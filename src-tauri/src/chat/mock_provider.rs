//! Mock inference provider — a deterministic stand-in for testing Rolo's chat
//! pipeline without hitting any real API.
//!
//! Streams a pre-configured response one word at a time with configurable delay,
//! making it perfect for UI development and integration tests.

#![allow(dead_code)]

use async_trait::async_trait;
use tokio::time::{sleep, Duration};

use super::provider::{GenerationConfig, InferenceError, InferenceProvider, ProviderChatMessage};

/// A mock provider that streams a canned response word-by-word.
pub struct MockProvider {
    /// The full response text to stream.
    pub response: String,
    /// Delay in milliseconds between each word token.
    pub delay_ms: u64,
    /// If true, generate() returns an error instead of the response.
    pub should_fail: bool,
}

impl MockProvider {
    /// Create a new mock provider with default happy-path settings.
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
            delay_ms: 50,
            should_fail: false,
        }
    }

    /// Create a mock provider that will always fail.
    pub fn failing(error_message: impl Into<String>) -> Self {
        Self {
            response: error_message.into(),
            delay_ms: 0,
            should_fail: true,
        }
    }
}

#[async_trait]
impl InferenceProvider for MockProvider {
    async fn generate(
        &self,
        _messages: Vec<ProviderChatMessage>,
        _config: GenerationConfig,
        token_tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<String, InferenceError> {
        if self.should_fail {
            return Err(InferenceError::ProviderError(self.response.clone()));
        }

        let words: Vec<&str> = self.response.split_whitespace().collect();
        let mut assembled = String::new();

        for (i, word) in words.iter().enumerate() {
            if self.delay_ms > 0 {
                sleep(Duration::from_millis(self.delay_ms)).await;
            }

            // Add space before word (except the first)
            let token = if i == 0 {
                word.to_string()
            } else {
                format!(" {}", word)
            };

            assembled.push_str(&token);

            // If the receiver has dropped (UI disconnected), stop generating.
            if token_tx.send(token).await.is_err() {
                log::warn!("[Rolo] Mock provider: token receiver dropped — stopping generation");
                return Err(InferenceError::Cancelled);
            }
        }

        Ok(assembled)
    }

    async fn health_check(&self) -> bool {
        // Mock is always healthy (unless configured to fail)
        !self.should_fail
    }

    fn provider_name(&self) -> &str {
        "Mock"
    }

    fn model_name(&self) -> &str {
        "mock-v1"
    }

    fn provider_id(&self) -> &str {
        "mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn streams_words_one_at_a_time() {
        let provider = MockProvider {
            response: "Hello Rolo is happy".to_string(),
            delay_ms: 0,
            should_fail: false,
        };

        let (tx, mut rx) = tokio::sync::mpsc::channel(32);
        let config = GenerationConfig::default();

        let result = provider
            .generate(vec![], config, tx)
            .await
            .expect("should succeed");

        assert_eq!(result, "Hello Rolo is happy");

        // Collect all streamed tokens
        let mut tokens = Vec::new();
        while let Ok(token) = rx.try_recv() {
            tokens.push(token);
        }

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], "Hello");
        assert_eq!(tokens[1], " Rolo");
        assert_eq!(tokens[2], " is");
        assert_eq!(tokens[3], " happy");
    }

    #[tokio::test]
    async fn failing_provider_returns_error() {
        let provider = MockProvider::failing("Rolo's brain is offline");

        let (tx, _rx) = tokio::sync::mpsc::channel(32);
        let config = GenerationConfig::default();

        let result = provider.generate(vec![], config, tx).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            InferenceError::ProviderError(msg) => {
                assert_eq!(msg, "Rolo's brain is offline");
            }
            other => panic!("Expected ProviderError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn health_check_reflects_should_fail() {
        let healthy = MockProvider::new("hi");
        assert!(healthy.health_check().await);

        let unhealthy = MockProvider::failing("nope");
        assert!(!unhealthy.health_check().await);
    }

    #[tokio::test]
    async fn cancelled_when_receiver_dropped() {
        let provider = MockProvider {
            response: "word1 word2 word3 word4 word5".to_string(),
            delay_ms: 0,
            should_fail: false,
        };

        // Create a channel with capacity 1 and drop the receiver immediately
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);

        let config = GenerationConfig::default();
        let result = provider.generate(vec![], config, tx).await;

        match result {
            Err(InferenceError::Cancelled) => {} // expected
            other => panic!("Expected Cancelled, got {:?}", other),
        }
    }

    #[test]
    fn provider_name_and_model() {
        let p = MockProvider::new("hi");
        assert_eq!(p.provider_name(), "Mock");
        assert_eq!(p.model_name(), "mock-v1");
    }
}
