//! Inference provider trait — the contract that any LLM backend must fulfill
//! to power Rolo's conversations.
//!
//! This abstraction allows swapping between local models, OpenAI-compatible
//! APIs, mock providers for testing, and future backends without changing
//! the chat orchestration layer.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// A message in the inference context. Deliberately separate from ChatMessage
/// (the persistence type) — the provider only cares about role and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderChatMessage {
    pub role: String,
    pub content: String,
}

/// Configuration knobs for text generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_p: 0.95,
            max_tokens: 200,
            stop_sequences: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during inference.
#[derive(Debug)]
pub enum InferenceError {
    /// HTTP or connection failure.
    NetworkError(String),
    /// The provider returned an error response (4xx, 5xx, malformed body).
    ProviderError(String),
    /// The request timed out.
    Timeout,
    /// The request was cancelled by the caller.
    Cancelled,
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            InferenceError::ProviderError(msg) => write!(f, "Provider error: {}", msg),
            InferenceError::Timeout => write!(f, "Request timed out"),
            InferenceError::Cancelled => write!(f, "Request was cancelled"),
        }
    }
}

impl std::error::Error for InferenceError {}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// The contract for any LLM inference backend.
///
/// Implementations must be `Send + Sync` so they can be stored in Tauri managed
/// state and called from async command handlers.
///
/// Streaming is handled via `token_tx` — the provider sends each token as it
/// arrives so the frontend can display Rolo's words letter-by-letter, giving
/// him that charming thinking-out-loud quality.
#[async_trait]
pub trait InferenceProvider: Send + Sync {
    /// Generate a response to the given message history.
    ///
    /// Tokens are streamed one-by-one through `token_tx`. The full assembled
    /// response is returned on completion. If the sender is dropped (frontend
    /// disconnected), the provider should stop generating and return what it
    /// has so far, or `InferenceError::Cancelled`.
    async fn generate(
        &self,
        messages: Vec<ProviderChatMessage>,
        config: GenerationConfig,
        token_tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<String, InferenceError>;

    /// Quick connectivity check. Returns `true` if the provider is reachable
    /// and ready to accept requests.
    async fn health_check(&self) -> bool;

    /// Human-readable name of this provider (e.g., "OpenAI", "Ollama", "Mock").
    fn provider_name(&self) -> &str;

    /// The model identifier being used (e.g., "gpt-4o-mini", "llama3").
    fn model_name(&self) -> &str;

    /// Stable lowercase identifier for this provider family. Used by the tool
    /// dispatcher to decide whether to allow tool routing (Ollama-only today)
    /// and by the Brain tab to display the active provider.
    ///
    /// Must be one of: "ollama" | "openai_compat" | "anthropic" | "gemini" | "mock".
    fn provider_id(&self) -> &str;
}
