//! OpenAI-compatible inference provider — connects Rolo to any API that speaks
//! the OpenAI chat completions protocol (OpenAI, Ollama, LM Studio, etc.).
//!
//! Uses HTTP streaming (SSE) to deliver tokens in real-time, so Rolo's words
//! appear one at a time as if he's actually thinking.

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::json;
use std::time::Duration;

use super::provider::{GenerationConfig, InferenceError, InferenceProvider, ProviderChatMessage};
use crate::http::chat_client;

/// An OpenAI-compatible chat completions client with SSE streaming support.
pub struct OpenAICompatProvider {
    /// Base URL of the API (e.g., "https://api.openai.com/v1" or "http://localhost:11434/v1").
    pub base_url: String,
    /// API key for authentication. `None` for local providers that don't require auth.
    pub api_key: Option<String>,
    /// Model identifier (e.g., "gpt-4o-mini", "llama3").
    pub model: String,
    /// Stable provider-family identifier returned by `provider_id()`. The same
    /// transport speaks to OpenAI-compatible endpoints AND to Ollama's `/v1`
    /// endpoint, so the caller picks the brand at construction time — Ollama
    /// uses `new_with_id(..., "ollama")` so the tool dispatcher can recognize
    /// it; everything else falls back to the default `"openai_compat"`.
    provider_id: String,
    /// HTTP client with pre-configured timeouts.
    client: Client,
}

impl OpenAICompatProvider {
    /// Create a new provider pointing at the given API endpoint.
    ///
    /// `base_url` should be the versioned API root (e.g., "https://api.openai.com/v1").
    /// The provider appends `/chat/completions` for generation requests.
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::new_with_id(base_url, api_key, model, "openai_compat")
    }

    /// Create a new provider with an explicit `provider_id`. Used by the
    /// Ollama branch in `engine.rs::build_provider_from_config` so the same
    /// HTTP transport reports as `"ollama"` for the dispatcher's brain check.
    pub fn new_with_id(
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: impl Into<String>,
        provider_id: impl Into<String>,
    ) -> Self {
        let client = chat_client().unwrap_or_else(|_| Client::new());

        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            model: model.into(),
            provider_id: provider_id.into(),
            client,
        }
    }
}

#[async_trait]
impl InferenceProvider for OpenAICompatProvider {
    async fn generate(
        &self,
        messages: Vec<ProviderChatMessage>,
        config: GenerationConfig,
        token_tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<String, InferenceError> {
        let url = format!("{}/chat/completions", self.base_url);

        // Build the message array for the API
        let api_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let mut body = json!({
            "model": self.model,
            "messages": api_messages,
            "stream": true,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        });

        if !config.stop_sequences.is_empty() {
            body["stop"] = json!(config.stop_sequences);
        }

        crate::dev_log::log_llm_prompt("chat-window", &body);

        // Build the request
        let mut request = self.client.post(&url).json(&body);

        if let Some(ref key) = self.api_key {
            request = request.bearer_auth(key);
        }

        // Send and begin streaming
        let response = request.send().await.map_err(|e| {
            if e.is_timeout() {
                InferenceError::Timeout
            } else {
                InferenceError::NetworkError(e.to_string())
            }
        })?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Could not read error body".to_string());
            return Err(InferenceError::ProviderError(format!(
                "HTTP {}: {}",
                status, error_body
            )));
        }

        // Parse SSE stream
        let mut stream = response.bytes_stream();
        let mut assembled = String::new();
        let mut line_buffer = String::new();
        // TODO(finetune): drop this stripper once the Rolo finetune ships —
        // see plans/rolo-finetune-pipeline.md. Filters Gemma's quote wrapping
        // out of the streamed token feed so the chat window never shows it.
        let mut quote_stripper = crate::ollama::StreamingQuoteStripper::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| InferenceError::NetworkError(e.to_string()))?;

            let chunk_str = String::from_utf8_lossy(&chunk);
            line_buffer.push_str(&chunk_str);

            // SSE protocol: events are separated by double newlines
            while let Some(boundary) = line_buffer.find("\n\n") {
                let event_block = line_buffer[..boundary].to_string();
                line_buffer = line_buffer[boundary + 2..].to_string();

                // Process each line in the event block
                for line in event_block.lines() {
                    let line = line.trim();

                    if !line.starts_with("data: ") {
                        continue;
                    }

                    let data = &line[6..]; // Strip "data: " prefix

                    // [DONE] marks the end of the stream
                    if data == "[DONE]" {
                        assembled.push_str(&quote_stripper.finish());
                        crate::dev_log::log_llm_response("chat-window", &assembled);
                        return Ok(assembled);
                    }

                    // Parse the JSON chunk
                    let parsed: serde_json::Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(e) => {
                            log::warn!("[Rolo] Failed to parse SSE chunk: {} — data: {}", e, data);
                            continue;
                        }
                    };

                    // Extract the delta content
                    if let Some(content) = parsed["choices"][0]["delta"]["content"].as_str() {
                        if !content.is_empty() {
                            // TODO(finetune): the stripper pass goes away once
                            // the Rolo finetune ships.
                            let emitted = quote_stripper.push(content);
                            if !emitted.is_empty() {
                                assembled.push_str(&emitted);

                                // Stream token to the frontend — if the receiver is
                                // gone (user closed the chat), stop generating.
                                if token_tx.send(emitted).await.is_err() {
                                    log::info!(
                                        "[Rolo] Token receiver dropped — stopping generation"
                                    );
                                    return Err(InferenceError::Cancelled);
                                }
                            }
                        }
                    }

                    // Check for finish_reason to handle early stops
                    if let Some(reason) = parsed["choices"][0]["finish_reason"].as_str() {
                        if reason == "stop" || reason == "length" {
                            assembled.push_str(&quote_stripper.finish());
                            crate::dev_log::log_llm_response("chat-window", &assembled);
                            return Ok(assembled);
                        }
                    }
                }
            }
        }

        // Stream ended without [DONE] — return what we have
        assembled.push_str(&quote_stripper.finish());
        crate::dev_log::log_llm_response("chat-window", &assembled);
        Ok(assembled)
    }

    async fn health_check(&self) -> bool {
        let url = format!("{}/models", self.base_url);

        let mut request = self.client.get(&url);
        if let Some(ref key) = self.api_key {
            request = request.bearer_auth(key);
        }

        match request.timeout(Duration::from_secs(5)).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    fn provider_name(&self) -> &str {
        "OpenAI-Compatible"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_id(&self) -> &str {
        &self.provider_id
    }
}
