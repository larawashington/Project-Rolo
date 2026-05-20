//! Anthropic Messages API provider — speaks the native Anthropic protocol
//! (NOT the OpenAI-compatible shim).
//!
//! PRD/rolo-command-center.md Brain panel — one of six provider families the
//! user can pick from the Brain tab. Uses SSE streaming so Rolo's voice still
//! arrives token-by-token even when his brain lives in Anthropic's cloud.
//!
//! System messages: Anthropic's API puts the system prompt in a top-level
//! `system` field instead of mixing it into `messages` like OpenAI does. We
//! concatenate any `system`-role `ProviderChatMessage` entries with a blank
//! line between them, then send only `user`/`assistant` messages in the
//! `messages` array.

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::json;
use std::time::Duration;

use super::provider::{GenerationConfig, InferenceError, InferenceProvider, ProviderChatMessage};
use crate::http::chat_client;

const ANTHROPIC_MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_MODELS_URL: &str = "https://api.anthropic.com/v1/models";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Native Anthropic Messages-API client with SSE streaming support.
pub struct AnthropicProvider {
    api_key: String,
    model: String,
    client: Client,
}

impl AnthropicProvider {
    /// Build a new Anthropic provider. The API key is sent on every request
    /// via the `x-api-key` header; it lives in process memory for as long as
    /// the slot holds this provider.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let client = chat_client().unwrap_or_else(|_| Client::new());

        Self {
            api_key: api_key.into(),
            model: model.into(),
            client,
        }
    }

    /// Split a flat `ProviderChatMessage` list into (system_prompt, messages).
    /// All `role=="system"` entries are concatenated with blank lines between
    /// them; everything else passes through with original ordering. Anthropic
    /// only recognises `user` and `assistant` roles inside `messages`, so any
    /// other roles are dropped with a warn (would only happen on malformed
    /// upstream input).
    fn partition_messages(messages: Vec<ProviderChatMessage>) -> (String, Vec<serde_json::Value>) {
        let mut system_parts: Vec<String> = Vec::new();
        let mut out: Vec<serde_json::Value> = Vec::with_capacity(messages.len());
        for m in messages {
            match m.role.as_str() {
                "system" => system_parts.push(m.content),
                "user" | "assistant" => {
                    out.push(json!({ "role": m.role, "content": m.content }));
                }
                other => {
                    log::warn!(
                        "[Rolo Anthropic] Dropping message with unexpected role '{}'",
                        other
                    );
                }
            }
        }
        (system_parts.join("\n\n"), out)
    }
}

#[async_trait]
impl InferenceProvider for AnthropicProvider {
    async fn generate(
        &self,
        messages: Vec<ProviderChatMessage>,
        config: GenerationConfig,
        token_tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<String, InferenceError> {
        let (system_prompt, api_messages) = Self::partition_messages(messages);

        let mut body = json!({
            "model": self.model,
            "max_tokens": config.max_tokens,
            "messages": api_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": true,
        });
        if !system_prompt.is_empty() {
            body["system"] = json!(system_prompt);
        }
        if !config.stop_sequences.is_empty() {
            body["stop_sequences"] = json!(config.stop_sequences);
        }

        crate::dev_log::log_llm_prompt("chat-window-anthropic", &body);

        let response = self
            .client
            .post(ANTHROPIC_MESSAGES_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
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

        // Parse Anthropic SSE stream. Events arrive as pairs:
        //   event: content_block_delta
        //   data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}
        // We only need the `data:` lines — the `event:` lines are
        // redundant since the JSON `type` field disambiguates.
        let mut stream = response.bytes_stream();
        let mut assembled = String::new();
        let mut line_buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| InferenceError::NetworkError(e.to_string()))?;
            let chunk_str = String::from_utf8_lossy(&chunk);
            line_buffer.push_str(&chunk_str);

            while let Some(boundary) = line_buffer.find("\n\n") {
                let event_block = line_buffer[..boundary].to_string();
                line_buffer = line_buffer[boundary + 2..].to_string();

                for line in event_block.lines() {
                    let line = line.trim();
                    if !line.starts_with("data: ") {
                        continue;
                    }
                    let data = &line[6..];

                    let parsed: serde_json::Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(e) => {
                            log::warn!(
                                "[Rolo Anthropic] Failed to parse SSE chunk: {} — data: {}",
                                e,
                                data
                            );
                            continue;
                        }
                    };

                    let event_type = parsed["type"].as_str().unwrap_or("");
                    match event_type {
                        "content_block_delta" => {
                            if let Some(text) = parsed["delta"]["text"].as_str() {
                                if !text.is_empty() {
                                    assembled.push_str(text);
                                    if token_tx.send(text.to_string()).await.is_err() {
                                        log::info!(
                                            "[Rolo Anthropic] Token receiver dropped — stopping generation"
                                        );
                                        return Err(InferenceError::Cancelled);
                                    }
                                }
                            }
                        }
                        "message_stop" => {
                            crate::dev_log::log_llm_response("chat-window-anthropic", &assembled);
                            return Ok(assembled);
                        }
                        "error" => {
                            let err_msg = parsed["error"]["message"]
                                .as_str()
                                .unwrap_or("unknown anthropic error")
                                .to_string();
                            return Err(InferenceError::ProviderError(err_msg));
                        }
                        _ => {
                            // message_start, content_block_start, content_block_stop,
                            // message_delta, ping — all ignored.
                        }
                    }
                }
            }
        }

        // Stream ended without an explicit message_stop — return what we have.
        crate::dev_log::log_llm_response("chat-window-anthropic", &assembled);
        Ok(assembled)
    }

    async fn health_check(&self) -> bool {
        match self
            .client
            .get(ANTHROPIC_MODELS_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    fn provider_name(&self) -> &str {
        "Anthropic"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_id(&self) -> &str {
        "anthropic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_match_constructor() {
        let p = AnthropicProvider::new("sk-test", "claude-haiku-4-5-20251001");
        assert_eq!(p.provider_id(), "anthropic");
        assert_eq!(p.provider_name(), "Anthropic");
        assert_eq!(p.model_name(), "claude-haiku-4-5-20251001");
    }

    #[test]
    fn partition_messages_concatenates_system_blocks() {
        let msgs = vec![
            ProviderChatMessage {
                role: "system".into(),
                content: "first system".into(),
            },
            ProviderChatMessage {
                role: "user".into(),
                content: "hi".into(),
            },
            ProviderChatMessage {
                role: "system".into(),
                content: "second system".into(),
            },
            ProviderChatMessage {
                role: "assistant".into(),
                content: "hello".into(),
            },
        ];
        let (system, api_msgs) = AnthropicProvider::partition_messages(msgs);
        assert_eq!(system, "first system\n\nsecond system");
        assert_eq!(api_msgs.len(), 2);
        assert_eq!(api_msgs[0]["role"], "user");
        assert_eq!(api_msgs[1]["role"], "assistant");
    }
}
