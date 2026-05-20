//! Google Gemini (Generative Language API) provider — speaks the native
//! Gemini protocol via `streamGenerateContent?alt=sse`.
//!
//! PRD/rolo-command-center.md Brain panel — one of six provider families.
//! Gemini uses a different shape from OpenAI: roles are `user` / `model`
//! instead of `user` / `assistant`, system prompts go into `systemInstruction`,
//! and the SSE stream emits whole `candidates` chunks rather than token
//! deltas. We coalesce `parts[].text` from each chunk into the streaming
//! channel.

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::json;
use std::time::Duration;

use super::provider::{GenerationConfig, InferenceError, InferenceProvider, ProviderChatMessage};
use crate::http::chat_client;

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Native Gemini streaming client.
pub struct GeminiProvider {
    api_key: String,
    model: String,
    client: Client,
}

impl GeminiProvider {
    /// Build a new Gemini provider. The API key travels in the URL query
    /// string `?key=...` on every request — that's how Google's API expects
    /// it.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let client = chat_client().unwrap_or_else(|_| Client::new());

        Self {
            api_key: api_key.into(),
            model: model.into(),
            client,
        }
    }

    /// Map a flat `ProviderChatMessage` history into Gemini's expected shape.
    ///
    /// Returns `(contents, systemInstruction)` where:
    ///   * `system` messages collapse into `systemInstruction.parts[0].text`
    ///   * `assistant` becomes `model`
    ///   * `user` stays `user`
    ///
    /// Anything else is dropped with a warn.
    fn partition_messages(messages: Vec<ProviderChatMessage>) -> (Vec<serde_json::Value>, String) {
        let mut system_parts: Vec<String> = Vec::new();
        let mut contents: Vec<serde_json::Value> = Vec::with_capacity(messages.len());
        for m in messages {
            let role = match m.role.as_str() {
                "system" => {
                    system_parts.push(m.content);
                    continue;
                }
                "assistant" => "model",
                "user" => "user",
                other => {
                    log::warn!(
                        "[Rolo Gemini] Dropping message with unexpected role '{}'",
                        other
                    );
                    continue;
                }
            };
            contents.push(json!({
                "role": role,
                "parts": [{ "text": m.content }],
            }));
        }
        (contents, system_parts.join("\n\n"))
    }
}

#[async_trait]
impl InferenceProvider for GeminiProvider {
    async fn generate(
        &self,
        messages: Vec<ProviderChatMessage>,
        config: GenerationConfig,
        token_tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<String, InferenceError> {
        let (contents, system_text) = Self::partition_messages(messages);
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse&key={}",
            GEMINI_API_BASE, self.model, self.api_key
        );

        let mut body = json!({
            "contents": contents,
            "generationConfig": {
                "temperature": config.temperature,
                "topP": config.top_p,
                "maxOutputTokens": config.max_tokens,
            },
        });
        if !system_text.is_empty() {
            body["systemInstruction"] = json!({
                "parts": [{ "text": system_text }],
            });
        }
        if !config.stop_sequences.is_empty() {
            body["generationConfig"]["stopSequences"] = json!(config.stop_sequences);
        }

        // Log the body but redact the URL — it contains the API key.
        crate::dev_log::log_llm_prompt("chat-window-gemini", &body);

        let response = self
            .client
            .post(&url)
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
                                "[Rolo Gemini] Failed to parse SSE chunk: {} — data: {}",
                                e,
                                data
                            );
                            continue;
                        }
                    };

                    // Collect text from candidates[0].content.parts[*].text.
                    if let Some(parts) = parsed["candidates"][0]["content"]["parts"].as_array() {
                        for part in parts {
                            if let Some(text) = part["text"].as_str() {
                                if !text.is_empty() {
                                    assembled.push_str(text);
                                    if token_tx.send(text.to_string()).await.is_err() {
                                        log::info!(
                                            "[Rolo Gemini] Token receiver dropped — stopping generation"
                                        );
                                        return Err(InferenceError::Cancelled);
                                    }
                                }
                            }
                        }
                    }

                    // finishReason marks completion; common values: STOP,
                    // MAX_TOKENS, SAFETY, RECITATION. Any non-empty value
                    // terminates this turn.
                    if let Some(reason) = parsed["candidates"][0]["finishReason"].as_str() {
                        if !reason.is_empty() {
                            crate::dev_log::log_llm_response("chat-window-gemini", &assembled);
                            return Ok(assembled);
                        }
                    }
                }
            }
        }

        crate::dev_log::log_llm_response("chat-window-gemini", &assembled);
        Ok(assembled)
    }

    async fn health_check(&self) -> bool {
        let url = format!("{}/models?key={}", GEMINI_API_BASE, self.api_key);
        match self
            .client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    fn provider_name(&self) -> &str {
        "Google Gemini"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_id(&self) -> &str {
        "gemini"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_match_constructor() {
        let p = GeminiProvider::new("key-test", "gemini-2.0-flash");
        assert_eq!(p.provider_id(), "gemini");
        assert_eq!(p.provider_name(), "Google Gemini");
        assert_eq!(p.model_name(), "gemini-2.0-flash");
    }

    #[test]
    fn partition_messages_maps_assistant_to_model() {
        let msgs = vec![
            ProviderChatMessage {
                role: "system".into(),
                content: "be helpful".into(),
            },
            ProviderChatMessage {
                role: "user".into(),
                content: "hi".into(),
            },
            ProviderChatMessage {
                role: "assistant".into(),
                content: "hello".into(),
            },
        ];
        let (contents, system_text) = GeminiProvider::partition_messages(msgs);
        assert_eq!(system_text, "be helpful");
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[1]["parts"][0]["text"], "hello");
    }
}
