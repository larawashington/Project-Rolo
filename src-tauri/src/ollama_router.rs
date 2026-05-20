//! Non-streaming Ollama transport for the tool-layer router (PRD §5.5 / T4).
//!
//! `ollama.rs` owns the streaming bubble pipeline (sync `ureq` over a thread +
//! channels). The router is a different shape entirely: a single async call,
//! grammar-constrained via Ollama's `format` parameter, that returns one JSON
//! envelope. Sharing transports would force one of those use cases to mimic
//! the other awkwardly. We therefore keep them in sibling modules and treat
//! `ollama.rs` as the streaming-bubble path; `ollama_router.rs` as the
//! constrained-envelope path. Both speak HTTP to the same `localhost:11434`.
//!
//! The PRD §5.7 line "ollama.rs becomes the shared low-level transport — the
//! router uses it with `format` set" is preserved in spirit: this module is
//! the second arm of that shared transport, just routed through a different
//! crate (`reqwest`) because it lives on the async/Tokio side of the app.

use std::time::Duration;

use serde_json::{json, Value};

/// Default Ollama base URL (localhost daemon). Exported so the dispatcher and
/// tests share a single source of truth.
pub const OLLAMA_BASE_URL_DEFAULT: &str = "http://localhost:11434";

/// Failure modes for `ollama_chat_with_format`. Each variant carries a short
/// reason string so the dispatcher can log it before falling back to the
/// legacy prompt path.
#[derive(Debug)]
pub enum OllamaRouterError {
    /// Network-level failure (connection refused, DNS, TLS, etc.).
    Network(String),
    /// Ollama returned 200 but the body wasn't valid JSON, OR the
    /// assistant `message.content` string wasn't valid JSON either.
    InvalidJson(String),
    /// Body parsed but didn't conform to the Ollama `/api/chat` shape
    /// (missing `message.content`, etc.).
    BadEnvelope(String),
    /// Request didn't complete inside the supplied timeout.
    Timeout,
    /// Anything else — non-200 status, unexpected error type, etc.
    Other(String),
}

impl std::fmt::Display for OllamaRouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OllamaRouterError::Network(s) => write!(f, "network error: {s}"),
            OllamaRouterError::InvalidJson(s) => write!(f, "invalid JSON: {s}"),
            OllamaRouterError::BadEnvelope(s) => write!(f, "bad envelope: {s}"),
            OllamaRouterError::Timeout => write!(f, "request timed out"),
            OllamaRouterError::Other(s) => write!(f, "other: {s}"),
        }
    }
}

impl std::error::Error for OllamaRouterError {}

/// Calls Ollama `/api/chat` with `format` set to a JSON schema, NOT streaming.
/// Used by the tool-layer router (Pass 1). Returns the parsed JSON content
/// from the assistant message — caller validates the envelope shape against
/// `RouterEnvelope`.
///
/// `format_schema` is the JSON Schema object itself (not a string); Ollama
/// accepts it verbatim and constrains decoding to outputs that match.
///
/// Logs both the prompt body and the assistant's response via `dev_log` so
/// `ROLO_LOG_PROMPTS=1` covers this path the same way it covers the bubble
/// pipeline.
pub async fn ollama_chat_with_format(
    base_url: &str,
    model: &str,
    system_prompt: &str,
    user_input: &str,
    format_schema: &Value,
    timeout: Duration,
) -> Result<Value, OllamaRouterError> {
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| OllamaRouterError::Other(format!("client build failed: {e}")))?;

    let body = json!({
        "model": model,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user",   "content": user_input },
        ],
        "stream": false,
        "format": format_schema,
        "options": { "temperature": 0.0 },
    });

    crate::dev_log::log_llm_prompt("router", &body);

    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));
    let resp = match client.post(&url).json(&body).send().await {
        Ok(r) => r,
        Err(e) => {
            if e.is_timeout() {
                return Err(OllamaRouterError::Timeout);
            }
            return Err(OllamaRouterError::Network(e.to_string()));
        }
    };

    if !resp.status().is_success() {
        return Err(OllamaRouterError::Other(format!(
            "non-200 status: {}",
            resp.status()
        )));
    }

    let outer: Value = match resp.json::<Value>().await {
        Ok(v) => v,
        Err(e) => {
            if e.is_timeout() {
                return Err(OllamaRouterError::Timeout);
            }
            return Err(OllamaRouterError::InvalidJson(format!(
                "outer body not JSON: {e}"
            )));
        }
    };

    // Ollama's non-streaming /api/chat returns `{ "message": { "role":
    // "assistant", "content": "<json string>" }, ... }`. Under `format`,
    // `content` is a string of JSON we have to parse a second time.
    let content_str = outer
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .ok_or_else(|| {
            OllamaRouterError::BadEnvelope(
                "missing message.content in /api/chat response".to_string(),
            )
        })?;

    crate::dev_log::log_llm_response("router", content_str);

    let parsed: Value = serde_json::from_str(content_str).map_err(|e| {
        OllamaRouterError::InvalidJson(format!("message.content not valid JSON: {e}"))
    })?;

    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_router_error_display_renders_each_variant() {
        // Just verify Display doesn't panic on each variant — these strings
        // surface in dispatcher logs, so a missing arm would be silent today.
        let _ = OllamaRouterError::Network("x".into()).to_string();
        let _ = OllamaRouterError::InvalidJson("x".into()).to_string();
        let _ = OllamaRouterError::BadEnvelope("x".into()).to_string();
        let _ = OllamaRouterError::Timeout.to_string();
        let _ = OllamaRouterError::Other("x".into()).to_string();
    }

    #[tokio::test]
    async fn unreachable_endpoint_returns_network_error() {
        // Port 1 is reserved/never bound on Darwin → connection refused.
        // Confirms we map low-level reqwest errors to Network, not panic.
        let schema = json!({"type": "object"});
        let res = ollama_chat_with_format(
            "http://127.0.0.1:1",
            "gemma3:4b",
            "sys",
            "user",
            &schema,
            Duration::from_millis(200),
        )
        .await;
        assert!(
            matches!(
                res,
                Err(OllamaRouterError::Network(_) | OllamaRouterError::Timeout)
            ),
            "expected Network or Timeout, got {res:?}"
        );
    }
}
