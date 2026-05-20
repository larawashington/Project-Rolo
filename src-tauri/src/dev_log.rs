//! Developer logging helpers — gated by `ROLO_LOG_PROMPTS=1`.
//!
//! Used to inspect what's being sent to the LLM during development. Logs the
//! full JSON request body so dynamic fields (state context, memories, history,
//! generation options) all surface automatically as the prompt grows.
//!
//! Output goes to the standard log pipeline (`log::info!`), so it respects
//! `RUST_LOG`. Use `RUST_LOG=info` or wider to actually see it.

use std::sync::OnceLock;

static PROMPT_LOGGING_ENABLED: OnceLock<bool> = OnceLock::new();

/// Returns true if `ROLO_LOG_PROMPTS=1` was set at process start.
pub fn prompt_logging_enabled() -> bool {
    *PROMPT_LOGGING_ENABLED.get_or_init(|| std::env::var("ROLO_LOG_PROMPTS").is_ok())
}

/// Pretty-prints the request body sent to the LLM if prompt logging is on.
///
/// `label` distinguishes call sites in the log (e.g., "speech-bubble",
/// "chat-window"). `body` is the full JSON payload — logging the whole body
/// future-proofs this against new dynamic fields without touching this helper.
pub fn log_llm_prompt(label: &str, body: &serde_json::Value) {
    if !prompt_logging_enabled() {
        return;
    }
    let pretty = serde_json::to_string_pretty(body).unwrap_or_else(|_| body.to_string());
    log::info!(
        "[Rolo PROMPT::{}]\n----- BEGIN PROMPT -----\n{}\n----- END PROMPT -----",
        label,
        pretty
    );
}

/// Logs the LLM's raw response text alongside its prompt, when enabled.
pub fn log_llm_response(label: &str, response: &str) {
    if !prompt_logging_enabled() {
        return;
    }
    log::info!(
        "[Rolo RESPONSE::{}]\n----- BEGIN RESPONSE -----\n{}\n----- END RESPONSE -----",
        label,
        response
    );
}

/// Logs a tool-layer routing decision (PRD/rolo-tool-layer.md §5.6).
///
/// Always logged at `info` (not gated by `ROLO_LOG_PROMPTS`) because dispatcher
/// telemetry is the only audit trail for *why* Rolo retrieved a given memory in
/// v1. `outcome` is one of the labels documented in `tools::dispatcher`; user
/// `input` is truncated to 80 chars to keep the log line bounded.
pub fn log_routing_decision(input: &str, outcome: &str, latency_ms: u128) {
    log::info!(
        "[Rolo ROUTE] outcome={} latency_ms={} input={:?}",
        outcome,
        latency_ms,
        truncate_for_log(input)
    );
}

/// Truncate `s` to ≤80 characters at a UTF-8 boundary, appending an ellipsis
/// when shortened. Bounded at the byte level — the dispatcher logs untrusted
/// user input, so we don't want a single huge paste to flood the log file.
fn truncate_for_log(s: &str) -> String {
    if s.len() <= 80 {
        return s.to_string();
    }
    let mut cut = 80;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &s[..cut])
}
