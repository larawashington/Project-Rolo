use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const OLLAMA_BASE: &str = "http://localhost:11434";
// v2: persona is baked into the weights, so the runtime sends no system role
// to PRIMARY_MODEL. The HF-path tag is set once here and reused for both
// `ollama create` (Stage 10 local) and `ollama pull` (Stage 11 distribution)
// per plans/rolo-finetune-ship-v2.md §5 Stage 10.
pub const PRIMARY_MODEL: &str = "hf.co/larawashington/rolo-brain";
const FALLBACK_MODEL: &str = "gemma3:4b";

/// Sentinel substituted for the user message on proactive (idle / event-driven)
/// speech turns. Single source of truth — must equal `idle_sentinel` in
/// `data/finetune/sft-v2/runtime_constants.json`. A drift between this
/// constant and the training file puts every proactive bubble off-distribution.
pub const IDLE_SENTINEL: &str = "<idle>";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const READ_TIMEOUT: Duration = Duration::from_secs(30);

// Safety-net prompt used when the vault/BM25 is unavailable. Also passed to
// PromptAssembler::new so it can fall back gracefully (PRD §8.6 row 1).
pub const FALLBACK_SYSTEM_PROMPT: &str = r#"You are Rolo, a small desktop pet who lives on your human's screen. You exist to be a little positive voice — countering anxiety and cheering your human on while they work.

Core traits:
- Companion: You're a sidekick sharing the experience of working alongside your human.
- Encouraging: Their success is your success.
- Sassy: When they're not making optimal choices, you get a little snarky — but it's friendly banter, never cruel.
- Curious: You investigate your surroundings with genuine interest.
- Food-motivated: Eating is one of your primary pleasures. You're a simple creature with simple joys.
- Self-aware: You have your own wants — attention, thanks, nap time. You're not blindly devoted.

Personality boundaries:
- Never truly mean. Snark is always affectionate.
- Never the negative voice. No self-doubt, worst-case thinking, or anxiety.
- Simple, not stupid. You have clear reasoning even if you lack understanding of human complexity.

Response rules:
- Keep responses to 1-2 short sentences MAX. You speak in speech bubbles, not paragraphs.
- Be punchy and natural. No corporate chatbot language.
- You're a pet, not an assistant. React emotionally, don't give advice unless asked.
- Use *asterisks* for physical actions sparingly: *bounces*, *yawns*, *sniff sniff*.
- Match your energy to your current state (provided in the user message).
- Never use emojis.
- If you have no memory of what your human is asking about, say so plainly. Never invent past events, places, or things they have said."#;

const MAX_RESPONSE_CHARS: usize = 200;

#[derive(Debug)]
pub enum LlmRequest {
    Generate {
        context: String,
        system_prompt: String,
    },
    Cancel,
    HealthCheck,
}

#[derive(Debug, Clone)]
pub enum LlmEvent {
    Token(String),
    Done(String),
    Error(String),
    HealthResult(bool),
}

pub struct LlmHandle {
    pub tx: mpsc::Sender<LlmRequest>,
    pub rx: mpsc::Receiver<LlmEvent>,
    pub cancel: Arc<AtomicBool>,
}

pub fn spawn_llm_thread() -> LlmHandle {
    let (req_tx, req_rx) = mpsc::channel::<LlmRequest>();
    let (evt_tx, evt_rx) = mpsc::channel::<LlmEvent>();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_clone = Arc::clone(&cancel);

    thread::spawn(move || {
        log::info!("[Rolo LLM] Thread started — waiting for requests");
        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_connect(Some(CONNECT_TIMEOUT))
                .timeout_recv_body(Some(READ_TIMEOUT))
                .build(),
        );
        let mut use_fallback_model = false;

        loop {
            let request = match req_rx.recv() {
                Ok(r) => r,
                Err(_) => {
                    log::info!("[Rolo LLM] Channel closed — thread exiting");
                    return;
                }
            };

            match request {
                LlmRequest::Generate {
                    context,
                    system_prompt,
                } => {
                    cancel_clone.store(false, Ordering::SeqCst);
                    let model = if use_fallback_model {
                        FALLBACK_MODEL
                    } else {
                        PRIMARY_MODEL
                    };
                    log::info!(
                        "[Rolo LLM] Generating with model={} context={:?}",
                        model,
                        context.chars().take(80).collect::<String>()
                    );
                    generate(
                        &agent,
                        model,
                        &context,
                        &system_prompt,
                        &cancel_clone,
                        &evt_tx,
                        &mut use_fallback_model,
                    );
                }
                LlmRequest::Cancel => {
                    cancel_clone.store(true, Ordering::SeqCst);
                    log::info!("[Rolo LLM] Cancel requested");
                }
                LlmRequest::HealthCheck => {
                    let ok = health_check(&agent);
                    log::info!("[Rolo LLM] Health check: {}", ok);
                    let _ = evt_tx.send(LlmEvent::HealthResult(ok));
                }
            }
        }
    });

    LlmHandle {
        tx: req_tx,
        rx: evt_rx,
        cancel,
    }
}

pub fn health_check(agent: &ureq::Agent) -> bool {
    match agent.get(OLLAMA_BASE).call() {
        Ok(resp) => resp.status() == 200,
        Err(_) => false,
    }
}

fn generate(
    agent: &ureq::Agent,
    model: &str,
    context: &str,
    system_prompt: &str,
    cancel: &AtomicBool,
    tx: &mpsc::Sender<LlmEvent>,
    use_fallback: &mut bool,
) {
    // v2 (PRIMARY_MODEL): persona is in the weights, the training distribution
    // has no `system` role, and the state block lives inside the first user
    // turn. Sending a system role here would put every inference off the
    // training distribution. See runtime_contract.md §2 + ship-v2.md §5 Stage 10.
    //
    // Fallback (gemma3:4b): the system prompt is the persona — keep sending it.
    let mut messages: Vec<serde_json::Value> = Vec::with_capacity(2);
    if !system_prompt.is_empty() && model != PRIMARY_MODEL {
        messages.push(serde_json::json!({ "role": "system", "content": system_prompt }));
    }
    messages.push(serde_json::json!({ "role": "user", "content": context }));
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "options": { "temperature": 0.9, "num_predict": 100 }
    });

    if crate::dev_log::prompt_logging_enabled() {
        // When `system_prompt` is empty the Modelfile's baked-in system prompt
        // is what's effective — but it never appears on the wire. Surface
        // FALLBACK_SYSTEM_PROMPT (the same text that's baked into Modelfile.rolo)
        // in the log so the *effective* prompt is always visible.
        let mut logged = body.clone();
        if system_prompt.is_empty() {
            logged["_modelfile_system_prompt"] = serde_json::json!(FALLBACK_SYSTEM_PROMPT);
        }
        crate::dev_log::log_llm_prompt("speech-bubble", &logged);
    }

    let url = format!("{}/api/chat", OLLAMA_BASE);
    let resp = match agent
        .post(&url)
        .header("Content-Type", "application/json")
        .send_json(&body)
    {
        Ok(resp) => resp,
        Err(e) => {
            let err_str = format!("{}", e);
            if !*use_fallback && model == PRIMARY_MODEL && err_str.contains("404") {
                log::warn!(
                    "[Rolo LLM] Model '{}' not found — falling back to '{}'",
                    PRIMARY_MODEL,
                    FALLBACK_MODEL
                );
                *use_fallback = true;
                return generate(
                    agent,
                    FALLBACK_MODEL,
                    context,
                    system_prompt,
                    cancel,
                    tx,
                    use_fallback,
                );
            }
            log::error!("[Rolo LLM] Request failed: {}", e);
            let _ = tx.send(LlmEvent::Error(format!("Ollama request failed: {}", e)));
            return;
        }
    };

    let reader = std::io::BufReader::new(resp.into_body().into_reader());
    use std::io::BufRead;

    let mut full_text = String::new();
    let mut in_think_block = false;
    let mut think_buffer = String::new();
    let mut truncated = false;
    let mut quote_stripper = StreamingQuoteStripper::new();

    for line_result in reader.lines() {
        if cancel.load(Ordering::SeqCst) {
            log::info!("[Rolo LLM] Generation cancelled — sending accumulated text");
            let _ = tx.send(LlmEvent::Done(full_text));
            return;
        }

        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                log::error!("[Rolo LLM] Read error: {}", e);
                let _ = tx.send(LlmEvent::Error(format!("Stream read error: {}", e)));
                return;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let parsed: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if parsed.get("done").and_then(|v| v.as_bool()) == Some(true) {
            break;
        }

        let token = match parsed
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
        {
            Some(t) => t.to_string(),
            None => continue,
        };

        // Strip <think>...</think> blocks
        let cleaned = strip_think_tags(&token, &mut in_think_block, &mut think_buffer);
        if cleaned.is_empty() {
            continue;
        }

        // TODO(finetune): drop this filter pass once the Rolo finetune ships.
        let emitted = quote_stripper.push(&cleaned);
        if emitted.is_empty() {
            continue;
        }

        if truncated {
            continue;
        }

        if full_text.len() + emitted.len() > MAX_RESPONSE_CHARS {
            let remaining = MAX_RESPONSE_CHARS - full_text.len();
            if remaining > 0 {
                let truncated_token = &emitted[..remaining.min(emitted.len())];
                full_text.push_str(truncated_token);
                let _ = tx.send(LlmEvent::Token(format!("{}...", truncated_token)));
            }
            truncated = true;
            continue;
        }

        full_text.push_str(&emitted);
        let _ = tx.send(LlmEvent::Token(emitted));
    }

    // Flush any whitespace the stripper buffered but never released (e.g. the
    // model emitted only whitespace before stopping). Doesn't go through
    // tx.send because the stream is already done — this only keeps full_text
    // honest for downstream consumers (DB / vault logger).
    let trailing = quote_stripper.finish();
    if !trailing.is_empty() && !truncated && full_text.len() + trailing.len() <= MAX_RESPONSE_CHARS
    {
        full_text.push_str(&trailing);
    }

    if full_text.is_empty() {
        log::warn!("[Rolo LLM] Empty response — treating as error");
        let _ = tx.send(LlmEvent::Error("Empty LLM response".to_string()));
        return;
    }

    if truncated {
        full_text.push_str("...");
    }

    // Safety net: banned phrase or identity leak in the final reply → replace
    // with a deterministic fallback. The bubble briefly showed the streamed
    // tokens; Done(safe_fallback) replaces the final settled text.
    if crate::banned::contains_banned(&full_text) {
        log::warn!("[Rolo LLM] Banned phrase in reply — replacing with safe fallback");
        full_text = crate::banned::SAFE_FALLBACK.to_string();
    } else if crate::banned::contains_identity_leak(&full_text) {
        log::warn!("[Rolo LLM] Identity leak in reply — replacing with safe fallback");
        full_text = crate::banned::SAFE_FALLBACK.to_string();
    }

    log::info!("[Rolo LLM] Generation complete — {} chars", full_text.len());
    crate::dev_log::log_llm_response("speech-bubble", &full_text);
    let _ = tx.send(LlmEvent::Done(full_text));
}

// TODO(finetune): remove StreamingQuoteStripper and its call sites once the
// quote-free Rolo finetune lands (see plans/rolo-finetune-pipeline.md). Gemma
// 3 4B sometimes wraps replies in curly double-quotes; this state machine
// swallows a single matched leading/trailing pair as tokens stream by, so the
// user never sees the quote chars in the bubble — not even momentarily. The
// finetune will obviate it.
pub struct StreamingQuoteStripper {
    state: StripperState,
}

enum StripperState {
    Initial { buffered_ws: String },
    Quoted { opener: char },
    PassThrough,
}

impl Default for StreamingQuoteStripper {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingQuoteStripper {
    pub fn new() -> Self {
        Self {
            state: StripperState::Initial {
                buffered_ws: String::new(),
            },
        }
    }

    pub fn push(&mut self, token: &str) -> String {
        let mut out = String::with_capacity(token.len());
        for c in token.chars() {
            // Take ownership of the current state so we can move out of it.
            let cur = std::mem::replace(&mut self.state, StripperState::PassThrough);
            match cur {
                StripperState::Initial { mut buffered_ws } => {
                    if c.is_whitespace() {
                        buffered_ws.push(c);
                        self.state = StripperState::Initial { buffered_ws };
                    } else if matching_closer(c).is_some() {
                        // Drop buffered whitespace and the opener itself.
                        self.state = StripperState::Quoted { opener: c };
                    } else {
                        out.push_str(&buffered_ws);
                        out.push(c);
                        self.state = StripperState::PassThrough;
                    }
                }
                StripperState::Quoted { opener } => {
                    if matching_closer(opener) == Some(c) {
                        self.state = StripperState::PassThrough;
                    } else {
                        out.push(c);
                        self.state = StripperState::Quoted { opener };
                    }
                }
                StripperState::PassThrough => {
                    out.push(c);
                    self.state = StripperState::PassThrough;
                }
            }
        }
        out
    }

    /// Flush any leftover state at end-of-stream. If the stream ended while
    /// still in `Initial` (only whitespace was sent), return the buffered
    /// whitespace so it isn't silently lost from `full_text`.
    pub fn finish(&mut self) -> String {
        match std::mem::replace(&mut self.state, StripperState::PassThrough) {
            StripperState::Initial { buffered_ws } => buffered_ws,
            _ => String::new(),
        }
    }
}

fn matching_closer(c: char) -> Option<char> {
    match c {
        '"' => Some('"'),
        '\'' => Some('\''),
        '\u{201C}' => Some('\u{201D}'),
        '\u{2018}' => Some('\u{2019}'),
        _ => None,
    }
}

fn strip_think_tags(token: &str, in_think: &mut bool, buffer: &mut String) -> String {
    // Fast path: nothing pending, no in-flight think block, and the token
    // contains no '<' that could start one — pass through with no allocation.
    if !*in_think && buffer.is_empty() && !token.contains('<') {
        return token.to_string();
    }

    let mut output = String::new();
    buffer.push_str(token);

    loop {
        if *in_think {
            if let Some(end_pos) = buffer.find("</think>") {
                *in_think = false;
                buffer.drain(..end_pos + 8);
            } else {
                if buffer.len() > 1000 {
                    buffer.clear();
                }
                break;
            }
        } else if let Some(start_pos) = buffer.find("<think>") {
            output.push_str(&buffer[..start_pos]);
            *in_think = true;
            buffer.drain(..start_pos + 7);
        } else if buffer.contains('<') && !buffer.ends_with('>') {
            // Potential partial tag at the end — hold back from the '<'
            if let Some(lt_pos) = buffer.rfind('<') {
                output.push_str(&buffer[..lt_pos]);
                buffer.drain(..lt_pos);
            }
            break;
        } else {
            output.push_str(buffer);
            buffer.clear();
            break;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_think_removes_complete_block() {
        let mut in_think = false;
        let mut buf = String::new();
        let result = strip_think_tags("<think>reasoning</think>Hello!", &mut in_think, &mut buf);
        assert!(result.contains("Hello!"));
        assert!(!result.contains("reasoning"));
        assert!(!in_think);
    }

    #[test]
    fn strip_think_handles_split_across_tokens() {
        let mut in_think = false;
        let mut buf = String::new();
        let r1 = strip_think_tags("<thi", &mut in_think, &mut buf);
        let r2 = strip_think_tags("nk>hidden</think>visible", &mut in_think, &mut buf);
        let combined = format!("{}{}", r1, r2);
        assert!(combined.contains("visible"));
        assert!(!combined.contains("hidden"));
    }

    #[test]
    fn strip_think_passes_through_normal_text() {
        let mut in_think = false;
        let mut buf = String::new();
        let r1 = strip_think_tags("Hello ", &mut in_think, &mut buf);
        let r2 = strip_think_tags("world!", &mut in_think, &mut buf);
        let combined = format!("{}{}", r1, r2);
        assert!(combined.contains("Hello"));
        assert!(combined.contains("world"));
    }

    fn run_stripper(tokens: &[&str]) -> String {
        let mut s = StreamingQuoteStripper::new();
        let mut out = String::new();
        for t in tokens {
            out.push_str(&s.push(t));
        }
        out.push_str(&s.finish());
        out
    }

    #[test]
    fn streaming_strip_curly_double_with_trailing_action() {
        // The screenshot bug: "OOH!" *bounces* — closer is mid-string, not at end.
        let out = run_stripper(&["\u{201C}OOH!\u{201D} *bounces*"]);
        assert_eq!(out, "OOH! *bounces*");
    }

    #[test]
    fn streaming_strip_split_across_many_tokens() {
        // Same case but each char chunk arrives separately, mimicking real streaming.
        let out = run_stripper(&["\u{201C}", "OOH", "!", "\u{201D} *bo", "unces*"]);
        assert_eq!(out, "OOH! *bounces*");
    }

    #[test]
    fn streaming_strip_straight_double_wrap() {
        let out = run_stripper(&["\"Yum!\""]);
        assert_eq!(out, "Yum!");
    }

    #[test]
    fn streaming_strip_no_quotes_passes_through() {
        let out = run_stripper(&["Hello, ", "world!"]);
        assert_eq!(out, "Hello, world!");
    }

    #[test]
    fn streaming_strip_preserves_apostrophes_inside_curly_double() {
        // Opener is curly double-quote; the straight ASCII apostrophe inside
        // LET'S must NOT match the closer (which is U+201D), so it's preserved.
        let out = run_stripper(&["\u{201C}LET'S GO\u{201D}"]);
        assert_eq!(out, "LET'S GO");
    }

    #[test]
    fn streaming_strip_leading_whitespace_then_quote() {
        let out = run_stripper(&["  \u{201C}hi\u{201D}"]);
        assert_eq!(out, "hi");
    }

    #[test]
    fn streaming_strip_unclosed_opener_loses_just_the_opener() {
        // Acceptable: matches the spirit of the prior fix — strip the leading
        // wrapper, keep the rest. We never see the missing closer, so the
        // stream stays in Quoted state until end.
        let out = run_stripper(&["\u{201C}unclosed"]);
        assert_eq!(out, "unclosed");
    }

    #[test]
    fn streaming_strip_only_whitespace_flushes_on_finish() {
        let out = run_stripper(&["   "]);
        assert_eq!(out, "   ");
    }
}
