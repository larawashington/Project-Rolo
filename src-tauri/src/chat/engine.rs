use std::sync::{Arc, Mutex};

use crate::chat::config::ChatConfig;
use crate::chat::context::ContextBudget;
use crate::chat::memory::MemoryExtractor;
use crate::chat::openai_provider::OpenAICompatProvider;
use crate::chat::prompt::StateContext;
use crate::chat::provider::{GenerationConfig, InferenceProvider, ProviderChatMessage};
use crate::chat::safety::{FilterResult, SafetyFilter};
use crate::chat::store::ChatStore;
use crate::command_center::SharedProviderSlot;
use crate::commands::{SharedMood, SharedPet};
use crate::state_snapshot::{StateSnapshot, SystemClock};

/// Build the closure that splices a (chat) first-user-turn for the rolo-brain
/// path. The closure takes the user message and returns the byte-exact
/// content for the final `role: "user"` message sent to PRIMARY_MODEL.
///
/// Shape:
/// - `tool_augmentation = None`:  `{state_slot}\n\n{user_message}`
/// - `tool_augmentation = Some(L)`: `{state_slot}\n{L}\n\n{user_message}`
///
/// `L` is the dispatcher's Ok-value — already wrapped in the
/// `[Context from <tool>: "..."]` envelope (see
/// `tools/dispatcher.rs::run_tool_and_log`), so we paste it verbatim.
///
/// Whitespace rule (per `data/finetune/sft-v1/runtime_contract.md` §2):
/// exactly **one blank line** between the bracket-stack and the user
/// message. The bracketed lines themselves are joined by single newlines.
fn build_rolo_brain_first_user_turn<'a>(
    state_slot: &'a str,
    tool_augmentation: Option<&'a str>,
) -> impl Fn(&str) -> String + 'a {
    move |user_message: &str| {
        let mut combined = String::with_capacity(
            state_slot.len()
                + tool_augmentation.map(|l| l.len() + 1).unwrap_or(0)
                + 2
                + user_message.len(),
        );
        combined.push_str(state_slot);
        if let Some(line) = tool_augmentation {
            combined.push('\n');
            combined.push_str(line);
        }
        combined.push_str("\n\n");
        combined.push_str(user_message);
        combined
    }
}

pub struct ChatEngine {
    store: Arc<Mutex<ChatStore>>,
    /// Atomically-swappable provider slot. Phase 4 of
    /// PRD/rolo-command-center.md introduced this in place of the fixed
    /// `primary`/`fallback` `Box<dyn …>` pair so the Brain panel can hot-swap
    /// Rolo's brain at runtime. Each call to `send_message` takes a snapshot
    /// before awaiting the provider, so in-flight inferences survive a swap.
    provider_slot: SharedProviderSlot,
    context_budget: Mutex<ContextBudget>,
    /// Optional vault logger — when wired, every assistant turn appends a
    /// `ExperienceEvent::Chat` to today's JSONL so the dream compiler can see
    /// what Rolo and the user actually said.
    experience_logger: Option<Arc<crate::vault::logger::ExperienceLogger>>,
    /// Optional vault prompt assembler — when wired, the system prompt is
    /// composed from the wiki via BM25 + vector hybrid search instead of the
    /// SQLite memories table. Wired in `lib.rs::setup` after the vault is
    /// opened. Without this, the engine falls back to a minimal prompt
    /// (system + state + history) so tests that don't stand up a vault still
    /// run.
    vault_assembler: Option<Arc<crate::vault::prompt::PromptAssembler>>,
    /// Optional shared mood handle. When wired alongside `pet`, `send_message`
    /// captures a live `StateSnapshot` for the state-slot text instead of the
    /// mood-blind legacy `StateContext::from_pet_state(None, ...)` path.
    /// PRD/rolo-prompt-consolidation T1.
    mood: Option<SharedMood>,
    /// Optional shared pet handle. Paired with `mood` above.
    pet: Option<SharedPet>,
    /// Optional vault handle for the tool-layer dispatcher
    /// (PRD/rolo-tool-layer.md T5). When wired alongside `tool_registry`,
    /// `mood`, and `pet`, `send_message` consults the dispatcher first; on
    /// success the natural-language tool result is appended to the system
    /// message. On any failure the legacy `vault_assembler.assemble(...)`
    /// path runs unchanged. Without this handle the dispatcher is skipped
    /// entirely so existing tests compile.
    vault: Option<Arc<crate::vault::Vault>>,
    /// Optional tool registry for the tool-layer dispatcher
    /// (PRD/rolo-tool-layer.md T5). Paired with `vault` above.
    tool_registry: Option<Arc<crate::tools::ToolRegistry>>,
}

impl ChatEngine {
    /// Build an initial `Arc<dyn InferenceProvider>` from a `ChatConfig`.
    /// Phase 4 PRD/rolo-command-center.md: the engine no longer owns the
    /// provider — `lib.rs::setup` constructs one provider Arc, seeds a
    /// `SharedProviderSlot` with it, and hands the slot to both the chat
    /// engine and the dreaming poll loop. Exposing the construction logic as
    /// a free function lets both `lib.rs` and `new` share one source of
    /// truth (and keeps Phase 5's `cc_apply_brain` honest when it rebuilds a
    /// provider for the swap).
    pub fn build_provider_from_config(config: &ChatConfig) -> Arc<dyn InferenceProvider> {
        use crate::chat::config::Provider;
        let inf = &config.inference;
        match inf.provider {
            Provider::Ollama => {
                let ollama_url = format!("{}/v1", inf.ollama.base_url);
                Arc::new(OpenAICompatProvider::new_with_id(
                    ollama_url,
                    None,
                    inf.ollama.model.clone(),
                    "ollama",
                ))
            }
            Provider::Anthropic => {
                Arc::new(crate::chat::anthropic_provider::AnthropicProvider::new(
                    inf.anthropic.api_key.clone(),
                    inf.anthropic.model.clone(),
                ))
            }
            Provider::Gemini => Arc::new(crate::chat::gemini_provider::GeminiProvider::new(
                inf.gemini.api_key.clone(),
                inf.gemini.model.clone(),
            )),
            // `OpenaiCompat` — and the UI-only Huggingface / Deepinfra
            // presets, which are also stored with `provider: OpenaiCompat`
            // plus a preset base URL — all land here.
            Provider::OpenaiCompat | Provider::Huggingface | Provider::Deepinfra => {
                Arc::new(OpenAICompatProvider::new(
                    inf.openai_compat.base_url.clone(),
                    Some(inf.openai_compat.api_key.clone()),
                    inf.openai_compat.model.clone(),
                ))
            }
        }
    }

    /// Construct a new `ChatEngine` reading its provider from the given slot.
    /// The slot is the single source of truth — Phase 5's `cc_apply_brain`
    /// hot-swaps providers by calling `slot.replace(new)` and the engine
    /// picks up the change on the next `send_message`.
    ///
    /// `config` is consulted only to size the context budget; the provider
    /// itself comes from the slot. Pre-Phase 4 callers used
    /// `ChatEngine::from_config` which built the provider and the engine
    /// together; that signature was removed when the fallback path was
    /// retired (one selected brain serves all four call-sites per
    /// PRD/rolo-command-center.md decision #5).
    pub fn new(
        store: Arc<Mutex<ChatStore>>,
        config: &ChatConfig,
        provider_slot: SharedProviderSlot,
    ) -> Self {
        use crate::chat::config::Provider;
        let inf = &config.inference;
        let context_window = match inf.provider {
            Provider::Ollama => inf.ollama.context_window,
            Provider::Anthropic => inf.anthropic.context_window,
            Provider::Gemini => inf.gemini.context_window,
            Provider::OpenaiCompat | Provider::Huggingface | Provider::Deepinfra => {
                inf.openai_compat.context_window
            }
        };

        Self {
            store,
            provider_slot,
            context_budget: Mutex::new(ContextBudget::new(context_window)),
            experience_logger: None,
            vault_assembler: None,
            mood: None,
            pet: None,
            vault: None,
            tool_registry: None,
        }
    }

    /// Attach an `ExperienceLogger` so every successful chat turn lands in the
    /// vault as an `ExperienceEvent::Chat`. Called by `lib.rs::setup` after the
    /// vault is opened. Without this, chat turns are persisted to SQLite but
    /// not visible to the dream compiler.
    pub fn with_experience_logger(
        mut self,
        logger: Arc<crate::vault::logger::ExperienceLogger>,
    ) -> Self {
        self.experience_logger = Some(logger);
        self
    }

    /// Attach a vault `PromptAssembler` so the system prompt is composed via
    /// hybrid search over the wiki (PRD §6 — "loop closure"). Without this,
    /// `send_message` falls back to a minimal prompt (system + state +
    /// history) so unit tests that don't stand up a vault still run.
    pub fn with_vault_assembler(
        mut self,
        assembler: Arc<crate::vault::prompt::PromptAssembler>,
    ) -> Self {
        self.vault_assembler = Some(assembler);
        self
    }

    /// Attach shared `mood` and `pet` handles so `send_message` can capture a
    /// live `StateSnapshot` and surface the actual mood (anxious / lonely /
    /// hungry / etc.) in the system prompt's state slot. Without this, chat is
    /// mood-blind: the legacy `StateContext::from_pet_state(None, ...)` path
    /// always renders "content / normal". PRD/rolo-prompt-consolidation T1.
    pub fn with_state_handles(mut self, mood: SharedMood, pet: SharedPet) -> Self {
        self.mood = Some(mood);
        self.pet = Some(pet);
        self
    }

    /// Attach the vault for the tool-layer dispatcher (PRD/rolo-tool-layer.md
    /// T5). Without this handle the chat path skips the dispatcher and runs
    /// the legacy `vault_assembler` path unchanged.
    pub fn with_vault(mut self, vault: Arc<crate::vault::Vault>) -> Self {
        self.vault = Some(vault);
        self
    }

    /// Attach the tool registry for the tool-layer dispatcher
    /// (PRD/rolo-tool-layer.md T5). Paired with `with_vault`.
    pub fn with_tool_registry(mut self, registry: Arc<crate::tools::ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    pub async fn send_message(
        &self,
        session_id: &str,
        user_text: &str,
        token_tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<(String, i64), String> {
        // Store the user message
        {
            let store = self.store.lock().map_err(|e| e.to_string())?;
            store
                .insert_message(session_id, "user", user_text)
                .map_err(|e| e.to_string())?;
        }

        // Rule-based memory extraction on user input
        {
            let user_msgs = vec![user_text];
            let extracted = MemoryExtractor::extract_rules(&user_msgs);
            if !extracted.is_empty() {
                MemoryExtractor::store_memories(&self.store, session_id, extracted).ok();
            }
        }

        // Build conversation history from DB
        let history = {
            let store = self.store.lock().map_err(|e| e.to_string())?;
            let msgs = store
                .get_session_messages(session_id)
                .map_err(|e| e.to_string())?;
            msgs.into_iter()
                .map(|m| ProviderChatMessage {
                    role: m.role,
                    content: m.content,
                })
                .collect::<Vec<_>>()
        };

        // Build the state slot. PRD/rolo-prompt-consolidation T1: when both
        // mood and pet handles are wired (lib.rs::setup), capture a live
        // `StateSnapshot` so the slot reflects actual mood / energy / social /
        // hunger. Otherwise fall back to the legacy mood-blind `StateContext`
        // so tests that don't stand up the full state machine still run.
        let state_slot = match (&self.mood, &self.pet) {
            (Some(mood), Some(pet)) => {
                let pet_guard = pet.lock().map_err(|e| e.to_string())?;
                StateSnapshot::capture(mood, &pet_guard, &SystemClock).render_for_speech()
            }
            _ => {
                let now = chrono::Local::now();
                let hour = now.format("%H").to_string().parse::<u32>().unwrap_or(12);
                let is_late_night = !(6..22).contains(&hour);
                StateContext::from_pet_state(None, is_late_night).to_message()
            }
        };

        // Tool-layer dispatcher (PRD/rolo-tool-layer.md §5.6 / T5). Same flag
        // and behavior as the bubble path: on success append the
        // natural-language tool result to the system message; on any error
        // (including `Disabled` / `Unknown` / `Speak`) fall through to the
        // legacy `vault_assembler.assemble(...)` path unchanged. This must
        // sit on the same system block — the legacy `[Mood: ...]` line in
        // `state_slot` already lives there, and Gemma anchors better to a
        // single consolidated system message than to fragmented ones.
        let tool_augmentation: Option<String> =
            if let (Some(vault), Some(registry), Some(mood), Some(pet)) =
                (&self.vault, &self.tool_registry, &self.mood, &self.pet)
            {
                // Clone the pet snapshot under the lock so the dispatcher's
                // async work doesn't hold the std::sync::Mutex across an await.
                let pet_clone = {
                    let g = pet.lock().map_err(|e| e.to_string())?;
                    g.clone()
                };
                let clock = crate::state_snapshot::SystemClock;
                // Tool routing is Ollama-only (PRD/rolo-command-center.md
                // Phase 5 Step E). Snapshot the slot's current provider id
                // BEFORE await so we don't race with a hot-swap, and pass it
                // to the dispatcher. A non-Ollama brain returns
                // `Err(NonOllamaBrain)` which falls through like every other
                // dispatcher error.
                let is_ollama = self.provider_slot.snapshot().provider_id() == "ollama";
                // Every `Err(_)` arm (Disabled / Speak / Unknown /
                // ToolFailed / NonOllamaBrain / etc.) means "fall through to
                // the legacy assembler" — `.ok()` collapses them all to None.
                crate::tools::dispatcher::dispatch(
                    user_text,
                    vault.as_ref(),
                    mood,
                    &pet_clone,
                    &clock,
                    registry.as_ref(),
                    is_ollama,
                )
                .await
                .ok()
            } else {
                None
            };

        // v2 (rolo-brain) detour: the SFT corpus has no `system` role and
        // every state block lives inside the first user turn. Sending the
        // multi-slot vault prompt to PRIMARY_MODEL puts every inference
        // off-distribution. For rolo-brain we drop the system message
        // entirely and fold the state slot into the latest user message,
        // matching `data/finetune/sft-v1/runtime_contract.md` §2.
        //
        // Identity / response rules are in the weights. Tool-layer
        // retrieved-memory snippets — when the dispatcher returns one —
        // land here too at the slot §7 reserves: a bracketed line stacked
        // immediately after the state block, before the blank-line
        // separator. Without this splice the SFT brain runs tool-blind
        // even though every other provider gets the augmentation, which
        // is the regression the runtime contract §7 was reserved to fix.
        // Non-rolo-brain providers (Anthropic / Gemini / fallback Ollama)
        // continue down the legacy path so their system prompts still work.
        let active_provider = self.provider_slot.snapshot();
        let is_rolo_brain = active_provider.model_name() == crate::ollama::PRIMARY_MODEL;

        let mut prompt_messages = if is_rolo_brain {
            let first_user_turn =
                build_rolo_brain_first_user_turn(&state_slot, tool_augmentation.as_deref());
            let mut msgs = Vec::with_capacity(history.len());
            for (i, m) in history.iter().enumerate() {
                if i == history.len() - 1 && m.role == "user" {
                    let combined = first_user_turn(&m.content);
                    msgs.push(ProviderChatMessage {
                        role: "user".to_string(),
                        content: combined,
                    });
                } else {
                    msgs.push(m.clone());
                }
            }
            // If the last history entry wasn't a user turn (defensive — chat
            // engine always inserts the user message first), append a synthetic
            // user turn carrying just the state slot so we never send an
            // assistant-only prompt to the model. We deliberately do NOT
            // splice the tool-augmentation line here: with no user message
            // the augmentation has nothing to anchor against, and the
            // fallthrough is only hit by malformed test wiring.
            if msgs.last().map(|m| m.role.as_str()) != Some("user") {
                msgs.push(ProviderChatMessage {
                    role: "user".to_string(),
                    content: state_slot.clone(),
                });
            }
            msgs
        } else if let Some(va) = &self.vault_assembler {
            // Legacy path — multi-slot system prompt from the vault assembler.
            // Used for Anthropic / Gemini / fallback Ollama (gemma3:4b).
            let mut system = va.assemble(user_text, &state_slot);
            if let Some(line) = &tool_augmentation {
                system.push('\n');
                system.push_str(line);
            }
            let mut msgs = Vec::with_capacity(history.len() + 1);
            msgs.push(ProviderChatMessage {
                role: "system".to_string(),
                content: system,
            });
            msgs.extend_from_slice(&history);
            msgs
        } else {
            let base_system = crate::chat::prompt::SYSTEM_PROMPT.to_string();
            let system = if let Some(line) = &tool_augmentation {
                format!("{base_system}\n{line}")
            } else {
                base_system
            };
            let mut msgs = Vec::with_capacity(history.len() + 2);
            msgs.push(ProviderChatMessage {
                role: "system".to_string(),
                content: system,
            });
            msgs.push(ProviderChatMessage {
                role: "system".to_string(),
                content: state_slot,
            });
            msgs.extend_from_slice(&history);
            msgs
        };

        // Context budget: trim old messages if conversation is getting long
        {
            let budget = self.context_budget.lock().map_err(|e| e.to_string())?;
            if budget.needs_summarization(&prompt_messages) {
                let (_, kept) = ContextBudget::split_for_summarization(&prompt_messages);
                prompt_messages = kept;
                log::info!("[Rolo] Context trimmed — conversation was getting long");
            }
        }

        let gen_config = GenerationConfig::default();

        // Reuse the `active_provider` snapshot captured above for the
        // model-name check so the generation call and the rolo-brain
        // formatting decision can never disagree (the slot can hot-swap
        // mid-method otherwise). The original "snapshot BEFORE await"
        // intent — that an in-flight call survives `cc_apply_brain` — still
        // holds, just one snapshot earlier in the function.
        let provider = active_provider;
        let full_text = provider
            .generate(prompt_messages, gen_config, token_tx)
            .await
            .map_err(|e| format!("Provider failed: {}", e))?;

        // Strip any <think> tags. Quote stripping happens inline in the
        // streaming providers (TODO(finetune): both go away once the Rolo
        // finetune ships) so `full_text` is already quote-free here.
        let cleaned = strip_think_tags(&full_text);

        // Safety filter on the final response
        let final_text = match SafetyFilter::check(&cleaned) {
            FilterResult::Safe(text) => text,
            FilterResult::Filtered { replacement, .. } => {
                log::warn!("[Rolo] Response was safety-filtered");
                replacement
            }
        };

        // Store Rolo's response
        let msg_id = {
            let store = self.store.lock().map_err(|e| e.to_string())?;
            store
                .insert_message(session_id, "assistant", &final_text)
                .map_err(|e| e.to_string())?
        };

        // A successful turn lifts the social bar. Sentiment detection lands
        // later; for now every turn that reached the user is a small social
        // boost. Mood handle is optional so test wiring without state still
        // works.
        if let Some(mood) = &self.mood {
            if let Ok(mut g) = mood.lock() {
                g.apply_event(crate::mood::MoodEvent::ChatMessage { positive: true });
            }
        }

        // Append a Chat event to the vault (one event per assistant turn).
        // Errors are swallowed inside ExperienceLogger::log so a bad write
        // never breaks the chat path — Rolo's voice is more important than
        // the ledger here, and the logger already self-recovers on next call.
        if let Some(logger) = &self.experience_logger {
            logger.log(&crate::vault::events::ExperienceEvent::Chat {
                event_id: crate::vault::events::new_id(),
                ts: chrono::Local::now(),
                session: session_id.to_string(),
                user: user_text.to_string(),
                rolo: final_text.clone(),
                mood_signal: None,
            });
        }

        Ok((final_text, msg_id))
    }

    pub async fn close_session(&self, session_id: &str, source: &str) -> Result<(), String> {
        // For check-in sessions, classify mood from conversation
        if source == "checkin" {
            let history = {
                let store = self.store.lock().map_err(|e| e.to_string())?;
                store
                    .get_session_messages(session_id)
                    .map_err(|e| e.to_string())?
            };

            if !history.is_empty() {
                let user_msgs: Vec<&str> = history
                    .iter()
                    .filter(|m| m.role == "user")
                    .map(|m| m.content.as_str())
                    .collect();

                let (mood, confidence) = classify_mood_from_text(&user_msgs);

                let store = self.store.lock().map_err(|e| e.to_string())?;
                store
                    .insert_mood_reading(session_id, &mood, confidence)
                    .map_err(|e| e.to_string())?;
                log::info!(
                    "[Rolo] Mood classified from chat: {} (confidence: {:.2})",
                    mood,
                    confidence,
                );
            }
        }

        // Close the session in the store
        {
            let store = self.store.lock().map_err(|e| e.to_string())?;
            store
                .close_session(session_id, None)
                .map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub async fn health_check(&self) -> bool {
        self.provider_slot.snapshot().health_check().await
    }
}

fn classify_mood_from_text(user_messages: &[&str]) -> (String, f64) {
    let combined: String = user_messages.join(" ").to_lowercase();

    let positive = [
        "great",
        "good",
        "happy",
        "awesome",
        "fantastic",
        "wonderful",
        "amazing",
        "excited",
        "love",
        "perfect",
        "excellent",
    ];
    let negative = [
        "bad",
        "terrible",
        "awful",
        "stressed",
        "anxious",
        "sad",
        "tired",
        "exhausted",
        "frustrated",
        "angry",
        "worried",
        "overwhelmed",
        "depressed",
    ];

    let pos_count = positive.iter().filter(|w| combined.contains(**w)).count();
    let neg_count = negative.iter().filter(|w| combined.contains(**w)).count();

    if pos_count == 0 && neg_count == 0 {
        return ("okay".to_string(), 0.3);
    }

    let total = (pos_count + neg_count) as f64;
    if pos_count > neg_count {
        let confidence = (pos_count as f64 / total).min(0.95);
        ("good".to_string(), confidence)
    } else if neg_count > pos_count {
        let confidence = (neg_count as f64 / total).min(0.95);
        ("not great".to_string(), confidence)
    } else {
        ("okay".to_string(), 0.5)
    }
}

fn strip_think_tags(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            result = format!("{}{}", &result[..start], &result[end + 8..]);
        } else {
            // Unclosed think tag — remove everything from <think> onwards
            result = result[..start].to_string();
            break;
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
impl ChatEngine {
    /// Test-only constructor — bypasses the config-driven provider wiring so
    /// tests can inject a `MockProvider` without standing up a real Ollama or
    /// OpenAI endpoint. Takes an `Arc<dyn InferenceProvider>` (not `Box`) so
    /// the underlying slot machinery is exercised the same way production
    /// code exercises it.
    pub(crate) fn for_test(
        store: Arc<Mutex<ChatStore>>,
        primary: Arc<dyn InferenceProvider>,
    ) -> Self {
        Self {
            store,
            provider_slot: SharedProviderSlot::new(primary),
            context_budget: Mutex::new(ContextBudget::new(8192)),
            experience_logger: None,
            vault_assembler: None,
            mood: None,
            pet: None,
            vault: None,
            tool_registry: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------
    // Rolo-brain first-user-turn shape (runtime_contract.md §2 + §7).
    //
    // These pin the byte-exact wire bytes for the PRIMARY_MODEL path so the
    // SFT brain's input distribution stays on the contract no matter what
    // the dispatcher returns. A regression here puts every chat turn off-
    // distribution and is invisible without these golden strings.
    // ---------------------------------------------------------------------

    #[test]
    fn rolo_brain_first_user_turn_without_tool_augmentation_matches_contract_section_2() {
        let state_slot =
            "[Mood: content | Energy: medium | Social: ok | Time: 2:30 PM | Hunger: just ate]";
        let build = build_rolo_brain_first_user_turn(state_slot, None);
        let got = build("hello rolo");
        // §2: exactly one blank line between the bracket-stack and the user
        // message. No tool line stacked above. This is the regression guard
        // for the pre-T5/T6 byte shape.
        let expected =
            "[Mood: content | Energy: medium | Social: ok | Time: 2:30 PM | Hunger: just ate]\n\
                        \n\
                        hello rolo";
        assert_eq!(got, expected);
    }

    #[test]
    fn rolo_brain_first_user_turn_with_tool_augmentation_matches_contract_section_7() {
        let state_slot =
            "[Mood: content | Energy: medium | Social: ok | Time: 2:30 PM | Hunger: just ate]";
        // Dispatcher Ok-value shape (see tools/dispatcher.rs::run_tool_and_log) —
        // the bracketed envelope is ready-to-paste, double-quote
        // wrapping included.
        let tool_line = "[Context from search_vault: \"we went hiking last weekend\"]";
        let build = build_rolo_brain_first_user_turn(state_slot, Some(tool_line));
        let got = build("remember when we went hiking");
        // §7: tool line stacked immediately after the state block with a
        // single `\n` separator, then exactly one blank line before the
        // user message.
        let expected =
            "[Mood: content | Energy: medium | Social: ok | Time: 2:30 PM | Hunger: just ate]\n\
                        [Context from search_vault: \"we went hiking last weekend\"]\n\
                        \n\
                        remember when we went hiking";
        assert_eq!(got, expected);
    }

    /// Belt-and-braces — confirm the two layouts differ by exactly the
    /// bracketed line + one separator newline. If this ever drifts, one of
    /// the goldens above is wrong.
    #[test]
    fn rolo_brain_layouts_differ_by_exactly_the_tool_line_plus_one_newline() {
        let state_slot = "[Mood: a | Energy: b | Social: c | Time: 1:00 PM]";
        let tool_line = "[Context from get_mood_state: \"feeling great\"]";
        let user = "ping";
        let none_out = build_rolo_brain_first_user_turn(state_slot, None)(user);
        let some_out = build_rolo_brain_first_user_turn(state_slot, Some(tool_line))(user);
        assert_eq!(some_out.len(), none_out.len() + tool_line.len() + 1);
    }

    #[test]
    fn strip_think_tags_removes_thinking() {
        assert_eq!(
            strip_think_tags("<think>reasoning here</think>hey!"),
            "hey!"
        );
    }

    #[test]
    fn strip_think_tags_handles_unclosed() {
        assert_eq!(strip_think_tags("hello<think>partial"), "hello");
    }

    #[test]
    fn strip_think_tags_no_tags_unchanged() {
        assert_eq!(strip_think_tags("just text"), "just text");
    }

    #[test]
    fn strip_think_tags_multiple() {
        assert_eq!(
            strip_think_tags("<think>a</think>hello <think>b</think>world"),
            "hello world"
        );
    }

    #[test]
    fn mood_positive() {
        let msgs = vec!["I'm feeling great today!", "Everything is awesome"];
        let (mood, conf) = classify_mood_from_text(&msgs);
        assert_eq!(mood, "good");
        assert!(conf > 0.5);
    }

    #[test]
    fn mood_negative() {
        let msgs = vec!["I'm so stressed and tired"];
        let (mood, conf) = classify_mood_from_text(&msgs);
        assert_eq!(mood, "not great");
        assert!(conf > 0.5);
    }

    #[test]
    fn mood_neutral() {
        let msgs = vec!["just working on some code"];
        let (mood, conf) = classify_mood_from_text(&msgs);
        assert_eq!(mood, "okay");
        assert!(conf < 0.5);
    }

    #[test]
    fn mood_mixed() {
        let msgs = vec!["I'm happy but also a bit stressed"];
        let (mood, _) = classify_mood_from_text(&msgs);
        assert_eq!(mood, "okay");
    }

    #[tokio::test]
    async fn chat_engine_logs_chat_event_after_send() {
        use crate::chat::mock_provider::MockProvider;
        use crate::chat::store::ChatStore;
        use crate::vault::events::ExperienceEvent;
        use crate::vault::logger::ExperienceLogger;
        use std::fs;
        use tempfile::TempDir;

        // 1. Build a ChatStore in memory and create a session.
        let store = Arc::new(Mutex::new(
            ChatStore::open_in_memory().expect("in-memory store"),
        ));
        let session_id = {
            let s = store.lock().unwrap();
            s.create_session("test", "hello").expect("session create")
        };

        // 2. Stand up a real ExperienceLogger pointed at a tempdir.
        let tmp = TempDir::new().unwrap();
        let events_dir = tmp.path().join("events");
        fs::create_dir_all(&events_dir).unwrap();
        let logger = ExperienceLogger::new(events_dir.clone());

        // 3. Build a ChatEngine with the MockProvider and wire the logger in.
        let mock: Arc<dyn InferenceProvider> = Arc::new(MockProvider {
            response: "Hi there friend".to_string(),
            delay_ms: 0,
            should_fail: false,
        });
        let engine = ChatEngine::for_test(store, mock).with_experience_logger(logger.clone());

        // 4. Send a message end-to-end.
        let user_text = "hello rolo";
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(32);
        // Drain the streaming channel concurrently so MockProvider can send tokens.
        let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
        let (final_text, _msg_id) = engine
            .send_message(&session_id, user_text, tx)
            .await
            .expect("send_message must succeed");
        drain.await.unwrap();

        assert_eq!(final_text, "Hi there friend");

        // 5. Drop engine and logger so writes are flushed; the logger's Drop
        //    appends a SessionEnd, so the Chat event is the second-to-last line.
        drop(engine);
        // Logger is held by an Arc inside the engine + this scope; dropping
        // our local Arc reference here triggers Drop because the engine
        // already released its clone.
        drop(logger);

        let today = chrono::Local::now().date_naive();
        let log_path = events_dir.join(format!("{today}.jsonl"));
        let content = fs::read_to_string(&log_path).expect("today's JSONL must exist");
        let lines: Vec<&str> = content.lines().collect();
        assert!(
            lines.len() >= 2,
            "expected at least Chat + SessionEnd, got {} lines",
            lines.len()
        );

        // Find the last non-SessionEnd line — that's our Chat event.
        let chat_line = lines
            .iter()
            .rev()
            .find_map(|l| {
                let parsed: ExperienceEvent = serde_json::from_str(l).ok()?;
                if matches!(parsed, ExperienceEvent::SessionEnd { .. }) {
                    None
                } else {
                    Some(parsed)
                }
            })
            .expect("must find a non-SessionEnd event");

        match chat_line {
            ExperienceEvent::Chat {
                user,
                rolo,
                session,
                ..
            } => {
                assert_eq!(user, user_text, "logged user text must match");
                assert_eq!(rolo, "Hi there friend", "logged rolo text must match");
                assert_eq!(session, session_id, "session id must match");
            }
            other => panic!("expected Chat event, got {:?}", other),
        }
    }

    /// PRD/rolo-command-center.md Phase 5 AC #5: when `cc_apply_brain`
    /// hot-swaps the provider in `SharedProviderSlot`, the very next
    /// `send_message` must route through the new provider. We don't need to
    /// invoke the command itself — the slot's `replace` is the unit under
    /// test, and the engine's contract is "snapshot before await".
    #[tokio::test]
    async fn chat_engine_uses_new_provider_after_slot_swap() {
        use crate::chat::mock_provider::MockProvider;
        use crate::chat::store::ChatStore;
        use crate::command_center::SharedProviderSlot;

        let store = Arc::new(Mutex::new(
            ChatStore::open_in_memory().expect("in-memory store"),
        ));
        let session_id = {
            let s = store.lock().unwrap();
            s.create_session("test", "hello").expect("session create")
        };

        // First provider — recognizable response.
        let first: Arc<dyn InferenceProvider> = Arc::new(MockProvider {
            response: "FIRST-BRAIN".to_string(),
            delay_ms: 0,
            should_fail: false,
        });
        let slot = SharedProviderSlot::new(Arc::clone(&first));

        // Build the engine reading from the slot (mimics the production
        // setup-block wiring in lib.rs).
        let context_budget_window = 8192u32;
        let engine = ChatEngine {
            store: Arc::clone(&store),
            provider_slot: slot.clone(),
            context_budget: Mutex::new(ContextBudget::new(context_budget_window)),
            experience_logger: None,
            vault_assembler: None,
            mood: None,
            pet: None,
            vault: None,
            tool_registry: None,
        };

        // First call — exercises `first`.
        let (tx1, mut rx1) = tokio::sync::mpsc::channel::<String>(32);
        let drain1 = tokio::spawn(async move { while rx1.recv().await.is_some() {} });
        let (out1, _) = engine
            .send_message(&session_id, "ping", tx1)
            .await
            .expect("first send_message");
        drain1.await.unwrap();
        assert_eq!(out1, "FIRST-BRAIN", "first call uses the seeded provider");

        // Hot-swap mid-session — same path cc_apply_brain takes.
        let second: Arc<dyn InferenceProvider> = Arc::new(MockProvider {
            response: "SECOND-BRAIN".to_string(),
            delay_ms: 0,
            should_fail: false,
        });
        let _previous = slot.replace(Arc::clone(&second));

        // Second call — must now exercise `second`.
        let (tx2, mut rx2) = tokio::sync::mpsc::channel::<String>(32);
        let drain2 = tokio::spawn(async move { while rx2.recv().await.is_some() {} });
        let (out2, _) = engine
            .send_message(&session_id, "ping again", tx2)
            .await
            .expect("second send_message");
        drain2.await.unwrap();
        assert_eq!(
            out2, "SECOND-BRAIN",
            "after slot.replace, next call must use the new provider"
        );
    }

    /// PRD/rolo-tool-layer.md T5 byte-identity check: with neither `vault`
    /// nor `tool_registry` wired, `send_message` MUST take exactly the same
    /// path as before T5 — `tool_augmentation` resolves to `None` and the
    /// system message contains no `[Context from ...]` block.
    ///
    /// This is the entire unique surface T5 introduces in chat. The richer
    /// "wire a tool registry, get an augmented prompt" flow is already
    /// covered by the in-module dispatcher tests in
    /// `tools/dispatcher.rs::tests`.
    #[tokio::test]
    async fn chat_engine_unwired_dispatcher_is_byte_identical() {
        use crate::chat::provider::{
            GenerationConfig, InferenceError, InferenceProvider, ProviderChatMessage,
        };
        use crate::chat::store::ChatStore;
        use async_trait::async_trait;

        /// Minimal capturing provider — pushes the prompt sent by the engine
        /// into a shared vector so the test can inspect the system message.
        struct CapturingProvider {
            sink: Arc<Mutex<Vec<Vec<ProviderChatMessage>>>>,
            response: String,
        }

        #[async_trait]
        impl InferenceProvider for CapturingProvider {
            async fn generate(
                &self,
                messages: Vec<ProviderChatMessage>,
                _config: GenerationConfig,
                token_tx: tokio::sync::mpsc::Sender<String>,
            ) -> Result<String, InferenceError> {
                if let Ok(mut g) = self.sink.lock() {
                    g.push(messages);
                }
                // Stream the canned response as one token so the receiver
                // sees something but the test can drain it cheaply.
                let _ = token_tx.send(self.response.clone()).await;
                Ok(self.response.clone())
            }
            async fn health_check(&self) -> bool {
                true
            }
            fn provider_name(&self) -> &str {
                "Capturing"
            }
            fn model_name(&self) -> &str {
                "capturing-v1"
            }
            fn provider_id(&self) -> &str {
                "mock"
            }
        }

        let store = Arc::new(Mutex::new(
            ChatStore::open_in_memory().expect("in-memory store"),
        ));
        let session_id = {
            let s = store.lock().unwrap();
            s.create_session("test", "hello").expect("session create")
        };

        let sink: Arc<Mutex<Vec<Vec<ProviderChatMessage>>>> = Arc::new(Mutex::new(Vec::new()));
        let provider: Arc<dyn InferenceProvider> = Arc::new(CapturingProvider {
            sink: Arc::clone(&sink),
            response: "Hi there friend".to_string(),
        });

        // No `with_vault`, no `with_tool_registry` — the dispatcher branch
        // must resolve to `None` and the prompt must be byte-identical to
        // the pre-T5 shape.
        let engine = ChatEngine::for_test(store, provider);

        let user_text = "remember when we went hiking";
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(32);
        let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
        let (_final_text, _msg_id) = engine
            .send_message(&session_id, user_text, tx)
            .await
            .expect("send_message must succeed");
        drain.await.unwrap();

        let captured = sink.lock().unwrap();
        assert_eq!(captured.len(), 1, "expected exactly one provider call");
        let prompt = &captured[0];

        // The unwired path uses the two-system-message fallback: SYSTEM_PROMPT
        // then state_slot. Neither must contain a tool-context envelope.
        for msg in prompt {
            assert!(
                !msg.content.contains("[Context from"),
                "unwired chat path must not contain tool augmentation, got: {}",
                msg.content
            );
        }
    }
}
