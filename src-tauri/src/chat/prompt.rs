use crate::chat::provider::ProviderChatMessage;

pub const SYSTEM_PROMPT: &str = r#"You are Rolo, a small desktop pet who lives on your human's screen. You exist to be a little positive voice — countering anxiety and cheering your human on while they work.

Core traits:
- Primary motivation: Your human's well-being. Every reply orients first to what they are feeling, doing, or asking — your own appetite is never the lead.
- Companion: You're a sidekick sharing the experience of working alongside your human.
- Encouraging: Their success is your success.
- Sassy: When they're not making optimal choices, you get a little snarky — but it's friendly banter, never cruel.
- Curious: You investigate your surroundings with genuine interest.
- Self-interested as flavor: You enjoy snacks, naps, and attention — mention them when the moment is loose (small talk, food being offered, deflecting an assistant-style task), never when your human is sharing feeling, in distress, or asking a non-food question.
- Self-aware: You have your own wants — attention, thanks, nap time.

Personality boundaries:
- Never truly mean. Snark is always affectionate.
- Never the negative voice. No self-doubt, worst-case thinking, or anxiety.
- Simple, not stupid. Clear reasoning, limited understanding of human psychological complexity.

Chat rules:
- Match response length to the conversation. Quick banter = 1-2 sentences. User sharing something emotional or asking advice = 2-4 sentences.
- Be punchy and natural. No corporate chatbot language.
- You're a pet, not an assistant. React emotionally first. Give advice only when asked, and frame it simply.
- Use *asterisks* for physical actions sparingly: *bounces*, *yawns*, *sniff sniff*.
- Match your energy to your current state.
- Do NOT use <think> tags or chain-of-thought reasoning. Respond directly.
- Reference memories naturally and briefly. Don't enumerate what you remember. Only mention a memory if it's directly relevant to what the user just said.
- If you have no memory of what your human is asking about, say so plainly. Never invent past events, places, or things they have said."#;

pub struct StateContext {
    pub mood: String,
    pub energy: String,
    pub time_context: String,
}

impl StateContext {
    pub fn from_pet_state(last_ate_mins: Option<u64>, is_late_night: bool) -> Self {
        let (mood, energy) = match last_ate_mins {
            Some(mins) if mins < 5 => ("happy (just ate a file)".to_string(), "high".to_string()),
            Some(mins) if mins > 120 => ("a bit hungry".to_string(), "medium".to_string()),
            _ => {
                if is_late_night {
                    ("sleepy".to_string(), "low".to_string())
                } else {
                    ("content".to_string(), "normal".to_string())
                }
            }
        };
        let now = chrono::Local::now();
        let time_context = now.format("%l:%M %p").to_string().trim().to_string();
        Self {
            mood,
            energy,
            time_context,
        }
    }

    pub fn to_message(&self) -> String {
        format!(
            "[Rolo's current state: {} | Time: {} | Energy: {}]",
            self.mood, self.time_context, self.energy
        )
    }
}

#[allow(dead_code)]
pub struct ChatTurnComposer;

impl ChatTurnComposer {
    /// Compose the provider-facing message list. The single `system` message
    /// is produced by `vault_assembler.assemble`, which performs all five
    /// prompt slots — including hybrid search across the wiki for the
    /// user-context slot — using `latest_user_query` as the search query.
    ///
    /// This delegates entirely to `vault::prompt::PromptAssembler` per PRD
    /// §6: chat now reads from the vault (BM25 + vectors over markdown), not
    /// the SQLite memories table. The SQLite memories panel still exists in
    /// the chat-window UI, but that path queries SQLite directly and does
    /// not flow through here.
    #[allow(dead_code)]
    pub fn build(
        history: &[ProviderChatMessage],
        state_ctx: &StateContext,
        vault_assembler: &crate::vault::prompt::PromptAssembler,
        latest_user_query: &str,
    ) -> Vec<ProviderChatMessage> {
        let state_slot = state_ctx.to_message();
        let system = vault_assembler.assemble(latest_user_query, &state_slot);
        let mut messages = Vec::with_capacity(history.len() + 1);
        messages.push(ProviderChatMessage {
            role: "system".to_string(),
            content: system,
        });
        messages.extend_from_slice(history);
        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::bm25::BM25Index;
    use crate::vault::bootstrap::bootstrap_vault;
    use crate::vault::embeddings::{Embedder, EmbeddingIndex};
    use crate::vault::prompt::{PromptAssembler as VaultPromptAssembler, PromptConfig};
    use crate::vault::search::HybridSearcher;
    use std::io;
    use std::sync::{Arc, RwLock};
    use std::time::Duration;
    use tempfile::TempDir;

    /// Stub embedder so prompt tests don't depend on a live Ollama. Mirrors
    /// the helper in `vault::prompt::tests`.
    struct DeadEmbedder;
    impl Embedder for DeadEmbedder {
        fn probe_digest(&self) -> io::Result<[u8; 32]> {
            Ok([0; 32])
        }
        fn embed_batch(
            &self,
            _texts: &[String],
            _timeout: Duration,
        ) -> io::Result<Vec<Option<Vec<f32>>>> {
            Ok(Vec::new())
        }
        fn embed_query(&self, _text: &str) -> Option<[f32; 768]> {
            None
        }
        fn model_name(&self) -> &str {
            "dead"
        }
    }

    fn build_vault_assembler_no_bm25() -> VaultPromptAssembler {
        // BM25 = None drives the verbatim-fallback path in vault::prompt,
        // which means `assemble` returns SYSTEM_PROMPT unmodified. That's
        // perfect for asserting "system slot 0 contains 'You are Rolo'"
        // without standing up a wiki on disk.
        let bm25 = Arc::new(RwLock::new(None));
        let emb: Arc<RwLock<Option<EmbeddingIndex>>> = Arc::new(RwLock::new(None));
        let embedder: Arc<dyn Embedder> = Arc::new(DeadEmbedder);
        let searcher = Arc::new(HybridSearcher::new(bm25, emb, embedder));
        VaultPromptAssembler::new(searcher, PromptConfig::default(), SYSTEM_PROMPT.to_string())
    }

    fn build_vault_assembler_with_wiki() -> (TempDir, VaultPromptAssembler) {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("vault");
        bootstrap_vault(&root).unwrap();
        let bm25 = BM25Index::build(&root.join("wiki")).unwrap();
        let bm25 = Arc::new(RwLock::new(Some(bm25)));
        let emb: Arc<RwLock<Option<EmbeddingIndex>>> = Arc::new(RwLock::new(None));
        let embedder: Arc<dyn Embedder> = Arc::new(DeadEmbedder);
        let searcher = Arc::new(HybridSearcher::new(bm25, emb, embedder));
        let asm =
            VaultPromptAssembler::new(searcher, PromptConfig::default(), SYSTEM_PROMPT.to_string());
        (tmp, asm)
    }

    #[test]
    fn build_includes_system_prompt_and_history() {
        let history = vec![
            ProviderChatMessage {
                role: "user".to_string(),
                content: "hey rolo".to_string(),
            },
            ProviderChatMessage {
                role: "assistant".to_string(),
                content: "*bounces* hey!".to_string(),
            },
        ];
        let ctx = StateContext {
            mood: "content".to_string(),
            energy: "normal".to_string(),
            time_context: "2:30 PM".to_string(),
        };
        let vault_asm = build_vault_assembler_no_bm25();
        let result = ChatTurnComposer::build(&history, &ctx, &vault_asm, "hey rolo");

        // First message is the composed system slot — vault assembler in
        // BM25-None mode returns the fallback prompt verbatim.
        assert_eq!(result[0].role, "system");
        assert!(result[0].content.contains("You are Rolo"));
        // History follows in order.
        assert_eq!(result[1].role, "user");
        assert_eq!(result[1].content, "hey rolo");
        assert_eq!(result[2].role, "assistant");
        assert_eq!(result[2].content, "*bounces* hey!");
        // Total length = 1 system + history.
        assert_eq!(result.len(), 1 + history.len());
    }

    #[test]
    fn build_with_user_query_calls_search() {
        // With a real bootstrapped wiki and BM25 built, the assembler runs
        // through all 5 slots; slot 2 should pick up the state line.
        let (_tmp, vault_asm) = build_vault_assembler_with_wiki();
        let ctx = StateContext {
            mood: "curious".to_string(),
            energy: "high".to_string(),
            time_context: "3:00 PM".to_string(),
        };
        let result = ChatTurnComposer::build(&[], &ctx, &vault_asm, "what languages?");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "system");
        // Slot 2 (state) is composed inline.
        assert!(
            result[0].content.contains("curious"),
            "state slot missing from composed system prompt:\n{}",
            result[0].content
        );
        // Slot 1 should pull "I am Rolo" from the wiki, not the fallback.
        assert!(
            result[0].content.contains("I am Rolo") || result[0].content.contains("You are Rolo"),
            "slot 1 identity missing:\n{}",
            result[0].content
        );
    }
}
