use std::sync::{Arc, RwLock};

use crate::vault::chunker::Chunk;
use crate::vault::search::{FusedHit, HybridSearcher};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextSource {
    SqliteInbox,
    Wiki,
}

#[derive(Debug, Clone)]
pub struct PromptConfig {
    pub max_context_tokens: usize,
    pub max_user_context_tokens: usize,
    pub max_learned_behaviors_tokens: usize,
    pub bm25_top_k: usize,
    pub source: ContextSource,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 4096,
            max_user_context_tokens: 100,
            max_learned_behaviors_tokens: 50,
            bm25_top_k: 5,
            source: ContextSource::SqliteInbox,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InboxMemory {
    pub content: String,
    pub category: String,
}

pub trait MemoryInbox: Send + Sync {
    fn active_memories(&self) -> Vec<InboxMemory>;
}

pub struct PromptAssembler {
    searcher: Arc<HybridSearcher>,
    inbox: RwLock<Option<Arc<dyn MemoryInbox>>>,
    config: PromptConfig,
    fallback_system_prompt: String,
}

impl PromptAssembler {
    pub fn new(
        searcher: Arc<HybridSearcher>,
        config: PromptConfig,
        fallback_system_prompt: String,
    ) -> Self {
        Self {
            searcher,
            inbox: RwLock::new(None),
            config,
            fallback_system_prompt,
        }
    }

    pub fn set_inbox(&self, inbox: Arc<dyn MemoryInbox>) {
        let mut guard = self.inbox.write().unwrap_or_else(|p| p.into_inner());
        *guard = Some(inbox);
    }

    /// Build the system prompt. `state_slot` is the literal `[Mood: ... | Time: ...]`
    /// line (caller assembles it because they have the live mood/time data).
    pub fn assemble(&self, query: &str, state_slot: &str) -> String {
        // BM25 unavailable → verbatim fallback per PRD §8.6 row 1. An empty
        // (but Some) BM25 still runs through the slot pipeline so slot 2
        // (state) and slot 5 (response-rules tail) get composed.
        if !self.searcher.bm25_is_loaded() {
            return self.fallback_system_prompt.clone();
        }

        // Slot 1/4 pull specific files via BM25; slot 3 fuses BM25+vector via
        // HybridSearcher.
        let identity_chunks = self
            .searcher
            .bm25_chunks_in_file(crate::vault::config::WIKI_CORE_IDENTITY);
        let learned_chunks = self
            .searcher
            .bm25_chunks_in_file(crate::vault::config::WIKI_LEARNED_BEHAVIORS);

        // ---- Slot [1]: Core Identity --------------------------------------
        let identity_body = chunk_bodies_for_headings(
            &identity_chunks,
            &["Who I Am", "Core Traits", "Personality Boundaries"],
        );
        let slot1 = if identity_body.trim().is_empty() {
            self.fallback_system_prompt.clone()
        } else {
            identity_body
        };

        // ---- Slot [2]: Current State (caller-supplied) --------------------
        let slot2 = state_slot.trim().to_string();

        // ---- Slot [3]: User Context ---------------------------------------
        let slot3 = match self.config.source {
            ContextSource::SqliteInbox => {
                let inbox_guard = self.inbox.read().unwrap_or_else(|p| p.into_inner());
                if let Some(inbox) = inbox_guard.as_ref() {
                    let mems = inbox.active_memories();
                    format_inbox(&mems, self.config.max_user_context_tokens)
                } else {
                    String::new()
                }
            }
            ContextSource::Wiki => {
                let hits = self
                    .searcher
                    .search_user_scope(query, self.config.bm25_top_k);
                format_wiki_user_context(&hits, self.config.max_user_context_tokens)
            }
        };

        // ---- Slot [4]: Learned Behaviors ----------------------------------
        // `[PINNED]` is internal metadata for the dreaming compiler — it
        // marks rules the compiler must not rewrite. The LLM should never
        // see the marker (or it parrots it back as a message prefix), so
        // strip it here along with the meta explainer paragraph.
        let slot4_full =
            sanitize_learned_behaviors(&collect_all_chunk_bodies_from(&learned_chunks));
        let slot4 = crate::tokens::truncate_to_budget(
            &slot4_full,
            self.config.max_learned_behaviors_tokens,
        );

        // Drop slot 4 if remaining headroom is tight (< 50 tokens) per §8.5.
        let used_so_far = crate::tokens::estimate_tokens(&slot1)
            + crate::tokens::estimate_tokens(&slot2)
            + crate::tokens::estimate_tokens(&slot3)
            + crate::tokens::estimate_tokens(&slot4);
        let slot4 = if self.config.max_context_tokens.saturating_sub(used_so_far) < 50 {
            String::new()
        } else {
            slot4
        };

        // ---- Slot [5]: Chat Rules -----------------------------------------
        let rules_body = chunk_bodies_for_headings(&identity_chunks, &["Response Rules"]);
        let slot5 = if rules_body.trim().is_empty() {
            // Pull just the response-rules section out of the fallback if we can.
            extract_response_rules(&self.fallback_system_prompt)
        } else {
            rules_body
        };

        // ---- Compose ------------------------------------------------------
        let mut out = String::new();
        out.push_str(slot1.trim());
        if !slot2.is_empty() {
            out.push_str("\n\n");
            out.push_str(&slot2);
        }
        if !slot3.trim().is_empty() {
            out.push_str("\n\n");
            out.push_str(slot3.trim());
        }
        if !slot4.trim().is_empty() {
            out.push_str("\n\nLearned behaviors:\n");
            out.push_str(slot4.trim());
        }
        if !slot5.trim().is_empty() {
            out.push_str("\n\nResponse rules:\n");
            out.push_str(slot5.trim());
        }
        out
    }
}

/// Strip the leading `# Title\n## Heading\n` prefix from a chunk body so the
/// content reads naturally when concatenated into a prompt.
fn strip_chunk_headings(body: &str) -> &str {
    let mut rest = body;
    if let Some(after_h1) = rest.strip_prefix("# ") {
        if let Some(idx) = after_h1.find('\n') {
            rest = &after_h1[idx + 1..];
        }
    }
    if let Some(after_h2) = rest.strip_prefix("## ") {
        if let Some(idx) = after_h2.find('\n') {
            rest = &after_h2[idx + 1..];
        }
    }
    rest
}

/// Pull chunk bodies whose heading matches one of `headings`, in the order
/// `headings` lists them. Operates on a pre-fetched chunk slice so the caller
/// can reuse one `chunks_in_file` lookup for several heading subsets.
fn chunk_bodies_for_headings(chunks: &[Chunk], headings: &[&str]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for h in headings {
        for c in chunks {
            if c.heading == *h {
                parts.push(strip_chunk_headings(&c.body).trim().to_string());
            }
        }
    }
    parts.join("\n\n")
}

/// Strip the `[PINNED] ` marker from rule lines and drop the editorial
/// paragraph that explains what `[PINNED]` means. Both are useful to the
/// dreaming compiler and to a human reading the markdown, but if either
/// reaches the LLM it tends to echo `[PINNED]` as a message prefix.
fn sanitize_learned_behaviors(body: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut skip_paragraph = false;
    for raw_line in body.lines() {
        let line = raw_line.trim_end();
        if line.is_empty() {
            skip_paragraph = false;
            out.push(String::new());
            continue;
        }
        if skip_paragraph {
            continue;
        }
        if line.trim_start().starts_with("Lines below starting with") {
            skip_paragraph = true;
            continue;
        }
        if let Some(rest) = line.strip_prefix("[PINNED] ") {
            out.push(rest.to_string());
        } else {
            out.push(line.to_string());
        }
    }
    // Collapse runs of blank lines and trim.
    let joined = out.join("\n");
    let mut collapsed = String::with_capacity(joined.len());
    let mut prev_blank = true;
    for line in joined.lines() {
        let blank = line.trim().is_empty();
        if blank && prev_blank {
            continue;
        }
        collapsed.push_str(line);
        collapsed.push('\n');
        prev_blank = blank;
    }
    collapsed.trim().to_string()
}

fn collect_all_chunk_bodies_from(chunks: &[Chunk]) -> String {
    chunks
        .iter()
        .map(|c| strip_chunk_headings(&c.body).trim().to_string())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn format_inbox(memories: &[InboxMemory], max_tokens: usize) -> String {
    if memories.is_empty() {
        return String::new();
    }
    let lines: Vec<String> = memories
        .iter()
        .map(|m| format!("- {}", m.content))
        .collect();
    crate::tokens::pop_oldest_to_fit(lines, "What I know about my human:\n", "\n", max_tokens)
        .unwrap_or_default()
}

fn format_wiki_user_context(hits: &[FusedHit], max_tokens: usize) -> String {
    // HybridSearcher::search_user_scope already filters to user/ and
    // relationships/ — we don't re-filter here.
    let blocks: Vec<String> = hits
        .iter()
        .map(|h| {
            let body = strip_chunk_headings(&h.chunk.body).trim().to_string();
            let path = h
                .chunk
                .file_path
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            format!("From {}#{}:\n{}", path, h.chunk.heading, body)
        })
        .collect();
    crate::tokens::pop_oldest_to_fit(blocks, "", "\n\n", max_tokens).unwrap_or_default()
}

/// Best-effort: pull the "Response rules:" tail out of the fallback prompt for
/// when the wiki has lost its response-rules chunk. Returns the whole fallback
/// if the marker isn't present.
fn extract_response_rules(fallback: &str) -> String {
    if let Some(idx) = fallback.find("Response rules:") {
        fallback[idx + "Response rules:".len()..].trim().to_string()
    } else {
        fallback.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::bm25::BM25Index;
    use crate::vault::bootstrap::bootstrap_vault;
    use crate::vault::embeddings::{Embedder, EmbeddingIndex};
    use crate::vault::search::HybridSearcher;
    use std::io;
    use std::sync::{Arc, RwLock};
    use std::time::Duration;
    use tempfile::TempDir;

    const FAKE_FALLBACK: &str = "FALLBACK_SYSTEM_PROMPT_MARKER\n\nResponse rules:\n- be brief.";

    /// Stub embedder that always returns None — drives the BM25-only path so
    /// PromptAssembler tests don't depend on a live Ollama.
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

    fn build_searcher(bm25: Option<BM25Index>) -> Arc<HybridSearcher> {
        let bm25 = Arc::new(RwLock::new(bm25));
        let emb: Arc<RwLock<Option<EmbeddingIndex>>> = Arc::new(RwLock::new(None));
        let embedder: Arc<dyn Embedder> = Arc::new(DeadEmbedder);
        Arc::new(HybridSearcher::new(bm25, emb, embedder))
    }

    fn build_assembler_from_bootstrap() -> (TempDir, PromptAssembler) {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("vault");
        bootstrap_vault(&root).unwrap();
        let bm25 = BM25Index::build(&root.join("wiki")).unwrap();
        let searcher = build_searcher(Some(bm25));
        let asm =
            PromptAssembler::new(searcher, PromptConfig::default(), FAKE_FALLBACK.to_string());
        (tmp, asm)
    }

    #[test]
    fn bm25_none_returns_fallback_verbatim() {
        let searcher = build_searcher(None);
        let asm =
            PromptAssembler::new(searcher, PromptConfig::default(), FAKE_FALLBACK.to_string());
        let out = asm.assemble("hi", "[Mood: ok]");
        assert_eq!(out, FAKE_FALLBACK);
    }

    #[test]
    fn empty_index_falls_back_for_slot1_and_slot5() {
        let tmp = TempDir::new().unwrap();
        let bm25 = BM25Index::build(tmp.path()).unwrap();
        let searcher = build_searcher(Some(bm25));
        let asm =
            PromptAssembler::new(searcher, PromptConfig::default(), FAKE_FALLBACK.to_string());
        let out = asm.assemble("hi", "[Mood: ok]");
        // slot 1 = full fallback (verbatim); slot 5 = response-rules-only tail.
        assert!(out.contains("FALLBACK_SYSTEM_PROMPT_MARKER"));
        assert!(out.contains("Response rules:"));
        assert!(out.contains("[Mood: ok]"));
    }

    #[test]
    fn full_assembly_contains_required_strings() {
        let (_tmp, asm) = build_assembler_from_bootstrap();
        let out = asm.assemble(
            "how is the user feeling?",
            "[Mood: curious | Energy: 0.6 | Time: 2:30 PM]",
        );
        // §11.5 #4: contains "You are Rolo" — actually wiki Who-I-Am uses "I am Rolo".
        // Our slot text starts with "I am Rolo, a small desktop pet..." per the bootstrap content.
        assert!(
            out.contains("I am Rolo"),
            "missing 'I am Rolo' in:\n{}",
            out
        );
        // §11.5 #5: contains "Response rules"
        assert!(out.contains("Response rules"));
        // §11.5 #6: state slot
        assert!(out.contains("[Mood: curious"));
        // Slot 4 should pull in the pinned rule content but strip the
        // `[PINNED]` marker — the LLM mimics it as a message prefix
        // (see `sanitize_learned_behaviors`).
        assert!(
            out.contains("Never use emojis"),
            "missing pinned rule body in:\n{}",
            out
        );
        assert!(
            !out.contains("[PINNED]"),
            "[PINNED] marker leaked into prompt:\n{}",
            out
        );
        assert!(
            !out.contains("Lines below starting with"),
            "meta explainer leaked into prompt:\n{}",
            out
        );
    }

    struct MockInbox(Vec<InboxMemory>);
    impl MemoryInbox for MockInbox {
        fn active_memories(&self) -> Vec<InboxMemory> {
            self.0.clone()
        }
    }

    #[test]
    fn sqlite_inbox_mode_includes_inbox_content() {
        let (_tmp, asm) = build_assembler_from_bootstrap();
        let inbox = Arc::new(MockInbox(vec![InboxMemory {
            content: "user is named Lara".to_string(),
            category: "fact".to_string(),
        }]));
        asm.set_inbox(inbox);
        let out = asm.assemble("anything", "[Mood: ok]");
        assert!(out.contains("Lara"), "missing 'Lara' in:\n{}", out);
        assert!(out.contains("What I know about my human:"));
    }

    #[test]
    fn token_budget_truncates_huge_learned_behaviors() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("vault");
        bootstrap_vault(&root).unwrap();
        // Overwrite learned-behaviors.md with a giant body so slot 4 needs truncation.
        let huge = format!(
            "<!-- last_compiled_event_ts: null -->\n# Learned Behaviors\n\n## Discovered\n{}\n",
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet ".repeat(500)
        );
        std::fs::write(root.join("wiki/personality/learned-behaviors.md"), huge).unwrap();
        let bm25 = BM25Index::build(&root.join("wiki")).unwrap();
        let searcher = build_searcher(Some(bm25));
        let cfg = PromptConfig {
            max_learned_behaviors_tokens: 30,
            ..Default::default()
        };
        let asm = PromptAssembler::new(searcher, cfg, FAKE_FALLBACK.to_string());
        let out = asm.assemble("anything", "[Mood: ok]");

        // Find slot 4 region in output and assert its char count is bounded
        // (~4 chars/token * 30 tokens = ~120 chars + small framing tolerance).
        if let Some(start) = out.find("Learned behaviors:\n") {
            let after = &out[start + "Learned behaviors:\n".len()..];
            let region = match after.find("\n\nResponse rules:") {
                Some(end) => &after[..end],
                None => after,
            };
            assert!(
                region.chars().count() <= 30 * 4 + 8,
                "learned-behaviors region too long: {} chars",
                region.chars().count()
            );
        }
        // The full huge body's middle/tail words must be absent (proves truncation).
        // Pick a token that only exists very late in the repeated pattern.
        // Since the pattern repeats, just assert overall length sanity.
        assert!(
            out.chars().count() < 5000,
            "prompt blew past sanity ceiling"
        );
    }
}
