pub const RETENTION_DAYS: i64 = 30;
pub const SOFT_TOKEN_CAP: usize = 8000;
pub const BOOTSTRAP_TOKEN_BUDGET: usize = 4000;
/// Bumped from 1 → 2 with the dreaming compiler foundation (PRD §A2).
/// Old `meta.json` files migrate via `VaultMeta::load_or_default`'s
/// "preserve created_at, reset everything else" path (see `meta.rs`).
pub const SCHEMA_VERSION: u32 = 2;

// Wiki file paths, relative to `wiki/`. Centralized so a rename surfaces as a
// compile error in every consumer instead of silently breaking BM25 lookups.
pub const WIKI_CORE_IDENTITY: &str = "personality/core-identity.md";
pub const WIKI_LEARNED_BEHAVIORS: &str = "personality/learned-behaviors.md";

/// Path prefixes (relative to `wiki/`) the dreaming compiler is allowed to
/// write/edit. Anything outside these prefixes is read-only — protects
/// `personality/core-identity.md` and other curated content from being
/// overwritten by a hallucinated patch (PRD §A2, §B).
///
/// `personality/learned-behaviors.md` is a single explicit file (not a
/// directory) — Rolo's compiled habit log is the only personality file the
/// dreamer can touch.
pub const WIKI_WRITABLE_PREFIXES: &[&str] = &[
    "user/",
    "world/",
    "relationships/",
    "personality/learned-behaviors.md",
];
