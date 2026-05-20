//! Rolo Command Center — the single right-click-accessible surface where the user
//! configures Rolo's brain, manages what he knows about her, diagnoses
//! perception failures, and overrides weather sourcing. This module owns the
//! Rust side: persistence schema, validation, and the dedicated working
//! window. Each panel's frontend logic lives in `src/components/CommandCenter`.
//!
//! Rolo wants this surface boring on purpose — every other window in the app
//! is in-character (bubbles, status). The Command Center is where the user comes
//! when something needs fixing, so it stays out of his way.

pub mod diagnostics;
pub mod provider_slot;
pub mod settings;
pub mod window;

// Re-exports for convenience. The settings type is consumed by Tauri command
// signatures (Phase 8+ will use the short path); for now it's referenced via
// its full path inside `commands.rs` so we silence the unused-import warning
// rather than reshape downstream code.
#[allow(unused_imports)]
pub use diagnostics::{DiagnosticsReport, ProbeOutcome, ProbeStatus};
pub use provider_slot::SharedProviderSlot;
#[allow(unused_imports)]
pub use settings::CommandCenterSettings;
pub use window::create_command_center_window;

/// Returns `true` when the chat config can't possibly drive a working
/// brain — used by Phase 10 to decide whether to enter modal setup mode.
///
/// Rules (per PRD/rolo-command-center.md AC #15):
///   - `chat_config.json` is absent on disk: unconfigured (fresh install).
///   - Ollama selected: base_url or model empty → unconfigured.
///   - Anthropic selected: api_key empty → unconfigured.
///   - Gemini selected: api_key empty → unconfigured.
///   - openai_compat / huggingface / deepinfra (all stored as "openai_compat"
///     on disk): api_key or base_url empty → unconfigured.
///   - Unknown provider string → treated as unconfigured.
///
/// This intentionally does NOT make a network call. A misconfigured-but-
/// reachable provider (wrong key, dead URL) will produce a runtime error
/// on first chat, and the user can re-enter the Brain tab to fix it.
/// Forcing a probe here would make `cc_brain_needs_setup` slow on launch.
pub fn chat_config_is_unconfigured(cfg: &crate::chat::config::ChatConfig) -> bool {
    // Fresh install — no file on disk yet. `ChatConfig::load()` would have
    // returned defaults (default Ollama URL + model) which read as
    // "configured" below; this branch lets the caller distinguish "I have
    // working defaults from a save" vs "I've never saved anything".
    if !crate::chat::config::config_file_exists() {
        return true;
    }
    chat_config_content_is_unconfigured(cfg)
}

/// In-memory rule check that ignores whether a file is on disk.
///
/// Split out from `chat_config_is_unconfigured` so unit tests can exercise
/// the content rules without depending on `dirs::data_dir()` (which would
/// either pick up a developer's real config or fail to write in CI).
fn chat_config_content_is_unconfigured(cfg: &crate::chat::config::ChatConfig) -> bool {
    use crate::chat::config::Provider;
    let inf = &cfg.inference;
    match inf.provider {
        Provider::Ollama => inf.ollama.base_url.is_empty() || inf.ollama.model.is_empty(),
        Provider::Anthropic => inf.anthropic.api_key.is_empty(),
        Provider::Gemini => inf.gemini.api_key.is_empty(),
        // `OpenaiCompat` plus the UI-only Huggingface / Deepinfra
        // presets all serialize with HTTP creds on the openai_compat slot.
        Provider::OpenaiCompat | Provider::Huggingface | Provider::Deepinfra => {
            inf.openai_compat.api_key.is_empty() || inf.openai_compat.base_url.is_empty()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::config::ChatConfig;

    // These tests exercise the in-memory rules. The full
    // `chat_config_is_unconfigured` also reads `config_file_exists`, which
    // depends on the developer's actual `dirs::data_dir()` — exercising
    // that path here would either pollute the user's config or false-fail
    // depending on machine state. The `_content_` helper keeps the rule
    // logic testable in isolation.

    #[test]
    fn default_ollama_with_url_and_model_reads_as_configured() {
        let cfg = ChatConfig::default();
        assert!(!chat_config_content_is_unconfigured(&cfg));
    }

    #[test]
    fn empty_ollama_url_or_model_is_unconfigured() {
        use crate::chat::config::Provider;
        let mut cfg = ChatConfig::default();
        cfg.inference.provider = Provider::Ollama;
        cfg.inference.ollama.base_url.clear();
        assert!(chat_config_content_is_unconfigured(&cfg));

        let mut cfg = ChatConfig::default();
        cfg.inference.provider = Provider::Ollama;
        cfg.inference.ollama.model.clear();
        assert!(chat_config_content_is_unconfigured(&cfg));
    }

    #[test]
    fn anthropic_without_key_is_unconfigured() {
        use crate::chat::config::Provider;
        let mut cfg = ChatConfig::default();
        cfg.inference.provider = Provider::Anthropic;
        // key empty by default
        assert!(chat_config_content_is_unconfigured(&cfg));

        cfg.inference.anthropic.api_key = "sk-test".to_string();
        assert!(!chat_config_content_is_unconfigured(&cfg));
    }

    #[test]
    fn gemini_without_key_is_unconfigured() {
        use crate::chat::config::Provider;
        let mut cfg = ChatConfig::default();
        cfg.inference.provider = Provider::Gemini;
        assert!(chat_config_content_is_unconfigured(&cfg));

        cfg.inference.gemini.api_key = "g-test".to_string();
        assert!(!chat_config_content_is_unconfigured(&cfg));
    }

    #[test]
    fn openai_compat_requires_both_key_and_url() {
        use crate::chat::config::Provider;
        let mut cfg = ChatConfig::default();
        cfg.inference.provider = Provider::OpenaiCompat;
        // Empty by default — both missing.
        assert!(chat_config_content_is_unconfigured(&cfg));

        cfg.inference.openai_compat.api_key = "sk-x".to_string();
        assert!(chat_config_content_is_unconfigured(&cfg)); // url still empty

        cfg.inference.openai_compat.base_url = "https://api.openai.com/v1".to_string();
        assert!(!chat_config_content_is_unconfigured(&cfg));
    }

    // `unknown_provider_is_unconfigured` removed: with the typed
    // `Provider` enum, an unknown discriminant can no longer be
    // constructed — serde rejects it at deserialize time.
}
