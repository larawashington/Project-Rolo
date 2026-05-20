use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// LLM provider families Rolo can connect to. `OpenaiCompat` covers any
/// service that speaks OpenAI's chat-completions protocol — OpenAI itself,
/// LM Studio, Hugging Face Inference, DeepInfra, etc. The narrower
/// `Huggingface` / `Deepinfra` variants exist for the *UI dropdown* (`ui_provider`)
/// so the Brain tab can render their preset form fields; on disk
/// `inference.provider` for those choices stays `OpenaiCompat`.
///
/// Keep variant ordering + the snake_case rename in lockstep with
/// `src/types/commandCenter.ts::UiProvider` — the TS union pins the wire
/// strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    Ollama,
    OpenaiCompat,
    Anthropic,
    Gemini,
    Huggingface,
    Deepinfra,
}

impl Provider {
    /// Wire string used in logs and provider-id comparisons. Mirrors the
    /// snake_case rename so callsites match what serializes.
    pub fn as_wire_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::OpenaiCompat => "openai_compat",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
            Self::Huggingface => "huggingface",
            Self::Deepinfra => "deepinfra",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatConfig {
    #[serde(default)]
    pub inference: InferenceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    #[serde(default = "default_provider")]
    pub provider: Provider,
    /// UI-level provider choice (PRD/rolo-command-center.md Phase 5 — Brain
    /// tab). The runtime IGNORES this field — it only exists so the
    /// dropdown can restore its position on reload. HF and DeepInfra are
    /// stored with `provider: OpenaiCompat` but with `ui_provider: Huggingface`
    /// (or `Deepinfra`), so the UI knows to render the right preset.
    #[serde(default)]
    pub ui_provider: Option<Provider>,
    #[serde(default)]
    pub ollama: OllamaConfig,
    #[serde(default)]
    pub openai_compat: OpenAICompatConfig,
    #[serde(default)]
    pub anthropic: AnthropicConfig,
    #[serde(default)]
    pub gemini: GeminiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_url")]
    pub base_url: String,
    #[serde(default = "default_ollama_model")]
    pub model: String,
    #[serde(default = "default_context_window")]
    pub context_window: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatConfig {
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub model: String,
    #[serde(default = "default_context_window")]
    pub context_window: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    #[serde(default)]
    pub api_key: String,
    #[serde(default = "default_anthropic_model")]
    pub model: String,
    #[serde(default = "default_context_window")]
    pub context_window: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiConfig {
    #[serde(default)]
    pub api_key: String,
    #[serde(default = "default_gemini_model")]
    pub model: String,
    #[serde(default = "default_context_window")]
    pub context_window: u32,
}

fn default_provider() -> Provider {
    Provider::Ollama
}
fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}
fn default_ollama_model() -> String {
    crate::ollama::PRIMARY_MODEL.to_string()
}
fn default_context_window() -> u32 {
    4096
}
fn default_anthropic_model() -> String {
    "claude-haiku-4-5-20251001".to_string()
}
fn default_gemini_model() -> String {
    "gemini-2.0-flash".to_string()
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            ui_provider: None,
            ollama: OllamaConfig::default(),
            openai_compat: OpenAICompatConfig::default(),
            anthropic: AnthropicConfig::default(),
            gemini: GeminiConfig::default(),
        }
    }
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: default_anthropic_model(),
            context_window: default_context_window(),
        }
    }
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: default_gemini_model(),
            context_window: default_context_window(),
        }
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: default_ollama_url(),
            model: default_ollama_model(),
            context_window: default_context_window(),
        }
    }
}

impl Default for OpenAICompatConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: String::new(),
            model: String::new(),
            context_window: default_context_window(),
        }
    }
}

fn config_path() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("com.rolo.desktop-pet").join("config.json"))
}

/// PRD/rolo-command-center.md Phase 10 — fresh-install detection helper.
///
/// Returns `true` when `chat_config.json` is absent on disk. Used together
/// with `chat_config_is_unconfigured(&ChatConfig::load())` so the Command
/// Center can route a brand-new install into modal Brain-setup mode even
/// when `ChatConfig::load()` would otherwise produce a "default but usable"
/// configuration (default Ollama URL + model). A returning user whose
/// config sits on disk gets `false` here regardless of correctness — the
/// `chat_config_is_unconfigured` check handles corrupt/empty content.
pub fn config_file_exists() -> bool {
    match config_path() {
        Some(p) => p.exists(),
        None => false,
    }
}

impl ChatConfig {
    pub fn load() -> Self {
        let path = match config_path() {
            Some(p) => p,
            None => return Self::default(),
        };
        match std::fs::read_to_string(&path) {
            Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let path = config_path().ok_or("Cannot determine data directory")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Cannot create config directory: {}", e))?;
        }
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("Cannot serialize: {}", e))?;
        std::fs::write(&path, json).map_err(|e| format!("Cannot write config: {}", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PRD/rolo-command-center.md Phase 5: the new Anthropic / Gemini /
    /// ui_provider fields must round-trip through JSON unchanged.
    #[test]
    fn config_with_new_fields_round_trips_through_json() {
        let mut cfg = ChatConfig::default();
        cfg.inference.provider = Provider::Anthropic;
        cfg.inference.ui_provider = Some(Provider::Anthropic);
        cfg.inference.anthropic.api_key = "sk-ant-test".into();
        cfg.inference.anthropic.model = "claude-haiku-4-5-20251001".into();
        cfg.inference.gemini.api_key = "g-test".into();
        cfg.inference.gemini.model = "gemini-2.0-flash".into();

        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: ChatConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.inference.provider, Provider::Anthropic);
        assert_eq!(back.inference.ui_provider, Some(Provider::Anthropic));
        assert_eq!(back.inference.anthropic.api_key, "sk-ant-test");
        assert_eq!(back.inference.anthropic.model, "claude-haiku-4-5-20251001");
        assert_eq!(back.inference.gemini.api_key, "g-test");
        assert_eq!(back.inference.gemini.model, "gemini-2.0-flash");
    }

    /// An old `config.json` written before Phase 5 won't have the new keys.
    /// Serde defaults must hydrate them so Rolo's brain stays bootable.
    #[test]
    fn legacy_config_without_anthropic_gemini_loads_with_defaults() {
        let legacy = r#"{
            "inference": {
                "provider": "ollama",
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "gemma3:4b",
                    "context_window": 4096
                },
                "openai_compat": {
                    "base_url": "",
                    "api_key": "",
                    "model": "",
                    "context_window": 4096
                }
            }
        }"#;
        let cfg: ChatConfig = serde_json::from_str(legacy).expect("legacy parse");
        let inf = &cfg.inference;
        assert_eq!(inf.provider, Provider::Ollama);
        assert!(
            inf.ui_provider.is_none(),
            "ui_provider must default to None on legacy reads"
        );
        assert_eq!(inf.anthropic.model, "claude-haiku-4-5-20251001");
        assert!(inf.anthropic.api_key.is_empty());
        assert_eq!(inf.gemini.model, "gemini-2.0-flash");
        assert!(inf.gemini.api_key.is_empty());
    }
}
