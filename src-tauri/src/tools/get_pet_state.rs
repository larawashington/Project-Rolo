//! `get_pet_state` — describe what Rolo is doing right now.
//!
//! Reads `Pet::state()`, `last_interaction_elapsed_ms()`, and the public
//! `file_drag_active` field, then renders a one-line description like
//! `"Idle on screen, last interaction 8 minutes ago, no file being dragged."`
//!
//! We don't reuse `StateSnapshot::render_compact` here because that line
//! is mood-flavoured. The pet-state tool answers "what are you doing?",
//! not "how are you?", and Pass 2 routes those through different tools.

use std::sync::OnceLock;

use serde_json::{json, Value};

use crate::state_machine::PetState;

use super::{RoloTool, ToolContext, ToolError, ToolOutput};

pub struct GetPetState;

impl GetPetState {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GetPetState {
    fn default() -> Self {
        Self::new()
    }
}

fn schema_static() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true
        })
    })
}

/// Humanize a pet state into a short verb-phrase. Stays English-leaning so
/// Pass 2 can paraphrase naturally — the speaker model is told this is
/// context, not a verbatim quote.
fn describe_state(state: PetState) -> &'static str {
    match state {
        PetState::Idle => "Idle on screen",
        PetState::Sleeping => "Sleeping",
        PetState::WalkLeft => "Walking left",
        PetState::WalkRight => "Walking right",
        PetState::Happy => "Looking happy",
        PetState::DragHover => "Watching a file hover over him",
        PetState::Sniffing => "Sniffing a file",
        PetState::Eating => "Eating",
        PetState::Satisfied => "Satisfied after eating",
        PetState::Disappointed => "Disappointed",
    }
}

/// Format a millisecond elapsed value as a coarse human duration.
/// Buckets: <60s = "just now", <60min = "{n} minutes ago", else "{n} hours ago".
/// Coarse on purpose — small models are bad at large unit-precision math
/// and we don't want them paraphrasing "5400000ms" as a number.
fn humanize_elapsed_ms(ms: u64) -> String {
    let secs = ms / 1000;
    if secs < 30 {
        return "just now".to_string();
    }
    if secs < 60 {
        return "less than a minute ago".to_string();
    }
    let mins = secs / 60;
    if mins < 60 {
        if mins == 1 {
            return "1 minute ago".to_string();
        }
        return format!("{} minutes ago", mins);
    }
    let hours = mins / 60;
    if hours == 1 {
        return "1 hour ago".to_string();
    }
    format!("{} hours ago", hours)
}

#[async_trait::async_trait]
impl RoloTool for GetPetState {
    fn name(&self) -> &'static str {
        "get_pet_state"
    }

    fn description(&self) -> &'static str {
        "Read Rolo's current activity: which animation state he is in, how long since the last \
         user interaction, and whether a file is currently being dragged over him. Use when the \
         input asks what Rolo is doing right now."
    }

    fn schema(&self) -> &'static Value {
        schema_static()
    }

    async fn invoke(&self, _args: &Value, ctx: &ToolContext<'_>) -> Result<ToolOutput, ToolError> {
        let state = ctx.pet.state();
        let elapsed_ms = ctx.pet.last_interaction_elapsed_ms().max(0) as u64;
        let file_drag = ctx.pet.file_drag_active;

        let line = format!(
            "{}, last interaction {}, {}.",
            describe_state(state),
            humanize_elapsed_ms(elapsed_ms),
            if file_drag {
                "a file is being dragged over him"
            } else {
                "no file being dragged"
            }
        );

        Ok(ToolOutput {
            natural_language: line,
            citations: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::SharedMood;
    use crate::mood::MoodState;
    use crate::state_machine::Pet;
    use crate::state_snapshot::SystemClock;
    use crate::vault::embeddings::Embedder;
    use crate::vault::Vault;
    use std::io;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tempfile::TempDir;

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

    fn fresh_pet() -> Pet {
        Pet::new(0, 0, 1920, 1080, 100, 2)
    }

    fn fresh_vault() -> (TempDir, Arc<Vault>) {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("vault");
        let embedder: Arc<dyn Embedder> = Arc::new(DeadEmbedder);
        let vault = Vault::open_or_init_with_embedder(root, embedder);
        (tmp, vault)
    }

    #[tokio::test]
    async fn invoke_produces_non_empty_idle_line() {
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = SystemClock;
        let ctx = ToolContext {
            vault: &vault,
            mood: &mood,
            pet: &pet,
            clock: &clock,
        };
        let tool = GetPetState::new();
        let out = tool.invoke(&json!({}), &ctx).await.unwrap();
        assert!(!out.natural_language.is_empty());
        assert!(
            out.natural_language.starts_with("Idle"),
            "fresh pet should be idle, got: {}",
            out.natural_language
        );
        assert!(
            out.natural_language.contains("no file being dragged"),
            "fresh pet should report no drag, got: {}",
            out.natural_language
        );
        assert!(out.citations.is_empty());
    }

    #[test]
    fn humanize_elapsed_buckets_correctly() {
        assert_eq!(humanize_elapsed_ms(0), "just now");
        assert_eq!(humanize_elapsed_ms(15_000), "just now");
        assert_eq!(humanize_elapsed_ms(45_000), "less than a minute ago");
        assert_eq!(humanize_elapsed_ms(60_000), "1 minute ago");
        assert_eq!(humanize_elapsed_ms(8 * 60_000), "8 minutes ago");
        assert_eq!(humanize_elapsed_ms(60 * 60_000), "1 hour ago");
        assert_eq!(humanize_elapsed_ms(3 * 60 * 60_000), "3 hours ago");
    }
}
