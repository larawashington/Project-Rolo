//! `get_mood_state` — return Rolo's current mood as a one-line summary.
//!
//! Wraps `StateSnapshot::capture(...).render_compact()`, the existing
//! single source of truth for mood-rendering. We deliberately reuse that
//! renderer rather than reformatting raw bars: every call site converging
//! on `StateSnapshot` is the whole point of B1, and adding a fourth
//! divergent renderer here would re-open the bug B1 just closed.

use std::sync::OnceLock;

use serde_json::{json, Value};

use crate::state_snapshot::StateSnapshot;

use super::{RoloTool, ToolContext, ToolError, ToolOutput};

pub struct GetMoodState;

impl GetMoodState {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GetMoodState {
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

#[async_trait::async_trait]
impl RoloTool for GetMoodState {
    fn name(&self) -> &'static str {
        "get_mood_state"
    }

    fn description(&self) -> &'static str {
        "Read Rolo's current mood, energy, social, hunger, and time of day. Use when the input \
         asks how Rolo is feeling, or when the response should be coloured by his current mood."
    }

    fn schema(&self) -> &'static Value {
        schema_static()
    }

    async fn invoke(&self, _args: &Value, ctx: &ToolContext<'_>) -> Result<ToolOutput, ToolError> {
        // No-arg tool — small models will sometimes still pass random keys.
        // We accept silently rather than rejecting, because the only fix
        // for a stray key is to ignore it.
        let snapshot = StateSnapshot::capture(ctx.mood, ctx.pet, ctx.clock);
        let line = snapshot.render_compact();
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
    use crate::state_snapshot::Clock;
    use crate::vault::embeddings::Embedder;
    use crate::vault::Vault;
    use chrono::TimeZone;
    use std::io;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tempfile::TempDir;

    /// Test clock pinned to a deterministic moment — afternoon so the
    /// energy/mood rendering doesn't drift with wall time.
    struct FixedClock(chrono::DateTime<chrono::Local>);
    impl Clock for FixedClock {
        fn now(&self) -> chrono::DateTime<chrono::Local> {
            self.0
        }
    }

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
    async fn invoke_produces_non_empty_natural_language() {
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = FixedClock(
            chrono::Local
                .with_ymd_and_hms(2026, 5, 6, 14, 0, 0)
                .single()
                .unwrap(),
        );

        let ctx = ToolContext {
            vault: &vault,
            mood: &mood,
            pet: &pet,
            clock: &clock,
        };
        let tool = GetMoodState::new();
        let out = tool.invoke(&json!({}), &ctx).await.unwrap();
        assert!(
            !out.natural_language.is_empty(),
            "mood line must be non-empty"
        );
        // Sanity: the compact renderer always emits "Mood: ..." up front.
        assert!(
            out.natural_language.starts_with("Mood: "),
            "expected compact mood line, got: {}",
            out.natural_language
        );
        assert!(out.citations.is_empty());
    }

    #[tokio::test]
    async fn invoke_tolerates_extraneous_args() {
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = FixedClock(
            chrono::Local
                .with_ymd_and_hms(2026, 5, 6, 14, 0, 0)
                .single()
                .unwrap(),
        );

        let ctx = ToolContext {
            vault: &vault,
            mood: &mood,
            pet: &pet,
            clock: &clock,
        };
        let tool = GetMoodState::new();
        // Small models sometimes hallucinate keys — must not error.
        let out = tool
            .invoke(&json!({ "garbage": 1, "stray": "key" }), &ctx)
            .await
            .unwrap();
        assert!(!out.natural_language.is_empty());
    }
}
