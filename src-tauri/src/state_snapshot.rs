//! Single source of truth for rendering Rolo's state into a prompt-ready
//! string. Replaces four ad-hoc renderers (`mood::render_state_line`,
//! `speech::build_context_prompt`, `tick::build_state_slot_placeholder`,
//! `chat::prompt::StateContext::from_pet_state`) per PRD/rolo-prompt-
//! consolidation.md §5 T1.
//!
//! Why one type: every state-rendering call site converges on the same shape
//! — bucketed mood word, bucketed energy/social, time-of-day, the current
//! `PetState`, optional file-drag info — and the four legacy renderers had
//! drifted into producing slightly different surface forms for the same
//! underlying snapshot. PRD 3 (tool layer) needs a *stable* contract: one
//! type, three render flavors, each tested for byte-identity.
//!
//! `Clock` is a trait so snapshot tests can pin {morning, afternoon,
//! late-night} without globally mutating `chrono::Local::now()`. PRD §6 A2.
//!
//! `PetStateView` and `FileDragView` are projection types — `PetState` is an
//! enum and `Pet` is a live runtime struct, neither of which we want to
//! expose directly to a future serializer.

use crate::commands::SharedMood;
use crate::mood::{self, bucket_energy, bucket_hunger, bucket_sass, bucket_social, MoodSnapshot};
use crate::state_machine::{Pet, PetState};

// ---------------------------------------------------------------------------
// Clock — trait so tests can pin time-of-day without touching `chrono::Local`.
// ---------------------------------------------------------------------------

/// A source of "now". Production uses `SystemClock`; tests use `MockClock`.
pub trait Clock: Send + Sync {
    fn now(&self) -> chrono::DateTime<chrono::Local>;
}

/// Wall-clock implementation. Always returns `chrono::Local::now()`.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> chrono::DateTime<chrono::Local> {
        chrono::Local::now()
    }
}

/// Test-only fixed clock. Use to build deterministic snapshot tests.
#[cfg(test)]
pub struct MockClock(pub chrono::DateTime<chrono::Local>);

#[cfg(test)]
impl Clock for MockClock {
    fn now(&self) -> chrono::DateTime<chrono::Local> {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Projection types
// ---------------------------------------------------------------------------

/// Pet-runtime data we care about for rendering. `Pet::state()` returns the
/// raw enum, but we render via Debug formatting (matching legacy behavior of
/// `tick::build_state_slot_placeholder`) so we keep both the enum (for the
/// `state_word` form used by the speech path's `[State: ...]` suffix) and a
/// pre-formatted Debug string here.
#[derive(Debug, Clone)]
pub struct PetStateView {
    pub state: PetState,
    pub last_interaction_ms: u64,
}

/// File-drag context. Only present when Rolo is hovering a real file (not a
/// cursor drag). `kind` is reserved for a future MIME/category — today this
/// is always "file" because the underlying Pet doesn't expose a kind yet.
#[derive(Debug, Clone)]
pub struct FileDragView {
    pub kind: String,
    pub since_ms: u64,
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// A frozen view of Rolo's state at one instant. Cheap to build — locks the
/// `SharedMood` mutex once and produces a `MoodSnapshot` (which already
/// allocates only the mood-word string).
pub struct StateSnapshot {
    pub mood: MoodSnapshot,
    pub pet: PetStateView,
    pub now: chrono::DateTime<chrono::Local>,
    pub file_drag: Option<FileDragView>,
    /// Internal copy of the bucketed bars + sass-token derived from the live
    /// `MoodState` at capture time. We keep this here rather than re-deriving
    /// from `MoodSnapshot` because (a) `MoodSnapshot` doesn't carry the sass
    /// modifier, and (b) hunger uses `last_fed_secs` which the snapshot
    /// flattens to a 0..1 bar. The render methods need both.
    bars: BarsView,
}

#[derive(Debug, Clone)]
struct BarsView {
    /// Bucketed energy ("low" / "medium" / "high").
    energy_word: &'static str,
    /// Bucketed social ("lonely" / "ok" / "engaged").
    social_word: &'static str,
    /// Sass parenthetical like `(snarky)`, or None if below threshold.
    sass_word: Option<&'static str>,
    /// Hunger token ("just ate" / "peckish") at the extremes only.
    hunger_word: Option<&'static str>,
}

impl StateSnapshot {
    /// Lock the mood mutex once, read the pet, freeze the clock, and build a
    /// snapshot. Caller has already locked `pet` (we take `&Pet`, not
    /// `&SharedPet`, by design — the typical caller already holds the pet
    /// guard for unrelated reasons and we don't want to re-enter the mutex).
    pub fn capture(mood: &SharedMood, pet: &Pet, clock: &dyn Clock) -> Self {
        let now = clock.now();
        let (mood_snapshot, bars) = {
            let guard: std::sync::MutexGuard<'_, mood::MoodState> =
                mood.lock().unwrap_or_else(|p| p.into_inner());
            let snap = guard.snapshot();
            let bars = BarsView {
                energy_word: bucket_energy(guard.energy()),
                social_word: bucket_social(guard.social),
                sass_word: bucket_sass(guard.sass_level),
                hunger_word: bucket_hunger(guard.last_fed_secs),
            };
            (snap, bars)
        };

        let last_interaction_ms = pet.last_interaction_elapsed_ms().max(0) as u64;
        let file_drag = if pet.file_drag_active {
            Some(FileDragView {
                kind: "file".to_string(),
                since_ms: 0,
            })
        } else {
            None
        };

        Self {
            mood: mood_snapshot,
            pet: PetStateView {
                state: pet.state(),
                last_interaction_ms,
            },
            now,
            file_drag,
            bars,
        }
    }

    /// Build a snapshot from raw parts. Used by tests and by the legacy
    /// shims during the migration window — they have a `MoodState` reference
    /// and don't want to round-trip through `SharedMood`.
    pub fn from_parts(
        mood: &mood::MoodState,
        pet_state: PetState,
        last_interaction_ms: i64,
        file_drag_active: bool,
        now: chrono::DateTime<chrono::Local>,
    ) -> Self {
        let mood_snapshot = mood.snapshot();
        let bars = BarsView {
            energy_word: bucket_energy(mood.energy()),
            social_word: bucket_social(mood.social),
            sass_word: bucket_sass(mood.sass_level),
            hunger_word: bucket_hunger(mood.last_fed_secs),
        };

        Self {
            mood: mood_snapshot,
            pet: PetStateView {
                state: pet_state,
                last_interaction_ms: last_interaction_ms.max(0) as u64,
            },
            now,
            file_drag: if file_drag_active {
                Some(FileDragView {
                    kind: "file".to_string(),
                    since_ms: 0,
                })
            } else {
                None
            },
            bars,
        }
    }

    /// Compact rendering for tool returns (PRD 3 tool-layer use).
    /// Single line, semicolon-separated, no surrounding brackets — designed
    /// to embed cleanly in JSON tool-result strings without escaping.
    pub fn render_compact(&self) -> String {
        let time = self.now.format("%-I:%M %p").to_string();
        let mut out = format!(
            "Mood: {}; Energy: {}; Social: {}; State: {:?}; Time: {}",
            self.mood.mood_word, self.bars.energy_word, self.bars.social_word, self.pet.state, time,
        );
        if let Some(sass) = self.bars.sass_word {
            out.push_str("; Sass: ");
            out.push_str(sass);
        }
        if let Some(h) = self.bars.hunger_word {
            out.push_str("; Hunger: ");
            out.push_str(h);
        }
        if self.file_drag.is_some() {
            out.push_str("; FileDrag: yes");
        }
        out
    }

    /// Pass-1 (router) system block. PRD 3 will use this to give the
    /// dispatcher a terse but complete state view. Bracketed, pipe-separated,
    /// fixed slot order so prompt templates can grep on it.
    pub fn render_for_router(&self) -> String {
        let time = self.now.format("%-I:%M %p").to_string();
        let mood_token = match self.bars.sass_word {
            Some(s) => format!("{} ({})", self.mood.mood_word, s),
            None => self.mood.mood_word.clone(),
        };
        let mut out = format!(
            "[Rolo: Mood: {} | Energy: {} | Social: {} | State: {:?} | Time: {}",
            mood_token, self.bars.energy_word, self.bars.social_word, self.pet.state, time,
        );
        if let Some(h) = self.bars.hunger_word {
            out.push_str(" | Hunger: ");
            out.push_str(h);
        }
        if self.file_drag.is_some() {
            out.push_str(" | FileDrag: yes");
        }
        out.push(']');
        out
    }

    /// Legacy speech-path rendering. Output is byte-identical to
    /// `MoodState::render_state_line(now)` for equivalent inputs.
    /// `speech::build_context_prompt` calls this then appends its own
    /// `[State: ...]` suffix; chat/engine.rs uses this as the slot-4 state
    /// line for the vault assembler.
    pub fn render_for_speech(&self) -> String {
        let mood_token = match self.bars.sass_word {
            Some(s) => format!("Mood: {} ({})", self.mood.mood_word, s),
            None => format!("Mood: {}", self.mood.mood_word),
        };
        let time = self.now.format("%-I:%M %p").to_string();

        let mut line = format!(
            "[{} | Energy: {} | Social: {} | Time: {}",
            mood_token, self.bars.energy_word, self.bars.social_word, time,
        );
        if let Some(h) = self.bars.hunger_word {
            line.push_str(" | Hunger: ");
            line.push_str(h);
        }
        line.push(']');
        line
    }
}

// ---------------------------------------------------------------------------
// Free function — slot-4 placeholder for the tick.rs callsites.
// ---------------------------------------------------------------------------

/// Mood-blind state-slot placeholder. Reproduces the legacy
/// `tick::build_state_slot_placeholder` output byte-for-byte. Used as the
/// `state_slot` argument to `vault::prompt::PromptAssembler::assemble` from
/// the tick loop, where the *real* mood line is already in the prompt's
/// `context` argument and a second copy in slot 4 would just inflate tokens.
///
/// Time format: `%-l:%M %p` (lowercase L) matches the legacy implementation
/// in tick.rs — note the difference vs the speech renderer's `%-I:%M %p`.
pub fn render_state_slot_placeholder(
    state: PetState,
    now: chrono::DateTime<chrono::Local>,
) -> String {
    let time_str = now.format("%-l:%M %p").to_string();
    format!(
        "[Mood: unknown | Energy: unknown | State: {:?} | Time: {}]",
        state, time_str
    )
}

// ---------------------------------------------------------------------------
// Tests — ≥6 snapshot fixtures across {morning, afternoon, late-night} ×
// {content, anxious}. PRD T1 verification block.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mood::MoodState;
    use chrono::TimeZone;

    fn local(year: i32, mon: u32, day: u32, h: u32, m: u32) -> chrono::DateTime<chrono::Local> {
        chrono::Local
            .with_ymd_and_hms(year, mon, day, h, m, 0)
            .single()
            .expect("fixture timestamp is unambiguous")
    }

    /// "Content" mood — mid-baseline bars at the given clock time. Energy
    /// is derived from `now` so the bucketed energy word reflects the
    /// time-of-day cosine curve from `MoodState::compute_energy_from_clock`.
    /// `last_fed_secs = 3600` puts hunger in the "no token" middle band
    /// (between the 30-min "just ate" cutoff and the 4-hour "peckish" cutoff).
    fn content_mood(now: chrono::DateTime<chrono::Local>) -> MoodState {
        MoodState {
            energy_cached: MoodState::compute_energy_from_clock(now),
            social: 0.6,
            happiness: 0.5,
            sass_level: 0.0,
            last_fed_secs: 3600,
            ..MoodState::default()
        }
    }

    /// "Anxious" mood — low social, low happiness, mid sass.
    /// The bucket vocabulary doesn't include "anxious" — it's modeled by
    /// `low` derived mood + `lonely` social bucket. This is the
    /// regression-fixture for the chat mood-blind bug: chat used to render
    /// "content / normal" no matter what the live mood looked like.
    fn anxious_mood() -> MoodState {
        MoodState {
            energy_cached: 0.2,    // low energy
            social: 0.1,           // lonely
            happiness: 0.05,       // unhappy
            sass_level: 0.7,       // snarky
            last_fed_secs: 16_000, // peckish
            ..MoodState::default()
        }
    }

    #[test]
    fn morning_content() {
        let now = local(2026, 5, 6, 9, 0);
        let snap = StateSnapshot::from_parts(&content_mood(now), PetState::Idle, 0, false, now);
        // 9 AM is near the cosine peak (10:00) — energy buckets "high".
        assert_eq!(
            snap.render_for_speech(),
            "[Mood: content | Energy: high | Social: ok | Time: 9:00 AM]",
        );
    }

    #[test]
    fn afternoon_content() {
        let now = local(2026, 5, 6, 14, 0);
        let snap = StateSnapshot::from_parts(&content_mood(now), PetState::Idle, 0, false, now);
        // 2 PM cosine gives ~0.75; hunger=0.75 (last_fed=3600) scales
        // energy() to ~0.656 → "medium" bucket. The trimmed energy
        // also pulls derived_mood just under +0.2 → "neutral".
        assert_eq!(
            snap.render_for_speech(),
            "[Mood: neutral | Energy: medium | Social: ok | Time: 2:00 PM]",
        );
    }

    #[test]
    fn late_night_content() {
        let now = local(2026, 5, 6, 1, 30);
        let snap = StateSnapshot::from_parts(&content_mood(now), PetState::Idle, 0, false, now);
        // 1:30 AM: energy is "low" from the cosine, and the energy drop
        // pulls `derived_mood` below the +0.2 threshold so the mood word
        // buckets to "neutral" rather than "content". This is correct
        // behavior — late-night Rolo is more subdued.
        assert_eq!(
            snap.render_for_speech(),
            "[Mood: neutral | Energy: low | Social: ok | Time: 1:30 AM]",
        );
    }

    #[test]
    fn morning_anxious() {
        let now = local(2026, 5, 6, 9, 0);
        let snap = StateSnapshot::from_parts(&anxious_mood(), PetState::Idle, 0, false, now);
        // sad mood + snarky sass + lonely social + peckish hunger.
        assert_eq!(
            snap.render_for_speech(),
            "[Mood: sad (snarky) | Energy: low | Social: lonely | Time: 9:00 AM | Hunger: peckish]",
        );
    }

    #[test]
    fn afternoon_anxious() {
        let now = local(2026, 5, 6, 14, 0);
        let snap = StateSnapshot::from_parts(&anxious_mood(), PetState::Idle, 0, false, now);
        assert_eq!(
            snap.render_for_speech(),
            "[Mood: sad (snarky) | Energy: low | Social: lonely | Time: 2:00 PM | Hunger: peckish]",
        );
    }

    #[test]
    fn late_night_anxious() {
        let now = local(2026, 5, 6, 1, 30);
        let snap = StateSnapshot::from_parts(&anxious_mood(), PetState::Idle, 0, false, now);
        assert_eq!(
            snap.render_for_speech(),
            "[Mood: sad (snarky) | Energy: low | Social: lonely | Time: 1:30 AM | Hunger: peckish]",
        );
    }

    // ----- Cross-renderer parity: snapshot must match `mood::render_state_line` -----

    #[test]
    fn render_for_speech_matches_legacy_render_state_line_content() {
        let now = local(2026, 5, 6, 14, 30);
        let mood = content_mood(now);
        let snap = StateSnapshot::from_parts(&mood, PetState::Idle, 0, false, now);
        assert_eq!(snap.render_for_speech(), mood.render_state_line(now));
    }

    #[test]
    fn render_for_speech_matches_legacy_render_state_line_anxious() {
        let now = local(2026, 5, 6, 1, 30);
        let mood = anxious_mood();
        let snap = StateSnapshot::from_parts(&mood, PetState::Sleeping, 0, false, now);
        assert_eq!(snap.render_for_speech(), mood.render_state_line(now));
    }

    // ----- Slot-4 placeholder parity with the legacy tick.rs free fn -----

    #[test]
    fn placeholder_matches_legacy_tick_format() {
        let now = local(2026, 5, 6, 14, 5);
        let out = render_state_slot_placeholder(PetState::Idle, now);
        // %-l strips leading zero/space, so 2:05 PM (not " 2:05 PM").
        assert_eq!(
            out,
            "[Mood: unknown | Energy: unknown | State: Idle | Time: 2:05 PM]",
        );
    }

    // ----- Compact and router renderings (new — for PRD 3) -----

    #[test]
    fn render_compact_includes_all_axes() {
        let now = local(2026, 5, 6, 14, 0);
        let snap =
            StateSnapshot::from_parts(&anxious_mood(), PetState::DragHover, 5_000, true, now);
        let out = snap.render_compact();
        assert!(out.starts_with("Mood: sad; "), "got: {}", out);
        assert!(out.contains("State: DragHover"), "got: {}", out);
        assert!(out.contains("Sass: snarky"), "got: {}", out);
        assert!(out.contains("Hunger: peckish"), "got: {}", out);
        assert!(out.contains("FileDrag: yes"), "got: {}", out);
    }

    #[test]
    fn render_for_router_is_bracketed() {
        let now = local(2026, 5, 6, 14, 0);
        let snap = StateSnapshot::from_parts(&content_mood(now), PetState::Idle, 0, false, now);
        let out = snap.render_for_router();
        // At 2 PM with mild hunger, content_mood now buckets to "neutral"
        // (hunger→energy coupling trims derived_mood just under +0.2).
        assert!(out.starts_with("[Rolo: Mood: neutral"), "got: {}", out);
        assert!(out.ends_with(']'), "got: {}", out);
    }

    /// Regression for the chat mood-blind bug — given an anxious live mood,
    /// `render_for_speech` MUST surface "lonely" and "snarky", not the legacy
    /// "content/normal" stub. PRD T1 verification.
    #[test]
    fn anxious_live_mood_surfaces_in_speech_render() {
        let now = local(2026, 5, 6, 14, 0);
        let snap = StateSnapshot::from_parts(&anxious_mood(), PetState::Idle, 0, false, now);
        let out = snap.render_for_speech();
        assert!(
            out.contains("Social: lonely"),
            "anxious mood must show lonely social, got: {}",
            out
        );
        assert!(
            out.contains("(snarky)"),
            "high sass must show snarky modifier, got: {}",
            out
        );
        assert!(
            !out.contains("content"),
            "anxious mood must NOT render as content, got: {}",
            out
        );
    }

    /// Snapshot fixture file lives at `src-tauri/tests/fixtures/
    /// state_renderings.snap`. We don't pull in `insta`; this test writes the
    /// canonical fixture string and asserts it matches what we render today.
    /// Per PRD T1 stopping condition, the fixture's body is what the user reviews
    /// before T2 begins.
    #[test]
    fn fixture_file_matches_current_renderings() {
        // Six combos × render_for_speech. content_mood() recomputes
        // energy from the supplied `now` so the bucketed energy word
        // varies across morning / afternoon / late-night.
        let morning = local(2026, 5, 6, 9, 0);
        let afternoon = local(2026, 5, 6, 14, 0);
        let late_night = local(2026, 5, 6, 1, 30);
        let cases = [
            ("morning_content", content_mood(morning), morning),
            ("afternoon_content", content_mood(afternoon), afternoon),
            ("late_night_content", content_mood(late_night), late_night),
            ("morning_anxious", anxious_mood(), morning),
            ("afternoon_anxious", anxious_mood(), afternoon),
            ("late_night_anxious", anxious_mood(), late_night),
        ];

        let mut rendered = String::new();
        for (name, mood, now) in &cases {
            let snap = StateSnapshot::from_parts(mood, PetState::Idle, 0, false, *now);
            rendered.push_str(&format!("# {}\n", name));
            rendered.push_str("speech:  ");
            rendered.push_str(&snap.render_for_speech());
            rendered.push('\n');
            rendered.push_str("router:  ");
            rendered.push_str(&snap.render_for_router());
            rendered.push('\n');
            rendered.push_str("compact: ");
            rendered.push_str(&snap.render_compact());
            rendered.push_str("\n\n");
        }

        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("state_renderings.snap");

        // Write-on-first-run pattern: if the fixture is missing, write it
        // and fail with a clear message so the user knows to review.
        if !fixture_path.exists() {
            std::fs::write(&fixture_path, &rendered)
                .expect("failed to write state_renderings.snap fixture");
            panic!(
                "Wrote new fixture to {} — review the contents and re-run.",
                fixture_path.display()
            );
        }

        let on_disk = std::fs::read_to_string(&fixture_path)
            .expect("failed to read state_renderings.snap fixture");
        assert_eq!(
            on_disk, rendered,
            "state_renderings.snap drifted from rendered output. If intentional, delete the file and re-run to regenerate."
        );
    }
}
