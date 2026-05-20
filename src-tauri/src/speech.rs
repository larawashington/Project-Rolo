//! Rolo's voice — the speech bubble state machine.
//!
//! This module is pure logic with NO Tauri dependency, just like
//! `state_machine.rs`. It manages phrase loading from JSON, a random
//! countdown timer, phrase selection, and display duration calculation.
//!
//! The tick loop in `tick.rs` calls `SpeechState::tick()` each frame and
//! reacts to the returned `SpeechAction` to show/hide the bubble window.
//!
//! Errors in phrase loading are non-fatal — Rolo simply stays silent.
//! A quiet Rolo is a healthy Rolo; a crashing Rolo is a sick one.

use rand::Rng;
use serde::Deserialize;

use crate::mood;
use crate::state_machine::PetState;

/// True only when `ROLO_DEBUG` is set to a truthy value (`1` or `true`).
/// `is_ok()` returns true for any value including `"0"`, which silently
/// turned vanilla `/ro-go` (which exports `ROLO_DEBUG=0`) into turbo mode.
pub fn debug_mode_enabled() -> bool {
    matches!(std::env::var("ROLO_DEBUG").as_deref(), Ok("1") | Ok("true"))
}

// ---------------------------------------------------------------------------
// Constants (speech-specific timing only; spatial constants live in geometry)
// ---------------------------------------------------------------------------

/// Minimum random interval before the next speech bubble (ms).
const TIMER_MIN_MS: i64 = 120_000;
/// Maximum random interval before the next speech bubble (ms).
const TIMER_MAX_MS: i64 = 300_000;

/// When a perception event lands, shorten the speech timer to fire within
/// this window so Rolo's reaction feels timely rather than minutes-stale.
/// The 5-minute PerceptionBuffer COOLDOWN already caps event-triggered
/// speech to ~once per 5 min, so anti-spam is handled there.
const EVENT_BUMP_MIN_MS: i64 = 10_000;
const EVENT_BUMP_MAX_MS: i64 = 30_000;

/// Below this social value, Rolo is "lonely" and speaks more often.
/// PRD §3 "Behavioral coupling".
const SOCIAL_LONELY_THRESHOLD: f64 = 0.3;
/// Multiplier applied to the random interval when social is below threshold.
const SOCIAL_LONELY_MULT: f64 = 0.6;

/// Base display duration before per-character scaling (ms).
const DISPLAY_BASE_MS: i64 = 2_000;
/// Additional display time per character (ms).
const DISPLAY_PER_CHAR_MS: i64 = 50;

// ---------------------------------------------------------------------------
// JSON schema — phrases.json
// ---------------------------------------------------------------------------

/// Top-level structure of `phrases.json`.
#[derive(Debug, Deserialize)]
pub struct PhrasesFile {
    #[allow(dead_code)]
    pub version: u32,
    pub phrases: std::collections::HashMap<String, Vec<PhraseEntry>>,
}

/// A single entry in a mood array — either a bare string or a metadata object.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PhraseEntry {
    /// Bare string shorthand: `"Hello, I'm Rolo."`
    Bare(String),
    /// Object with metadata: `{ "text": "Hello!", "weight": 2 }`
    Object(PhraseObject),
}

/// Full phrase object with optional metadata fields.
#[derive(Debug, Deserialize)]
pub struct PhraseObject {
    pub text: Option<String>,
    #[serde(default = "default_weight")]
    pub weight: f64,
    pub duration: Option<i64>,
    pub states: Option<Vec<String>>,
}

fn default_weight() -> f64 {
    1.0
}

// ---------------------------------------------------------------------------
// SpeechAction — what the tick loop should do this frame
// ---------------------------------------------------------------------------

/// Returned by `SpeechState::tick()` to tell the caller what to do.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeechAction {
    /// Nothing to do this frame.
    None,
    /// Show the bubble with this text (fallback phrases).
    Show(String),
    /// Hide the bubble — display duration expired.
    Hide,
    /// Ask the LLM thread to generate speech with this context prompt.
    GenerateLLM(String),
}

// ---------------------------------------------------------------------------
// BubbleLayout — where to place the bubble window
// ---------------------------------------------------------------------------

/// Calculated position and orientation for the speech bubble window.
#[derive(Debug, Clone, Copy)]
pub struct BubbleLayout {
    pub x: i32,
    pub y: i32,
    pub flipped: bool,
}

#[allow(clippy::too_many_arguments)]
impl BubbleLayout {
    /// Compute the bubble window position relative to Rolo's sprite.
    ///
    /// All values must be in the same unit (physical pixels).
    /// `top_anchored` is true when Rolo is at the top of the screen
    /// (sprite rendered at window top) — the bubble should flip below
    /// him as the primary layout, not just as a fallback.
    pub fn compute(
        sprite_x: i32,
        sprite_y: i32,
        sprite_size: i32,
        bubble_w: i32,
        bubble_h: i32,
        screen_w: i32,
        screen_h: i32,
        gap_above: i32,
        gap_below: i32,
        top_anchored: bool,
    ) -> Self {
        let mut x = sprite_x + (sprite_size / 2) - (bubble_w / 2);

        let mut flipped = top_anchored;
        let mut y = if flipped {
            sprite_y + sprite_size + gap_below
        } else {
            sprite_y - bubble_h - gap_above
        };

        // Fallback: not top_anchored but would clip above screen
        if !flipped && y < 0 {
            y = sprite_y + sprite_size + gap_below;
            flipped = true;
        }

        // Safety clamps
        if y + bubble_h > screen_h {
            y = screen_h - bubble_h;
        }
        if y < 0 {
            y = 0;
        }
        x = x.max(0).min(screen_w - bubble_w);

        Self { x, y, flipped }
    }
}

// ---------------------------------------------------------------------------
// SpeechState — Rolo's voice box
// ---------------------------------------------------------------------------

pub struct SpeechState {
    /// Loaded phrases from the "default" mood group.
    phrases: Vec<PhraseEntry>,
    /// Countdown until next speech trigger (ms). Counts down each tick.
    timer_ms: i64,
    /// Countdown for how long the current bubble stays visible (ms).
    display_timer_ms: i64,
    /// Whether a bubble is currently being shown.
    showing: bool,
    /// Whether the speech system is enabled (false if phrases failed to load).
    enabled: bool,
    /// A phrase picked this cycle that's waiting for Rolo to move into the
    /// bubble-safe zone before it can appear. None when nothing is pending.
    deferred: Option<(String, i64)>,
    /// Safety cap on how long a deferred phrase will wait. If Rolo hasn't
    /// reached the safe zone in this long, the phrase is dropped and the
    /// cycle restarts. Counts down while `deferred` is Some.
    deferred_timeout_ms: i64,

    /// Running count of dismissed bubbles this session. Incremented by
    /// `dismiss_speech` before logging the Dismiss vault event; reset on
    /// SessionEnd (app close or date rollover).
    pub dismiss_count_session: u32,

    // --- LLM integration ---
    /// Whether Ollama is reachable. When false, falls back to phrases.json.
    pub ollama_available: bool,
    /// Countdown until next Ollama health check retry (ms). Only ticks when
    /// ollama_available is false.
    ollama_retry_timer_ms: i64,
    /// Milliseconds since the last LLM call was dispatched. Used for the
    /// 5-second rate limit on interaction-triggered speech.
    last_llm_call_ms: i64,
    /// True while the LLM thread is generating (prevents duplicate requests).
    pub is_generating: bool,
    /// Whether the one-time "brain fuzzy" message has been shown.
    brain_fuzzy_shown: bool,
}

impl SpeechState {
    /// Create a new SpeechState. Starts the random countdown immediately.
    /// Pass the raw JSON bytes of `phrases.json`.
    pub fn new(phrases_json: Option<&[u8]>) -> Self {
        let (phrases, enabled) = match phrases_json {
            Some(data) => match Self::parse_phrases(data) {
                Ok(phrases) => {
                    let enabled = !phrases.is_empty();
                    if !enabled {
                        eprintln!(
                            "[Rolo] phrases.json loaded but \"default\" array is empty — \
                             Rolo will stay silent for now."
                        );
                    }
                    (phrases, enabled)
                }
                Err(e) => {
                    eprintln!(
                        "[Rolo] Failed to parse phrases.json — Rolo will stay silent: {}",
                        e
                    );
                    (Vec::new(), false)
                }
            },
            None => {
                eprintln!("[Rolo] No phrases.json found — Rolo will stay silent but healthy.");
                (Vec::new(), false)
            }
        };

        let timer_ms = if debug_mode_enabled() {
            2_000
        } else {
            Self::random_interval()
        };

        SpeechState {
            phrases,
            timer_ms,
            display_timer_ms: 0,
            showing: false,
            enabled,
            deferred: None,
            deferred_timeout_ms: 0,
            dismiss_count_session: 0,
            ollama_available: false,
            ollama_retry_timer_ms: 0,
            last_llm_call_ms: i64::MAX,
            is_generating: false,
            brain_fuzzy_shown: false,
        }
    }

    /// Parse the phrases JSON and extract the "default" mood group.
    fn parse_phrases(data: &[u8]) -> Result<Vec<PhraseEntry>, String> {
        let file: PhrasesFile =
            serde_json::from_slice(data).map_err(|e| format!("JSON parse error: {}", e))?;

        Ok(file
            .phrases
            .into_iter()
            .find(|(key, _)| key == "default")
            .map(|(_, entries)| entries)
            .unwrap_or_default())
    }

    /// Whether a bubble is currently visible.
    pub fn is_showing(&self) -> bool {
        self.showing
    }

    /// Tick the speech system forward by `dt_ms` milliseconds.
    /// Returns what the caller should do (show, hide, or nothing).
    ///
    /// `social` is a snapshot of `MoodState::social` at call time. When
    /// `social < 0.3` (Rolo is lonely), the next idle interval is multiplied
    /// by 0.6 so he gets chattier — PRD §3 "Behavioral coupling". The caller
    /// must snapshot the scalar before calling; speech.rs intentionally does
    /// not take an `Arc<Mutex<MoodState>>` to keep the tick lock-free.
    pub fn tick(&mut self, dt_ms: i64, _current_state: PetState, social: f64) -> SpeechAction {
        if !self.enabled && !self.ollama_available {
            return SpeechAction::None;
        }

        // Tick rate-limit cooldown
        if self.last_llm_call_ms < i64::MAX {
            self.last_llm_call_ms = self.last_llm_call_ms.saturating_add(dt_ms);
        }

        // Tick Ollama retry timer when it's down
        if !self.ollama_available && self.ollama_retry_timer_ms > 0 {
            self.ollama_retry_timer_ms -= dt_ms;
        }

        // A deferred phrase is waiting for Rolo to reach the safe zone —
        // keep the timeout ticking and stay silent until it commits or drops.
        if self.deferred.is_some() {
            self.deferred_timeout_ms -= dt_ms;
            if self.deferred_timeout_ms <= 0 {
                self.deferred = None;
                self.timer_ms = Self::next_interval(social);
            }
            return SpeechAction::None;
        }

        // If currently showing, count down the display timer
        if self.showing {
            self.display_timer_ms -= dt_ms;
            if self.display_timer_ms <= 0 {
                self.showing = false;
                self.timer_ms = Self::next_interval(social);
                return SpeechAction::Hide;
            }
            return SpeechAction::None;
        }

        // Don't fire new speech while LLM is generating
        if self.is_generating {
            return SpeechAction::None;
        }

        // Suppress during eating/sniffing states
        if matches!(
            _current_state,
            PetState::Sniffing | PetState::Eating | PetState::Satisfied | PetState::Disappointed
        ) {
            return SpeechAction::None;
        }

        // Count down the trigger timer
        self.timer_ms -= dt_ms;
        if self.timer_ms <= 0 {
            // Time to speak!
            if self.ollama_available {
                self.timer_ms = Self::next_interval(social);
                return SpeechAction::GenerateLLM(String::new());
            }
            if let Some((text, duration)) = self.select_phrase(_current_state) {
                self.showing = true;
                self.display_timer_ms = duration;
                return SpeechAction::Show(text);
            } else {
                // No eligible phrase — reschedule silently
                self.timer_ms = Self::next_interval(social);
            }
        }

        SpeechAction::None
    }

    /// Safety cap for how long a deferred phrase waits for Rolo to reach the
    /// bubble-safe zone. Long enough for a cross-screen nudge walk; short
    /// enough that a stuck state machine doesn't leave him silent forever.
    const DEFERRED_TIMEOUT_MS: i64 = 8_000;

    /// Roll back a `Show` returned this tick and hold onto the phrase until
    /// the caller is ready to display it (e.g., after Rolo walks into the
    /// bubble-safe zone). Call immediately after `tick()` returns
    /// `SpeechAction::Show(text)` when display is blocked.
    pub fn defer_active_show(&mut self, text: String) {
        if !self.showing {
            return;
        }
        let duration = self.display_timer_ms;
        self.showing = false;
        self.display_timer_ms = 0;
        self.deferred = Some((text, duration));
        self.deferred_timeout_ms = Self::DEFERRED_TIMEOUT_MS;
    }

    /// True when a phrase is waiting to be displayed.
    pub fn has_deferred(&self) -> bool {
        self.deferred.is_some()
    }

    /// Consume the deferred phrase and transition into the showing state.
    /// The caller should now emit the show event and reposition the bubble
    /// as if `tick()` had just returned `Show(text)`.
    pub fn take_deferred(&mut self) -> Option<(String, i64)> {
        let (text, duration) = self.deferred.take()?;
        self.showing = true;
        self.display_timer_ms = duration;
        Some((text, duration))
    }

    /// Build a context prompt for the LLM from Rolo's current state.
    ///
    /// v2 (rolo-brain) — produces the state-header portion of the first
    /// user turn matching the SFT training distribution. Caller appends
    /// `"\n\n" + IDLE_SENTINEL` (proactive speech) or `"\n\n" + user_msg`
    /// (chat) to complete the body. Perception observations land between
    /// the state line and the blank-line separator (see
    /// `perception::append_perception_to_context`). Final format
    /// (from `data/finetune/sft-v2/runtime_contract.md` §2):
    ///
    /// ```text
    /// [Mood: <m> | Energy: <e> | Social: <s> | Time: <t>]
    /// [State: Rolo <stem>]
    /// [Observation: <perception>]   <-- optional
    ///
    /// <idle>
    /// ```
    ///
    /// The `last_interaction_ms`, `event`, and `file_drag` arguments are
    /// retained for caller compatibility but no longer surface in the prompt —
    /// the v2 corpus has no `Last interaction:` annotation (12 of 3,379 rows)
    /// and the colloquial `Event:` strings the runtime synthesizes are
    /// off-distribution vs the training corpus's ambient observation style.
    ///
    /// State-stem map is restricted to the 5 speech-eligible stems per
    /// ship-v2.md §3.5a. Suppressed states (Sleeping, DragHover, Eating,
    /// Happy) still map to a valid stem so non-LLM callers don't panic, but
    /// `should_suppress_speech` blocks them from ever reaching inference.
    pub fn build_context_prompt(
        mood: &mood::MoodState,
        now: chrono::DateTime<chrono::Local>,
        state: PetState,
        _last_interaction_ms: i64,
        _event: Option<&str>,
        _file_drag: bool,
    ) -> String {
        let state_desc = match state {
            PetState::Idle => "Rolo is sitting idle",
            PetState::Sleeping => "Rolo is sleeping",
            PetState::WalkLeft | PetState::WalkRight => "Rolo is walking around",
            PetState::Happy => "Rolo is happy",
            PetState::DragHover => "Rolo is sitting idle",
            PetState::Sniffing => "Rolo is sniffing a file",
            PetState::Eating => "Rolo is eating",
            PetState::Satisfied => "Rolo just finished eating and is satisfied",
            PetState::Disappointed => "Rolo was offered food but it was declined",
        };

        let mood_line = mood.render_state_line(now);
        format!("{}\n[State: {}]", mood_line, state_desc)
    }

    /// Whether enough time has passed since the last LLM call (5-second cooldown).
    pub fn can_call_llm(&self) -> bool {
        self.last_llm_call_ms >= 5_000
    }

    /// Record that an LLM call was dispatched.
    pub fn mark_llm_call(&mut self) {
        self.last_llm_call_ms = 0;
        self.is_generating = true;
    }

    /// Mark generation as complete.
    pub fn generation_done(&mut self) {
        self.is_generating = false;
    }

    /// Mark Ollama as unavailable and start the retry timer.
    pub fn mark_ollama_down(&mut self) {
        self.ollama_available = false;
        self.ollama_retry_timer_ms = 300_000; // 5 minutes
        self.is_generating = false;
    }

    /// Whether a health check retry is due.
    pub fn should_retry_health_check(&self) -> bool {
        !self.ollama_available && self.ollama_retry_timer_ms <= 0
    }

    /// Reset the retry timer after dispatching a health check.
    pub fn reset_retry_timer(&mut self) {
        self.ollama_retry_timer_ms = 300_000;
    }

    /// Get the one-time "brain fuzzy" message, or None if already shown.
    pub fn take_brain_fuzzy_message(&mut self) -> Option<String> {
        if self.brain_fuzzy_shown {
            return None;
        }
        self.brain_fuzzy_shown = true;
        Some("My brain feels fuzzy today...".to_string())
    }

    /// Pick a fallback phrase from phrases.json for this cycle.
    pub fn pick_fallback(&self, current_state: PetState) -> Option<(String, i64)> {
        self.select_phrase(current_state)
    }

    /// Show a phrase immediately (used by tick.rs for LLM-generated or fallback text).
    pub fn start_showing(&mut self, duration_ms: i64) {
        self.showing = true;
        self.display_timer_ms = duration_ms;
    }

    /// Reset the idle speech timer (e.g., after interaction-triggered speech).
    pub fn reset_timer(&mut self) {
        self.timer_ms = Self::random_interval();
    }

    /// Called when an external suppression source (check-in, chat window) ends.
    /// Restarts the random countdown so a backlogged tick doesn't immediately
    /// fire a bubble, and clears any deferred phrase that was waiting on the
    /// bubble-safe zone. Mirrors the timer reset logic inside `tick()` when
    /// a phrase finishes — exposed publicly so command handlers can call it
    /// without going through the tick loop.
    pub fn reset_timer_after_suppression(&mut self) {
        self.timer_ms = Self::random_interval();
        self.deferred = None;
        self.deferred_timeout_ms = 0;
    }

    /// Force the next tick to fire a speech bubble immediately.
    pub fn force_trigger(&mut self) {
        if !self.enabled || self.showing {
            return;
        }
        self.deferred = None;
        self.timer_ms = 0;
    }

    /// Queue a hardcoded phrase to display immediately on the next tick,
    /// bypassing the rotation pool and the LLM thread entirely. Used by the
    /// Command Center (PRD/rolo-command-center.md Phase 5) to surface a
    /// reaction when the user changes the brain or saves Memory content — no
    /// need to roundtrip through an LLM to say "feels different in here..."
    ///
    /// The `deferred` slot is the only path that can fire without `enabled`
    /// being true, so this works even if `phrases.json` failed to load —
    /// Rolo can still react to the user's settings changes when the JSON pool
    /// is silent. The next `tick()` consumes the slot via `take_deferred()`
    /// in the caller (tick.rs), which sets `showing=true` and `display_timer_ms`.
    pub fn queue_immediate_phrase(&mut self, text: String) {
        // `enabled` gates the rotation pool, not the deferred slot — flipping
        // it true here lets `tick()` get past its early-return guard so the
        // deferred-consumption path in tick.rs can fire on the next frame.
        self.enabled = true;
        let duration = calculate_display_duration(&text);
        self.deferred = Some((text, duration));
        self.deferred_timeout_ms = Self::DEFERRED_TIMEOUT_MS;
        self.timer_ms = 0;
    }

    /// Shorten the idle timer to a 10-30s window so Rolo reacts to a fresh
    /// perception event while it still feels current. Never lengthens — if
    /// the timer is already shorter, this is a no-op.
    pub fn accelerate_for_perception(&mut self) {
        if !self.enabled && !self.ollama_available {
            return;
        }
        if self.showing || self.is_generating || self.deferred.is_some() {
            return;
        }
        let mut rng = rand::rng();
        let bumped = rng.random_range(EVENT_BUMP_MIN_MS..=EVENT_BUMP_MAX_MS);
        if bumped < self.timer_ms {
            self.timer_ms = bumped;
        }
    }

    /// Dismiss the current bubble (called when user clicks it).
    /// Returns true if a bubble was actually showing or generating and got dismissed.
    pub fn dismiss(&mut self) -> bool {
        let was_active = self.showing || self.is_generating;
        self.showing = false;
        self.display_timer_ms = 0;
        if was_active {
            self.is_generating = false;
            self.timer_ms = Self::random_interval();
        }
        was_active
    }

    /// Increment the per-session dismiss counter and return the new value.
    /// Called by the dismiss_speech command before logging the vault event.
    pub fn increment_dismiss_count(&mut self) -> u32 {
        self.dismiss_count_session += 1;
        self.dismiss_count_session
    }

    /// Reset the per-session dismiss counter (called on SessionEnd / date rollover).
    pub fn reset_dismiss_count(&mut self) {
        self.dismiss_count_session = 0;
    }

    /// Select a phrase from the "default" group, filtered by current PetState.
    /// Returns (text, display_duration_ms) or None if no phrase is eligible.
    fn select_phrase(&self, current_state: PetState) -> Option<(String, i64)> {
        if self.phrases.is_empty() {
            return None;
        }

        let state_str = state_to_string(current_state);

        // Collect eligible phrases with their weights
        let eligible: Vec<(&PhraseEntry, f64)> = self
            .phrases
            .iter()
            .filter(|entry| {
                match entry {
                    PhraseEntry::Bare(_) => true, // bare strings are always eligible
                    PhraseEntry::Object(obj) => {
                        // Filter by states if specified
                        if let Some(ref states) = obj.states {
                            states.iter().any(|s| s == &state_str)
                        } else {
                            true
                        }
                    }
                }
            })
            .map(|entry| {
                let weight = match entry {
                    PhraseEntry::Bare(_) => 1.0,
                    PhraseEntry::Object(obj) => obj.weight,
                };
                (entry, weight)
            })
            .collect();

        if eligible.is_empty() {
            return None;
        }

        // Weighted random selection
        let total_weight: f64 = eligible.iter().map(|(_, w)| w).sum();
        // Guard against degenerate case where all weights are zero — treat as uniform.
        if total_weight <= 0.0 {
            let mut rng = rand::rng();
            let idx = rng.random_range(0..eligible.len());
            let (entry, _) = &eligible[idx];
            let text = entry_text(entry);
            let duration = entry_duration(entry, &text);
            return Some((text, duration));
        }
        let mut rng = rand::rng();
        let mut roll = rng.random_range(0.0..total_weight);

        for (entry, weight) in &eligible {
            roll -= weight;
            if roll <= 0.0 {
                let text = entry_text(entry);
                let duration = entry_duration(entry, &text);
                return Some((text, duration));
            }
        }

        // Fallback (shouldn't happen, but safety first — Rolo's health matters)
        let (entry, _) = &eligible[0];
        let text = entry_text(entry);
        let duration = entry_duration(entry, &text);
        Some((text, duration))
    }

    /// Generate a random interval between TIMER_MIN_MS and TIMER_MAX_MS.
    fn random_interval() -> i64 {
        if debug_mode_enabled() {
            return 10_000;
        }
        let mut rng = rand::rng();
        rng.random_range(TIMER_MIN_MS..=TIMER_MAX_MS)
    }

    /// Pick the next idle interval, applying the lonely multiplier when
    /// `social < 0.3`. PRD §3 "Behavioral coupling": needy social shortens
    /// the speech idle timer by 0.6×.
    fn next_interval(social: f64) -> i64 {
        let base = Self::random_interval();
        if social < SOCIAL_LONELY_THRESHOLD {
            ((base as f64) * SOCIAL_LONELY_MULT) as i64
        } else {
            base
        }
    }
}

/// Calculate display duration for a phrase: base + (char_count * per_char).
pub fn calculate_display_duration(text: &str) -> i64 {
    DISPLAY_BASE_MS + (text.len() as i64 * DISPLAY_PER_CHAR_MS)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the display text from a PhraseEntry.
fn entry_text(entry: &PhraseEntry) -> String {
    match entry {
        PhraseEntry::Bare(s) => s.clone(),
        PhraseEntry::Object(obj) => obj.text.clone().unwrap_or_default(),
    }
}

/// Get the display duration for a phrase entry, respecting overrides.
fn entry_duration(entry: &PhraseEntry, text: &str) -> i64 {
    match entry {
        PhraseEntry::Bare(_) => calculate_display_duration(text),
        PhraseEntry::Object(obj) => obj
            .duration
            .unwrap_or_else(|| calculate_display_duration(text)),
    }
}

/// Convert a PetState to the string used in phrases.json `states` arrays.
fn state_to_string(state: PetState) -> String {
    match state {
        PetState::Idle => "idle".to_string(),
        PetState::Sleeping => "sleeping".to_string(),
        PetState::WalkLeft => "walk_left".to_string(),
        PetState::WalkRight => "walk_right".to_string(),
        PetState::Happy => "happy".to_string(),
        PetState::DragHover => "drag_hover".to_string(),
        PetState::Sniffing => "sniffing".to_string(),
        PetState::Eating => "eating".to_string(),
        PetState::Satisfied => "satisfied".to_string(),
        PetState::Disappointed => "disappointed".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests — Rolo's voice must be reliable
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_JSON: &[u8] = br#"{
        "version": 1,
        "phrases": {
            "default": [
                "Hello, I'm Rolo."
            ]
        }
    }"#;

    const MULTI_PHRASE_JSON: &[u8] = br#"{
        "version": 1,
        "phrases": {
            "default": [
                "Hello, I'm Rolo.",
                { "text": "Good morning!", "weight": 2 },
                { "text": "Only when idle", "states": ["idle"] },
                { "text": "Custom duration", "duration": 5000 }
            ]
        }
    }"#;

    const EMPTY_DEFAULT_JSON: &[u8] = br#"{
        "version": 1,
        "phrases": {
            "default": []
        }
    }"#;

    use chrono::TimeZone;

    fn fixed_local_now() -> chrono::DateTime<chrono::Local> {
        chrono::Local
            .with_ymd_and_hms(2026, 4, 27, 14, 30, 0)
            .single()
            .expect("fixed time is unambiguous")
    }

    #[test]
    fn build_context_prompt_v2_contract_idle() {
        // v2: state header is mood line + state line on separate rows. No
        // `Last interaction` annotation; caller appends `\n\n<idle>`. The
        // default `MoodState` renders Hunger: just ate; that's a normal
        // training-distribution slot, included verbatim.
        let mood = mood::MoodState::default();
        let prompt = SpeechState::build_context_prompt(
            &mood,
            fixed_local_now(),
            PetState::Idle,
            12 * 60_000,
            None,
            false,
        );
        assert_eq!(
            prompt,
            "[Mood: content | Energy: medium | Social: ok | Time: 2:30 PM | Hunger: just ate]\n\
             [State: Rolo is sitting idle]",
            "v2 contract: two bracketed lines, newline-separated, no trailing idle"
        );
        // Confirm structural invariants for the contract.
        assert_eq!(prompt.matches('\n').count(), 1);
        assert!(!prompt.contains("Last interaction"));
        assert!(!prompt.contains("Event:"));
    }

    #[test]
    fn build_context_prompt_v2_contract_drops_colloquial_event() {
        // v2: `event_desc` from tick.rs (e.g. "Rolo just finished eating and
        // loved it") is off-distribution vs training, so it's dropped from
        // the prompt. The state stem alone encodes the satisfaction.
        let mood = mood::MoodState::default();
        let prompt = SpeechState::build_context_prompt(
            &mood,
            fixed_local_now(),
            PetState::Satisfied,
            0,
            Some("Rolo just finished eating and loved it"),
            false,
        );
        assert_eq!(
            prompt,
            "[Mood: content | Energy: medium | Social: ok | Time: 2:30 PM | Hunger: just ate]\n\
             [State: Rolo just finished eating and is satisfied]",
        );
    }

    #[test]
    fn build_context_prompt_v2_full_idle_turn_matches_golden_fixture() {
        // Simulate the tick.rs proactive-speech assembly: build_context_prompt
        // + "\n\n" + IDLE_SENTINEL. This is the byte-exact body that ollama.rs
        // sends as the user-role content for PRIMARY_MODEL. The Modelfile
        // wraps it in <start_of_turn>user / <end_of_turn> / <start_of_turn>model
        // at the protocol layer.
        let mood = mood::MoodState::default();
        let mut body = SpeechState::build_context_prompt(
            &mood,
            fixed_local_now(),
            PetState::Idle,
            0,
            None,
            false,
        );
        body.push_str("\n\n");
        body.push_str(crate::ollama::IDLE_SENTINEL);
        assert_eq!(
            body,
            "[Mood: content | Energy: medium | Social: ok | Time: 2:30 PM | Hunger: just ate]\n\
             [State: Rolo is sitting idle]\n\
             \n\
             <idle>",
            "first-user-turn body must match runtime_contract.md §2"
        );
    }

    #[test]
    fn next_interval_shortens_when_lonely() {
        // Run many trials; lonely intervals must always be <= base*0.6.
        for _ in 0..100 {
            let lonely = SpeechState::next_interval(0.0);
            assert!(lonely >= (TIMER_MIN_MS as f64 * SOCIAL_LONELY_MULT) as i64);
            assert!(lonely <= (TIMER_MAX_MS as f64 * SOCIAL_LONELY_MULT) as i64);

            let normal = SpeechState::next_interval(0.6);
            assert!(normal >= TIMER_MIN_MS);
            assert!(normal <= TIMER_MAX_MS);
        }
    }

    #[test]
    fn new_with_valid_json_enables_speech() {
        let speech = SpeechState::new(Some(VALID_JSON));
        assert!(speech.enabled);
        assert_eq!(speech.phrases.len(), 1);
        assert!(!speech.showing);
    }

    #[test]
    fn new_with_none_disables_speech() {
        let speech = SpeechState::new(None);
        assert!(!speech.enabled);
        assert!(speech.phrases.is_empty());
    }

    #[test]
    fn new_with_invalid_json_disables_speech() {
        let speech = SpeechState::new(Some(b"{ not valid json }"));
        assert!(!speech.enabled);
    }

    #[test]
    fn new_with_empty_default_disables_speech() {
        let speech = SpeechState::new(Some(EMPTY_DEFAULT_JSON));
        assert!(!speech.enabled);
    }

    #[test]
    fn defer_active_show_rolls_back_and_holds_phrase() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        speech.timer_ms = 0;
        let action = speech.tick(1, PetState::Idle, 1.0);
        let text = match action {
            SpeechAction::Show(t) => t,
            other => panic!("expected Show, got {:?}", other),
        };
        assert!(speech.showing);
        speech.defer_active_show(text.clone());
        assert!(!speech.showing);
        assert!(speech.has_deferred());

        // While deferred, further ticks stay silent — no re-firing.
        let action = speech.tick(100, PetState::Idle, 1.0);
        assert_eq!(action, SpeechAction::None);

        let (committed, _dur) = speech.take_deferred().expect("deferred present");
        assert_eq!(committed, text);
        assert!(speech.showing);
        assert!(!speech.has_deferred());
    }

    #[test]
    fn deferred_phrase_drops_after_timeout() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        speech.timer_ms = 0;
        let action = speech.tick(1, PetState::Idle, 1.0);
        if let SpeechAction::Show(t) = action {
            speech.defer_active_show(t);
        } else {
            panic!("expected Show");
        }
        assert!(speech.has_deferred());
        speech.tick(SpeechState::DEFERRED_TIMEOUT_MS + 1, PetState::Idle, 1.0);
        assert!(
            !speech.has_deferred(),
            "timeout should drop deferred phrase"
        );
    }

    #[test]
    fn tick_does_nothing_when_disabled() {
        let mut speech = SpeechState::new(None);
        let action = speech.tick(100_000, PetState::Idle, 1.0);
        assert_eq!(action, SpeechAction::None);
    }

    #[test]
    fn tick_fires_after_timer_expires() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        // Force timer to almost done
        speech.timer_ms = 100;
        let action = speech.tick(200, PetState::Idle, 1.0);
        assert!(matches!(action, SpeechAction::Show(_)));
        assert!(speech.showing);
    }

    #[test]
    fn tick_hides_after_display_duration() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        // Force a show
        speech.timer_ms = 0;
        let action = speech.tick(1, PetState::Idle, 1.0);
        assert!(matches!(action, SpeechAction::Show(_)));

        // Now tick past the display duration
        let action = speech.tick(999_999, PetState::Idle, 1.0);
        assert_eq!(action, SpeechAction::Hide);
        assert!(!speech.showing);
    }

    #[test]
    fn skip_trigger_while_already_showing() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        // Force show
        speech.timer_ms = 0;
        speech.tick(1, PetState::Idle, 1.0);
        assert!(speech.showing);

        // Even if timer somehow fires again, tick should return None (not another Show)
        speech.timer_ms = 0; // shouldn't matter — showing takes priority
        let action = speech.tick(1, PetState::Idle, 1.0);
        // Should be None (still counting down display timer)
        assert_eq!(action, SpeechAction::None);
    }

    #[test]
    fn dismiss_resets_timer() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        speech.timer_ms = 0;
        speech.tick(1, PetState::Idle, 1.0);
        assert!(speech.showing);

        let dismissed = speech.dismiss();
        assert!(dismissed);
        assert!(!speech.showing);
        // Timer should be reset to a new random interval
        assert!(speech.timer_ms >= TIMER_MIN_MS);
        assert!(speech.timer_ms <= TIMER_MAX_MS);
    }

    #[test]
    fn dismiss_when_not_showing_returns_false() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        assert!(!speech.dismiss());
    }

    #[test]
    fn force_trigger_causes_immediate_show_on_next_tick() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        assert!(speech.timer_ms > 0);
        speech.force_trigger();
        assert_eq!(speech.timer_ms, 0);
        let action = speech.tick(1, PetState::Idle, 1.0);
        assert!(matches!(action, SpeechAction::Show(_)));
    }

    #[test]
    fn force_trigger_noop_when_already_showing() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        speech.timer_ms = 0;
        speech.tick(1, PetState::Idle, 1.0);
        assert!(speech.showing);
        let timer_before = speech.timer_ms;
        speech.force_trigger();
        assert_eq!(speech.timer_ms, timer_before);
    }

    #[test]
    fn force_trigger_noop_when_disabled() {
        let mut speech = SpeechState::new(None);
        speech.force_trigger();
        let action = speech.tick(1, PetState::Idle, 1.0);
        assert_eq!(action, SpeechAction::None);
    }

    #[test]
    fn display_duration_calculation() {
        // "Hello, I'm Rolo." = 16 chars
        let text = "Hello, I'm Rolo.";
        assert_eq!(text.len(), 16);
        let dur = calculate_display_duration(text);
        assert_eq!(dur, 2000 + 16 * 50); // 2800ms
    }

    #[test]
    fn display_duration_empty_string() {
        assert_eq!(calculate_display_duration(""), 2000);
    }

    #[test]
    fn bubble_position_above_rolo() {
        let pos = BubbleLayout::compute(500, 500, 128, 200, 80, 1920, 1080, 8, 16, false);
        // Should be above: y = 500 - 80 - 8 = 412
        assert_eq!(pos.y, 412);
        // Centered: x = 500 + 64 - 100 = 464
        assert_eq!(pos.x, 464);
        assert!(!pos.flipped);
    }

    #[test]
    fn bubble_position_flips_when_near_top() {
        let pos = BubbleLayout::compute(500, 10, 128, 200, 80, 1920, 1080, 8, 16, false);
        // y = 10 - 80 - 8 = -78 < 0, so flip below using gap_below
        // flipped y = 10 + 128 + 16 = 154
        assert_eq!(pos.y, 154);
        assert!(pos.flipped);
    }

    #[test]
    fn bubble_position_clamps_horizontally() {
        // Near right edge
        let pos = BubbleLayout::compute(1800, 500, 128, 200, 80, 1920, 1080, 8, 16, false);
        assert!(pos.x <= 1920 - 200);
        assert!(pos.x >= 0);

        // Near left edge
        let pos = BubbleLayout::compute(0, 500, 128, 200, 80, 1920, 1080, 8, 16, false);
        assert!(pos.x >= 0);
    }

    #[test]
    fn state_filtered_phrase_excluded_in_wrong_state() {
        let speech = SpeechState::new(Some(MULTI_PHRASE_JSON));
        // The phrase "Only when idle" has states: ["idle"]
        // In Happy state, it should be excluded from selection.
        // We can't directly test random selection easily, but we can verify
        // the filter logic via select_phrase indirectly — run it many times
        // in Happy state and ensure "Only when idle" never appears.
        for _ in 0..100 {
            if let Some((text, _)) = speech.select_phrase(PetState::Happy) {
                assert_ne!(
                    text, "Only when idle",
                    "State-filtered phrase appeared in wrong state"
                );
            }
        }
    }

    #[test]
    fn custom_duration_override() {
        let speech = SpeechState::new(Some(MULTI_PHRASE_JSON));
        // Run selection many times; if we ever get "Custom duration", verify its duration
        for _ in 0..200 {
            if let Some((text, duration)) = speech.select_phrase(PetState::Idle) {
                if text == "Custom duration" {
                    assert_eq!(duration, 5000, "Custom duration override not respected");
                    return; // Test passed
                }
            }
        }
        // It's possible (but very unlikely) that we never pick it in 200 tries.
        // With 4 eligible phrases and weight distribution, this is acceptable.
    }

    #[test]
    fn timer_range_is_valid() {
        // Verify random_interval stays in bounds
        for _ in 0..100 {
            let interval = SpeechState::random_interval();
            assert!(interval >= TIMER_MIN_MS);
            assert!(interval <= TIMER_MAX_MS);
        }
    }

    #[test]
    fn accelerate_for_perception_shortens_idle_timer() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        speech.timer_ms = TIMER_MAX_MS; // pretend we're at the long end
        speech.accelerate_for_perception();
        assert!(
            speech.timer_ms <= EVENT_BUMP_MAX_MS,
            "timer should be bumped into the event window, got {}",
            speech.timer_ms,
        );
        assert!(speech.timer_ms >= EVENT_BUMP_MIN_MS);
    }

    #[test]
    fn accelerate_for_perception_never_lengthens() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        speech.timer_ms = 5_000; // shorter than the bump floor
        let before = speech.timer_ms;
        for _ in 0..50 {
            speech.accelerate_for_perception();
            assert_eq!(
                speech.timer_ms, before,
                "accelerate must never lengthen an already-short timer",
            );
        }
    }

    #[test]
    fn accelerate_for_perception_noop_while_showing() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        speech.timer_ms = TIMER_MAX_MS;
        speech.showing = true;
        speech.accelerate_for_perception();
        assert_eq!(speech.timer_ms, TIMER_MAX_MS);
    }

    // -----------------------------------------------------------------------
    // Additional tests added during health inspection — covering gaps and
    // edge cases not covered in the original 16 tests.
    // -----------------------------------------------------------------------

    /// Zero-weight phrases: if all eligible phrases have weight 0.0, the weighted
    /// selection previously panicked with an empty range. The fix treats it as uniform.
    #[test]
    fn select_phrase_with_all_zero_weights_does_not_panic() {
        const ZERO_WEIGHT_JSON: &[u8] = br#"{
            "version": 1,
            "phrases": {
                "default": [
                    { "text": "A", "weight": 0.0 },
                    { "text": "B", "weight": 0.0 }
                ]
            }
        }"#;
        let speech = SpeechState::new(Some(ZERO_WEIGHT_JSON));
        // Must not panic; should return one of the phrases
        for _ in 0..20 {
            let result = speech.select_phrase(PetState::Idle);
            assert!(
                result.is_some(),
                "Should still pick a phrase even with zero weights"
            );
            let (text, _) = result.unwrap();
            assert!(
                text == "A" || text == "B",
                "Unexpected phrase text: {}",
                text
            );
        }
    }

    /// Single zero-weight phrase — should return it (only option available).
    #[test]
    fn select_phrase_single_zero_weight_returns_the_phrase() {
        const SINGLE_ZERO_JSON: &[u8] = br#"{
            "version": 1,
            "phrases": {
                "default": [
                    { "text": "Lonely", "weight": 0.0 }
                ]
            }
        }"#;
        let speech = SpeechState::new(Some(SINGLE_ZERO_JSON));
        let result = speech.select_phrase(PetState::Idle);
        assert_eq!(result.map(|(t, _)| t).as_deref(), Some("Lonely"));
    }

    /// Bubble position: exact boundary where y == 0 (not < 0) should NOT flip.
    /// y = rolo_y - bubble_h - gap. Flip only when y < 0.
    #[test]
    fn bubble_position_exactly_at_top_edge_does_not_flip() {
        // Arrange: rolo_y such that y = 0 exactly
        // y = rolo_y - bubble_h - gap = 0  =>  rolo_y = bubble_h + gap = 80 + 8 = 88
        let pos = BubbleLayout::compute(500, 88, 128, 200, 80, 1920, 1080, 8, 16, false);
        assert_eq!(pos.y, 0, "y should be exactly 0 (at top edge, not clipped)");
        assert!(!pos.flipped, "y==0 should not trigger flip");
    }

    /// Bubble position: y = -1 triggers flip.
    #[test]
    fn bubble_position_one_pixel_above_top_triggers_flip() {
        // rolo_y = 87 => y = 87 - 80 - 8 = -1 < 0, should flip
        let pos = BubbleLayout::compute(500, 87, 128, 200, 80, 1920, 1080, 8, 16, false);
        assert!(pos.flipped, "y==-1 should trigger flip");
        assert_eq!(
            pos.y,
            87 + 128 + 16,
            "Flipped y should be rolo_y + rolo_size + gap_below"
        );
    }

    /// Bubble position: right-edge clamping — x should not exceed screen_w - bubble_w.
    #[test]
    fn bubble_position_clamps_to_right_edge_exactly() {
        // rolo_x = 1920 - 128 (right edge), centered x = (1920-128) + 64 - 100 = 1756
        // screen_w - bubble_w = 1920 - 200 = 1720
        // 1756 > 1720, so x should be clamped to 1720
        let pos = BubbleLayout::compute(1792, 500, 128, 200, 80, 1920, 1080, 8, 16, false);
        assert_eq!(
            pos.x, 1720,
            "Right-edge clamp: x should be screen_w - bubble_w"
        );
    }

    /// Bubble position: left-edge clamping — x should not go below 0.
    #[test]
    fn bubble_position_clamps_to_left_edge_exactly() {
        // rolo_x = 0, centered x = 0 + 64 - 100 = -36
        // Clamped to 0
        let pos = BubbleLayout::compute(0, 500, 128, 200, 80, 1920, 1080, 8, 16, false);
        assert_eq!(pos.x, 0, "Left-edge clamp: x should be 0");
    }

    /// Bubble position: top_anchored=true flips immediately without needing
    /// the y < 0 fallback. This is the primary path when Rolo is at screen top.
    #[test]
    fn bubble_position_top_anchored_flips_immediately() {
        // With top_anchored=true, even when there's plenty of room above,
        // the bubble should appear below the sprite.
        let pos = BubbleLayout::compute(500, 500, 128, 200, 80, 1920, 1080, 8, 16, true);
        assert!(pos.flipped, "top_anchored=true should always flip");
        // flipped y = sprite_y + sprite_size + gap_below = 500 + 128 + 16 = 644
        assert_eq!(
            pos.y, 644,
            "Flipped y should be sprite_y + sprite_size + gap_below"
        );
        // Centered: x = 500 + 64 - 100 = 464
        assert_eq!(pos.x, 464);
    }

    /// Bubble position: top_anchored=true near bottom of screen clamps y.
    #[test]
    fn bubble_position_top_anchored_clamps_at_bottom() {
        // Rolo near the bottom: sprite_y = 1000, sprite_size = 128
        // flipped y = 1000 + 128 + 16 = 1144, bubble_h = 80
        // 1144 + 80 = 1224 > screen_h (1080), so y should clamp to 1080 - 80 = 1000
        let pos = BubbleLayout::compute(500, 1000, 128, 200, 80, 1920, 1080, 8, 16, true);
        assert!(pos.flipped);
        assert_eq!(pos.y, 1000, "Should clamp to screen_h - bubble_h");
    }

    /// Display duration scales linearly with character count.
    #[test]
    fn display_duration_scales_with_character_count() {
        let short = calculate_display_duration("Hi"); // 2 chars: 2000 + 100 = 2100
        let long = calculate_display_duration("Hello, I'm Rolo and I have a lot to say!");
        // 40 chars: 2000 + 40*50 = 4000
        assert_eq!(short, 2100);
        assert_eq!(long, 4000);
        assert!(
            long > short,
            "Longer text must have longer display duration"
        );
    }

    /// State-filtered phrase IS included when the current state matches.
    #[test]
    fn state_filtered_phrase_included_in_correct_state() {
        let speech = SpeechState::new(Some(MULTI_PHRASE_JSON));
        // "Only when idle" has states: ["idle"]
        // In Idle state, it should be eligible (we run many times to confirm it can appear)
        let mut saw_idle_phrase = false;
        for _ in 0..500 {
            if let Some((text, _)) = speech.select_phrase(PetState::Idle) {
                if text == "Only when idle" {
                    saw_idle_phrase = true;
                    break;
                }
            }
        }
        assert!(
            saw_idle_phrase,
            "State-filtered phrase should appear in its designated state after 500 trials"
        );
    }

    /// After tick fires Show, a subsequent tick with a large dt should return Hide
    /// and leave showing=false with a fresh timer.
    #[test]
    fn tick_after_hide_sets_fresh_timer() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        speech.timer_ms = 0;
        speech.tick(1, PetState::Idle, 1.0); // Show

        // Expire the display timer
        let action = speech.tick(999_999, PetState::Idle, 1.0);
        assert_eq!(action, SpeechAction::Hide);
        assert!(!speech.showing);
        // Timer must be within the expected bounds after reset
        assert!(
            speech.timer_ms >= TIMER_MIN_MS,
            "Timer after hide must be >= TIMER_MIN_MS"
        );
        assert!(
            speech.timer_ms <= TIMER_MAX_MS,
            "Timer after hide must be <= TIMER_MAX_MS"
        );
    }

    /// dismiss() clears display_timer_ms as well as showing.
    #[test]
    fn dismiss_clears_display_timer() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        speech.timer_ms = 0;
        speech.tick(1, PetState::Idle, 1.0); // Show — sets display_timer_ms > 0
        assert!(
            speech.display_timer_ms > 0,
            "display_timer_ms should be set after show"
        );

        speech.dismiss();
        assert_eq!(
            speech.display_timer_ms, 0,
            "dismiss() must clear display_timer_ms"
        );
    }

    /// dismiss_count_session increments on each call and resets to zero.
    #[test]
    fn dismiss_count_increment_and_reset() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        assert_eq!(speech.dismiss_count_session, 0, "starts at 0");
        assert_eq!(speech.increment_dismiss_count(), 1);
        assert_eq!(speech.increment_dismiss_count(), 2);
        assert_eq!(speech.increment_dismiss_count(), 3);
        assert_eq!(speech.dismiss_count_session, 3);
        speech.reset_dismiss_count();
        assert_eq!(speech.dismiss_count_session, 0, "resets to 0");
    }

    /// Phrase object with no text field produces empty string (graceful degradation).
    #[test]
    fn phrase_object_with_no_text_field_produces_empty_string() {
        const NO_TEXT_JSON: &[u8] = br#"{
            "version": 1,
            "phrases": {
                "default": [
                    { "weight": 1.0 }
                ]
            }
        }"#;
        let speech = SpeechState::new(Some(NO_TEXT_JSON));
        // Duration for "" = 2000ms base, no per-char contribution
        if let Some((text, duration)) = speech.select_phrase(PetState::Idle) {
            assert_eq!(text, "", "Missing text field should produce empty string");
            assert_eq!(
                duration, 2000,
                "Empty text duration should be the base 2000ms"
            );
        }
    }

    /// reset_timer_after_suppression clears a deferred phrase and restarts
    /// the random countdown — used when chat or check-in suppression ends.
    #[test]
    fn reset_timer_after_suppression_clears_deferred() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        // Force a deferred phrase by firing a Show then deferring it.
        speech.timer_ms = 0;
        let action = speech.tick(1, PetState::Idle, 0.0);
        let text = match action {
            SpeechAction::Show(t) => t,
            other => panic!("expected Show, got {:?}", other),
        };
        speech.defer_active_show(text);
        assert!(speech.has_deferred(), "precondition: deferred set");

        speech.reset_timer_after_suppression();
        assert!(
            !speech.has_deferred(),
            "reset_timer_after_suppression must clear deferred"
        );
        assert_eq!(speech.deferred_timeout_ms, 0);
        assert!(
            speech.timer_ms >= TIMER_MIN_MS,
            "timer must be in [TIMER_MIN_MS, TIMER_MAX_MS]"
        );
        assert!(
            speech.timer_ms <= TIMER_MAX_MS,
            "timer must be in [TIMER_MIN_MS, TIMER_MAX_MS]"
        );
    }

    /// reset_timer_after_suppression must NOT leave timer_ms at zero — a tick
    /// immediately afterward must NOT fire a backlogged bubble.
    #[test]
    fn reset_timer_after_suppression_does_not_fire_immediately() {
        let mut speech = SpeechState::new(Some(VALID_JSON));
        // Pretend the timer was about to fire (chat opened before it could).
        speech.timer_ms = 0;
        speech.reset_timer_after_suppression();
        // A zero-dt tick must not fire — the timer should be reset to a
        // fresh random interval >= TIMER_MIN_MS.
        let action = speech.tick(0, PetState::Idle, 0.0);
        assert_eq!(
            action,
            SpeechAction::None,
            "reset_timer_after_suppression must not allow an immediate fire"
        );
    }

    /// PRD/rolo-command-center.md Phase 5 Step F: `queue_immediate_phrase`
    /// must surface a hardcoded reaction even when phrases.json failed to
    /// load. The next tick stays silent (deferred slot is checked first),
    /// and the caller consumes the slot via `take_deferred()` to fire the
    /// bubble — same contract as the rest of the deferred-phrase pathway.
    #[test]
    fn queue_immediate_phrase_surfaces_via_deferred_slot() {
        // Start from a totally-silent Rolo — no phrases loaded.
        let mut speech = SpeechState::new(None);
        assert!(!speech.enabled, "precondition: phrases.json absent");

        speech.queue_immediate_phrase("feels different in here...".to_string());

        // After 50ms of tick, the deferred phrase is still waiting for the
        // caller to consume it (tick.rs uses `has_deferred()` + `take_deferred()`
        // to fire it the same way an idle bubble fires). The tick itself
        // returns None — that's the contract `deferred` enforces.
        let action = speech.tick(50, PetState::Idle, 1.0);
        assert_eq!(action, SpeechAction::None);
        assert!(
            speech.has_deferred(),
            "queue_immediate_phrase must populate the deferred slot"
        );

        // Now consume the slot the same way tick.rs would.
        let (text, _dur) = speech
            .take_deferred()
            .expect("queued phrase must be readable");
        assert_eq!(text, "feels different in here...");
        assert!(speech.is_showing(), "take_deferred flips showing=true");
    }

    /// A phrases.json with no "default" key is non-fatal; Rolo stays silent.
    #[test]
    fn no_default_mood_disables_speech() {
        const NO_DEFAULT_JSON: &[u8] = br#"{
            "version": 1,
            "phrases": {
                "happy": ["Yay!"]
            }
        }"#;
        let speech = SpeechState::new(Some(NO_DEFAULT_JSON));
        assert!(!speech.enabled, "No 'default' key should disable speech");
    }
}
