//! Generic interaction engine — the foundation for all interactive conversations.
//!
//! This module is pure logic with NO Tauri dependency, just like `speech.rs`
//! and `state_machine.rs`. It manages timed interaction triggers, active
//! prompts, responses, suppression, and the mood check-in scheduling logic.
//!
//! The tick loop in `tick.rs` calls `InteractionState::tick()` each frame and
//! reacts to the returned `InteractionAction` to show/hide interactive bubbles.
//!
//! Rolo's mood check-ins are scheduled using wall-clock time: two daily windows
//! (morning 9am–12pm, afternoon 1pm–5pm) with gaussian-ish jitter, quiet hours
//! enforcement (8am–10pm), and minimum cooldown between check-ins.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use chrono::{NaiveDate, NaiveTime, Timelike};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

use crate::state_machine::PetState;

// ---------------------------------------------------------------------------
// Constants — normal mode
// ---------------------------------------------------------------------------

/// Jitter range in minutes (±90 minutes from midpoint).
const JITTER_MINUTES: i64 = 90;

/// A scheduled check-in window: a midpoint with jitter, clamped to a range.
///
/// Replaces the old 8-arg `jittered_time` call sites: callers now describe the
/// window declaratively via this struct instead of passing eight loose primitives.
struct WindowSpec {
    midpoint: NaiveTime,
    jitter_minutes: i64,
    clamp_start: NaiveTime,
    clamp_end: NaiveTime,
}

/// Build a `NaiveTime` from `h:m` at runtime. Panics on invalid inputs — only
/// used for the morning/afternoon constants below, which are static and valid.
fn t(h: u32, m: u32) -> NaiveTime {
    NaiveTime::from_hms_opt(h, m, 0).expect("static window constant must be valid")
}

/// Morning check-in window: 10:30 AM ± 90 minutes, clamped to 9:00–12:00.
fn morning_window() -> WindowSpec {
    WindowSpec {
        midpoint: t(10, 30),
        jitter_minutes: JITTER_MINUTES,
        clamp_start: t(9, 0),
        clamp_end: t(12, 0),
    }
}

/// Afternoon check-in window: 3:00 PM ± 90 minutes, clamped to 1:00–5:00 PM.
fn afternoon_window() -> WindowSpec {
    WindowSpec {
        midpoint: t(15, 0),
        jitter_minutes: JITTER_MINUTES,
        clamp_start: t(13, 0),
        clamp_end: t(17, 0),
    }
}

/// Quiet hours start: 8:00 AM.
const QUIET_START_HOUR: u32 = 8;
/// Quiet hours end: 10:00 PM.
const QUIET_END_HOUR: u32 = 22;

/// Minimum cooldown between check-ins in milliseconds (3 hours).
const COOLDOWN_MS: i64 = 3 * 60 * 60 * 1000;

/// After the chat window closes, suppress check-ins for this long. Gives the
/// user breathing room — Rolo shouldn't ping the moment the user finishes
/// talking to him. Applies in both normal and turbo mode.
const POST_CHAT_DELAY_MS: i64 = 120_000;

/// After Rolo wakes from sleep, suppress check-ins for this long. Prevents the
/// "check-in ambushes you the moment Rolo opens his eyes" experience when a
/// trigger landed mid-dream. Applies in both normal and turbo mode.
const POST_SLEEP_DELAY_MS: i64 = 30_000;

/// Default timeout for an interaction prompt in milliseconds (2 minutes).
const DEFAULT_TIMEOUT_MS: i64 = 120_000;

/// Maximum number of mood check-ins that may fire in a single calendar day in
/// normal mode. Restart-pile-up bug fix: without this cap, every fresh launch
/// re-schedules both daily targets in memory, so launching the app multiple
/// times in an evening could fire 4, 6, 8 check-ins. Turbo mode bypasses the
/// cap so testing still cycles freely.
pub const DAILY_FIRE_CAP: u32 = 2;

/// Grace window for "past" wall-clock targets at schedule time. Targets older
/// than this from `now` are dropped (they're stale — the app wasn't running);
/// targets within the window are kept so a slightly-late launch (e.g., 4pm
/// after a 3pm afternoon target) still fires the check-in if the cap allows.
/// Two hours mirrors the existing COOLDOWN_MS rhythm.
const PAST_TARGET_GRACE_MS: i64 = 2 * 60 * 60 * 1000;

// ---------------------------------------------------------------------------
// Constants — turbo/debug mode
// ---------------------------------------------------------------------------

/// Debug mode: check-in interval (15 seconds).
pub(crate) const DEBUG_INTERVAL_MS: i64 = 15_000;
/// Debug mode: minimum cooldown (5 seconds).
const DEBUG_COOLDOWN_MS: i64 = 5_000;
/// Debug mode: prompt timeout (20 seconds). Normal mode uses 2 minutes, which
/// makes the turbo-cadence feel broken — by the time a new check-in could fire,
/// the old prompt is still pending suppression. 20s is long enough to click but
/// short enough to see the ~15s cadence actually cycle during testing.
const DEBUG_TIMEOUT_MS: i64 = 20_000;

// ---------------------------------------------------------------------------
// Types — serialized to/from the frontend
// ---------------------------------------------------------------------------

/// A single UI element within an interaction prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InteractionElement {
    /// A text label — Rolo's question or instruction.
    Text { content: String },
    /// A row of clickable buttons.
    ButtonRow { buttons: Vec<ButtonDef> },
    /// A text input field for free-form responses.
    TextInput {
        placeholder: String,
        max_length: usize,
    },
}

/// A single button definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ButtonDef {
    /// Display label on the button.
    pub label: String,
    /// Value sent back when this button is clicked.
    pub value: String,
    /// Optional visual style hint (e.g., "primary", "secondary").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
}

/// An interaction prompt shown to the user — contains elements to render.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionPrompt {
    /// Identifies the type of interaction (e.g., "mood_checkin").
    pub interaction_id: String,
    /// Unique instance ID for this specific prompt occurrence.
    pub instance_id: String,
    /// UI elements to render, in order.
    pub elements: Vec<InteractionElement>,
    /// Optional timeout in milliseconds before auto-dismiss.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<i64>,
}

/// The user's response to an interaction prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionResponse {
    /// The instance_id of the prompt being responded to.
    pub instance_id: String,
    /// The interaction_id for routing.
    pub interaction_id: String,
    /// What the user chose.
    pub response: ResponseValue,
    /// ISO-8601 timestamp of the response.
    pub timestamp: String,
}

/// The value carried by a response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseValue {
    /// User clicked a button.
    ButtonPress { value: String },
    /// User submitted text.
    TextSubmit { text: String },
    /// User dismissed without responding.
    Dismissed,
}

// ---------------------------------------------------------------------------
// Mood-checkin button vocabulary
// ---------------------------------------------------------------------------
//
// The mood_checkin prompt has exactly two buttons — "Good" and "Just OK".
// The wire-format values ("good" / "ok") were sprinkled across the prompt
// builder, the response router (`reaction_for_response`,
// `mood_event_for_response`), tests, and the eat-trash confirmation.
// The `Mood` enum centralizes that vocabulary; helpers convert at the
// `ButtonDef.value` / wire boundary so the rest of the codebase can match
// on a typed enum exhaustively.

/// The two mood-checkin answers Rolo accepts. `Good` ⇒ Happy reaction;
/// `Ok` ⇒ Disappointed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mood {
    Good,
    Ok,
}

impl Mood {
    /// Wire-format value embedded in `ButtonDef.value` and surfaced to the
    /// frontend. Snake_case matches every other public-API enum in this
    /// codebase.
    pub fn as_wire_str(self) -> &'static str {
        match self {
            Self::Good => "good",
            Self::Ok => "ok",
        }
    }

    /// Parse a wire value back into a `Mood`. Returns `None` for any
    /// other string — non-mood buttons (future check-ins, eat-trash
    /// confirmations) carry their own vocabulary and shouldn't accidentally
    /// route through the mood path.
    pub fn from_wire_str(s: &str) -> Option<Self> {
        match s {
            "good" => Some(Self::Good),
            "ok" => Some(Self::Ok),
            _ => None,
        }
    }
}

/// Identifier for the daily mood check-in prompt. Mirrors the
/// `interaction_id` field on `InteractionPrompt` so routing sites can
/// compare against this constant instead of a bare string literal.
pub const INTERACTION_ID_MOOD_CHECKIN: &str = "mood_checkin";

/// What the tick loop should do after calling `InteractionState::tick()`.
#[derive(Debug, Clone, PartialEq)]
pub enum InteractionAction {
    /// Nothing to do this frame.
    None,
    /// Show this interaction prompt to the user.
    Show(InteractionPrompt),
    /// Hide the current interaction (timeout or dismiss).
    Hide,
}

// ---------------------------------------------------------------------------
// DebugConfig
// ---------------------------------------------------------------------------

/// Configuration read once at startup from `ROLO_DEBUG` env var.
#[derive(Debug, Clone, Default)]
pub struct DebugConfig {
    /// When true: intervals collapse to seconds, quiet hours disabled.
    pub turbo_mode: bool,
}

// ---------------------------------------------------------------------------
// InteractionState — Rolo's conversation scheduler
// ---------------------------------------------------------------------------

/// Manages when and how interactions are triggered and resolved.
///
/// For the mood check-in feature, interactions are scheduled using wall-clock
/// windows (morning and afternoon) with jitter. The engine is generic enough
/// to support future interaction types added via additional triggers.
pub struct InteractionState {
    /// Debug/turbo configuration.
    debug: DebugConfig,

    /// Whether interaction triggers are suppressed (e.g., during eating).
    suppressed: bool,

    /// The currently active prompt, if any.
    active_prompt: Option<InteractionPrompt>,

    /// Remaining timeout for the active prompt (ms).
    active_timeout_ms: i64,

    /// Timestamp (ms since epoch conceptually, but we track as cooldown countdown).
    /// Countdown until another check-in is allowed.
    cooldown_remaining_ms: i64,

    /// The date for which we've scheduled today's check-in targets.
    scheduled_date: Option<NaiveDate>,

    /// Target times for today's check-ins (morning, afternoon).
    /// Once a target is "consumed" (the check-in fires), it's removed.
    scheduled_targets: Vec<NaiveTime>,

    /// Whether we have a pending check-in that was blocked by suppression.
    /// When suppression lifts, we fire it on the next eligible tick.
    pending_trigger: bool,

    /// Counter for generating unique instance IDs.
    instance_counter: u64,

    /// Countdown timer for turbo/debug mode (ms). Only used in turbo mode.
    debug_timer_ms: i64,

    /// Number of check-ins that have fired today (normal mode only — turbo
    /// mode increments but never enforces). Resets on calendar-date rollover.
    fires_today: u32,

    /// The local calendar date `fires_today` is counting against. `None` until
    /// the first fire (or the first restore_from). When `tick()` sees a date
    /// mismatch, the counter resets.
    fires_today_date: Option<NaiveDate>,

    /// Whether the cap counter changed since the last save. The tick loop
    /// reads this flag to decide whether to schedule an off-thread save.
    /// Cleared by `take_dirty()`.
    dirty: bool,

    /// Countdown after the chat window closes before a check-in may fire (ms).
    /// Armed on the falling edge of `chat_open` in `tick()`. Session-scoped,
    /// not persisted — same lifetime as `cooldown_remaining_ms`.
    post_chat_delay_ms: i64,

    /// Whether the chat window was open on the previous tick. Used to detect
    /// the falling edge so we can arm `post_chat_delay_ms`.
    chat_was_open: bool,

    /// Countdown after Rolo wakes from sleep before a check-in may fire (ms).
    /// Armed on the falling edge of `pet_state == Sleeping` in `tick()`.
    /// Same lifetime as `cooldown_remaining_ms`.
    post_sleep_delay_ms: i64,

    /// Whether Rolo was sleeping on the previous tick. Used to detect the
    /// falling edge so we can arm `post_sleep_delay_ms`.
    was_sleeping: bool,
}

// ---------------------------------------------------------------------------
// Persisted state — the durable subset of InteractionState
// ---------------------------------------------------------------------------

/// Subset of `InteractionState` that survives across restarts. The schedule,
/// active prompt, cooldown, and pending_trigger are intentionally NOT persisted
/// — they're per-session ephemera. Only the daily fire counter needs to outlive
/// a process restart, otherwise the cap resets every launch.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct InteractionPersistedState {
    /// How many check-ins fired today.
    #[serde(default)]
    pub fires_today: u32,
    /// The date `fires_today` counts against (local). None on first run.
    #[serde(default)]
    pub fires_today_date: Option<NaiveDate>,
    /// Schema version for future-proofing.
    #[serde(default = "InteractionPersistedState::default_version")]
    pub version: u32,
}

impl InteractionPersistedState {
    pub fn default_version() -> u32 {
        1
    }
}

impl InteractionState {
    /// Create a new InteractionState with the given debug configuration.
    pub fn new(debug: DebugConfig) -> Self {
        let debug_timer_ms = if debug.turbo_mode {
            DEBUG_INTERVAL_MS
        } else {
            0
        };

        InteractionState {
            debug,
            suppressed: false,
            active_prompt: None,
            active_timeout_ms: 0,
            cooldown_remaining_ms: 0,
            scheduled_date: None,
            scheduled_targets: Vec::new(),
            pending_trigger: false,
            instance_counter: 0,
            debug_timer_ms,
            fires_today: 0,
            fires_today_date: None,
            dirty: false,
            post_chat_delay_ms: 0,
            chat_was_open: false,
            post_sleep_delay_ms: 0,
            was_sleeping: false,
        }
    }

    /// Restore persisted fields from disk. Called once at startup, before the
    /// state is wrapped in a Mutex. Idempotent — safe to call on a fresh state.
    pub fn restore_from(&mut self, persisted: InteractionPersistedState) {
        self.fires_today = persisted.fires_today;
        self.fires_today_date = persisted.fires_today_date;
        // Restoring durable state is not itself a change — only mutations
        // after restore should mark dirty.
        self.dirty = false;
    }

    /// Snapshot of the persisted subset. Used by the tick loop to schedule
    /// off-thread saves without holding the InteractionState mutex.
    pub fn persisted_snapshot(&self) -> InteractionPersistedState {
        InteractionPersistedState {
            fires_today: self.fires_today,
            fires_today_date: self.fires_today_date,
            version: InteractionPersistedState::default_version(),
        }
    }

    /// Returns `true` if the daily counter changed since the last call, and
    /// clears the dirty flag. The tick loop calls this each frame and only
    /// dispatches a save when something actually changed.
    pub fn take_dirty(&mut self) -> bool {
        let was = self.dirty;
        self.dirty = false;
        was
    }

    /// Test/inspection: how many check-ins have fired today.
    #[cfg(test)]
    fn fires_today(&self) -> u32 {
        self.fires_today
    }

    /// Whether an interaction is currently being shown to the user.
    pub fn is_active(&self) -> bool {
        self.active_prompt.is_some()
    }

    /// Toggle suppression (e.g., during eating, dragging).
    pub fn suppress(&mut self, on: bool) {
        self.suppressed = on;
    }

    #[allow(dead_code)]
    pub fn is_suppressed(&self) -> bool {
        self.suppressed
    }

    /// Tick the interaction engine forward.
    ///
    /// `dt_ms` — elapsed milliseconds since last tick.
    /// `pet_state` — Rolo's current behavioral state.
    /// `now_time` — current wall-clock time (local).
    /// `now_date` — current date (local).
    /// `chat_open` — whether the chat window is currently open. Suppresses
    ///   firing while open, and arms a `POST_CHAT_DELAY_MS` breathing-room
    ///   countdown on the falling edge.
    ///
    /// Returns an `InteractionAction` telling the tick loop what to do.
    pub fn tick(
        &mut self,
        dt_ms: i64,
        pet_state: PetState,
        now_time: NaiveTime,
        now_date: NaiveDate,
        chat_open: bool,
    ) -> InteractionAction {
        let is_sleeping = pet_state == PetState::Sleeping;

        // --- Handle active prompt timeout ---
        if let Some(_prompt) = &self.active_prompt {
            // Sleep dismisses any showing prompt outright — the bubble would
            // otherwise stay glued to a sleeping Rolo and confuse the user.
            if is_sleeping {
                self.active_prompt = None;
                self.active_timeout_ms = 0;
                self.was_sleeping = true;
                return InteractionAction::Hide;
            }
            self.active_timeout_ms -= dt_ms;
            if self.active_timeout_ms <= 0 {
                // Timed out — auto-dismiss
                self.active_prompt = None;
                self.active_timeout_ms = 0;
                return InteractionAction::Hide;
            }
            return InteractionAction::None;
        }

        // --- Track chat window state ---
        // On the falling edge (chat just closed), arm the post-chat delay so
        // Rolo gives the user breathing room before pinging them again.
        if self.chat_was_open && !chat_open {
            self.post_chat_delay_ms = POST_CHAT_DELAY_MS;
        }
        self.chat_was_open = chat_open;

        // --- Track sleep state ---
        // On the falling edge (just woke up), arm the post-sleep delay so a
        // pending trigger doesn't ambush the user the instant Rolo opens his
        // eyes. Mirror chat's pattern exactly.
        if self.was_sleeping && !is_sleeping {
            self.post_sleep_delay_ms = POST_SLEEP_DELAY_MS;
        }
        self.was_sleeping = is_sleeping;

        // --- Count down cooldown + delays ---
        if self.cooldown_remaining_ms > 0 {
            self.cooldown_remaining_ms = (self.cooldown_remaining_ms - dt_ms).max(0);
        }
        if self.post_chat_delay_ms > 0 {
            self.post_chat_delay_ms = (self.post_chat_delay_ms - dt_ms).max(0);
        }
        if self.post_sleep_delay_ms > 0 {
            self.post_sleep_delay_ms = (self.post_sleep_delay_ms - dt_ms).max(0);
        }

        // --- Check if we should fire a check-in ---
        let should_fire = if self.debug.turbo_mode {
            self.tick_debug_timer(dt_ms)
        } else {
            self.tick_wall_clock(now_time, now_date)
        };

        if should_fire || self.pending_trigger {
            // Whitelist gate: only fire when Rolo is genuinely Idle. Any other
            // state (eating, walking, dragging, future ones) parks the trigger
            // until he returns to Idle.
            if self.suppressed || chat_open || pet_state != PetState::Idle {
                self.pending_trigger = true;
                return InteractionAction::None;
            }

            // Post-chat breathing room. Hold off without clearing
            // `pending_trigger` so we fire as soon as the delay elapses.
            if self.post_chat_delay_ms > 0 {
                self.pending_trigger = true;
                return InteractionAction::None;
            }

            // Post-sleep breathing room — same parked-trigger pattern.
            if self.post_sleep_delay_ms > 0 {
                self.pending_trigger = true;
                return InteractionAction::None;
            }

            // Check cooldown
            if self.cooldown_remaining_ms > 0 && !self.pending_trigger {
                // Cooldown still active — skip this trigger, don't mark pending
                return InteractionAction::None;
            }

            // Check quiet hours (not in turbo mode)
            if !self.debug.turbo_mode && !is_within_quiet_hours(now_time) {
                return InteractionAction::None;
            }

            // Daily cap (normal mode only). Roll the counter over on a new
            // calendar day BEFORE checking the cap so a date change resets us.
            // If the cap is exhausted, also clear `pending_trigger` so we don't
            // retry every tick — the user's eat-then-unsuppress flow would
            // otherwise be a slow retry loop.
            if !self.debug.turbo_mode {
                self.maybe_roll_daily_counter(now_date);
                if self.fires_today >= DAILY_FIRE_CAP {
                    self.pending_trigger = false;
                    return InteractionAction::None;
                }
            }

            // Fire the check-in!
            self.pending_trigger = false;
            let cooldown = if self.debug.turbo_mode {
                DEBUG_COOLDOWN_MS
            } else {
                COOLDOWN_MS
            };
            self.cooldown_remaining_ms = cooldown;

            // Increment the daily counter (normal mode only). Mark dirty so
            // the tick loop saves the change to disk on its next save tick.
            if !self.debug.turbo_mode {
                self.fires_today = self.fires_today.saturating_add(1);
                self.fires_today_date = Some(now_date);
                self.dirty = true;
            }

            let prompt = self.build_mood_checkin_prompt();
            self.active_prompt = Some(prompt.clone());
            self.active_timeout_ms = prompt.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS);

            return InteractionAction::Show(prompt);
        }

        InteractionAction::None
    }

    /// Reset `fires_today` to 0 if `now_date` differs from the stored date.
    /// Also marks dirty so the rollover persists. No-op if dates already match
    /// or `fires_today_date` is None (the first fire will set the date).
    fn maybe_roll_daily_counter(&mut self, now_date: NaiveDate) {
        match self.fires_today_date {
            Some(d) if d == now_date => {} // same day — keep counting
            Some(_) => {
                // Date changed — reset.
                self.fires_today = 0;
                self.fires_today_date = Some(now_date);
                self.dirty = true;
            }
            None => {
                // First check ever. Don't mark dirty yet; the actual fire
                // (or first save tick) will pin the date.
            }
        }
    }

    /// Handle a user response to the active interaction.
    ///
    /// Returns the response for routing to feature-specific handlers (mood
    /// dispatch, vault logging), or None if no interaction was active.
    pub fn respond(&mut self, response: InteractionResponse) -> Option<InteractionResponse> {
        if let Some(ref prompt) = self.active_prompt {
            if prompt.instance_id == response.instance_id {
                self.active_prompt = None;
                self.active_timeout_ms = 0;
                return Some(response);
            }
        }
        None
    }

    #[allow(dead_code)]
    pub fn dismiss(&mut self) -> bool {
        if self.active_prompt.is_some() {
            self.active_prompt = None;
            self.active_timeout_ms = 0;
            true
        } else {
            false
        }
    }

    #[allow(dead_code)]
    pub fn active_prompt(&self) -> Option<&InteractionPrompt> {
        self.active_prompt.as_ref()
    }

    // -----------------------------------------------------------------------
    // Private — wall-clock scheduling
    // -----------------------------------------------------------------------

    /// Check wall-clock schedule and return true if a check-in should fire.
    fn tick_wall_clock(&mut self, now_time: NaiveTime, now_date: NaiveDate) -> bool {
        // Ensure we have targets for today. Pass `now_time` so we can drop
        // targets that lay too far in the past — restart-pile-up bug fix.
        if self.scheduled_date != Some(now_date) {
            self.schedule_day(now_date, now_time);
        }

        // Check if any target time has passed
        if let Some(pos) = self
            .scheduled_targets
            .iter()
            .position(|target| now_time >= *target)
        {
            // Consume this target
            self.scheduled_targets.remove(pos);
            return true;
        }

        false
    }

    /// Generate the scheduled check-in times for a given date.
    ///
    /// Targets older than `PAST_TARGET_GRACE_MS` from `now_time` are dropped
    /// — they're stale (the app wasn't running). Targets within the grace
    /// window are kept so a slightly-late launch (e.g., 4pm after a 3pm
    /// afternoon target) can still fire if the daily cap allows.
    fn schedule_day(&mut self, date: NaiveDate, now_time: NaiveTime) {
        self.scheduled_date = Some(date);
        self.scheduled_targets.clear();

        let mut rng = rand::rng();

        // Morning target: 10:30 AM ± 90 minutes → clamp to 9:00–12:00
        let morning = jittered_time(&mut rng, &morning_window());

        // Afternoon target: 3:00 PM ± 90 minutes → clamp to 1:00–5:00 PM
        let afternoon = jittered_time(&mut rng, &afternoon_window());

        let grace = chrono::Duration::milliseconds(PAST_TARGET_GRACE_MS);
        let cutoff = now_time - grace;
        for target in [morning, afternoon] {
            // Keep target if it's in the future, OR within the grace window.
            // `cutoff` may underflow into the previous day — chrono's NaiveTime
            // wraps, so compare via signed_duration_since to avoid wraparound
            // bugs. A target from "today" is at most 24h ahead/behind cutoff.
            let delta = target.signed_duration_since(now_time).num_milliseconds();
            // delta > 0 → target in future today; keep.
            // delta in [-grace, 0] → target just barely passed; keep.
            // delta < -grace → too stale, drop.
            let _ = cutoff; // kept for documentation clarity
            if delta >= -PAST_TARGET_GRACE_MS {
                self.scheduled_targets.push(target);
            }
        }
    }

    /// Debug/turbo mode: simple countdown timer.
    fn tick_debug_timer(&mut self, dt_ms: i64) -> bool {
        self.debug_timer_ms -= dt_ms;
        if self.debug_timer_ms <= 0 {
            self.debug_timer_ms = DEBUG_INTERVAL_MS;
            return true;
        }
        false
    }

    /// Build the mood check-in prompt.
    fn build_mood_checkin_prompt(&mut self) -> InteractionPrompt {
        self.instance_counter += 1;
        InteractionPrompt {
            interaction_id: INTERACTION_ID_MOOD_CHECKIN.to_string(),
            instance_id: format!("mood_checkin_{}", self.instance_counter),
            elements: vec![
                InteractionElement::Text {
                    content: "How ya doing?".to_string(),
                },
                InteractionElement::ButtonRow {
                    buttons: vec![
                        ButtonDef {
                            label: "Good".to_string(),
                            value: Mood::Good.as_wire_str().to_string(),
                            style: Some("primary".to_string()),
                        },
                        ButtonDef {
                            label: "Just OK".to_string(),
                            value: Mood::Ok.as_wire_str().to_string(),
                            style: Some("secondary".to_string()),
                        },
                    ],
                },
                InteractionElement::TextInput {
                    placeholder: "or something else?".to_string(),
                    max_length: 280,
                },
            ],
            timeout_ms: Some(if self.debug.turbo_mode {
                DEBUG_TIMEOUT_MS
            } else {
                DEFAULT_TIMEOUT_MS
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Whether the given time is within quiet hours (8am–10pm inclusive of 8am,
/// exclusive of 10pm).
fn is_within_quiet_hours(time: NaiveTime) -> bool {
    let start = NaiveTime::from_hms_opt(QUIET_START_HOUR, 0, 0).unwrap();
    let end = NaiveTime::from_hms_opt(QUIET_END_HOUR, 0, 0).unwrap();
    time >= start && time < end
}

/// Generate a jittered time around a midpoint, clamped to a window.
fn jittered_time(rng: &mut impl Rng, spec: &WindowSpec) -> NaiveTime {
    let mid_total_min = (spec.midpoint.hour() as i64) * 60 + spec.midpoint.minute() as i64;
    let offset: i64 = rng.random_range(-spec.jitter_minutes..=spec.jitter_minutes);
    let target_min = (mid_total_min + offset).max(0) as u32;

    let h = (target_min / 60).min(23);
    let m = target_min % 60;

    let time = NaiveTime::from_hms_opt(h, m, 0).unwrap_or(spec.midpoint);

    if time < spec.clamp_start {
        spec.clamp_start
    } else if time > spec.clamp_end {
        spec.clamp_end
    } else {
        time
    }
}

// ---------------------------------------------------------------------------
// Persistence — InteractionPersistedState ↔ disk
// ---------------------------------------------------------------------------
//
// Mirrors `mood::path_for` / `load_from_disk` / `save_to_disk`. The daily
// fire counter has to outlive process restarts or the cap is meaningless —
// otherwise launching the app twice in an evening would simply re-fire both
// daily targets. Persistence failures are non-fatal: a confused Rolo who
// remembers nothing is still healthier than one who refuses to start.

/// Resolve the on-disk path for `interaction_state.json`. Lives under the
/// same vault directory as `mood-state.json` for consistency.
pub fn path_for(app: &AppHandle) -> PathBuf {
    let base = app
        .path()
        .app_data_dir()
        .unwrap_or_else(|_| std::env::temp_dir());
    base.join("vault")
        .join("wiki")
        .join("personality")
        .join("interaction-state.json")
}

/// Write the persisted state to disk atomically (write-then-rename). Creates
/// the parent directory if missing. Failure is the caller's to log; we just
/// surface the error.
pub fn save_to_disk(path: &Path, state: &InteractionPersistedState) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(state)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, json)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Load persisted state from disk. Always returns a usable value — missing
/// file, parse failure, or unknown version yield defaults so Rolo always
/// boots. Corrupt files are sidecar-renamed for inspection.
pub fn load_from_disk(path: &Path) -> InteractionPersistedState {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return InteractionPersistedState::default();
        }
        Err(e) => {
            log::warn!("[Rolo] interaction-state load read failed: {}", e);
            return InteractionPersistedState::default();
        }
    };

    match serde_json::from_slice::<InteractionPersistedState>(&bytes) {
        Ok(s) => s,
        Err(e) => {
            log::warn!(
                "[Rolo] interaction-state parse failed: {} — sidecaring corrupt file",
                e
            );
            sidecar_corrupt(path);
            InteractionPersistedState::default()
        }
    }
}

/// Rename a corrupt interaction-state file to a timestamped sidecar.
/// Failure to rename is non-fatal.
fn sidecar_corrupt(path: &Path) {
    let ts = chrono::Utc::now().timestamp();
    let sidecar = path.with_extension(format!("json.corrupt-{}", ts));
    if let Err(e) = fs::rename(path, &sidecar) {
        log::warn!("[Rolo] interaction-state sidecar rename failed: {}", e);
    }
}

// ---------------------------------------------------------------------------
// Tests — the interaction engine must be airtight for Rolo's safety
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_machine::is_eating_state;

    /// Helper to create a default (non-turbo) InteractionState.
    fn normal_state() -> InteractionState {
        InteractionState::new(DebugConfig { turbo_mode: false })
    }

    /// Helper to create a turbo-mode InteractionState.
    fn turbo_state() -> InteractionState {
        InteractionState::new(DebugConfig { turbo_mode: true })
    }

    /// Helper: a time inside quiet hours.
    fn mid_day() -> NaiveTime {
        NaiveTime::from_hms_opt(14, 0, 0).unwrap()
    }

    /// Helper: today's date.
    fn today() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 4, 13).unwrap() // Monday
    }

    /// Helper: a time outside quiet hours (after 10pm).
    fn late_night() -> NaiveTime {
        NaiveTime::from_hms_opt(23, 0, 0).unwrap()
    }

    #[allow(dead_code)]
    fn early_morning() -> NaiveTime {
        NaiveTime::from_hms_opt(6, 0, 0).unwrap()
    }

    /// Helper: build a mock response for a given instance_id.
    fn mock_response(instance_id: &str) -> InteractionResponse {
        InteractionResponse {
            instance_id: instance_id.to_string(),
            interaction_id: "mood_checkin".to_string(),
            response: ResponseValue::ButtonPress {
                value: "good".to_string(),
            },
            timestamp: "2026-04-13T14:30:00".to_string(),
        }
    }

    // -----------------------------------------------------------------------
    // Basic construction
    // -----------------------------------------------------------------------

    #[test]
    fn new_state_is_inactive() {
        let state = normal_state();
        assert!(!state.is_active());
        assert!(!state.is_suppressed());
    }

    #[test]
    fn turbo_mode_starts_with_debug_timer() {
        let state = turbo_state();
        assert_eq!(state.debug_timer_ms, DEBUG_INTERVAL_MS);
    }

    // -----------------------------------------------------------------------
    // Turbo mode — timer countdown and trigger
    // -----------------------------------------------------------------------

    #[test]
    fn turbo_timer_counts_down_and_fires() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Tick forward but not enough to fire
        let action = state.tick(5_000, PetState::Idle, time, date, false);
        assert_eq!(action, InteractionAction::None);
        assert_eq!(state.debug_timer_ms, DEBUG_INTERVAL_MS - 5_000);

        // Tick past the threshold
        let action = state.tick(DEBUG_INTERVAL_MS - 4_000, PetState::Idle, time, date, false);
        assert!(matches!(action, InteractionAction::Show(_)));
        assert!(state.is_active());
    }

    #[test]
    fn turbo_timer_resets_after_firing() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Force fire
        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Idle, time, date, false);
        assert!(matches!(action, InteractionAction::Show(_)));

        // Dismiss so we can fire again
        state.dismiss();

        // Timer should have reset
        assert_eq!(state.debug_timer_ms, DEBUG_INTERVAL_MS);
    }

    #[test]
    fn turbo_mode_ignores_quiet_hours() {
        let mut state = turbo_state();
        let date = today();

        // Fire at 11pm — would be outside quiet hours in normal mode
        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Idle, late_night(), date, false);
        assert!(
            matches!(action, InteractionAction::Show(_)),
            "Turbo mode should ignore quiet hours"
        );
    }

    #[test]
    fn turbo_cooldown_is_short() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Fire first check-in
        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Idle, time, date, false);
        assert!(matches!(action, InteractionAction::Show(_)));
        state.dismiss();

        // Cooldown should be DEBUG_COOLDOWN_MS (5s)
        assert_eq!(state.cooldown_remaining_ms, DEBUG_COOLDOWN_MS);

        // Tick past debug timer but within cooldown
        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Idle, time, date, false);
        assert_eq!(
            action,
            InteractionAction::None,
            "Should not fire during cooldown"
        );

        // Tick past cooldown
        state.cooldown_remaining_ms = 0;
        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Idle, time, date, false);
        assert!(
            matches!(action, InteractionAction::Show(_)),
            "Should fire after cooldown expires"
        );
    }

    // -----------------------------------------------------------------------
    // Quiet hours
    // -----------------------------------------------------------------------

    #[test]
    fn quiet_hours_boundaries() {
        // 8:00 AM is within quiet hours
        assert!(is_within_quiet_hours(
            NaiveTime::from_hms_opt(8, 0, 0).unwrap()
        ));
        // 9:59 PM is within quiet hours
        assert!(is_within_quiet_hours(
            NaiveTime::from_hms_opt(21, 59, 0).unwrap()
        ));
        // 10:00 PM is NOT within quiet hours (exclusive end)
        assert!(!is_within_quiet_hours(
            NaiveTime::from_hms_opt(22, 0, 0).unwrap()
        ));
        // 7:59 AM is NOT within quiet hours
        assert!(!is_within_quiet_hours(
            NaiveTime::from_hms_opt(7, 59, 0).unwrap()
        ));
        // Midnight is NOT within quiet hours
        assert!(!is_within_quiet_hours(
            NaiveTime::from_hms_opt(0, 0, 0).unwrap()
        ));
    }

    #[test]
    fn normal_mode_respects_quiet_hours() {
        let mut state = normal_state();
        let date = today();

        // Manually inject a target that has already passed (to force a trigger)
        state.scheduled_date = Some(date);
        state.scheduled_targets = vec![NaiveTime::from_hms_opt(22, 30, 0).unwrap()];

        // Tick at 11pm — target passed but outside quiet hours
        let action = state.tick(100, PetState::Idle, late_night(), date, false);
        assert_eq!(
            action,
            InteractionAction::None,
            "Should not fire outside quiet hours"
        );
    }

    // -----------------------------------------------------------------------
    // Wall-clock scheduling
    // -----------------------------------------------------------------------

    #[test]
    fn wall_clock_schedules_two_targets_per_day() {
        let mut state = normal_state();
        let date = today();
        let early = NaiveTime::from_hms_opt(7, 0, 0).unwrap();

        // Tick once to trigger scheduling
        state.tick(100, PetState::Idle, early, date, false);

        assert_eq!(state.scheduled_date, Some(date));
        assert_eq!(
            state.scheduled_targets.len(),
            2,
            "Should schedule morning and afternoon targets"
        );
    }

    #[test]
    fn wall_clock_fires_when_target_time_passes() {
        let mut state = normal_state();
        let date = today();

        // Manually set a target time that has already passed
        state.scheduled_date = Some(date);
        state.scheduled_targets = vec![NaiveTime::from_hms_opt(10, 0, 0).unwrap()];

        // Tick at 10:30 — target should fire
        let time = NaiveTime::from_hms_opt(10, 30, 0).unwrap();
        let action = state.tick(100, PetState::Idle, time, date, false);

        assert!(
            matches!(action, InteractionAction::Show(_)),
            "Should fire when wall-clock target passes"
        );
        assert!(
            state.scheduled_targets.is_empty(),
            "Consumed target should be removed"
        );
    }

    #[test]
    fn wall_clock_does_not_fire_before_target() {
        let mut state = normal_state();
        let date = today();

        // Set a target in the future
        state.scheduled_date = Some(date);
        state.scheduled_targets = vec![NaiveTime::from_hms_opt(15, 0, 0).unwrap()];

        // Tick at 10:00 — before target
        let time = NaiveTime::from_hms_opt(10, 0, 0).unwrap();
        let action = state.tick(100, PetState::Idle, time, date, false);

        assert_eq!(
            action,
            InteractionAction::None,
            "Should not fire before target time"
        );
        assert_eq!(state.scheduled_targets.len(), 1, "Target should remain");
    }

    #[test]
    fn new_day_reschedules_targets() {
        let mut state = normal_state();
        let date1 = NaiveDate::from_ymd_opt(2026, 4, 13).unwrap();
        let date2 = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let time = NaiveTime::from_hms_opt(7, 0, 0).unwrap();

        // Schedule for day 1
        state.tick(100, PetState::Idle, time, date1, false);
        assert_eq!(state.scheduled_date, Some(date1));

        // Advance to day 2
        state.tick(100, PetState::Idle, time, date2, false);
        assert_eq!(state.scheduled_date, Some(date2));
        assert_eq!(
            state.scheduled_targets.len(),
            2,
            "New day should get fresh targets"
        );
    }

    #[test]
    fn app_launched_after_quiet_hours_does_not_fire() {
        let mut state = normal_state();
        let date = today();

        // Launch at 11pm — no targets will have been scheduled yet,
        // and quiet hours should prevent firing.
        let time = NaiveTime::from_hms_opt(23, 0, 0).unwrap();
        let action = state.tick(100, PetState::Idle, time, date, false);

        // Even though all daily targets have technically "passed",
        // quiet hours should block them.
        assert_eq!(
            action,
            InteractionAction::None,
            "Should not fire after quiet hours even if targets passed"
        );
    }

    #[test]
    fn app_launched_mid_day_fires_passed_target() {
        let mut state = normal_state();
        let date = today();

        // Manually set morning target at 10:00 AM
        state.scheduled_date = Some(date);
        state.scheduled_targets = vec![
            NaiveTime::from_hms_opt(10, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(15, 0, 0).unwrap(),
        ];

        // App launches at 11am — morning target has passed
        let time = NaiveTime::from_hms_opt(11, 0, 0).unwrap();
        let action = state.tick(100, PetState::Idle, time, date, false);

        assert!(
            matches!(action, InteractionAction::Show(_)),
            "Should fire for a passed morning target"
        );
        // Only afternoon target should remain
        assert_eq!(state.scheduled_targets.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Cooldown enforcement
    // -----------------------------------------------------------------------

    #[test]
    fn cooldown_prevents_rapid_fire() {
        let mut state = normal_state();
        let date = today();
        let time = mid_day();

        // Set up two targets, both already passed
        state.scheduled_date = Some(date);
        state.scheduled_targets = vec![
            NaiveTime::from_hms_opt(10, 0, 0).unwrap(),
            NaiveTime::from_hms_opt(11, 0, 0).unwrap(),
        ];

        // First fires
        let action = state.tick(100, PetState::Idle, time, date, false);
        assert!(matches!(action, InteractionAction::Show(_)));
        state.dismiss();

        // Second target has passed, but cooldown is active
        let action = state.tick(100, PetState::Idle, time, date, false);
        assert_eq!(
            action,
            InteractionAction::None,
            "Cooldown should prevent second check-in"
        );
    }

    #[test]
    fn cooldown_counts_down() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Fire and dismiss
        state.debug_timer_ms = 1;
        state.tick(2, PetState::Idle, time, date, false);
        state.dismiss();

        let initial_cooldown = state.cooldown_remaining_ms;
        assert!(initial_cooldown > 0);

        // Tick forward
        state.tick(2_000, PetState::Idle, time, date, false);
        assert_eq!(
            state.cooldown_remaining_ms,
            (initial_cooldown - 2_000).max(0)
        );
    }

    // -----------------------------------------------------------------------
    // Suppression
    // -----------------------------------------------------------------------

    #[test]
    fn suppression_blocks_trigger() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        state.suppress(true);
        state.debug_timer_ms = 1;

        let action = state.tick(2, PetState::Idle, time, date, false);
        assert_eq!(
            action,
            InteractionAction::None,
            "Suppressed state should block triggers"
        );
        assert!(
            state.pending_trigger,
            "Blocked trigger should be marked pending"
        );
    }

    #[test]
    fn suppression_lifts_fires_pending() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Suppress and trigger
        state.suppress(true);
        state.debug_timer_ms = 1;
        state.tick(2, PetState::Idle, time, date, false);
        assert!(state.pending_trigger);

        // Lift suppression
        state.suppress(false);
        state.cooldown_remaining_ms = 0; // Clear cooldown for this test

        let action = state.tick(1, PetState::Idle, time, date, false);
        assert!(
            matches!(action, InteractionAction::Show(_)),
            "Lifting suppression should fire pending trigger"
        );
        assert!(!state.pending_trigger);
    }

    #[test]
    fn eating_state_suppresses_trigger() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        state.debug_timer_ms = 1;

        // Rolo is eating — should suppress
        let action = state.tick(2, PetState::Eating, time, date, false);
        assert_eq!(action, InteractionAction::None);
        assert!(state.pending_trigger);
    }

    #[test]
    fn sniffing_state_suppresses_trigger() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        state.debug_timer_ms = 1;

        let action = state.tick(2, PetState::Sniffing, time, date, false);
        assert_eq!(action, InteractionAction::None);
        assert!(state.pending_trigger);
    }

    #[test]
    fn drag_hover_suppresses_trigger() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        state.debug_timer_ms = 1;

        let action = state.tick(2, PetState::DragHover, time, date, false);
        assert_eq!(action, InteractionAction::None);
        assert!(state.pending_trigger);
    }

    #[test]
    fn idle_state_does_not_suppress() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        state.debug_timer_ms = 1;

        let action = state.tick(2, PetState::Idle, time, date, false);
        assert!(matches!(action, InteractionAction::Show(_)));
    }

    #[test]
    fn non_idle_states_suppress_trigger() {
        // Future-proof: only PetState::Idle lets a check-in fire. Adding new
        // PetStates should automatically suppress without touching this gate.
        for state_under_test in [
            PetState::WalkLeft,
            PetState::WalkRight,
            PetState::Happy,
            PetState::Satisfied,
            PetState::Disappointed,
            PetState::Sleeping,
        ] {
            let mut state = turbo_state();
            let time = mid_day();
            let date = today();
            state.debug_timer_ms = 1;

            let action = state.tick(2, state_under_test, time, date, false);
            assert_eq!(
                action,
                InteractionAction::None,
                "{:?} should suppress check-ins",
                state_under_test
            );
            assert!(
                state.pending_trigger,
                "{:?} should park the trigger as pending",
                state_under_test
            );
        }
    }

    // -----------------------------------------------------------------------
    // Response handling
    // -----------------------------------------------------------------------

    #[test]
    fn respond_clears_active_prompt() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Trigger a prompt
        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Idle, time, date, false);
        let prompt = match action {
            InteractionAction::Show(p) => p,
            _ => panic!("Expected Show"),
        };

        // Respond
        let response = mock_response(&prompt.instance_id);
        let result = state.respond(response);

        assert!(result.is_some());
        assert!(!state.is_active());
    }

    #[test]
    fn respond_with_wrong_instance_id_returns_none() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Trigger a prompt
        state.debug_timer_ms = 1;
        state.tick(2, PetState::Idle, time, date, false);

        // Respond with wrong instance_id
        let response = mock_response("wrong_id_12345");
        let result = state.respond(response);

        assert!(result.is_none(), "Wrong instance_id should be rejected");
        assert!(state.is_active(), "Prompt should still be active");
    }

    #[test]
    fn respond_when_no_prompt_active_returns_none() {
        let mut state = normal_state();

        let response = mock_response("mood_checkin_1");
        let result = state.respond(response);

        assert!(result.is_none());
    }

    #[test]
    fn respond_carries_response_value_through() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Idle, time, date, false);
        let prompt = match action {
            InteractionAction::Show(p) => p,
            _ => panic!("Expected Show"),
        };

        let response = InteractionResponse {
            instance_id: prompt.instance_id,
            interaction_id: "mood_checkin".to_string(),
            response: ResponseValue::TextSubmit {
                text: "feeling great".to_string(),
            },
            timestamp: "2026-04-13T14:30:00".to_string(),
        };

        let result = state.respond(response).unwrap();
        assert_eq!(
            result.response,
            ResponseValue::TextSubmit {
                text: "feeling great".to_string()
            }
        );
    }

    // -----------------------------------------------------------------------
    // Timeout auto-dismiss
    // -----------------------------------------------------------------------

    #[test]
    fn timeout_auto_dismisses() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Trigger a prompt
        state.debug_timer_ms = 1;
        state.tick(2, PetState::Idle, time, date, false);
        assert!(state.is_active());

        // Tick past the timeout
        let action = state.tick(DEFAULT_TIMEOUT_MS + 1, PetState::Idle, time, date, false);
        assert_eq!(action, InteractionAction::Hide);
        assert!(!state.is_active());
    }

    #[test]
    fn timeout_counts_down_each_tick() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        state.debug_timer_ms = 1;
        state.tick(2, PetState::Idle, time, date, false);

        let initial_timeout = state.active_timeout_ms;
        assert!(initial_timeout > 0);

        // Tick partially
        state.tick(1_000, PetState::Idle, time, date, false);
        assert_eq!(state.active_timeout_ms, initial_timeout - 1_000);

        // Still active
        assert!(state.is_active());
    }

    #[test]
    fn no_new_trigger_while_prompt_active() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Fire first
        state.debug_timer_ms = 1;
        let action1 = state.tick(2, PetState::Idle, time, date, false);
        assert!(matches!(action1, InteractionAction::Show(_)));

        // Try to fire another — should return None because a prompt is active
        state.debug_timer_ms = 1;
        let action2 = state.tick(2, PetState::Idle, time, date, false);
        assert_eq!(
            action2,
            InteractionAction::None,
            "Should not fire while another prompt is active"
        );
    }

    // -----------------------------------------------------------------------
    // Dismiss
    // -----------------------------------------------------------------------

    #[test]
    fn dismiss_clears_active_prompt() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        state.debug_timer_ms = 1;
        state.tick(2, PetState::Idle, time, date, false);
        assert!(state.is_active());

        let dismissed = state.dismiss();
        assert!(dismissed);
        assert!(!state.is_active());
    }

    #[test]
    fn dismiss_when_no_prompt_returns_false() {
        let mut state = normal_state();
        assert!(!state.dismiss());
    }

    // -----------------------------------------------------------------------
    // Prompt content
    // -----------------------------------------------------------------------

    #[test]
    fn mood_checkin_prompt_has_correct_elements() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Idle, time, date, false);
        let prompt = match action {
            InteractionAction::Show(p) => p,
            _ => panic!("Expected Show"),
        };

        assert_eq!(prompt.interaction_id, "mood_checkin");
        assert_eq!(prompt.elements.len(), 3);

        // Text element
        match &prompt.elements[0] {
            InteractionElement::Text { content } => {
                assert_eq!(content, "How ya doing?");
            }
            _ => panic!("First element should be Text"),
        }

        // ButtonRow
        match &prompt.elements[1] {
            InteractionElement::ButtonRow { buttons } => {
                assert_eq!(buttons.len(), 2);
                assert_eq!(buttons[0].label, "Good");
                assert_eq!(buttons[0].value, "good");
                assert_eq!(buttons[1].label, "Just OK");
                assert_eq!(buttons[1].value, "ok");
            }
            _ => panic!("Second element should be ButtonRow"),
        }

        // TextInput
        match &prompt.elements[2] {
            InteractionElement::TextInput {
                placeholder,
                max_length,
            } => {
                assert_eq!(placeholder, "or something else?");
                assert_eq!(*max_length, 280);
            }
            _ => panic!("Third element should be TextInput"),
        }
    }

    #[test]
    fn instance_ids_are_unique() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Fire first
        state.debug_timer_ms = 1;
        let action1 = state.tick(2, PetState::Idle, time, date, false);
        let id1 = match action1 {
            InteractionAction::Show(p) => p.instance_id,
            _ => panic!("Expected Show"),
        };
        state.dismiss();
        state.cooldown_remaining_ms = 0;

        // Fire second
        state.debug_timer_ms = 1;
        let action2 = state.tick(2, PetState::Idle, time, date, false);
        let id2 = match action2 {
            InteractionAction::Show(p) => p.instance_id,
            _ => panic!("Expected Show"),
        };

        assert_ne!(id1, id2, "Instance IDs must be unique");
    }

    // -----------------------------------------------------------------------
    // Jittered time helper
    // -----------------------------------------------------------------------

    #[test]
    fn jittered_time_stays_within_clamp() {
        let mut rng = rand::rng();
        let morning = morning_window();
        for _ in 0..200 {
            let time = jittered_time(&mut rng, &morning);
            assert!(
                time >= morning.clamp_start && time <= morning.clamp_end,
                "Jittered time {:?} outside morning window",
                time
            );
        }

        let afternoon = afternoon_window();
        for _ in 0..200 {
            let time = jittered_time(&mut rng, &afternoon);
            assert!(
                time >= afternoon.clamp_start && time <= afternoon.clamp_end,
                "Jittered time {:?} outside afternoon window",
                time
            );
        }
    }

    // -----------------------------------------------------------------------
    // Eating state detection
    // -----------------------------------------------------------------------

    #[test]
    fn eating_states_detected_correctly() {
        assert!(is_eating_state(PetState::Sniffing));
        assert!(is_eating_state(PetState::Eating));
        assert!(is_eating_state(PetState::Satisfied));
        assert!(is_eating_state(PetState::Disappointed));
        assert!(is_eating_state(PetState::DragHover));

        assert!(!is_eating_state(PetState::Idle));
        assert!(!is_eating_state(PetState::WalkLeft));
        assert!(!is_eating_state(PetState::WalkRight));
        assert!(!is_eating_state(PetState::Happy));
    }

    // -----------------------------------------------------------------------
    // Serialization — types must round-trip for frontend IPC
    // -----------------------------------------------------------------------

    #[test]
    fn interaction_prompt_serializes_correctly() {
        let prompt = InteractionPrompt {
            interaction_id: "mood_checkin".to_string(),
            instance_id: "mood_checkin_1".to_string(),
            elements: vec![
                InteractionElement::Text {
                    content: "How ya doing?".to_string(),
                },
                InteractionElement::ButtonRow {
                    buttons: vec![ButtonDef {
                        label: "Good".to_string(),
                        value: "good".to_string(),
                        style: Some("primary".to_string()),
                    }],
                },
                InteractionElement::TextInput {
                    placeholder: "type here".to_string(),
                    max_length: 280,
                },
            ],
            timeout_ms: Some(120_000),
        };

        let json = serde_json::to_string(&prompt).unwrap();
        let deserialized: InteractionPrompt = serde_json::from_str(&json).unwrap();
        assert_eq!(prompt, deserialized);
    }

    #[test]
    fn response_value_variants_serialize() {
        let button = ResponseValue::ButtonPress {
            value: "good".to_string(),
        };
        let json = serde_json::to_string(&button).unwrap();
        assert!(json.contains("button_press"));

        let text = ResponseValue::TextSubmit {
            text: "feeling ok".to_string(),
        };
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains("text_submit"));

        let dismissed = ResponseValue::Dismissed;
        let json = serde_json::to_string(&dismissed).unwrap();
        assert!(json.contains("dismissed"));
    }

    #[test]
    fn interaction_response_round_trips() {
        let response = InteractionResponse {
            instance_id: "mood_checkin_1".to_string(),
            interaction_id: "mood_checkin".to_string(),
            response: ResponseValue::ButtonPress {
                value: "ok".to_string(),
            },
            timestamp: "2026-04-13T14:30:00".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: InteractionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response, deserialized);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn pending_trigger_fires_after_eating_ends() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Trigger while eating
        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Eating, time, date, false);
        assert_eq!(action, InteractionAction::None);
        assert!(state.pending_trigger);

        // Eating ends — next tick with Idle should fire
        state.cooldown_remaining_ms = 0;
        let action = state.tick(1, PetState::Idle, time, date, false);
        assert!(
            matches!(action, InteractionAction::Show(_)),
            "Pending trigger should fire when eating ends"
        );
    }

    #[test]
    fn pending_trigger_respects_quiet_hours_in_normal_mode() {
        let mut state = normal_state();
        let date = today();

        // Mark a pending trigger
        state.pending_trigger = true;
        state.cooldown_remaining_ms = 0;

        // Tick outside quiet hours
        let action = state.tick(100, PetState::Idle, late_night(), date, false);
        assert_eq!(
            action,
            InteractionAction::None,
            "Pending trigger should still respect quiet hours"
        );
        // Pending remains set
        assert!(state.pending_trigger);
    }

    #[test]
    fn multiple_suppress_unsuppress_cycles() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Suppress
        state.suppress(true);
        assert!(state.is_suppressed());

        // Unsuppress
        state.suppress(false);
        assert!(!state.is_suppressed());

        // Suppress again
        state.suppress(true);

        // Trigger while suppressed
        state.debug_timer_ms = 1;
        state.tick(2, PetState::Idle, time, date, false);
        assert!(state.pending_trigger);

        // Unsuppress — should fire
        state.suppress(false);
        state.cooldown_remaining_ms = 0;
        let action = state.tick(1, PetState::Idle, time, date, false);
        assert!(matches!(action, InteractionAction::Show(_)));
    }

    #[test]
    fn zero_dt_tick_does_not_panic() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Zero dt should be fine
        let action = state.tick(0, PetState::Idle, time, date, false);
        assert_eq!(action, InteractionAction::None);
    }

    #[test]
    fn large_dt_tick_does_not_panic() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Very large dt
        let action = state.tick(999_999_999, PetState::Idle, time, date, false);
        // Should fire because debug timer expires
        assert!(matches!(action, InteractionAction::Show(_)));
    }

    // -----------------------------------------------------------------------
    // Daily fire cap — Phase 1 of chat-bug-fix
    // -----------------------------------------------------------------------

    /// Helper: make a normal-mode state with N already-passed targets queued
    /// so we can drive `tick()` in normal mode without waiting on jitter.
    fn normal_state_with_passed_targets(date: NaiveDate, n: usize) -> InteractionState {
        let mut state = normal_state();
        state.scheduled_date = Some(date);
        state.scheduled_targets = (0..n)
            .map(|i| NaiveTime::from_hms_opt(9 + i as u32, 0, 0).unwrap())
            .collect();
        state
    }

    #[test]
    fn cap_blocks_third_fire_in_one_day() {
        let date = today();
        let time = mid_day();
        // Queue 3 already-passed targets — more than the cap allows.
        let mut state = normal_state_with_passed_targets(date, 3);

        // Fire #1
        let a1 = state.tick(100, PetState::Idle, time, date, false);
        assert!(matches!(a1, InteractionAction::Show(_)), "fire #1");
        assert_eq!(state.fires_today(), 1);
        state.dismiss();
        state.cooldown_remaining_ms = 0;

        // Fire #2
        let a2 = state.tick(100, PetState::Idle, time, date, false);
        assert!(matches!(a2, InteractionAction::Show(_)), "fire #2");
        assert_eq!(state.fires_today(), 2);
        state.dismiss();
        state.cooldown_remaining_ms = 0;

        // Fire #3 should be blocked by the cap.
        let a3 = state.tick(100, PetState::Idle, time, date, false);
        assert_eq!(
            a3,
            InteractionAction::None,
            "third fire must be blocked by daily cap"
        );
        assert_eq!(state.fires_today(), 2, "counter must not advance past cap");
    }

    #[test]
    fn cap_does_not_apply_in_turbo_mode() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Force three fires in a row.
        for i in 0..3 {
            state.debug_timer_ms = 1;
            state.cooldown_remaining_ms = 0;
            let action = state.tick(2, PetState::Idle, time, date, false);
            assert!(
                matches!(action, InteractionAction::Show(_)),
                "turbo fire #{} must succeed (cap is bypassed in turbo)",
                i + 1
            );
            state.dismiss();
        }
        // Counter is not incremented in turbo mode.
        assert_eq!(state.fires_today(), 0);
    }

    #[test]
    fn cap_resets_on_date_rollover() {
        let date1 = NaiveDate::from_ymd_opt(2026, 4, 13).unwrap();
        let date2 = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let time = mid_day();

        let mut state = normal_state_with_passed_targets(date1, 2);

        // Burn the cap on day 1.
        for _ in 0..2 {
            let a = state.tick(100, PetState::Idle, time, date1, false);
            assert!(matches!(a, InteractionAction::Show(_)));
            state.dismiss();
            state.cooldown_remaining_ms = 0;
        }
        assert_eq!(state.fires_today(), 2);

        // Day 2: targets get rescheduled. Force them to be passed so we can
        // observe a fire (the rescheduled jitter targets won't have passed
        // at mid_day() reliably, depending on jitter).
        state.scheduled_date = Some(date2);
        state.scheduled_targets = vec![NaiveTime::from_hms_opt(10, 0, 0).unwrap()];

        let a = state.tick(100, PetState::Idle, time, date2, false);
        assert!(
            matches!(a, InteractionAction::Show(_)),
            "rollover must reset the cap and allow firing again"
        );
        assert_eq!(
            state.fires_today(),
            1,
            "counter resets to 0 then increments"
        );
        assert_eq!(state.fires_today_date, Some(date2));
    }

    #[test]
    fn pending_trigger_respects_cap() {
        let date = today();
        let time = mid_day();
        let mut state = normal_state_with_passed_targets(date, 3);

        // Burn the cap.
        for _ in 0..2 {
            let a = state.tick(100, PetState::Idle, time, date, false);
            assert!(matches!(a, InteractionAction::Show(_)));
            state.dismiss();
            state.cooldown_remaining_ms = 0;
        }
        assert_eq!(state.fires_today(), 2);

        // Now simulate the eat-then-unsuppress flow that previously bypassed
        // every other guard: mark a pending trigger and tick with no cooldown.
        state.pending_trigger = true;
        state.cooldown_remaining_ms = 0;

        let action = state.tick(100, PetState::Idle, time, date, false);
        assert_eq!(
            action,
            InteractionAction::None,
            "pending_trigger must NOT bypass the daily cap"
        );
        assert!(
            !state.pending_trigger,
            "pending_trigger must be cleared once cap is exhausted, \
             otherwise we retry every tick forever"
        );
    }

    #[test]
    fn schedule_day_drops_targets_older_than_two_hours() {
        let mut state = normal_state();
        let date = today();
        // Launch at 8pm. The afternoon window's latest possible target is 5pm
        // (clamped); at 8pm that is 3h stale — beyond the 2h grace. The
        // morning window tops out at noon — 8h stale. So both targets are
        // unconditionally dropped regardless of jitter.
        let now_time = NaiveTime::from_hms_opt(20, 0, 0).unwrap();
        // Run 20 times to guard against any jitter-dependent flake.
        for _ in 0..20 {
            state.schedule_day(date, now_time);
            assert!(
                state.scheduled_targets.is_empty(),
                "all daily targets are >2h stale at 8pm — must be dropped, \
                 got {:?}",
                state.scheduled_targets
            );
        }
    }

    #[test]
    fn schedule_day_keeps_targets_within_two_hour_grace() {
        let mut state = normal_state();
        let date = today();
        // Launch at 4pm. The afternoon target falls in [1pm, 5pm], so the
        // worst case is a 1pm target — three hours old, which would be
        // dropped. The best case is a 5pm target — one hour in the future,
        // kept. Run schedule_day 50 times: at least one run should have an
        // afternoon target within the 2h grace window (2pm or later).
        let now_time = NaiveTime::from_hms_opt(16, 0, 0).unwrap();
        let mut saw_kept = false;
        for _ in 0..50 {
            state.schedule_day(date, now_time);
            // Any target kept must be no more than 2h before now_time.
            for t in &state.scheduled_targets {
                let delta_ms = t.signed_duration_since(now_time).num_milliseconds();
                assert!(
                    delta_ms >= -PAST_TARGET_GRACE_MS,
                    "kept target {:?} is more than 2h before now {:?}",
                    t,
                    now_time
                );
                saw_kept = true;
            }
        }
        assert!(
            saw_kept,
            "across 50 runs at 4pm, at least one afternoon target should fall in the 2h grace window"
        );
    }

    #[test]
    fn persisted_state_round_trips_through_json() {
        let original = InteractionPersistedState {
            fires_today: 1,
            fires_today_date: Some(NaiveDate::from_ymd_opt(2026, 4, 28).unwrap()),
            version: 1,
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: InteractionPersistedState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, original);

        // Default round-trip too.
        let default = InteractionPersistedState::default();
        let json = serde_json::to_string(&default).expect("serialize default");
        let parsed: InteractionPersistedState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, default);
    }

    #[test]
    fn restore_from_seeds_counter_without_marking_dirty() {
        let mut state = normal_state();
        state.restore_from(InteractionPersistedState {
            fires_today: 1,
            fires_today_date: Some(today()),
            version: 1,
        });
        assert_eq!(state.fires_today(), 1);
        assert_eq!(state.fires_today_date, Some(today()));
        // restore_from is the load path — it should not look like a mutation.
        assert!(!state.dirty);
    }

    #[test]
    fn save_and_load_round_trip_through_disk() {
        let dir = std::env::temp_dir().join(format!(
            "rolo-interaction-tests-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("interaction-state.json");

        let original = InteractionPersistedState {
            fires_today: 2,
            fires_today_date: Some(NaiveDate::from_ymd_opt(2026, 4, 28).unwrap()),
            version: 1,
        };
        save_to_disk(&path, &original).expect("save");
        let loaded = load_from_disk(&path);
        assert_eq!(loaded, original);

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_from_disk_missing_file_returns_default() {
        let path = std::env::temp_dir()
            .join(format!("rolo-interaction-missing-{}", std::process::id()))
            .join("interaction-state.json");
        // Ensure the file does NOT exist.
        let _ = fs::remove_file(&path);
        let loaded = load_from_disk(&path);
        assert_eq!(loaded, InteractionPersistedState::default());
    }

    #[test]
    fn load_from_disk_corrupt_file_sidecars_and_returns_default() {
        let dir = std::env::temp_dir().join(format!(
            "rolo-interaction-corrupt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("interaction-state.json");
        fs::write(&path, b"not-json {{").expect("write corrupt");

        let loaded = load_from_disk(&path);
        assert_eq!(loaded, InteractionPersistedState::default());
        assert!(!path.exists(), "corrupt file should be sidecar-renamed");

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // Chat-window suppression + post-chat delay
    // -----------------------------------------------------------------------

    #[test]
    fn chat_open_suppresses_turbo_fire_and_marks_pending() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Force the timer to fire on the next tick, but with chat open.
        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Idle, time, date, /* chat_open */ true);
        assert_eq!(
            action,
            InteractionAction::None,
            "Check-in must not fire while chat window is open"
        );
        assert!(
            state.pending_trigger,
            "Suppressed fire must be remembered as pending"
        );
        assert!(!state.is_active());
    }

    #[test]
    fn closing_chat_arms_post_chat_delay() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Tick with chat open — establishes prior chat_was_open.
        state.tick(100, PetState::Idle, time, date, true);
        assert_eq!(state.post_chat_delay_ms, 0, "Delay armed only on close");

        // Tick with chat closed — falling edge arms the delay.
        // dt_ms is consumed by the same tick that arms the delay, so the
        // remaining value is POST_CHAT_DELAY_MS - dt_ms.
        state.tick(100, PetState::Idle, time, date, false);
        assert_eq!(
            state.post_chat_delay_ms,
            POST_CHAT_DELAY_MS - 100,
            "Falling edge of chat_open must arm the post-chat delay"
        );
    }

    #[test]
    fn pending_fire_holds_until_post_chat_delay_elapses() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Build up a pending fire while the chat is open.
        state.debug_timer_ms = 1;
        state.tick(2, PetState::Idle, time, date, true);
        assert!(state.pending_trigger);

        // Close the chat — arms the 2-min delay; pending should NOT fire yet.
        let action = state.tick(2, PetState::Idle, time, date, false);
        assert_eq!(action, InteractionAction::None);
        assert!(state.pending_trigger, "Still waiting on post-chat delay");

        // Tick forward almost the full delay — still suppressed.
        let action = state.tick(POST_CHAT_DELAY_MS - 10, PetState::Idle, time, date, false);
        assert_eq!(action, InteractionAction::None);
        assert!(state.pending_trigger);

        // One more tick past the delay — now the pending fire goes.
        let action = state.tick(100, PetState::Idle, time, date, false);
        assert!(
            matches!(action, InteractionAction::Show(_)),
            "Pending check-in must fire once post-chat delay elapses"
        );
        assert!(!state.pending_trigger);
        assert_eq!(state.post_chat_delay_ms, 0);
    }

    #[test]
    fn no_chat_means_no_post_chat_delay() {
        // Regression: with no chat ever opened, cadence is unchanged.
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Idle, time, date, false);
        assert!(matches!(action, InteractionAction::Show(_)));
        assert_eq!(state.post_chat_delay_ms, 0);
        assert!(!state.chat_was_open);
    }

    // -----------------------------------------------------------------------
    // Sleep suppression — check-ins must not fire while Rolo dreams
    // -----------------------------------------------------------------------

    #[test]
    fn active_prompt_is_dismissed_when_pet_enters_sleeping() {
        // Fire a check-in first.
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();
        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Idle, time, date, false);
        assert!(matches!(action, InteractionAction::Show(_)));
        assert!(state.is_active());

        // Now Rolo falls asleep mid-prompt — the bubble must go away.
        let action = state.tick(100, PetState::Sleeping, time, date, false);
        assert_eq!(action, InteractionAction::Hide);
        assert!(!state.is_active());
    }

    #[test]
    fn turbo_trigger_does_not_fire_while_sleeping() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Force the timer to expire while Rolo is asleep.
        state.debug_timer_ms = 1;
        let action = state.tick(2, PetState::Sleeping, time, date, false);

        // The trigger should be parked, NOT fired.
        assert_eq!(action, InteractionAction::None);
        assert!(state.pending_trigger);
        assert!(!state.is_active());
    }

    #[test]
    fn pending_trigger_does_not_fire_during_post_sleep_grace() {
        let mut state = turbo_state();
        let time = mid_day();
        let date = today();

        // Trigger fires while sleeping → parks as pending.
        state.debug_timer_ms = 1;
        state.tick(2, PetState::Sleeping, time, date, false);
        assert!(state.pending_trigger);

        // Wake up on the next tick. The falling edge arms post_sleep_delay.
        let action = state.tick(10, PetState::Idle, time, date, false);
        assert_eq!(action, InteractionAction::None);
        assert!(state.pending_trigger);
        assert!(state.post_sleep_delay_ms > 0);

        // Tick partway through the grace period — still parked.
        let action = state.tick(POST_SLEEP_DELAY_MS / 2, PetState::Idle, time, date, false);
        assert_eq!(action, InteractionAction::None);
        assert!(state.pending_trigger);

        // Tick past the grace period — now it fires.
        let action = state.tick(POST_SLEEP_DELAY_MS, PetState::Idle, time, date, false);
        assert!(matches!(action, InteractionAction::Show(_)));
        assert_eq!(state.post_sleep_delay_ms, 0);
    }
}
