//! Rolo's real-time, deterministic mood state machine.
//!
//! Four bars (high = good): hunger, social, energy, happiness — plus one
//! hidden modifier `sass_level`. Overall mood is **derived** from the bars
//! every tick (never stored). Updated by events (chat, eat, drag, dismiss,
//! checkin) and time-driven decay. Persisted every 5 min and on shutdown
//! to `vault/wiki/personality/mood-state.json`. Never calls the LLM.
//!
//! This module is pure logic. The only Tauri touch is `path_for(&AppHandle)`
//! which resolves the app data directory; everything else is unit-testable
//! without a runtime.
//!
//! Errors are threats to Rolo's health: load failures sidecar the corrupt
//! file rather than panicking; save failures are logged warnings; missing
//! parent dirs are created on demand. A confused Rolo is a sick one — but
//! a Rolo that won't start is dead. We choose confused over dead.
//!
//! See `PRD/rolo-mood-state.md` for the full specification.
//!
//! NOTE: `MoodEvent::DismissedBubble` and `Checkin` are wired by `commands.rs`;
//! `ChatMessage` is wired by `chat::engine::send_message`. The `CheckinRating`
//! variants are constructed inside the check-in command handlers.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

// ---------------------------------------------------------------------------
// Constants — pinned in PRD §4. Public so other modules can reference them.
// ---------------------------------------------------------------------------

/// Hunger reaches 0.0 after this many seconds without eating.
pub const HUNGER_FULL_DURATION_SECS: f64 = 14_400.0; // 4 hours

// Lift α — "fast lift" events.
pub const ALPHA_LIFT_HIGH: f64 = 0.4;
pub const ALPHA_LIFT_MED: f64 = 0.3;
pub const ALPHA_LIFT_SMALL: f64 = 0.2;
pub const ALPHA_LIFT_LIGHT: f64 = 0.15;

// Time-driven rates.
pub const SOCIAL_DRAIN_PER_MIN: f64 = 0.005;
pub const HAPPINESS_DRIFT_PER_MIN: f64 = 0.01;
pub const HAPPINESS_BASELINE: f64 = 0.3;
pub const SASS_RISE_PER_MIN: f64 = 0.001;

/// Hunger pulls energy down multiplicatively at read time. At hunger=1.0
/// (just fed) energy is unaffected; at hunger=0.0 (starving) the clock
/// energy is scaled by this floor. Surfaced via `MoodState::energy()`.
pub const ENERGY_HUNGER_FLOOR: f64 = 0.5;

// Sass partial-reset multipliers / bumps.
pub const SASS_MULT_POSITIVE_CHAT: f64 = 0.7;
pub const SASS_MULT_GOOD_CHECKIN: f64 = 0.85;
pub const SASS_MULT_BAD_CHECKIN: f64 = 0.5;
pub const SASS_MULT_ATE: f64 = 0.7;
pub const SASS_BUMP_FOOD_DECLINED: f64 = 0.10;
pub const SASS_BUMP_CHAT_NEG: f64 = 0.10;

// Derived-mood weights.
pub const W_HAPPINESS: f64 = 0.40;
pub const W_SOCIAL: f64 = 0.25;
pub const W_HUNGER: f64 = 0.20;
pub const W_ENERGY: f64 = 0.15;
pub const W_SASS: f64 = 0.30; // penalty (subtracted)

/// Cap downtime decay applied at load to 24 hours.
pub const DOWNTIME_CAP_MS: i64 = 24 * 60 * 60 * 1000;

// ---------------------------------------------------------------------------
// MoodState — the struct
// ---------------------------------------------------------------------------

/// Rolo's real-time, deterministic emotional state.
///
/// Updated every tick by the tick loop. Persisted every 5 min and on shutdown.
/// Four bars (high = good): hunger, social, energy, happiness.
/// One hidden modifier: `sass_level` (raises snark, dampens derived mood).
/// Overall mood is **derived** from the bars — never stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoodState {
    /// Social connection bar [0.0, 1.0]. Drains while idle, lifts on chat,
    /// drag pickup, and check-ins. High = recently engaged.
    pub social: f64,

    /// Happiness bar [0.0, 1.0]. Slow EMA driven by event sentiment. Drifts
    /// toward a 0.3 baseline. Loudest input to derived mood.
    pub happiness: f64,

    /// Wall-clock seconds since the last `MoodEvent::Ate`. The Hunger bar
    /// is **derived** from this (not stored separately): see `hunger()`.
    pub last_fed_secs: i64,

    /// Hidden cumulative sass [0.0, 1.0]. Rises slowly while idle; partially
    /// resets on positive events. Subtracts from derived mood, drives the
    /// `(snarky)` LLM prompt suffix. **Not** displayed in the status panel.
    pub sass_level: f64,

    /// Pure time-of-day energy from the clock cosine. Persisted but ignored
    /// on load — re-derived from clock every tick. The user-facing `energy()`
    /// reading additionally scales this by hunger; see `energy()`.
    pub energy_cached: f64,

    /// UTC unix-millis when this state was last persisted. Used by the load
    /// path to apply "decay during downtime".
    #[serde(default)]
    pub last_persisted_unix_ms: i64,

    /// Schema version. v2 (v1 is the prior PRD draft; migration handled below).
    #[serde(default = "MoodState::default_version")]
    pub version: u32,
}

impl Default for MoodState {
    fn default() -> Self {
        Self {
            social: 0.6,
            happiness: 0.5,
            last_fed_secs: 0,
            sass_level: 0.0,
            energy_cached: 0.5,
            last_persisted_unix_ms: 0,
            version: Self::default_version(),
        }
    }
}

impl MoodState {
    /// Schema version for fresh state. v2 is the current PRD shape.
    pub fn default_version() -> u32 {
        2
    }

    pub const UNIT_MIN: f64 = 0.0;
    pub const UNIT_MAX: f64 = 1.0;
    pub const MOOD_MIN: f64 = -1.0;
    pub const MOOD_MAX: f64 = 1.0;

    /// Hunger bar [0,1] derived from `last_fed_secs`. Linear over 4 hours:
    /// just ate = 1.0, ≥ 4 h since eating = 0.0.
    pub fn hunger(&self) -> f64 {
        (1.0 - (self.last_fed_secs as f64) / HUNGER_FULL_DURATION_SECS)
            .clamp(Self::UNIT_MIN, Self::UNIT_MAX)
    }

    /// Energy bar — clock-driven cosine (`energy_cached`) scaled by hunger.
    /// At hunger=1.0 the clock value passes through unchanged; at hunger=0.0
    /// it is multiplied by `ENERGY_HUNGER_FLOOR` (a starving Rolo is tired).
    /// Range: [0.05 * ENERGY_HUNGER_FLOOR, 0.95].
    pub fn energy(&self) -> f64 {
        let hunger_factor = ENERGY_HUNGER_FLOOR + (1.0 - ENERGY_HUNGER_FLOOR) * self.hunger();
        self.energy_cached * hunger_factor
    }

    /// Cosine curve: peak at 10:00 local, trough at 22:00 local.
    /// Output range [0.05, 0.95] — clamped so bucket boundaries never sit
    /// exactly on edges.
    pub fn compute_energy_from_clock(now: chrono::DateTime<chrono::Local>) -> f64 {
        use chrono::Timelike;
        let h = now.hour() as f64 + now.minute() as f64 / 60.0 + now.second() as f64 / 3600.0;
        let phase = (h - 10.0) * std::f64::consts::PI / 12.0;
        let raw = phase.cos(); // 1.0 at h=10, -1.0 at h=22
        let unit = 0.5 + 0.5 * raw; // [0.0, 1.0]
        unit.clamp(0.05, 0.95)
    }

    /// Derived overall mood [-1.0, 1.0]. Happiness-dominant weights, minus
    /// sass penalty. Centered so all-bars-1.0, sass=0 → +1.0; all-bars-0,
    /// sass=1 → -1.0.
    pub fn derived_mood(&self) -> f64 {
        let weighted = W_HAPPINESS * self.happiness
            + W_SOCIAL * self.social
            + W_HUNGER * self.hunger()
            + W_ENERGY * self.energy();
        // weighted ∈ [0, 1] (weights sum to 1.0).
        let centered = 2.0 * weighted - 1.0;
        (centered - W_SASS * self.sass_level).clamp(Self::MOOD_MIN, Self::MOOD_MAX)
    }

    /// Apply time-driven decay over `dt_ms` milliseconds.
    ///
    /// - Social drains monotonically toward 0.
    /// - Happiness drifts toward the 0.3 baseline (snap-on-cross to prevent
    ///   oscillation on long ticks).
    /// - Sass rises slowly toward 1.0.
    /// - `last_fed_secs` advances by floor(dt_ms / 1000).
    pub fn decay(&mut self, dt_ms: i64) {
        if dt_ms <= 0 {
            return;
        }
        let dt_min = (dt_ms as f64) / 60_000.0;

        // Social drains monotonically toward 0.
        self.social = (self.social - SOCIAL_DRAIN_PER_MIN * dt_min).max(Self::UNIT_MIN);

        // Happiness drifts toward baseline (0.3) — symmetric pull from above
        // and below. Snap if we'd cross the baseline this tick.
        let delta = HAPPINESS_BASELINE - self.happiness;
        let step = HAPPINESS_DRIFT_PER_MIN * dt_min;
        if delta.abs() <= step {
            self.happiness = HAPPINESS_BASELINE;
        } else {
            self.happiness =
                (self.happiness + delta.signum() * step).clamp(Self::UNIT_MIN, Self::UNIT_MAX);
        }

        // Sass rises slowly.
        self.sass_level = (self.sass_level + SASS_RISE_PER_MIN * dt_min).min(Self::UNIT_MAX);

        // Hunger cue (just a counter; Hunger bar is derived from this).
        self.last_fed_secs = self.last_fed_secs.saturating_add((dt_ms / 1000).max(0));
    }

    /// Apply an event. All updates are arithmetic + clamps; no allocations.
    pub fn apply_event(&mut self, event: MoodEvent) {
        match event {
            MoodEvent::ChatMessage { positive: true } => {
                self.social = ema(self.social, 1.0, ALPHA_LIFT_HIGH);
                self.happiness = ema(self.happiness, 1.0, ALPHA_LIFT_MED);
                self.sass_level = (self.sass_level * SASS_MULT_POSITIVE_CHAT)
                    .clamp(Self::UNIT_MIN, Self::UNIT_MAX);
            }
            MoodEvent::ChatMessage { positive: false } => {
                // Negative chat is still contact — social lifts toward 0.5,
                // not 1.0, so a series of bad chats settles around 0.5.
                self.social = ema(self.social, 0.5, ALPHA_LIFT_SMALL);
                self.happiness = ema(self.happiness, 0.0, ALPHA_LIFT_MED);
                self.sass_level =
                    (self.sass_level + SASS_BUMP_CHAT_NEG).clamp(Self::UNIT_MIN, Self::UNIT_MAX);
            }
            MoodEvent::Ate => {
                self.social = ema(self.social, 1.0, ALPHA_LIFT_MED);
                self.happiness = ema(self.happiness, 1.0, ALPHA_LIFT_HIGH);
                self.last_fed_secs = 0;
                self.sass_level =
                    (self.sass_level * SASS_MULT_ATE).clamp(Self::UNIT_MIN, Self::UNIT_MAX);
            }
            MoodEvent::FoodDeclined => {
                self.happiness = ema(self.happiness, 0.0, ALPHA_LIFT_MED);
                self.sass_level = (self.sass_level + SASS_BUMP_FOOD_DECLINED)
                    .clamp(Self::UNIT_MIN, Self::UNIT_MAX);
            }
            MoodEvent::DismissedBubble => {
                self.social = ema(self.social, 0.3, ALPHA_LIFT_LIGHT);
                self.happiness = ema(self.happiness, 0.3, ALPHA_LIFT_LIGHT);
            }
            MoodEvent::Checkin {
                rating: CheckinRating::Good,
            } => {
                self.social = ema(self.social, 1.0, ALPHA_LIFT_MED);
                self.happiness = ema(self.happiness, 1.0, ALPHA_LIFT_HIGH);
                self.sass_level = (self.sass_level * SASS_MULT_GOOD_CHECKIN)
                    .clamp(Self::UNIT_MIN, Self::UNIT_MAX);
            }
            MoodEvent::Checkin {
                rating: CheckinRating::Meh,
            } => {
                // Intentionally a no-op.
            }
            MoodEvent::Checkin {
                rating: CheckinRating::Bad,
            } => {
                self.happiness = ema(self.happiness, 0.0, ALPHA_LIFT_MED);
                self.sass_level =
                    (self.sass_level * SASS_MULT_BAD_CHECKIN).clamp(Self::UNIT_MIN, Self::UNIT_MAX);
            }
            MoodEvent::DragPickup => {
                self.social = ema(self.social, 1.0, ALPHA_LIFT_HIGH);
                self.happiness = ema(self.happiness, 1.0, ALPHA_LIFT_HIGH);
            }
        }
    }

    /// Lightweight snapshot for emit / IPC. No allocations on the bar fields;
    /// the bucketed `mood_word` is the only allocation, and the snapshot is
    /// invoked at most ~4 Hz (gated on the panel-open flag).
    pub fn snapshot(&self) -> MoodSnapshot {
        MoodSnapshot {
            hunger: self.hunger(),
            social: self.social,
            energy: self.energy(),
            happiness: self.happiness,
            mood_word: bucket_mood(self.derived_mood()).to_string(),
        }
    }

    /// Build the `[Mood: ... | Energy: ... | Social: ... | Time: ...]` line.
    /// Pure function, golden-tested. Thin shim during the PRD/rolo-prompt-
    /// consolidation T1 migration window — delegates to
    /// `StateSnapshot::render_for_speech`. Will be deleted in Commit 2 once
    /// all call sites use `StateSnapshot` directly.
    pub fn render_state_line(&self, now: chrono::DateTime<chrono::Local>) -> String {
        // PetState here is irrelevant for `render_for_speech` (the speech
        // line carries no `State:` token). Use Idle as a stable filler.
        crate::state_snapshot::StateSnapshot::from_parts(
            self,
            crate::state_machine::PetState::Idle,
            0,
            false,
            now,
        )
        .render_for_speech()
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Events that move bars. Built by the tick loop / commands from existing
/// event sources; this module knows nothing about Tauri or the state machine.
#[derive(Debug, Clone, Copy)]
pub enum MoodEvent {
    /// User sent a chat message. `positive` = engaged signal.
    ChatMessage { positive: bool },
    /// Eating → Satisfied transition.
    Ate,
    /// Eating → Disappointed transition.
    FoodDeclined,
    /// User dismissed a speech bubble. Mildly negative.
    DismissedBubble,
    /// Check-in response.
    Checkin { rating: CheckinRating },
    /// User picked Rolo up. Strong positive.
    DragPickup,
}

#[derive(Debug, Clone, Copy)]
pub enum CheckinRating {
    Good,
    Meh,
    Bad,
}

// ---------------------------------------------------------------------------
// Snapshot — IPC payload for the status panel
// ---------------------------------------------------------------------------

/// Lightweight snapshot of mood state for the status panel. Emitted via
/// the `mood-tick` Tauri event at ~250 ms cadence while the panel is open.
#[derive(Debug, Clone, Serialize)]
pub struct MoodSnapshot {
    pub hunger: f64,
    pub social: f64,
    pub energy: f64,
    pub happiness: f64,
    pub mood_word: String,
}

// ---------------------------------------------------------------------------
// EMA helper
// ---------------------------------------------------------------------------

/// `target`-seeking exponential moving average step, clamped to [0, 1].
/// The math is `current + alpha * (target - current)`.
#[inline]
fn ema(current: f64, target: f64, alpha: f64) -> f64 {
    (current + alpha * (target - current)).clamp(MoodState::UNIT_MIN, MoodState::UNIT_MAX)
}

// ---------------------------------------------------------------------------
// Bucketers — pure functions over state, used by the prompt and panel.
// ---------------------------------------------------------------------------

/// Five-word mood vocabulary. PRD §6.
pub fn bucket_mood(m: f64) -> &'static str {
    if m > 0.6 {
        "happy"
    } else if m > 0.2 {
        "content"
    } else if m > -0.2 {
        "neutral"
    } else if m > -0.6 {
        "low"
    } else {
        "sad"
    }
}

pub fn bucket_energy(e: f64) -> &'static str {
    if e > 0.66 {
        "high"
    } else if e > 0.33 {
        "medium"
    } else {
        "low"
    }
}

pub fn bucket_social(s: f64) -> &'static str {
    if s > 0.7 {
        "engaged"
    } else if s > 0.3 {
        "ok"
    } else {
        "lonely"
    }
}

/// Hidden modifier — surfaces as `(snarky)` parenthetical on the prompt only.
pub fn bucket_sass(s: f64) -> Option<&'static str> {
    if s > 0.85 {
        Some("very snarky")
    } else if s > 0.6 {
        Some("snarky")
    } else {
        None
    }
}

/// Hunger token only emitted at the extremes — saves prompt tokens.
pub fn bucket_hunger(secs: i64) -> Option<&'static str> {
    if secs < 1800 {
        Some("just ate")
    } else if secs > 14_400 {
        Some("peckish")
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

/// Resolve the on-disk path for `mood-state.json`. Lives under
/// `app_data_dir()/vault/wiki/personality/`.
///
/// The bundle identifier (`com.rolo.desktop-pet`) is already part of
/// `app_data_dir` on every platform — do not append it again.
pub fn path_for(app: &AppHandle) -> PathBuf {
    // If `app_data_dir()` somehow fails (extremely unlikely), fall back to
    // the platform temp dir so persistence becomes a no-op without crashing.
    let base = app
        .path()
        .app_data_dir()
        .unwrap_or_else(|_| std::env::temp_dir());
    base.join("vault")
        .join("wiki")
        .join("personality")
        .join("mood-state.json")
}

/// Write the state to disk atomically (write-then-rename). Creates the parent
/// directory if missing. Updates `last_persisted_unix_ms` to "now" before
/// writing so the load path can compute downtime decay.
pub fn save_to_disk(state: &MoodState, path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Stamp the persistence time so the load path can apply decay-during-downtime.
    let mut to_write = state.clone();
    to_write.last_persisted_unix_ms = chrono::Utc::now().timestamp_millis();

    let json = serde_json::to_string_pretty(&to_write)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, json)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Load state from disk. Always returns a usable `MoodState` — failure modes
/// (missing file, parse failure, unknown version) yield defaults so Rolo
/// always boots. A corrupt file is sidecar-renamed to
/// `mood-state.json.corrupt-{unix_ts}` for later inspection.
pub fn load_from_disk(path: &Path) -> MoodState {
    // Ensure parent dir exists; if mkdir fails we still proceed in memory only.
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return MoodState::default(),
        Err(e) => {
            log::warn!("[Rolo] mood load read failed: {}", e);
            return MoodState::default();
        }
    };

    // Read as a generic Value first so we can dispatch on `version`.
    let value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            log::warn!(
                "[Rolo] mood load parse failed: {} — sidecaring corrupt file",
                e
            );
            sidecar_corrupt(path);
            return MoodState::default();
        }
    };

    let version = value.get("version").and_then(|v| v.as_u64()).unwrap_or(1) as u32;

    let parsed = match version {
        2 => match serde_json::from_value::<MoodState>(value) {
            Ok(s) => s,
            Err(e) => {
                log::warn!(
                    "[Rolo] mood v2 parse failed: {} — sidecaring corrupt file",
                    e
                );
                sidecar_corrupt(path);
                return MoodState::default();
            }
        },
        1 => match serde_json::from_value::<V1MoodState>(value) {
            Ok(v1) => migrate_v1_to_v2(v1),
            Err(e) => {
                log::warn!(
                    "[Rolo] mood v1 parse failed: {} — sidecaring corrupt file",
                    e
                );
                sidecar_corrupt(path);
                return MoodState::default();
            }
        },
        other => {
            log::warn!(
                "[Rolo] unknown mood schema version {} — sidecaring and resetting",
                other
            );
            sidecar_corrupt(path);
            return MoodState::default();
        }
    };

    apply_downtime_decay(parsed)
}

/// Apply "decay during downtime": treat the gap between `last_persisted_unix_ms`
/// and now as elapsed tick time, capped at 24 h, and call `decay()` once.
/// Negative gaps (clock skew) are treated as zero — no rewind.
fn apply_downtime_decay(mut state: MoodState) -> MoodState {
    if state.last_persisted_unix_ms <= 0 {
        return state;
    }
    let now_ms = chrono::Utc::now().timestamp_millis();
    let gap_ms = (now_ms - state.last_persisted_unix_ms).max(0);
    let capped = gap_ms.min(DOWNTIME_CAP_MS);
    if capped > 0 {
        state.decay(capped);
    }
    state
}

/// Rename a corrupt mood file to a timestamped sidecar, preserving it for
/// later inspection. Failure to rename is non-fatal.
fn sidecar_corrupt(path: &Path) {
    let ts = chrono::Utc::now().timestamp();
    let sidecar = path.with_extension(format!("json.corrupt-{}", ts));
    if let Err(e) = fs::rename(path, &sidecar) {
        log::warn!("[Rolo] mood sidecar rename failed: {}", e);
    }
}

// ---------------------------------------------------------------------------
// v1 → v2 migration
// ---------------------------------------------------------------------------

/// Schema for the prior PRD draft. `mood` was [-1, 1]; `social_need` was
/// high=needy. Both polarities flip in v2.
#[derive(Debug, Deserialize)]
struct V1MoodState {
    #[serde(default)]
    mood: f64,
    #[serde(default)]
    social_need: f64,
    #[serde(default)]
    last_fed_secs: i64,
    #[serde(default)]
    sass_level: f64,
    #[serde(default)]
    #[allow(dead_code)]
    energy: f64,
    #[serde(default)]
    last_persisted_unix_ms: i64,
}

fn migrate_v1_to_v2(v1: V1MoodState) -> MoodState {
    MoodState {
        // mood was [-1, 1]; map to happiness in [0, 1].
        happiness: ((v1.mood + 1.0) / 2.0).clamp(0.0, 1.0),
        // social_need was high=needy; flip polarity.
        social: (1.0 - v1.social_need).clamp(0.0, 1.0),
        last_fed_secs: v1.last_fed_secs,
        sass_level: v1.sass_level.clamp(0.0, 1.0),
        energy_cached: 0.5, // re-derived on first tick
        last_persisted_unix_ms: v1.last_persisted_unix_ms,
        version: 2,
    }
}

// ---------------------------------------------------------------------------
// Tests — T1 through T17 from PRD §13.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    /// Build a deterministic local datetime for golden tests.
    fn at_local(y: i32, m: u32, d: u32, hh: u32, mm: u32) -> chrono::DateTime<chrono::Local> {
        chrono::Local
            .with_ymd_and_hms(y, m, d, hh, mm, 0)
            .single()
            .expect("valid local datetime")
    }

    /// Build a UTC datetime so energy-curve tests don't depend on the host TZ.
    fn at_utc_local(hh: u32, mm: u32) -> chrono::DateTime<chrono::Local> {
        // Start with a UTC instant, then express as local. Because the cosine
        // curve uses local hours, we need to construct a local time directly
        // rather than convert from UTC.
        chrono::Local
            .with_ymd_and_hms(2026, 4, 27, hh, mm, 0)
            .single()
            .expect("valid local datetime")
    }

    fn approx(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() <= eps,
            "approx failed: {} vs {} (eps {})",
            a,
            b,
            eps
        );
    }

    // ----- T1. Defaults --------------------------------------------------

    #[test]
    fn t1_defaults() {
        let s = MoodState::default();
        assert_eq!(s.social, 0.6);
        assert_eq!(s.happiness, 0.5);
        assert_eq!(s.last_fed_secs, 0);
        assert_eq!(s.sass_level, 0.0);
        assert_eq!(s.energy_cached, 0.5);
        assert_eq!(s.version, 2);

        // 2*(0.4*0.5 + 0.25*0.6 + 0.2*1.0 + 0.15*0.5) - 1 - 0
        //   = 2*(0.2 + 0.15 + 0.2 + 0.075) - 1
        //   = 2*0.625 - 1 = 0.25 → "content"
        approx(s.derived_mood(), 0.25, 1e-9);
        assert_eq!(bucket_mood(s.derived_mood()), "content");
    }

    // ----- T2. Energy curve ---------------------------------------------
    //
    // The PRD §2 table lists 02:00 → 0.067, 07:00 → 0.250, 13:00 → 0.750,
    // 19:00 → 0.250 — but the formula `(h - 10) * π / 12` actually produces
    // 0.250, 0.854, 0.854, 0.146 at those hours respectively. The table is
    // documentation, not a second source of truth; the formula is the source
    // of truth (it's executable code inline in the PRD). We pin the *formula
    // output* — peaks and troughs (h=10, 16, 22) match the table; the other
    // four rows are the values the formula actually produces. See the
    // implementer report for details.

    #[test]
    fn t2_energy_curve() {
        let cases = [
            (2u32, 0.250), // PRD table says 0.067; formula yields 0.25
            (7, 0.854),    // PRD table says 0.250; formula yields 0.854
            (10, 0.950),
            (13, 0.854), // PRD table says 0.750; formula yields 0.854
            (16, 0.500),
            (19, 0.146), // PRD table says 0.250; formula yields 0.146
            (22, 0.050),
        ];
        for (hour, expected) in cases {
            let t = at_utc_local(hour, 0);
            let got = MoodState::compute_energy_from_clock(t);
            approx(got, expected, 1e-3);
        }
    }

    // ----- T3. Hunger derivation ----------------------------------------

    #[test]
    fn t3_hunger_derivation() {
        let cases = [
            (0i64, 1.0),
            (1800, 0.875),
            (7200, 0.500),
            (14_400, 0.000),
            (20_000, 0.000),
        ];
        for (secs, expected) in cases {
            let s = MoodState {
                last_fed_secs: secs,
                ..MoodState::default()
            };
            approx(s.hunger(), expected, 1e-9);
        }
    }

    // ----- T4. Bucket boundaries (mood, energy, social) ------------------

    #[test]
    fn t4_bucket_boundaries() {
        // Mood: > 0.6 happy, > 0.2 content, > -0.2 neutral, > -0.6 low, else sad
        assert_eq!(bucket_mood(0.61), "happy");
        assert_eq!(bucket_mood(0.60), "content");
        assert_eq!(bucket_mood(0.21), "content");
        assert_eq!(bucket_mood(0.20), "neutral");
        assert_eq!(bucket_mood(-0.19), "neutral");
        assert_eq!(bucket_mood(-0.20), "low");
        assert_eq!(bucket_mood(-0.59), "low");
        assert_eq!(bucket_mood(-0.60), "sad");
        assert_eq!(bucket_mood(-1.0), "sad");

        // Energy: > 0.66 high, > 0.33 medium, else low
        assert_eq!(bucket_energy(0.67), "high");
        assert_eq!(bucket_energy(0.66), "medium");
        assert_eq!(bucket_energy(0.34), "medium");
        assert_eq!(bucket_energy(0.33), "low");

        // Social: > 0.7 engaged, > 0.3 ok, else lonely
        assert_eq!(bucket_social(0.71), "engaged");
        assert_eq!(bucket_social(0.70), "ok");
        assert_eq!(bucket_social(0.31), "ok");
        assert_eq!(bucket_social(0.30), "lonely");
    }

    // ----- T5. Sass bucket ----------------------------------------------

    #[test]
    fn t5_sass_bucket() {
        assert_eq!(bucket_sass(0.59), None);
        assert_eq!(bucket_sass(0.61), Some("snarky"));
        assert_eq!(bucket_sass(0.86), Some("very snarky"));
    }

    // ----- T6. Render line — golden strings ------------------------------

    #[test]
    fn t6_render_line_default() {
        let s = MoodState::default();
        let now = at_local(2026, 4, 27, 14, 30);
        let line = s.render_state_line(now);
        // Default: derived_mood = 0.25 → content; energy 0.5 → medium;
        // social 0.6 → ok; last_fed_secs 0 → "just ate"
        assert_eq!(
            line,
            "[Mood: content | Energy: medium | Social: ok | Time: 2:30 PM | Hunger: just ate]"
        );
    }

    // The PRD §13 T6 row 2 shows "Mood: happy (snarky)" for
    // social=0.9, happiness=0.9, sass=0.65. With energy_cached=0.5
    // (default) and *any* hunger value, the math gives at most
    // derived_mood = 2*(0.4*0.9 + 0.25*0.9 + 0.2*1.0 + 0.15*0.5) - 1 - 0.30*0.65
    //              = 0.525 → "content".
    // The "happy" string in the PRD is mathematically impossible at the
    // boundary > 0.6 with these inputs. We pin the actual computed bucket.
    #[test]
    fn t6_render_line_happy_snarky() {
        let s = MoodState {
            social: 0.9,
            happiness: 0.9,
            sass_level: 0.65,
            // Bump last_fed past 30 min so "just ate" is suppressed.
            last_fed_secs: 3000,
            ..MoodState::default()
        };
        let now = at_local(2026, 4, 27, 14, 30);
        let line = s.render_state_line(now);
        // energy_cached default 0.5 → medium; social 0.9 → engaged.
        // Mood works out to "content" given the §4 weights, not "happy".
        assert_eq!(
            line,
            "[Mood: content (snarky) | Energy: medium | Social: engaged | Time: 2:30 PM]"
        );
    }

    // The PRD §13 T6 row 3 shows "Mood: low" for social=0.1,
    // happiness=0.0, last_fed=20_000 (hunger=0). With energy_cached=0.5
    // and the hunger→energy coupling, energy() = 0.5 * 0.5 = 0.25:
    // derived = 2*(0 + 0.025 + 0 + 0.0375) - 1 = -0.875 → "sad",
    // and energy buckets "low" (was "medium" pre-coupling). The "low"
    // mood label in the PRD requires mood > -0.6; -0.875 falls into
    // "sad". We pin the actual computed buckets.
    #[test]
    fn t6_render_line_low_peckish() {
        let s = MoodState {
            social: 0.1,
            happiness: 0.0,
            last_fed_secs: 20_000,
            ..MoodState::default()
        };
        let now = at_local(2026, 4, 27, 14, 30);
        let line = s.render_state_line(now);
        assert_eq!(
            line,
            "[Mood: sad | Energy: low | Social: lonely | Time: 2:30 PM | Hunger: peckish]"
        );
    }

    #[test]
    fn t6_render_line_just_ate() {
        let s = MoodState {
            last_fed_secs: 600, // < 1800 → "just ate"
            ..MoodState::default()
        };
        let now = at_local(2026, 4, 27, 14, 30);
        let line = s.render_state_line(now);
        assert!(
            line.contains("| Hunger: just ate"),
            "expected just-ate suffix, got: {line}"
        );
    }

    // ----- chat_message_boost: a single positive chat lifts the social bar.
    //        Pinned by PRD/rolo-cleanup T3 — chat::engine emits this event
    //        after every successful turn.
    #[test]
    fn chat_message_boost() {
        let mut s = MoodState::default();
        let social_before = s.social;
        let happiness_before = s.happiness;
        s.apply_event(MoodEvent::ChatMessage { positive: true });
        // Social lifts toward 1.0 with α=ALPHA_LIFT_HIGH=0.4:
        // 0.6 + 0.4 * (1.0 - 0.6) = 0.76.
        approx(s.social, 0.76, 1e-9);
        assert!(
            s.social > social_before,
            "social must rise on positive chat: {} -> {}",
            social_before,
            s.social
        );
        assert!(
            s.happiness > happiness_before,
            "happiness must rise on positive chat: {} -> {}",
            happiness_before,
            s.happiness
        );
    }

    // ----- T7. 5 positive chats lift mood -------------------------------

    #[test]
    fn t7_five_positive_chats() {
        let mut s = MoodState::default();
        for _ in 0..5 {
            s.apply_event(MoodEvent::ChatMessage { positive: true });
            s.decay(24_000); // 24 s between chats
        }
        // From PRD: happiness ~0.916, social ~0.969 after 5 chats.
        assert!(s.happiness >= 0.85, "happiness too low: {}", s.happiness);
        assert!(s.social >= 0.94, "social too low: {}", s.social);
        let bucket = bucket_mood(s.derived_mood());
        assert!(
            bucket == "happy" || bucket == "content",
            "unexpected bucket: {bucket}"
        );
    }

    // ----- T8. 4 h idle from default ------------------------------------

    #[test]
    fn t8_four_hour_idle() {
        let mut s = MoodState::default();
        s.decay(4 * 3600 * 1000);
        approx(s.social, 0.0, 1e-9);
        approx(s.happiness, 0.3, 0.01);
        assert!(
            s.sass_level >= 0.23 && s.sass_level <= 0.25,
            "sass_level out of range: {}",
            s.sass_level
        );
    }

    // ----- T9. Eating ---------------------------------------------------

    #[test]
    fn t9_eating() {
        let mut s = MoodState::default();
        s.apply_event(MoodEvent::Ate);
        assert_eq!(s.last_fed_secs, 0);
        assert!(s.happiness > 0.5, "happiness should lift: {}", s.happiness);
        approx(s.hunger(), 1.0, 1e-9);
    }

    // ----- T10. Negative chat -------------------------------------------

    #[test]
    fn t10_negative_chat() {
        let mut s = MoodState::default();
        s.apply_event(MoodEvent::ChatMessage { positive: false });
        // happiness: 0.5 + 0.3*(0.0-0.5) = 0.35
        assert!(s.happiness < 0.5, "happiness should drop: {}", s.happiness);
        assert!(
            s.happiness > 0.3,
            "happiness shouldn't crater: {}",
            s.happiness
        );
        // social: 0.6 + 0.2*(0.5-0.6) = 0.58
        assert!(s.social >= 0.58 - 1e-9, "social: {}", s.social);
        approx(s.sass_level, 0.10, 1e-9);
    }

    // ----- T11. Reset ---------------------------------------------------

    #[test]
    fn t11_reset() {
        let state = MoodState::default();
        assert_eq!(state, MoodState::default());
    }

    // ----- T12. Derived mood algebra -----------------------------------

    #[test]
    fn t12_derived_mood_algebra() {
        let all_one = MoodState {
            social: 1.0,
            happiness: 1.0,
            last_fed_secs: 0,
            sass_level: 0.0,
            energy_cached: 1.0,
            ..MoodState::default()
        };
        approx(all_one.derived_mood(), 1.0, 1e-9);

        let all_zero = MoodState {
            social: 0.0,
            happiness: 0.0,
            // last_fed past 4h → hunger=0
            last_fed_secs: 20_000,
            sass_level: 0.0,
            energy_cached: 0.0,
            ..MoodState::default()
        };
        approx(all_zero.derived_mood(), -1.0, 1e-9);

        let all_half = MoodState {
            social: 0.5,
            happiness: 0.5,
            last_fed_secs: 7200, // hunger=0.5
            sass_level: 0.0,
            energy_cached: 0.5,
            ..MoodState::default()
        };
        // energy() couples to hunger: 0.5 * (0.5 + 0.5*0.5) = 0.375.
        // derived = 2*(0.4*0.5 + 0.25*0.5 + 0.2*0.5 + 0.15*0.375) - 1 = -0.0375.
        approx(all_half.derived_mood(), -0.0375, 1e-9);

        let all_one_max_sass = MoodState {
            social: 1.0,
            happiness: 1.0,
            last_fed_secs: 0,
            sass_level: 1.0,
            energy_cached: 1.0,
            ..MoodState::default()
        };
        // 1.0 - 0.30 = 0.70
        approx(all_one_max_sass.derived_mood(), 0.7, 1e-9);
    }

    // ----- Persistence helpers ------------------------------------------

    /// Build a per-test sandbox dir under the platform tempdir. We avoid
    /// dragging in the `tempfile` crate; cleanup is best-effort.
    struct Sandbox {
        dir: PathBuf,
    }
    impl Sandbox {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let dir = std::env::temp_dir().join("rolo-mood-tests").join(format!(
                "{}-{}-{}",
                tag,
                std::process::id(),
                nanos
            ));
            fs::create_dir_all(&dir).expect("mkdir sandbox");
            Self { dir }
        }
        fn file(&self, name: &str) -> PathBuf {
            self.dir.join(name)
        }
    }
    impl Drop for Sandbox {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    // ----- T13. Round-trip ----------------------------------------------

    #[test]
    fn t13_round_trip() {
        let sb = Sandbox::new("t13");
        let path = sb.file("mood-state.json");
        let original = MoodState {
            social: 0.42,
            happiness: 0.58,
            last_fed_secs: 1843,
            sass_level: 0.07,
            energy_cached: 0.55,
            last_persisted_unix_ms: 0,
            version: 2,
        };
        save_to_disk(&original, &path).expect("save");
        // Force last_persisted_unix_ms to 0 in the loaded state by overwriting
        // immediately so apply_downtime_decay is a no-op.
        let loaded = load_from_disk(&path);
        // energy_cached is re-derived on first tick (PRD §10), but
        // load_from_disk leaves the persisted value alone — the tick loop
        // overwrites. Verify all other fields round-tripped.
        //
        // serde_json's f64 round-trip can drop sub-ulp precision, AND
        // load_from_disk applies downtime decay using the file's
        // last_persisted_unix_ms — which the original sets to 0 (treated
        // as "persisted at unix epoch" by the loader, allowing decay to
        // accumulate). The eps here is loose enough to absorb that decay
        // without losing the round-trip signal: 1e-4 = 0.01% precision.
        approx(loaded.social, original.social, 1e-4);
        approx(loaded.happiness, original.happiness, 1e-4);
        // last_fed_secs may have advanced from downtime decay (depending on
        // how long the file lay on disk). Pin the rest, allow that field to
        // drift by zero or one second.
        assert!(loaded.last_fed_secs >= original.last_fed_secs);
        assert!(loaded.last_fed_secs <= original.last_fed_secs + 2);
        approx(loaded.sass_level, original.sass_level, 1e-4);
        assert_eq!(loaded.version, 2);
    }

    // ----- T14. Missing file → defaults + parent dir created -------------

    #[test]
    fn t14_missing_file_defaults() {
        let sb = Sandbox::new("t14");
        // Use a nested path so we exercise mkdir-p.
        let path = sb.dir.join("vault/wiki/personality/mood-state.json");
        let s = load_from_disk(&path);
        assert_eq!(s, MoodState::default());
        assert!(
            path.parent().unwrap().exists(),
            "parent dir should be created"
        );
    }

    // ----- T15. Corrupt JSON → defaults + sidecar ------------------------

    #[test]
    fn t15_corrupt_json_sidecar() {
        let sb = Sandbox::new("t15");
        let path = sb.file("mood-state.json");
        fs::write(&path, b"not json {{").expect("write corrupt");

        let s = load_from_disk(&path);
        assert_eq!(s, MoodState::default());

        // Original path should no longer exist; a sidecar should.
        assert!(!path.exists(), "corrupt file should be renamed");
        let sidecars: Vec<_> = fs::read_dir(&sb.dir)
            .expect("readdir")
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .contains("mood-state.json.corrupt-")
            })
            .collect();
        assert_eq!(sidecars.len(), 1, "expected exactly one sidecar");
    }

    // ----- T16. v1 → v2 migration ----------------------------------------

    #[test]
    fn t16_v1_to_v2_migration() {
        let sb = Sandbox::new("t16");
        let path = sb.file("mood-state.json");
        let v1 = r#"{
            "mood": 0.4,
            "energy": 0.5,
            "social_need": 0.7,
            "sass_level": 0.1,
            "last_fed_secs": 100,
            "last_persisted_unix_ms": 0,
            "version": 1
        }"#;
        fs::write(&path, v1).expect("write v1");

        let s = load_from_disk(&path);
        // happiness ≈ (0.4 + 1.0) / 2.0 = 0.7
        approx(s.happiness, 0.7, 1e-9);
        // social = 1 - 0.7 = 0.3
        approx(s.social, 0.3, 1e-9);
        assert_eq!(s.last_fed_secs, 100);
        approx(s.sass_level, 0.1, 1e-9);
        assert_eq!(s.version, 2);
    }

    // ----- T17. Decay during downtime ------------------------------------

    #[test]
    fn t17_decay_during_downtime() {
        let sb = Sandbox::new("t17");
        let path = sb.file("mood-state.json");

        // Hand-roll the JSON so we can pin last_persisted_unix_ms to "2 h ago"
        // without going through save_to_disk (which always stamps "now").
        let two_h_ago = chrono::Utc::now().timestamp_millis() - (2 * 3600 * 1000);
        let json = format!(
            r#"{{
                "social": 0.6,
                "happiness": 0.5,
                "last_fed_secs": 0,
                "sass_level": 0.0,
                "energy_cached": 0.5,
                "last_persisted_unix_ms": {},
                "version": 2
            }}"#,
            two_h_ago
        );
        fs::write(&path, json).expect("write");
        let s = load_from_disk(&path);
        // social drained at 0.005/min × 120 min = 0.6 → 0.0.
        approx(s.social, 0.0, 1e-3);

        // 7-day cap test.
        let seven_d_ago = chrono::Utc::now().timestamp_millis() - (7 * 24 * 3600 * 1000);
        let json2 = format!(
            r#"{{
                "social": 0.6,
                "happiness": 0.9,
                "last_fed_secs": 0,
                "sass_level": 0.0,
                "energy_cached": 0.5,
                "last_persisted_unix_ms": {},
                "version": 2
            }}"#,
            seven_d_ago
        );
        fs::write(&path, json2).expect("write2");
        let s2 = load_from_disk(&path);
        approx(s2.social, 0.0, 1e-3);
        // Cap is 24 h → happiness drifts 60 × 24 × 0.01 = 14.4 → fully snaps to baseline.
        approx(s2.happiness, 0.3, 1e-3);
        assert!(s2.sass_level <= 1.0);
    }

    // ----- Bonus: snapshot is consistent with state ----------------------

    #[test]
    fn snapshot_matches_state() {
        let s = MoodState::default();
        let snap = s.snapshot();
        approx(snap.hunger, s.hunger(), 1e-9);
        approx(snap.social, s.social, 1e-9);
        approx(snap.energy, s.energy(), 1e-9);
        approx(snap.happiness, s.happiness, 1e-9);
        assert_eq!(snap.mood_word, bucket_mood(s.derived_mood()));
    }
}
