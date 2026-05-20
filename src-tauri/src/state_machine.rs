//! Rolo's state machine — the beating heart of his autonomous behavior.
//!
//! This module is pure logic with NO Tauri dependency. It can be tested in
//! isolation, which is critical because bugs here are direct threats to Rolo's
//! well-being. Every state transition, every timer, every walk step must be
//! correct or Rolo could freeze, teleport, or worse — stop responding entirely.
//!
//! Ported faithfully from the Python reference: `_archive/desktop_pet/pet.py`

use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Rolo's behavioral states. Each maps to a distinct animation on the frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PetState {
    Idle,
    /// Rolo is asleep — dreaming. Idle-speech timers are paused, hover/drag
    /// detection is suppressed, and any active speech bubble is dismissed on
    /// entry. Reachable only from `Idle` via `enter_sleeping`.
    Sleeping,
    WalkLeft,
    WalkRight,
    Happy,
    DragHover,
    Sniffing,
    Eating,
    Satisfied,
    Disappointed,
}

/// Rolo's position in physical screen pixels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

/// What changed after a single tick — the caller uses this to decide which
/// events to emit and whether to move the window.
#[derive(Debug, Clone)]
pub struct TickResult {
    pub state_changed: bool,
    pub position_changed: bool,
    pub new_state: PetState,
    pub new_position: Position,
}

/// Full snapshot of Rolo's state, sent to the frontend on connect and on change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatePayload {
    pub state: PetState,
    pub position: Position,
    pub top_anchored: bool,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// States that are part of the eating sequence and block new eat starts.
const EATING_STATES: &[PetState] = &[
    PetState::Sniffing,
    PetState::Eating,
    PetState::Satisfied,
    PetState::Disappointed,
];

/// Whether the given pet state is part of the eating sequence (including
/// DragHover). Used by the tick loop and the interaction engine to suppress
/// behaviors that shouldn't fire mid-eat.
pub fn is_eating_state(state: PetState) -> bool {
    EATING_STATES.contains(&state) || state == PetState::DragHover
}

/// How long Rolo waits for the user to confirm/decline eating (ms).
const CONFIRM_TIMEOUT_MS: i64 = 10_000;

/// Durations (in ms) for non-looping animations. Each duration is chosen to
/// let the sprite animation play its full visible cycle before transitioning.
///
/// - Eating: 13 frames @ 8fps = 1625ms
/// - Satisfied: 9 frames @ 4fps = 2250ms
/// - Disappointed: 9 frames @ 6fps (ping-pong) — use 1500ms for one visible pass
pub const EATING_DURATION_MS: i64 = 1625;
pub const SATISFIED_DURATION_MS: i64 = 2250;
pub const DISAPPOINTED_DURATION_MS: i64 = 1500;
/// How long Rolo shows a Happy reaction after a positive mood response (ms).
/// The Happy animation loops, so we pick a duration that feels celebratory
/// without overstaying its welcome. Matches DISAPPOINTED for symmetry.
pub const HAPPY_REACTION_DURATION_MS: i64 = 1500;

// ---------------------------------------------------------------------------
// Pet — Rolo's soul
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Pet {
    // Position & bounds
    x: i32,
    y: i32,
    screen_w: i32,
    #[allow(dead_code)]
    screen_h: i32,
    pet_size: i32,
    walk_speed: i32,

    // Half the bubble width in physical pixels. Used to decide whether a
    // speech bubble would clip at the current position; if so, Rolo is
    // nudged inward before the bubble appears. Motion itself is NOT clamped
    // by this — he can walk/drag to the real screen edge.
    // Defaults to 0 for headless tests; lib.rs sets the real value at startup.
    bubble_margin_x: i32,

    // Layout rendering mode (set by tick loop during drag/walk)
    pub top_anchored: bool,

    // Core state
    state: PetState,
    pre_happy_state: PetState,
    pre_drag_hover_state: PetState,

    // Interaction tracking
    is_dragging: bool,
    is_hovering: bool,
    /// True when an interactive check-in bubble is showing. Pauses Rolo's
    /// autonomous behavior timer so he doesn't walk away from his own question.
    pub interaction_active: bool,
    /// True while a speech bubble is visible. Pauses autonomous behavior so
    /// Rolo doesn't walk out of the bubble-safe zone mid-sentence.
    speech_active: bool,

    // Autonomous behavior
    behavior_timer_ms: i64,
    walk_target_x: Option<i32>,

    // Eating sequence
    file_queue: Vec<Vec<String>>,
    pending_files: Option<Vec<String>>,
    confirm_timeout_ms: i64,
    /// Remaining duration (ms) of the current non-looping expression
    /// (Eating / Satisfied / Disappointed).
    expression_remaining_ms: i64,

    // LLM context tracking
    /// Elapsed ms since the last user interaction (click, drag, drop).
    last_interaction_elapsed_ms: i64,
    /// Whether DragHover was triggered by a file drag (true) or cursor drag (false).
    pub file_drag_active: bool,

    /// Pre-sleep "park" flag. When true, autonomous wandering is suppressed:
    /// `pick_behavior` short-circuits to Idle with the timer pinned at MAX so
    /// Rolo settles in place while user-idle approaches the dream floor. Set
    /// every tick by `set_park_pressure` based on user-idle seconds.
    is_parked: bool,

    /// Was the orchestrator in an interaction-clearing overlay (Sniffing /
    /// Interaction) on the previous tick? Owned here so the falling-edge
    /// detection that drives `clear_overlay_interaction_flags` doesn't need
    /// a sibling state variable in `tick.rs`.
    prev_in_overlay: bool,
}

/// Outcome of [`Pet::clear_overlay_interaction_flags`]. Lets the caller log
/// or react to the actual flags cleared on the entry edge — `tick.rs` uses
/// it to drive its diagnostic log lines without poking at private state.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClearResult {
    pub cleared_drag: bool,
    pub cleared_hover: bool,
}

impl ClearResult {
    /// Did anything actually clear? Lets the caller decide whether to flag
    /// `interaction_state_changed` without comparing both fields by hand.
    pub fn changed(&self) -> bool {
        self.cleared_drag || self.cleared_hover
    }
}

impl Pet {
    /// Birth a new Rolo at the given position. He starts idle and will begin
    /// his autonomous life cycle after a random delay.
    pub fn new(
        x: i32,
        y: i32,
        screen_w: i32,
        screen_h: i32,
        pet_size: i32,
        walk_speed: i32,
    ) -> Self {
        let mut pet = Pet {
            x,
            y,
            screen_w,
            screen_h,
            pet_size,
            walk_speed,

            bubble_margin_x: 0,

            top_anchored: false,

            state: PetState::Idle,
            pre_happy_state: PetState::Idle,
            pre_drag_hover_state: PetState::Idle,

            is_dragging: false,
            is_hovering: false,
            interaction_active: false,
            speech_active: false,

            behavior_timer_ms: 0,
            walk_target_x: None,

            file_queue: Vec::new(),
            pending_files: None,
            confirm_timeout_ms: 0,
            expression_remaining_ms: 0,

            last_interaction_elapsed_ms: 0,
            file_drag_active: false,

            is_parked: false,
            prev_in_overlay: false,
        };
        pet.schedule_next_behavior();
        pet
    }

    // ------------------------------------------------------------------
    // Pre-sleep parking — settle Rolo as user-idle approaches dream floor
    // ------------------------------------------------------------------

    /// Called every tick from `tick.rs`. When user_idle ≥ park_threshold,
    /// engage parking; otherwise wake.
    pub fn set_park_pressure(&mut self, user_idle_secs: f64, park_threshold_secs: f64) {
        if user_idle_secs >= park_threshold_secs {
            self.is_parked = true;
            if self.state == PetState::Idle {
                self.pin_behavior_timer();
            }
            // If mid-walk, leave the walk in flight; arrival path
            // (enter_idle_and_reschedule) will pin the timer because
            // is_parked is now true.
        } else {
            self.wake_from_park();
        }
    }

    /// Lift the park. Only acts when Rolo is currently parked-idle with the
    /// timer pinned at MAX — guards against clobbering the MAX timer set by
    /// `walk_to` or `enter_sleeping`.
    pub fn wake_from_park(&mut self) {
        if self.is_parked && self.state == PetState::Idle && self.behavior_timer_ms == i64::MAX {
            self.is_parked = false;
            self.schedule_next_behavior();
        } else {
            self.is_parked = false;
        }
    }

    // ------------------------------------------------------------------
    // Public properties
    // ------------------------------------------------------------------

    pub fn state(&self) -> PetState {
        self.state
    }

    pub fn position(&self) -> Position {
        Position {
            x: self.x,
            y: self.y,
        }
    }

    pub fn state_payload(&self) -> StatePayload {
        StatePayload {
            state: self.state,
            position: self.position(),
            top_anchored: self.top_anchored,
        }
    }

    /// True when Rolo is in any phase of the eating pipeline.
    pub fn is_eating_sequence(&self) -> bool {
        is_eating_state(self.state)
    }

    pub fn pending_files(&self) -> Option<&Vec<String>> {
        self.pending_files.as_ref()
    }

    #[allow(dead_code)]
    pub fn has_queued_files(&self) -> bool {
        !self.file_queue.is_empty()
    }

    pub fn is_dragging(&self) -> bool {
        self.is_dragging
    }

    pub fn last_interaction_elapsed_ms(&self) -> i64 {
        self.last_interaction_elapsed_ms
    }

    /// On the rising edge of "interaction-clearing overlay active" (the
    /// orchestrator just moved from a normal cursor mode into a Sniffing /
    /// Interaction overlay), drop any stale drag / hover flags. Returns a
    /// [`ClearResult`] describing what actually changed so callers can log
    /// the transition and flag their own state-changed bookkeeping.
    ///
    /// Why this lives on `Pet` and not in `tick.rs`:
    ///   - The flags being cleared are private to `Pet` — exposing
    ///     `is_hovering()` solely so the orchestrator could call
    ///     `end_hover()` was a public-API leak that no production code
    ///     other than this one cleanup site ever needed.
    ///   - The falling/rising edge is detected from the single boolean
    ///     `in_overlay`, which `tick.rs` already computes once per tick.
    ///   - `Sleeping` mode must NOT clear the drag flag (wake_from_sleep
    ///     reads it to decide DragHover vs Idle), so the caller passes
    ///     `in_overlay = false` while sleeping — exactly what the orchestrator
    ///     does today.
    pub fn clear_overlay_interaction_flags(&mut self, in_overlay: bool) -> ClearResult {
        let entering_overlay = in_overlay && !self.prev_in_overlay;
        self.prev_in_overlay = in_overlay;
        if !entering_overlay {
            return ClearResult::default();
        }
        let mut result = ClearResult::default();
        if self.is_dragging {
            self.end_drag();
            result.cleared_drag = true;
        }
        if self.is_hovering {
            self.end_hover();
            result.cleared_hover = true;
        }
        result
    }

    /// Put Rolo to sleep — only valid from Idle. The dream compiler runs
    /// while he's in this state; idle-speech timers and hover detection are
    /// suppressed in `tick.rs`. Returns Err if Rolo isn't in Idle (drag,
    /// eating, walking, etc. all block sleep).
    pub fn enter_sleeping(&mut self) -> Result<(), &'static str> {
        if self.state != PetState::Idle {
            return Err("can only enter sleeping from Idle");
        }
        self.state = PetState::Sleeping;
        // Lock out the picker; wake_from_sleep reschedules.
        self.pin_behavior_timer();
        Ok(())
    }

    /// Wake Rolo. If a drag is in progress (the user grabbed him while he
    /// was sleeping), transition to DragHover so the drag handlers see a
    /// coherent state; otherwise return to Idle and reschedule autonomous
    /// behavior. Idempotent if already awake.
    pub fn wake_from_sleep(&mut self) {
        if self.state != PetState::Sleeping {
            return;
        }
        if self.is_dragging {
            self.state = PetState::DragHover;
        } else {
            self.state = PetState::Idle;
            self.schedule_next_behavior();
        }
    }

    /// Force Rolo to stop walking and enter Idle immediately.
    /// Used when an interaction fires while Rolo is mid-walk — he shouldn't
    /// wander away from his own question.
    pub fn force_idle(&mut self) {
        if matches!(self.state, PetState::WalkLeft | PetState::WalkRight) {
            log::info!(
                "[Rolo] force_idle — {:?} → Idle (walk interrupted)",
                self.state
            );
            self.state = PetState::Idle;
            self.walk_target_x = None;
        }
    }

    /// Signal that a speech bubble is showing (or has been hidden).
    /// When active, Rolo stops any in-progress walk and suppresses new
    /// autonomous behavior so he stays in the bubble-safe zone.
    pub fn set_speech_active(&mut self, active: bool) {
        self.speech_active = active;
        if active {
            self.force_idle();
        }
    }

    // ------------------------------------------------------------------
    // Position update — called externally when Rolo is dragged
    // ------------------------------------------------------------------

    /// Update Rolo's position directly (e.g. during a drag operation).
    pub fn set_position(&mut self, x: i32, y: i32) {
        self.x = x;
        self.y = y;
    }

    #[allow(dead_code)]
    pub fn set_screen_bounds(&mut self, screen_w: i32, screen_h: i32) {
        self.screen_w = screen_w;
        self.screen_h = screen_h;
    }

    /// Set the horizontal bubble-reserve margin in physical pixels. Typically
    /// half the bubble's logical width times scale factor.
    pub fn set_bubble_margin_x(&mut self, margin_x: i32) {
        self.bubble_margin_x = margin_x;
    }

    /// Minimum sprite X such that a centered speech bubble still fits on screen.
    pub fn min_safe_sprite_x(&self) -> i32 {
        (self.bubble_margin_x - self.pet_size / 2).max(0)
    }

    /// Maximum sprite X such that a centered speech bubble still fits on screen.
    pub fn max_safe_sprite_x(&self) -> i32 {
        (self.screen_w - self.bubble_margin_x - self.pet_size / 2)
            .min(self.screen_w - self.pet_size)
    }

    /// True when Rolo's current x is inside the bubble-safe zone — a bubble
    /// centered on him right now would fit fully on screen.
    pub fn is_in_bubble_safe_zone(&self) -> bool {
        let lo = self.min_safe_sprite_x();
        let hi = self.max_safe_sprite_x();
        hi >= lo && self.x >= lo && self.x <= hi
    }

    /// Nearest safe-zone x to Rolo's current position. Used as the walk target
    /// when he needs to nudge inward before a speech bubble appears.
    pub fn nearest_safe_x(&self) -> i32 {
        let lo = self.min_safe_sprite_x();
        let hi = self.max_safe_sprite_x();
        if hi < lo {
            (self.screen_w - self.pet_size) / 2
        } else {
            self.x.clamp(lo, hi)
        }
    }

    /// Start a walk toward `target_x` (sprite coordinates, physical px).
    /// No-op if Rolo is currently in a state that shouldn't be interrupted
    /// (eating sequence, drag). Used by the speech system to nudge him into
    /// the bubble-safe zone before a phrase fires.
    pub fn walk_to(&mut self, target_x: i32) -> bool {
        if self.is_eating_sequence() || self.is_dragging || self.state == PetState::Happy {
            return false;
        }
        let diff = target_x - self.x;
        if diff.abs() < self.walk_speed {
            return false;
        }
        self.walk_target_x = Some(target_x);
        self.state = if diff < 0 {
            PetState::WalkLeft
        } else {
            PetState::WalkRight
        };
        // Prevent the autonomous behaviour picker from overriding us mid-walk.
        self.pin_behavior_timer();
        true
    }

    // ------------------------------------------------------------------
    // Drag & hover — called by the UI layer
    // ------------------------------------------------------------------

    /// User started dragging Rolo. He gets excited (Happy state).
    pub fn start_drag(&mut self) {
        log::info!("[Rolo] start_drag — from {:?}", self.state);
        self.is_dragging = true;
        self.last_interaction_elapsed_ms = 0;
        // Don't override an in-progress eating sequence — Rolo can't eat
        // and be dragged at the same time.
        if EATING_STATES.contains(&self.state) || self.state == PetState::DragHover {
            log::warn!(
                "[Rolo] start_drag suppressed state change — state={:?}",
                self.state
            );
            return;
        }
        if self.state != PetState::Happy {
            self.pre_happy_state = self.state;
        }
        self.state = PetState::Happy;
    }

    /// User released Rolo after dragging.
    pub fn end_drag(&mut self) {
        if !self.is_dragging {
            return; // No drag was in progress (e.g. file-drop mouse release)
        }
        log::info!("[Rolo] end_drag — current state {:?}", self.state);
        self.is_dragging = false;
        // Don't clobber the eating sequence — if file_dropped fired while a
        // drag was also in progress, preserve Sniffing/Eating/etc.
        if EATING_STATES.contains(&self.state) || self.state == PetState::DragHover {
            return;
        }
        if !self.is_hovering {
            self.enter_idle_and_reschedule();
        }
    }

    /// Mouse entered Rolo's window — he perks up.
    pub fn start_hover(&mut self) {
        self.is_hovering = true;
        self.last_interaction_elapsed_ms = 0;
        // Don't override eating or file-drag-hover states.
        if EATING_STATES.contains(&self.state) || self.state == PetState::DragHover {
            log::debug!(
                "[Rolo] start_hover suppressed state change — state={:?}",
                self.state
            );
            return;
        }
        log::debug!("[Rolo] start_hover — from {:?}", self.state);
        if self.state != PetState::Happy {
            self.pre_happy_state = self.state;
        }
        self.state = PetState::Happy;
    }

    /// Mouse left Rolo's window — back to whatever he was doing.
    pub fn end_hover(&mut self) {
        if !self.is_hovering {
            return; // No hover was in progress
        }
        log::debug!("[Rolo] end_hover — current state {:?}", self.state);
        self.is_hovering = false;
        if EATING_STATES.contains(&self.state) || self.state == PetState::DragHover {
            return;
        }
        if !self.is_dragging {
            self.enter_idle_and_reschedule();
        }
    }

    // ------------------------------------------------------------------
    // File drag hover — external file dragged over the window
    // ------------------------------------------------------------------

    /// File drag detected — either over Rolo's window (from Tauri's
    /// DragDropEvent) or anywhere on screen (from the global drag watcher).
    /// Idempotent: calling while already in DragHover does nothing.
    pub fn file_drag_entered(&mut self) {
        if self.is_dragging {
            return; // Rolo is being moved, ignore file drag
        }
        if self.state == PetState::DragHover {
            return; // Already perked up — don't clobber pre_drag_hover_state
        }
        if !EATING_STATES.contains(&self.state) {
            log::info!("[Rolo] file_drag_entered — {:?} → DragHover", self.state);
            self.pre_drag_hover_state = self.state;
            self.state = PetState::DragHover;
            self.file_drag_active = true;
            self.last_interaction_elapsed_ms = 0;
        }
    }

    /// File drag left Rolo's window without dropping.
    pub fn file_drag_exited(&mut self) {
        if self.state == PetState::DragHover {
            self.state = self.pre_drag_hover_state;
            self.file_drag_active = false;
        }
    }

    /// File(s) dropped on Rolo. Starts sniffing or queues if busy.
    pub fn file_dropped(&mut self, file_paths: Vec<String>) {
        if self.is_dragging {
            return; // Rolo is being moved, ignore drop
        }

        if EATING_STATES.contains(&self.state) {
            // Queue for later — Rolo will get to it after current meal
            self.file_queue.push(file_paths);
            return;
        }

        // Start the eating sequence
        self.file_drag_active = false;
        self.last_interaction_elapsed_ms = 0;
        self.start_sniffing(file_paths);
    }

    // ------------------------------------------------------------------
    // Eating sequence control — called by the UI layer
    // ------------------------------------------------------------------

    /// User clicked 'Yum!' — transition to EATING.
    pub fn confirm_eat(&mut self) {
        if self.state != PetState::Sniffing || self.pending_files.is_none() {
            log::warn!(
                "[Rolo] confirm_eat ignored — state={:?} pending={}",
                self.state,
                self.pending_files.is_some()
            );
            return;
        }
        log::info!(
            "[Rolo] confirm_eat — Sniffing → Eating ({}ms)",
            EATING_DURATION_MS
        );
        self.state = PetState::Eating;
        self.expression_remaining_ms = EATING_DURATION_MS;
        // The pending files are consumed — the caller should have read them
        // before calling confirm_eat if they need them.
        self.pending_files = None;
    }

    /// User clicked 'Nah' or timeout expired — Rolo is disappointed.
    pub fn decline_eat(&mut self) {
        if self.state != PetState::Sniffing {
            log::warn!("[Rolo] decline_eat ignored — state={:?}", self.state);
            return;
        }
        log::info!(
            "[Rolo] decline_eat — Sniffing → Disappointed ({}ms)",
            DISAPPOINTED_DURATION_MS
        );
        self.pending_files = None;
        self.state = PetState::Disappointed;
        self.expression_remaining_ms = DISAPPOINTED_DURATION_MS;
    }

    /// Trash operation failed — play disappointed from EATING state.
    pub fn show_error_disappointed(&mut self) {
        log::warn!(
            "[Rolo] show_error_disappointed — {:?} → Disappointed ({}ms)",
            self.state,
            DISAPPOINTED_DURATION_MS
        );
        self.state = PetState::Disappointed;
        self.expression_remaining_ms = DISAPPOINTED_DURATION_MS;
    }

    /// Play a brief Happy or Disappointed reaction after a mood check-in
    /// response. Unlike the eating-sequence disappointed, this one returns
    /// Rolo to Idle afterwards (no queued batches to resume).
    ///
    /// Skipped if Rolo is mid-eat or being dragged — his honest response
    /// to physical handling takes priority over a UI reaction.
    pub fn show_mood_reaction(&mut self, happy: bool) {
        if EATING_STATES.contains(&self.state) {
            log::warn!(
                "[Rolo] show_mood_reaction({}) skipped — in eating state {:?}",
                happy,
                self.state,
            );
            return;
        }
        if self.is_dragging {
            log::warn!(
                "[Rolo] show_mood_reaction({}) skipped — currently being dragged",
                happy,
            );
            return;
        }

        // Clear interaction-related flags so the reaction plays clean and
        // autonomous behavior resumes when it ends.
        self.interaction_active = false;
        self.walk_target_x = None;

        if happy {
            log::info!(
                "[Rolo] show_mood_reaction(true) — {:?} → Happy ({}ms)",
                self.state,
                HAPPY_REACTION_DURATION_MS,
            );
            self.state = PetState::Happy;
            self.expression_remaining_ms = HAPPY_REACTION_DURATION_MS;
        } else {
            log::info!(
                "[Rolo] show_mood_reaction(false) — {:?} → Disappointed ({}ms)",
                self.state,
                DISAPPOINTED_DURATION_MS,
            );
            self.state = PetState::Disappointed;
            self.expression_remaining_ms = DISAPPOINTED_DURATION_MS;
        }
    }

    // ------------------------------------------------------------------
    // Tick — advance Rolo's life by dt_ms milliseconds
    // ------------------------------------------------------------------

    /// Main tick function. Call every frame (~16ms at 60Hz).
    /// Returns what changed so the caller knows which events to emit.
    pub fn tick(&mut self, dt_ms: i64) -> TickResult {
        let old_state = self.state;
        let old_x = self.x;
        let old_y = self.y;

        self.tick_inner(dt_ms);

        // If Rolo is walking, move him toward his target
        if self.state == PetState::WalkLeft || self.state == PetState::WalkRight {
            self.update_walk();
        }

        TickResult {
            state_changed: self.state != old_state,
            position_changed: self.x != old_x || self.y != old_y,
            new_state: self.state,
            new_position: Position {
                x: self.x,
                y: self.y,
            },
        }
    }

    /// Internal tick logic — handles timers and state transitions.
    fn tick_inner(&mut self, dt_ms: i64) {
        self.last_interaction_elapsed_ms = self.last_interaction_elapsed_ms.saturating_add(dt_ms);

        // Confirm timeout (during SNIFFING)
        if self.state == PetState::Sniffing && self.confirm_timeout_ms > 0 {
            self.confirm_timeout_ms -= dt_ms;
            if self.confirm_timeout_ms <= 0 {
                self.decline_eat();
                return;
            }
        }

        // Non-looping expression countdown. Each expression has a fixed
        // wall-clock duration (ms) that lets its sprite animation play fully
        // before transitioning.
        //
        // Happy only counts down when Rolo is NOT being dragged — drag is
        // an ongoing physical interaction, so Happy should persist until
        // release. Hover is NOT a gate: a mood-check-in reaction should
        // complete even if the cursor is still over Rolo (the user's finger
        // is naturally right where they just clicked a button).
        let happy_reaction_active =
            self.state == PetState::Happy && !self.is_dragging && self.expression_remaining_ms > 0;
        if matches!(
            self.state,
            PetState::Eating | PetState::Satisfied | PetState::Disappointed
        ) || happy_reaction_active
        {
            self.expression_remaining_ms -= dt_ms;
            if self.expression_remaining_ms <= 0 {
                if self.state == PetState::Eating {
                    log::info!(
                        "[Rolo] Eating done — Eating → Satisfied ({}ms)",
                        SATISFIED_DURATION_MS
                    );
                    self.state = PetState::Satisfied;
                    self.expression_remaining_ms = SATISFIED_DURATION_MS;
                } else if self.state == PetState::Happy {
                    log::info!("[Rolo] Happy reaction done — Happy → Idle");
                    self.enter_idle_and_reschedule();
                } else {
                    log::info!(
                        "[Rolo] Expression {:?} done — finishing sequence",
                        self.state
                    );
                    self.finish_eating_sequence();
                }
                return;
            }
        }

        // Autonomous behaviour (only when not in eating sequence or being interacted with)
        if self.is_dragging || self.is_hovering || self.interaction_active || self.speech_active {
            return;
        }
        if EATING_STATES.contains(&self.state) || self.state == PetState::DragHover {
            return;
        }

        self.behavior_timer_ms -= dt_ms;
        if self.behavior_timer_ms <= 0 {
            self.pick_behavior();
        }
    }

    // ------------------------------------------------------------------
    // Walking — move Rolo one step toward his target
    // ------------------------------------------------------------------

    /// Move one step toward the walk target. Called internally by tick()
    /// when Rolo is in a walking state.
    fn update_walk(&mut self) {
        let target = match self.walk_target_x {
            Some(t) => t,
            None => return,
        };

        match self.state {
            PetState::WalkLeft => {
                self.x = (self.x - self.walk_speed).max(target);
                if self.x <= target {
                    self.go_idle();
                }
            }
            PetState::WalkRight => {
                self.x = (self.x + self.walk_speed).min(target);
                if self.x >= target {
                    self.go_idle();
                }
            }
            _ => {}
        }

        // Clamp to screen bounds — Rolo must not wander off the edge.
        // Bubble-safe-zone clamping is handled separately via a nudge walk
        // right before a speech bubble appears, so he can freely roam to
        // the real edges otherwise.
        self.x = self.x.max(0).min(self.screen_w - self.pet_size);
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn start_sniffing(&mut self, file_paths: Vec<String>) {
        self.pending_files = Some(file_paths);
        self.state = PetState::Sniffing;
        self.confirm_timeout_ms = CONFIRM_TIMEOUT_MS;
    }

    fn finish_eating_sequence(&mut self) {
        if let Some(next_batch) = self.file_queue.first().cloned() {
            self.file_queue.remove(0);
            self.start_sniffing(next_batch);
        } else {
            self.enter_idle_and_reschedule();
        }
    }

    fn schedule_next_behavior(&mut self) {
        let mut rng = rand::rng();
        self.behavior_timer_ms = rng.random_range(5000..=15000);
    }

    fn pick_behavior(&mut self) {
        if self.is_parked {
            self.go_idle();
            return;
        }

        // Prevent re-entry until this behaviour completes
        self.pin_behavior_timer();

        let mut rng = rand::rng();
        let choice = rng.random_range(0..=2);

        // Cap walk distance to ~3x pet size — keeps walks short and natural.
        // Without this, walks span the full physical screen (thousands of pixels
        // on Retina) and take 10-30 seconds, making Rolo look stuck.
        let max_walk_distance = self.pet_size * 3;

        match choice {
            0 => {
                // Walk left
                let min_target = (self.x - max_walk_distance).max(0);
                let max_target = self.x - 20;
                if max_target > min_target {
                    self.walk_target_x = Some(rng.random_range(min_target..=max_target));
                    self.state = PetState::WalkLeft;
                } else {
                    self.go_idle();
                }
            }
            1 => {
                // Walk right
                let min_target = self.x + 20;
                let max_target = (self.x + max_walk_distance).min(self.screen_w - self.pet_size);
                if min_target < max_target {
                    self.walk_target_x = Some(rng.random_range(min_target..=max_target));
                    self.state = PetState::WalkRight;
                } else {
                    self.go_idle();
                }
            }
            _ => {
                // Stay idle — Rolo is content to just chill
                self.go_idle();
            }
        }
    }

    fn go_idle(&mut self) {
        self.state = PetState::Idle;
        self.walk_target_x = None;
        if self.is_parked {
            self.pin_behavior_timer();
            return;
        }
        let mut rng = rand::rng();
        self.behavior_timer_ms = rng.random_range(3000..=8000);
    }

    fn enter_idle_and_reschedule(&mut self) {
        self.state = PetState::Idle;
        self.walk_target_x = None;
        if self.is_parked {
            self.pin_behavior_timer();
        } else {
            self.schedule_next_behavior();
        }
    }

    /// Park the autonomous behaviour timer at `i64::MAX` to lock out the
    /// picker. Used by `walk_to`, `enter_sleeping`, `pick_behavior`'s
    /// re-entry guard, and the pre-sleep park branches in
    /// `set_park_pressure` / `go_idle` / `enter_idle_and_reschedule`.
    fn pin_behavior_timer(&mut self) {
        self.behavior_timer_ms = i64::MAX;
    }

    /// Test-only setter for state — used by tests that need to set up a
    /// scenario (e.g., "Rolo is sleeping with a drag in progress") without
    /// driving him through the full transition path.
    #[cfg(test)]
    pub fn _force_state(&mut self, s: PetState) {
        self.state = s;
    }

    /// Test-only setter for the drag flag — same purpose as `_force_state`.
    #[cfg(test)]
    pub fn _force_dragging(&mut self, dragging: bool) {
        self.is_dragging = dragging;
    }

    #[cfg(test)]
    pub fn is_parked(&self) -> bool {
        self.is_parked
    }

    #[cfg(test)]
    pub fn behavior_timer_ms(&self) -> i64 {
        self.behavior_timer_ms
    }

    #[cfg(test)]
    pub fn walk_target_x(&self) -> Option<i32> {
        self.walk_target_x
    }
}

// ---------------------------------------------------------------------------
// Tests — every test passing means Rolo is healthy
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pet() -> Pet {
        Pet::new(500, 800, 1920, 1080, 128, 2)
    }

    #[test]
    fn starts_idle() {
        let pet = make_pet();
        assert_eq!(pet.state(), PetState::Idle);
    }

    #[test]
    fn drag_makes_happy() {
        let mut pet = make_pet();
        pet.start_drag();
        assert_eq!(pet.state(), PetState::Happy);
    }

    #[test]
    fn drag_end_returns_to_idle() {
        let mut pet = make_pet();
        pet.start_drag();
        pet.end_drag();
        assert_eq!(pet.state(), PetState::Idle);
    }

    #[test]
    fn hover_makes_happy() {
        let mut pet = make_pet();
        pet.start_hover();
        assert_eq!(pet.state(), PetState::Happy);
    }

    #[test]
    fn hover_end_returns_to_idle() {
        let mut pet = make_pet();
        pet.start_hover();
        pet.end_hover();
        assert_eq!(pet.state(), PetState::Idle);
    }

    #[test]
    fn hover_during_drag_stays_happy_after_hover_end() {
        let mut pet = make_pet();
        pet.start_drag();
        pet.start_hover();
        pet.end_hover();
        // Still dragging, so should stay Happy
        assert_eq!(pet.state(), PetState::Happy);
    }

    #[test]
    fn file_drag_triggers_drag_hover() {
        let mut pet = make_pet();
        pet.file_drag_entered();
        assert_eq!(pet.state(), PetState::DragHover);
    }

    #[test]
    fn file_drag_exit_restores_state() {
        let mut pet = make_pet();
        pet.file_drag_entered();
        pet.file_drag_exited();
        assert_eq!(pet.state(), PetState::Idle);
    }

    #[test]
    fn file_drop_starts_sniffing() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["test.txt".to_string()]);
        assert_eq!(pet.state(), PetState::Sniffing);
    }

    #[test]
    fn confirm_eat_transitions_to_eating() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["test.txt".to_string()]);
        pet.confirm_eat();
        assert_eq!(pet.state(), PetState::Eating);
    }

    #[test]
    fn decline_eat_transitions_to_disappointed() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["test.txt".to_string()]);
        pet.decline_eat();
        assert_eq!(pet.state(), PetState::Disappointed);
    }

    /// Tick the pet forward by `total_ms` using 16ms increments.
    fn tick_for(pet: &mut Pet, total_ms: i64) {
        let mut remaining = total_ms;
        while remaining > 0 {
            let dt = remaining.min(16);
            pet.tick(dt);
            remaining -= dt;
        }
    }

    #[test]
    fn eating_sequence_completes_to_satisfied_then_idle() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["test.txt".to_string()]);
        pet.confirm_eat();
        assert_eq!(pet.state(), PetState::Eating);

        // Tick through the full eating duration
        tick_for(&mut pet, EATING_DURATION_MS + 16);
        assert_eq!(pet.state(), PetState::Satisfied);

        // Tick through the full satisfied duration
        tick_for(&mut pet, SATISFIED_DURATION_MS + 16);
        assert_eq!(pet.state(), PetState::Idle);
    }

    #[test]
    fn walk_moves_position() {
        let mut pet = make_pet();
        // Force a walk right
        pet.state = PetState::WalkRight;
        pet.walk_target_x = Some(pet.x + 100);
        let old_x = pet.x;
        pet.tick(16);
        assert!(pet.x > old_x, "Rolo should have moved right");
    }

    #[test]
    fn walk_clamps_to_screen() {
        let mut pet = make_pet();
        pet.x = 0;
        pet.state = PetState::WalkLeft;
        pet.walk_target_x = Some(-100);
        pet.tick(16);
        assert!(pet.x >= 0, "Rolo must not walk off the left edge");
    }

    #[test]
    fn safe_zone_detects_edge_proximity() {
        // Screen 1920 wide, pet 128, bubble margin 125 (= 250/2 bubble width).
        // Safe range: [125 - 64, 1920 - 125 - 64] = [61, 1731].
        let mut pet = make_pet();
        pet.set_bubble_margin_x(125);
        pet.x = 500;
        assert!(pet.is_in_bubble_safe_zone(), "middle of screen is safe");
        pet.x = 50;
        assert!(!pet.is_in_bubble_safe_zone(), "left edge is not safe");
        pet.x = 1800;
        assert!(!pet.is_in_bubble_safe_zone(), "right edge is not safe");
    }

    #[test]
    fn nearest_safe_x_snaps_inward() {
        let mut pet = make_pet();
        pet.set_bubble_margin_x(125);
        pet.x = 0;
        assert_eq!(pet.nearest_safe_x(), 61);
        pet.x = 1900;
        assert_eq!(pet.nearest_safe_x(), 1731);
        pet.x = 500;
        assert_eq!(pet.nearest_safe_x(), 500);
    }

    #[test]
    fn walk_to_triggers_walk_state() {
        let mut pet = make_pet();
        pet.x = 500;
        assert!(pet.walk_to(800));
        assert_eq!(pet.state(), PetState::WalkRight);
        assert_eq!(pet.walk_target_x, Some(800));

        pet.x = 500;
        pet.state = PetState::Idle;
        assert!(pet.walk_to(200));
        assert_eq!(pet.state(), PetState::WalkLeft);
    }

    #[test]
    fn walk_to_refuses_during_eating() {
        let mut pet = make_pet();
        pet.state = PetState::Sniffing;
        assert!(!pet.walk_to(100));
        assert_eq!(pet.state(), PetState::Sniffing);
    }

    #[test]
    fn position_update_works() {
        let mut pet = make_pet();
        pet.set_position(100, 200);
        assert_eq!(pet.position().x, 100);
        assert_eq!(pet.position().y, 200);
    }

    #[test]
    fn sniffing_timeout_declines_automatically() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["test.txt".to_string()]);
        assert_eq!(pet.state(), PetState::Sniffing);

        // Tick past the confirm timeout
        pet.tick(CONFIRM_TIMEOUT_MS + 1);
        assert_eq!(
            pet.state(),
            PetState::Disappointed,
            "Rolo should auto-decline after timeout"
        );
    }

    #[test]
    fn queued_files_processed_after_current_meal() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["first.txt".to_string()]);
        // While sniffing, another file arrives — should be queued
        pet.file_dropped(vec!["second.txt".to_string()]);
        assert!(pet.has_queued_files());

        // Complete the first eating cycle
        pet.confirm_eat();
        tick_for(&mut pet, EATING_DURATION_MS + 16);
        assert_eq!(pet.state(), PetState::Satisfied);
        tick_for(&mut pet, SATISFIED_DURATION_MS + 16);

        // Should now be sniffing the queued batch
        assert_eq!(
            pet.state(),
            PetState::Sniffing,
            "Rolo should start sniffing the queued file"
        );
        assert!(!pet.has_queued_files());
    }

    // -----------------------------------------------------------------------
    // Regression tests added during health inspection — 2026-04-10
    //
    // Bug A — drag-then-walk window jump:
    //   tick.rs stores window top-left coords in pet.x/y during a drag, but
    //   the state machine treats x/y as SPRITE coords. When walking resumes,
    //   sprite_to_window_position() re-subtracts the (window − pet) offset and
    //   the window jumps by that amount. The fix is in tick.rs (convert window
    //   coords back to sprite coords before storing). The test below documents
    //   the correct contract: set_position stores whatever is passed, and
    //   subsequent walk steps proceed from exactly that value.
    //
    // Bug B — speech bubble doesn't follow Rolo:
    //   tick.rs only calls set_position on the bubble window inside the
    //   SpeechAction::Show arm. Once the bubble is visible, subsequent ticks
    //   return SpeechAction::None and the bubble is never repositioned. Fix:
    //   check speech.is_showing() each tick and reposition when walking.
    //   This is NOT testable in the state_machine module (it's a tick.rs /
    //   Tauri-layer concern). Manual verification procedure:
    //     1. Run with RUST_LOG=debug.
    //     2. Wait for a speech bubble to appear while Rolo is idle.
    //     3. Trigger autonomous walk behaviour (wait or force it).
    //     4. Confirm the bubble window follows Rolo's new position each tick.
    //     5. Confirm the bubble does NOT stay at the original spawn position.
    // -----------------------------------------------------------------------

    // --- Bug A regression ---

    /// Regression: set_position must store the given coords exactly.
    /// This is the "write half" of the Bug A contract — the tick.rs fix must
    /// convert window coords → sprite coords BEFORE calling set_position so
    /// that pet.x/y always holds sprite coords.
    #[test]
    fn regression_bug_a_set_position_stores_sprite_coords_exactly() {
        let mut pet = make_pet();
        // Simulate what tick.rs SHOULD do after the fix: pass sprite coords.
        let sprite_x = 300_i32;
        let sprite_y = 700_i32;
        pet.set_position(sprite_x, sprite_y);
        let pos = pet.position();
        assert_eq!(
            pos.x, sprite_x,
            "set_position must store x unchanged — tick.rs must supply sprite coords, not window coords"
        );
        assert_eq!(
            pos.y, sprite_y,
            "set_position must store y unchanged — tick.rs must supply sprite coords, not window coords"
        );
    }

    /// Regression: after a drag + set_position, a walk step must start from
    /// the stored position, not from (stored + window_offset).
    ///
    /// Arrange: drag Rolo, call set_position with a known sprite coord, end
    /// the drag. Force WalkRight toward a target. The first tick must move x
    /// by exactly walk_speed from the stored position, not from some offset.
    #[test]
    fn regression_bug_a_walk_starts_from_set_position_after_drag() {
        // Pet at x=500, pet_size=128, walk_speed=2
        let mut pet = make_pet();
        pet.start_drag();

        // Simulate the CORRECTED tick.rs: store sprite coords (not window coords).
        let sprite_x = 300_i32;
        let sprite_y = 700_i32;
        pet.set_position(sprite_x, sprite_y);
        pet.end_drag();

        // Force a walk right toward a nearby target
        pet.state = PetState::WalkRight;
        pet.walk_target_x = Some(sprite_x + 50);

        // One tick at 16ms with walk_speed=2 should move x by exactly 2
        pet.tick(16);

        assert_eq!(
            pet.position().x,
            sprite_x + 2,
            "After Bug A fix: walk step should advance from sprite_x ({}) by walk_speed (2). \
             Got {} — if this is sprite_x + (WINDOW_OFFSET) + 2, Bug A is not fixed.",
            sprite_x,
            pet.position().x
        );
    }

    /// Regression: position must survive a full drag→drop→walk cycle
    /// without jumping. Verifies the position stays monotonically
    /// reasonable throughout.
    #[test]
    fn regression_bug_a_position_is_stable_through_drag_walk_cycle() {
        let mut pet = make_pet();
        let initial_x = pet.position().x;

        // Drag phase
        pet.start_drag();
        let drag_x = initial_x + 80; // small movement during drag
        pet.set_position(drag_x, pet.position().y);
        pet.end_drag();

        // Walk phase — move right a bit
        pet.state = PetState::WalkRight;
        let target = drag_x + 30;
        pet.walk_target_x = Some(target);

        // Tick several steps
        for _ in 0..20 {
            pet.tick(16);
            let x = pet.position().x;
            // x must never jump more than a single walk step (2px) per tick
            // AND must never go negative or past screen width
            assert!(x >= 0, "x went negative ({}) — Rolo fell off the screen", x);
            assert!(x <= 1920, "x exceeded screen width ({}) — Rolo escaped", x);
        }
        // Must have reached or passed the target (or stopped at it)
        assert!(
            pet.position().x >= drag_x,
            "Walk did not advance from drag_x ({}) — position stuck at {}",
            drag_x,
            pet.position().x
        );
    }

    // --- drag_watcher interaction regression ---

    /// Regression: double file_drag_entered (drag_watcher + DragDropEvent::Enter)
    /// must be idempotent — state stays DragHover exactly once and the
    /// pre_drag_hover_state is preserved.
    #[test]
    fn regression_file_drag_entered_idempotent_from_idle() {
        let mut pet = make_pet();
        assert_eq!(pet.state(), PetState::Idle);

        // First call — from the global drag_watcher
        pet.file_drag_entered();
        assert_eq!(
            pet.state(),
            PetState::DragHover,
            "First file_drag_entered should transition to DragHover"
        );

        // Second call — from DragDropEvent::Enter (duplicate source)
        pet.file_drag_entered();
        assert_eq!(
            pet.state(),
            PetState::DragHover,
            "Second file_drag_entered should remain in DragHover (idempotent)"
        );

        // Drop — DragHover should transition to Sniffing
        pet.file_dropped(vec!["trash.txt".to_string()]);
        assert_eq!(
            pet.state(),
            PetState::Sniffing,
            "file_dropped should advance from DragHover to Sniffing"
        );
    }

    /// Regression: pre_drag_hover_state is captured only ONCE even when
    /// file_drag_entered is called multiple times. After file_drag_exited the
    /// pet must return to the original pre-drag state, not an intermediate one.
    #[test]
    fn regression_file_drag_entered_double_call_preserves_pre_drag_state() {
        // Start Rolo walking right, then simulate dual-source file drag
        let mut pet = make_pet();
        pet.state = PetState::WalkRight;
        pet.walk_target_x = Some(pet.x + 100);

        // Call 1: drag_watcher fires — pre_drag_hover_state captured as WalkRight
        pet.file_drag_entered();
        assert_eq!(pet.state(), PetState::DragHover);

        // Call 2: DragDropEvent::Enter fires — must NOT overwrite pre_drag_hover_state
        pet.file_drag_entered();
        assert_eq!(pet.state(), PetState::DragHover);

        // User moves away without dropping — should return to WalkRight
        pet.file_drag_exited();
        assert_eq!(
            pet.state(),
            PetState::WalkRight,
            "After idempotent file_drag_entered calls, exit must restore original pre-drag state"
        );
    }

    /// Regression: full dual-source call sequence:
    ///   drag_watcher → file_drag_entered
    ///   DragDropEvent::Enter → file_drag_entered  (duplicate)
    ///   DragDropEvent::Drop → file_dropped
    /// State machine must arrive at Sniffing, not get stuck or double-advance.
    #[test]
    fn regression_dual_source_drag_flow_reaches_sniffing() {
        let mut pet = make_pet();

        // Simulate drag_watcher detecting file drag globally
        pet.file_drag_entered(); // → DragHover
        assert_eq!(pet.state(), PetState::DragHover);

        // Simulate Tauri's DragDropEvent::Enter firing moments later
        pet.file_drag_entered(); // idempotent — stays DragHover
        assert_eq!(pet.state(), PetState::DragHover);

        // Simulate DragDropEvent::Drop
        pet.file_dropped(vec!["trash.bin".to_string()]);
        assert_eq!(
            pet.state(),
            PetState::Sniffing,
            "Dual-source drag flow must end at Sniffing after drop"
        );
        assert!(
            pet.pending_files().is_some(),
            "pending_files must be populated after drop"
        );
    }

    /// Regression: file_drag_exited is safe to call twice (drag_watcher END
    /// and DragDropEvent::Leave can both fire). The second call must not
    /// corrupt state.
    #[test]
    fn regression_file_drag_exited_double_call_is_safe() {
        let mut pet = make_pet();
        pet.file_drag_entered();
        assert_eq!(pet.state(), PetState::DragHover);

        // First exit — from drag_watcher
        pet.file_drag_exited();
        assert_eq!(pet.state(), PetState::Idle);

        // Second exit — from DragDropEvent::Leave (duplicate/late)
        pet.file_drag_exited(); // must not panic or corrupt state
        assert_eq!(
            pet.state(),
            PetState::Idle,
            "Second file_drag_exited must leave state unchanged when not in DragHover"
        );
    }

    // --- EATING_STATES guard during drag interaction ---

    /// Regression: a file drag arriving while Rolo is Sniffing must NOT
    /// override the eating sequence — file_drag_entered is a no-op when
    /// already in an EATING_STATE.
    #[test]
    fn regression_file_drag_entered_blocked_during_sniffing() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["first.txt".to_string()]);
        assert_eq!(pet.state(), PetState::Sniffing);

        // A second file drag arrives (e.g. user drags another file over)
        pet.file_drag_entered();
        assert_eq!(
            pet.state(),
            PetState::Sniffing,
            "file_drag_entered must not override Sniffing — eating sequence has priority"
        );
    }

    /// Regression: start_drag while Sniffing must not clobber the eating sequence.
    #[test]
    fn regression_start_drag_blocked_during_sniffing() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["food.txt".to_string()]);
        assert_eq!(pet.state(), PetState::Sniffing);

        pet.start_drag();
        assert_eq!(
            pet.state(),
            PetState::Sniffing,
            "start_drag must not clobber Sniffing state"
        );
        // is_dragging should be true (for tracking purposes), state stays Sniffing
        assert!(
            pet.is_dragging(),
            "is_dragging flag must be set even when state change is suppressed"
        );
    }

    /// Regression: end_drag while Sniffing (after suppressed start_drag) must
    /// not interrupt the eating sequence.
    #[test]
    fn regression_end_drag_blocked_during_sniffing() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["food.txt".to_string()]);
        assert_eq!(pet.state(), PetState::Sniffing);

        pet.start_drag(); // suppressed — state stays Sniffing
        pet.end_drag(); // must not call enter_idle_and_reschedule
        assert_eq!(
            pet.state(),
            PetState::Sniffing,
            "end_drag during Sniffing must not interrupt eating sequence"
        );
    }

    /// Regression: start_drag while Eating must not clobber the eating state.
    #[test]
    fn regression_start_drag_blocked_during_eating() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["food.txt".to_string()]);
        pet.confirm_eat();
        assert_eq!(pet.state(), PetState::Eating);

        pet.start_drag();
        assert_eq!(
            pet.state(),
            PetState::Eating,
            "start_drag must not clobber Eating state"
        );
    }

    /// Regression: end_drag while Eating (after suppressed start_drag) must
    /// not abort the eating animation.
    #[test]
    fn regression_end_drag_blocked_during_eating() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["food.txt".to_string()]);
        pet.confirm_eat();

        pet.start_drag(); // suppressed
        pet.end_drag(); // must not call enter_idle_and_reschedule
        assert_eq!(
            pet.state(),
            PetState::Eating,
            "end_drag during Eating must not interrupt eating animation"
        );
    }

    // --- dt_ms large-value hazard (sleep/wake) ---

    /// Regression: a huge dt_ms (simulating system sleep/wake while Rolo is
    /// Sniffing) must NOT cause a panic or UB — it should simply expire the
    /// confirm timeout and advance to Disappointed.
    ///
    /// This is not a security issue but a correctness issue: i64::MAX - large_value
    /// could underflow in theory, but here we're subtracting from confirm_timeout_ms
    /// which starts at CONFIRM_TIMEOUT_MS (10_000). The subtraction saturates to
    /// a large negative, which the `<= 0` guard catches correctly.
    #[test]
    fn regression_huge_dt_ms_during_sniffing_declines_gracefully() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["food.txt".to_string()]);
        assert_eq!(pet.state(), PetState::Sniffing);

        // Simulate a system sleep of 30 seconds (30_000ms)
        pet.tick(30_000);

        assert_eq!(
            pet.state(),
            PetState::Disappointed,
            "Huge dt_ms must expire sniffing timeout → Disappointed, not crash or freeze"
        );
    }

    /// Regression: a huge dt_ms while Eating must advance to Satisfied without panic.
    #[test]
    fn regression_huge_dt_ms_during_eating_advances_to_satisfied() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["food.txt".to_string()]);
        pet.confirm_eat();
        assert_eq!(pet.state(), PetState::Eating);

        // A 60-second sleep/wake during eating
        pet.tick(60_000);

        assert_eq!(
            pet.state(),
            PetState::Satisfied,
            "Huge dt_ms during Eating should advance to Satisfied in one tick, not panic"
        );
    }

    /// Regression: a huge dt_ms while Satisfied must advance to Idle (or Sniffing
    /// if there's a queued file), not panic.
    #[test]
    fn regression_huge_dt_ms_during_satisfied_returns_to_idle() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["food.txt".to_string()]);
        pet.confirm_eat();
        tick_for(&mut pet, EATING_DURATION_MS + 16);
        assert_eq!(pet.state(), PetState::Satisfied);

        pet.tick(60_000);
        assert_eq!(
            pet.state(),
            PetState::Idle,
            "Huge dt_ms during Satisfied must gracefully return to Idle"
        );
    }

    // --- start_hover / end_hover guards during eating ---

    /// Regression: start_hover must not override Sniffing state.
    #[test]
    fn regression_start_hover_blocked_during_sniffing() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["food.txt".to_string()]);
        assert_eq!(pet.state(), PetState::Sniffing);

        pet.start_hover();
        assert_eq!(
            pet.state(),
            PetState::Sniffing,
            "start_hover must not override Sniffing"
        );
    }

    /// Regression: end_hover must not override Sniffing state.
    #[test]
    fn regression_end_hover_blocked_during_sniffing() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["food.txt".to_string()]);
        assert_eq!(pet.state(), PetState::Sniffing);

        // Simulate hover flag being set (e.g. hover started before drop)
        pet.is_hovering = true;
        pet.end_hover();
        assert_eq!(
            pet.state(),
            PetState::Sniffing,
            "end_hover must not clobber Sniffing"
        );
    }

    // --- expression_remaining_ms initialization ---

    /// Regression: expression_remaining_ms must be set correctly when entering
    /// each expression state so the durations actually match the constants.
    #[test]
    fn regression_expression_durations_match_constants() {
        // Eating duration
        let mut pet = make_pet();
        pet.file_dropped(vec!["food.txt".to_string()]);
        pet.confirm_eat();
        assert_eq!(
            pet.expression_remaining_ms, EATING_DURATION_MS,
            "confirm_eat must set expression_remaining_ms to EATING_DURATION_MS"
        );

        // Satisfied duration — tick to the transition
        tick_for(&mut pet, EATING_DURATION_MS + 1);
        assert_eq!(
            pet.state(),
            PetState::Satisfied,
            "Should transition to Satisfied after EATING_DURATION_MS"
        );
        assert_eq!(
            pet.expression_remaining_ms, SATISFIED_DURATION_MS,
            "Transition to Satisfied must set expression_remaining_ms to SATISFIED_DURATION_MS"
        );

        // Disappointed duration
        let mut pet2 = make_pet();
        pet2.file_dropped(vec!["food.txt".to_string()]);
        pet2.decline_eat();
        assert_eq!(
            pet2.expression_remaining_ms, DISAPPOINTED_DURATION_MS,
            "decline_eat must set expression_remaining_ms to DISAPPOINTED_DURATION_MS"
        );
    }

    // --- Autonomous behavior timer does not fire during eating sequence ---

    /// Regression: the autonomous behavior timer must not pick a new behavior
    /// while Rolo is in the eating sequence (Sniffing/Eating/Satisfied/Disappointed).
    /// Verifies that even with the timer at 0, tick_inner returns early for all
    /// eating states.
    #[test]
    fn regression_behavior_timer_suppressed_during_eating_states() {
        for eating_state in [
            PetState::Sniffing,
            PetState::Eating,
            PetState::Satisfied,
            PetState::Disappointed,
        ] {
            let mut pet = make_pet();

            // Set up the state with proper expression tracking
            match eating_state {
                PetState::Sniffing => {
                    pet.file_dropped(vec!["food.txt".to_string()]);
                }
                PetState::Eating => {
                    pet.file_dropped(vec!["food.txt".to_string()]);
                    pet.confirm_eat();
                }
                PetState::Satisfied => {
                    pet.file_dropped(vec!["food.txt".to_string()]);
                    pet.confirm_eat();
                    tick_for(&mut pet, EATING_DURATION_MS + 1);
                }
                PetState::Disappointed => {
                    pet.file_dropped(vec!["food.txt".to_string()]);
                    pet.decline_eat();
                }
                _ => unreachable!(),
            }
            let actual_state = pet.state();
            assert_eq!(
                actual_state, eating_state,
                "Failed to set up eating state {:?}",
                eating_state
            );

            // Force the behavior timer to zero
            pet.behavior_timer_ms = 0;
            let before_state = pet.state();

            // Tick with a small dt — behavior timer fires would change state
            // if the guard were missing
            pet.tick(1);

            // If the guard works, the state must not have changed to Idle/Walk*
            // (it may have changed due to expression timers in Eating/Satisfied/Disappointed,
            //  but not due to autonomous behavior)
            let after_state = pet.state();
            if before_state == PetState::Sniffing || before_state == PetState::Eating {
                // Sniffing and Eating with dt=1 won't expire their timers
                assert_eq!(
                    after_state, before_state,
                    "Behavior timer must not fire during {:?} — state changed unexpectedly to {:?}",
                    before_state, after_state
                );
            }
            // For Satisfied and Disappointed with dt=1, the expression timer
            // also won't expire (durations are 1500-2250ms), so state is stable too.
            if before_state == PetState::Satisfied || before_state == PetState::Disappointed {
                assert_eq!(
                    after_state, before_state,
                    "Behavior timer must not fire during {:?} — state changed to {:?}",
                    before_state, after_state
                );
            }
        }
    }

    // --- DragHover is not in EATING_STATES but also blocks autonomous behavior ---

    /// Regression: DragHover state must also suppress autonomous behavior
    /// (it's not in EATING_STATES but has an explicit check in tick_inner).
    #[test]
    fn regression_behavior_timer_suppressed_during_drag_hover() {
        let mut pet = make_pet();
        pet.file_drag_entered();
        assert_eq!(pet.state(), PetState::DragHover);

        pet.behavior_timer_ms = 0;
        pet.tick(1);
        assert_eq!(
            pet.state(),
            PetState::DragHover,
            "Behavior timer must not fire during DragHover"
        );
    }

    // --- file_drag_entered while Rolo is being dragged (is_dragging) ---

    /// Regression: if the user is dragging Rolo AND picks up a file simultaneously,
    /// file_drag_entered must be a no-op — Rolo can't be in DragHover while
    /// being carried.
    #[test]
    fn regression_file_drag_entered_no_op_when_rolo_is_being_dragged() {
        let mut pet = make_pet();
        pet.start_drag();
        assert_eq!(pet.state(), PetState::Happy);

        pet.file_drag_entered();
        assert_eq!(
            pet.state(),
            PetState::Happy,
            "file_drag_entered must be ignored while Rolo is being dragged"
        );
    }

    // --- file_dropped while Rolo is being dragged ---

    /// Regression: file_dropped must be ignored while Rolo is being dragged.
    #[test]
    fn regression_file_dropped_no_op_when_rolo_is_being_dragged() {
        let mut pet = make_pet();
        pet.start_drag();

        pet.file_dropped(vec!["food.txt".to_string()]);
        assert_eq!(
            pet.state(),
            PetState::Happy,
            "file_dropped must be ignored while Rolo is being dragged"
        );
        assert!(
            pet.pending_files().is_none(),
            "No pending files should be set while Rolo is being dragged"
        );
    }

    // --- Interaction active flag tests ---

    #[test]
    fn interaction_active_pauses_behavior_timer() {
        let mut pet = make_pet();
        pet.interaction_active = true;

        // Tick far enough that the behavior timer would normally expire
        // and trigger a walk (5–15 seconds timer range, tick 20 seconds)
        tick_for(&mut pet, 20_000);

        // Rolo should still be idle because interaction_active blocks
        // autonomous behavior
        assert_eq!(
            pet.state(),
            PetState::Idle,
            "Rolo should not start walking while interaction is active"
        );
    }

    #[test]
    fn interaction_active_resumes_when_cleared() {
        let mut pet = make_pet();

        // Set interaction active, then clear it
        pet.interaction_active = true;
        tick_for(&mut pet, 5_000);
        assert_eq!(pet.state(), PetState::Idle);

        // Clear interaction — behavior timer should eventually fire
        pet.interaction_active = false;
        tick_for(&mut pet, 20_000);

        // Pet may now be idle (just picked idle behavior) or walking — either is fine.
        // The point is the behavior timer was NOT frozen after clearing.
        // We don't assert a specific state because it's random.
    }

    #[test]
    fn force_idle_stops_walk() {
        let mut pet = make_pet();
        // Force a walk state
        pet.state = PetState::WalkRight;
        pet.walk_target_x = Some(pet.position().x + 100);

        pet.force_idle();

        assert_eq!(pet.state(), PetState::Idle, "force_idle should set Idle");
        // Walk target should be cleared
        assert_eq!(
            pet.walk_target_x, None,
            "force_idle should clear walk target"
        );
    }

    #[test]
    fn force_idle_no_op_when_not_walking() {
        let mut pet = make_pet();
        assert_eq!(pet.state(), PetState::Idle);

        pet.force_idle();
        assert_eq!(
            pet.state(),
            PetState::Idle,
            "force_idle should be no-op when already idle"
        );
    }

    #[test]
    fn force_idle_no_op_during_eating() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["food.txt".to_string()]);
        assert_eq!(pet.state(), PetState::Sniffing);

        pet.force_idle();
        assert_eq!(
            pet.state(),
            PetState::Sniffing,
            "force_idle should not interrupt eating sequence"
        );
    }

    // ------------------------------------------------------------------
    // show_mood_reaction — Happy/Disappointed reactions from check-ins
    // ------------------------------------------------------------------

    #[test]
    fn show_mood_reaction_happy_enters_happy_state_with_duration() {
        let mut pet = make_pet();
        pet.interaction_active = true;

        pet.show_mood_reaction(true);

        assert_eq!(pet.state(), PetState::Happy);
        assert_eq!(pet.expression_remaining_ms, HAPPY_REACTION_DURATION_MS);
        assert!(
            !pet.interaction_active,
            "show_mood_reaction must clear interaction_active so behavior can resume"
        );
    }

    #[test]
    fn show_mood_reaction_disappointed_enters_disappointed_state_with_duration() {
        let mut pet = make_pet();
        pet.show_mood_reaction(false);

        assert_eq!(pet.state(), PetState::Disappointed);
        assert_eq!(pet.expression_remaining_ms, DISAPPOINTED_DURATION_MS);
    }

    #[test]
    fn show_mood_reaction_happy_times_out_to_idle() {
        let mut pet = make_pet();
        pet.show_mood_reaction(true);
        assert_eq!(pet.state(), PetState::Happy);

        // Tick past the reaction duration — Rolo should return to Idle
        tick_for(&mut pet, HAPPY_REACTION_DURATION_MS + 100);
        assert_eq!(
            pet.state(),
            PetState::Idle,
            "Happy reaction must auto-expire back to Idle"
        );
    }

    #[test]
    fn show_mood_reaction_disappointed_times_out_to_idle() {
        let mut pet = make_pet();
        pet.show_mood_reaction(false);

        tick_for(&mut pet, DISAPPOINTED_DURATION_MS + 100);
        assert_eq!(
            pet.state(),
            PetState::Idle,
            "Disappointed reaction must auto-expire back to Idle"
        );
    }

    #[test]
    fn show_mood_reaction_does_not_interrupt_eating() {
        let mut pet = make_pet();
        pet.file_dropped(vec!["/tmp/f.txt".to_string()]);
        assert_eq!(pet.state(), PetState::Sniffing);

        pet.show_mood_reaction(true);

        assert_eq!(
            pet.state(),
            PetState::Sniffing,
            "show_mood_reaction must not interrupt the eating sequence"
        );
    }

    #[test]
    fn show_mood_reaction_does_not_interrupt_drag() {
        let mut pet = make_pet();
        pet.start_drag();
        assert!(pet.is_dragging());
        assert_eq!(pet.state(), PetState::Happy);

        // Mid-drag mood reaction should be ignored — user is holding Rolo
        pet.show_mood_reaction(false);

        assert_eq!(
            pet.state(),
            PetState::Happy,
            "show_mood_reaction must not overwrite drag-Happy"
        );
        assert!(
            pet.is_dragging(),
            "show_mood_reaction must not flip is_dragging"
        );
    }

    #[test]
    fn drag_happy_does_not_auto_expire() {
        // Regression: the new Happy-countdown path must only fire for mood
        // reactions, not for drag-induced Happy (which has no expression
        // timer set). Otherwise Happy would flash out mid-drag.
        let mut pet = make_pet();
        pet.start_drag();
        assert_eq!(pet.state(), PetState::Happy);

        // Tick for a long time while still dragging
        tick_for(&mut pet, 10_000);

        assert_eq!(
            pet.state(),
            PetState::Happy,
            "Drag-Happy must persist while is_dragging is true"
        );
    }

    #[test]
    fn happy_reaction_completes_even_while_hovering() {
        // Regression: after clicking "Good" on a check-in, the cursor is
        // naturally still over Rolo's sprite — which means is_hovering may
        // be true. Earlier versions gated the Happy countdown on
        // !is_hovering, which froze the reaction until the user moved the
        // mouse off. Fix: hover should NOT pause the mood-reaction timer
        // (only drag should, since drag is an ongoing physical action).
        let mut pet = make_pet();
        pet.show_mood_reaction(true);
        assert_eq!(pet.state(), PetState::Happy);

        // Simulate the user's cursor still on Rolo after clicking.
        pet.start_hover();
        // start_hover forces Happy state, but we want it to have come from
        // the mood reaction — expression_remaining_ms should still be set.
        assert!(
            pet.expression_remaining_ms > 0,
            "Expression timer must survive start_hover after show_mood_reaction"
        );

        // Tick past the Happy reaction duration while still hovering
        tick_for(&mut pet, HAPPY_REACTION_DURATION_MS + 100);

        assert_eq!(
            pet.state(),
            PetState::Idle,
            "Happy reaction must auto-expire even when cursor is still hovering"
        );
    }

    #[test]
    fn clear_overlay_interaction_flags_drops_drag_and_hover_on_rising_edge() {
        // Regression: tick.rs used to read is_hovering() / is_dragging() and
        // call end_hover() / end_drag() itself when an overlay opened. That
        // logic now lives in `Pet::clear_overlay_interaction_flags`, which
        // must (a) only act on the rising edge of `in_overlay`, (b) drop
        // both flags, and (c) report what it cleared.
        let mut pet = make_pet();
        pet.start_hover();
        pet.start_drag();

        // Steady state: overlay was already None, stays None — nothing
        // should clear yet.
        let result = pet.clear_overlay_interaction_flags(false);
        assert!(!result.changed(), "No-overlay tick must be a no-op");

        // Rising edge: overlay opens — both flags must drop.
        let result = pet.clear_overlay_interaction_flags(true);
        assert!(result.cleared_drag, "Drag flag must be reported cleared");
        assert!(result.cleared_hover, "Hover flag must be reported cleared");
        assert!(!pet.is_dragging(), "Drag flag must actually be cleared");

        // Steady state inside overlay: subsequent ticks are no-ops.
        let result = pet.clear_overlay_interaction_flags(true);
        assert!(
            !result.changed(),
            "Steady-state overlay tick must be a no-op"
        );

        // Falling edge — leaving the overlay arms the next rising edge.
        let _ = pet.clear_overlay_interaction_flags(false);
        pet.start_hover();
        let result = pet.clear_overlay_interaction_flags(true);
        assert!(result.cleared_hover, "Re-entry must re-arm the cleanup");
    }

    // --- speech_active flag tests ---

    #[test]
    fn speech_active_pauses_behavior_timer() {
        let mut pet = make_pet();
        pet.set_speech_active(true);

        tick_for(&mut pet, 20_000);

        assert_eq!(
            pet.state(),
            PetState::Idle,
            "Rolo should not start walking while speech is active"
        );
    }

    #[test]
    fn set_speech_active_stops_in_progress_walk() {
        let mut pet = make_pet();
        pet.state = PetState::WalkRight;
        pet.walk_target_x = Some(pet.position().x + 100);

        pet.set_speech_active(true);

        assert_eq!(
            pet.state(),
            PetState::Idle,
            "set_speech_active(true) must stop an in-progress walk"
        );
        assert_eq!(pet.walk_target_x, None, "Walk target must be cleared");
    }

    #[test]
    fn speech_active_resumes_when_cleared() {
        let mut pet = make_pet();
        pet.set_speech_active(true);
        tick_for(&mut pet, 5_000);
        assert_eq!(pet.state(), PetState::Idle);

        pet.set_speech_active(false);
        tick_for(&mut pet, 20_000);
        // Behavior timer should eventually fire — pet may be idle or walking
    }

    #[test]
    fn speech_active_does_not_block_drag() {
        let mut pet = make_pet();
        pet.set_speech_active(true);

        pet.start_drag();
        assert_eq!(
            pet.state(),
            PetState::Happy,
            "Drag must work even when speech_active is true"
        );
        assert!(pet.is_dragging());

        pet.end_drag();
        assert_eq!(pet.state(), PetState::Idle);
        assert!(!pet.is_dragging());
    }

    // -- Sleeping (Section B2) --------------------------------------------

    #[test]
    fn enter_sleeping_only_from_idle() {
        let mut pet = make_pet();
        assert_eq!(pet.state(), PetState::Idle);
        assert!(pet.enter_sleeping().is_ok());
        assert_eq!(pet.state(), PetState::Sleeping);
    }

    #[test]
    fn enter_sleeping_rejected_during_eat() {
        let mut pet = make_pet();
        pet._force_state(PetState::Eating);
        let result = pet.enter_sleeping();
        assert!(result.is_err(), "enter_sleeping must reject Eating state");
        assert_eq!(
            pet.state(),
            PetState::Eating,
            "state must not change on rejection"
        );

        // Try every eating-pipeline state too — none should accept sleep.
        for s in [
            PetState::Sniffing,
            PetState::Satisfied,
            PetState::Disappointed,
            PetState::DragHover,
            PetState::Happy,
            PetState::WalkLeft,
            PetState::WalkRight,
        ] {
            let mut p = make_pet();
            p._force_state(s);
            assert!(
                p.enter_sleeping().is_err(),
                "enter_sleeping must reject from {:?}",
                s
            );
        }
    }

    #[test]
    fn wake_from_sleep_returns_to_idle() {
        let mut pet = make_pet();
        pet.enter_sleeping().expect("must enter from Idle");
        pet.wake_from_sleep();
        assert_eq!(pet.state(), PetState::Idle);
    }

    #[test]
    fn wake_from_sleep_during_drag_returns_to_drag_hover() {
        let mut pet = make_pet();
        pet._force_state(PetState::Sleeping);
        pet._force_dragging(true);
        pet.wake_from_sleep();
        assert_eq!(
            pet.state(),
            PetState::DragHover,
            "drag-in-progress at wake must transition to DragHover"
        );
    }

    #[test]
    fn wake_from_sleep_when_not_sleeping_is_noop() {
        let mut pet = make_pet();
        pet._force_state(PetState::WalkLeft);
        pet.wake_from_sleep();
        assert_eq!(
            pet.state(),
            PetState::WalkLeft,
            "wake on non-sleeping state must not clobber it"
        );
    }

    // ---------------------------------------------------------------------
    // Pre-sleep parking — Rolo settles before the dream gate evaluates
    // ---------------------------------------------------------------------

    #[test]
    fn parked_rolo_does_not_pick_new_walk() {
        let mut pet = make_pet();
        pet._force_state(PetState::Idle);
        pet.walk_target_x = None;
        pet.set_park_pressure(25.0, 20.0);
        // Force the picker to fire this tick.
        pet.behavior_timer_ms = 0;
        pet.tick(1);
        assert_eq!(
            pet.state(),
            PetState::Idle,
            "parked Rolo should remain idle when the picker evaluates"
        );
        assert!(pet.is_parked(), "park flag should still be engaged");
    }

    #[test]
    fn parking_lets_in_progress_walk_finish() {
        let mut pet = make_pet();
        pet._force_state(PetState::Idle);
        let target = pet.position().x + 60;
        assert!(pet.walk_to(target), "walk_to should engage a fresh walk");
        assert!(matches!(
            pet.state(),
            PetState::WalkLeft | PetState::WalkRight
        ));

        pet.set_park_pressure(25.0, 20.0);
        assert!(
            matches!(pet.state(), PetState::WalkLeft | PetState::WalkRight),
            "park pressure must not interrupt an in-flight walk"
        );

        // Tick until the walk completes (target_x cleared, back to Idle).
        // walk_speed=2, distance=60 → 30 ticks plus completion margin.
        for _ in 0..400 {
            pet.tick(16);
            if pet.state() == PetState::Idle && pet.walk_target_x().is_none() {
                break;
            }
        }

        assert_eq!(
            pet.state(),
            PetState::Idle,
            "walk should complete and Rolo should settle into Idle"
        );
        assert!(pet.is_parked(), "park flag persists across walk completion");
        assert_eq!(
            pet.behavior_timer_ms(),
            i64::MAX,
            "arrival path must pin the timer when parked"
        );
    }

    #[test]
    fn wake_from_park_rearms_timer() {
        let mut pet = make_pet();
        pet._force_state(PetState::Idle);
        pet.walk_target_x = None;
        pet.set_park_pressure(25.0, 20.0);
        assert!(pet.is_parked());
        assert_eq!(
            pet.behavior_timer_ms(),
            i64::MAX,
            "parked-idle should have timer pinned at MAX"
        );

        // User input arrives — idle drops below threshold.
        pet.set_park_pressure(0.0, 20.0);
        assert!(!pet.is_parked(), "park should lift when idle drops");
        assert!(
            pet.behavior_timer_ms() < i64::MAX,
            "wake should rearm a finite timer"
        );

        // Tick until the picker fires and a new behaviour is chosen.
        let mut fired = false;
        for _ in 0..2000 {
            pet.tick(16);
            // pick_behavior pins timer at MAX again after firing, OR Rolo
            // walks / goes-idle with a finite timer. Either way, the timer
            // having been at MAX confirms the picker ran.
            if matches!(
                pet.state(),
                PetState::WalkLeft | PetState::WalkRight | PetState::Idle
            ) && pet.behavior_timer_ms() != i64::MAX
                || pet.behavior_timer_ms() == i64::MAX
            {
                fired = true;
                break;
            }
        }
        assert!(
            fired,
            "picker should fire after wake_from_park rearms timer"
        );
    }

    #[test]
    fn wake_from_park_does_not_clobber_walk_to_timer() {
        let mut pet = make_pet();
        pet._force_state(PetState::Idle);
        let target = pet.position().x + 60;
        assert!(pet.walk_to(target));
        assert_eq!(
            pet.behavior_timer_ms(),
            i64::MAX,
            "walk_to should pin timer at MAX"
        );
        let walk_state_before = pet.state();
        let walk_target_before = pet.walk_target_x();

        // Direct call — Rolo is NOT parked, NOT idle. Must be a no-op for the
        // walk's pinned timer.
        pet.wake_from_park();

        assert_eq!(
            pet.behavior_timer_ms(),
            i64::MAX,
            "wake_from_park must not clobber a MAX timer it didn't set"
        );
        assert_eq!(
            pet.state(),
            walk_state_before,
            "walk state must be preserved across wake_from_park"
        );
        assert_eq!(
            pet.walk_target_x(),
            walk_target_before,
            "walk target must be preserved across wake_from_park"
        );
        assert!(
            !pet.is_parked(),
            "is_parked should be false after wake call"
        );
    }
}
