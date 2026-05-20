//! Rolo's heartbeat — the background tick loop that keeps him alive.
//!
//! This module spawns a dedicated thread that ticks Rolo's state machine at
//! a steady rate, emits Tauri events when things change, and moves his window
//! when he walks. If this loop stops, Rolo stops. It is his pulse.
//!
//! The tick loop also polls the cursor position via CoreGraphics (see
//! `platform.rs`) and drives a CursorState machine to handle hover and drag
//! interactions. This bypasses macOS's refusal to deliver mouse events to
//! unfocused transparent windows.
//!
//! The speech bubble system is also ticked here — when a phrase fires, the
//! bubble window is positioned, resized, shown, and the event is emitted.
//! When the display timer expires, the bubble is hidden.
//!
//! Design notes:
//! - Uses std::thread::spawn + sleep for simplicity and reliability.
//!   Async runtimes add complexity without benefit for a fixed-rate tick loop.
//! - The tick rate is ~60Hz (16ms) for smooth window movement during walks.
//! - State machine ticks use elapsed wall-clock time for accurate timers.
//! - Window position updates use PhysicalPosition to respect display scaling.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use chrono::Local;
use tauri::{Emitter, LogicalSize, Manager, PhysicalPosition, WebviewWindow};

use crate::commands::{LastRightClickPos, SharedMood, StatusPanelOpenFlag};
use crate::drag_watch::{DragEvent, DragWatcher};
use crate::interaction::{self, InteractionAction, InteractionState};
use crate::mood::{self, MoodEvent};
use crate::ollama::{LlmEvent, LlmRequest};
#[cfg(target_os = "macos")]
use crate::perception::{self, PerceptionEvent, SharedPerceptionBuffer};
// `platform` and `drag_watch` are platform-cfg'd internally; the public API
// exists on all targets (with no-op stubs on non-macOS) so this `use` works
// unconditionally and tick.rs stays free of cfg noise at the call sites.
use crate::platform;
use crate::speech::{self, SpeechAction, SpeechState};
use crate::state_machine::{is_eating_state, Pet, PetState};
use crate::vault::events::ExperienceEvent;
use crate::vault::Vault;
use crate::{MENU_ID_CHAT_WITH_ROLO, MENU_ID_OPEN_COMMAND_CENTER, MENU_ID_QUIT};

/// Tick interval target — 60Hz gives smooth walking motion.
const TICK_INTERVAL: Duration = Duration::from_millis(16);

/// Event name for state changes. The frontend subscribes to this.
pub const EVENT_STATE_CHANGED: &str = "rolo://state-changed";

/// Event name for position changes. Emitted alongside window moves.
pub const EVENT_POSITION_CHANGED: &str = "rolo://position-changed";

/// Event name for files pending confirmation (queued batch auto-started).
pub const EVENT_FILES_PENDING: &str = "rolo://files-pending";

/// Event name for streaming LLM tokens to the speech bubble.
pub const EVENT_SPEECH_TOKEN: &str = "rolo://speech-token";

/// Event name to show an interactive bubble (check-in prompt).
pub const EVENT_SHOW_INTERACTION: &str = "rolo://show-interaction";

/// Event name to hide the interactive bubble.
pub const EVENT_HIDE_INTERACTION: &str = "rolo://hide-interaction";

// ---------------------------------------------------------------------------
// OverlayMode — what kind of interactive overlay is active in the window
// ---------------------------------------------------------------------------

/// Determines how `tick_cursor` handles the cursor: normal sprite hit-testing
/// or full-window interactive mode for overlay UI (buttons, text input).
#[derive(Debug, Clone, Copy, PartialEq)]
enum OverlayMode {
    /// Normal mode — only the sprite area triggers hover/drag.
    None,
    /// Sleeping mode — Rolo is asleep, so neither hover nor drag should
    /// register on his sprite. Cursor events stay ignored so clicks pass
    /// through to whatever's behind him; tick_cursor early-returns.
    Sleeping,
    /// Sniffing mode — the confirmation bubble fills the window.
    Sniffing,
    /// Interaction mode — the check-in bubble fills the window.
    Interaction,
}

// ---------------------------------------------------------------------------
// Cursor state machine — drives hover/drag from Rust-side polling
// ---------------------------------------------------------------------------

/// Tracks whether the cursor is idle, hovering over Rolo, or dragging him.
#[derive(Debug)]
enum CursorState {
    /// Cursor is not over Rolo's sprite.
    Idle,
    /// Cursor is over Rolo's sprite but the mouse button is not pressed.
    Hovering,
    /// User is dragging Rolo. We store the offset from the cursor to the
    /// sprite's top-left corner so the sprite follows smoothly regardless
    /// of layout mode changes.
    Dragging { offset_x: f64, offset_y: f64 },
}

/// Hit-test the cursor against Rolo's visible pixels.
/// The 320x320 window has the 128x128 sprite at bottom-center. We first
/// reject cursors outside the sprite rect, then — when a hit mask is
/// available — sample the merged idle-alpha mask so hover only triggers on
/// visible pixels, not transparent regions inside the sprite bounds.
/// All values in physical pixels.
#[allow(clippy::too_many_arguments)]
fn cursor_is_over_sprite(
    cursor_phys_x: f64,
    cursor_phys_y: f64,
    win_x: f64,
    win_y: f64,
    scale_factor: f64,
    window_size_logical: f64,
    pet_size_logical: f64,
    hit_mask: Option<&crate::hitmask::HitMask>,
    top_anchored: bool,
) -> bool {
    let win_size = window_size_logical * scale_factor;
    let pet_size = pet_size_logical * scale_factor;

    let sprite_left = win_x + (win_size - pet_size) / 2.0;
    let sprite_top = if top_anchored {
        win_y
    } else {
        win_y + (win_size - pet_size)
    };
    let sprite_right = sprite_left + pet_size;
    let sprite_bottom = sprite_top + pet_size;

    if cursor_phys_x < sprite_left
        || cursor_phys_x >= sprite_right
        || cursor_phys_y < sprite_top
        || cursor_phys_y >= sprite_bottom
    {
        return false;
    }

    match hit_mask {
        Some(mask) => {
            let u = (cursor_phys_x - sprite_left) / pet_size;
            let v = (cursor_phys_y - sprite_top) / pet_size;
            mask.contains_normalized(u, v)
        }
        // Fallback if the mask failed to load — full sprite rect, keeping
        // Rolo hoverable rather than dead.
        None => true,
    }
}

/// Hit-test the cursor against the full window bounds (for sniffing mode).
/// All values in physical pixels.
fn cursor_is_in_window(
    cursor_phys_x: f64,
    cursor_phys_y: f64,
    win_x: f64,
    win_y: f64,
    win_size_phys: f64,
) -> bool {
    cursor_phys_x >= win_x
        && cursor_phys_x < win_x + win_size_phys
        && cursor_phys_y >= win_y
        && cursor_phys_y < win_y + win_size_phys
}

/// Result of one tick of cursor processing.
struct CursorTickResult {
    /// Window was moved by a drag operation.
    dragged: bool,
    /// The cursor state machine changed the pet's state (hover/drag start/end).
    pet_state_changed: bool,
}

/// Process one tick of cursor state, updating the pet and window as needed.
///
/// When `overlay_mode` is `Sniffing` or `Interaction`, the hover/drag state
/// machine is suspended and the entire 320x320 window becomes interactive so
/// buttons and text inputs can receive clicks. Otherwise, only the sprite area
/// (128x128 at bottom-center) triggers hover/drag.
#[allow(clippy::too_many_arguments)]
fn tick_cursor(
    cursor_state: &mut CursorState,
    pet: &Arc<Mutex<Pet>>,
    window: &WebviewWindow,
    scale_factor: f64,
    win_size_phys: f64,
    window_size_logical: f64,
    pet_size_logical: f64,
    overlay_mode: OverlayMode,
    ignore_cursor_active: &mut bool,
    tick_count: u64,
    hit_mask: Option<&crate::hitmask::HitMask>,
    screen_w: i32,
    screen_h: i32,
    pet_size_phys: i32,
    top_anchored: &mut bool,
) -> CursorTickResult {
    let no_change = CursorTickResult {
        dragged: false,
        pet_state_changed: false,
    };

    let cursor = match platform::get_cursor_info() {
        Some(c) => c,
        None => {
            if tick_count.is_multiple_of(300) {
                log::warn!("[Rolo] CoreGraphics cursor poll failed — feeling unwell");
            }
            return no_change;
        }
    };

    // Convert cursor from points (logical) to physical pixels
    let cx = cursor.x * scale_factor;
    let cy = cursor.y * scale_factor;

    // Get current window position (physical pixels)
    let win_pos = match window.outer_position() {
        Ok(pos) => pos,
        Err(e) => {
            if tick_count.is_multiple_of(300) {
                log::warn!("[Rolo] Can't read window position: {}", e);
            }
            return no_change;
        }
    };
    let wx = win_pos.x as f64;
    let wy = win_pos.y as f64;

    // --- Sleeping: cursor passes through entirely. Don't toggle the window
    // into interactive mode — Rolo is unreachable while asleep. Reset cursor
    // state so any in-progress hover/drag dialogue cleans up.
    if overlay_mode == OverlayMode::Sleeping {
        if !*ignore_cursor_active {
            let _ = window.set_ignore_cursor_events(true);
            *ignore_cursor_active = true;
        }
        if !matches!(cursor_state, CursorState::Idle) {
            *cursor_state = CursorState::Idle;
        }
        return no_change;
    }

    // --- Overlay mode: full window interactive for buttons/text input ---
    if overlay_mode != OverlayMode::None {
        let in_window = cursor_is_in_window(cx, cy, wx, wy, win_size_phys);
        if in_window && *ignore_cursor_active {
            let _ = window.set_ignore_cursor_events(false);
            *ignore_cursor_active = false;
        } else if !in_window && !*ignore_cursor_active {
            let _ = window.set_ignore_cursor_events(true);
            *ignore_cursor_active = true;
        }
        // Reset hover/drag state — no hover/drag interaction during sniffing
        if !matches!(cursor_state, CursorState::Idle) {
            *cursor_state = CursorState::Idle;
        }
        return no_change;
    }

    // --- Normal mode: sprite-area hit-test for hover/drag ---
    let inside = cursor_is_over_sprite(
        cx,
        cy,
        wx,
        wy,
        scale_factor,
        window_size_logical,
        pet_size_logical,
        hit_mask,
        *top_anchored,
    );

    // Periodic cursor diagnostic (every ~1s)
    if tick_count.is_multiple_of(60) {
        log::trace!(
            "[Rolo] cursor=({:.0},{:.0})phys window=({:.0},{:.0}) inside={} btn={} state={:?}",
            cx,
            cy,
            wx,
            wy,
            inside,
            cursor.left_down,
            cursor_state,
        );
    }

    // --- OS passthrough (ignore_cursor_events) — based purely on cursor
    // position, NOT on button state or pet state. This is critical: macOS
    // only delivers DragDropEvent to a window if cursor events are NOT
    // being ignored. If we gate this on "button up" we'd never receive
    // file drops (user holds the file with the button down the whole time).
    // During Dragging we don't toggle — the window is actively being moved
    // and we want cursor events to flow through uninterrupted.
    let is_cursor_dragging = matches!(cursor_state, CursorState::Dragging { .. });
    if !is_cursor_dragging {
        if inside && *ignore_cursor_active {
            log::info!(
                "[Rolo] Cursor over sprite — enabling cursor events (OS can deliver drag-drop)"
            );
            let _ = window.set_ignore_cursor_events(false);
            *ignore_cursor_active = false;
        } else if !inside && !*ignore_cursor_active {
            log::info!("[Rolo] Cursor left sprite area — restoring click-through");
            let _ = window.set_ignore_cursor_events(true);
            *ignore_cursor_active = true;
        }
    }

    let mut dragged = false;
    let mut pet_state_changed = false;

    match cursor_state {
        CursorState::Idle => {
            // Pet hover/drag state transitions only trigger when the mouse
            // button is UP. If the cursor arrives with the button already
            // held, the user is dragging a file from Finder — let the OS
            // DragDropEvent handler take care of it (file_drag_entered).
            // The passthrough toggle above has already cleared
            // ignore_cursor_events so macOS can deliver the drag events.
            if inside && !cursor.left_down {
                log::info!("[Rolo] Cursor over sprite, button up — Idle → Hovering");
                *cursor_state = CursorState::Hovering;
                if let Ok(mut pet_guard) = pet.lock() {
                    pet_guard.start_hover();
                    pet_state_changed = true;
                }
            }
        }

        CursorState::Hovering => {
            if cursor.left_down {
                // Capture cursor-to-SPRITE offset (not window offset) so the
                // drag math is layout-mode-independent — the sprite stays at
                // cursor - offset regardless of whether we flip mid-drag.
                let (sx, sy) = pet
                    .lock()
                    .map(|g| {
                        let p = g.position();
                        (p.x as f64, p.y as f64)
                    })
                    .unwrap_or_else(|_| {
                        let (dx, dy) = crate::geometry::sprite_window_offsets(scale_factor);
                        (wx + dx as f64, wy + dy as f64)
                    });
                let offset_x = cx - sx;
                let offset_y = cy - sy;
                log::info!(
                    "[Rolo] Mouse down — Hovering → Dragging (sprite offset={:.0},{:.0})",
                    offset_x,
                    offset_y,
                );
                *cursor_state = CursorState::Dragging { offset_x, offset_y };
                if let Ok(mut pet_guard) = pet.lock() {
                    pet_guard.start_drag();
                    pet_state_changed = true;
                }
            } else if !inside {
                log::info!("[Rolo] Cursor left sprite — Hovering → Idle");
                *cursor_state = CursorState::Idle;
                if let Ok(mut pet_guard) = pet.lock() {
                    pet_guard.end_hover();
                    pet_state_changed = true;
                }
            }
        }

        CursorState::Dragging { offset_x, offset_y } => {
            if !cursor.left_down {
                let new_state = if inside {
                    log::info!("[Rolo] Mouse released — Dragging → Hovering");
                    CursorState::Hovering
                } else {
                    log::info!("[Rolo] Mouse released outside — Dragging → Idle");
                    // Passthrough will be re-enabled by the top-of-function
                    // toggle on the next tick (inside=false).
                    CursorState::Idle
                };
                if let Ok(mut pet_guard) = pet.lock() {
                    pet_guard.end_drag();
                    pet_state_changed = true;
                }
                *cursor_state = new_state;
            } else {
                // Offsets are cursor-to-sprite, so this yields sprite coords.
                let raw_sprite_x = (cx - *offset_x).round() as i32;
                let raw_sprite_y = (cy - *offset_y).round() as i32;

                // Clamp so Rolo's sprite stays fully on-screen.
                let new_sprite_x = raw_sprite_x.clamp(0, (screen_w - pet_size_phys).max(0));
                let new_sprite_y = raw_sprite_y.clamp(0, (screen_h - pet_size_phys).max(0));

                // Compute layout mode — flip to top-anchored when the
                // normal layout would require a negative window Y.
                let current_layout = if *top_anchored {
                    crate::LayoutMode::TopAnchored
                } else {
                    crate::LayoutMode::Normal
                };
                let new_layout =
                    crate::compute_layout_mode(new_sprite_y, scale_factor, current_layout);
                let layout_changed = new_layout != current_layout;

                // If layout mode changed, synchronously toggle the CSS class
                // BEFORE moving the window to prevent a 1-2 frame visual jump.
                if layout_changed {
                    let class_op = match new_layout {
                        crate::LayoutMode::TopAnchored => "add",
                        crate::LayoutMode::Normal => "remove",
                    };
                    let _ = window.eval(format!(
                        "document.querySelector('.rolo-world')?.classList.{}('top-anchored')",
                        class_op,
                    ));
                }

                let (new_win_x, new_win_y) = crate::sprite_to_window_position(
                    new_sprite_x,
                    new_sprite_y,
                    scale_factor,
                    new_layout,
                );

                if let Err(e) = window.set_position(PhysicalPosition::new(new_win_x, new_win_y)) {
                    log::warn!(
                        "[Rolo] Drag move failed to ({},{}) — his paws slipped: {}",
                        new_win_x,
                        new_win_y,
                        e,
                    );
                } else {
                    if let Ok(mut pet_guard) = pet.lock() {
                        pet_guard.set_position(new_sprite_x, new_sprite_y);
                        pet_guard.top_anchored = new_layout == crate::LayoutMode::TopAnchored;
                    }
                    *top_anchored = new_layout == crate::LayoutMode::TopAnchored;
                    dragged = true;
                }
            }
        }
    }

    CursorTickResult {
        dragged,
        pet_state_changed,
    }
}

// ---------------------------------------------------------------------------
// Speech events
// ---------------------------------------------------------------------------

/// Event name to tell the bubble frontend to show a phrase.
pub const EVENT_SHOW_SPEECH: &str = "rolo://show-speech";

/// Event name to tell the bubble frontend to hide.
pub const EVENT_HIDE_SPEECH: &str = "rolo://hide-speech";

/// Event name sent when the bubble flips orientation during a reposition.
pub const EVENT_BUBBLE_REPOSITION: &str = "rolo://bubble-reposition";

/// Payload sent with the show-speech event.
#[derive(Clone, serde::Serialize)]
pub struct ShowSpeechPayload {
    pub text: String,
    pub flipped: bool,
}

/// Payload sent with each streaming token event.
#[derive(Clone, serde::Serialize)]
pub struct SpeechTokenPayload {
    pub token: String,
}

// ---------------------------------------------------------------------------
// Main tick loop
// ---------------------------------------------------------------------------

/// Build the [2] Current State slot for the prompt template (PRD §8.3).
/// Mood and energy are unknown until Phase 7a ships; this is the placeholder.
/// Decide whether the speech subsystem must be muted on this tick. Sleeping is
/// a hard mute regardless of whether anything else asked for one — Rolo cannot
/// speak while dreaming. The other two paths preserve the existing behavior:
/// an active interaction or an open chat window blocks autonomous speech.
///
/// Pulled out so the rule is unit-testable without spinning up the full tick
/// loop (which depends on Tauri AppHandle).
pub(crate) fn should_suppress_speech(
    pet_state: PetState,
    interaction_active: bool,
    chat_open: bool,
) -> bool {
    pet_state == PetState::Sleeping || interaction_active || chat_open
}

fn build_state_slot_placeholder(state: PetState) -> String {
    // Thin shim during the PRD/rolo-prompt-consolidation T1 migration window.
    // Delegates to `state_snapshot::render_state_slot_placeholder`. Will be
    // deleted in Commit 2 once tick.rs call sites use the free fn directly.
    crate::state_snapshot::render_state_slot_placeholder(state, chrono::Local::now())
}

/// The dispatcher performs a synchronous Ollama round-trip to the router
/// model (~1-3s on local Gemma). Running it on the tick thread would freeze
/// window movement for that duration — Rolo would walk in place. We spawn
/// onto Tauri's async runtime so the tick loop keeps ticking; when dispatch
/// returns, the spawned task sends `Generate` itself.
///
/// Idle bubbles pass `input = ""` — the pre-router routes to Unknown almost
/// always, the dispatcher returns Err, and the legacy prompt path runs
/// unchanged. Disabled-flag and missing-registry paths skip the spawn and
/// send `Generate` synchronously with the unaugmented prompt.
#[allow(clippy::too_many_arguments)]
fn dispatch_and_send_generate(
    app_handle: &tauri::AppHandle,
    vault: &Arc<crate::vault::Vault>,
    mood: &SharedMood,
    pet: &Arc<Mutex<crate::state_machine::Pet>>,
    llm_tx: &std::sync::mpsc::Sender<LlmRequest>,
    input: &str,
    base_prompt: String,
    context: String,
) {
    let send = |system_prompt: String, context: String| {
        let _ = llm_tx.send(LlmRequest::Generate {
            context,
            system_prompt,
        });
    };

    if !crate::tools::dispatcher::dispatcher_enabled() {
        send(base_prompt, context);
        return;
    }

    let registry = match app_handle.try_state::<Arc<crate::tools::ToolRegistry>>() {
        Some(s) => s.inner().clone(),
        None => {
            send(base_prompt, context);
            return;
        }
    };

    let pet_snapshot = pet.lock().unwrap_or_else(|p| p.into_inner()).clone();
    let vault = vault.clone();
    let mood = mood.clone();
    let llm_tx = llm_tx.clone();
    let input = input.to_string();

    // PRD/rolo-command-center.md Phase 5 Step E: tool routing is Ollama-only.
    // Snapshot the slot's current provider id BEFORE the spawn so we don't
    // race with a hot-swap from `cc_apply_brain`. If the slot isn't managed
    // (e.g., partial test wiring) we conservatively treat it as Ollama so
    // existing tests' behavior is preserved.
    let is_ollama = match app_handle.try_state::<crate::command_center::SharedProviderSlot>() {
        Some(slot) => slot.snapshot().provider_id() == "ollama",
        None => true,
    };

    tauri::async_runtime::spawn(async move {
        let clock = crate::state_snapshot::SystemClock;
        let dispatch_result = crate::tools::dispatcher::dispatch(
            &input,
            &vault,
            &mood,
            &pet_snapshot,
            &clock,
            &registry,
            is_ollama,
        )
        .await;
        let (augmented_system, augmented_context) = match dispatch_result {
            Ok(line) => (
                // Legacy path (fallback gemma3:4b uses system_prompt; the
                // chat-engine branches do not reach this function): keep
                // appending the tool line to the system prompt with a
                // leading newline — the pre-T5 behavior we don't want to
                // regress on.
                format!("{base_prompt}\n{line}"),
                // v2 (PRIMARY_MODEL) path: ollama.rs::generate drops the
                // system message on the wire when model == PRIMARY_MODEL,
                // so the tool line must ride inside `context` itself.
                // runtime_contract.md §7 reserves a bracketed line stacked
                // immediately after the state block and before the blank
                // line separator. `context` ends in `\n\n<idle>` here
                // (both call sites in start_tick_loop append the sentinel),
                // so we splice in `\n{line}` before that trailing pair.
                splice_tool_line_before_idle_sentinel(&context, &line),
            ),
            Err(_) => (base_prompt, context),
        };
        let _ = llm_tx.send(LlmRequest::Generate {
            context: augmented_context,
            system_prompt: augmented_system,
        });
    });
}

/// Splice a `[Context from <tool>: "..."]` line into a proactive-speech
/// `context` string at the slot reserved by runtime_contract.md §7.
///
/// `context` is expected to end with the IDLE_SENTINEL tail
/// (`\n\n<idle>`); the dispatcher only fires from `start_tick_loop` and
/// both call sites append the sentinel before invoking
/// `dispatch_and_send_generate`. We find the LAST occurrence of the
/// `\n\n<idle>` boundary and insert `\n{line}` immediately before it. The
/// resulting tail is `\n{line}\n\n<idle>` — exactly one blank line
/// between the bracket-stack and the user-message sentinel.
///
/// If the expected tail is somehow missing (defensive — would mean a
/// future caller forgot to append the sentinel), fall back to appending
/// `\n{line}` so the line at least reaches the model rather than getting
/// silently dropped. Logged at warn so the regression is visible.
fn splice_tool_line_before_idle_sentinel(context: &str, line: &str) -> String {
    let tail = format!("\n\n{}", crate::ollama::IDLE_SENTINEL);
    if let Some(idx) = context.rfind(&tail) {
        let mut out = String::with_capacity(context.len() + line.len() + 1);
        out.push_str(&context[..idx]);
        out.push('\n');
        out.push_str(line);
        out.push_str(&context[idx..]);
        out
    } else {
        log::warn!(
            "[Rolo dispatcher] proactive-speech context did not end with \
             IDLE_SENTINEL tail; appending tool line at the end as a \
             defensive fallback. Off-distribution by one blank-line — \
             check that the call site still pushes \\n\\n<idle>."
        );
        format!("{context}\n{line}")
    }
}

/// Append an `IdleSpeech` event for `text` to the vault. `dismissed` resolves
/// retroactively when a paired `Dismiss(IdleSpeech)` event lands within the
/// bubble's display window (PRD §4.4).
fn log_idle_speech(vault: &Arc<crate::vault::Vault>, text: &str) {
    vault
        .logger
        .log(&crate::vault::events::ExperienceEvent::IdleSpeech {
            event_id: crate::vault::events::new_id(),
            ts: chrono::Local::now(),
            text: text.to_string(),
            dismissed: false,
        });
}

/// Start Rolo's heartbeat. Call this once during app setup.
#[allow(clippy::too_many_arguments)]
pub fn start_tick_loop(
    app_handle: tauri::AppHandle,
    pet: Arc<Mutex<Pet>>,
    speech: Arc<Mutex<SpeechState>>,
    interaction: Arc<Mutex<InteractionState>>,
    mood: SharedMood,
    status_panel_open: StatusPanelOpenFlag,
    vault: Arc<Vault>,
    scale_factor: f64,
    window_size_logical: u32,
    pet_size_logical: u32,
    pet_size_phys: i32,
    screen_w: i32,
    screen_h: i32,
    hit_mask: Option<Arc<crate::hitmask::HitMask>>,
    heartbeat: crate::command_center::diagnostics::DragWatchHeartbeat,
    #[cfg(target_os = "macos")] perception_rx: std::sync::mpsc::Receiver<PerceptionEvent>,
    #[cfg(target_os = "macos")] perception_buffer: SharedPerceptionBuffer,
) {
    let win_size_phys = f64::from(window_size_logical) * scale_factor;
    let win_size_logical_f = f64::from(window_size_logical);
    let pet_size_logical_f = f64::from(pet_size_logical);

    // Resolve mood-state.json path once (depends on AppHandle, never changes
    // after startup). Periodic save inside the loop reuses this clone.
    let mood_path = mood::path_for(&app_handle);
    // Same pattern for interaction-state.json (the daily-fire counter that
    // enforces the 2-per-day cap across restarts).
    let interaction_path = interaction::path_for(&app_handle);

    thread::spawn(move || {
        let mut last_tick = Instant::now();
        let mut tick_count: u64 = 0;
        // Mood-loop accumulators. Save-to-disk every 5 min off-thread; emit
        // a MoodSnapshot every 250 ms while the status panel is open.
        let mut mood_save_accum_ms: i64 = 0;
        let mut mood_emit_accum_ms: i64 = 0;
        let mut cursor_state = CursorState::Idle;
        let mut ignore_cursor_active = true;
        // Right-click rising edge detection — only fires once per click.
        let mut prev_right_down = false;
        // Track the last state we emitted to the frontend, so we detect
        // between-tick state changes (e.g., drag-drop handler calling
        // file_dropped() or file_drag_entered() outside the tick loop).
        let mut last_emitted_state: Option<PetState> = None;
        // Layout mode: whether the sprite renders at the top or bottom of
        // the 320x320 window. Flips when sprite approaches screen top.
        let mut top_anchored = false;
        // Track the last flip direction emitted for the speech bubble so
        // we can detect threshold crossings during drag and emit a
        // reposition event only when the direction actually changes.
        let mut last_bubble_flipped: Option<bool> = None;
        let window = app_handle.get_webview_window("main");
        let bubble_window = app_handle.get_webview_window("speech-bubble");

        // Global file-drag watcher — polls the macOS drag pasteboard so
        // Rolo perks up the moment the user picks up any file on screen,
        // not just when the file is over Rolo's window.
        let mut drag_watcher = DragWatcher::new();

        // LLM thread — Rolo's brain
        let llm = crate::ollama::spawn_llm_thread();
        let llm_tx = llm.tx;
        let llm_rx = llm.rx;
        let llm_cancel = llm.cancel;

        // Fire an initial health check so we know if Ollama is up
        let _ = llm_tx.send(LlmRequest::HealthCheck);

        // Track whether we've shown the first token (to show/position bubble)
        let mut llm_first_token_pending = false;
        // Buffer tokens when Rolo isn't in the safe zone yet
        let mut llm_token_buffer: Vec<String> = Vec::new();
        let mut llm_waiting_for_safe_zone = false;
        // Track previous state for interaction-triggered speech
        let mut prev_pet_state: Option<PetState> = None;

        // True when the pending LLM call was timer-fired (idle speech), false
        // when interaction-triggered. Controls IdleSpeech vault event emission.
        let mut llm_call_idle_fired = false;
        // Drag event tracking — emit Drag vault event on the falling edge.
        let mut prev_is_dragging = false;
        let mut drag_start: Option<Instant> = None;
        // How many ticks per minute — used for once-per-minute rotate_if_needed.
        let ticks_per_minute: u64 = 60_000 / TICK_INTERVAL.as_millis() as u64;
        // Track local date so we can reset session-scoped speech counters when
        // the day rolls over (the same boundary at which the JSONL rotates).
        let mut last_local_date = Local::now().date_naive();

        let bubble_w_phys = crate::geometry::scaled_speech_width(scale_factor);
        let bubble_h_phys = crate::geometry::scaled_speech_height(scale_factor);
        let gap_above_phys = crate::geometry::scaled_speech_gap_above(scale_factor);
        let gap_below_phys = crate::geometry::scaled_speech_gap_below(scale_factor);

        log::info!(
            "[Rolo] Heartbeat started — scale={:.1} win_size_phys={:.0} pet_size_phys={}",
            scale_factor,
            win_size_phys,
            pet_size_phys,
        );

        loop {
            tick_count += 1;
            let now = Instant::now();

            // Stamp the drag-watch heartbeat so the Perception tab's
            // drag-drop probe can confirm this thread is still alive.
            // SystemTime avoids the per-tick libc timezone lookup that
            // `chrono::Local::now()` triggers; the probe only compares
            // against "now", no calendar arithmetic.
            // Relaxed ordering is enough — the probe only cares about a
            // recent value, not a memory-ordering relationship.
            heartbeat.store(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0),
                std::sync::atomic::Ordering::Relaxed,
            );
            let dt = now.duration_since(last_tick);
            last_tick = now;

            // Clamp so sleep/wake stalls can't drain every timer in one tick.
            let dt_ms_raw = dt.as_millis() as i64;
            let dt_ms = dt_ms_raw.min(500);

            // Looked up once per tick — both interaction firing (suppress
            // while open, arm post-chat delay on close) and speech-bubble
            // suppression read this. `get_webview_window` walks the window
            // manager's list, so checking it twice per ~16ms tick adds up.
            let chat_open_now = app_handle.get_webview_window("chat").is_some();

            // Once per minute: rotate the event log if the day changed and
            // reset session-scoped speech counters at the same boundary.
            if ticks_per_minute > 0 && tick_count.is_multiple_of(ticks_per_minute) {
                vault.logger.rotate_if_needed();
                let today = Local::now().date_naive();
                if today > last_local_date {
                    last_local_date = today;
                    if let Ok(mut g) = speech.lock() {
                        g.reset_dismiss_count();
                    }
                }
            }

            // Lock Rolo's state and tick him forward. Sample is_dragging in
            // the same lock so the per-tick drag-edge detection (and the two
            // downstream is_dragging reads below) don't take their own locks.
            let (tick_result, current_pet_state, pending_files_for_emit, now_is_dragging) = {
                let mut pet_guard = match pet.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => {
                        eprintln!(
                            "[Rolo] CRITICAL: State mutex poisoned! \
                             Attempting recovery — Rolo may behave erratically."
                        );
                        poisoned.into_inner()
                    }
                };

                let old_state = pet_guard.state();

                // Pre-sleep parking: as user-idle approaches the dream floor,
                // engage the park so Rolo settles into Idle for the gate.
                let user_idle_secs = crate::vault::idle::seconds_since_last_input().unwrap_or(0.0);
                let park_threshold = crate::vault::dreaming::dream_thresholds().park_threshold_secs;
                pet_guard.set_park_pressure(user_idle_secs, park_threshold);

                let result = pet_guard.tick(dt_ms);

                // If state transitioned to Sniffing (e.g., queued batch auto-started),
                // grab the pending files so we can emit them for the React bubble.
                let pending = if result.state_changed
                    && result.new_state == PetState::Sniffing
                    && old_state != PetState::Sniffing
                {
                    pet_guard.pending_files().cloned()
                } else {
                    None
                };

                let state = pet_guard.state();
                let dragging = pet_guard.is_dragging();
                (result, state, pending, dragging)
            };

            // Tick-loop hitch diagnostic. Target tick interval is ~16ms;
            // anything >50ms means something stalled the loop and Rolo
            // missed walk-steps / state-event emits. Log to surface the
            // "walks in place" symptom.
            if dt_ms_raw > 50 {
                log::warn!(
                    "[Rolo] Tick hitch: dt={}ms (clamped to {}ms) tick={} state={:?} walking={}",
                    dt_ms_raw,
                    dt_ms,
                    tick_count,
                    current_pet_state,
                    matches!(current_pet_state, PetState::WalkLeft | PetState::WalkRight),
                );
            }

            // Drag vault event — emit on the falling edge if the drag lasted >= 250 ms.
            if !prev_is_dragging && now_is_dragging {
                drag_start = Some(Instant::now());
            } else if prev_is_dragging && !now_is_dragging {
                let duration_ms = drag_start
                    .take()
                    .map(|s| s.elapsed().as_millis() as u64)
                    .unwrap_or(0);
                if duration_ms >= 250 {
                    vault.logger.log(&ExperienceEvent::Drag {
                        event_id: crate::vault::events::new_id(),
                        ts: Local::now(),
                        duration_ms,
                    });
                }
            }
            prev_is_dragging = now_is_dragging;

            // --- Tick the interaction engine ---
            let interaction_active_before;
            let mut interaction_state_changed = false;
            {
                let mut ix_guard = match interaction.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => {
                        eprintln!(
                            "[Rolo] CRITICAL: Interaction mutex poisoned! \
                             Rolo may miss check-ins."
                        );
                        poisoned.into_inner()
                    }
                };

                interaction_active_before = ix_guard.is_active();

                // Suppress during eating sequence or dragging
                let should_suppress = is_eating_state(current_pet_state) || now_is_dragging;
                ix_guard.suppress(should_suppress);

                // Get wall-clock time for scheduling
                let now_local = chrono::Local::now();
                let now_time = now_local.time();
                let now_date = now_local.date_naive();

                // chat_open_now is hoisted above; passed in so InteractionState
                // can suppress firing while open and arm the post-chat delay.
                let ix_action =
                    ix_guard.tick(dt_ms, current_pet_state, now_time, now_date, chat_open_now);

                // Snapshot persisted state under the lock if it changed —
                // we save off-thread below so we never hold the mutex across
                // an fs::rename. fires_today only changes ~2x/day so an
                // immediate save (vs. periodic) costs nothing and means the
                // cap is durable even on a hard kill.
                let interaction_save: Option<interaction::InteractionPersistedState> =
                    if ix_guard.take_dirty() {
                        Some(ix_guard.persisted_snapshot())
                    } else {
                        None
                    };

                drop(ix_guard);

                if let Some(snapshot) = interaction_save {
                    let path = interaction_path.clone();
                    std::thread::spawn(move || {
                        if let Err(e) = interaction::save_to_disk(&path, &snapshot) {
                            log::warn!("[Rolo] interaction-state save failed: {}", e);
                        }
                    });
                }

                match ix_action {
                    InteractionAction::Show(prompt) => {
                        log::info!(
                            "[Rolo] Interaction triggered: {} (instance {})",
                            prompt.interaction_id,
                            prompt.instance_id,
                        );

                        // Suppress active speech bubble first — no overlapping UI
                        {
                            let mut speech_guard = match speech.lock() {
                                Ok(g) => g,
                                Err(p) => p.into_inner(),
                            };
                            if speech_guard.is_showing() {
                                speech_guard.dismiss();
                                if let Some(ref bw) = bubble_window {
                                    let _ = bw.hide();
                                }
                                let _ = app_handle.emit(EVENT_HIDE_SPEECH, ());
                                log::info!(
                                    "[Rolo] Speech bubble hidden to make way for interaction"
                                );
                            }
                        }

                        // Force Rolo to Idle if he's walking — he shouldn't
                        // wander away from his own question.
                        if let Ok(mut pet_guard) = pet.lock() {
                            pet_guard.force_idle();
                            pet_guard.set_speech_active(false);
                            pet_guard.interaction_active = true;
                            interaction_state_changed = true;
                        }

                        // Emit event to the frontend
                        if let Err(e) = app_handle.emit(EVENT_SHOW_INTERACTION, &prompt) {
                            log::error!("[Rolo] Failed to emit show-interaction event: {}", e);
                        }
                    }
                    InteractionAction::Hide => {
                        log::info!("[Rolo] Interaction ended (timeout/dismiss)");

                        if let Ok(mut pet_guard) = pet.lock() {
                            pet_guard.interaction_active = false;
                            interaction_state_changed = true;
                        }

                        if let Err(e) = app_handle.emit(EVENT_HIDE_INTERACTION, ()) {
                            log::error!("[Rolo] Failed to emit hide-interaction event: {}", e);
                        }
                    }
                    InteractionAction::None => {}
                }
            }

            // Check if interaction was cleared externally (e.g., by submit_interaction_response command)
            let interaction_active_now = interaction.lock().map(|g| g.is_active()).unwrap_or(false);

            // Detect that the command handler cleared the interaction
            if interaction_active_before && !interaction_active_now {
                // The command handler responded — clear the pet flag
                if let Ok(mut pet_guard) = pet.lock() {
                    if pet_guard.interaction_active {
                        pet_guard.interaction_active = false;
                        interaction_state_changed = true;
                        log::info!("[Rolo] Interaction cleared by command handler");
                    }
                }
                // Emit hide event so the frontend knows to clean up
                let _ = app_handle.emit(EVENT_HIDE_INTERACTION, ());
            }

            // Cancel in-flight LLM if eating/sniffing starts
            if is_eating_state(current_pet_state) {
                let is_gen = speech.lock().map(|g| g.is_generating).unwrap_or(false);
                if is_gen {
                    llm_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                    if let Ok(mut g) = speech.lock() {
                        g.generation_done();
                    }
                    llm_first_token_pending = false;
                    llm_waiting_for_safe_zone = false;
                    llm_token_buffer.clear();
                    if let Some(ref bw) = bubble_window {
                        let _ = bw.hide();
                    }
                    let _ = app_handle.emit(EVENT_HIDE_SPEECH, ());
                    log::info!("[Rolo] LLM generation cancelled — eating sequence takes priority");
                }
            }

            // Determine overlay mode for cursor handling
            let pet_is_sniffing = current_pet_state == PetState::Sniffing;
            let pet_is_sleeping = current_pet_state == PetState::Sleeping;
            let overlay_mode = if pet_is_sleeping {
                // Sleeping wins over Sniffing/Interaction because both of those
                // require Rolo to NOT be asleep (enter_sleeping rejects from
                // any non-Idle state). This branch is defense-in-depth.
                OverlayMode::Sleeping
            } else if pet_is_sniffing {
                OverlayMode::Sniffing
            } else if interaction_active_now {
                OverlayMode::Interaction
            } else {
                OverlayMode::None
            };

            // Overlay entry: the cursor state machine is about to suspend
            // (tick_cursor returns early for non-None overlay mode), so
            // we need to clear any stale hover/drag flags on the Pet.
            // Otherwise, if Rolo is mid-drag or mid-hover when a check-in
            // fires, the flags stay true forever — freezing behavior_timer
            // (which gates on !is_dragging && !is_hovering) and blocking
            // Happy-reaction countdowns.
            //
            // Sleeping must preserve is_dragging — wake_from_sleep reads it
            // to decide whether to wake into DragHover or Idle. So we treat
            // Sleeping as "not in a clearing overlay" when computing the
            // rising-edge boolean, while still letting `Pet` own the edge
            // detection internally.
            let in_clearing_overlay = matches!(
                overlay_mode,
                OverlayMode::Sniffing | OverlayMode::Interaction,
            );
            if let Ok(mut pet_guard) = pet.lock() {
                let cleared = pet_guard.clear_overlay_interaction_flags(in_clearing_overlay);
                if cleared.cleared_drag {
                    log::info!(
                        "[Rolo] Overlay {:?} entered mid-drag — clearing is_dragging",
                        overlay_mode,
                    );
                }
                if cleared.cleared_hover {
                    log::info!(
                        "[Rolo] Overlay {:?} entered mid-hover — clearing is_hovering",
                        overlay_mode,
                    );
                }
                if cleared.changed() {
                    interaction_state_changed = true;
                }
            }
            // Also reset the cursor state machine on the same rising edge so
            // it matches what tick_cursor will do anyway — but log it this
            // time. `clear_overlay_interaction_flags` already swallowed the
            // edge bookkeeping, so we mirror its detection here for the log.
            if in_clearing_overlay && !matches!(cursor_state, CursorState::Idle) {
                log::info!(
                    "[Rolo] Overlay {:?} entered — resetting cursor_state {:?} → Idle",
                    overlay_mode,
                    cursor_state,
                );
                cursor_state = CursorState::Idle;
            }

            // Heartbeat log every ~5 seconds
            if tick_count.is_multiple_of(300) {
                log::debug!(
                    "[Rolo] Heartbeat tick={} state={:?} pos=({},{}) cursor={:?} overlay={:?}",
                    tick_count,
                    tick_result.new_state,
                    tick_result.new_position.x,
                    tick_result.new_position.y,
                    cursor_state,
                    overlay_mode,
                );
            }

            // --- Global file-drag polling ---
            // Poll the drag pasteboard each tick. If a new file drag starts
            // anywhere on screen, Rolo perks up (DragHover). If it ends
            // (mouse released), he settles back to whatever he was doing.
            let mut drag_watch_state_changed = false;
            if let Some(cursor) = platform::get_cursor_info() {
                if let Some(event) = drag_watcher.poll(cursor.left_down) {
                    if let Ok(mut pet_guard) = pet.lock() {
                        match event {
                            DragEvent::Started => {
                                log::info!(
                                    "[Rolo] Global drag START — pet state was {:?}",
                                    pet_guard.state()
                                );
                                pet_guard.file_drag_entered();
                                drag_watch_state_changed = true;
                            }
                            DragEvent::Ended => {
                                log::info!(
                                    "[Rolo] Global drag END — pet state was {:?}",
                                    pet_guard.state()
                                );
                                pet_guard.file_drag_exited();
                                drag_watch_state_changed = true;
                            }
                        }
                    }
                }
            }

            // --- Cursor polling (hover/drag/overlay click-through) ---
            let cursor_result = if let Some(ref win) = window {
                tick_cursor(
                    &mut cursor_state,
                    &pet,
                    win,
                    scale_factor,
                    win_size_phys,
                    win_size_logical_f,
                    pet_size_logical_f,
                    overlay_mode,
                    &mut ignore_cursor_active,
                    tick_count,
                    hit_mask.as_deref(),
                    screen_w,
                    screen_h,
                    pet_size_phys,
                    &mut top_anchored,
                )
            } else {
                CursorTickResult {
                    dragged: false,
                    pet_state_changed: false,
                }
            };

            // Dismiss speech bubble when a drag starts — picking Rolo up
            // interrupts his train of thought (and prevents bubble misalignment
            // when dragged outside the bubble-safe zone).
            if cursor_result.pet_state_changed && pet.lock().ok().is_some_and(|g| g.is_dragging()) {
                let mut sg = match speech.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                if sg.dismiss() {
                    if let Some(ref bw) = bubble_window {
                        let _ = bw.hide();
                    }
                    let _ = app_handle.emit(EVENT_HIDE_SPEECH, ());
                    if let Ok(mut pg) = pet.lock() {
                        pg.set_speech_active(false);
                    }
                }
            }

            // --- Right-click context menu detection ---
            // Detect right-click rising edge on the sprite area. The menu
            // must be created and shown on the main thread (macOS requirement).
            //
            // Note: we do NOT gate on overlay_mode — right-click should always
            // offer the context menu when the cursor is over the sprite,
            // even if a confirmation/interaction bubble is showing. The menu
            // is a separate OS-level popup and won't interfere.
            if let Some(ref win) = window {
                if let Some(cursor) = platform::get_cursor_info() {
                    let right_rising = cursor.right_down && !prev_right_down;
                    prev_right_down = cursor.right_down;

                    if right_rising {
                        let cx = cursor.x * scale_factor;
                        let cy = cursor.y * scale_factor;
                        let win_pos = win.outer_position();
                        if let Ok(pos) = win_pos {
                            let wx = pos.x as f64;
                            let wy = pos.y as f64;
                            let over_sprite = cursor_is_over_sprite(
                                cx,
                                cy,
                                wx,
                                wy,
                                scale_factor,
                                win_size_logical_f,
                                pet_size_logical_f,
                                hit_mask.as_deref(),
                                top_anchored,
                            );
                            if over_sprite {
                                log::info!(
                                    "[Rolo] Right-click detected on sprite — showing context menu"
                                );
                                // Stash the cursor (logical coords) so Phase 6
                                // can anchor the status-panel window. Phase 7
                                // adds the menu item that consumes this; for
                                // now, just keep the slot live so we never
                                // ship a phase that overwrites stale data.
                                if let Some(cell) = app_handle.try_state::<LastRightClickPos>() {
                                    if let Ok(mut g) = cell.lock() {
                                        *g = Some((cursor.x, cursor.y));
                                    }
                                }
                                let menu_win = win.clone();
                                let _ = win.run_on_main_thread(move || {
                                    use tauri::menu::{MenuBuilder, MenuItemBuilder};
                                    let command_center_item = match MenuItemBuilder::new(
                                        "Open Rolo Command Center",
                                    )
                                    .id(MENU_ID_OPEN_COMMAND_CENTER)
                                    .build(&menu_win)
                                    {
                                        Ok(item) => item,
                                        Err(e) => {
                                            log::warn!(
                                                "[Rolo] Failed to build open-command-center menu item: {}",
                                                e
                                            );
                                            return;
                                        }
                                    };
                                    let chat_item = match MenuItemBuilder::new("Chat with Rolo")
                                        .id(MENU_ID_CHAT_WITH_ROLO)
                                        .build(&menu_win)
                                    {
                                        Ok(item) => item,
                                        Err(e) => {
                                            log::warn!(
                                                "[Rolo] Failed to build chat-with-rolo menu item: {}",
                                                e
                                            );
                                            return;
                                        }
                                    };
                                    let view_status = match MenuItemBuilder::new("View Status")
                                        .id("view-status")
                                        .build(&menu_win)
                                    {
                                        Ok(item) => item,
                                        Err(e) => {
                                            log::warn!(
                                                "[Rolo] Failed to build view-status menu item: {}",
                                                e
                                            );
                                            return;
                                        }
                                    };
                                    let quit_item = match MenuItemBuilder::new("Quit Rolo")
                                        .id(MENU_ID_QUIT)
                                        .build(&menu_win)
                                    {
                                        Ok(item) => item,
                                        Err(e) => {
                                            log::warn!(
                                                "[Rolo] Failed to build quit-rolo menu item: {}",
                                                e
                                            );
                                            return;
                                        }
                                    };
                                    match MenuBuilder::new(&menu_win)
                                        .item(&chat_item)
                                        .item(&view_status)
                                        .item(&command_center_item)
                                        .separator()
                                        .item(&quit_item)
                                        .build()
                                    {
                                        Ok(menu) => {
                                            if let Err(e) = menu_win.popup_menu(&menu) {
                                                log::warn!(
                                                    "[Rolo] Failed to show context menu: {}",
                                                    e
                                                );
                                            }
                                        }
                                        Err(e) => log::warn!(
                                            "[Rolo] Failed to build right-click menu: {}",
                                            e
                                        ),
                                    }
                                });
                            }
                        }
                    }
                } else {
                    prev_right_down = false;
                }
            }

            // Detect between-tick state changes (drag-drop handler, commands, etc.)
            let external_state_changed = last_emitted_state != Some(current_pet_state);

            // Emit state-changed event when anything changed the pet's state
            // or the layout mode flipped (so the frontend can toggle CSS).
            if tick_result.state_changed
                || cursor_result.pet_state_changed
                || drag_watch_state_changed
                || external_state_changed
                || interaction_state_changed
            {
                let payload = if cursor_result.pet_state_changed
                    || drag_watch_state_changed
                    || external_state_changed
                    || interaction_state_changed
                {
                    pet.lock().ok().map(|g| g.state_payload())
                } else {
                    None
                }
                .unwrap_or(crate::state_machine::StatePayload {
                    state: tick_result.new_state,
                    position: tick_result.new_position,
                    top_anchored,
                });
                last_emitted_state = Some(payload.state);
                if let Err(e) = app_handle.emit(EVENT_STATE_CHANGED, &payload) {
                    eprintln!(
                        "[Rolo] Failed to emit state change event — \
                         the frontend may not know Rolo changed to {:?}: {}",
                        payload.state, e
                    );
                }
            }

            // Emit files-pending for sniffing transitions
            if let Some(files) = pending_files_for_emit {
                if let Err(e) = app_handle.emit(EVENT_FILES_PENDING, &files) {
                    eprintln!(
                        "[Rolo] Failed to emit files-pending event — \
                         the bubble won't know what he's sniffing: {}",
                        e
                    );
                }
            } else if external_state_changed && current_pet_state == PetState::Sniffing {
                // Between-tick transition to Sniffing (e.g., file drop) —
                // grab and emit the pending files now.
                if let Ok(guard) = pet.lock() {
                    if let Some(files) = guard.pending_files() {
                        let _ = app_handle.emit(EVENT_FILES_PENDING, files);
                    }
                }
            }

            // Current sprite position — freshest source of truth. During a
            // walk, tick_result.new_position is accurate. During a drag,
            // tick_cursor wrote new sprite coords directly to pet.x/y, so
            // read them back.
            let current_rolo_pos = if cursor_result.dragged {
                pet.lock()
                    .ok()
                    .map(|g| g.position())
                    .unwrap_or(tick_result.new_position)
            } else {
                tick_result.new_position
            };

            // macOS clamps windows below the menu bar, so the stored
            // sprite position can diverge from where the window actually is.
            let bubble_sprite_pos = window
                .as_ref()
                .and_then(|w| w.outer_position().ok())
                .map(|pos| {
                    let mode = if top_anchored {
                        crate::LayoutMode::TopAnchored
                    } else {
                        crate::LayoutMode::Normal
                    };
                    let (sx, sy) =
                        crate::window_to_sprite_position(pos.x, pos.y, scale_factor, mode);
                    crate::state_machine::Position { x: sx, y: sy }
                })
                .unwrap_or(current_rolo_pos);

            let reposition_bubble =
                |rolo_x: i32, rolo_y: i32, anchored: bool| -> speech::BubbleLayout {
                    let layout = speech::BubbleLayout::compute(
                        rolo_x,
                        rolo_y,
                        pet_size_phys,
                        bubble_w_phys,
                        bubble_h_phys,
                        screen_w,
                        screen_h,
                        gap_above_phys,
                        gap_below_phys,
                        anchored,
                    );
                    if let Some(ref bw) = bubble_window {
                        let _ = bw.set_position(PhysicalPosition::new(layout.x, layout.y));
                    }
                    layout
                };

            if tick_result.position_changed || cursor_result.dragged {
                if tick_result.position_changed && !cursor_result.dragged {
                    if let Some(ref win) = window {
                        let walk_layout = if top_anchored {
                            crate::LayoutMode::TopAnchored
                        } else {
                            crate::LayoutMode::Normal
                        };
                        let (win_x, win_y) = crate::sprite_to_window_position(
                            current_rolo_pos.x,
                            current_rolo_pos.y,
                            scale_factor,
                            walk_layout,
                        );
                        if let Err(e) = win.set_position(PhysicalPosition::new(win_x, win_y)) {
                            eprintln!(
                                "[Rolo] Failed to move window to ({}, {}) — \
                                 Rolo wanted to walk but his legs aren't working: {}",
                                win_x, win_y, e
                            );
                        }
                    }
                }

                if let Err(e) = app_handle.emit(EVENT_POSITION_CHANGED, &current_rolo_pos) {
                    eprintln!(
                        "[Rolo] Failed to emit position change — \
                         the frontend may show Rolo in the wrong place: {}",
                        e
                    );
                }
            }

            // Show/emit helpers for the LLM streaming path and phrases.json
            // show path. `flipped` is the tail orientation.
            let show_bubble_at = |_rolo_x: i32, _rolo_y: i32| -> speech::BubbleLayout {
                let layout =
                    reposition_bubble(bubble_sprite_pos.x, bubble_sprite_pos.y, top_anchored);
                if let Some(ref bw) = bubble_window {
                    let _ = bw.set_size(LogicalSize::new(
                        crate::geometry::SPEECH_MAX_WIDTH as f64,
                        crate::geometry::SPEECH_WINDOW_HEIGHT as f64,
                    ));
                    let _ = bw.show();
                }
                layout
            };
            let emit_show = |text: String| -> bool {
                let layout = show_bubble_at(bubble_sprite_pos.x, bubble_sprite_pos.y);
                let payload = ShowSpeechPayload {
                    text,
                    flipped: layout.flipped,
                };
                let _ = app_handle.emit(EVENT_SHOW_SPEECH, &payload);
                if let Ok(mut pg) = pet.lock() {
                    pg.set_speech_active(true);
                }
                layout.flipped
            };

            // --- LLM event polling ---
            // Drain all pending LLM events each tick (non-blocking).
            loop {
                match llm_rx.try_recv() {
                    Ok(LlmEvent::Token(t)) => {
                        if llm_waiting_for_safe_zone {
                            llm_token_buffer.push(t);
                            continue;
                        }
                        if llm_first_token_pending {
                            llm_first_token_pending = false;
                            show_bubble_at(current_rolo_pos.x, current_rolo_pos.y);
                        }
                        let _ =
                            app_handle.emit(EVENT_SPEECH_TOKEN, &SpeechTokenPayload { token: t });
                    }
                    Ok(LlmEvent::Done(full_text)) => {
                        log::info!(
                            "[Rolo] LLM done: {:?}",
                            full_text.chars().take(60).collect::<String>()
                        );
                        let is_empty = full_text.trim().is_empty();
                        {
                            let mut speech_guard = match speech.lock() {
                                Ok(g) => g,
                                Err(p) => p.into_inner(),
                            };
                            speech_guard.generation_done();
                            if !is_empty {
                                let duration = speech::calculate_display_duration(&full_text);
                                speech_guard.start_showing(duration);
                            }
                        }
                        llm_first_token_pending = false;
                        llm_waiting_for_safe_zone = false;
                        llm_token_buffer.clear();
                        if is_empty {
                            log::warn!("[Rolo] LLM returned empty text — skipping bubble");
                        } else {
                            if llm_call_idle_fired {
                                log_idle_speech(&vault, &full_text);
                            }
                            emit_show(full_text);
                        }
                        llm_call_idle_fired = false;
                    }
                    Ok(LlmEvent::Error(e)) => {
                        log::warn!("[Rolo] LLM error — Rolo's brain hiccup: {}", e);
                        let mut speech_guard = match speech.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        speech_guard.generation_done();
                        llm_first_token_pending = false;
                        llm_waiting_for_safe_zone = false;
                        llm_token_buffer.clear();

                        // Show "brain fuzzy" message once, then fall back
                        let fallback_text = speech_guard
                            .take_brain_fuzzy_message()
                            .or_else(|| {
                                speech_guard
                                    .pick_fallback(current_pet_state)
                                    .map(|(t, _)| t)
                            })
                            .unwrap_or_default();

                        if !fallback_text.is_empty() {
                            let duration = speech::calculate_display_duration(&fallback_text);
                            speech_guard.start_showing(duration);
                            drop(speech_guard);
                            emit_show(fallback_text);
                        } else {
                            speech_guard.mark_ollama_down();
                        }
                    }
                    Ok(LlmEvent::HealthResult(ok)) => {
                        log::info!(
                            "[Rolo] Ollama health: {} — Rolo's brain is {}",
                            ok,
                            if ok { "online" } else { "offline" }
                        );
                        let mut speech_guard = match speech.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        speech_guard.ollama_available = ok;
                        if !ok {
                            speech_guard.mark_ollama_down();
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        log::error!("[Rolo] LLM channel disconnected — Rolo lost his brain!");
                        break;
                    }
                }
            }

            // Flush buffered tokens when Rolo reaches the safe zone
            if llm_waiting_for_safe_zone {
                let in_safe = pet
                    .lock()
                    .map(|g| g.is_in_bubble_safe_zone())
                    .unwrap_or(true);
                if in_safe {
                    llm_waiting_for_safe_zone = false;
                    if !llm_token_buffer.is_empty() {
                        llm_first_token_pending = false;
                        show_bubble_at(current_rolo_pos.x, current_rolo_pos.y);
                        for buffered_token in llm_token_buffer.drain(..) {
                            let _ = app_handle.emit(
                                EVENT_SPEECH_TOKEN,
                                &SpeechTokenPayload {
                                    token: buffered_token,
                                },
                            );
                        }
                    }
                }
            }

            // Periodic Ollama health check retry
            {
                let should_retry = speech
                    .lock()
                    .map(|g| g.should_retry_health_check())
                    .unwrap_or(false);
                if should_retry {
                    let _ = llm_tx.send(LlmRequest::HealthCheck);
                    if let Ok(mut g) = speech.lock() {
                        g.reset_retry_timer();
                    }
                }
            }

            // --- Mood state update ---
            // Time-driven decay every tick + transition-driven events. Reads
            // (but does not advance) `prev_pet_state` — the existing
            // assignment at end-of-tick keeps that responsibility. The lock
            // is held only for arithmetic — never across a Tauri call (PRD
            // §9 lock-ordering invariant).
            {
                let mut g = mood.lock().unwrap_or_else(|p| p.into_inner());
                g.energy_cached = mood::MoodState::compute_energy_from_clock(chrono::Local::now());
                g.decay(dt_ms);

                if let Some(prev) = prev_pet_state {
                    // Drag pickup — same predicate as the speech-trigger block
                    // below, so both fire on the same transition.
                    let pickup = matches!(
                        (prev, current_pet_state),
                        (
                            PetState::Idle | PetState::WalkLeft | PetState::WalkRight,
                            PetState::Happy,
                        )
                    ) && pet.lock().map(|p| p.is_dragging()).unwrap_or(false);
                    if pickup {
                        g.apply_event(MoodEvent::DragPickup);
                    }

                    if matches!(
                        (prev, current_pet_state),
                        (PetState::Eating, PetState::Satisfied)
                    ) {
                        g.apply_event(MoodEvent::Ate);
                    }
                    if matches!(
                        (prev, current_pet_state),
                        (PetState::Eating, PetState::Disappointed)
                    ) {
                        g.apply_event(MoodEvent::FoodDeclined);
                    }
                }
            }

            // Periodic save — every 5 min, off-thread. Snapshot the state
            // under the lock, then drop the guard before spawning the I/O
            // thread (lock-ordering: no Tauri/IO call under the mood mutex).
            mood_save_accum_ms += dt_ms;
            if mood_save_accum_ms >= 300_000 {
                mood_save_accum_ms = 0;
                let snap_for_save = mood.lock().ok().map(|g| g.clone());
                if let Some(state) = snap_for_save {
                    let path = mood_path.clone();
                    std::thread::spawn(move || {
                        if let Err(e) = mood::save_to_disk(&state, &path) {
                            log::warn!("[Rolo] mood save failed: {}", e);
                        }
                    });
                }
            }

            // 250 ms snapshot emit while the status panel is open. The flag
            // is always false until Phase 6, so this is a no-op today —
            // wired now so Phase 6 just flips the bit. `emit_to` to a missing
            // window is a silent no-op; we ignore the Result either way.
            if status_panel_open.load(Ordering::Relaxed) {
                mood_emit_accum_ms += dt_ms;
                if mood_emit_accum_ms >= 250 {
                    mood_emit_accum_ms = 0;
                    let snap = mood.lock().ok().map(|g| g.snapshot());
                    if let Some(snap) = snap {
                        let _ = app_handle.emit_to("status-panel", "mood-tick", snap);
                    }
                }
            }

            // --- Interaction-triggered LLM speech ---
            // On specific state transitions, generate contextual speech.
            if let Some(prev) = prev_pet_state {
                let should_trigger = match (prev, current_pet_state) {
                    // Cursor drag only (not file drag)
                    (PetState::Idle, PetState::DragHover) => {
                        !pet.lock().map(|g| g.file_drag_active).unwrap_or(true)
                    }
                    (
                        PetState::Idle | PetState::WalkLeft | PetState::WalkRight,
                        PetState::Happy,
                    ) => now_is_dragging,
                    (PetState::Eating, PetState::Satisfied) => true,
                    (PetState::Eating, PetState::Disappointed) => true,
                    _ => false,
                };

                if should_trigger {
                    let mut speech_guard = match speech.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    if speech_guard.ollama_available
                        && speech_guard.can_call_llm()
                        && !speech_guard.is_generating
                        && !speech_guard.is_showing()
                    {
                        let event_desc = match (prev, current_pet_state) {
                            (_, PetState::Happy) => Some("user started dragging Rolo"),
                            (PetState::Eating, PetState::Satisfied) => {
                                Some("Rolo just finished eating and loved it")
                            }
                            (PetState::Eating, PetState::Disappointed) => {
                                Some("Rolo was offered food but declined")
                            }
                            _ => None,
                        };
                        let (interaction_ms, file_drag) = pet
                            .lock()
                            .map(|g| (g.last_interaction_elapsed_ms(), g.file_drag_active))
                            .unwrap_or((0, false));
                        let mood_snap = {
                            let g = mood.lock().unwrap_or_else(|p| p.into_inner());
                            g.clone()
                        };
                        let now_local = chrono::Local::now();
                        let mut context = SpeechState::build_context_prompt(
                            &mood_snap,
                            now_local,
                            current_pet_state,
                            interaction_ms,
                            event_desc,
                            file_drag,
                        );
                        #[cfg(target_os = "macos")]
                        perception::append_perception_to_context(&mut context, &perception_buffer);
                        // v2 contract: complete the first-user-turn with the
                        // blank-line separator + IDLE_SENTINEL. See
                        // data/finetune/sft-v2/runtime_contract.md §2-§3.
                        context.push_str("\n\n");
                        context.push_str(crate::ollama::IDLE_SENTINEL);
                        let state_slot = build_state_slot_placeholder(current_pet_state);
                        let system_prompt = vault.assembler.assemble(&context, &state_slot);
                        speech_guard.mark_llm_call();
                        speech_guard.reset_timer();
                        drop(speech_guard);

                        llm_call_idle_fired = false; // interaction-triggered, not idle timer
                        llm_first_token_pending = true;
                        // Check safe zone
                        let in_safe = pet
                            .lock()
                            .map(|g| g.is_in_bubble_safe_zone())
                            .unwrap_or(true);
                        if !in_safe {
                            llm_waiting_for_safe_zone = true;
                        }
                        dispatch_and_send_generate(
                            &app_handle,
                            &vault,
                            &mood,
                            &pet,
                            &llm_tx,
                            event_desc.unwrap_or(""),
                            system_prompt,
                            context,
                        );
                    }
                }
            }
            prev_pet_state = Some(current_pet_state);

            // --- Perception drain ---
            // Drop events arriving while Rolo can't speak. Producer threads
            // are state-agnostic; this is the single chokepoint that
            // enforces the suppression rule. `evict_stale` runs every tick
            // regardless so a 3-min-old observation never reaches a prompt.
            #[cfg(target_os = "macos")]
            {
                let suppress_perception = should_suppress_speech(
                    current_pet_state,
                    interaction_active_now,
                    chat_open_now,
                );
                let buffer_for_drain = perception_buffer.clone();
                let mut buf = match buffer_for_drain.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                // Drain non-blocking. `Err(_)` covers both Empty and Disconnected;
                // we treat the latter as end-of-batch so the tick keeps running
                // even if every producer thread has died.
                let mut accepted_an_event = false;
                while let Ok(ev) = perception_rx.try_recv() {
                    if !suppress_perception && current_pet_state == PetState::Idle {
                        // `push` returns false during the 5-min post-consume
                        // COOLDOWN; only accelerate for events that will
                        // actually reach the next prompt.
                        if buf.push(ev, Instant::now()) {
                            accepted_an_event = true;
                        }
                    }
                    // else: drop. Producers never know — that's the contract.
                }
                buf.evict_stale(Instant::now());
                drop(buf);
                if accepted_an_event {
                    let mut sg = match speech.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    sg.accelerate_for_perception();
                }
            }

            // --- Speech bubble system ---
            // Suppress speech entirely when an interaction is active OR the
            // chat window is open OR Rolo is asleep. If a speech bubble is
            // currently showing, hide it before skipping. Sleeping is a hard
            // mute: idle-speech timers don't tick, in-flight LLM is cancelled,
            // and any active bubble is dismissed on entry.
            let suppress_speech =
                should_suppress_speech(current_pet_state, interaction_active_now, chat_open_now);
            let (speech_action, speech_showing) = if suppress_speech {
                let mut speech_guard = match speech.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => {
                        eprintln!(
                            "[Rolo] CRITICAL: Speech mutex poisoned! \
                             Rolo may lose his voice."
                        );
                        poisoned.into_inner()
                    }
                };
                // Cancel in-flight LLM generation
                if speech_guard.is_generating {
                    llm_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                    speech_guard.generation_done();
                    llm_first_token_pending = false;
                    llm_waiting_for_safe_zone = false;
                    llm_token_buffer.clear();
                    log::info!(
                        "[Rolo] LLM generation cancelled — {}",
                        if interaction_active_now {
                            "interaction takes priority"
                        } else if pet_is_sleeping {
                            "Rolo is sleeping"
                        } else {
                            "chat window takes priority"
                        },
                    );
                }
                if speech_guard.is_showing() {
                    speech_guard.dismiss();
                    if let Some(ref bw) = bubble_window {
                        let _ = bw.hide();
                    }
                    let _ = app_handle.emit(EVENT_HIDE_SPEECH, ());
                    log::info!(
                        "[Rolo] Speech suppressed — {}",
                        if interaction_active_now {
                            "interaction active"
                        } else if pet_is_sleeping {
                            "Rolo is sleeping"
                        } else {
                            "chat open"
                        },
                    );
                }
                // Don't tick the speech timer — it stays paused during suppression
                (SpeechAction::None, false)
            } else {
                // Normal speech tick
                let mut speech_guard = match speech.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => {
                        eprintln!(
                            "[Rolo] CRITICAL: Speech mutex poisoned! \
                             Rolo may lose his voice."
                        );
                        poisoned.into_inner()
                    }
                };
                let social_now = mood.lock().map(|g| g.social).unwrap_or(0.6);
                let action = speech_guard.tick(dt_ms, current_pet_state, social_now);
                (action, speech_guard.is_showing())
            };

            match speech_action {
                SpeechAction::GenerateLLM(_) => {
                    // Timer fired and Ollama is available — build context and send to LLM
                    let mut speech_guard = match speech.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    if !speech_guard.is_generating {
                        let (interaction_ms, file_drag) = pet
                            .lock()
                            .map(|g| (g.last_interaction_elapsed_ms(), g.file_drag_active))
                            .unwrap_or((0, false));
                        let mood_snap = {
                            let g = mood.lock().unwrap_or_else(|p| p.into_inner());
                            g.clone()
                        };
                        let now_local = chrono::Local::now();
                        let mut context = SpeechState::build_context_prompt(
                            &mood_snap,
                            now_local,
                            current_pet_state,
                            interaction_ms,
                            None,
                            file_drag,
                        );
                        #[cfg(target_os = "macos")]
                        perception::append_perception_to_context(&mut context, &perception_buffer);
                        // v2 contract: see other call site above.
                        context.push_str("\n\n");
                        context.push_str(crate::ollama::IDLE_SENTINEL);
                        let state_slot = build_state_slot_placeholder(current_pet_state);
                        let system_prompt = vault.assembler.assemble(&context, &state_slot);
                        speech_guard.mark_llm_call();
                        drop(speech_guard);

                        llm_call_idle_fired = true; // timer-fired idle speech
                        llm_first_token_pending = true;
                        let in_safe = pet
                            .lock()
                            .map(|g| g.is_in_bubble_safe_zone())
                            .unwrap_or(true);
                        if !in_safe {
                            llm_waiting_for_safe_zone = true;
                            let nudge_target = pet.lock().map(|g| g.nearest_safe_x()).unwrap_or(0);
                            if let Ok(mut pg) = pet.lock() {
                                let _ = pg.walk_to(nudge_target);
                            }
                        }
                        dispatch_and_send_generate(
                            &app_handle,
                            &vault,
                            &mood,
                            &pet,
                            &llm_tx,
                            "",
                            system_prompt,
                            context,
                        );
                    }
                }
                SpeechAction::Show(text) => {
                    // If Rolo is too close to an edge, nudge him inward to
                    // the nearest safe x BEFORE showing so the bubble fits
                    // without clipping. The phrase is parked in speech state
                    // and emitted when he arrives.
                    let (in_safe_zone, nudge_target) = {
                        let pet_guard = pet.lock();
                        match pet_guard {
                            Ok(g) => (g.is_in_bubble_safe_zone(), g.nearest_safe_x()),
                            Err(_) => (true, 0), // on poison, don't block speech
                        }
                    };

                    if in_safe_zone {
                        log_idle_speech(&vault, &text);
                        last_bubble_flipped = Some(emit_show(text));
                    } else {
                        let walked = {
                            let mut pet_guard = match pet.lock() {
                                Ok(g) => g,
                                Err(poisoned) => poisoned.into_inner(),
                            };
                            pet_guard.walk_to(nudge_target)
                        };
                        if walked {
                            let mut speech_guard = match speech.lock() {
                                Ok(g) => g,
                                Err(poisoned) => poisoned.into_inner(),
                            };
                            speech_guard.defer_active_show(text);
                            log::info!(
                                "[Rolo] Deferred phrase — nudging toward safe_x={}",
                                nudge_target,
                            );
                        } else {
                            // Can't nudge (eating, dragging, etc.) — show
                            // where he is; the existing bubble-clamp will
                            // at least keep it on-screen even if off-center.
                            log_idle_speech(&vault, &text);
                            last_bubble_flipped = Some(emit_show(text));
                        }
                    }
                }
                SpeechAction::Hide => {
                    if let Some(ref bw) = bubble_window {
                        let _ = bw.hide();
                    }
                    let _ = app_handle.emit(EVENT_HIDE_SPEECH, ());
                    last_bubble_flipped = None;
                    if let Ok(mut pg) = pet.lock() {
                        pg.set_speech_active(false);
                    }
                }
                SpeechAction::None => {
                    // Follow Rolo only when he actually moved — avoid
                    // redundant set_position calls on idle ticks.
                    // Also reposition during LLM streaming.
                    let is_gen = speech.lock().map(|g| g.is_generating).unwrap_or(false);
                    if (speech_showing || is_gen)
                        && (tick_result.position_changed || cursor_result.dragged)
                    {
                        let layout = reposition_bubble(
                            bubble_sprite_pos.x,
                            bubble_sprite_pos.y,
                            top_anchored,
                        );
                        if last_bubble_flipped != Some(layout.flipped) {
                            last_bubble_flipped = Some(layout.flipped);
                            #[derive(Clone, serde::Serialize)]
                            struct ReposPayload {
                                flipped: bool,
                            }
                            let _ = app_handle.emit(
                                EVENT_BUBBLE_REPOSITION,
                                ReposPayload {
                                    flipped: layout.flipped,
                                },
                            );
                        }
                    }

                    // If a phrase is deferred and Rolo has reached the safe
                    // zone, commit and show it now.
                    let ready = {
                        let speech_guard = speech.lock();
                        speech_guard.map(|g| g.has_deferred()).unwrap_or(false)
                    };
                    if ready {
                        let pet_ready = match pet.lock() {
                            Ok(g) => {
                                let idle = matches!(g.state(), PetState::Idle);
                                idle && g.is_in_bubble_safe_zone()
                            }
                            Err(_) => false,
                        };
                        if pet_ready {
                            let text = {
                                let mut speech_guard = match speech.lock() {
                                    Ok(g) => g,
                                    Err(poisoned) => poisoned.into_inner(),
                                };
                                speech_guard.take_deferred().map(|(t, _)| t)
                            };
                            if let Some(t) = text {
                                log::info!("[Rolo] Committing deferred phrase after nudge");
                                log_idle_speech(&vault, &t);
                                last_bubble_flipped = Some(emit_show(t));
                            }
                        }
                    }
                }
            }

            // Sleep to maintain tick rate
            let elapsed = Instant::now().duration_since(now);
            if let Some(remaining) = TICK_INTERVAL.checked_sub(elapsed) {
                thread::sleep(remaining);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------
    // Proactive-speech tool-line splicer (runtime_contract.md §2 + §7).
    //
    // For PRIMARY_MODEL the system prompt is dropped on the wire by
    // ollama.rs::generate, so the dispatcher's `[Context from ...]` line
    // must ride inside `context`. These tests pin the byte-exact splice
    // shape so the SFT brain's bubble input stays on-contract.
    // ---------------------------------------------------------------------

    #[test]
    fn splice_tool_line_inserts_before_idle_sentinel_with_one_blank_line() {
        let context =
            "[Mood: content | Energy: medium | Social: ok | Time: 2:30 PM | Hunger: just ate]\n\
                       [State: Rolo is sitting idle]\n\
                       \n\
                       <idle>";
        let line = "[Context from search_vault: \"we went hiking last weekend\"]";
        let got = splice_tool_line_before_idle_sentinel(context, line);
        let expected =
            "[Mood: content | Energy: medium | Social: ok | Time: 2:30 PM | Hunger: just ate]\n\
                        [State: Rolo is sitting idle]\n\
                        [Context from search_vault: \"we went hiking last weekend\"]\n\
                        \n\
                        <idle>";
        assert_eq!(got, expected);
    }

    /// Defensive fallback — if a future caller forgets to append the
    /// IDLE_SENTINEL tail, the splicer must not silently drop the tool
    /// line. It appends with a single newline so the line at least reaches
    /// the model, and warns in the log.
    #[test]
    fn splice_tool_line_appends_at_end_when_sentinel_tail_missing() {
        let context = "[Mood: a]\n[State: b]";
        let line = "[Context from x: \"y\"]";
        let got = splice_tool_line_before_idle_sentinel(context, line);
        assert_eq!(got, "[Mood: a]\n[State: b]\n[Context from x: \"y\"]");
    }

    /// Section B3: the speech-suppression gate must mute Rolo while he's
    /// asleep, regardless of the other flags. This boolean is the single gate
    /// the tick loop reads — every Sleeping side-effect (don't tick the speech
    /// timer, dismiss bubbles on entry, cancel in-flight LLM) hangs off of it.
    #[test]
    fn idle_speech_does_not_fire_while_sleeping() {
        // Sleeping must suppress regardless of what else is happening.
        assert!(
            should_suppress_speech(PetState::Sleeping, false, false),
            "Sleeping must suppress when no other reason exists"
        );
        assert!(
            should_suppress_speech(PetState::Sleeping, true, false),
            "Sleeping + interaction still suppresses"
        );
        assert!(
            should_suppress_speech(PetState::Sleeping, false, true),
            "Sleeping + chat still suppresses"
        );

        // Idle with no other reason must NOT suppress — proves the gate
        // isn't simply returning true for everything.
        assert!(
            !should_suppress_speech(PetState::Idle, false, false),
            "Idle with no other suppressor must allow speech"
        );

        // Existing suppressors must still fire from non-Sleeping states.
        assert!(should_suppress_speech(PetState::Idle, true, false));
        assert!(should_suppress_speech(PetState::Idle, false, true));
        assert!(should_suppress_speech(PetState::WalkLeft, true, false));
    }

    #[test]
    fn sleeping_overlay_mode_is_distinct() {
        // Sleeping is treated by tick_cursor as a passthrough — cursor events
        // stay ignored, the cursor state machine resets to Idle. We can't
        // drive a full Tauri window from a unit test, so we just assert the
        // enum variant exists and is distinct from None/Sniffing/Interaction
        // so the match arms above stay live.
        assert_ne!(OverlayMode::Sleeping, OverlayMode::None);
        assert_ne!(OverlayMode::Sleeping, OverlayMode::Sniffing);
        assert_ne!(OverlayMode::Sleeping, OverlayMode::Interaction);
    }
}
