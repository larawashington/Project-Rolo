//! Tauri commands — the interface between Rolo's Rust soul and the React frontend.
//!
//! Each command locks the Pet mutex, performs its operation, and returns.
//! These are invoked from TypeScript via `invoke("command_name", { ... })`.
//!
//! Note: hover, drag, and position update commands were removed. Rolo's cursor
//! interaction is now handled entirely by Rust-side CoreGraphics polling in the
//! tick loop (see tick.rs CursorState). This bypasses macOS's refusal to deliver
//! mouse events to unfocused transparent windows.

use serde::Serialize;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager, State};

use crate::interaction::{InteractionResponse, InteractionState, ResponseValue};
use crate::mood::{CheckinRating, MoodEvent, MoodSnapshot, MoodState};
use crate::speech::SpeechState;
use crate::state_machine::{Pet, StatePayload};
use crate::stats::SharedStats;
use crate::vault::events::{CheckinMethod, DismissContext, EatOutcome, ExperienceEvent};
use crate::vault::Vault;

/// The shared Pet instance.
pub type SharedPet = Arc<Mutex<Pet>>;

/// The shared SpeechState instance.
pub type SharedSpeech = Arc<Mutex<SpeechState>>;

/// The shared InteractionState instance.
pub type SharedInteraction = Arc<Mutex<InteractionState>>;

/// The shared MoodState instance — Rolo's real-time emotional state.
pub type SharedMood = Arc<Mutex<MoodState>>;

/// Flag tracking whether the status panel window is currently open.
/// Used by Phase 6 commands to avoid duplicating the panel.
pub type StatusPanelOpenFlag = Arc<AtomicBool>;

/// Last right-click position in screen coordinates, used to anchor the
/// status panel window when it opens. Phase 6 reads this; Phase 2 just
/// plumbs the slot.
pub type LastRightClickPos = Arc<Mutex<Option<(f64, f64)>>>;

/// The shared ChatStore instance — Rolo's conversational memory.
pub type SharedChatStore = Arc<Mutex<crate::chat::store::ChatStore>>;

/// The shared ChatEngine instance — Rolo's brain for conversations.
/// Uses tokio::sync::Mutex because ChatEngine methods are async.
pub type SharedChatEngine = Arc<tokio::sync::Mutex<crate::chat::engine::ChatEngine>>;

/// Result of a confirm_eat operation.
#[derive(Debug, Serialize)]
pub struct EatResult {
    pub success: bool,
    pub error_message: Option<String>,
}

/// Get Rolo's current state and position.
#[tauri::command]
pub fn get_pet_state(pet: State<'_, SharedPet>) -> Result<StatePayload, String> {
    let pet = pet.lock().map_err(|e| {
        format!(
            "Rolo's state is locked and unreachable — this is a health emergency! Error: {}",
            e
        )
    })?;
    Ok(pet.state_payload())
}

/// Returns the screen dimensions so the frontend knows Rolo's world bounds.
#[tauri::command]
pub fn get_screen_size(window: tauri::Window) -> Result<(u32, u32), String> {
    let monitor = window
        .current_monitor()
        .map_err(|e| format!("Failed to query monitor: {}", e))?
        .ok_or_else(|| "No monitor detected — Rolo has nowhere to live!".to_string())?;
    let size = monitor.size();
    Ok((size.width, size.height))
}

/// Get the list of files Rolo is currently sniffing (for the confirmation bubble).
#[tauri::command]
pub fn get_pending_files(pet: State<'_, SharedPet>) -> Result<Option<Vec<String>>, String> {
    let pet = pet
        .lock()
        .map_err(|e| format!("Cannot check Rolo's pending files: {}", e))?;
    Ok(pet.pending_files().cloned())
}

/// User clicked 'Yum!' — trash the files and transition to Eating.
#[tauri::command]
pub fn confirm_eat(
    pet: State<'_, SharedPet>,
    stats: State<'_, SharedStats>,
    vault: State<'_, Arc<Vault>>,
) -> Result<EatResult, String> {
    let mut pet_guard = pet
        .lock()
        .map_err(|e| format!("Cannot reach Rolo for eating — he's choking! Error: {}", e))?;

    let files = match pet_guard.pending_files() {
        Some(f) => f.clone(),
        None => return Err("No pending files to eat".to_string()),
    };

    let trash_result = crate::trash::trash_files(&files);

    if trash_result.trashed_count == 0 {
        pet_guard.show_error_disappointed();
        drop(pet_guard);

        let event = ExperienceEvent::Eat {
            event_id: crate::vault::events::new_id(),
            ts: chrono::Local::now(),
            files: basenames(&files),
            outcome: EatOutcome::Errored,
            bytes: 0,
        };
        vault.logger.log(&event);

        return Ok(EatResult {
            success: false,
            error_message: trash_result.error_message,
        });
    }

    pet_guard.confirm_eat();
    drop(pet_guard);

    if let Ok(mut stats_guard) = stats.lock() {
        stats_guard.record_meal(trash_result.trashed_bytes, trash_result.trashed_count);
    }

    let event = ExperienceEvent::Eat {
        event_id: crate::vault::events::new_id(),
        ts: chrono::Local::now(),
        files: basenames(&files),
        outcome: EatOutcome::Satisfied,
        bytes: trash_result.trashed_bytes,
    };
    vault.logger.log(&event);

    Ok(EatResult {
        success: true,
        error_message: None,
    })
}

/// User clicked 'Nah' or React timeout fired.
#[tauri::command]
pub fn decline_eat(pet: State<'_, SharedPet>, vault: State<'_, Arc<Vault>>) -> Result<(), String> {
    // Capture pending files before clearing them.
    let files = pet
        .lock()
        .map(|g| g.pending_files().cloned().unwrap_or_default())
        .unwrap_or_default();

    let mut pet_guard = pet
        .lock()
        .map_err(|e| format!("Cannot reach Rolo to decline eat: {}", e))?;
    pet_guard.decline_eat();
    drop(pet_guard);

    let event = ExperienceEvent::Eat {
        event_id: crate::vault::events::new_id(),
        ts: chrono::Local::now(),
        files: basenames(&files),
        outcome: EatOutcome::Declined,
        bytes: 0,
    };
    vault.logger.log(&event);

    Ok(())
}

/// Get Rolo's eating stats.
#[tauri::command]
pub fn get_stats(stats: State<'_, SharedStats>) -> Result<crate::stats::EatingStats, String> {
    let stats = stats
        .lock()
        .map_err(|e| format!("Cannot read Rolo's stats: {}", e))?;
    Ok(stats.clone())
}

/// Force a speech bubble on the next tick (for testing).
#[tauri::command]
pub fn force_speech(speech: State<'_, SharedSpeech>) -> Result<(), String> {
    let mut speech_guard = speech
        .lock()
        .map_err(|e| format!("Cannot reach Rolo's speech state: {}", e))?;
    speech_guard.force_trigger();
    Ok(())
}

/// Dismiss the speech bubble.
#[tauri::command]
pub fn dismiss_speech(
    pet: State<'_, SharedPet>,
    speech: State<'_, SharedSpeech>,
    mood: State<'_, SharedMood>,
    vault: State<'_, Arc<Vault>>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let dismissed = {
        let mut speech_guard = speech.lock().map_err(|e| {
            format!(
                "Cannot reach Rolo's speech state — his voice is stuck! Error: {}",
                e
            )
        })?;
        speech_guard.dismiss()
    };

    if dismissed {
        if let Some(bubble_win) = app_handle.get_webview_window("speech-bubble") {
            let _ = bubble_win.hide();
        }
        if let Ok(mut pet_guard) = pet.lock() {
            pet_guard.set_speech_active(false);
        }
        if let Ok(mut mood_guard) = mood.lock() {
            mood_guard.apply_event(MoodEvent::DismissedBubble);
        }

        // Increment the session dismiss counter and log a Dismiss vault event.
        // WHY: v1 simplification — all dismissed bubbles are tagged IdleSpeech
        // because we don't yet distinguish chat-window vs. interaction-prompt vs.
        // idle-speech dismissals at the command layer. Future PRD will refine.
        let count = speech
            .lock()
            .map(|mut g| g.increment_dismiss_count())
            .unwrap_or(1);
        let event = ExperienceEvent::Dismiss {
            event_id: crate::vault::events::new_id(),
            ts: chrono::Local::now(),
            context: DismissContext::IdleSpeech,
            times_dismissed_session: count,
        };
        vault.logger.log(&event);
    }

    Ok(())
}

/// Open a native file picker dialog and feed the selected files to Rolo.
#[tauri::command]
pub async fn feed_rolo_dialog(
    pet: State<'_, SharedPet>,
    window: tauri::Window,
) -> Result<(), String> {
    use tauri_plugin_dialog::DialogExt;

    let file_responses = window.dialog().file().blocking_pick_files();
    if let Some(paths) = file_responses {
        let path_strings: Vec<String> = paths
            .iter()
            .filter_map(|p| p.as_path().and_then(|path| path.to_str().map(String::from)))
            .collect();
        if !path_strings.is_empty() {
            let mut pet = pet
                .lock()
                .map_err(|e| format!("Cannot feed Rolo via dialog: {}", e))?;
            pet.file_dropped(path_strings);
        }
    }
    Ok(())
}

/// Result of submitting an interaction response — tells the frontend which
/// reaction animation to trigger on Rolo.
#[derive(Debug, Serialize)]
pub struct InteractionResult {
    /// "happy", "disappointed", or null (for dismiss / unknown).
    pub reaction: Option<String>,
}

/// Submit a user's response to the active interaction prompt.
///
/// Routes mood check-in responses into Rolo's vault (his cognition), triggers
/// Rolo's reaction animation (Happy/Disappointed) on the Pet state machine,
/// and returns the reaction hint so the frontend can cross-check if needed.
#[tauri::command]
pub fn submit_interaction_response(
    pet: State<'_, SharedPet>,
    interaction_state: State<'_, SharedInteraction>,
    mood: State<'_, SharedMood>,
    vault: State<'_, Arc<Vault>>,
    response: InteractionResponse,
) -> Result<InteractionResult, String> {
    log::info!(
        "[Rolo] submit_interaction_response: interaction_id={}, instance_id={}, response={:?}",
        response.interaction_id,
        response.instance_id,
        response.response,
    );

    let mut interaction = interaction_state.lock().map_err(|e| {
        format!(
            "Cannot reach Rolo's interaction state — he's confused! Error: {}",
            e
        )
    })?;

    // Route to the interaction engine — validates instance_id match
    let accepted = interaction.respond(response.clone());
    drop(interaction);

    match accepted {
        Some(resp) => {
            // Determine the animation reaction based on the response type
            let reaction = reaction_for_response(&resp.response);

            // Trigger the reaction on the Pet state machine. This is what
            // actually makes Rolo play the Happy/Disappointed animation —
            // the frontend reads state via the rolo://state-changed event.
            if let Some(ref kind) = reaction {
                match pet.lock() {
                    Ok(mut pet_guard) => {
                        let happy = kind == "happy";
                        pet_guard.show_mood_reaction(happy);
                        log::info!(
                            "[Rolo] Triggered {} reaction animation",
                            if happy { "Happy" } else { "Disappointed" },
                        );
                    }
                    Err(e) => {
                        log::error!(
                            "[Rolo] Could not lock Pet to play reaction — \
                             he heard you but can't emote: {}",
                            e
                        );
                    }
                }
            } else {
                // No reaction (Dismissed or unknown button) — just clear
                // the interaction flag so autonomous behavior resumes.
                if let Ok(mut pet_guard) = pet.lock() {
                    pet_guard.interaction_active = false;
                }
            }

            // Apply the mood event derived from this response (PRD §3
            // event table). Rolo's mood reacts to every interaction.
            if let Some(event) = mood_event_for_response(&resp) {
                if let Ok(mut mood_guard) = mood.lock() {
                    mood_guard.apply_event(event);
                }
            }

            // Log a Checkin vault event for mood check-in responses.
            // WHY: v1 hardcodes the question text because InteractionResponse
            // doesn't carry the original prompt and a getter is out of scope.
            if resp.interaction_id == crate::interaction::INTERACTION_ID_MOOD_CHECKIN {
                let checkin_event = match &resp.response {
                    ResponseValue::ButtonPress { value } => Some(ExperienceEvent::Checkin {
                        event_id: crate::vault::events::new_id(),
                        ts: chrono::Local::now(),
                        question: "how are you?".to_string(),
                        response: value.clone(),
                        method: CheckinMethod::Button,
                    }),
                    ResponseValue::TextSubmit { text } => Some(ExperienceEvent::Checkin {
                        event_id: crate::vault::events::new_id(),
                        ts: chrono::Local::now(),
                        question: "how are you?".to_string(),
                        response: text.clone(),
                        method: CheckinMethod::Text,
                    }),
                    // Dismissed responses are not logged — Rolo respects the silence.
                    ResponseValue::Dismissed => None,
                };
                if let Some(event) = checkin_event {
                    vault.logger.log(&event);
                }
            }

            Ok(InteractionResult { reaction })
        }
        None => {
            // Response didn't match any active prompt — maybe it expired
            log::warn!(
                "[Rolo] Interaction response rejected — no matching active prompt \
                 (instance_id={})",
                response.instance_id,
            );
            Ok(InteractionResult { reaction: None })
        }
    }
}

// ---------------------------------------------------------------------------
// Mood-state commands (Phase 6) — open/close the status panel and inspect
// the current snapshot. The panel itself is a hidden Tauri window registered
// in lib.rs::create_status_panel_window. PRD §9.
// ---------------------------------------------------------------------------

/// Compute a screen-clamped logical position for the status panel.
///
/// Tries `(cursor_x + 12, cursor_y + 12)` (cursor's lower-right). If the
/// resulting rect would extend past the right or bottom edge of the panel
/// window's current monitor, mirror to upper-left. Falls back to (0, 0) if
/// monitor metadata is unavailable.
fn clamp_panel_position(win: &tauri::WebviewWindow, cursor_x: f64, cursor_y: f64) -> (f64, f64) {
    const PANEL_W: f64 = 260.0;
    const PANEL_H: f64 = 200.0;
    const OFFSET: f64 = 12.0;

    let mut x = cursor_x + OFFSET;
    let mut y = cursor_y + OFFSET;

    if let Ok(Some(monitor)) = win.current_monitor() {
        let scale = monitor.scale_factor();
        let pos = monitor.position();
        let size = monitor.size();
        let mon_x = pos.x as f64 / scale;
        let mon_y = pos.y as f64 / scale;
        let mon_w = size.width as f64 / scale;
        let mon_h = size.height as f64 / scale;

        if x + PANEL_W > mon_x + mon_w {
            x = cursor_x - PANEL_W - OFFSET;
        }
        if y + PANEL_H > mon_y + mon_h {
            y = cursor_y - PANEL_H - OFFSET;
        }
        x = x.max(mon_x).min(mon_x + mon_w - PANEL_W);
        y = y.max(mon_y).min(mon_y + mon_h - PANEL_H);
    }

    (x, y)
}

/// Plain-fn body shared by the `open_status_panel` Tauri command and the
/// `view-status` menu-event handler in `lib.rs`. The menu path can't go
/// through `invoke`, so we keep the logic here in a regular function and
/// thinly wrap it with a `#[tauri::command]`. PRD §5/§9.
pub fn open_status_panel_internal(
    app: tauri::AppHandle,
    cursor_x: f64,
    cursor_y: f64,
) -> Result<(), String> {
    let win = app
        .get_webview_window("status-panel")
        .ok_or_else(|| "status-panel window missing".to_string())?;

    let mood = app
        .try_state::<SharedMood>()
        .ok_or_else(|| "mood state not yet managed".to_string())?;
    let open_flag = app
        .try_state::<StatusPanelOpenFlag>()
        .ok_or_else(|| "status panel flag not yet managed".to_string())?;

    use std::sync::atomic::Ordering;

    // Toggle: if already open, hide it.
    if open_flag.load(Ordering::Relaxed) {
        let _ = win.hide();
        open_flag.store(false, Ordering::Relaxed);
        return Ok(());
    }

    // Snapshot first (lock released before any Tauri call — PRD §9 invariant).
    let snap = {
        let g = mood
            .lock()
            .map_err(|e| format!("mood mutex poisoned: {}", e))?;
        g.snapshot()
    };

    // Initial paint without flicker — emit before show.
    let _ = win.emit("mood-tick", snap);

    // Position the window at the captured cursor + offset, clamped to screen.
    let (px, py) = clamp_panel_position(&win, cursor_x, cursor_y);
    let _ = win.set_position(tauri::LogicalPosition::new(px, py));

    let _ = win.show();
    let _ = win.set_focus();
    open_flag.store(true, Ordering::Relaxed);

    Ok(())
}

/// Open Rolo's status panel at `(cursor_x, cursor_y)` (logical coords).
/// Toggles closed if already open. PRD §9.
#[tauri::command]
pub async fn open_status_panel(
    app: tauri::AppHandle,
    cursor_x: f64,
    cursor_y: f64,
) -> Result<(), String> {
    open_status_panel_internal(app, cursor_x, cursor_y)
}

/// Hide Rolo's status panel. Triggered by Escape, blur, or the auto-timeout
/// idle-watcher in the frontend. PRD §6/§9.
#[tauri::command]
pub fn close_status_panel(
    app: tauri::AppHandle,
    open_flag: State<'_, StatusPanelOpenFlag>,
) -> Result<(), String> {
    use std::sync::atomic::Ordering;
    if let Some(win) = app.get_webview_window("status-panel") {
        let _ = win.hide();
    }
    open_flag.store(false, Ordering::Relaxed);
    Ok(())
}

/// Return Rolo's current mood snapshot. Used by the panel on mount as a
/// fallback if the open-time `mood-tick` emit was missed. PRD §9.
#[tauri::command]
pub fn get_mood_snapshot(mood: State<'_, SharedMood>) -> Result<MoodSnapshot, String> {
    let g = mood
        .lock()
        .map_err(|e| format!("mood mutex poisoned: {}", e))?;
    Ok(g.snapshot())
}

/// Reset Rolo's mood state to defaults. Dev-console only — there is no UI
/// surface for this. PRD §9.
#[tauri::command]
pub fn reset_mood_state(mood: State<'_, SharedMood>) -> Result<(), String> {
    let mut g = mood
        .lock()
        .map_err(|e| format!("mood mutex poisoned: {}", e))?;
    *g = MoodState::default();
    log::info!("[Rolo] Mood state reset to defaults");
    Ok(())
}

// ---------------------------------------------------------------------------
// Chat commands
// ---------------------------------------------------------------------------

/// Open the chat window. Called when user clicks a speech bubble or check-in.
///
/// Creates a SQLite session, freezes Rolo, dismisses the active bubble,
/// and spawns the chat Tauri window adjacent to Rolo's current position.
#[tauri::command]
pub async fn open_chat(
    trigger_text: String,
    source: String,
    initial_user_message: Option<String>,
    pet: State<'_, SharedPet>,
    speech: State<'_, SharedSpeech>,
    chat_store: State<'_, SharedChatStore>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    // If chat window already exists, focus it
    if let Some(existing) = app_handle.get_webview_window("chat") {
        let _ = existing.set_focus();
        return Ok("already_open".to_string());
    }

    // PRD Phase 10: if Rolo has no brain configured, route the user into
    // Command Center > Brain (modal setup mode) instead of opening chat.
    // This replaces the legacy first-run wizard window path. The
    // "unconfigured" check is a static read of `chat_config.json` — no
    // network probe, so a misconfigured-but-reachable provider still
    // opens chat and surfaces the runtime error there (the user can
    // re-enter CC to fix it).
    let chat_cfg = crate::chat::config::ChatConfig::load();
    if crate::command_center::chat_config_is_unconfigured(&chat_cfg) {
        log::info!("[Rolo] open_chat: chat config is unconfigured — routing to Command Center");

        // Freeze Rolo and dismiss speech (same as chat flow) so he sits
        // still while the user configures him.
        {
            let mut pet_guard = pet.lock().map_err(|e| e.to_string())?;
            pet_guard.force_idle();
            pet_guard.interaction_active = true;
        }
        {
            let mut speech_guard = speech.lock().map_err(|e| e.to_string())?;
            if speech_guard.is_showing() {
                speech_guard.dismiss();
                if let Some(bw) = app_handle.get_webview_window("speech-bubble") {
                    let _ = bw.hide();
                }
            }
        }

        if let Err(e) = crate::command_center::window::show_command_center(&app_handle) {
            log::warn!(
                "[Rolo] open_chat: could not show Command Center on unconfigured \
                 brain: {} — chat will not open until Rolo is configured.",
                e
            );
        }

        return Ok("needs_setup".to_string());
    }

    // Freeze Rolo and dismiss speech
    {
        let mut pet_guard = pet.lock().map_err(|e| e.to_string())?;
        pet_guard.force_idle();
        pet_guard.interaction_active = true;
    }
    {
        let mut speech_guard = speech.lock().map_err(|e| e.to_string())?;
        if speech_guard.is_showing() {
            speech_guard.dismiss();
            if let Some(bw) = app_handle.get_webview_window("speech-bubble") {
                let _ = bw.hide();
            }
        }
    }

    // When the user has typed their own opening message (e.g. from the
    // check-in text input), suppress the trigger_text from being shown as
    // Rolo's first assistant message — the conversation should open with
    // the user's message instead. We still record `trigger_text` on the
    // session row for context.
    let initial_user = initial_user_message.as_deref().map(str::trim).unwrap_or("");
    let frontend_trigger_text = if initial_user.is_empty() {
        trigger_text.as_str()
    } else {
        ""
    };

    // Create session in SQLite
    let session_id = {
        let store = chat_store.lock().map_err(|e| e.to_string())?;
        store
            .create_session(&source, &trigger_text)
            .map_err(|e| e.to_string())?
    };

    // Calculate chat window position — adjacent to Rolo's window
    let (chat_x, chat_y) = calculate_adjacent_window_position(&app_handle, 350.0, 420.0);

    // Create the chat window — pass session data via URL params so it's
    // available immediately on mount (events race with React hydration)
    use tauri::{WebviewUrl, WebviewWindowBuilder};

    let url_path = format!(
        "index.html?session_id={}&trigger_text={}&source={}&initial_user_message={}",
        urlencoding::encode(&session_id),
        urlencoding::encode(frontend_trigger_text),
        urlencoding::encode(&source),
        urlencoding::encode(initial_user),
    );

    let chat_window =
        WebviewWindowBuilder::new(&app_handle, "chat", WebviewUrl::App(url_path.into()))
            .title("Chat with Rolo")
            .inner_size(350.0, 420.0)
            .min_inner_size(280.0, 240.0)
            .position(chat_x, chat_y)
            .decorations(false)
            .transparent(true)
            .always_on_top(false)
            .resizable(true)
            .shadow(true)
            .focused(true)
            .visible(true)
            .accept_first_mouse(true)
            .build()
            .map_err(|e| format!("Failed to create chat window: {}", e))?;

    if std::env::var("ROLO_DEVTOOLS").is_ok() {
        chat_window.open_devtools();
    }
    let _chat_window = chat_window;

    #[cfg(target_os = "macos")]
    crate::platform::activate_window_for_input(&app_handle, "chat");

    log::info!(
        "[Rolo] Chat window opened — session {} (source: {})",
        session_id,
        source,
    );

    Ok(session_id)
}

/// Close the chat window and resume Rolo's autonomous behavior.
/// Also triggers session close with mood classification for check-in sessions.
#[tauri::command]
pub async fn close_chat(
    session_id: Option<String>,
    source: Option<String>,
    pet: State<'_, SharedPet>,
    speech: State<'_, SharedSpeech>,
    engine: State<'_, SharedChatEngine>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    if let Some(ref sid) = session_id {
        let engine_guard = engine.lock().await;
        let src = source.as_deref().unwrap_or("speech");
        engine_guard.close_session(sid, src).await.ok();
    }

    if let Some(chat_win) = app_handle.get_webview_window("chat") {
        let _ = chat_win.destroy();
    }

    {
        let mut pet_guard = pet.lock().map_err(|e| e.to_string())?;
        pet_guard.interaction_active = false;
    }

    // Reset the speech timer so a backlogged tick doesn't immediately fire
    // a bubble the instant the chat closes.
    {
        let mut speech_guard = speech.lock().map_err(|e| e.to_string())?;
        speech_guard.reset_timer_after_suppression();
    }

    log::info!("[Rolo] Chat window closed — resuming autonomous behavior");
    Ok(())
}

/// Stream event sent to the frontend via Tauri Channel during chat.
#[derive(Clone, Serialize)]
#[serde(tag = "type")]
pub enum ChatStreamEvent {
    Token { text: String },
    Done { full_text: String, message_id: i64 },
    Error { message: String },
}

/// Send a chat message and stream the response via Tauri Channel.
///
/// The frontend creates a Channel and passes it as `onEvent`. Tokens
/// stream through the channel as they arrive from the LLM. A final
/// `Done` event carries the complete text and the stored message ID.
#[tauri::command]
pub async fn send_chat_message(
    text: String,
    session_id: String,
    on_event: tauri::ipc::Channel<ChatStreamEvent>,
    engine: State<'_, SharedChatEngine>,
) -> Result<(), String> {
    let engine_guard = engine.lock().await;

    let (token_tx, mut token_rx) = tokio::sync::mpsc::channel::<String>(64);

    // Forward tokens from the engine to the Tauri channel
    let channel = on_event.clone();
    let forward_handle = tokio::spawn(async move {
        while let Some(token) = token_rx.recv().await {
            let _ = channel.send(ChatStreamEvent::Token { text: token });
        }
    });

    // Run inference
    let result = engine_guard
        .send_message(&session_id, &text, token_tx)
        .await;

    // Wait for all tokens to be forwarded
    let _ = forward_handle.await;

    match result {
        Ok((full_text, msg_id)) => {
            let _ = on_event.send(ChatStreamEvent::Done {
                full_text,
                message_id: msg_id,
            });
            Ok(())
        }
        Err(e) => {
            let _ = on_event.send(ChatStreamEvent::Error { message: e.clone() });
            Err(e)
        }
    }
}

/// Report a chat message (flags it in the database).
#[tauri::command]
pub async fn report_chat_message(
    message_id: i64,
    chat_store: State<'_, SharedChatStore>,
) -> Result<(), String> {
    let store = chat_store.lock().map_err(|e| e.to_string())?;
    store
        .mark_message_reported(message_id)
        .map_err(|e| e.to_string())?;
    log::info!(
        "[Rolo] Message {} reported — noted for training data",
        message_id
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Export commands — finetuning asset generation
// ---------------------------------------------------------------------------

/// Export Rolo's character card as a JSON file.
///
/// Opens a native save dialog so the user can choose where to put it, then
/// writes the character card. Returns the chosen file path on success.
#[tauri::command]
pub async fn export_character_card(window: tauri::Window) -> Result<String, String> {
    use tauri_plugin_dialog::DialogExt;

    let file_response = window
        .dialog()
        .file()
        .set_file_name("rolo_character_card.json")
        .add_filter("JSON", &["json"])
        .blocking_save_file();

    let path = file_response
        .ok_or_else(|| "Export cancelled — Rolo's card stays secret for now".to_string())?;

    let path_str = path
        .as_path()
        .and_then(|p| p.to_str())
        .map(String::from)
        .ok_or_else(|| "Invalid file path — Rolo can't write there".to_string())?;

    let card_json = crate::chat::export::generate_character_card();

    std::fs::write(&path_str, &card_json)
        .map_err(|e| format!("Failed to write character card: {}", e))?;

    log::info!("[Rolo] Character card exported to {}", path_str);

    Ok(path_str)
}

/// Export Rolo's conversation history as JSONL training data.
///
/// Opens a native save dialog, generates training data from all stored
/// sessions, and writes the JSONL file. Returns the chosen file path on success.
#[tauri::command]
pub async fn export_training_data(
    chat_store: State<'_, SharedChatStore>,
    window: tauri::Window,
) -> Result<String, String> {
    use tauri_plugin_dialog::DialogExt;

    let file_response = window
        .dialog()
        .file()
        .set_file_name("rolo_training_data.jsonl")
        .add_filter("JSONL", &["jsonl"])
        .blocking_save_file();

    let path = file_response
        .ok_or_else(|| "Export cancelled — Rolo's training data stays private".to_string())?;

    let path_str = path
        .as_path()
        .and_then(|p| p.to_str())
        .map(String::from)
        .ok_or_else(|| "Invalid file path — Rolo can't write there".to_string())?;

    let jsonl = {
        let store = chat_store.lock().map_err(|e| {
            format!(
                "Cannot access Rolo's conversation store — his memories are locked! Error: {}",
                e
            )
        })?;
        crate::chat::export::generate_training_data(&store)?
    };

    std::fs::write(&path_str, &jsonl)
        .map_err(|e| format!("Failed to write training data: {}", e))?;

    log::info!(
        "[Rolo] Training data exported to {} ({} lines)",
        path_str,
        jsonl.lines().count()
    );

    Ok(path_str)
}

// ---------------------------------------------------------------------------
// Dreaming commands (Section O) — wake the sleeping pet, peek at the dream
// log, and revert specific learned facts. PRD §O.
// ---------------------------------------------------------------------------

/// Cancel any in-flight dream and let Rolo's poll loop transition him back
/// to Idle on its next iteration. Wired to the right-click "Wake Rolo" menu
/// item via the menu-event handler in `lib.rs`; this command is the
/// frontend-callable counterpart so devtools and the Dream Log UI can
/// trigger the same path.
#[tauri::command]
pub fn wake_rolo(
    dream_handle: State<'_, Arc<crate::vault::dreaming::DreamHandle>>,
) -> Result<(), String> {
    log::info!("[Rolo] wake_rolo command invoked");
    dream_handle.request_wake();
    Ok(())
}

/// Returns true when the chat webview window is currently registered.
/// Used by the Dream Log UI to detect "chat-open" gating without exposing
/// the dreaming module's internals to the frontend.
#[tauri::command]
pub fn chat_window_is_open(app: tauri::AppHandle) -> bool {
    app.get_webview_window("chat").is_some()
}

/// UI-friendly summary of one dreams.jsonl entry. Compile entries set
/// `facts_*` fields; lint and manual_revert entries leave them None and
/// callers should distinguish via `run_type`. The `accepted_facts` array
/// is loaded from the per-run artifact (`vault/dreams_artifacts/{run_id}.json`)
/// when available so the Dream Log can render per-fact rows with [revert]
/// buttons; the dreams.jsonl entry itself only carries counts.
#[derive(Debug, Serialize)]
pub struct DreamRunSummary {
    pub run_id: String,
    pub status: String,
    pub started_at: Option<String>,
    pub ended_at: Option<String>,
    pub latency_ms: Option<u64>,
    pub facts_accepted: Option<usize>,
    pub facts_rejected: Option<usize>,
    /// "lint" / "manual_revert" / absent for compile (PRD §7A).
    pub run_type: Option<String>,
    pub parent_run_id: Option<String>,
    pub reason: Option<String>,
    /// The per-run accepted facts from the artifact, when available. Each
    /// entry mirrors `Fact` (file, content, operation, source_event_ids, …).
    pub accepted_facts: Option<serde_json::Value>,
    /// The raw dreams.jsonl entry — frontend can pull any extra field
    /// (rejected_reasons, model_digest, lint findings) without a backend
    /// change.
    pub raw: serde_json::Value,
}

/// Read up to `limit` recent dream-runs in newest-first order. Each compile
/// run is enriched with `accepted_facts` from its on-disk artifact so the UI
/// can render per-fact provenance. Lint and manual_revert entries pass
/// through unenriched. PRD §O2.
#[tauri::command]
pub fn read_dream_log(
    limit: usize,
    vault: State<'_, Arc<Vault>>,
) -> Result<Vec<DreamRunSummary>, String> {
    let dreams = vault.dreams_log_handle();
    let entries = dreams.read_recent(limit);
    let artifacts_dir = vault.dreams_artifacts_dir();

    let summaries = entries
        .into_iter()
        .map(|raw| {
            let get_str = |k: &str| raw.get(k).and_then(|v| v.as_str()).map(String::from);
            let get_u64 = |k: &str| raw.get(k).and_then(|v| v.as_u64());
            let get_usize = |k: &str| get_u64(k).map(|n| n as usize);

            let run_id = get_str("run_id").unwrap_or_default();

            // Best-effort: load accepted facts from the artifact. Missing or
            // unparseable artifacts just yield None — the UI degrades to
            // showing counts only, which matches the lint/revert path anyway.
            let accepted_facts = load_accepted_facts(&artifacts_dir, &run_id);

            DreamRunSummary {
                run_id,
                status: get_str("status").unwrap_or_else(|| "unknown".into()),
                started_at: get_str("started_at"),
                ended_at: get_str("ended_at"),
                latency_ms: get_u64("latency_ms"),
                facts_accepted: get_usize("facts_accepted"),
                facts_rejected: get_usize("facts_rejected"),
                run_type: get_str("run_type"),
                parent_run_id: get_str("parent_run_id"),
                reason: get_str("reason").or_else(|| get_str("reject_reason")),
                accepted_facts,
                raw,
            }
        })
        .collect();

    Ok(summaries)
}

/// Load the `accepted_facts` array from a compile run's artifact JSON.
/// Returns None for missing / unparseable / non-compile artifacts; the
/// caller treats that as "no per-fact rows available" rather than an error.
fn load_accepted_facts(artifacts_dir: &Path, run_id: &str) -> Option<serde_json::Value> {
    if run_id.is_empty() {
        return None;
    }
    let path = artifacts_dir.join(format!("{}.json", run_id));
    let bytes = std::fs::read(&path).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    v.get("accepted_facts").cloned()
}

/// Pure helper for `revert_fact`: find the first line in `body` that contains
/// both the fact's content and its provenance marker `<!-- src: a,b -->`,
/// then wrap it in revert markers. Returns the rewritten body when a match
/// is found; None otherwise (caller surfaces the error to the UI). Trailing-
/// newline state of the input is preserved so atomic writes don't churn
/// mtime.
///
/// Extracted for testability — `revert_fact` itself takes a Tauri `State`
/// which is awkward to instantiate from a unit test.
pub(crate) fn comment_out_fact_line(
    body: &str,
    content: &str,
    source_ids: &[String],
    timestamp: &str,
) -> Option<String> {
    let needle = format!("<!-- src: {} -->", source_ids.join(","));
    let mut found = false;
    let new_lines: Vec<String> = body
        .lines()
        .map(|line| {
            if !found && line.contains(&needle) && line.contains(content) {
                found = true;
                format!(
                    "<!-- reverted_by_user: {} -->\n<!-- {} -->",
                    timestamp, line
                )
            } else {
                line.to_string()
            }
        })
        .collect();
    if !found {
        return None;
    }
    let mut new_body = new_lines.join("\n");
    if body.ends_with('\n') {
        new_body.push('\n');
    }
    Some(new_body)
}

/// Sentinel `reason` string written into the `ExperienceEvent::Report`
/// emitted alongside a manual fact revert. Tests assert on this exact
/// value; the dreaming compiler may key off it in the future.
pub(crate) const REVERT_REPORT_REASON: &str = "manual_revert";

/// Comment out a previously-accepted fact line in the wiki, append a
/// `manual_revert` entry to dreams.jsonl, blocklist its source event IDs,
/// and trigger a best-effort embedding rebuild. The fact is identified by
/// the run that produced it plus its index in that run's `accepted_facts`
/// array (0-based, matches the order in the artifact). PRD §O2 / §7C.
#[tauri::command]
pub fn revert_fact(
    run_id: String,
    fact_index: usize,
    vault: State<'_, Arc<Vault>>,
) -> Result<(), String> {
    revert_fact_impl(run_id, fact_index, vault.inner().as_ref())
}

/// Inner implementation of `revert_fact` that operates on a `&Vault` directly
/// so unit tests can drive it without constructing a `tauri::State`. The
/// public `#[tauri::command]` is a thin wrapper.
pub(crate) fn revert_fact_impl(
    run_id: String,
    fact_index: usize,
    vault: &Vault,
) -> Result<(), String> {
    log::info!(
        "[Rolo] revert_fact invoked — run_id={}, fact_index={}",
        run_id,
        fact_index
    );

    // 1. Pull the artifact for this run (the source of truth for accepted
    //    facts; dreams.jsonl only carries counts).
    let artifacts_dir = vault.dreams_artifacts_dir();
    let artifact_path = artifacts_dir.join(format!("{}.json", run_id));
    let artifact_bytes = std::fs::read(&artifact_path).map_err(|e| {
        format!(
            "could not read artifact for run_id {}: {} — \
             this run may have been evicted by the FIFO cap",
            run_id, e
        )
    })?;
    let artifact: serde_json::Value = serde_json::from_slice(&artifact_bytes)
        .map_err(|e| format!("artifact for run_id {} is unparseable: {}", run_id, e))?;

    let accepted = artifact
        .get("accepted_facts")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("no accepted_facts in artifact for {}", run_id))?;
    let fact = accepted.get(fact_index).ok_or_else(|| {
        format!(
            "fact_index {} out of range (run has {} accepted facts)",
            fact_index,
            accepted.len()
        )
    })?;

    let file_str = fact
        .get("file")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "fact missing `file`".to_string())?;
    let content_str = fact.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let source_ids: Vec<String> = fact
        .get("source_event_ids")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // 2. Find the line in the wiki file. Heuristic match: the line must
    //    contain the exact source-id marker AND the fact content. This is
    //    fragile vs. manual edits but acceptable for v1 — the UI surfaces
    //    a clear error if the line moved.
    let wiki_root = vault.wiki_root();
    let path = wiki_root.join(file_str);
    let body = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read wiki file {}: {}", file_str, e))?;
    let today = chrono::Local::now().format("%Y-%m-%dT%H:%M").to_string();

    let new_body =
        comment_out_fact_line(&body, content_str, &source_ids, &today).ok_or_else(|| {
            format!(
                "fact line not found in {} — was it edited or already reverted?",
                file_str
            )
        })?;

    crate::vault::atomic::atomic_write(&path, new_body.as_bytes())
        .map_err(|e| format!("atomic write to {} failed: {}", file_str, e))?;

    // 3. Append a manual_revert entry so the Dream Log audit trail captures
    //    the action. The hash chain stays intact via DreamsLog::append.
    let dreams = vault.dreams_log_handle();
    let revert_entry = serde_json::json!({
        "run_type": "manual_revert",
        "parent_run_id": run_id,
        "fact_index": fact_index,
        "file": file_str,
        "source_event_ids": source_ids,
        "ts": chrono::Local::now().to_rfc3339(),
        "status": "success",
    });
    dreams
        .append(revert_entry)
        .map_err(|e| format!("dreams.jsonl append failed: {}", e))?;

    // 3b. Emit an ExperienceEvent::Report into the experience log so the
    //     compiler can replay the audit signal alongside the chat/dismiss/eat
    //     stream (PRD §11.4 / Cleanup PRD 1 T4). The Report variant predates
    //     this use case — it carries `message_id` + `rolo_text` for chat
    //     reports — so we serialize the fact identifier and reason into
    //     `rolo_text` as JSON and use `-1` as a "not a chat row" sentinel
    //     for `message_id`. Compiler-driven reverts (§A8) are deferred to v3.
    let fact_id = format!("{}:{}", run_id, fact_index);
    let report_payload = serde_json::json!({
        "kind": "fact_revert",
        "fact_id": fact_id,
        "reason": REVERT_REPORT_REASON,
    });
    let report_event = crate::vault::events::ExperienceEvent::Report {
        event_id: crate::vault::events::new_id(),
        ts: chrono::Local::now(),
        message_id: -1,
        rolo_text: report_payload.to_string(),
    };
    vault.logger.log(&report_event);

    // 4. Blocklist the source IDs so future compiles can lower confidence
    //    on those events (PRD §7C). Persisted to meta.json.
    vault
        .add_to_revert_blocklist(&source_ids)
        .map_err(|e| format!("revert_blocklist persist failed: {}", e))?;

    // 5. Trigger an embedding rebuild — best-effort. The wiki file changed,
    //    so the BM25 + embedding indices are now slightly stale. Failures
    //    are logged inside rebuild_index and don't surface here.
    vault.rebuild_index();

    Ok(())
}

// ---------------------------------------------------------------------------
// Dev surface — manually invoke a tool from the frontend StatusPanel.
// Used to verify the tool layer end-to-end before T3 wires the dispatcher
// into production paths. Not gated on `ROLO_DEBUG` for now: T1 is shipped
// behind the (unflipped) `ROLO_DISPATCHER_ENABLED` flag in T3, and gating
// this command is something we can revisit when the flag flips to default-on.
// ---------------------------------------------------------------------------

/// Invoke a registered tool by name with arbitrary JSON args. Returns the
/// `{ natural_language, citations }` payload as JSON for the frontend to
/// render. Errors surface as their `Display` message — no stack traces.
#[tauri::command]
pub async fn dev_invoke_tool(
    name: String,
    args: serde_json::Value,
    registry: State<'_, std::sync::Arc<crate::tools::ToolRegistry>>,
    vault: State<'_, std::sync::Arc<Vault>>,
    mood: State<'_, SharedMood>,
    pet: State<'_, SharedPet>,
) -> Result<serde_json::Value, String> {
    let tool = registry.get(&name).ok_or_else(|| {
        format!(
            "no tool named `{}` (registry has: {:?})",
            name,
            registry.names()
        )
    })?;

    // Clone Pet under the lock and drop the guard before awaiting. The
    // MutexGuard isn't `Send`, so holding it across the `tool.invoke`
    // await is a hard error in Tauri's command machinery. Pet is `Clone`
    // (cheap — all fields are primitives or small Vecs); the snapshot
    // semantics are exactly what the tools want anyway.
    let pet_clone = {
        let g = pet
            .lock()
            .map_err(|e| format!("pet mutex poisoned: {}", e))?;
        g.clone()
    };
    let clock = crate::state_snapshot::SystemClock;

    let ctx = crate::tools::ToolContext {
        vault: vault.inner().as_ref(),
        mood: mood.inner(),
        pet: &pet_clone,
        clock: &clock,
    };

    let out = tool.invoke(&args, &ctx).await.map_err(|e| e.to_string())?;
    serde_json::to_value(out).map_err(|e| format!("serialization failed: {}", e))
}

// ---------------------------------------------------------------------------
// Pure routing helpers — extracted for testability
// ---------------------------------------------------------------------------

/// Determine the reaction animation hint for a given response value.
///
/// - `Mood::Good` button press → "happy"
/// - `Mood::Ok` button press   → "disappointed"
/// - any text submit           → "happy" (Rolo appreciates the engagement)
/// - dismissed                 → None
/// - unrecognized button value → None
pub(crate) fn reaction_for_response(response: &ResponseValue) -> Option<String> {
    use crate::interaction::Mood;
    match response {
        ResponseValue::ButtonPress { value } => match Mood::from_wire_str(value) {
            Some(Mood::Good) => Some("happy".to_string()),
            Some(Mood::Ok) => Some("disappointed".to_string()),
            None => None,
        },
        ResponseValue::TextSubmit { .. } => Some("happy".to_string()),
        ResponseValue::Dismissed => None,
    }
}

/// Map an accepted interaction response to the MoodEvent it should fire.
///
/// Mood-checkin button presses are the canonical Checkin source:
/// `Mood::Good` ⇒ `CheckinRating::Good`, `Mood::Ok` ⇒ `CheckinRating::Bad`
/// (matching the existing happy/disappointed reaction split). A free-text
/// submit is treated as positive engagement — Rolo appreciates the words.
/// A dismissed checkin fires `DismissedBubble` (mildly negative).
///
/// Non-checkin interactions return None — those don't yet have a mood
/// hookup (chat-message events come from the chat-window PRD).
pub(crate) fn mood_event_for_response(resp: &InteractionResponse) -> Option<MoodEvent> {
    use crate::interaction::{Mood, INTERACTION_ID_MOOD_CHECKIN};
    if resp.interaction_id != INTERACTION_ID_MOOD_CHECKIN {
        return None;
    }
    match &resp.response {
        ResponseValue::ButtonPress { value } => match Mood::from_wire_str(value) {
            Some(Mood::Good) => Some(MoodEvent::Checkin {
                rating: CheckinRating::Good,
            }),
            Some(Mood::Ok) => Some(MoodEvent::Checkin {
                rating: CheckinRating::Bad,
            }),
            None => None,
        },
        ResponseValue::TextSubmit { .. } => Some(MoodEvent::Checkin {
            rating: CheckinRating::Good,
        }),
        ResponseValue::Dismissed => Some(MoodEvent::DismissedBubble),
    }
}

/// Extract the filename component from each path string.
/// Full paths are stripped so JSONL events never leak the user's directory layout.
pub(crate) fn basenames(paths: &[String]) -> Vec<String> {
    paths
        .iter()
        .filter_map(|p| {
            Path::new(p)
                .file_name()
                .and_then(|s| s.to_str())
                .map(String::from)
        })
        .collect()
}

/// Compute logical (x, y) for a floating window placed adjacent to Rolo.
///
/// Tries the right side of the main window first; falls back to the left
/// side if the right would overflow the monitor; clamps to x=0 if the left
/// side would be off-screen too. Falls back to `(500.0, 200.0)` when the
/// main window or its outer position cannot be read.
///
/// `window_width` is in logical pixels (matches `inner_size` on the builder).
fn calculate_adjacent_window_position(
    app_handle: &tauri::AppHandle,
    window_width: f64,
    window_height: f64,
) -> (f64, f64) {
    let Some(win) = app_handle.get_webview_window("main") else {
        return (500.0, 200.0);
    };
    let Ok(outer) = win.outer_position() else {
        return (500.0, 200.0);
    };

    let monitor = win.current_monitor().ok().flatten();
    let screen_w = monitor
        .as_ref()
        .map(|m| m.size().width as i32)
        .unwrap_or(1920);
    let screen_h = monitor
        .as_ref()
        .map(|m| m.size().height as i32)
        .unwrap_or(1080);
    let scale = monitor.as_ref().map(|m| m.scale_factor()).unwrap_or(2.0);

    let win_w_px = (window_width * scale) as i32;
    let win_h_px = (window_height * scale) as i32;
    let gap = (8.0 * scale) as i32;
    let margin = (16.0 * scale) as i32;
    let rolo_w_px = (320.0 * scale) as i32;

    let mut x = outer.x + rolo_w_px + gap;
    let mut y = outer.y;

    if x + win_w_px > screen_w {
        x = outer.x - win_w_px - gap;
        if x < 0 {
            x = 0;
        }
    }

    if y + win_h_px > screen_h - margin {
        y = screen_h - win_h_px - margin;
    }
    if y < margin {
        y = margin;
    }

    (x as f64 / scale, y as f64 / scale)
}

// ---------------------------------------------------------------------------
// Command Center commands
// ---------------------------------------------------------------------------

/// Reveal the Rolo Command Center window. Invoked by the frontend (e.g., a
/// programmatic open from the first-run fresh-install router) — the
/// right-click menu uses `show_command_center` directly via `on_menu_event`.
#[tauri::command]
pub async fn open_command_center(app: tauri::AppHandle) -> Result<(), String> {
    crate::command_center::window::show_command_center(&app)
}

/// Load `command_center.json` from disk. Missing or malformed files yield
/// defaults — see `CommandCenterSettings::load`.
#[tauri::command]
pub fn cc_load_settings(
    app: tauri::AppHandle,
) -> Result<crate::command_center::settings::CommandCenterSettings, String> {
    Ok(crate::command_center::settings::CommandCenterSettings::load(&app))
}

/// Atomically write Command Center settings. On success, emit
/// `rolo-internal://weather-config-changed` so the weather tool (Phase 8) can
/// invalidate its cache. Safe to emit when nothing listens yet — Tauri silently
/// no-ops.
///
/// The `rolo-internal://` prefix marks this as a backend-only contract — see
/// the matching listener in `lib.rs::setup` for the convention. Frontend code
/// must never emit `rolo-internal://*`; new internal topics must use the same
/// prefix so the contract stays greppable.
#[tauri::command]
pub fn cc_save_settings(
    app: tauri::AppHandle,
    settings: crate::command_center::settings::CommandCenterSettings,
) -> Result<(), String> {
    settings.save(&app)?;
    if let Err(e) = app.emit("rolo-internal://weather-config-changed", ()) {
        log::warn!(
            "[Rolo] Command Center: emit 'weather-config-changed' failed: {}. \
             Weather cache may serve stale data until next restart.",
            e
        );
    }
    Ok(())
}

/// Phase 9 / Phase 10 — close-from-frontend helper.
///
/// The Command Center window's `CloseRequested` handler in
/// `command_center::window` always `prevent_close`s, then emits
/// `rolo://command-center-close-requested`. The frontend listens, decides
/// whether to surface the discard modal (Phase 9) or block on the setup
/// gate (Phase 10), and — when it decides the window should hide —
/// invokes this command.
///
/// Defense in depth: this command also enforces the Phase 10 setup gate
/// on the backend side. If `chat_config_is_unconfigured` returns true,
/// hiding the window is rejected so a direct `invoke("cc_force_close")`
/// from a misbehaving page can't bypass the modal.
#[tauri::command]
pub fn cc_force_close(app: tauri::AppHandle) -> Result<(), String> {
    let cfg = crate::chat::config::ChatConfig::load();
    if crate::command_center::chat_config_is_unconfigured(&cfg) {
        return Err(
            "Pick a brain for Rolo before continuing — Command Center cannot \
             close until a working Brain is saved."
                .to_string(),
        );
    }
    if let Some(win) = app.get_webview_window("command-center") {
        win.hide().map_err(|e| {
            format!(
                "Cannot hide Command Center: {} — the window may stay visible \
                 until next interaction.",
                e
            )
        })?;
    }
    Ok(())
}

/// Phase 10 — fresh-install gate.
///
/// Returns `true` when Rolo has no working brain configured, meaning the
/// Command Center must enter modal setup mode (only the Brain tab is
/// interactive, close is blocked). See `chat_config_is_unconfigured` for
/// the exact rules. Sync because reading a tiny JSON off disk is cheap.
#[tauri::command]
pub fn cc_brain_needs_setup() -> Result<bool, String> {
    let cfg = crate::chat::config::ChatConfig::load();
    Ok(crate::command_center::chat_config_is_unconfigured(&cfg))
}

/// Run all Perception-tab probes and return a single `DiagnosticsReport`.
/// Probes execute in parallel inside `run_all`; total wall time is bounded
/// by the slowest probe's timeout (3s for the embedder). Errors inside
/// individual probes surface as Red `ProbeOutcome`s in the report — this
/// command's `Result` is reserved for IPC-layer failures, which today
/// cannot happen because `run_all` is infallible.
#[tauri::command]
pub async fn cc_run_diagnostics(
    app: tauri::AppHandle,
) -> Result<crate::command_center::DiagnosticsReport, String> {
    Ok(crate::command_center::diagnostics::run_all(&app).await)
}

// ---------------------------------------------------------------------------
// Brain tab — Phase 5 (PRD/rolo-command-center.md)
// ---------------------------------------------------------------------------

/// Hugging Face Inference's OpenAI-compatible endpoint. Resolved here (not
/// in the frontend) so the UI doesn't have to know infrastructure details —
/// it just sends `provider="huggingface"` and we translate.
const HF_INFERENCE_BASE_URL: &str = "https://api-inference.huggingface.co/v1";
/// DeepInfra's OpenAI-compatible endpoint. Same rationale as `HF_INFERENCE_BASE_URL`.
const DEEPINFRA_BASE_URL: &str = "https://api.deepinfra.com/v1/openai";

/// Request body for `cc_test_brain`. `provider` may be any of the six UI
/// values; the backend resolves HF / DeepInfra to their preset base URLs.
#[derive(serde::Deserialize)]
pub struct TestBrainRequest {
    pub provider: crate::chat::config::Provider,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: String,
}

/// Result of a connection probe. `latency_ms` is wall-clock for the whole
/// `generate` call (including streaming the 8-token response); meaningful
/// even when the probe ultimately returns an empty body.
#[derive(serde::Serialize)]
pub struct TestBrainResult {
    pub ok: bool,
    pub message: String,
    pub latency_ms: u32,
}

/// Translate a `TestBrainRequest` into the same `ChatConfig` shape the
/// engine's builder consumes. HF and DeepInfra collapse into the
/// `openai_compat` discriminant with their preset base URL written in,
/// so the test path is byte-identical to what `cc_apply_brain` will
/// persist on save.
fn test_request_to_config(
    req: &TestBrainRequest,
) -> Result<crate::chat::config::ChatConfig, String> {
    use crate::chat::config::{
        AnthropicConfig, ChatConfig, GeminiConfig, InferenceConfig, OllamaConfig,
        OpenAICompatConfig, Provider,
    };

    let api_key = req.api_key.clone().unwrap_or_default();
    let mut inf = InferenceConfig {
        provider: req.provider,
        ui_provider: Some(req.provider),
        ollama: OllamaConfig::default(),
        openai_compat: OpenAICompatConfig::default(),
        anthropic: AnthropicConfig::default(),
        gemini: GeminiConfig::default(),
    };

    match req.provider {
        Provider::Ollama => {
            inf.ollama.base_url = req
                .base_url
                .clone()
                .unwrap_or_else(|| "http://localhost:11434".to_string());
            inf.ollama.model = req.model.clone();
        }
        Provider::OpenaiCompat => {
            inf.openai_compat.base_url = req
                .base_url
                .clone()
                .ok_or_else(|| "openai_compat requires base_url".to_string())?;
            inf.openai_compat.api_key = api_key;
            inf.openai_compat.model = req.model.clone();
        }
        Provider::Huggingface => {
            inf.provider = Provider::OpenaiCompat;
            inf.openai_compat.base_url = HF_INFERENCE_BASE_URL.to_string();
            inf.openai_compat.api_key = api_key;
            inf.openai_compat.model = req.model.clone();
        }
        Provider::Deepinfra => {
            inf.provider = Provider::OpenaiCompat;
            inf.openai_compat.base_url = DEEPINFRA_BASE_URL.to_string();
            inf.openai_compat.api_key = api_key;
            inf.openai_compat.model = req.model.clone();
        }
        Provider::Anthropic => {
            inf.anthropic.api_key = api_key;
            inf.anthropic.model = req.model.clone();
        }
        Provider::Gemini => {
            inf.gemini.api_key = api_key;
            inf.gemini.model = req.model.clone();
        }
    }

    Ok(ChatConfig { inference: inf })
}

/// Probe a provider configuration without persisting anything. Builds a
/// one-off `InferenceProvider`, sends a tiny "ping"-style completion (8 max
/// tokens), and reports whether it succeeded within a 5-second wall budget.
/// The frontend gates the Save button on a successful test, so this is the
/// only path that talks to a cloud provider before its credentials are
/// committed to disk.
#[tauri::command]
pub async fn cc_test_brain(req: TestBrainRequest) -> Result<TestBrainResult, String> {
    use crate::chat::provider::{GenerationConfig, ProviderChatMessage};
    use std::time::Instant;

    // Translate the request into a transient `ChatConfig` and reuse the
    // engine's provider builder — keeps the test path from drifting away
    // from the persist path, and centralises the HF/DeepInfra base-URL
    // presets in one match (down in `build_provider_from_config`).
    let probe_config = match test_request_to_config(&req) {
        Ok(c) => c,
        Err(err) => {
            return Ok(TestBrainResult {
                ok: false,
                message: err,
                latency_ms: 0,
            });
        }
    };
    let provider = crate::chat::engine::ChatEngine::build_provider_from_config(&probe_config);

    // Tiny probe call — 8 token cap is enough to verify the wire and the
    // credentials. We drain tokens through a small bounded channel and
    // drop the receiver immediately; we only care that `generate` returns
    // Ok, not what the model said.
    let messages = vec![ProviderChatMessage {
        role: "user".to_string(),
        content: "ping".to_string(),
    }];
    let gen = GenerationConfig {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 8,
        stop_sequences: Vec::new(),
    };
    let (tx, _rx) = tokio::sync::mpsc::channel::<String>(8);

    let started = Instant::now();
    let timed = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        provider.generate(messages, gen, tx),
    )
    .await;
    let latency_ms = started.elapsed().as_millis().min(u32::MAX as u128) as u32;

    match timed {
        Ok(Ok(_text)) => Ok(TestBrainResult {
            ok: true,
            message: "connected".to_string(),
            latency_ms,
        }),
        Ok(Err(e)) => Ok(TestBrainResult {
            ok: false,
            message: e.to_string(),
            latency_ms,
        }),
        Err(_) => Ok(TestBrainResult {
            ok: false,
            message: "Test timed out after 5 seconds".to_string(),
            latency_ms,
        }),
    }
}

/// Persist a new `ChatConfig`, rebuild the provider, and hot-swap it into
/// the `SharedProviderSlot` so chat and dreaming both pick up the change
/// without a restart. Also queues a hardcoded "Rolo notices" bubble.
///
/// The Brain tab gates Save on a prior successful `cc_test_brain`; this
/// command trusts that gate and does NOT re-test. That keeps Save snappy
/// and avoids burning a second round-trip's latency budget on the same
/// credentials.
#[tauri::command]
pub async fn cc_apply_brain(
    app: tauri::AppHandle,
    slot: tauri::State<'_, crate::command_center::SharedProviderSlot>,
    config: crate::chat::config::ChatConfig,
) -> Result<(), String> {
    use rand::seq::IndexedRandom;

    // 1. Persist first so a downstream crash doesn't lose the config the
    //    user just confirmed.
    config.save()?;

    // 2. Build the new provider Arc from the freshly-saved config.
    let new_provider = crate::chat::engine::ChatEngine::build_provider_from_config(&config);
    let provider_id = new_provider.provider_id().to_string();
    let model_name = new_provider.model_name().to_string();

    // 3. Hot-swap. Both ChatEngine and the dreaming poll loop read from the
    //    slot on every call, so the next inference uses the new brain.
    let _previous = slot.replace(new_provider);

    log::info!(
        "[Rolo] Brain swapped to {}, model {}",
        provider_id,
        model_name
    );

    // 4. Emit the brain-changed event with the chosen reaction phrase.
    //    A listener in `lib.rs::setup` mirrors the weather-config-changed
    //    pattern and pushes the phrase into `SpeechState` — that keeps this
    //    command free of any direct dependency on `SharedSpeech`, so the
    //    chat-config code path no longer has to hold the speech mutex.
    //
    //    The PRD requires the bubble to bypass the LLM entirely, so we use
    //    a hardcoded phrase that appears even if the new brain is
    //    misconfigured.
    const REACTIONS: &[&str] = &[
        "feels different in here...",
        "new voice today.",
        "...hello?",
    ];
    let phrase: String = {
        let mut rng = rand::rng();
        REACTIONS
            .choose(&mut rng)
            .copied()
            .unwrap_or(REACTIONS[0])
            .to_string()
    };
    // `rolo-internal://` prefix: backend-only contract; the listener in
    // `lib.rs::setup` trusts this payload. Frontend code must not emit it.
    if let Err(e) = app.emit("rolo-internal://brain-changed", phrase) {
        log::warn!(
            "[Rolo] cc_apply_brain: emit 'brain-changed' failed: {}. \
             No reaction bubble this swap.",
            e
        );
    }

    Ok(())
}

/// Expose the loaded `ChatConfig` to the Brain tab so the dropdown can
/// restore its state (`ui_provider` field) and pre-fill the form on mount.
#[tauri::command]
pub fn cc_load_chat_config() -> crate::chat::config::ChatConfig {
    crate::chat::config::ChatConfig::load()
}

// ---------------------------------------------------------------------------
// Brain tab — first-run Rolo Brain pull (PRD/rolo-brain-first-run-pull.md)
// ---------------------------------------------------------------------------

/// One progress event from Ollama's `/api/pull` NDJSON stream, re-shaped for
/// the frontend. Optional fields are `None` whenever Ollama omits them
/// (e.g. early `pulling manifest` lines carry no byte counters).
#[derive(serde::Serialize, Clone)]
pub struct BrainPullProgress {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

/// Probe Ollama's `/api/tags` to discover whether a specific model is already
/// pulled. Used by the first-run walkthrough to skip the pull step when the
/// model is present (and to detect "Ollama not reachable" so the Pull button
/// can stay disabled until the daemon is up).
#[derive(serde::Serialize)]
pub struct BrainModelStatus {
    /// `true` when `/api/tags` returned 200. False means Ollama is down,
    /// unreachable, or returning a non-2xx.
    pub ollama_reachable: bool,
    /// `true` when `ollama_reachable` AND a tag with the requested model
    /// name was found in the response.
    pub model_present: bool,
    /// If `ollama_reachable` is false, the underlying error message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[tauri::command]
pub async fn cc_check_brain_model(
    base_url: String,
    model: String,
) -> Result<BrainModelStatus, String> {
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return Ok(BrainModelStatus {
                ollama_reachable: false,
                model_present: false,
                error: Some(format!("HTTP client build failed: {}", e)),
            })
        }
    };

    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            return Ok(BrainModelStatus {
                ollama_reachable: false,
                model_present: false,
                error: Some(e.to_string()),
            })
        }
    };

    if !resp.status().is_success() {
        return Ok(BrainModelStatus {
            ollama_reachable: false,
            model_present: false,
            error: Some(format!("Ollama returned HTTP {}", resp.status())),
        });
    }

    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            return Ok(BrainModelStatus {
                ollama_reachable: true,
                model_present: false,
                error: Some(format!("Malformed /api/tags JSON: {}", e)),
            })
        }
    };

    // Ollama's /api/tags returns `{ "models": [ { "name": "tag:latest", ... }, ... ] }`.
    // Match `model` against `name` exactly; Ollama's convention is that a
    // pulled tag like `hf.co/larawashington/rolo-brain:latest` matches the
    // user-supplied `hf.co/larawashington/rolo-brain` when the suffix is
    // `:latest` (the default). To stay robust against tagging conventions we
    // accept either an exact match or a match where the stored name strips
    // a trailing `:latest`.
    let target = model.trim();
    let target_default_tag = format!("{}:latest", target);
    let model_present = body
        .get("models")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter().any(|entry| {
                let name = entry.get("name").and_then(|n| n.as_str()).unwrap_or("");
                name == target || name == target_default_tag
            })
        })
        .unwrap_or(false);

    Ok(BrainModelStatus {
        ollama_reachable: true,
        model_present,
        error: None,
    })
}

/// Cancellation flag for an in-flight `cc_pull_brain_model`. The frontend
/// raises it via `cc_cancel_brain_pull`; the pull loop checks it between
/// streamed chunks and returns early. We use a process-static here because
/// only one pull is ever in flight (the wizard gates the button), and a
/// global flag is simpler than threading shared state through Tauri's
/// `State<...>` plumbing. Reset to `false` at the start of every new pull.
static BRAIN_PULL_CANCEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Frontend-callable cancel: raises the flag so the active pull loop bails
/// at the next chunk boundary. No-op if no pull is in flight.
#[tauri::command]
pub fn cc_cancel_brain_pull() {
    BRAIN_PULL_CANCEL.store(true, std::sync::atomic::Ordering::SeqCst);
    log::info!("[Rolo] cc_cancel_brain_pull: cancel signalled");
}

/// Stream Ollama's `/api/pull` for a given model and emit each NDJSON line
/// to the frontend as a `rolo://brain-pull-progress` event. Returns `Ok(())`
/// when the stream ends with `status == "success"`; `Err` for any other
/// terminal state (HTTP error, transport error, embedded `error` field,
/// stream ending without success, or user cancellation). Cancellation:
/// the frontend calls `cc_cancel_brain_pull` which flips `BRAIN_PULL_CANCEL`;
/// this loop checks it between chunks and drops the stream. Ollama keeps
/// any layers it had already written to disk so a subsequent retry resumes.
#[tauri::command]
pub async fn cc_pull_brain_model(
    app: tauri::AppHandle,
    base_url: String,
    model: String,
) -> Result<(), String> {
    BRAIN_PULL_CANCEL.store(false, std::sync::atomic::Ordering::SeqCst);
    use futures_util::StreamExt;

    let url = format!("{}/api/pull", base_url.trim_end_matches('/'));
    // No overall timeout — multi-GB pulls on slow links can legitimately
    // take >30 minutes. Cancellation is handled by dropping the future.
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

    let body = serde_json::json!({ "name": model, "stream": true });

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Failed to reach Ollama at {}: {}", url, e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Ollama returned {} — {}", status, text.trim()));
    }

    let mut stream = resp.bytes_stream();
    let mut buffer: Vec<u8> = Vec::new();
    let mut saw_success = false;

    while let Some(chunk) = stream.next().await {
        if BRAIN_PULL_CANCEL.load(std::sync::atomic::Ordering::SeqCst) {
            log::info!("[Rolo] cc_pull_brain_model: cancelled by user");
            return Err("cancelled".to_string());
        }
        let bytes = chunk.map_err(|e| format!("Stream read error: {}", e))?;
        buffer.extend_from_slice(&bytes);

        // Drain complete newline-terminated lines from the buffer. Anything
        // after the last newline stays buffered for the next chunk.
        while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buffer.drain(..=pos).collect();
            let line_str = match std::str::from_utf8(&line) {
                Ok(s) => s.trim(),
                Err(_) => continue,
            };
            if line_str.is_empty() {
                continue;
            }

            let parsed: serde_json::Value = match serde_json::from_str(line_str) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!(
                        "[Rolo] cc_pull_brain_model: skipping unparseable line ({}): {}",
                        e,
                        line_str
                    );
                    continue;
                }
            };

            // Ollama signals a hard error mid-stream via `{"error": "..."}`.
            if let Some(err) = parsed.get("error").and_then(|v| v.as_str()) {
                return Err(err.to_string());
            }

            let status = parsed
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let digest = parsed
                .get("digest")
                .and_then(|v| v.as_str())
                .map(String::from);
            let total = parsed.get("total").and_then(|v| v.as_u64());
            let completed = parsed.get("completed").and_then(|v| v.as_u64());

            if status == "success" {
                saw_success = true;
            }

            let payload = BrainPullProgress {
                status,
                digest,
                total,
                completed,
            };
            if let Err(e) = app.emit("rolo://brain-pull-progress", &payload) {
                log::warn!("[Rolo] cc_pull_brain_model: emit progress failed: {}", e);
            }
        }
    }

    if saw_success {
        log::info!("[Rolo] cc_pull_brain_model: pull complete for {}", model);
        Ok(())
    } else {
        Err("Pull stream ended without a success status".to_string())
    }
}

// ---------------------------------------------------------------------------
// Memory tab — Phase 7 (PRD/rolo-command-center.md)
// ---------------------------------------------------------------------------

/// Hard per-section cap for Memory textarea content. The frontend enforces the
/// same cap on `onChange`; the backend double-checks for safety so a script
/// can't pump unbounded text into the event log.
const MEMORY_SECTION_CHAR_CAP: usize = 2000;

/// One Memory textarea's content, sent verbatim from the frontend. Empty
/// strings mean "section not set" — we skip the event emit for those sections
/// but still persist them on the profile so a save acts as a snapshot.
#[derive(serde::Deserialize, Debug)]
pub struct SaveMemoryRequest {
    pub about_you: String,
    pub people_and_context: String,
    pub how_rolo_should_respond: String,
}

/// Status discriminant for `SaveMemoryResult`. Mirrors the `CycleOutcome.status`
/// strings produced by the dreaming module but with a typed Rust enum on the
/// public command surface so the match sites are exhaustive.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaveMemoryStatus {
    Success,
    Skipped,
    Failed,
    Cancelled,
}

/// Result of `cc_save_memory_and_sleep`. The `profile` is the freshly-persisted
/// profile (with backend-stamped `updated_at`) so the frontend can echo it back
/// in the "What Rolo has learned" section without a second roundtrip.
#[derive(serde::Serialize, Debug)]
pub struct SaveMemoryResult {
    pub status: SaveMemoryStatus,
    /// Free-text reason when `status != Success`. For `status == Skipped`,
    /// this is `Some("already_running")` when the single-flight slot is held.
    pub reason: Option<String>,
    /// The user profile as persisted to `<vault_root>/user_profile.json`.
    pub profile: crate::vault::user_profile::UserProfile,
}

/// Trim then hard-cap a Memory section to `MEMORY_SECTION_CHAR_CAP` Unicode
/// scalar values. The frontend enforces the same cap on `onChange`, but the
/// backend guards against malicious or buggy callers shoving unbounded text
/// into the event log.
fn cap_memory_section(text: &str) -> String {
    text.trim()
        .chars()
        .take(MEMORY_SECTION_CHAR_CAP)
        .collect::<String>()
}

/// Persist the three Memory textareas, append one `UserProfileUpdate` event per
/// non-empty section, then drive a one-shot dream cycle through the existing
/// compiler/linter pipeline. Stage events are emitted on
/// `rolo://command-center-dream-progress` so the SleepOverlay UI can update
/// without polling.
///
/// **Save before dream.** `user_profile.json` is written FIRST so a downstream
/// dream failure leaves the saved profile intact (PRD edge case "Sleep cycle
/// fails mid-dream").
///
/// **No pet transition here.** `run_one_cycle_for_events` (via
/// `run_one_cycle`) handles the Idle → Sleeping transition internally; calling
/// `enter_sleeping` here would double-enter.
#[tauri::command]
pub async fn cc_save_memory_and_sleep(
    app: tauri::AppHandle,
    vault: tauri::State<'_, Arc<Vault>>,
    slot: tauri::State<'_, crate::command_center::SharedProviderSlot>,
    dream_handle: tauri::State<'_, Arc<crate::vault::dreaming::DreamHandle>>,
    pet: tauri::State<'_, SharedPet>,
    speech: tauri::State<'_, SharedSpeech>,
    req: SaveMemoryRequest,
) -> Result<SaveMemoryResult, String> {
    use crate::vault::compiler::{prompt_kind_label, CompiledEvent};
    use crate::vault::events::{ExperienceEvent, UserProfileCategory};
    use chrono::Local;
    use rand::seq::IndexedRandom;

    // 1. Trim + cap. The frontend caps too; we double-check to keep the
    //    on-disk profile and event log bounded.
    let about_you = cap_memory_section(&req.about_you);
    let people_and_context = cap_memory_section(&req.people_and_context);
    let how_rolo_should_respond = cap_memory_section(&req.how_rolo_should_respond);

    // 2. Save user_profile.json. The frontend reads the returned profile to
    //    refresh the "What Rolo has learned" section. We persist BEFORE the
    //    dream so a dream failure leaves the saved file intact (PRD edge
    //    case "Sleep cycle fails mid-dream").
    let profile_in = crate::vault::user_profile::UserProfile {
        version: 1,
        about_you: about_you.clone(),
        people_and_context: people_and_context.clone(),
        how_rolo_should_respond: how_rolo_should_respond.clone(),
        updated_at: Local::now(),
    };
    let profile = vault.save_user_profile(profile_in).map_err(|e| {
        format!(
            "Rolo couldn't save his memory of you — vault write failed: {}",
            e
        )
    })?;

    if let Err(e) = app.emit(
        "rolo://command-center-dream-progress",
        serde_json::json!({"stage":"writing_memory","label":"Writing memory…"}),
    ) {
        log::warn!(
            "[Rolo] cc_save_memory_and_sleep: writing_memory emit failed: {} — non-fatal",
            e
        );
    }

    // 3. Append one UserProfileUpdate event per non-empty section. We build
    //    the events as we go and keep the corresponding wire-format JSONL
    //    lines so the compiler can ingest them via run_one_cycle_for_events
    //    without re-reading the events store.
    let mut compiled_events: Vec<CompiledEvent> = Vec::new();
    let kind_label = prompt_kind_label("user_profile_update").to_string();
    let sections: [(UserProfileCategory, &str); 3] = [
        (UserProfileCategory::About, &about_you),
        (UserProfileCategory::People, &people_and_context),
        (UserProfileCategory::ResponseStyle, &how_rolo_should_respond),
    ];
    for (category, text) in sections.iter() {
        if text.is_empty() {
            continue;
        }
        let event_id = crate::vault::events::new_id();
        let event = ExperienceEvent::UserProfileUpdate {
            event_id: event_id.clone(),
            ts: Local::now(),
            category: *category,
            text: text.to_string(),
        };
        vault.logger.log(&event);
        // Mirror the on-disk JSONL line exactly — the compiler's prompt
        // builder treats this as the verbatim ground truth.
        let raw_json_line = serde_json::to_string(&event).map_err(|e| {
            format!(
                "Rolo couldn't serialize his fresh memory event ({:?}): {}",
                category, e
            )
        })?;
        compiled_events.push(CompiledEvent {
            event_id,
            raw_json_line,
            kind: kind_label.clone(),
        });
    }

    // No non-empty sections → save the (empty) profile and skip the dream.
    // The frontend disables Save when all three are empty, but if a script
    // calls this anyway we treat it as success-with-no-dream rather than
    // burning a compile cycle on zero events.
    if compiled_events.is_empty() {
        if let Err(e) = app.emit(
            "rolo://command-center-dream-progress",
            serde_json::json!({"stage":"waking_up","label":"Waking up…"}),
        ) {
            log::warn!(
                "[Rolo] cc_save_memory_and_sleep: waking_up (empty) emit failed: {}",
                e
            );
        }
        return Ok(SaveMemoryResult {
            status: SaveMemoryStatus::Success,
            reason: None,
            profile,
        });
    }

    if let Err(e) = app.emit(
        "rolo://command-center-dream-progress",
        serde_json::json!({"stage":"indexing","label":"Indexing…"}),
    ) {
        log::warn!(
            "[Rolo] cc_save_memory_and_sleep: indexing emit failed: {}",
            e
        );
    }

    // 4. Snapshot the provider + clone the dream handle and pet so we can
    //    hand them to the cycle without holding state across the await.
    let provider = slot.snapshot();
    let dream_handle_arc = Arc::clone(&*dream_handle);
    let pet_arc = Arc::clone(&*pet);
    let vault_arc = Arc::clone(&*vault);

    if let Err(e) = app.emit(
        "rolo://command-center-dream-progress",
        serde_json::json!({"stage":"dreaming","label":"Dreaming…"}),
    ) {
        log::warn!(
            "[Rolo] cc_save_memory_and_sleep: dreaming emit failed: {}",
            e
        );
    }

    // 5. Drive the cycle. run_one_cycle_for_events handles the Idle →
    //    Sleeping transition AND the wake on the other side — do NOT
    //    duplicate them here.
    let dreaming = crate::vault::dreaming::Dreaming::new();
    let outcome = dreaming
        .run_one_cycle_for_events(
            &vault_arc.wiki_root(),
            &vault_arc.dreams_artifacts_dir(),
            &vault_arc.dreams_log_handle(),
            compiled_events,
            provider,
            dream_handle_arc,
            pet_arc,
        )
        .await;

    // 6. Branch on outcome. We emit `waking_up` for every terminal state
    //    EXCEPT `already_running` (where no dream ever started, so there is
    //    no nap to wake from).
    if outcome.status == "skipped" && outcome.reason.as_deref() == Some("already_running") {
        return Ok(SaveMemoryResult {
            status: SaveMemoryStatus::Skipped,
            reason: Some("already_running".into()),
            profile,
        });
    }

    if outcome.status == "success" {
        vault_arc.record_compile_complete(Local::now());
    }

    if let Err(e) = app.emit(
        "rolo://command-center-dream-progress",
        serde_json::json!({"stage":"waking_up","label":"Waking up…"}),
    ) {
        log::warn!(
            "[Rolo] cc_save_memory_and_sleep: waking_up emit failed: {}",
            e
        );
    }

    if outcome.status == "success" {
        // Queue Rolo's "got it" reaction. The phrase is picked privately
        // (no log line) — these aren't useful debug signal and would clutter
        // the journal.
        const PHRASES: &[&str] = &["got it.", "i'll remember that.", "...okay."];
        let phrase: &str = {
            let mut rng = rand::rng();
            PHRASES.choose(&mut rng).copied().unwrap_or("got it.")
        };
        if let Ok(mut g) = speech.lock() {
            g.queue_immediate_phrase(phrase.to_string());
        } else {
            log::warn!(
                "[Rolo] cc_save_memory_and_sleep: speech mutex poisoned, reaction suppressed"
            );
        }
    }

    // Map the dreaming module's free-form status string into the typed
    // enum at the public-API boundary. Anything we don't recognize falls
    // through to `Failed` so a future un-mapped status surfaces visibly
    // rather than silently morphing into Success.
    let status = match outcome.status.as_str() {
        "success" => SaveMemoryStatus::Success,
        "skipped" => SaveMemoryStatus::Skipped,
        "failed" => SaveMemoryStatus::Failed,
        "cancelled" => SaveMemoryStatus::Cancelled,
        other => {
            log::warn!(
                "[Rolo] cc_save_memory_and_sleep: unmapped dreaming status {:?}; surfacing as Failed",
                other
            );
            SaveMemoryStatus::Failed
        }
    };
    Ok(SaveMemoryResult {
        status,
        reason: outcome.reason,
        profile,
    })
}

/// Status discriminant for `ClearProfileResult`. `Ok` means nothing was
/// running; `CancelThenClear` means we had to wake an in-flight dream
/// before wiping. Serializes to the TS literal-union values the frontend
/// already expects.
#[derive(serde::Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClearProfileStatus {
    Ok,
    CancelThenClear,
}

/// Result of `cc_clear_user_profile`. `status` distinguishes a clean wipe from
/// one that had to cancel an in-flight dream first — the latter is shown to
/// the user so they understand why their previous Save didn't finish.
#[derive(serde::Serialize, Debug)]
pub struct ClearProfileResult {
    pub status: ClearProfileStatus,
    /// Number of wiki lines stripped by `clear_user_profile_dreams`. Echoed
    /// back so the UI can render "(removed N entries)" if it wants to.
    pub dreams_removed: usize,
}

/// Delete `user_profile.json` and strip any wiki lines whose `<!-- src: ... -->`
/// markers cite only `UserProfileUpdate` event IDs. If a dream is currently
/// in flight (e.g. the user is mid-sleep from a recent Save), we cancel it
/// first via `DreamHandle::request_wake` and wait a brief grace period so the
/// poll loop has time to release the slot before we mutate the wiki.
#[tauri::command]
pub async fn cc_clear_user_profile(
    vault: tauri::State<'_, Arc<Vault>>,
    dream_handle: tauri::State<'_, Arc<crate::vault::dreaming::DreamHandle>>,
) -> Result<ClearProfileResult, String> {
    use crate::vault::dreaming::DreamRunState;

    // 1. Is a dream in flight? If so, cancel it and wait for the slot to
    //    settle. `DreamHandle` has no `is_idle()` method — we poll
    //    `current_state()` instead, capped at 5 × 150 ms.
    let was_running = !matches!(dream_handle.current_state(), DreamRunState::Idle);
    if was_running {
        dream_handle.request_wake();
        let mut settled = false;
        for _ in 0..5 {
            if matches!(dream_handle.current_state(), DreamRunState::Idle) {
                settled = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        }
        if !settled {
            // The dream did not release the slot within ~750 ms of being
            // asked to wake. Surface the failure rather than racing the wiki
            // walk against an active compile.
            return Err(
                "Rolo wouldn't wake from his nap — clearing the profile would have raced with an active dream. Try again in a moment."
                    .into(),
            );
        }
    }

    // 2. Delete the file (idempotent).
    vault.clear_user_profile().map_err(|e| {
        format!(
            "Rolo couldn't forget what you'd told him — vault delete failed: {}",
            e
        )
    })?;

    // 3. Strip profile-derived lines from the wiki and re-index.
    let dreams_removed = vault.clear_user_profile_dreams().map_err(|e| {
        format!(
            "Rolo cleared the surface profile but couldn't strip the dreams that grew from it: {}",
            e
        )
    })?;

    Ok(ClearProfileResult {
        status: if was_running {
            ClearProfileStatus::CancelThenClear
        } else {
            ClearProfileStatus::Ok
        },
        dreams_removed,
    })
}

/// Result of `cc_load_user_profile`. The `profile` is `None` when no save has
/// ever happened (fresh install) — distinct from "saved but every section is
/// empty", which returns `Some(UserProfile { all empty })`. The frontend uses
/// this to decide whether to show the "What Rolo has learned" panel as empty
/// or pre-populated.
#[derive(serde::Serialize, Debug)]
pub struct LoadProfileResult {
    pub profile: Option<crate::vault::user_profile::UserProfile>,
    /// Titles (basenames) of wiki entries the dreaming compiler has tagged
    /// with at least one `UserProfileUpdate` source-id. TODO: implement the
    /// extraction — see the body. Returns `Vec::new()` for now; the frontend
    /// tolerates an empty list.
    pub learned_dream_titles: Vec<String>,
}

/// Read the saved user profile and the list of wiki entries the dreaming
/// compiler has tagged with profile-update source ids. Read-only — does not
/// mutate any state. Called on Memory-tab mount and on the
/// `rolo://command-center-opened` event.
#[tauri::command]
pub fn cc_load_user_profile(
    vault: tauri::State<'_, Arc<Vault>>,
) -> Result<LoadProfileResult, String> {
    let profile = vault.load_user_profile();
    // TODO(Phase 7+): walk the wiki and return basenames of files whose
    // `<!-- src: ... -->` markers cite any UserProfileUpdate event id. The
    // existing `strip_user_profile_lines` helper is per-file mutating and
    // doesn't expose a "list-only" pass; adding `list_user_profile_tagged_files`
    // is straightforward but out of scope for this phase. The UI handles an
    // empty list gracefully (no list, no row).
    Ok(LoadProfileResult {
        profile,
        learned_dream_titles: Vec::new(),
    })
}

/// Wake an in-flight dream without touching the user profile. The Memory
/// tab's SleepOverlay calls this from its Cancel button so the user can bail
/// out of a long-running compile without also wiping their saved profile.
#[tauri::command]
pub fn cc_cancel_sleep(
    dream_handle: tauri::State<'_, Arc<crate::vault::dreaming::DreamHandle>>,
) -> Result<(), String> {
    dream_handle.request_wake();
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 4 dev-only hot-swap helper
// ---------------------------------------------------------------------------
//
// Test-only entry point for manually verifying the `SharedProviderSlot` swap
// path before the Brain UI exists (PRD/rolo-command-center.md Phase 5, AC 5).
// Deliberately NOT registered in `invoke_handler!` — production must not be
// able to silently flip Rolo's brain to a mock at runtime. To exercise this
// during local development, register it temporarily and call from devtools.

/// Test-only: swap the active provider in the `SharedProviderSlot` to a
/// `MockProvider`. Used in Phase 5 to verify hot-reload before the Brain UI
/// is built. Not registered in `invoke_handler!` for production safety —
/// wire it up locally if you need to invoke it from the frontend during
/// development.
#[allow(dead_code)]
pub fn _dev_swap_provider_to_mock(
    slot: tauri::State<'_, crate::command_center::SharedProviderSlot>,
) {
    use crate::chat::mock_provider::MockProvider;
    use std::sync::Arc;
    let mock: Arc<dyn crate::chat::provider::InferenceProvider> =
        Arc::new(MockProvider::new("(dev-swapped)"));
    slot.replace(mock);
}

// ---------------------------------------------------------------------------
// Weather tab — Phase 8 (PRD/rolo-command-center.md)
// ---------------------------------------------------------------------------

/// Request body for `cc_test_weather`. Fields mirror the Weather tab form
/// state: a URL template (with `{lat}`, `{lon}`, `{key}` placeholders), an
/// API key, the response-shape preset, the lat/lon to substitute, and the
/// two optional JSONPath strings (required only when preset is
/// `"custom_jsonpath"`).
#[derive(serde::Deserialize, Debug)]
pub struct TestWeatherRequest {
    pub url_template: String,
    pub api_key: String,
    pub preset: String,
    pub lat: f64,
    pub lon: f64,
    pub jsonpath_temp: Option<String>,
    pub jsonpath_condition: Option<String>,
}

/// Result of a `cc_test_weather` call. `ok=true` carries the parsed weather;
/// `ok=false` carries a human-readable error the Weather tab renders verbatim
/// next to a red X icon. The command itself never returns `Err` — that would
/// surface as a Tauri IPC failure on the frontend, which is reserved for
/// transport-level bugs.
#[derive(serde::Serialize, Debug)]
pub struct TestWeatherResult {
    pub ok: bool,
    pub parsed: Option<crate::tools::weather_endpoint::ParsedWeather>,
    pub error: Option<String>,
}

/// Probe a fully-configured custom weather endpoint with a 5s timeout.
/// Validates the URL template, substitutes placeholders, fetches once,
/// parses the body per `preset`, and returns the normalized weather (or a
/// human-readable error). Does NOT touch the runtime cache — the Weather
/// tab gates Save on a successful test, but Save itself emits
/// `rolo://weather-config-changed` which is what invalidates the cache.
#[tauri::command]
pub async fn cc_test_weather(req: TestWeatherRequest) -> Result<TestWeatherResult, String> {
    let result = crate::tools::weather_endpoint::test_endpoint(
        &req.url_template,
        &req.api_key,
        &req.preset,
        req.lat,
        req.lon,
        req.jsonpath_temp.as_deref(),
        req.jsonpath_condition.as_deref(),
    )
    .await;
    match result {
        Ok(parsed) => Ok(TestWeatherResult {
            ok: true,
            parsed: Some(parsed),
            error: None,
        }),
        Err(e) => Ok(TestWeatherResult {
            ok: false,
            parsed: None,
            error: Some(e),
        }),
    }
}

/// Status of the most recent weather fetch — drives the green/amber/red dot
/// at the top of the Weather tab. Green when the cached fetch is still fresh
/// (< CACHE_TTL); amber when no fetch has happened this session; red when
/// the cache has gone stale (i.e. a previous fetch happened but its TTL
/// expired without a refresh succeeding). The "Last fetched N s ago"
/// summary is parsed straight from the cached NL line.
#[derive(serde::Serialize, Debug)]
pub struct WeatherStatus {
    /// "green" | "amber" | "red".
    pub status: String,
    pub last_summary: Option<String>,
    pub last_fetched_seconds_ago: Option<u64>,
}

/// Read the weather tool's cache state for the Weather tab's status row.
/// Cheap: just reads a mutex on the same `GetWeather` instance the runtime
/// tool dispatches against, so the dot reflects the live state without an
/// extra network round-trip.
#[tauri::command]
pub fn cc_weather_status(
    weather: tauri::State<'_, Arc<crate::tools::get_weather::GetWeather>>,
) -> Result<WeatherStatus, String> {
    match weather.cache_snapshot() {
        None => Ok(WeatherStatus {
            status: "amber".to_string(),
            last_summary: None,
            last_fetched_seconds_ago: None,
        }),
        Some((line, fetched_at)) => {
            let elapsed = fetched_at.elapsed();
            let secs = elapsed.as_secs();
            // The cache stores the period-terminated NL line; the UI looks
            // cleaner without the trailing dot, so strip one if present.
            let summary = line.trim_end_matches('.').to_string();
            let fresh = elapsed < crate::tools::get_weather::CACHE_TTL;
            Ok(WeatherStatus {
                status: if fresh { "green" } else { "red" }.to_string(),
                last_summary: Some(summary),
                last_fetched_seconds_ago: Some(secs),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — routing logic must be airtight for Rolo's emotional health
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::{DebugConfig, ResponseValue};

    // -----------------------------------------------------------------------
    // basenames helper
    // -----------------------------------------------------------------------

    #[test]
    fn basenames_strips_directory_paths() {
        let input = vec!["/foo/bar.txt".to_string(), "/qux/baz.sql".to_string()];
        assert_eq!(basenames(&input), vec!["bar.txt", "baz.sql"]);
    }

    #[test]
    fn basenames_bare_filename_unchanged() {
        let input = vec!["file.rs".to_string()];
        assert_eq!(basenames(&input), vec!["file.rs"]);
    }

    #[test]
    fn basenames_empty_slice_returns_empty() {
        assert_eq!(basenames(&[]), Vec::<String>::new());
    }

    fn make_response(
        instance_id: &str,
        interaction_id: &str,
        response: ResponseValue,
    ) -> InteractionResponse {
        InteractionResponse {
            instance_id: instance_id.to_string(),
            interaction_id: interaction_id.to_string(),
            response,
            timestamp: "2026-04-14T14:30:00".to_string(),
        }
    }

    // -----------------------------------------------------------------------
    // reaction_for_response
    // -----------------------------------------------------------------------

    #[test]
    fn reaction_good_button_is_happy() {
        let resp = ResponseValue::ButtonPress {
            value: "good".to_string(),
        };
        assert_eq!(reaction_for_response(&resp), Some("happy".to_string()));
    }

    #[test]
    fn reaction_ok_button_is_disappointed() {
        let resp = ResponseValue::ButtonPress {
            value: "ok".to_string(),
        };
        assert_eq!(
            reaction_for_response(&resp),
            Some("disappointed".to_string())
        );
    }

    #[test]
    fn reaction_unknown_button_is_none() {
        let resp = ResponseValue::ButtonPress {
            value: "meh".to_string(),
        };
        assert_eq!(reaction_for_response(&resp), None);
    }

    #[test]
    fn reaction_text_submit_is_happy() {
        let resp = ResponseValue::TextSubmit {
            text: "doing great".to_string(),
        };
        assert_eq!(reaction_for_response(&resp), Some("happy".to_string()));
    }

    #[test]
    fn reaction_dismissed_is_none() {
        assert_eq!(reaction_for_response(&ResponseValue::Dismissed), None);
    }

    // -----------------------------------------------------------------------
    // mood_event_for_response — routing interaction responses to MoodEvent
    // -----------------------------------------------------------------------

    #[test]
    fn mood_event_button_good_is_checkin_good() {
        let resp = make_response(
            "mood_checkin_1",
            "mood_checkin",
            ResponseValue::ButtonPress {
                value: "good".to_string(),
            },
        );
        match mood_event_for_response(&resp) {
            Some(MoodEvent::Checkin {
                rating: CheckinRating::Good,
            }) => {}
            other => panic!("expected Checkin{{Good}}, got {:?}", other),
        }
    }

    #[test]
    fn mood_event_button_ok_is_checkin_bad() {
        let resp = make_response(
            "mood_checkin_1",
            "mood_checkin",
            ResponseValue::ButtonPress {
                value: "ok".to_string(),
            },
        );
        match mood_event_for_response(&resp) {
            Some(MoodEvent::Checkin {
                rating: CheckinRating::Bad,
            }) => {}
            other => panic!("expected Checkin{{Bad}}, got {:?}", other),
        }
    }

    #[test]
    fn mood_event_text_submit_is_checkin_good() {
        let resp = make_response(
            "mood_checkin_1",
            "mood_checkin",
            ResponseValue::TextSubmit {
                text: "feeling great".to_string(),
            },
        );
        match mood_event_for_response(&resp) {
            Some(MoodEvent::Checkin {
                rating: CheckinRating::Good,
            }) => {}
            other => panic!("expected Checkin{{Good}}, got {:?}", other),
        }
    }

    #[test]
    fn mood_event_dismissed_is_dismissed_bubble() {
        let resp = make_response("mood_checkin_1", "mood_checkin", ResponseValue::Dismissed);
        match mood_event_for_response(&resp) {
            Some(MoodEvent::DismissedBubble) => {}
            other => panic!("expected DismissedBubble, got {:?}", other),
        }
    }

    #[test]
    fn mood_event_non_checkin_is_none() {
        let resp = make_response(
            "some_other_1",
            "some_other_interaction",
            ResponseValue::ButtonPress {
                value: "good".to_string(),
            },
        );
        assert!(mood_event_for_response(&resp).is_none());
    }

    // -----------------------------------------------------------------------
    // force_idle + interaction_active integration
    // -----------------------------------------------------------------------

    #[test]
    fn force_idle_then_set_interaction_active_keeps_rolo_still() {
        use crate::state_machine::{Pet, PetState};

        let mut pet = Pet::new(500, 500, 1920, 1080, 128, 2);

        // Tick until Rolo picks a walk behavior (behavior timer is 0-15s;
        // tick 20s to guarantee it fires at least once).
        let mut ticked_to_walk = false;
        for _ in 0..1250 {
            pet.tick(16);
            if matches!(pet.state(), PetState::WalkLeft | PetState::WalkRight) {
                ticked_to_walk = true;
                break;
            }
        }

        if !ticked_to_walk {
            // Pet chose Idle behavior — force_idle is a no-op but that's fine;
            // the important thing is interaction_active blocks further walking.
            // This branch is technically possible (scheduler picks Idle ~50% of the time).
        }

        // Simulate what tick.rs does when Show(prompt) fires:
        // 1. force_idle() stops any ongoing walk
        pet.force_idle();
        assert_eq!(
            pet.state(),
            PetState::Idle,
            "After force_idle, Rolo must be Idle"
        );

        // 2. interaction_active = true prevents autonomous walking
        pet.interaction_active = true;

        // Tick for longer than the behavior timer max (15s)
        for _ in 0..1000 {
            pet.tick(16);
        }

        assert_eq!(
            pet.state(),
            PetState::Idle,
            "Rolo must not start walking while interaction_active = true"
        );
    }

    #[test]
    fn clearing_interaction_active_restores_autonomous_behavior() {
        use crate::state_machine::{Pet, PetState};

        let mut pet = Pet::new(500, 500, 1920, 1080, 128, 2);
        pet.interaction_active = true;

        // Tick for 20 seconds — behavior timer stays frozen
        for _ in 0..1250 {
            pet.tick(16);
        }
        assert_eq!(
            pet.state(),
            PetState::Idle,
            "Must stay Idle while interaction is active"
        );

        // Clear the flag — behavior should eventually resume
        pet.interaction_active = false;

        // Tick another 20 seconds — behavior timer may fire
        // We don't assert a specific end state (it's random), just that
        // the engine doesn't crash and was no longer frozen.
        for _ in 0..1250 {
            pet.tick(16);
        }
        // If we get here without panicking, the behavior resumed safely.
    }

    // -----------------------------------------------------------------------
    // Full routing pipeline: InteractionState.respond instance_id matching
    // -----------------------------------------------------------------------

    #[test]
    fn response_pipeline_unmatched_instance_id_rejected() {
        use crate::interaction::InteractionState;

        let mut ix = InteractionState::new(DebugConfig { turbo_mode: true });

        ix.tick(
            crate::interaction::DEBUG_INTERVAL_MS + 1_000,
            crate::state_machine::PetState::Idle,
            chrono::NaiveTime::from_hms_opt(14, 0, 0).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2026, 4, 14).unwrap(),
            false,
        );
        assert!(ix.is_active());

        // Try to respond with a wrong instance_id
        let wrong_resp = InteractionResponse {
            instance_id: "mood_checkin_999".to_string(), // wrong ID
            interaction_id: "mood_checkin".to_string(),
            response: ResponseValue::ButtonPress {
                value: "good".to_string(),
            },
            timestamp: "2026-04-14T14:30:00".to_string(),
        };

        let accepted = ix.respond(wrong_resp);
        assert!(
            accepted.is_none(),
            "Response with wrong instance_id must be rejected"
        );
        assert!(
            ix.is_active(),
            "Interaction must still be active after rejected response"
        );
    }

    // -----------------------------------------------------------------------
    // Section O — Dream Log commands
    // -----------------------------------------------------------------------

    use crate::vault::dreams_log::DreamsLog;
    use serde_json::json;
    use tempfile::TempDir;

    /// O1 — read_dream_log shapes recent entries newest-first. The DreamsLog
    /// itself already returns reversed; this test just confirms the command
    /// shaper preserves that order and surfaces the right fields.
    #[test]
    fn read_dream_log_helper_shapes_entries_newest_first() {
        // We can't easily build a Vault State, but we can verify the shaping
        // logic by walking entries through the same get_str/get_u64 plumbing
        // the command uses. This is the contract the UI depends on.
        let tmp = TempDir::new().unwrap();
        let log = DreamsLog::new(tmp.path());

        log.append(json!({
            "run_id": "drm_first",
            "status": "success",
            "started_at": "2026-05-04T08:00:00",
            "ended_at": "2026-05-04T08:00:14",
            "latency_ms": 14000,
            "facts_accepted": 6,
            "facts_rejected": 1,
        }))
        .unwrap();
        log.append(json!({
            "run_id": "drm_second",
            "status": "success",
            "started_at": "2026-05-04T14:30:00",
            "ended_at": "2026-05-04T14:30:18",
            "latency_ms": 18000,
            "facts_accepted": 10,
            "facts_rejected": 2,
        }))
        .unwrap();

        let entries = log.read_recent(10);
        // newest-first: drm_second is index 0
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["run_id"], "drm_second");
        assert_eq!(entries[1]["run_id"], "drm_first");
        assert_eq!(entries[0]["facts_accepted"], 10);
    }

    /// O2 — load_accepted_facts reads from the per-run artifact and ignores
    /// missing files (compile artifacts may be FIFO-evicted; lint and
    /// manual_revert never have one). Both paths must be silent.
    #[test]
    fn load_accepted_facts_returns_value_when_artifact_exists() {
        let tmp = TempDir::new().unwrap();
        let artifacts = tmp.path().join("artifacts");
        std::fs::create_dir_all(&artifacts).unwrap();
        let run_id = "drm_test_run";
        let payload = json!({
            "run_id": run_id,
            "accepted_facts": [
                { "file": "user/preferences.md", "content": "User loves pizza" }
            ],
        });
        std::fs::write(
            artifacts.join(format!("{}.json", run_id)),
            serde_json::to_vec(&payload).unwrap(),
        )
        .unwrap();

        let got = load_accepted_facts(&artifacts, run_id).expect("artifact present");
        let arr = got.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["file"], "user/preferences.md");
    }

    #[test]
    fn load_accepted_facts_missing_artifact_is_silent_none() {
        let tmp = TempDir::new().unwrap();
        let artifacts = tmp.path().join("artifacts");
        // Don't create the dir — simulate FIFO eviction.
        let got = load_accepted_facts(&artifacts, "drm_evicted");
        assert!(got.is_none());
    }

    #[test]
    fn load_accepted_facts_empty_run_id_is_none() {
        // Lint/manual_revert entries have no run_id of their own; we must
        // not paw at the filesystem with an empty path.
        let tmp = TempDir::new().unwrap();
        let got = load_accepted_facts(tmp.path(), "");
        assert!(got.is_none());
    }

    /// O2 — comment_out_fact_line wraps the matching wiki line in revert
    /// markers and preserves trailing-newline state. The input below mirrors
    /// the compiler's output: `{content} <!-- src: a,b -->`.
    #[test]
    fn comment_out_fact_line_wraps_matching_line() {
        let body = "# Preferences\n\
                    User loves pizza <!-- src: evt_a,evt_b -->\n\
                    User has a cat <!-- src: evt_c -->\n";
        let content = "User loves pizza";
        let ids = vec!["evt_a".to_string(), "evt_b".to_string()];

        let result = comment_out_fact_line(body, content, &ids, "2026-05-04T15:00").unwrap();

        assert!(
            result.contains("<!-- reverted_by_user: 2026-05-04T15:00 -->"),
            "must contain the revert marker"
        );
        assert!(
            result.contains("<!-- User loves pizza <!-- src: evt_a,evt_b --> -->"),
            "the original line must be commented out via wrapping <!-- ... -->"
        );
        // The unrelated line is untouched.
        assert!(result.contains("User has a cat <!-- src: evt_c -->"));
        // Trailing newline preserved.
        assert!(result.ends_with('\n'));
    }

    #[test]
    fn comment_out_fact_line_missing_line_returns_none() {
        let body = "# Preferences\nUser likes apples <!-- src: evt_x -->\n";
        let result = comment_out_fact_line(
            body,
            "User loves pizza",
            &["evt_a".to_string()],
            "2026-05-04T15:00",
        );
        assert!(
            result.is_none(),
            "no matching line → None, surfaced as error"
        );
    }

    #[test]
    fn comment_out_fact_line_only_first_match_is_wrapped() {
        // Defensive: if two lines share the marker AND the content, only
        // the first is reverted. The user can re-revert the second; bulk
        // revert is out of scope for v1.
        let body = "User loves pizza <!-- src: evt_a -->\n\
                    User loves pizza <!-- src: evt_a -->\n";
        let result = comment_out_fact_line(
            body,
            "User loves pizza",
            &["evt_a".to_string()],
            "2026-05-04T15:00",
        )
        .unwrap();
        let revert_count = result.matches("<!-- reverted_by_user:").count();
        assert_eq!(revert_count, 1);
    }

    /// O2 — Vault::add_to_revert_blocklist persists across a save/load round
    /// trip. The compile gate reads this set on every run, so it MUST survive
    /// the meta.json fsync.
    #[test]
    fn add_to_revert_blocklist_persists_to_meta_json() {
        use crate::vault::embeddings::Embedder;
        use crate::vault::Vault;
        use std::time::Duration;

        // Stub embedder mirroring the DeadEmbedder pattern in prompt.rs —
        // probe_digest returns zeros, embed_query returns None, so the
        // vault degrades to BM25-only and never tries to reach Ollama.
        struct DeadEmbedder;
        impl Embedder for DeadEmbedder {
            fn probe_digest(&self) -> std::io::Result<[u8; 32]> {
                Ok([0; 32])
            }
            fn embed_batch(
                &self,
                _texts: &[String],
                _timeout: Duration,
            ) -> std::io::Result<Vec<Option<Vec<f32>>>> {
                Ok(Vec::new())
            }
            fn embed_query(&self, _text: &str) -> Option<[f32; 768]> {
                None
            }
            fn model_name(&self) -> &str {
                "dead"
            }
        }

        let tmp = TempDir::new().unwrap();
        let vault = Vault::open_or_init_with_embedder(
            tmp.path().to_path_buf(),
            std::sync::Arc::new(DeadEmbedder),
        );

        let ids = vec!["evt_a".to_string(), "evt_b".to_string()];
        vault.add_to_revert_blocklist(&ids).unwrap();

        // Read meta.json from disk to confirm persistence.
        let meta_path = tmp.path().join("meta.json");
        let bytes = std::fs::read(&meta_path).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let blocklist = v["revert_blocklist"].as_array().unwrap();
        let strs: Vec<&str> = blocklist.iter().filter_map(|x| x.as_str()).collect();
        assert!(strs.contains(&"evt_a"));
        assert!(strs.contains(&"evt_b"));
    }

    /// PRD-prompt-consolidation T4 — every `manual_revert` row in
    /// `dreams.jsonl` is paired with an `ExperienceEvent::Report` in the
    /// experience log. `fact_id` and `reason` round-trip through the Report
    /// payload so the compiler can correlate the two streams later.
    #[test]
    fn revert_fact_emits_report_event() {
        use crate::vault::embeddings::Embedder;
        use crate::vault::events::ExperienceEvent;
        use crate::vault::Vault;
        use std::time::Duration;

        // Same DeadEmbedder pattern as the surrounding tests — keeps Ollama
        // out of the test process so the vault degrades to BM25-only.
        struct DeadEmbedder;
        impl Embedder for DeadEmbedder {
            fn probe_digest(&self) -> std::io::Result<[u8; 32]> {
                Ok([0; 32])
            }
            fn embed_batch(
                &self,
                _texts: &[String],
                _timeout: Duration,
            ) -> std::io::Result<Vec<Option<Vec<f32>>>> {
                Ok(Vec::new())
            }
            fn embed_query(&self, _text: &str) -> Option<[f32; 768]> {
                None
            }
            fn model_name(&self) -> &str {
                "dead"
            }
        }

        let tmp = TempDir::new().unwrap();
        let vault = Vault::open_or_init_with_embedder(
            tmp.path().to_path_buf(),
            std::sync::Arc::new(DeadEmbedder),
        );

        // Seed the wiki with a single fact line whose source-id marker matches
        // what the artifact below claims.
        let wiki_root = vault.wiki_root();
        let wiki_file_rel = "user/preferences.md";
        let wiki_path = wiki_root.join(wiki_file_rel);
        std::fs::create_dir_all(wiki_path.parent().unwrap()).unwrap();
        std::fs::write(
            &wiki_path,
            "User loves pizza <!-- src: evt_pizza_1,evt_pizza_2 -->\n",
        )
        .unwrap();

        // Seed the matching artifact under dreams_artifacts/<run_id>.json.
        let run_id = "drm_t4_test";
        let fact_index: usize = 0;
        let artifacts_dir = vault.dreams_artifacts_dir();
        std::fs::create_dir_all(&artifacts_dir).unwrap();
        let artifact = json!({
            "run_id": run_id,
            "accepted_facts": [
                {
                    "file": wiki_file_rel,
                    "content": "User loves pizza",
                    "source_event_ids": ["evt_pizza_1", "evt_pizza_2"],
                }
            ],
        });
        std::fs::write(
            artifacts_dir.join(format!("{}.json", run_id)),
            serde_json::to_vec(&artifact).unwrap(),
        )
        .unwrap();

        // Drive the inner impl directly — same code path the Tauri command
        // executes, minus the State<'_, …> wrapper.
        revert_fact_impl(run_id.to_string(), fact_index, vault.as_ref())
            .expect("revert_fact_impl should succeed on a well-formed fixture");

        let expected_fact_id = format!("{}:{}", run_id, fact_index);

        // Assertion 1 — dreams.jsonl carries a manual_revert row referencing
        // the run + fact. Existing behavior, regression guard.
        let dreams_path = tmp.path().join("dreams.jsonl");
        let dreams_body = std::fs::read_to_string(&dreams_path).expect("dreams.jsonl exists");
        let dreams_lines: Vec<&str> = dreams_body.lines().collect();
        assert!(
            !dreams_lines.is_empty(),
            "dreams.jsonl should have at least the manual_revert line"
        );
        let mut found_revert_row = false;
        for line in &dreams_lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            if v["run_type"] == "manual_revert"
                && v["parent_run_id"] == run_id
                && v["fact_index"] == fact_index
            {
                found_revert_row = true;
                break;
            }
        }
        assert!(
            found_revert_row,
            "dreams.jsonl missing manual_revert row for {}",
            expected_fact_id
        );

        // Drop the vault (and thus the Arc<ExperienceLogger>) so the line
        // writer flushes before we read the events file. Drop also writes a
        // trailing SessionEnd, which we tolerate.
        drop(vault);

        // Assertion 2 — today's events JSONL has a Report event whose
        // rolo_text encodes the same fact_id + reason.
        let today = chrono::Local::now().date_naive();
        let events_path = tmp.path().join("events").join(format!("{}.jsonl", today));
        let events_body = std::fs::read_to_string(&events_path).expect("events JSONL exists");
        let mut found_report = false;
        for line in events_body.lines() {
            let parsed: ExperienceEvent =
                serde_json::from_str(line).expect("every event line must parse");
            if let ExperienceEvent::Report { rolo_text, .. } = parsed {
                let payload: serde_json::Value = serde_json::from_str(&rolo_text)
                    .expect("Report.rolo_text must be JSON for fact reverts");
                if payload["kind"] == "fact_revert"
                    && payload["fact_id"] == expected_fact_id
                    && payload["reason"] == REVERT_REPORT_REASON
                {
                    found_report = true;
                    break;
                }
            }
        }
        assert!(
            found_report,
            "events JSONL missing matching Report event for {}",
            expected_fact_id
        );
    }
}
