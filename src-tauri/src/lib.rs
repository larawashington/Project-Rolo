//! Rolo's Tauri application — the wiring layer that brings him to life.
//!
//! This module initializes Rolo's state machine with real screen dimensions,
//! stores it as managed Tauri state, starts his heartbeat (tick loop), and
//! registers all commands the frontend can invoke.
//!
//! Business logic lives in `state_machine.rs` and `speech.rs`.
//! This file is plumbing only.

mod banned;
mod chat;
mod command_center;
mod commands;
mod dev_log;
mod drag_watch;
pub mod geometry;
mod hitmask;
pub mod http;
mod interaction;
mod mood;
pub mod ollama;
pub mod ollama_router;
#[cfg(target_os = "macos")]
mod perception;
mod platform;
mod self_trash_ledger;
pub mod speech;
mod state_machine;
pub mod state_snapshot;
mod stats;
mod tick;
mod tokens;
pub mod tools;
mod trash;
pub mod vault;

use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tauri::{
    DragDropEvent, Emitter, Listener, Manager, PhysicalPosition, RunEvent, WebviewUrl,
    WebviewWindow, WebviewWindowBuilder, WindowEvent,
};

use commands::{
    LastRightClickPos, SharedChatStore, SharedInteraction, SharedMood, SharedPet, SharedSpeech,
    StatusPanelOpenFlag,
};
use interaction::{DebugConfig, InteractionState};
use speech::SpeechState;
use state_machine::Pet;
use stats::{EatingStats, SharedStats};

use geometry::{EDGE_OFFSET, LAYOUT_HYSTERESIS, SPRITE_SIZE, WINDOW_SIZE};

pub const MENU_ID_CHAT_WITH_ROLO: &str = "chat-with-rolo";
pub const MENU_ID_OPEN_COMMAND_CENTER: &str = "open-command-center";
pub const MENU_ID_QUIT: &str = "quit-rolo";

/// Whether the sprite sits at the bottom or top of the 320x320 window.
/// Top-anchored is used when sprite_y < dy so the window never needs a
/// negative y position (which macOS clamps).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LayoutMode {
    Normal,
    TopAnchored,
}

/// Decide whether to use top-anchored layout based on sprite Y position.
pub(crate) fn compute_layout_mode(
    sprite_y: i32,
    scale_factor: f64,
    current_mode: LayoutMode,
) -> LayoutMode {
    let (_, dy) = geometry::sprite_window_offsets(scale_factor);
    match current_mode {
        LayoutMode::Normal => {
            if sprite_y < dy {
                LayoutMode::TopAnchored
            } else {
                LayoutMode::Normal
            }
        }
        LayoutMode::TopAnchored => {
            if sprite_y > dy + LAYOUT_HYSTERESIS {
                LayoutMode::Normal
            } else {
                LayoutMode::TopAnchored
            }
        }
    }
}

/// Query the primary monitor for screen dimensions and scale factor.
/// Returns (screen_width, screen_height, scale_factor) in physical pixels.
fn get_monitor_info(app: &tauri::App) -> Result<(u32, u32, f64), Box<dyn std::error::Error>> {
    let window = app
        .get_webview_window("main")
        .ok_or("Could not find main window — Rolo's home is missing!")?;

    let monitor = window
        .current_monitor()?
        .ok_or("No monitor detected — Rolo can't find a screen to sit on!")?;

    let screen_size = monitor.size();
    let scale_factor = monitor.scale_factor();

    Ok((screen_size.width, screen_size.height, scale_factor))
}

/// Calculate Rolo's starting position — bottom-right corner of the screen.
/// Returns (sprite_x, sprite_y) in physical pixels.
fn calculate_start_position(screen_w: u32, screen_h: u32, scale_factor: f64) -> (i32, i32) {
    let physical_pet = (f64::from(SPRITE_SIZE) * scale_factor) as u32;
    let physical_offset = (f64::from(EDGE_OFFSET) * scale_factor) as u32;

    let x = screen_w.saturating_sub(physical_pet + physical_offset) as i32;
    let y = screen_h.saturating_sub(physical_pet + physical_offset) as i32;

    (x, y)
}

/// Convert sprite position to window position, accounting for layout mode.
/// Normal: sprite at bottom-center of the 320x320 window.
/// TopAnchored: sprite at top-center (used near the top of the screen).
pub(crate) fn sprite_to_window_position(
    sprite_x: i32,
    sprite_y: i32,
    scale_factor: f64,
    layout: LayoutMode,
) -> (i32, i32) {
    let (dx, dy) = geometry::sprite_window_offsets(scale_factor);
    match layout {
        LayoutMode::Normal => (sprite_x - dx, sprite_y - dy),
        LayoutMode::TopAnchored => (sprite_x - dx, sprite_y),
    }
}

#[allow(dead_code)]
pub(crate) fn window_to_sprite_position(
    window_x: i32,
    window_y: i32,
    scale_factor: f64,
    layout: LayoutMode,
) -> (i32, i32) {
    let (dx, dy) = geometry::sprite_window_offsets(scale_factor);
    match layout {
        LayoutMode::Normal => (window_x + dx, window_y + dy),
        LayoutMode::TopAnchored => (window_x + dx, window_y),
    }
}

/// Position Rolo's window at his starting coordinates.
fn position_window(
    app: &tauri::App,
    sprite_x: i32,
    sprite_y: i32,
    scale_factor: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let window = app
        .get_webview_window("main")
        .ok_or("Could not find main window — Rolo's home is missing!")?;

    let (win_x, win_y) =
        sprite_to_window_position(sprite_x, sprite_y, scale_factor, LayoutMode::Normal);
    window.set_position(PhysicalPosition::new(win_x, win_y))?;
    Ok(())
}

/// Resolve and load the idle sprite alphas into a single hit-mask. Returns
/// None if asset resolution fails or every frame fails to decode — Rolo
/// then falls back to the full sprite rect so hover still works.
fn load_idle_hit_mask(app: &tauri::App) -> Option<Arc<hitmask::HitMask>> {
    let mut paths = Vec::new();
    for i in 0..4 {
        let rel = format!("../ASSETS/IDLE/idle{}.png", i);
        match app
            .path()
            .resolve(&rel, tauri::path::BaseDirectory::Resource)
        {
            Ok(p) if p.exists() => paths.push(p),
            Ok(p) => {
                log::warn!("[Rolo] idle frame {:?} missing — skipping", p);
            }
            Err(e) => {
                log::warn!("[Rolo] Could not resolve {}: {}", rel, e);
            }
        }
    }
    if paths.is_empty() {
        log::warn!(
            "[Rolo] No idle frames found for hit-mask — hover will fall back to full sprite rect"
        );
        return None;
    }
    match hitmask::HitMask::from_png_paths(&paths) {
        Ok(mask) => Some(Arc::new(mask)),
        Err(e) => {
            log::warn!(
                "[Rolo] Failed to build hit-mask: {} — hover will use full sprite rect",
                e,
            );
            None
        }
    }
}

/// Load phrases.json from the bundled ASSETS directory.
fn load_phrases(app: &tauri::App) -> Option<Vec<u8>> {
    let resource_path = app.path().resolve(
        "../ASSETS/phrases.json",
        tauri::path::BaseDirectory::Resource,
    );
    match resource_path {
        Ok(path) => match std::fs::read(&path) {
            Ok(data) => Some(data),
            Err(e) => {
                eprintln!(
                    "[Rolo] Could not read phrases.json at {:?}: {}. \
                     He'll stay silent but healthy.",
                    path, e
                );
                None
            }
        },
        Err(e) => {
            eprintln!(
                "[Rolo] Could not resolve phrases.json path: {}. \
                 He'll stay silent but healthy.",
                e
            );
            None
        }
    }
}

/// Create the hidden speech bubble window. Returns Ok(()) on success.
fn create_bubble_window(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let bubble_window = WebviewWindowBuilder::new(app, "speech-bubble", WebviewUrl::default())
        .title("Rolo Speech")
        .inner_size(
            geometry::SPEECH_MAX_WIDTH as f64,
            geometry::SPEECH_WINDOW_HEIGHT as f64,
        )
        .decorations(false)
        .transparent(true)
        .always_on_top(true)
        .resizable(false)
        .shadow(false)
        .skip_taskbar(true)
        .visible_on_all_workspaces(true)
        .focused(false)
        .visible(false)
        .build()?;

    raise_bubble_z_order(&bubble_window);

    Ok(())
}

/// Create the hidden status-panel window. the user opens it via right-click ->
/// "View Status" (Phase 7). Reuses the speech-bubble pattern: frameless,
/// transparent, always-on-top, hidden until summoned. Width 260, height
/// 200 logical px fits four bars + header + footer link (PRD §6).
fn create_status_panel_window(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let panel = WebviewWindowBuilder::new(app, "status-panel", WebviewUrl::default())
        .title("Rolo Status")
        .inner_size(260.0, 200.0)
        .decorations(false)
        .transparent(true)
        .always_on_top(true)
        .resizable(false)
        .shadow(false)
        .skip_taskbar(true)
        .visible_on_all_workspaces(true)
        .focused(false)
        .visible(false)
        .build()?;

    raise_bubble_z_order(&panel);

    Ok(())
}

/// Set the speech bubble window one level above the main window so it
/// always renders in front of the main window's transparent overlap area.
#[cfg(target_os = "macos")]
fn raise_bubble_z_order(window: &WebviewWindow) {
    use objc2::msg_send;
    use objc2::runtime::AnyObject;

    let raw = window.ns_window();
    match raw {
        Ok(ptr) => {
            let ns_window = ptr as *mut AnyObject;
            unsafe {
                let current_level: i64 = msg_send![ns_window, level];
                let _: () = msg_send![ns_window, setLevel: current_level + 1];
            }
            log::info!("[Rolo] Speech bubble window z-order raised above main window");
        }
        Err(e) => {
            log::warn!(
                "[Rolo] Could not raise bubble z-order — overlapping windows \
                 may cause visual artifacts: {}",
                e
            );
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn raise_bubble_z_order(_window: &WebviewWindow) {
    // No-op on non-macOS platforms
}

/// Persist Rolo's state on shutdown — mood + interaction state.
///
/// Mirrors the logic the `RunEvent::Exit` arm used to perform inline, factored
/// here so the new "Quit Rolo" menu item can persist via the same code path
/// before calling `app.exit(0)`. Lock poisoning is swallowed: a shutdown-time
/// panic in another thread must not block Rolo's final save. Save failures
/// are logged rather than propagated — Rolo never refuses to die.
fn persist_state_on_exit(app: &tauri::AppHandle) {
    // Wake any in-flight dream BEFORE persisting state. The CancellationToken
    // lets the LLM call abort cleanly; the 500ms grace gives the compiler/lint
    // future a moment to unwind its dreams_log append. Raw JSONL is the ground
    // truth — anything we miss here recompiles next launch.
    if let Some(h) = app.try_state::<Arc<vault::dreaming::DreamHandle>>() {
        h.request_wake();
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    // Persist mood state synchronously on shutdown.
    if let Some(mood) = app.try_state::<SharedMood>() {
        let path = mood::path_for(app);
        let guard = mood.lock().unwrap_or_else(|p| p.into_inner());
        if let Err(e) = mood::save_to_disk(&guard, &path) {
            log::warn!("[Rolo] mood save on exit failed: {}", e);
        } else {
            log::info!("[Rolo] Mood persisted on exit");
        }
    }

    // Persist interaction state (daily-fire counter) on shutdown. The
    // off-thread save in tick.rs covers the common case, but a hard kill
    // immediately after a check-in fires could race past it. Saving
    // synchronously here closes that window for graceful shutdowns.
    if let Some(ix) = app.try_state::<SharedInteraction>() {
        let path = interaction::path_for(app);
        let guard = ix.lock().unwrap_or_else(|p| p.into_inner());
        let snapshot = guard.persisted_snapshot();
        if let Err(e) = interaction::save_to_disk(&path, &snapshot) {
            log::warn!("[Rolo] interaction-state save on exit failed: {}", e);
        } else {
            log::info!("[Rolo] Interaction state persisted on exit");
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();
    log::info!("[Rolo] Waking up... initializing Tauri application");

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            commands::cc_apply_brain,
            commands::cc_brain_needs_setup,
            commands::cc_cancel_brain_pull,
            commands::cc_cancel_sleep,
            commands::cc_check_brain_model,
            commands::cc_clear_user_profile,
            commands::cc_force_close,
            commands::cc_load_chat_config,
            commands::cc_load_settings,
            commands::cc_load_user_profile,
            commands::cc_pull_brain_model,
            commands::cc_run_diagnostics,
            commands::cc_save_memory_and_sleep,
            commands::cc_save_settings,
            commands::cc_test_brain,
            commands::cc_test_weather,
            commands::cc_weather_status,
            commands::open_command_center,
            commands::get_pet_state,
            commands::get_screen_size,
            commands::get_pending_files,
            commands::confirm_eat,
            commands::decline_eat,
            commands::get_stats,
            commands::feed_rolo_dialog,
            commands::dismiss_speech,
            commands::force_speech,
            commands::submit_interaction_response,
            commands::open_status_panel,
            commands::close_status_panel,
            commands::get_mood_snapshot,
            commands::reset_mood_state,
            commands::open_chat,
            commands::close_chat,
            commands::send_chat_message,
            commands::report_chat_message,
            commands::export_character_card,
            commands::export_training_data,
            commands::wake_rolo,
            commands::chat_window_is_open,
            commands::read_dream_log,
            commands::revert_fact,
            commands::dev_invoke_tool,
        ])
        .setup(|app| {
            // 1. Query the monitor
            let (screen_w, screen_h, scale_factor) = get_monitor_info(app)
                .unwrap_or_else(|e| {
                    eprintln!(
                        "[Rolo] Could not determine screen dimensions: {}. \
                         He'll try his best with defaults.",
                        e
                    );
                    (1920, 1080, 1.0)
                });

            // 2. Calculate starting position — bottom-right corner
            let (start_x, start_y) = calculate_start_position(screen_w, screen_h, scale_factor);

            // 3. Place the window
            if let Err(e) = position_window(app, start_x, start_y, scale_factor) {
                eprintln!(
                    "[Rolo] Window positioning failed: {}. \
                     He'll appear somewhere, at least.",
                    e
                );
            }

            // 4. Create Rolo's state machine
            let physical_pet = (f64::from(SPRITE_SIZE) * scale_factor) as i32;
            let walk_speed = (2.0 * scale_factor).round().max(2.0) as i32;
            let mut pet = Pet::new(
                start_x,
                start_y,
                screen_w as i32,
                screen_h as i32,
                physical_pet,
                walk_speed,
            );

            // Record half the bubble width so the speech system can detect
            // when Rolo is too close to a horizontal edge and nudge him
            // inward before a phrase fires. Motion is NOT clamped by this —
            // he's free to walk or be dragged to the actual screen edge.
            let bubble_margin_x =
                (f64::from(geometry::SPEECH_MAX_WIDTH) * scale_factor / 2.0) as i32;
            pet.set_bubble_margin_x(bubble_margin_x);

            // 5. Wrap in Arc<Mutex>
            let shared_pet: SharedPet = Arc::new(Mutex::new(pet));
            let tick_pet = Arc::clone(&shared_pet);
            let drag_pet = Arc::clone(&shared_pet);
            let chat_pet = Arc::clone(&shared_pet);
            app.manage(shared_pet);

            // 6. Set initial click-through
            if let Some(window) = app.get_webview_window("main") {
                if let Err(e) = window.set_ignore_cursor_events(true) {
                    log::warn!("[Rolo] Could not set initial click-through: {}", e);
                }
                log::info!("[Rolo] Click-through enabled (transparent areas pass through)");
            }

            // 7. Load eating stats
            let eating_stats = EatingStats::load();
            let shared_stats: SharedStats = Arc::new(Mutex::new(eating_stats));
            app.manage(shared_stats);

            // 8. Load phrases and create speech system
            let phrases_data = load_phrases(app);
            let speech_state = SpeechState::new(phrases_data.as_deref());
            let shared_speech: SharedSpeech = Arc::new(Mutex::new(speech_state));
            let tick_speech = Arc::clone(&shared_speech);
            let brain_changed_speech = Arc::clone(&shared_speech);
            app.manage(shared_speech);

            // 9. Create the hidden speech bubble window
            if let Err(e) = create_bubble_window(app) {
                eprintln!(
                    "[Rolo] Failed to create speech bubble window: {}. \
                     Rolo will be silent but otherwise healthy.",
                    e
                );
            }

            // 9b. Create the hidden status-panel window. Stays invisible
            // until the user right-clicks Rolo and picks "View Status".
            if let Err(e) = create_status_panel_window(app) {
                eprintln!(
                    "[Rolo] Failed to create status-panel window: {}. \
                     Rolo will hide his feelings but otherwise stay healthy.",
                    e
                );
            }

            // 9c. Create the hidden Command Center window. Opened from the
            // right-click menu — Rolo's single configuration surface.
            if let Err(e) = command_center::create_command_center_window(app) {
                eprintln!(
                    "[Rolo] Failed to create command-center window: {}. \
                     The settings surface won't open, but Rolo otherwise stays healthy.",
                    e
                );
            }

            // 9c-bis. PRD Phase 10 — fresh-install router.
            //
            // If `chat_config.json` is missing or empty enough that
            // `chat_config_is_unconfigured` returns true, show the Command
            // Center immediately so the user lands in Brain tab. The
            // frontend's `cc_brain_needs_setup` query gates the rest of
            // the UI into modal setup mode.
            //
            // We do this BEFORE the chat engine and main pet are made
            // available to the user so a fresh install never tries to
            // chat with a non-existent brain. The chat engine still
            // initializes with a (possibly-failing) provider below — the
            // setup gate is the user-facing fix.
            let fresh_install_chat_config = chat::config::ChatConfig::load();
            let needs_brain_setup = command_center::chat_config_is_unconfigured(
                &fresh_install_chat_config,
            );
            if needs_brain_setup {
                log::info!(
                    "[Rolo] Fresh install detected — chat config is unconfigured. \
                     Opening Command Center > Brain in modal setup mode."
                );
                if let Err(e) = command_center::window::show_command_center(app.handle()) {
                    log::warn!(
                        "[Rolo] Could not auto-show Command Center on fresh install: {}. \
                         User can open it from the right-click menu.",
                        e
                    );
                }
            }

            // 9d. Register the drag-watch heartbeat shared atomic. The tick
            // loop writes a unix-millis timestamp into it every iteration;
            // the Perception tab's drag-drop probe reads it to confirm the
            // watcher thread is still beating. Initialized to "now" so the
            // first probe before the tick loop spins up doesn't false-Red.
            let heartbeat: command_center::diagnostics::DragWatchHeartbeat =
                Arc::new(std::sync::atomic::AtomicI64::new(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as i64)
                        .unwrap_or(0),
                ));
            let tick_heartbeat = Arc::clone(&heartbeat);
            app.manage(heartbeat);

            // 10. Create the interaction engine (mood check-ins).
            // ROLO_DEBUG=1 collapses check-in cadence to seconds and disables
            // quiet hours so a full cycle can be tested in a single session.
            let turbo_mode = crate::speech::debug_mode_enabled();
            if turbo_mode {
                log::info!(
                    "[Rolo] Interaction engine: TURBO mode (ROLO_DEBUG=1) — \
                     check-ins fire every few seconds, quiet hours disabled."
                );
            } else {
                log::info!(
                    "[Rolo] Interaction engine: NORMAL mode — check-ins on \
                     the real schedule (few per day, 8am-10pm only)."
                );
            }
            let debug_config = DebugConfig { turbo_mode };
            let mut interaction_state = InteractionState::new(debug_config);
            // Restore the persisted daily-fire counter so the cap survives
            // restarts. Without this, every fresh launch resets fires_today
            // to 0 and the cap is effectively unenforced — see Phase 1 of
            // chat-bug-fix.
            let interaction_path = interaction::path_for(app.handle());
            let persisted_interaction = interaction::load_from_disk(&interaction_path);
            log::info!(
                "[Rolo] Interaction state loaded: fires_today={} (date={:?})",
                persisted_interaction.fires_today,
                persisted_interaction.fires_today_date,
            );
            interaction_state.restore_from(persisted_interaction);
            let shared_interaction: SharedInteraction = Arc::new(Mutex::new(interaction_state));
            let tick_interaction = Arc::clone(&shared_interaction);
            app.manage(shared_interaction);

            // 11. Load Rolo's mood state from disk (always returns a usable
            // value — corrupt files are sidecar-renamed and replaced with
            // defaults). Phase 2: persisted on shutdown via RunEvent::Exit.
            // Phase 3 will start the periodic save + decay/event integration.
            let mood_path = mood::path_for(app.handle());
            let mood_state = mood::load_from_disk(&mood_path);
            log::info!("[Rolo] Mood loaded: {:?}", mood_state.snapshot());
            let shared_mood: SharedMood = Arc::new(Mutex::new(mood_state));
            let tick_mood = Arc::clone(&shared_mood);
            let chat_mood = Arc::clone(&shared_mood);
            app.manage(shared_mood);

            // 11c. Status panel state slots — populated by Phase 6 commands
            // and the right-click handler. Empty/false on startup.
            let status_panel_open: StatusPanelOpenFlag = Arc::new(AtomicBool::new(false));
            let tick_status_panel_open = Arc::clone(&status_panel_open);
            app.manage(status_panel_open);
            let last_right_click: LastRightClickPos = Arc::new(Mutex::new(None));
            app.manage(last_right_click);

            // 12. Initialize chat store — Rolo's conversational memory banks
            let chat_store = match chat::store::ChatStore::open() {
                Ok(store) => {
                    log::info!("[Rolo] Chat store opened — memory banks online");
                    store
                }
                Err(e) => {
                    log::error!(
                        "[Rolo] Failed to open chat store: {}. Chat will use in-memory fallback.",
                        e
                    );
                    chat::store::ChatStore::open_in_memory()
                        .expect("In-memory chat store must succeed")
                }
            };
            let shared_chat_store: SharedChatStore = Arc::new(Mutex::new(chat_store));
            let engine_store = Arc::clone(&shared_chat_store);
            app.manage(shared_chat_store);

            // 12b. Initialize chat engine with config.
            //
            // Phase 4 of PRD/rolo-command-center.md: the provider lives in a
            // `SharedProviderSlot` managed at app scope so the Brain panel can
            // hot-swap it at runtime. We build one `Arc<dyn InferenceProvider>`
            // here, seed the slot with it, hand the slot to both `ChatEngine`
            // and the dreaming poll loop, and `app.manage()` the slot itself so
            // `cc_apply_brain` (Phase 5) can reach it from a command handler.
            // Reuse the ChatConfig we already loaded for the fresh-install
            // gate above — avoids a second disk read and guarantees both
            // checks see the same byte-for-byte state.
            let chat_config = fresh_install_chat_config;
            log::info!(
                "[Rolo] Chat config loaded — provider: {}, model: {}",
                chat_config.inference.provider.as_wire_str(),
                match chat_config.inference.provider {
                    chat::config::Provider::Ollama => &chat_config.inference.ollama.model,
                    chat::config::Provider::Anthropic => &chat_config.inference.anthropic.model,
                    chat::config::Provider::Gemini => &chat_config.inference.gemini.model,
                    chat::config::Provider::OpenaiCompat
                    | chat::config::Provider::Huggingface
                    | chat::config::Provider::Deepinfra =>
                        &chat_config.inference.openai_compat.model,
                },
            );
            let initial_provider =
                chat::engine::ChatEngine::build_provider_from_config(&chat_config);
            let provider_slot =
                command_center::SharedProviderSlot::new(initial_provider);
            app.manage(provider_slot.clone());
            let chat_engine =
                chat::engine::ChatEngine::new(engine_store, &chat_config, provider_slot.clone());

            // 12c. Open or initialize Rolo's Vault (his persistent file-based memory).
            //      Resolves to ~/Library/Application Support/com.larawashington.rolo/vault/
            //      Falls back to a temp dir if app_data_dir is unavailable so Rolo still runs
            //      (memory won't persist across launches in that case).
            let vault_root = app
                .path()
                .app_data_dir()
                .map(|d| d.join("vault"))
                .unwrap_or_else(|e| {
                    log::warn!(
                        "[Rolo] Could not resolve app_data_dir: {} — using temp dir; \
                         vault will not persist across runs",
                        e
                    );
                    std::env::temp_dir().join("rolo-vault-fallback")
                });
            let vault = vault::Vault::open_or_init(vault_root);

            // Wire the vault's ExperienceLogger into the ChatEngine BEFORE it
            // gets sealed in the SharedChatEngine Arc<Mutex<>>. Every chat turn
            // will then append an ExperienceEvent::Chat to the vault, which is
            // what the dream compiler reads.
            let chat_engine = chat_engine.with_experience_logger(vault.logger.clone());
            // Wire the vault's hybrid-search PromptAssembler into ChatEngine
            // so every chat turn's system prompt is composed from the wiki
            // (BM25 + vectors) rather than from SQLite memories. This is the
            // PRD §6 / Section E1 "loop closure" — dreams that produce wiki
            // content now actually reach Rolo's prompt.
            let chat_engine = chat_engine.with_vault_assembler(vault.assembler.clone());
            // PRD/rolo-prompt-consolidation T1: wire mood + pet handles so the
            // chat path can capture a live `StateSnapshot` instead of the
            // mood-blind `StateContext::from_pet_state(None, ...)` stub.
            let chat_engine = chat_engine.with_state_handles(chat_mood, chat_pet);

            // 12c-bis. Tool registry — Gemma-callable wrappers around vault,
            // mood, and pet state. T1 only uses this from the
            // `dev_invoke_tool` command; T3 wires the dispatcher into the
            // bubble path; T5 wires the same dispatcher into the chat path
            // (PRD/rolo-tool-layer.md §6 T5). Constructed BEFORE the chat
            // engine is sealed so `with_vault` / `with_tool_registry` can
            // hand it shared `Arc` handles.
            let tool_registry = Arc::new(crate::tools::ToolRegistry::standard());

            // PRD/rolo-tool-layer.md T5: wire vault + tool registry into the
            // chat engine so `send_message` can consult the dispatcher
            // (gated by `ROLO_DISPATCHER_ENABLED`, same flag as T3).
            let chat_engine = chat_engine.with_vault(Arc::clone(&vault));
            let chat_engine = chat_engine.with_tool_registry(Arc::clone(&tool_registry));

            let shared_engine: commands::SharedChatEngine =
                Arc::new(tokio::sync::Mutex::new(chat_engine));
            app.manage(shared_engine);

            let tick_vault = Arc::clone(&vault);
            let close_vault = Arc::clone(&vault);
            let dream_vault = Arc::clone(&vault);
            app.manage(vault);

            // Phase 8 of PRD/rolo-command-center.md: pull a typed handle to
            // the weather tool out of the registry BEFORE we move the
            // registry into Tauri-managed state. The handle is needed in
            // three places:
            //   1. `set_app_handle` — so the tool can read
            //      `CommandCenterSettings` per invocation (custom endpoint
            //      branching + manual-location override).
            //   2. `app.manage` — so `cc_weather_status` can read the cache.
            //   3. A `rolo://weather-config-changed` listener — so saving
            //      Weather-tab settings immediately invalidates the cache
            //      and the next fetch hits the new endpoint.
            let weather_handle = Arc::clone(tool_registry.get_weather());
            weather_handle.set_app_handle(app.handle().clone());
            app.manage(Arc::clone(&weather_handle));

            // Cache invalidation listener. Topic uses the `rolo-internal://`
            // prefix to mark it as a backend-only contract — the matching
            // emit lives in `cc_save_settings` (commands.rs). Frontend code
            // must never emit `rolo-internal://*`; the convention is what
            // keeps the listener's trust assumption (payload originated in
            // Rust) intact. Listener returns immediately — invalidating the
            // cache is just a mutex swap.
            let weather_for_listener = Arc::clone(&weather_handle);
            app.listen("rolo-internal://weather-config-changed", move |_evt| {
                weather_for_listener.invalidate_cache();
                log::info!("[Rolo WEATHER] Cache invalidated after config change");
            });

            // `cc_apply_brain` (commands.rs) emits this event after a
            // successful slot swap, carrying the hardcoded reaction phrase
            // as the payload. We push it through `SpeechState::queue_immediate_phrase`
            // here so the apply-brain command itself stays free of any
            // dependency on `SharedSpeech` — exactly the decoupling the
            // weather-config listener already models above.
            //
            // Topic uses the `rolo-internal://` prefix — backend-only
            // contract, frontend must not emit. Without this, a compromised
            // webview could fire arbitrary text into `queue_immediate_phrase`.
            app.listen("rolo-internal://brain-changed", move |evt| {
                let phrase: String = match serde_json::from_str::<String>(evt.payload()) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!(
                            "[Rolo] brain-changed: malformed payload {:?}: {}. Skipping reaction bubble.",
                            evt.payload(),
                            e
                        );
                        return;
                    }
                };
                match brain_changed_speech.lock() {
                    Ok(mut guard) => guard.queue_immediate_phrase(phrase),
                    Err(_) => log::warn!(
                        "[Rolo] brain-changed: speech mutex poisoned, skipping reaction bubble"
                    ),
                }
            });

            app.manage(tool_registry);

            // 12d. Spawn the dreaming poll loop.
            //
            // Phase 4 of PRD/rolo-command-center.md replaced the old parallel
            // `dream_provider` construction with a shared `SharedProviderSlot`:
            // the dreaming loop and the chat engine both read through the same
            // slot, so a Brain-panel swap (via Phase 5's `cc_apply_brain`)
            // takes effect for the very next dream cycle without restarting
            // the loop. The PRD decision #5 ("one selected brain serves all
            // four call-sites") drives this — no more duplicated wiring.
            let dream_handle = Arc::new(vault::dreaming::DreamHandle::new());
            app.manage(Arc::clone(&dream_handle));

            let app_handle_for_chat_check = app.handle().clone();
            let chat_open: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || {
                // The chat window is created on demand via `commands::open_chat`.
                // Presence in the webview registry is the cheapest signal that
                // matches the PRD §1 "chat window not open" gate.
                app_handle_for_chat_check
                    .get_webview_window("chat")
                    .is_some()
            });

            let dream_pet = Arc::clone(&drag_pet);
            vault::dreaming::spawn_poll_loop(
                dream_vault,
                dream_pet,
                provider_slot,
                Arc::clone(&dream_handle),
                chat_open,
                vault::dreaming::PollerConfig::default(),
            );

            // 13. Wire drag-drop events to the state machine
            let last_drop = Arc::new(Mutex::new(Instant::now()));
            let app_handle_drag = app.handle().clone();

            app.get_webview_window("main")
                .expect("Main window must exist for drag-drop wiring")
                .on_window_event(move |event| {
                    match event {
                        WindowEvent::DragDrop(drag_event) => {
                            log::info!("[Rolo] DragDropEvent received: {:?}", drag_event);
                            let mut pet = drag_pet.lock().unwrap_or_else(|p| p.into_inner());

                            match drag_event {
                                DragDropEvent::Enter { paths, .. } => {
                                    log::info!("[Rolo] DragDrop::Enter with {} paths", paths.len());
                                    if !paths.is_empty() {
                                        pet.file_drag_entered();
                                        log::info!("[Rolo] After file_drag_entered, state={:?}", pet.state());
                                    }
                                }
                                DragDropEvent::Drop { paths, .. } => {
                                    log::info!("[Rolo] DragDrop::Drop with {} paths", paths.len());
                                    // Debounce: ignore drops within 100ms (Tauri duplicate event bug)
                                    let mut last = last_drop.lock().unwrap_or_else(|p| p.into_inner());
                                    let now = Instant::now();
                                    if now.duration_since(*last).as_millis() < 100 {
                                        log::warn!("[Rolo] DragDrop::Drop debounced (duplicate within 100ms)");
                                        return;
                                    }
                                    *last = now;
                                    drop(last);

                                    let path_strings: Vec<String> = paths
                                        .iter()
                                        .filter_map(|p: &std::path::PathBuf| p.to_str().map(String::from))
                                        .collect();
                                    if !path_strings.is_empty() {
                                        log::info!("[Rolo] Feeding Rolo {} files via drop", path_strings.len());
                                        pet.file_dropped(path_strings);
                                        log::info!("[Rolo] After file_dropped, state={:?}", pet.state());
                                        if let Some(pending) = pet.pending_files() {
                                            match app_handle_drag.emit("rolo://files-pending", pending) {
                                                Ok(_) => log::info!("[Rolo] Emitted rolo://files-pending with {} files", pending.len()),
                                                Err(e) => log::error!("[Rolo] Failed to emit files-pending: {}", e),
                                            }
                                        } else {
                                            log::warn!("[Rolo] No pending files after file_dropped — state={:?}", pet.state());
                                        }
                                    }
                                }
                                DragDropEvent::Leave => {
                                    log::info!("[Rolo] DragDrop::Leave");
                                    pet.file_drag_exited();
                                }
                                _ => {
                                    log::debug!("[Rolo] DragDropEvent other variant");
                                }
                            }
                        }
                        WindowEvent::CloseRequested { .. } => {
                            // Flush is best-effort here; SessionEnd is written by
                            // ExperienceLogger::Drop on app teardown so every exit
                            // path emits exactly once.
                            close_vault.logger.flush();
                        }
                        _ => {}
                    }
                });

            // 14. Load hover hit-mask from idle sprites. Merged alpha across
            // frames so hover only fires on Rolo's visible pixels.
            let hit_mask = load_idle_hit_mask(app);

            // 15. Spawn passive-perception producers (Trash, Downloads, Foreground app).
            // Each producer is a thread that sends `PerceptionEvent`s on a single
            // channel; the receiver is moved into the tick thread closure since
            // `mpsc::Receiver` is `!Sync` and can't be Tauri-managed. The buffer
            // itself is `Arc<Mutex>`-managed so other code paths could read it later.
            #[cfg(target_os = "macos")]
            let (perception_rx, perception_buffer) = {
                use crate::perception::{PerceptionBuffer, SharedPerceptionBuffer};
                let buf: SharedPerceptionBuffer = Arc::new(Mutex::new(PerceptionBuffer::default()));
                app.manage(Arc::clone(&buf));
                let handles = crate::perception::spawn_all();
                // The producer-thread join handles inside `handles` (held only
                // to keep them from being considered detached) are dropped at
                // the end of this block — the threads leak to live for the
                // app's lifetime, which is the documented contract. `heartbeats`
                // is moved into Tauri-managed state so the Command Center's
                // Perception probes can read each producer's liveness.
                let crate::perception::PerceptionHandles {
                    rx, heartbeats, ..
                } = handles;
                app.manage(Arc::clone(&heartbeats));
                (rx, buf)
            };

            // 16. Start the heartbeat — Rolo comes alive
            let app_handle = app.handle().clone();
            tick::start_tick_loop(
                app_handle,
                tick_pet,
                tick_speech,
                tick_interaction,
                tick_mood,
                tick_status_panel_open,
                tick_vault,
                scale_factor,
                WINDOW_SIZE,
                SPRITE_SIZE,
                physical_pet,
                screen_w as i32,
                screen_h as i32,
                hit_mask,
                tick_heartbeat,
                #[cfg(target_os = "macos")]
                perception_rx,
                #[cfg(target_os = "macos")]
                perception_buffer,
            );

            Ok(())
        })
        .on_menu_event(|app_handle, event| match event.id().0.as_str() {
            "view-status" => {
                log::info!("[Rolo] Menu: 'View Status' selected");
                // Read the cursor position the right-click handler stashed
                // before popping the menu. Falls back to (0, 0) — the
                // panel's clamp_to_screen will pull it onto a monitor.
                let pos = app_handle
                    .try_state::<LastRightClickPos>()
                    .and_then(|s| s.lock().ok().and_then(|g| *g))
                    .unwrap_or((0.0, 0.0));
                let app_for_main = app_handle.clone();
                let _ = app_handle.run_on_main_thread(move || {
                    if let Err(e) =
                        commands::open_status_panel_internal(app_for_main, pos.0, pos.1)
                    {
                        log::warn!("[Rolo] open_status_panel_internal failed: {}", e);
                    }
                });
            }
            MENU_ID_OPEN_COMMAND_CENTER => {
                log::info!("[Rolo] Menu: 'Open Rolo Command Center' selected");
                let app_for_main = app_handle.clone();
                let _ = app_handle.run_on_main_thread(move || {
                    if let Err(e) = command_center::window::show_command_center(&app_for_main) {
                        log::warn!("[Rolo] show_command_center failed: {}", e);
                    }
                });
            }
            MENU_ID_CHAT_WITH_ROLO => {
                log::info!("[Rolo] Menu: 'Chat with Rolo' selected");
                let app_handle = app_handle.clone();
                tauri::async_runtime::spawn(async move {
                    let pet = app_handle.state::<SharedPet>();
                    let speech = app_handle.state::<SharedSpeech>();
                    let chat_store = app_handle.state::<SharedChatStore>();
                    match commands::open_chat(
                        String::new(),
                        "menu".to_string(),
                        None,
                        pet,
                        speech,
                        chat_store,
                        app_handle.clone(),
                    )
                    .await
                    {
                        Ok(result) => {
                            log::info!("[Rolo] open_chat from menu: {}", result);
                        }
                        Err(e) => {
                            log::warn!("[Rolo] open_chat from menu failed: {}", e);
                        }
                    }
                });
            }
            MENU_ID_QUIT => {
                log::info!("[Rolo] Menu: 'Quit Rolo' selected");
                // Persist mood + interaction state synchronously before
                // exiting, mirroring the RunEvent::Exit path. Then call
                // app_handle.exit(0) which triggers Exit too — the helper is
                // idempotent (just two extra save writes worst case).
                persist_state_on_exit(app_handle);
                app_handle.exit(0);
            }
            other => {
                log::warn!("[Rolo] Unknown menu event id: {}", other);
            }
        })
        .build(tauri::generate_context!())
        .expect("Rolo failed to start — this is a critical health emergency!")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                persist_state_on_exit(app);
            }
        });
}
