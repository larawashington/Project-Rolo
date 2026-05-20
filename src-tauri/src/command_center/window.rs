//! Command Center window plumbing: builder + show/hide lifecycle.
//!
//! Unlike the speech bubble and status panel — both frameless, transparent,
//! always-on-top — the Command Center is a regular working window with a
//! titlebar, resize handles, and a taskbar entry. It's hidden at startup
//! and shown on demand from the right-click menu. Closing the window hides
//! rather than destroys, so a second open is instant and any saved state in
//! the webview persists between sessions of the running app.

use tauri::{Emitter, Manager, WebviewUrl, WebviewWindowBuilder, WindowEvent};

const WINDOW_LABEL: &str = "command-center";
const WINDOW_TITLE: &str = "Rolo Command Center";
const WINDOW_WIDTH: f64 = 720.0;
const WINDOW_HEIGHT: f64 = 520.0;

/// Build the hidden Command Center window during app setup. The close-event
/// handler is attached here, once, so we don't need an external guard to
/// avoid double-registration on subsequent shows.
pub fn create_command_center_window(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let window = WebviewWindowBuilder::new(app, WINDOW_LABEL, WebviewUrl::default())
        .title(WINDOW_TITLE)
        .inner_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        .min_inner_size(600.0, 420.0)
        .decorations(false)
        .transparent(true)
        .always_on_top(false)
        .resizable(true)
        .skip_taskbar(false)
        .focused(false)
        .visible(false)
        .build()?;

    // Intercept the close button — Phase 9 / Phase 10.
    //
    // Always `prevent_close` and hand the decision to the frontend by
    // emitting `rolo://command-center-close-requested`. The webview
    // checks its `dirty` state and the Brain-needs-setup gate:
    //   - dirty: shows a "Discard changes?" modal; on confirm it invokes
    //     `cc_force_close` which hides the window.
    //   - setup mode: refuses to close until a Brain is saved.
    //   - clean & configured: immediately invokes `cc_force_close`.
    //
    // We do NOT hide synchronously here. If the emit fails (e.g. the
    // webview is mid-reload), we still hide as a fallback so the user
    // isn't stuck with an unkillable window — losing the discard prompt
    // in that rare race is preferable to a stuck close button.
    let close_app = window.app_handle().clone();
    let close_window = window.clone();
    window.on_window_event(move |event| {
        if let WindowEvent::CloseRequested { api, .. } = event {
            api.prevent_close();
            if let Err(e) = close_app.emit_to(
                "command-center",
                "rolo://command-center-close-requested",
                (),
            ) {
                log::warn!(
                    "[Rolo] Command Center: failed to emit close-requested ({}). \
                     Falling back to synchronous hide so the window doesn't get stuck.",
                    e
                );
                if let Err(hide_err) = close_window.hide() {
                    log::warn!(
                        "[Rolo] Command Center: fallback hide also failed: {}",
                        hide_err
                    );
                }
            }
        }
    });

    Ok(())
}

/// Reveal the Command Center, center it on whatever monitor the cursor is
/// currently on (falling back to the primary), focus it, and emit
/// `rolo://command-center-opened` so the frontend can refresh diagnostics.
/// Returns `Err` if the window wasn't built at startup — that's a
/// programmer error, not a runtime one, so the caller logs and moves on.
pub fn show_command_center(app: &tauri::AppHandle) -> Result<(), String> {
    let window = app
        .get_webview_window(WINDOW_LABEL)
        .ok_or_else(|| "command-center window missing".to_string())?;

    // `center()` picks the monitor the window currently sits on, which on
    // first show is the primary monitor. Good enough for v1 — multi-monitor
    // cursor-following can come later if needed.
    if let Err(e) = window.center() {
        log::warn!(
            "[Rolo] Command Center: could not center window: {}. \
             Showing at its previous position.",
            e
        );
    }

    window
        .show()
        .map_err(|e| format!("Cannot show Command Center window: {}", e))?;
    window
        .set_focus()
        .map_err(|e| format!("Cannot focus Command Center window: {}", e))?;

    if let Err(e) = app.emit("rolo://command-center-opened", ()) {
        log::warn!(
            "[Rolo] Command Center: emit 'opened' event failed: {}. \
             Diagnostics will refresh on next manual click.",
            e
        );
    }

    Ok(())
}
