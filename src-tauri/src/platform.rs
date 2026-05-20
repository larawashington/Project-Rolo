//! Cursor polling and window-activation helpers.
//!
//! On macOS, transparent unfocused windows don't receive DOM mouse events.
//! This module bypasses that limitation by polling the global cursor position
//! and mouse button state directly from CoreGraphics — see the macOS impl
//! block below.
//!
//! On other platforms, the API is preserved but every function is a no-op
//! stub: `get_cursor_info` returns `None` and `activate_window_for_input` is
//! a log-only fallback. This lets the crate compile on Linux and Windows for
//! CI, with the caveat that hover/drag features driven by global cursor
//! polling won't fire there. Native Tauri `WindowEvent::DragDrop` and
//! focused-window mouse events still work — only the macOS-specific bypass
//! for unfocused transparent windows is gone.
//!
//! Coordinate note (macOS): CoreGraphics returns **points** (logical pixels),
//! not physical pixels. Multiply by `scale_factor` before comparing with
//! Tauri's `PhysicalPosition` values.

#[cfg(not(target_os = "macos"))]
use tauri::AppHandle;

/// Snapshot of cursor state at a single point in time.
///
/// Always populated on macOS; never populated on non-macOS (callers must
/// handle `None` from [`get_cursor_info`]).
#[derive(Debug, Clone)]
pub struct CursorInfo {
    /// X position in points (logical pixels), screen origin top-left.
    pub x: f64,
    /// Y position in points (logical pixels), screen origin top-left.
    pub y: f64,
    /// Whether the left mouse button is currently held down.
    pub left_down: bool,
    /// Whether the right mouse button is currently held down.
    pub right_down: bool,
}

// ---------------------------------------------------------------------------
// macOS implementation
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod mac {
    use super::CursorInfo;
    use core_graphics::event::CGEvent;
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
    use objc2::msg_send;
    use objc2::rc::autoreleasepool;
    use objc2::runtime::{AnyObject, Sel};
    use objc2::sel;
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
    use objc2_foundation::MainThreadMarker;
    use std::ffi::CStr;
    use std::os::raw::c_char;
    use tauri::{AppHandle, Manager};

    /// Force a child window (wizard/chat) to become key and make the app active.
    ///
    /// Rolo runs under `macOSPrivateApi: true` with the main window's
    /// `skipTaskbar: true`, which puts NSApp in **accessory** activation policy.
    /// In accessory mode, no child window can become key on its own — Tauri's
    /// `.focused(true)` is a no-op because the app itself is not "active".
    ///
    /// Two things must happen:
    /// 1. Flip NSApp to `Regular` and `activateIgnoringOtherApps`, then
    ///    `makeKeyAndOrderFront:` so the window can actually be key. Side-effect:
    ///    a Dock icon while an activated window is open. Rolo's main window
    ///    stays click-through.
    /// 2. Defuse NSPanel defaults inherited from `macos-private-api`. tao
    ///    backs these windows with NSPanel, which has `hidesOnDeactivate` and
    ///    `becomesKeyOnlyIfNeeded` set, plus the `NonactivatingPanel` style
    ///    bit — without clearing them, the window auto-hides on outside clicks.
    ///
    /// Dispatches the AppKit work to the main thread via Tauri's
    /// `run_on_main_thread`, so it's safe to call from async command handlers
    /// that run on the tokio executor.
    pub fn activate_window_for_input(app_handle: &AppHandle, window_label: &str) {
        let label = window_label.to_string();
        let app_handle = app_handle.clone();
        let result = app_handle.clone().run_on_main_thread(move || {
            let window = match app_handle.get_webview_window(&label) {
                Some(w) => w,
                None => {
                    log::warn!(
                        "[Rolo] activate_window_for_input: window '{}' not found",
                        label
                    );
                    return;
                }
            };

            let ns_window_ptr = match window.ns_window() {
                Ok(p) => p,
                Err(e) => {
                    log::warn!("[Rolo] activate_window_for_input: ns_window() failed: {}", e);
                    return;
                }
            };
            if ns_window_ptr.is_null() {
                log::warn!("[Rolo] activate_window_for_input: NSWindow pointer is null");
                return;
            }

            let mtm = match MainThreadMarker::new() {
                Some(m) => m,
                None => {
                    log::error!(
                        "[Rolo] activate_window_for_input: run_on_main_thread closure not on main thread!"
                    );
                    return;
                }
            };

            autoreleasepool(|_| unsafe {
                let app = NSApplication::sharedApplication(mtm);
                app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
                #[allow(deprecated)]
                app.activateIgnoringOtherApps(true);

                let ns_window = ns_window_ptr as *mut AnyObject;

                let cls: *mut AnyObject = msg_send![ns_window, class];
                let cls_name_ptr: *const c_char = objc2::ffi::class_getName(cls as *const _);
                let class_name = if cls_name_ptr.is_null() {
                    "<null>".to_string()
                } else {
                    CStr::from_ptr(cls_name_ptr).to_string_lossy().into_owned()
                };

                let sel_hides: Sel = sel!(setHidesOnDeactivate:);
                let responds_hides: bool = msg_send![ns_window, respondsToSelector: sel_hides];
                if responds_hides {
                    let _: () = msg_send![ns_window, setHidesOnDeactivate: false];
                }

                let sel_becomes: Sel = sel!(setBecomesKeyOnlyIfNeeded:);
                let responds_becomes: bool = msg_send![ns_window, respondsToSelector: sel_becomes];
                if responds_becomes {
                    let _: () = msg_send![ns_window, setBecomesKeyOnlyIfNeeded: false];
                }

                // Clear NSWindowStyleMaskNonactivatingPanel (1<<7) only on NSPanel.
                if class_name == "NSPanel" {
                    let mask: usize = msg_send![ns_window, styleMask];
                    let cleared = mask & !(1usize << 7);
                    let _: () = msg_send![ns_window, setStyleMask: cleared];
                }

                let _: () = msg_send![ns_window, setLevel: 0i64];

                let _: () =
                    msg_send![ns_window, makeKeyAndOrderFront: std::ptr::null::<AnyObject>()];
            });
        });

        if let Err(e) = result {
            log::error!(
                "[Rolo] Could not dispatch activate_window_for_input to main thread: {}",
                e
            );
        }
    }

    extern "C" {
        fn CGEventSourceButtonState(state_id: CGEventSourceStateID, button: u32) -> bool;
    }

    const CG_MOUSE_BUTTON_LEFT: u32 = 0;
    const CG_MOUSE_BUTTON_RIGHT: u32 = 1;

    /// Poll the global cursor position and left-button state.
    ///
    /// Returns `None` if CoreGraphics refuses to create an event (rare — would
    /// indicate a system-level problem, like Rolo being very sick).
    pub fn get_cursor_info() -> Option<CursorInfo> {
        // Reuse the CGEventSource across calls — it's stateless for
        // CombinedSessionState and expensive to recreate 60 times/sec.
        thread_local! {
            static SOURCE: Option<CGEventSource> =
                CGEventSource::new(CGEventSourceStateID::CombinedSessionState).ok();
        }

        SOURCE.with(|maybe_source| {
            let source = maybe_source.as_ref()?;
            let event = CGEvent::new(source.clone()).ok()?;
            let point = event.location();
            let left_down = unsafe {
                CGEventSourceButtonState(
                    CGEventSourceStateID::CombinedSessionState,
                    CG_MOUSE_BUTTON_LEFT,
                )
            };
            let right_down = unsafe {
                CGEventSourceButtonState(
                    CGEventSourceStateID::CombinedSessionState,
                    CG_MOUSE_BUTTON_RIGHT,
                )
            };

            Some(CursorInfo {
                x: point.x,
                y: point.y,
                left_down,
                right_down,
            })
        })
    }
}

#[cfg(target_os = "macos")]
pub use mac::{activate_window_for_input, get_cursor_info};

// ---------------------------------------------------------------------------
// Non-macOS stubs — keep the public API alive so the rest of the crate
// (tick.rs, commands.rs) compiles on Linux and Windows for CI.
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "macos"))]
pub fn get_cursor_info() -> Option<CursorInfo> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn activate_window_for_input(_app_handle: &AppHandle, window_label: &str) {
    log::debug!(
        "[Rolo] activate_window_for_input: no-op on non-macOS (window='{}')",
        window_label
    );
}
