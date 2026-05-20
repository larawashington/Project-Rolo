//! Global file-drag detection via the macOS drag pasteboard.
//!
//! Tauri's `WindowEvent::DragDrop` only fires when a file is dragged over OUR
//! window. For Rolo to perk up the instant the user picks up ANY file on
//! screen, we need to observe system-wide drag sessions.
//!
//! On macOS, this module polls `NSPasteboard.pasteboardWithName(.drag).changeCount`
//! each tick. When the count increments and the mouse button is held, a new
//! drag session has started. We combine this with the existing CoreGraphics
//! left button state (from `platform::get_cursor_info`) to detect the end of
//! the drag. Used by Yoink, Dropover, and similar apps. No accessibility
//! permission required — the drag pasteboard is readable by any app.
//!
//! On other platforms, [`DragWatcher::poll`] always returns `None`. There's no
//! cross-platform "global drag in progress" signal, so the feature degrades
//! to "Rolo only reacts when the file enters his own window" (via Tauri's
//! native `WindowEvent::DragDrop`). The stub exists so the crate compiles on
//! Linux and Windows for CI.

/// Events emitted by the watcher when drag state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DragEvent {
    /// A new file drag session started somewhere on screen.
    Started,
    /// The active file drag session ended (mouse released).
    Ended,
}

// ---------------------------------------------------------------------------
// macOS implementation
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod mac {
    use super::DragEvent;
    use objc2::rc::autoreleasepool;
    use objc2_app_kit::{NSPasteboard, NSPasteboardNameDrag, NSPasteboardTypeFileURL};

    /// Polls the drag pasteboard each tick to detect system-wide file drags.
    pub struct DragWatcher {
        last_change_count: isize,
        active: bool,
    }

    impl DragWatcher {
        /// Build a new watcher, seeding with the current pasteboard changeCount so
        /// stale drag data from before Rolo woke up doesn't trigger a spurious
        /// Started event.
        pub fn new() -> Self {
            let initial = autoreleasepool(|_| unsafe {
                let pb = NSPasteboard::pasteboardWithName(NSPasteboardNameDrag);
                pb.changeCount()
            });
            log::info!(
                "[Rolo] DragWatcher initialized — baseline changeCount={}",
                initial
            );
            Self {
                last_change_count: initial,
                active: false,
            }
        }

        /// Poll the drag pasteboard and return any state transition.
        ///
        /// Call once per tick. `mouse_down` must be the current left-button state
        /// from CoreGraphics — we use it to detect drag end (pasteboard alone
        /// gives us start but not end).
        pub fn poll(&mut self, mouse_down: bool) -> Option<DragEvent> {
            // Read the pasteboard state inside an autorelease pool so NSString /
            // NSArray temporaries don't leak on each tick.
            let (bumped, has_file_url) = autoreleasepool(|_| unsafe {
                let pb = NSPasteboard::pasteboardWithName(NSPasteboardNameDrag);
                let cc = pb.changeCount();

                if cc == self.last_change_count {
                    return (false, false);
                }
                self.last_change_count = cc;

                // Check whether the drag contains file URLs. If it's only text or
                // something else, Rolo doesn't care.
                let has_file = if let Some(types) = pb.types() {
                    let count = types.count();
                    let mut found = false;
                    for i in 0..count {
                        let t = types.objectAtIndex(i);
                        if *t == *NSPasteboardTypeFileURL {
                            found = true;
                            break;
                        }
                    }
                    found
                } else {
                    false
                };

                (true, has_file)
            });

            // Start: pasteboard bumped AND contains files AND mouse is held AND
            // we're not already tracking an active drag. The mouse-down gate
            // filters out programmatic pasteboard changes (e.g. a Copy Files
            // action that doesn't involve a drag).
            if bumped && has_file_url && mouse_down && !self.active {
                self.active = true;
                log::info!("[Rolo] DragWatcher detected file drag START");
                return Some(DragEvent::Started);
            }

            // End: we were tracking a drag and the mouse is no longer held.
            if self.active && !mouse_down {
                self.active = false;
                log::info!("[Rolo] DragWatcher detected file drag END");
                return Some(DragEvent::Ended);
            }

            None
        }
    }
}

#[cfg(target_os = "macos")]
pub use mac::DragWatcher;

// ---------------------------------------------------------------------------
// Non-macOS stub — preserves the type and method shape so tick.rs compiles
// on Linux and Windows for CI. `poll` always returns None.
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "macos"))]
pub struct DragWatcher;

#[cfg(not(target_os = "macos"))]
impl DragWatcher {
    pub fn new() -> Self {
        log::debug!("[Rolo] DragWatcher: stub on non-macOS — global drag detection disabled");
        Self
    }

    pub fn poll(&mut self, _mouse_down: bool) -> Option<DragEvent> {
        None
    }
}
