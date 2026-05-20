//! Frontmost-app poller — emits `ForegroundChanged` when an app stays
//! foreground for ≥10 seconds.
//!
//! NSWorkspace gives us app name + PID; CGWindowList gives us the topmost
//! on-screen window title owned by that PID. We poll every 2 seconds — well
//! under the 10-second stability gate, so quick alt-tab churn never emits.

use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use core_foundation::array::{CFArrayGetCount, CFArrayGetValueAtIndex};
use core_foundation::base::{CFTypeRef, TCFType};
use core_foundation::dictionary::{CFDictionaryGetValueIfPresent, CFDictionaryRef};
use core_foundation::number::{CFNumber, CFNumberRef};
use core_foundation::string::{CFString, CFStringRef};
use core_graphics::window::{
    copy_window_info, kCGNullWindowID, kCGWindowListExcludeDesktopElements,
    kCGWindowListOptionOnScreenOnly, kCGWindowName, kCGWindowOwnerPID,
};
use objc2::rc::autoreleasepool;
use objc2_app_kit::NSWorkspace;

use super::{filters, stamp_heartbeat, PerceptionEvent, PerceptionHeartbeat};

const POLL_INTERVAL: Duration = Duration::from_secs(2);
const STABILITY: Duration = Duration::from_secs(10);

pub fn spawn(
    tx: mpsc::Sender<PerceptionEvent>,
    heartbeat: PerceptionHeartbeat,
) -> Option<JoinHandle<()>> {
    let handle = thread::Builder::new()
        .name("perception-foreground".into())
        .spawn(move || {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(tx, heartbeat)));
            if result.is_err() {
                log::error!("[Rolo perception] foreground poller panicked; exiting thread");
            }
        })
        .ok();
    if handle.is_none() {
        log::warn!("[Rolo perception] failed to spawn foreground poller thread");
    }
    handle
}

fn run(tx: mpsc::Sender<PerceptionEvent>, heartbeat: PerceptionHeartbeat) {
    log::info!("[Rolo perception] foreground poller started");
    let mut current: Option<(String, Option<String>, Instant)> = None;
    let mut last_emitted: Option<(String, Option<String>)> = None;

    loop {
        thread::sleep(POLL_INTERVAL);
        stamp_heartbeat(&heartbeat);

        let Some((app_name, pid)) = read_frontmost_app() else {
            continue;
        };

        if filters::is_app_sensitive(&app_name) {
            continue;
        }

        let raw_title = read_window_title(pid);
        let safe_title = raw_title.and_then(|t| {
            if filters::is_window_title_safe(&t) {
                Some(t)
            } else {
                None
            }
        });

        let now = Instant::now();
        let changed = match &current {
            Some((a, t, _)) => {
                a != &app_name || title_compare_key(t) != title_compare_key(&safe_title)
            }
            None => true,
        };

        if changed {
            current = Some((app_name, safe_title, now));
            continue;
        }

        let became_at = current.as_ref().unwrap().2;
        if now.duration_since(became_at) < STABILITY {
            continue;
        }

        let (a, t, _) = current.as_ref().unwrap();
        let tuple = (a.clone(), t.clone());
        if last_emitted
            .as_ref()
            .map(|(la, lt)| la == &tuple.0 && title_compare_key(lt) == title_compare_key(&tuple.1))
            .unwrap_or(false)
        {
            continue;
        }
        log::info!(
            "[Rolo perception] foreground emit: app={:?} title={:?}",
            tuple.0,
            tuple.1
        );
        let ev = PerceptionEvent::ForegroundChanged {
            app_name: tuple.0.clone(),
            window_title: tuple.1.clone(),
        };
        if tx.send(ev).is_err() {
            log::info!("[Rolo perception] channel closed; foreground thread exiting");
            return;
        }
        last_emitted = Some(tuple);
    }
}

fn read_frontmost_app() -> Option<(String, i32)> {
    autoreleasepool(|_| unsafe {
        let ws = NSWorkspace::sharedWorkspace();
        let app = ws.frontmostApplication()?;
        let name_ns = app.localizedName()?;
        let pid = app.processIdentifier();
        Some((name_ns.to_string(), pid))
    })
}

fn read_window_title(pid: i32) -> Option<String> {
    let option = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements;
    let arr = copy_window_info(option, kCGNullWindowID)?;
    let arr_ref = arr.as_concrete_TypeRef();
    let count = unsafe { CFArrayGetCount(arr_ref) };

    for i in 0..count {
        let dict_ptr = unsafe { CFArrayGetValueAtIndex(arr_ref, i) } as CFDictionaryRef;
        if dict_ptr.is_null() {
            continue;
        }
        // Filter by PID.
        let owner_pid = unsafe { dict_get_i32(dict_ptr, kCGWindowOwnerPID) };
        if owner_pid != Some(pid) {
            continue;
        }
        // Read title. Skip empty names (background helper windows have them);
        // try the next window owned by this PID.
        if let Some(name) = unsafe { dict_get_string(dict_ptr, kCGWindowName) } {
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    None
}

/// Comparison key for window-title change detection. Strips leading
/// non-alphanumeric characters (whitespace, spinner glyphs like `⣾⣷⣯⡿…`,
/// status dots, etc.) so apps like Ghostty that animate a prefix in the
/// title don't trip the 10s stability gate every tick.
fn title_compare_key(title: &Option<String>) -> Option<String> {
    title.as_ref().map(|t| {
        t.trim_start_matches(|c: char| !c.is_alphanumeric())
            .to_string()
    })
}

unsafe fn dict_get_i32(dict: CFDictionaryRef, key: CFStringRef) -> Option<i32> {
    let mut value: CFTypeRef = std::ptr::null();
    let found =
        CFDictionaryGetValueIfPresent(dict, key as *const _, &mut value as *mut _ as *mut _);
    if found == 0 || value.is_null() {
        return None;
    }
    // value is a CFNumberRef under "get" semantics.
    let n = CFNumber::wrap_under_get_rule(value as CFNumberRef);
    n.to_i32()
}

unsafe fn dict_get_string(dict: CFDictionaryRef, key: CFStringRef) -> Option<String> {
    let mut value: CFTypeRef = std::ptr::null();
    let found =
        CFDictionaryGetValueIfPresent(dict, key as *const _, &mut value as *mut _ as *mut _);
    if found == 0 || value.is_null() {
        return None;
    }
    let s = CFString::wrap_under_get_rule(value as CFStringRef);
    Some(s.to_string())
}
