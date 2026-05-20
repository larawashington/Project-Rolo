//! User idle-time detection.
//!
//! The dreaming compiler kicks off when the user has been idle long enough that
//! pegging the CPU on a 4B-param model won't disturb her work. macOS exposes
//! this directly via `CGEventSourceSecondsSinceLastEventType`. Other platforms
//! are stubbed: we return `None` so the dreaming scheduler can fall back to a
//! pure timer (PRD §A5).

/// Seconds since the last user input (any HID event) on the host system.
///
/// Returns `None` on platforms without a supported implementation; callers
/// should treat `None` as "I don't know" rather than "user is here" or "user
/// is away" — the safe behavior is usually to defer running heavy work.
pub fn seconds_since_last_input() -> Option<f64> {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: `CGEventSourceSecondsSinceLastEventType` is a pure read of
        // a system counter — no allocation, no callbacks, no thread-affinity
        // requirements. Crashing here would mean Quartz itself has gone, in
        // which case Rolo has bigger problems than scheduling dreams.
        const K_CG_EVENT_SOURCE_STATE_HID_SYSTEM_STATE: i32 = 1;
        const K_CG_ANY_INPUT_EVENT_TYPE: u32 = 0xFFFFFFFF;

        extern "C" {
            fn CGEventSourceSecondsSinceLastEventType(source_state: i32, event_type: u32) -> f64;
        }

        let secs = unsafe {
            CGEventSourceSecondsSinceLastEventType(
                K_CG_EVENT_SOURCE_STATE_HID_SYSTEM_STATE,
                K_CG_ANY_INPUT_EVENT_TYPE,
            )
        };
        // Quartz returns negative values only on internal failure — guard
        // against that so callers always see a non-negative or `None`.
        if secs.is_finite() && secs >= 0.0 {
            Some(secs)
        } else {
            None
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn returns_some_on_macos() {
        // Sanity: the call returns a finite, non-negative number. We don't
        // assert a specific value — the test runner itself has been moving
        // the cursor recently so the count is whatever it is.
        let secs = seconds_since_last_input().expect("macOS should always have a counter");
        assert!(
            secs.is_finite() && secs >= 0.0,
            "got nonsensical idle time: {secs}"
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn returns_none_off_macos() {
        assert!(seconds_since_last_input().is_none());
    }
}
