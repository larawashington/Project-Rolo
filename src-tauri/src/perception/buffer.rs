//! Single-slot perception buffer with TTL + cooldown.
//!
//! Holds at most one `PerceptionEvent`. New events overwrite older ones
//! (last-write-wins) unless cooldown is active. After an event is consumed
//! for a prompt, the cooldown blocks further pushes for COOLDOWN.

use std::time::{Duration, Instant};

use super::PerceptionEvent;

#[derive(Default)]
pub struct PerceptionBuffer {
    slot: Option<(PerceptionEvent, Instant)>,
    last_consumed: Option<Instant>,
}

impl PerceptionBuffer {
    pub const STALE_AFTER: Duration = Duration::from_secs(120);
    pub const COOLDOWN: Duration = Duration::from_secs(300);

    /// Returns true if the event was parked in the slot, false if the
    /// post-consume COOLDOWN silently dropped it. Callers use this to
    /// decide whether downstream wiring (e.g., speech acceleration) should
    /// react — an event that won't reach the next prompt shouldn't change
    /// the speech timer.
    pub fn push(&mut self, ev: PerceptionEvent, now: Instant) -> bool {
        if let Some(last) = self.last_consumed {
            if now.duration_since(last) < Self::COOLDOWN {
                return false;
            }
        }
        self.slot = Some((ev, now));
        true
    }

    pub fn take_for_prompt(&mut self, now: Instant) -> Option<PerceptionEvent> {
        let (ev, arrived) = self.slot.take()?;
        if now.duration_since(arrived) >= Self::STALE_AFTER {
            return None;
        }
        self.last_consumed = Some(now);
        Some(ev)
    }

    pub fn evict_stale(&mut self, now: Instant) {
        if let Some((_, arrived)) = &self.slot {
            if now.duration_since(*arrived) >= Self::STALE_AFTER {
                self.slot = None;
            }
        }
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.slot.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev() -> PerceptionEvent {
        PerceptionEvent::DownloadAdded {
            filename: "x.pdf".into(),
            ext: "pdf".into(),
            size_bytes: 1,
        }
    }

    #[test]
    fn push_then_take_returns_event() {
        let mut b = PerceptionBuffer::default();
        let t0 = Instant::now();
        b.push(ev(), t0);
        assert!(b.take_for_prompt(t0).is_some());
        assert!(b.is_empty());
    }

    #[test]
    fn overwrite_on_second_push() {
        let mut b = PerceptionBuffer::default();
        let t0 = Instant::now();
        b.push(
            PerceptionEvent::DownloadAdded {
                filename: "a".into(),
                ext: "".into(),
                size_bytes: 0,
            },
            t0,
        );
        b.push(
            PerceptionEvent::DownloadAdded {
                filename: "b".into(),
                ext: "".into(),
                size_bytes: 0,
            },
            t0 + Duration::from_secs(1),
        );
        let taken = b.take_for_prompt(t0 + Duration::from_secs(2)).unwrap();
        match taken {
            PerceptionEvent::DownloadAdded { filename, .. } => assert_eq!(filename, "b"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn cooldown_blocks_push() {
        let mut b = PerceptionBuffer::default();
        let t0 = Instant::now();
        b.push(ev(), t0);
        let _ = b.take_for_prompt(t0); // sets last_consumed
        b.push(ev(), t0 + Duration::from_secs(60));
        assert!(b.is_empty(), "cooldown should have blocked");
    }

    #[test]
    fn cooldown_expires_after_5min() {
        let mut b = PerceptionBuffer::default();
        let t0 = Instant::now();
        b.push(ev(), t0);
        let _ = b.take_for_prompt(t0);
        b.push(ev(), t0 + Duration::from_secs(301));
        assert!(!b.is_empty(), "cooldown should have expired");
    }

    #[test]
    fn stale_event_not_returned() {
        let mut b = PerceptionBuffer::default();
        let t0 = Instant::now();
        b.push(ev(), t0);
        let taken = b.take_for_prompt(t0 + Duration::from_secs(121));
        assert!(taken.is_none());
    }

    #[test]
    fn evict_stale_clears_slot() {
        let mut b = PerceptionBuffer::default();
        let t0 = Instant::now();
        b.push(ev(), t0);
        b.evict_stale(t0 + Duration::from_secs(121));
        assert!(b.is_empty());
    }

    #[test]
    fn evict_stale_keeps_fresh() {
        let mut b = PerceptionBuffer::default();
        let t0 = Instant::now();
        b.push(ev(), t0);
        b.evict_stale(t0 + Duration::from_secs(60));
        assert!(!b.is_empty());
    }
}
