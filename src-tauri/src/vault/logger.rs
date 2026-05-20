use std::fs::{self, File, OpenOptions};
use std::io::{LineWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{Local, NaiveDate};

use super::config::RETENTION_DAYS;
use super::events::ExperienceEvent;

const REOPEN_WARN_INTERVAL: Duration = Duration::from_secs(60);

pub struct ExperienceLogger {
    inner: Mutex<LoggerInner>,
    session_count: AtomicU64,
    last_open_warn: Mutex<Option<Instant>>,
}

struct LoggerInner {
    events_dir: PathBuf,
    current_date: NaiveDate,
    handle: Option<LineWriter<File>>,
}

impl ExperienceLogger {
    /// Open or create today's JSONL in append mode. Best-effort: on open failure
    /// the logger still returns; subsequent `log` calls retry the open.
    pub fn new(events_dir: PathBuf) -> Arc<Self> {
        let today = Local::now().date_naive();
        let handle = open_today(&events_dir, today);
        Arc::new(Self {
            inner: Mutex::new(LoggerInner {
                events_dir,
                current_date: today,
                handle,
            }),
            session_count: AtomicU64::new(0),
            last_open_warn: Mutex::new(None),
        })
    }

    /// Append one event. Errors are logged + dropped, never returned.
    pub fn log(&self, event: &ExperienceEvent) {
        let json = match serde_json::to_string(event) {
            Ok(s) => s,
            Err(e) => {
                log::error!("[Rolo vault] failed to serialize ExperienceEvent: {e}");
                return;
            }
        };

        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());

        // Cheap rollover check before writing — if the local date moved forward
        // since we last opened, rotate first so the event lands in the new day's
        // file (PRD §4.3 / §4.5: prefer rotate-then-write). We compare with `>`
        // (not `!=`) so a clock that briefly steps backward — or a test that
        // forces a future date — does not trigger spurious rotations.
        let today = Local::now().date_naive();
        if today > inner.current_date {
            self.rotate_locked(&mut inner, today);
        }

        if inner.handle.is_none() {
            inner.handle = open_today(&inner.events_dir, inner.current_date);
            if inner.handle.is_none() {
                self.maybe_warn_open_failure();
                return;
            }
        }

        if let Some(handle) = inner.handle.as_mut() {
            let write_result = handle
                .write_all(json.as_bytes())
                .and_then(|_| handle.write_all(b"\n"));
            match write_result {
                Ok(()) => {
                    self.session_count.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    log::error!("[Rolo vault] failed to write ExperienceEvent: {e}");
                    // Drop handle so next call retries open.
                    inner.handle = None;
                }
            }
        }
    }

    /// Detect date rollover; if so, write a SessionEnd into the OLD file, then
    /// flush+close and open the new day's file. Cheap when nothing to do.
    pub fn rotate_if_needed(&self) {
        let today = Local::now().date_naive();
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if today > inner.current_date {
            self.rotate_locked(&mut inner, today);
        }
    }

    /// Force-flush the writer. Called from the shutdown path.
    pub fn flush(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(handle) = inner.handle.as_mut() {
            if let Err(e) = handle.flush() {
                log::error!("[Rolo vault] flush failed: {e}");
            }
        }
    }

    /// Sweep `events/` for files older than RETENTION_DAYS. Returns the count
    /// of files deleted (for telemetry). Per-file errors are logged + skipped.
    pub fn retention_sweep(&self) -> usize {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let dir = inner.events_dir.clone();
        drop(inner);

        let today = Local::now().date_naive();
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) => {
                log::error!(
                    "[Rolo vault] retention sweep cannot read {}: {e}",
                    dir.display()
                );
                return 0;
            }
        };

        let mut deleted = 0usize;
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
                continue;
            };
            if ext != "jsonl" {
                continue;
            }
            // Filename must look like YYYY-MM-DD.jsonl. Anything else (notes.txt,
            // someone's hand-edited file) is left alone.
            let file_date = match NaiveDate::parse_from_str(stem, "%Y-%m-%d") {
                Ok(d) => d,
                Err(e) => {
                    log::warn!(
                        "[Rolo vault] retention sweep skipping unparseable name {}: {e}",
                        path.display()
                    );
                    continue;
                }
            };

            // Files MORE than RETENTION_DAYS old are deleted; exactly-N stays.
            // Future-dated files yield a negative diff and survive.
            let age_days = (today - file_date).num_days();
            if age_days > RETENTION_DAYS {
                match fs::remove_file(&path) {
                    Ok(()) => {
                        deleted += 1;
                        log::info!("[Rolo vault] retention sweep deleted {}", path.display());
                    }
                    Err(e) => {
                        log::error!(
                            "[Rolo vault] retention sweep failed to delete {}: {e}",
                            path.display()
                        );
                    }
                }
            }
        }
        deleted
    }

    pub fn session_event_count(&self) -> u64 {
        self.session_count.load(Ordering::Relaxed)
    }

    /// Internal: rotate the open handle to `new_date`. Caller holds the lock.
    /// Writes a SessionEnd into the OLD file (interactions = current session
    /// count), flushes, closes, then opens the new day's file.
    fn rotate_locked(&self, inner: &mut LoggerInner, new_date: NaiveDate) {
        if let Some(handle) = inner.handle.as_mut() {
            let session_end = ExperienceEvent::SessionEnd {
                event_id: crate::vault::events::new_id(),
                ts: Local::now(),
                idle_total_ms: 0,
                interactions: self.session_count.load(Ordering::Relaxed) as u32,
            };
            match serde_json::to_string(&session_end) {
                Ok(json) => {
                    if let Err(e) = handle
                        .write_all(json.as_bytes())
                        .and_then(|_| handle.write_all(b"\n"))
                    {
                        log::error!("[Rolo vault] failed to write SessionEnd on rotate: {e}");
                    }
                }
                Err(e) => log::error!("[Rolo vault] failed to serialize SessionEnd: {e}"),
            }
            if let Err(e) = handle.flush() {
                log::error!("[Rolo vault] flush failed during rotate: {e}");
            }
        }
        inner.handle = None;
        inner.current_date = new_date;
        inner.handle = open_today(&inner.events_dir, new_date);
        if inner.handle.is_none() {
            self.maybe_warn_open_failure();
        }
        log::info!("[Rolo vault] rotated event log to {new_date}");
    }

    fn maybe_warn_open_failure(&self) {
        let mut last = self
            .last_open_warn
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let now = Instant::now();
        let should_warn = match *last {
            None => true,
            Some(prev) => now.duration_since(prev) >= REOPEN_WARN_INTERVAL,
        };
        if should_warn {
            log::error!(
                "[Rolo vault] event log handle is not open; events will be dropped until re-open succeeds"
            );
            *last = Some(now);
        }
    }

    /// Test-only helper: simulate a date rollover without touching the clock.
    /// Forces the same rotation path `rotate_if_needed` runs in production.
    #[cfg(test)]
    fn rotate_to_date(&self, target_date: NaiveDate) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if target_date != inner.current_date {
            self.rotate_locked(&mut inner, target_date);
        }
    }
}

impl Drop for ExperienceLogger {
    fn drop(&mut self) {
        // Emit SessionEnd on every exit path that unwinds through Drop —
        // covers Cmd+Q, dock-quit, normal shutdown, and any path the
        // window-level CloseRequested handler doesn't see. Writing it here
        // is idempotent: callers should NOT also emit SessionEnd elsewhere.
        let session_count = self.session_count.load(Ordering::Relaxed) as u32;
        let event = ExperienceEvent::SessionEnd {
            event_id: crate::vault::events::new_id(),
            ts: Local::now(),
            idle_total_ms: 0,
            interactions: session_count,
        };
        if let Ok(json) = serde_json::to_string(&event) {
            if let Ok(mut inner) = self.inner.lock() {
                if let Some(handle) = inner.handle.as_mut() {
                    let _ = handle
                        .write_all(json.as_bytes())
                        .and_then(|_| handle.write_all(b"\n"));
                    let _ = handle.flush();
                }
            }
        }
    }
}

/// Open today's JSONL in append mode, wrapped in a `LineWriter` so each
/// event flushes on its trailing `\n` — survives force-kill without losing
/// emitted events. Returns None on failure (logged inside).
fn open_today(events_dir: &Path, date: NaiveDate) -> Option<LineWriter<File>> {
    let path = events_dir.join(format!("{date}.jsonl"));
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(f) => Some(LineWriter::new(f)),
        Err(e) => {
            log::error!(
                "[Rolo vault] failed to open event log {}: {e}",
                path.display()
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::events::{
        CheckinMethod, DismissContext, EatOutcome, ExperienceEvent, MoodSignal,
    };
    use std::sync::Arc;
    use std::thread;
    use tempfile::TempDir;

    fn ts() -> chrono::DateTime<Local> {
        Local::now()
    }

    fn read_lines(path: &std::path::Path) -> Vec<String> {
        let s = fs::read_to_string(path).unwrap_or_default();
        s.lines().map(|l| l.to_string()).collect()
    }

    fn make_logger(dir: &TempDir) -> Arc<ExperienceLogger> {
        let events = dir.path().join("events");
        fs::create_dir_all(&events).unwrap();
        ExperienceLogger::new(events)
    }

    fn today_path(dir: &TempDir) -> PathBuf {
        let today = Local::now().date_naive();
        dir.path().join("events").join(format!("{today}.jsonl"))
    }

    #[test]
    fn append_produces_one_line_per_event() {
        let tmp = TempDir::new().unwrap();
        let logger = make_logger(&tmp);
        for i in 0..3 {
            logger.log(&ExperienceEvent::Drag {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                duration_ms: 100 + i,
            });
        }
        // Drop should flush AND emit a trailing SessionEnd.
        drop(logger);
        let lines = read_lines(&today_path(&tmp));
        assert_eq!(lines.len(), 4, "expected 3 drags + 1 SessionEnd");
        for line in &lines[..3] {
            let parsed: ExperienceEvent = serde_json::from_str(line).expect("each line must parse");
            assert!(matches!(parsed, ExperienceEvent::Drag { .. }));
        }
        let last: ExperienceEvent = serde_json::from_str(&lines[3]).unwrap();
        assert!(matches!(last, ExperienceEvent::SessionEnd { .. }));
    }

    #[test]
    fn date_rollover_emits_session_end_in_old_file() {
        let tmp = TempDir::new().unwrap();
        let logger = make_logger(&tmp);

        let today = Local::now().date_naive();
        let tomorrow = today.succ_opt().unwrap();

        logger.log(&ExperienceEvent::Drag {
            event_id: crate::vault::events::new_id(),
            ts: ts(),
            duration_ms: 500,
        });
        logger.rotate_to_date(tomorrow);

        let today_lines = read_lines(&tmp.path().join("events").join(format!("{today}.jsonl")));
        assert_eq!(today_lines.len(), 2, "old file: 1 drag + 1 session_end");
        let last: ExperienceEvent = serde_json::from_str(&today_lines[1]).unwrap();
        assert!(matches!(last, ExperienceEvent::SessionEnd { .. }));

        // Tomorrow's file should now exist (created by open_today after rotate).
        let tomorrow_path = tmp.path().join("events").join(format!("{tomorrow}.jsonl"));
        assert!(tomorrow_path.exists(), "tomorrow's file should exist");

        logger.log(&ExperienceEvent::Drag {
            event_id: crate::vault::events::new_id(),
            ts: ts(),
            duration_ms: 999,
        });
        drop(logger);

        let tomorrow_lines = read_lines(&tomorrow_path);
        assert_eq!(
            tomorrow_lines.len(),
            2,
            "new file: 1 drag + Drop's SessionEnd"
        );
        let parsed: ExperienceEvent = serde_json::from_str(&tomorrow_lines[0]).unwrap();
        match parsed {
            ExperienceEvent::Drag { duration_ms, .. } => assert_eq!(duration_ms, 999),
            _ => panic!("expected Drag in tomorrow's file"),
        }
        let final_event: ExperienceEvent = serde_json::from_str(&tomorrow_lines[1]).unwrap();
        assert!(matches!(final_event, ExperienceEvent::SessionEnd { .. }));
    }

    #[test]
    fn disk_write_failure_does_not_panic_and_recovers() {
        // Strategy: point the logger at a non-existent dir so open_today fails
        // up front. (Removing the dir AFTER open is unreliable on Unix — an
        // already-open file handle keeps writing to an unlinked inode without
        // surfacing an error.) Then create the dir and verify recovery.
        let tmp = TempDir::new().unwrap();
        let events_dir = tmp.path().join("events_does_not_exist_yet");
        let logger = ExperienceLogger::new(events_dir.clone());

        for _ in 0..5 {
            logger.log(&ExperienceEvent::Drag {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                duration_ms: 1,
            });
        }
        // No panic — count stayed at 0 because all opens (and thus writes) failed.
        assert_eq!(logger.session_event_count(), 0);

        // Recreate the dir and verify recovery on the next log.
        fs::create_dir_all(&events_dir).unwrap();
        logger.log(&ExperienceEvent::Drag {
            event_id: crate::vault::events::new_id(),
            ts: ts(),
            duration_ms: 7,
        });
        drop(logger);
        let today = Local::now().date_naive();
        let path = events_dir.join(format!("{today}.jsonl"));
        let lines = read_lines(&path);
        assert_eq!(lines.len(), 2, "1 recovery write + Drop's SessionEnd");
        let recovered: ExperienceEvent = serde_json::from_str(&lines[0]).unwrap();
        assert!(matches!(
            recovered,
            ExperienceEvent::Drag { duration_ms: 7, .. }
        ));
        let last: ExperienceEvent = serde_json::from_str(&lines[1]).unwrap();
        assert!(matches!(last, ExperienceEvent::SessionEnd { .. }));
    }

    #[test]
    fn concurrent_calls_do_not_interleave() {
        let tmp = TempDir::new().unwrap();
        let logger = make_logger(&tmp);

        let mut handles = Vec::new();
        for thread_id in 0..4 {
            let l = logger.clone();
            handles.push(thread::spawn(move || {
                for i in 0..25 {
                    l.log(&ExperienceEvent::Drag {
                        event_id: crate::vault::events::new_id(),
                        ts: ts(),
                        duration_ms: thread_id * 1000 + i,
                    });
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        drop(logger);

        let lines = read_lines(&today_path(&tmp));
        assert_eq!(lines.len(), 101, "100 drags + 1 Drop SessionEnd");
        for line in lines {
            let _: ExperienceEvent =
                serde_json::from_str(&line).expect("no half-broken interleaved bytes");
        }
    }

    #[test]
    fn session_event_count_only_counts_successful_writes() {
        // Same trick as the disk-failure test: log against a non-existent dir
        // so opens fail (and thus the counter does not advance). Removing a
        // dir while a handle is open is unreliable cross-platform.
        let tmp = TempDir::new().unwrap();
        let events_dir = tmp.path().join("events");
        fs::create_dir_all(&events_dir).unwrap();
        let logger = ExperienceLogger::new(events_dir.clone());

        for _ in 0..3 {
            logger.log(&ExperienceEvent::Drag {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                duration_ms: 1,
            });
        }
        assert_eq!(logger.session_event_count(), 3);

        // Force the handle closed by simulating a write failure path: rebuild
        // the logger pointing at a path that cannot be opened, attempt logs.
        let bad_dir = tmp.path().join("nope_no_dir_here");
        let broken = ExperienceLogger::new(bad_dir);
        for _ in 0..2 {
            broken.log(&ExperienceEvent::Drag {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                duration_ms: 1,
            });
        }
        // Failed writes must not bump the counter.
        assert_eq!(broken.session_event_count(), 0);
    }

    #[test]
    fn retention_sweep_deletes_old_and_keeps_today() {
        let tmp = TempDir::new().unwrap();
        let logger = make_logger(&tmp);
        let events_dir = tmp.path().join("events");

        // Old file: 2026-03-15 is well over 30 days from any plausible test date.
        let old_path = events_dir.join("2026-03-15.jsonl");
        fs::write(&old_path, b"{}\n").unwrap();

        // Today's file already exists from logger init.
        let today_p = today_path(&tmp);
        assert!(today_p.exists());

        let deleted = logger.retention_sweep();
        // Don't assert exact count — the test could run on 2026-03-15 itself.
        // But: the old file should be gone IF it's older than 30d.
        let today = Local::now().date_naive();
        let old_date = NaiveDate::from_ymd_opt(2026, 3, 15).unwrap();
        let age = (today - old_date).num_days();
        if age > RETENTION_DAYS {
            assert!(!old_path.exists(), "old file should be gone");
            assert_eq!(deleted, 1);
        }
        assert!(today_p.exists(), "today's file must survive");

        // Idempotency: second sweep is a no-op.
        let deleted2 = logger.retention_sweep();
        assert_eq!(deleted2, 0, "second sweep should delete nothing");
    }

    #[test]
    fn retention_sweep_ignores_non_matching_filenames() {
        let tmp = TempDir::new().unwrap();
        let logger = make_logger(&tmp);
        let events_dir = tmp.path().join("events");
        fs::write(events_dir.join("notes.txt"), b"hello").unwrap();
        fs::write(events_dir.join("2026-99-99.jsonl"), b"{}\n").unwrap();

        // Should not panic, should not delete these.
        let _ = logger.retention_sweep();
        assert!(events_dir.join("notes.txt").exists());
        assert!(events_dir.join("2026-99-99.jsonl").exists());
    }

    #[test]
    fn future_dated_files_survive_sweep() {
        let tmp = TempDir::new().unwrap();
        let logger = make_logger(&tmp);
        let events_dir = tmp.path().join("events");
        let future = events_dir.join("9999-01-01.jsonl");
        fs::write(&future, b"{}\n").unwrap();

        let _ = logger.retention_sweep();
        assert!(future.exists(), "future-dated file must survive");
    }

    #[test]
    fn log_handles_every_variant() {
        // Smoke test mirroring §11.3 — ensures all variants serialize through
        // the logger path without panicking.
        let tmp = TempDir::new().unwrap();
        let logger = make_logger(&tmp);
        let events = vec![
            ExperienceEvent::Chat {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                session: "s".into(),
                user: "u".into(),
                rolo: "r".into(),
                mood_signal: None,
            },
            ExperienceEvent::Chat {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                session: "s".into(),
                user: "u".into(),
                rolo: "r".into(),
                mood_signal: Some(MoodSignal::Positive),
            },
            ExperienceEvent::Dismiss {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                context: DismissContext::ChatBubble,
                times_dismissed_session: 1,
            },
            ExperienceEvent::Eat {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                files: vec!["a.sql".into()],
                outcome: EatOutcome::Satisfied,
                bytes: 10,
            },
            ExperienceEvent::Checkin {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                question: "q".into(),
                response: "r".into(),
                method: CheckinMethod::Button,
            },
            ExperienceEvent::Drag {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                duration_ms: 1,
            },
            ExperienceEvent::IdleSpeech {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                text: "hi".into(),
                dismissed: false,
            },
            ExperienceEvent::Report {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                message_id: 1,
                rolo_text: "x".into(),
            },
            ExperienceEvent::SessionEnd {
                event_id: crate::vault::events::new_id(),
                ts: ts(),
                idle_total_ms: 0,
                interactions: 0,
            },
        ];
        for e in &events {
            logger.log(e);
        }
        drop(logger);
        let lines = read_lines(&today_path(&tmp));
        // Drop adds an extra SessionEnd at the end on top of every event we logged.
        assert_eq!(lines.len(), events.len() + 1);
    }
}
