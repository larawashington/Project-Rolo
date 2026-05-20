//! `~/Downloads` watcher — emits `DownloadAdded` when a file lands and
//! stays stable for ≥2 seconds.
//!
//! Uses FSEvents via `notify`. Browsers stream files as `.crdownload`,
//! `.part`, or `Unconfirmed N.tmp`; we ignore those until the final rename.
//! Permission failures (TCC denial) log one warn and exit cleanly — no
//! retry, no UI surfacing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use notify::{
    event::{CreateKind, ModifyKind, RenameMode},
    EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};

use super::{filters, stamp_heartbeat, PerceptionEvent, PerceptionHeartbeat};

const STABLE_AFTER: Duration = Duration::from_secs(2);
const ABANDON_AFTER: Duration = Duration::from_secs(60);
const POLL_INTERVAL: Duration = Duration::from_millis(500);

pub fn spawn(
    tx: mpsc::Sender<PerceptionEvent>,
    heartbeat: PerceptionHeartbeat,
) -> Option<JoinHandle<()>> {
    let downloads = match dirs::download_dir() {
        Some(p) => p,
        None => {
            log::warn!(
                "[Rolo perception] dirs::download_dir() returned None; downloads watcher disabled"
            );
            return None;
        }
    };

    thread::Builder::new()
        .name("perception-downloads".into())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(downloads, tx, heartbeat);
            }));
            if result.is_err() {
                log::error!("[Rolo perception] downloads watcher panicked; exiting thread");
            }
        })
        .ok()
}

fn run(downloads: PathBuf, tx: mpsc::Sender<PerceptionEvent>, heartbeat: PerceptionHeartbeat) {
    log::info!(
        "[Rolo perception] downloads watcher starting on {}",
        downloads.display()
    );

    let (fs_tx, fs_rx) = mpsc::channel::<notify::Result<notify::Event>>();
    let mut watcher: RecommendedWatcher = match notify::recommended_watcher(move |res| {
        let _ = fs_tx.send(res);
    }) {
        Ok(w) => w,
        Err(e) => {
            log::warn!(
                "[Rolo perception] failed to construct downloads watcher: {} — disabled until restart",
                e
            );
            return;
        }
    };
    if let Err(e) = watcher.watch(&downloads, RecursiveMode::NonRecursive) {
        log::warn!(
            "[Rolo perception] failed to watch ~/Downloads (TCC denied?): {} — disabled until restart",
            e
        );
        return;
    }

    // path -> (last-observed-size, last-changed-at)
    let mut tracking: HashMap<PathBuf, (u64, Instant)> = HashMap::new();
    let mut emitted: HashMap<PathBuf, Instant> = HashMap::new();

    loop {
        stamp_heartbeat(&heartbeat);
        // Drain any pending FS events without blocking.
        loop {
            match fs_rx.recv_timeout(POLL_INTERVAL) {
                Ok(Ok(ev)) => observe_event(ev, &mut tracking),
                Ok(Err(e)) => log::debug!("[Rolo perception] downloads notify error: {}", e),
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    log::warn!("[Rolo perception] downloads notify channel closed; thread exiting");
                    return;
                }
            }
        }

        let now = Instant::now();
        // Promote stable files to events; drop abandoned ones.
        let paths: Vec<PathBuf> = tracking.keys().cloned().collect();
        for path in paths {
            let Some((tracked_size, changed_at)) = tracking.get(&path).copied() else {
                continue;
            };

            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => {
                    // File removed before stabilising — drop.
                    tracking.remove(&path);
                    continue;
                }
            };
            let cur_size = if meta.is_dir() { 0 } else { meta.len() };

            if cur_size != tracked_size {
                // Still being written — reset the stability clock.
                tracking.insert(path, (cur_size, now));
                continue;
            }

            if now.duration_since(changed_at) < STABLE_AFTER {
                continue;
            }

            if now.duration_since(changed_at) > ABANDON_AFTER {
                tracking.remove(&path);
                continue;
            }

            // Stable — try to emit. Don't re-emit the same path within the
            // abandon window if we already fired (browsers can touch a file
            // multiple times after rename).
            tracking.remove(&path);

            // Dedupe: a file just emitted within ~60s shouldn't fire again.
            if let Some(prev) = emitted.get(&path) {
                if now.duration_since(*prev) < ABANDON_AFTER {
                    continue;
                }
            }

            let Some(filename) = path.file_name().and_then(|n| n.to_str()).map(String::from) else {
                continue;
            };
            if !is_acceptable_filename(&filename) {
                continue;
            }
            if !filters::is_filename_safe(&filename) {
                continue;
            }

            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(String::from)
                .unwrap_or_else(|| {
                    if meta.is_dir() {
                        "folder".into()
                    } else {
                        String::new()
                    }
                });

            log::info!(
                "[Rolo perception] download emit: filename={:?} size={}",
                filename,
                cur_size
            );
            let ev = PerceptionEvent::DownloadAdded {
                filename,
                ext,
                size_bytes: cur_size,
            };
            if tx.send(ev).is_err() {
                log::info!("[Rolo perception] channel closed; downloads thread exiting");
                return;
            }
            emitted.insert(path, now);
        }

        // GC the emitted-dedupe map.
        emitted.retain(|_, t| now.duration_since(*t) < Duration::from_secs(120));
    }
}

fn observe_event(ev: notify::Event, tracking: &mut HashMap<PathBuf, (u64, Instant)>) {
    let is_relevant = matches!(
        ev.kind,
        EventKind::Create(CreateKind::File)
            | EventKind::Create(CreateKind::Any)
            | EventKind::Create(CreateKind::Folder)
            | EventKind::Modify(ModifyKind::Data(_))
            | EventKind::Modify(ModifyKind::Name(RenameMode::To))
            | EventKind::Modify(ModifyKind::Name(RenameMode::Any))
            | EventKind::Modify(ModifyKind::Any)
    );
    if !is_relevant {
        return;
    }
    for path in ev.paths {
        if !path.exists() {
            continue;
        }
        let size = std::fs::metadata(&path)
            .map(|m| if m.is_dir() { 0 } else { m.len() })
            .unwrap_or(0);
        tracking.insert(path, (size, Instant::now()));
    }
}

fn is_acceptable_filename(name: &str) -> bool {
    if name.starts_with('.') {
        return false;
    }
    if name.starts_with("Unconfirmed ") {
        return false;
    }
    let lower = name.to_lowercase();
    !(lower.ends_with(".crdownload")
        || lower.ends_with(".part")
        || lower.ends_with(".tmp")
        || lower.ends_with(".download"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceptable_filename_normal() {
        assert!(is_acceptable_filename("report.pdf"));
        assert!(is_acceptable_filename("vacation.zip"));
    }

    #[test]
    fn acceptable_filename_rejects_in_progress() {
        assert!(!is_acceptable_filename("foo.crdownload"));
        assert!(!is_acceptable_filename("foo.part"));
        assert!(!is_acceptable_filename("foo.tmp"));
        assert!(!is_acceptable_filename("foo.download"));
        assert!(!is_acceptable_filename(".hidden"));
        assert!(!is_acceptable_filename("Unconfirmed 1234.crdownload"));
    }

    #[test]
    fn acceptable_filename_is_case_insensitive_on_ext() {
        assert!(!is_acceptable_filename("foo.CRDOWNLOAD"));
    }
}
