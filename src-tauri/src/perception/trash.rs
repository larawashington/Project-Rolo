//! `~/.Trash` watcher — emits `TrashAdded` when something Rolo did not eat
//! lands in the bin, and `TrashEmptied` when the bin transitions from
//! `>=1` item to `0` (user-initiated empty).
//!
//! Provenance: we try `mdls -name kMDItemWhereFroms` (set when the file was
//! originally downloaded from the web), but Finder does NOT set that xattr
//! for files it moves to Trash itself — so most user-doc trashes resolve
//! `original_parent = None`. We still emit; the prompt just drops the
//! "from X" clause.
//!
//! Empty detection: `notify` on macOS doesn't deliver per-file remove events
//! for a Finder empty operation, so we poll the directory's file count on
//! the same 500ms cadence as the notify receiver and fire on the N>=1 → 0
//! transition. Rolo never causes that transition (eating ADDS to trash),
//! so any 1→0 we see is the user organizing.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};

use super::{filters, stamp_heartbeat, PerceptionEvent, PerceptionHeartbeat};
use crate::self_trash_ledger;

const POLL_INTERVAL: Duration = Duration::from_millis(500);

pub fn spawn(
    tx: mpsc::Sender<PerceptionEvent>,
    heartbeat: PerceptionHeartbeat,
) -> Option<JoinHandle<()>> {
    let trash_dir = match dirs::home_dir() {
        Some(home) => home.join(".Trash"),
        None => {
            log::warn!("[Rolo perception] no home dir; trash watcher disabled");
            return None;
        }
    };

    thread::Builder::new()
        .name("perception-trash".into())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(trash_dir, tx, heartbeat);
            }));
            if result.is_err() {
                log::error!("[Rolo perception] trash watcher panicked; exiting thread");
            }
        })
        .ok()
}

fn run(trash_dir: PathBuf, tx: mpsc::Sender<PerceptionEvent>, heartbeat: PerceptionHeartbeat) {
    log::info!(
        "[Rolo perception] trash watcher starting on {}",
        trash_dir.display()
    );

    let (fs_tx, fs_rx) = mpsc::channel::<notify::Result<notify::Event>>();
    let mut watcher: RecommendedWatcher = match notify::recommended_watcher(move |res| {
        let _ = fs_tx.send(res);
    }) {
        Ok(w) => w,
        Err(e) => {
            log::warn!(
                "[Rolo perception] failed to construct trash watcher: {} — disabled",
                e
            );
            return;
        }
    };
    if let Err(e) = watcher.watch(&trash_dir, RecursiveMode::NonRecursive) {
        log::warn!(
            "[Rolo perception] failed to watch ~/.Trash: {} — disabled",
            e
        );
        return;
    }

    // Adds we already emitted, so a duplicate FS notification doesn't double-fire.
    let mut emitted: Vec<(PathBuf, Instant)> = Vec::new();
    // Last observed item count, for N→0 empty detection.
    let mut prev_count: usize = count_trash_items(&trash_dir);

    loop {
        let timeout_result = fs_rx.recv_timeout(POLL_INTERVAL);
        let now = Instant::now();
        stamp_heartbeat(&heartbeat);

        emitted.retain(|(_, t)| now.duration_since(*t) < Duration::from_secs(60));

        // Empty-detection runs every tick regardless of whether a notify
        // event arrived — Finder's "Empty Bin" doesn't generate per-file
        // remove events we can rely on.
        let current_count = count_trash_items(&trash_dir);
        if prev_count >= 1 && current_count == 0 {
            log::info!(
                "[Rolo perception] trash empty transition emit: count={}",
                prev_count
            );
            let event = PerceptionEvent::TrashEmptied {
                approximate_count: prev_count as u32,
            };
            if tx.send(event).is_err() {
                return;
            }
        }
        prev_count = current_count;

        let ev = match timeout_result {
            Ok(Ok(ev)) => ev,
            Ok(Err(e)) => {
                log::debug!("[Rolo perception] trash notify error: {}", e);
                continue;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                log::warn!("[Rolo perception] trash notify channel closed; thread exiting");
                return;
            }
        };

        if matches!(
            ev.kind,
            EventKind::Create(_) | EventKind::Modify(notify::event::ModifyKind::Name(_))
        ) {
            for path in ev.paths {
                handle_add(&path, &trash_dir, &tx, &mut emitted, now);
            }
        }
    }
}

fn handle_add(
    path: &Path,
    trash_dir: &Path,
    tx: &mpsc::Sender<PerceptionEvent>,
    emitted: &mut Vec<(PathBuf, Instant)>,
    now: Instant,
) {
    // Only items that landed directly in ~/.Trash — not files inside a
    // trashed folder that FSEvents may also surface.
    let parent = match path.parent() {
        Some(p) => p,
        None => return,
    };
    if parent != trash_dir {
        return;
    }

    if !path.exists() {
        return;
    }

    if self_trash_ledger::was_self_trashed(path, now) {
        return;
    }

    if emitted.iter().any(|(p, _)| p == path) {
        return;
    }

    let filename = match path.file_name().and_then(|n| n.to_str()).map(String::from) {
        Some(f) => f,
        None => return,
    };
    if filename == ".DS_Store" {
        return;
    }
    if !filters::is_filename_safe(&filename) {
        return;
    }

    // Best-effort — most Finder trashes don't carry kMDItemWhereFroms.
    let original_parent = resolve_original_parent(path);

    log::info!(
        "[Rolo perception] trash emit: filename={:?} parent={:?}",
        filename,
        original_parent
    );
    let ev = PerceptionEvent::TrashAdded {
        filename,
        original_parent,
    };
    if tx.send(ev).is_err() {
        return;
    }
    emitted.push((path.to_path_buf(), now));
}

/// Best-effort resolve of the file's original parent folder via
/// `kMDItemWhereFroms` (only present for items downloaded from the web).
/// Returns a short folder label if the source path is under one of the
/// user-doc allowlist roots, else `None`.
fn resolve_original_parent(path: &Path) -> Option<String> {
    let raw = read_where_froms_mdls(path)?;
    let parent_path = raw_to_parent_path(&raw)?;
    classify_parent(&parent_path)
}

fn read_where_froms_mdls(path: &Path) -> Option<String> {
    let out = Command::new("/usr/bin/mdls")
        .args(["-name", "kMDItemWhereFroms", "-raw"])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).to_string();
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed == "(null)" {
        return None;
    }
    Some(trimmed.to_string())
}

fn raw_to_parent_path(raw: &str) -> Option<PathBuf> {
    for line in raw.lines() {
        let line = line.trim().trim_end_matches(',');
        let line = line.trim_start_matches('"').trim_end_matches('"');
        if line.starts_with('/') {
            let p = PathBuf::from(line);
            return p.parent().map(|p| p.to_path_buf());
        }
    }
    None
}

fn classify_parent(parent: &Path) -> Option<String> {
    let home = dirs::home_dir()?;
    let allowlist = [
        ("Documents", home.join("Documents")),
        ("Desktop", home.join("Desktop")),
        ("Downloads", home.join("Downloads")),
        ("Pictures", home.join("Pictures")),
        ("Movies", home.join("Movies")),
        ("Music", home.join("Music")),
    ];
    for (label, root) in allowlist.iter() {
        if parent.starts_with(root) {
            return Some(label.to_string());
        }
    }
    None
}

fn count_trash_items(dir: &Path) -> usize {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    entries
        .flatten()
        .filter(|e| e.file_name() != ".DS_Store")
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_parses_finder_format() {
        let raw = "(\n    \"/Users/x/Documents/foo.md\",\n    \"https://example.com\"\n)";
        let parent = raw_to_parent_path(raw).expect("parent");
        assert_eq!(parent, PathBuf::from("/Users/x/Documents"));
    }

    #[test]
    fn raw_returns_none_on_empty() {
        assert!(raw_to_parent_path("").is_none());
        assert!(raw_to_parent_path("(\n)").is_none());
    }

    #[test]
    fn classify_documents() {
        let home = dirs::home_dir().expect("home");
        let parent = home.join("Documents").join("subdir");
        assert_eq!(classify_parent(&parent).as_deref(), Some("Documents"));
    }

    #[test]
    fn classify_cache_is_rejected() {
        let home = dirs::home_dir().expect("home");
        let parent = home.join("Library/Caches/Foo");
        assert!(classify_parent(&parent).is_none());
    }

    #[test]
    fn count_ignores_ds_store() {
        let tmp =
            std::env::temp_dir().join(format!("rolo-trash-count-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mk tmp");
        std::fs::write(tmp.join(".DS_Store"), b"x").expect("ds");
        assert_eq!(count_trash_items(&tmp), 0);
        std::fs::write(tmp.join("real.txt"), b"x").expect("real");
        assert_eq!(count_trash_items(&tmp), 1);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
