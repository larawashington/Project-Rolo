//! File-to-Trash operations — Rolo's digestive system.
//!
//! Uses the `trash` crate which calls `NSFileManager.trashItemAtURL` on macOS.
//! Files are moved to the OS Trash (recoverable), never permanently deleted.

use std::path::Path;

/// Result of a trash operation for a batch of files.
pub struct TrashResult {
    pub trashed_count: usize,
    pub trashed_bytes: u64,
    pub error_message: Option<String>,
}

/// Attempt to trash all files in the batch.
pub fn trash_files(paths: &[String]) -> TrashResult {
    let mut trashed_count = 0;
    let mut trashed_bytes = 0u64;
    let mut last_error = None;

    for path_str in paths {
        let path = Path::new(path_str);

        if !path.exists() {
            last_error = Some("Where'd it go?".to_string());
            continue;
        }

        let size = measure_size(path);

        // Tell the perception trash-watcher this is Rolo eating, not the user
        // moving a file to the bin — keeps Rolo from narrating his own meals.
        crate::self_trash_ledger::record(path);

        match trash::delete(path) {
            Ok(()) => {
                trashed_count += 1;
                trashed_bytes += size;
            }
            Err(e) => {
                eprintln!("[Rolo] Failed to trash {}: {}", path_str, e);
                last_error = Some("I can't eat that!".to_string());
            }
        }
    }

    TrashResult {
        trashed_count,
        trashed_bytes,
        error_message: if trashed_count == 0 { last_error } else { None },
    }
}

fn measure_size(path: &Path) -> u64 {
    if path.is_dir() {
        walkdir_size(path)
    } else {
        path.metadata().map(|m| m.len()).unwrap_or(0)
    }
}

fn walkdir_size(dir: &Path) -> u64 {
    let mut total = 0u64;
    let mut count = 0;
    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            count += 1;
            if count > 10_000 {
                return 0;
            }
        }
    }
    total
}
