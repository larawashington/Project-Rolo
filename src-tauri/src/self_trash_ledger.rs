//! Cross-module ledger of paths Rolo recently trashed himself.
//!
//! The eat-files mutation (`trash::trash_files`) calls `record` before
//! `trash::delete`; the trash watcher (`perception::trash`) calls
//! `was_self_trashed` to skip events that came from Rolo's own snacks.
//!
//! Lives in its own module so neither side depends on the other.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const TTL: Duration = Duration::from_secs(60);

fn ledger() -> &'static Mutex<HashMap<PathBuf, Instant>> {
    static LEDGER: OnceLock<Mutex<HashMap<PathBuf, Instant>>> = OnceLock::new();
    LEDGER.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn record(path: &Path) {
    let mut g = match ledger().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    g.insert(path.to_path_buf(), Instant::now());
}

pub fn was_self_trashed(path: &Path, now: Instant) -> bool {
    let mut g = match ledger().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    g.retain(|_, t| now.duration_since(*t) < TTL);
    g.contains_key(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn record_then_check_hits() {
        let path = p("/tmp/self-trash-test-a.txt");
        record(&path);
        assert!(was_self_trashed(&path, Instant::now()));
    }

    #[test]
    fn check_misses_when_not_recorded() {
        let path = p("/tmp/self-trash-test-not-recorded.txt");
        assert!(!was_self_trashed(&path, Instant::now()));
    }

    #[test]
    fn ages_out_after_60s() {
        let path = p("/tmp/self-trash-test-c.txt");
        record(&path);
        let future = Instant::now() + Duration::from_secs(61);
        assert!(!was_self_trashed(&path, future));
    }
}
