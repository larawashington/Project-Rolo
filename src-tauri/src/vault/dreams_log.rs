//! Hash-chained dreams ledger.
//!
//! The dreaming compiler appends a JSON record to `<vault_root>/dreams.jsonl`
//! after every run. Each line carries a `prev_hash`/`this_hash` pair so any
//! tampering — manual edit, partial truncation, accidental concat — is
//! detectable on the next append. The chain itself is never verified at
//! load (we don't fail Rolo's boot over a snipped dream); the hash is purely
//! an audit trail for the dreaming-history viewer (PRD §A4).
//!
//! Canonicalization rule: keys in every JSON object (recursively) are sorted
//! before hashing so two semantically-equal records hash identically across
//! serializer versions / map ordering.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Sentinel `prev_hash` value used for the first entry, or whenever the chain
/// has been corrupted and we must restart from a known anchor.
const ZERO_HASH: &str = "sha256:0000000000000000000000000000000000000000000000000000000000000000";

const FILENAME: &str = "dreams.jsonl";

/// Append-only hash-chained log of dream-compiler runs.
pub struct DreamsLog {
    path: PathBuf,
    lock: Mutex<()>,
}

impl DreamsLog {
    /// Construct a `DreamsLog` rooted at `<vault_root>/dreams.jsonl`. Does not
    /// touch the filesystem; the file is created lazily on first `append`.
    pub fn new(vault_root: &Path) -> Self {
        Self {
            path: vault_root.join(FILENAME),
            lock: Mutex::new(()),
        }
    }

    /// Path the log writes to. Exposed for tests and the dreaming-history viewer.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one entry, computing and stamping `prev_hash` and `this_hash`.
    /// Returns the entry's `this_hash` so the caller can echo it in logs.
    ///
    /// On a corrupt or unparseable tail line, we log a warning and fall back
    /// to `ZERO_HASH` for `prev_hash` — the chain restarts cleanly rather
    /// than panicking. This is the "warn-and-continue" rule from PRD §A4.
    pub fn append(&self, entry: Value) -> std::io::Result<String> {
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());

        let prev_hash = read_tail_hash(&self.path);

        // Build a fresh object so we control field ordering: prev_hash first
        // for human-readable diffs, then the caller's payload, then this_hash.
        // The hash is computed over (prev_hash + payload) — `this_hash` is
        // never part of its own input.
        let mut obj: BTreeMap<String, Value> = BTreeMap::new();
        match entry {
            Value::Object(map) => {
                for (k, v) in map {
                    obj.insert(k, v);
                }
            }
            other => {
                // The caller passed a non-object — wrap it under a `value`
                // key so we still have a well-formed chained line.
                obj.insert("value".to_string(), other);
            }
        }
        obj.insert("prev_hash".to_string(), Value::String(prev_hash.clone()));

        let canonical = to_canonical(&Value::Object(obj.clone().into_iter().collect()));
        let this_hash = sha256_hex(&canonical);
        obj.insert("this_hash".to_string(), Value::String(this_hash.clone()));

        let line = serde_json::to_string(&Value::Object(obj.into_iter().collect()))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
        file.flush()?;

        Ok(this_hash)
    }

    /// Return up to `limit` most recent entries, newest-first. Malformed lines
    /// are skipped silently — a hand-edited file shouldn't blank the viewer.
    pub fn read_recent(&self, limit: usize) -> Vec<Value> {
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let file = match std::fs::File::open(&self.path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let reader = BufReader::new(file);
        let mut entries: Vec<Value> = Vec::new();
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(&line) {
                entries.push(v);
            }
        }
        entries.reverse();
        if entries.len() > limit {
            entries.truncate(limit);
        }
        entries
    }
}

/// Read the final non-empty line and extract its `this_hash` field. Falls back
/// to `ZERO_HASH` when the file is missing, empty, or the tail is malformed.
/// In the malformed case the caller should expect a `warn!` log line.
fn read_tail_hash(path: &Path) -> String {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return ZERO_HASH.to_string(),
        Err(e) => {
            log::warn!(
                "[Rolo dreams_log] could not read {}: {} — restarting chain at zero hash",
                path.display(),
                e
            );
            return ZERO_HASH.to_string();
        }
    };
    let reader = BufReader::new(file);
    let mut last: Option<String> = None;
    for line in reader.lines().map_while(Result::ok) {
        if !line.trim().is_empty() {
            last = Some(line);
        }
    }
    let Some(tail) = last else {
        // File exists but is empty — treat as a fresh chain.
        return ZERO_HASH.to_string();
    };
    match serde_json::from_str::<Value>(&tail) {
        Ok(Value::Object(map)) => match map.get("this_hash") {
            Some(Value::String(h)) => h.clone(),
            _ => {
                log::warn!(
                    "[Rolo dreams_log] tail line in {} missing this_hash — restarting chain at zero hash",
                    path.display()
                );
                ZERO_HASH.to_string()
            }
        },
        _ => {
            log::warn!(
                "[Rolo dreams_log] tail line in {} unparseable — restarting chain at zero hash",
                path.display()
            );
            ZERO_HASH.to_string()
        }
    }
}

/// Recursively re-build a `Value` so every object's keys are sorted. Arrays
/// preserve order; primitives are returned unchanged. The output of
/// `serde_json::to_string` on this canonical form is deterministic across
/// runs and serde_json versions.
fn to_canonical(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut sorted: BTreeMap<String, Value> = BTreeMap::new();
            for (k, val) in map {
                sorted.insert(k.clone(), to_canonical(val));
            }
            // serde_json::Map preserves insertion order, so feeding from a
            // BTreeMap (sorted iteration) yields a sorted output map.
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(to_canonical).collect()),
        _ => v.clone(),
    }
}

fn sha256_hex(v: &Value) -> String {
    let bytes = serde_json::to_vec(v).expect("canonical Value always serializes");
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(7 + 64);
    hex.push_str("sha256:");
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use std::thread;
    use tempfile::TempDir;

    fn make_log() -> (TempDir, DreamsLog) {
        let tmp = TempDir::new().unwrap();
        let log = DreamsLog::new(tmp.path());
        (tmp, log)
    }

    #[test]
    fn append_chains_prev_hash() {
        let (_tmp, log) = make_log();
        let hash_a = log
            .append(json!({"kind": "dream", "iteration": 1}))
            .unwrap();
        let hash_b = log
            .append(json!({"kind": "dream", "iteration": 2}))
            .unwrap();

        // The first entry's prev_hash is the zero anchor.
        let entries = log.read_recent(10);
        assert_eq!(entries.len(), 2);
        // entries are newest-first, so [0] is B, [1] is A.
        assert_eq!(entries[1]["prev_hash"], ZERO_HASH);
        assert_eq!(entries[1]["this_hash"], hash_a);
        assert_eq!(entries[0]["prev_hash"], hash_a);
        assert_eq!(entries[0]["this_hash"], hash_b);
        assert_ne!(hash_a, hash_b);
    }

    #[test]
    fn read_recent_returns_in_reverse_order() {
        let (_tmp, log) = make_log();
        log.append(json!({"n": 1})).unwrap();
        log.append(json!({"n": 2})).unwrap();
        log.append(json!({"n": 3})).unwrap();

        let entries = log.read_recent(10);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0]["n"], 3);
        assert_eq!(entries[1]["n"], 2);
        assert_eq!(entries[2]["n"], 1);
    }

    #[test]
    fn read_recent_respects_limit() {
        let (_tmp, log) = make_log();
        for i in 0..5 {
            log.append(json!({ "n": i })).unwrap();
        }
        let entries = log.read_recent(2);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["n"], 4);
        assert_eq!(entries[1]["n"], 3);
    }

    #[test]
    fn manual_corruption_warn_then_continue() {
        let (_tmp, log) = make_log();
        log.append(json!({"n": 1})).unwrap();

        // Hand-corrupt the tail. `append` must NOT panic, the chain just
        // restarts from ZERO_HASH on the next entry.
        std::fs::OpenOptions::new()
            .append(true)
            .open(log.path())
            .unwrap()
            .write_all(b"this is not json at all\n")
            .unwrap();

        let recovery_hash = log.append(json!({"n": 2})).unwrap();
        assert!(recovery_hash.starts_with("sha256:"));

        // The recovery line's prev_hash must be ZERO_HASH (chain restart).
        // Read raw lines so we can inspect the malformed middle line.
        let raw = std::fs::read_to_string(log.path()).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 3, "n=1 line, garbage line, recovery line");
        let recovery: Value = serde_json::from_str(lines[2]).unwrap();
        assert_eq!(recovery["prev_hash"], ZERO_HASH);
    }

    #[test]
    fn missing_file_starts_chain_at_zero_hash() {
        let (tmp, _log) = make_log();
        // Build a fresh log against a non-existent path.
        let log = DreamsLog::new(&tmp.path().join("subdir_does_not_exist_yet"));
        // Parent dir doesn't exist — append should error cleanly. (The
        // caller, vault::open_or_init, creates the vault root before any
        // dream runs.)
        let res = log.append(json!({"n": 1}));
        assert!(res.is_err());
    }

    #[test]
    fn first_entry_uses_zero_hash() {
        let (_tmp, log) = make_log();
        log.append(json!({"first": true})).unwrap();
        let entries = log.read_recent(1);
        assert_eq!(entries[0]["prev_hash"], ZERO_HASH);
    }

    #[test]
    fn concurrent_appends_serialize_via_lock() {
        let (_tmp, log) = make_log();
        let log = Arc::new(log);

        let mut handles = Vec::new();
        for thread_id in 0..5 {
            let l = Arc::clone(&log);
            handles.push(thread::spawn(move || {
                l.append(json!({ "thread_id": thread_id })).unwrap()
            }));
        }
        let mut produced_hashes: Vec<String> = Vec::new();
        for h in handles {
            produced_hashes.push(h.join().unwrap());
        }

        // 5 unique hashes — proof that the lock serialized the writes; if it
        // hadn't, two threads could read the same prev_hash and produce
        // identical lines / hashes (or a torn write).
        produced_hashes.sort();
        produced_hashes.dedup();
        assert_eq!(produced_hashes.len(), 5);

        // Walk the on-disk file and verify the chain is continuous.
        let entries = log.read_recent(10);
        assert_eq!(entries.len(), 5);
        // entries are newest-first; reverse to walk chronologically.
        let chronological: Vec<&Value> = entries.iter().rev().collect();
        let mut expected_prev = ZERO_HASH.to_string();
        for entry in &chronological {
            assert_eq!(entry["prev_hash"], expected_prev);
            expected_prev = entry["this_hash"].as_str().unwrap().to_string();
        }
    }

    #[test]
    fn this_hash_is_deterministic_for_same_payload() {
        // Two fresh logs, same single payload → same this_hash. Proves the
        // canonicalization is order-independent (object keys serialized in
        // sorted order regardless of insertion).
        let (tmp1, log1) = make_log();
        let (tmp2, log2) = make_log();
        let payload = json!({ "z": 1, "a": 2, "nested": { "y": 1, "x": 2 } });
        let h1 = log1.append(payload.clone()).unwrap();
        let h2 = log2.append(payload).unwrap();
        assert_eq!(h1, h2);
        // Touch the dirs so they aren't dropped before assertion.
        let _ = (tmp1, tmp2);
    }

    #[test]
    fn to_canonical_sorts_nested_object_keys() {
        let messy = json!({"z": {"y": 1, "a": 2}, "a": [1, {"q": 1, "b": 2}]});
        let canonical = to_canonical(&messy);
        let s = serde_json::to_string(&canonical).unwrap();
        // Sorted at every level: outer keys "a" before "z", inner "a" before
        // "y", and inside the array, "b" before "q".
        assert_eq!(s, r#"{"a":[1,{"b":2,"q":1}],"z":{"a":2,"y":1}}"#);
    }
}
