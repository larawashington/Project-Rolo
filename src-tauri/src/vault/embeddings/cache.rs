//! Hash-keyed embedding cache. PRD §7.4.
//!
//! On-disk format is append-only `cache.jsonl`, one JSON object per line:
//! ```json
//! {"ns":"<model>:<digest_hex>:<dim>","hash":"<sha256_hex>","vec":[…f32…]}
//! ```
//!
//! Namespace gates everything. On `open`, only entries whose `ns` matches the
//! current namespace are loaded into RAM. Foreign-namespace entries are left
//! untouched in the file so multiple namespaces can coexist (e.g. during a
//! model swap mid-rebuild). `invalidate_namespace` rewrites the file with the
//! current namespace's lines stripped.
//!
//! Malformed lines are tolerated: drop and continue. A miss costs one
//! re-embed; a bad cache shouldn't block dreaming.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::vault::atomic::atomic_write;
use crate::vault::embeddings::hex::{decode_hex_32, encode_hex_32};

/// 50 MB threshold for `compact_if_oversized` (Risk R5).
const COMPACT_THRESHOLD_BYTES: u64 = 50 * 1024 * 1024;

pub struct EmbeddingCache {
    namespace: String,
    map: HashMap<[u8; 32], Vec<f32>>,
    log_path: PathBuf,
}

impl EmbeddingCache {
    /// Open or create the cache. Loads only entries matching `namespace` into
    /// memory. Missing file is fine. Malformed lines are warned and skipped.
    pub fn open(dir: &Path, namespace: String) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let log_path = dir.join("cache.jsonl");

        let mut map: HashMap<[u8; 32], Vec<f32>> = HashMap::new();
        if log_path.exists() {
            let f = OpenOptions::new().read(true).open(&log_path)?;
            let reader = BufReader::new(f);
            for (lineno, line) in reader.lines().enumerate() {
                let line = match line {
                    Ok(l) => l,
                    Err(e) => {
                        tracing::warn!(
                            target: "rolo::vault::embeddings",
                            "cache read error at line {}: {}",
                            lineno + 1,
                            e
                        );
                        continue;
                    }
                };
                if line.trim().is_empty() {
                    continue;
                }
                let parsed: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => {
                        tracing::warn!(
                            target: "rolo::vault::embeddings",
                            "dropping malformed cache line {}",
                            lineno + 1
                        );
                        continue;
                    }
                };
                let ns = parsed.get("ns").and_then(|v| v.as_str()).unwrap_or("");
                if ns != namespace {
                    continue;
                }
                let hash_str = match parsed.get("hash").and_then(|v| v.as_str()) {
                    Some(h) => h,
                    None => continue,
                };
                let hash = match decode_hex_32(hash_str) {
                    Some(h) => h,
                    None => continue,
                };
                let vec_arr = match parsed.get("vec").and_then(|v| v.as_array()) {
                    Some(a) => a,
                    None => continue,
                };
                let mut vec_f: Vec<f32> = Vec::with_capacity(vec_arr.len());
                let mut bad = false;
                for v in vec_arr {
                    match v.as_f64() {
                        Some(f) => vec_f.push(f as f32),
                        None => {
                            bad = true;
                            break;
                        }
                    }
                }
                if bad {
                    continue;
                }
                map.insert(hash, vec_f);
            }
        }

        Ok(Self {
            namespace,
            map,
            log_path,
        })
    }

    pub fn get(&self, hash: &[u8; 32]) -> Option<&[f32]> {
        self.map.get(hash).map(|v| v.as_slice())
    }

    /// Insert into memory and append a JSON line to disk. Append-only; a crash
    /// after the line lands but before the next write leaves a valid cache.
    pub fn put(&mut self, hash: [u8; 32], vec: Vec<f32>) -> io::Result<()> {
        if let Some(parent) = self.log_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let line = json!({
            "ns": self.namespace,
            "hash": encode_hex_32(&hash),
            "vec": vec,
        })
        .to_string();
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_all()?;
        self.map.insert(hash, vec);
        Ok(())
    }

    /// Drop all in-memory entries and rewrite `cache.jsonl` with all
    /// foreign-namespace lines preserved. Used when the model digest changes
    /// (PRD §7.9 step 5).
    pub fn invalidate_namespace(&mut self) -> io::Result<()> {
        let mut kept: Vec<String> = Vec::new();
        if self.log_path.exists() {
            let f = OpenOptions::new().read(true).open(&self.log_path)?;
            let reader = BufReader::new(f);
            for line in reader.lines().map_while(Result::ok) {
                if line.trim().is_empty() {
                    continue;
                }
                let parsed: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let ns = parsed.get("ns").and_then(|v| v.as_str()).unwrap_or("");
                if ns != self.namespace {
                    kept.push(line);
                }
            }
        }
        let mut buf = String::new();
        for l in &kept {
            buf.push_str(l);
            buf.push('\n');
        }
        atomic_write(&self.log_path, buf.as_bytes())?;
        self.map.clear();
        Ok(())
    }

    /// Rewrite from in-memory state if `cache.jsonl` exceeds 50 MB.
    /// Foreign-namespace lines from the prior file are dropped — we only
    /// preserve the current namespace's entries (Risk R5 chronic-disease check).
    pub fn compact_if_oversized(&mut self) -> io::Result<()> {
        let size = match fs::metadata(&self.log_path) {
            Ok(m) => m.len(),
            Err(_) => return Ok(()),
        };
        if size <= COMPACT_THRESHOLD_BYTES {
            return Ok(());
        }
        let mut buf = String::new();
        for (hash, vec) in &self.map {
            let line = json!({
                "ns": self.namespace,
                "hash": encode_hex_32(hash),
                "vec": vec,
            })
            .to_string();
            buf.push_str(&line);
            buf.push('\n');
        }
        atomic_write(&self.log_path, buf.as_bytes())?;
        Ok(())
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn ns(s: &str) -> String {
        s.to_string()
    }

    #[test]
    fn open_missing_file_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let cache = EmbeddingCache::open(tmp.path(), ns("a:b:768")).unwrap();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn put_then_get_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let mut cache = EmbeddingCache::open(tmp.path(), ns("a:b:768")).unwrap();
        let h = [7u8; 32];
        cache.put(h, vec![1.0, 2.0, 3.0]).unwrap();
        let got = cache.get(&h).unwrap();
        assert_eq!(got, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn open_filters_by_namespace() {
        let tmp = TempDir::new().unwrap();
        let mut cache = EmbeddingCache::open(tmp.path(), ns("model-a:dig:768")).unwrap();
        cache.put([1u8; 32], vec![0.1]).unwrap();
        let mut cache2 = EmbeddingCache::open(tmp.path(), ns("model-b:dig:768")).unwrap();
        cache2.put([2u8; 32], vec![0.2]).unwrap();

        let reopened = EmbeddingCache::open(tmp.path(), ns("model-a:dig:768")).unwrap();
        assert_eq!(reopened.len(), 1);
        assert!(reopened.get(&[1u8; 32]).is_some());
        assert!(reopened.get(&[2u8; 32]).is_none());
    }

    #[test]
    fn malformed_line_dropped_and_load_continues() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cache.jsonl");
        // Mix valid + invalid lines.
        let valid = json!({
            "ns": "x:y:768",
            "hash": encode_hex_32(&[3u8; 32]),
            "vec": [0.5_f32]
        })
        .to_string();
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{{garbage").unwrap();
        writeln!(f, "{}", valid).unwrap();
        writeln!(f, "not even json").unwrap();
        drop(f);

        let cache = EmbeddingCache::open(tmp.path(), ns("x:y:768")).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&[3u8; 32]).unwrap(), &[0.5]);
    }

    #[test]
    fn invalidate_namespace_preserves_foreign_entries() {
        let tmp = TempDir::new().unwrap();
        // Seed with current + foreign entries.
        let mut current = EmbeddingCache::open(tmp.path(), ns("cur:d:768")).unwrap();
        current.put([1u8; 32], vec![0.1]).unwrap();
        let mut foreign = EmbeddingCache::open(tmp.path(), ns("for:d:768")).unwrap();
        foreign.put([2u8; 32], vec![0.2]).unwrap();

        // Now invalidate the current namespace.
        let mut current = EmbeddingCache::open(tmp.path(), ns("cur:d:768")).unwrap();
        current.invalidate_namespace().unwrap();

        // Reopen as foreign and confirm foreign entry is still there.
        let foreign = EmbeddingCache::open(tmp.path(), ns("for:d:768")).unwrap();
        assert_eq!(foreign.len(), 1);
        assert_eq!(foreign.get(&[2u8; 32]).unwrap(), &[0.2]);

        // Reopen as current and confirm it's gone.
        let current = EmbeddingCache::open(tmp.path(), ns("cur:d:768")).unwrap();
        assert_eq!(current.len(), 0);
    }

    #[test]
    fn compact_noops_under_threshold() {
        let tmp = TempDir::new().unwrap();
        let mut cache = EmbeddingCache::open(tmp.path(), ns("a:b:768")).unwrap();
        cache.put([1u8; 32], vec![0.1, 0.2, 0.3]).unwrap();
        let before = std::fs::metadata(&cache.log_path).unwrap().len();
        cache.compact_if_oversized().unwrap();
        let after = std::fs::metadata(&cache.log_path).unwrap().len();
        assert_eq!(before, after);
    }

    #[test]
    fn hex_roundtrip() {
        let h = [0xab; 32];
        let s = encode_hex_32(&h);
        assert_eq!(s.len(), 64);
        assert_eq!(decode_hex_32(&s).unwrap(), h);
    }
}
