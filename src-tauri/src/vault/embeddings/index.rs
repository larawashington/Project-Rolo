//! Flat-vector embedding index. PRD §7.1, §7.3, §7.4.
//!
//! On disk: one binary `index.bin` (header + model name + raw f32 vectors)
//! plus a `chunks.jsonl` sidecar with row metadata. Both written via
//! `vault::atomic::atomic_write` in the order **chunks.jsonl first, then
//! index.bin** so a crash mid-persist is detectable on next boot
//! (`header.count != chunks.jsonl line count` → wipe both, set dirty).
//!
//! Search is brute-force cosine. Vectors are L2-normalized at insert so the
//! similarity reduces to a dot product (PRD §7.3, Decision H3).

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

use crate::vault::atomic::atomic_write;
use crate::vault::chunker::Chunk;
use crate::vault::embeddings::cache::EmbeddingCache;
use crate::vault::embeddings::embedder::Embedder;
use crate::vault::embeddings::hex::{decode_hex_32, encode_hex_32 as encode_hex, l2_normalize};

/// Fixed 64-byte header. `repr(C, packed)` so on-disk layout is byte-exact and
/// portable. Validated at compile time with the const-assert below.
#[repr(C, packed)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct IndexHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub dim: u32,
    pub count: u32,
    pub model_name_len: u16,
    pub model_digest: [u8; 32],
    pub _reserved: [u8; 14],
}

const _: () = assert!(std::mem::size_of::<IndexHeader>() == 64);

pub const INDEX_MAGIC: [u8; 4] = *b"ROLO";
pub const INDEX_VERSION: u32 = 1;
pub const VECTOR_DIM: usize = 768;

/// Per-row metadata kept in `chunks.jsonl`. Held in RAM at `vectors[i*768..]`.
#[derive(Clone, Debug)]
pub struct RowMeta {
    pub chunk_id: String,
    pub file_path: String,
    pub heading: String,
    pub content_hash: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct ScoredRow {
    pub row: RowMeta,
    pub score: f32,
}

pub struct EmbeddingIndex {
    /// Length = N * 768, row-major. L2-normalized at insert.
    vectors: Vec<f32>,
    rows: Vec<RowMeta>,
    header: IndexHeader,
    /// Model name bytes — written into the on-disk payload between header
    /// and vectors (PRD §7.4). Length matches `header.model_name_len`.
    model_name: String,
}

impl EmbeddingIndex {
    /// Build from a chunk slice, consulting the cache before issuing any
    /// network calls. Skips chunks whose token estimate exceeds 8192
    /// (Decision H13) and per-item batch failures (PRD §7.6) with warnings.
    pub fn build(
        chunks: &[Chunk],
        embedder: &dyn Embedder,
        cache: &mut EmbeddingCache,
        progress: impl Fn(usize, usize),
    ) -> io::Result<Self> {
        let model_digest = embedder.probe_digest()?;
        let model_name = embedder.model_name().to_string();

        const BATCH_SIZE: usize = 32;
        let total = chunks.len();
        let mut vectors: Vec<f32> = Vec::with_capacity(total * VECTOR_DIM);
        let mut rows: Vec<RowMeta> = Vec::with_capacity(total);
        let mut done: usize = 0;

        // Pre-pass: normalize bodies and compute hashes once. Filter out the
        // long-chunk skips here so batches are full of work.
        struct Prepared<'a> {
            chunk: &'a Chunk,
            normalized: String,
            hash: [u8; 32],
        }

        let mut prepared: Vec<Prepared> = Vec::with_capacity(total);
        for c in chunks {
            // chars/4 token estimator (master plan §3.4)
            let approx_tokens = c.body.chars().count() / 4;
            if approx_tokens > 8192 {
                tracing::warn!(
                    target: "rolo::vault::embeddings",
                    "skipping oversized chunk {} ({}#{}): est_tokens={}",
                    c.id,
                    c.file_path.display(),
                    c.heading,
                    approx_tokens
                );
                continue;
            }
            let normalized = normalize_body(&c.body);
            let mut hasher = Sha256::new();
            hasher.update(normalized.as_bytes());
            let hash: [u8; 32] = hasher.finalize().into();
            prepared.push(Prepared {
                chunk: c,
                normalized,
                hash,
            });
        }

        for batch in prepared.chunks(BATCH_SIZE) {
            // Split cache hits from misses.
            let mut miss_indices: Vec<usize> = Vec::new();
            let mut miss_texts: Vec<String> = Vec::new();
            let mut hit_vecs: Vec<Option<Vec<f32>>> = Vec::with_capacity(batch.len());
            for (i, item) in batch.iter().enumerate() {
                if let Some(v) = cache.get(&item.hash) {
                    hit_vecs.push(Some(v.to_vec()));
                } else {
                    hit_vecs.push(None);
                    miss_indices.push(i);
                    miss_texts.push(item.normalized.clone());
                }
            }

            let new_vectors: Vec<Option<Vec<f32>>> = if miss_texts.is_empty() {
                Vec::new()
            } else {
                match embedder.embed_batch(&miss_texts, Duration::from_secs(30)) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(
                            target: "rolo::vault::embeddings",
                            "embed_batch failed: {} — aborting build",
                            e
                        );
                        return Err(e);
                    }
                }
            };

            // Splice misses back into the slot ordering.
            for (slot, miss_pos) in miss_indices.iter().enumerate() {
                let v_opt = new_vectors.get(slot).cloned().flatten();
                if let Some(mut v) = v_opt {
                    if v.len() != VECTOR_DIM {
                        tracing::warn!(
                            target: "rolo::vault::embeddings",
                            "embed returned dim={} for chunk {}; skipping",
                            v.len(),
                            batch[*miss_pos].chunk.id
                        );
                        hit_vecs[*miss_pos] = None;
                        continue;
                    }
                    l2_normalize(&mut v);
                    if let Err(e) = cache.put(batch[*miss_pos].hash, v.clone()) {
                        tracing::warn!(
                            target: "rolo::vault::embeddings",
                            "cache put failed for {}: {}",
                            batch[*miss_pos].chunk.id,
                            e
                        );
                    }
                    hit_vecs[*miss_pos] = Some(v);
                } else {
                    tracing::warn!(
                        target: "rolo::vault::embeddings",
                        "per-item embed failure for chunk {}",
                        batch[*miss_pos].chunk.id
                    );
                }
            }

            for (i, vec_opt) in hit_vecs.into_iter().enumerate() {
                let item = &batch[i];
                let Some(v) = vec_opt else { continue };
                vectors.extend_from_slice(&v);
                rows.push(RowMeta {
                    chunk_id: item.chunk.id.clone(),
                    file_path: item
                        .chunk
                        .file_path
                        .to_string_lossy()
                        .replace(std::path::MAIN_SEPARATOR, "/"),
                    heading: item.chunk.heading.clone(),
                    content_hash: item.hash,
                });
            }

            done += batch.len();
            progress(done.min(total), total);
        }

        let count: u32 = rows.len() as u32;
        let model_name_bytes = model_name.as_bytes();
        let model_name_len: u16 = model_name_bytes.len() as u16;

        let header = IndexHeader {
            magic: INDEX_MAGIC,
            version: INDEX_VERSION,
            dim: VECTOR_DIM as u32,
            count,
            model_name_len,
            model_digest,
            _reserved: [0u8; 14],
        };

        Ok(Self {
            vectors,
            rows,
            header,
            model_name,
        })
    }

    /// Persist to `dir/chunks.jsonl` then `dir/index.bin` via atomic rename.
    /// Order is correctness-critical: a crash between the two leaves a
    /// chunks.jsonl whose line count won't match a stale index.bin → the
    /// next `load` returns `Ok(None)` and dreaming rebuilds (PRD §7.4).
    pub fn persist(&self, dir: &Path) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        let chunks_path = dir.join("chunks.jsonl");
        let index_path = dir.join("index.bin");

        // 1. chunks.jsonl
        let mut buf = String::new();
        for (i, row) in self.rows.iter().enumerate() {
            // tokens is best-effort metadata only; we don't have the original
            // chunk here, so emit 0 — see PRD §7.4 ("tokens field is best-effort").
            let line = json!({
                "row": i,
                "chunk_id": row.chunk_id,
                "file_path": row.file_path,
                "heading": row.heading,
                "content_hash": encode_hex(&row.content_hash),
                "tokens": 0,
            })
            .to_string();
            buf.push_str(&line);
            buf.push('\n');
        }
        atomic_write(&chunks_path, buf.as_bytes())?;

        // 2. index.bin: header (64) + model_name (padded to 8B) + vectors.
        let header_bytes = bytemuck::bytes_of(&self.header).to_vec();
        let mut payload: Vec<u8> =
            Vec::with_capacity(64 + 64 + self.vectors.len() * std::mem::size_of::<f32>());
        payload.extend_from_slice(&header_bytes);
        // Model name UTF-8 bytes, zero-padded so vectors land on an 8-byte
        // boundary (PRD §7.4 H = 64 + ((model_name_len + 7) & !7)).
        let mn_bytes = self.model_name.as_bytes();
        let mn_len = mn_bytes.len().min(self.header.model_name_len as usize);
        let aligned = ((self.header.model_name_len as usize) + 7) & !7;
        payload.extend_from_slice(&mn_bytes[..mn_len]);
        payload.extend(std::iter::repeat_n(0u8, aligned - mn_len));
        payload.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&self.vectors));
        atomic_write(&index_path, &payload)?;
        Ok(())
    }

    /// Load + validate. Returns `Ok(None)` on any header mismatch, magic fail,
    /// digest fail, or count/sidecar drift — the caller wipes both files and
    /// queues a rebuild (PRD §8 "Index header magic invalid").
    ///
    /// v1 simplicity: vectors are mmap-read but copied into `Vec<f32>` for
    /// search. Copy of 1000×768×4 = 3 MB is sub-ms; mmap is just for fast read.
    pub fn load(dir: &Path, expected_digest: &[u8; 32]) -> io::Result<Option<Self>> {
        let index_path = dir.join("index.bin");
        let chunks_path = dir.join("chunks.jsonl");
        if !index_path.exists() || !chunks_path.exists() {
            return Ok(None);
        }

        let f = File::open(&index_path)?;
        let metadata = f.metadata()?;
        if metadata.len() < 64 {
            return Ok(None);
        }
        // Safety: file is read-only and we drop the mmap before returning.
        let mmap = unsafe { Mmap::map(&f)? };
        let header_bytes = &mmap[..64];
        let header: IndexHeader =
            match bytemuck::try_pod_read_unaligned::<IndexHeader>(header_bytes) {
                Ok(h) => h,
                Err(_) => return Ok(None),
            };

        if header.magic != INDEX_MAGIC {
            return Ok(None);
        }
        if header.version != INDEX_VERSION {
            return Ok(None);
        }
        if header.dim as usize != VECTOR_DIM {
            return Ok(None);
        }
        if &header.model_digest != expected_digest {
            return Ok(None);
        }

        let count = header.count as usize;
        let mn_len = header.model_name_len as usize;
        let payload_offset = 64 + ((mn_len + 7) & !7);
        let vectors_bytes = count * VECTOR_DIM * std::mem::size_of::<f32>();
        if mmap.len() < payload_offset + vectors_bytes {
            return Ok(None);
        }
        let vec_slice = &mmap[payload_offset..payload_offset + vectors_bytes];
        let vectors: Vec<f32> = bytemuck::cast_slice::<u8, f32>(vec_slice).to_vec();

        // Sidecar count check.
        let cf = File::open(&chunks_path)?;
        let reader = BufReader::new(cf);
        let mut rows: Vec<RowMeta> = Vec::with_capacity(count);
        for line in reader.lines().map_while(Result::ok) {
            if line.trim().is_empty() {
                continue;
            }
            let v: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => return Ok(None),
            };
            let chunk_id = v
                .get("chunk_id")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let file_path = v
                .get("file_path")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let heading = v
                .get("heading")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let content_hash = match v.get("content_hash").and_then(|x| x.as_str()) {
                Some(s) => decode_hex_32(s).unwrap_or([0u8; 32]),
                None => [0u8; 32],
            };
            rows.push(RowMeta {
                chunk_id,
                file_path,
                heading,
                content_hash,
            });
        }

        if rows.len() != count {
            return Ok(None);
        }

        // Read the model name string from the same mmap before dropping it.
        let mn_end = 64 + mn_len;
        let model_name = if mn_end <= mmap.len() {
            String::from_utf8_lossy(&mmap[64..mn_end]).into_owned()
        } else {
            String::new()
        };

        // Drop mmap explicitly via end-of-scope; vectors already copied.
        let _ = mmap;
        let _ = f;
        let _ = vectors_bytes;

        Ok(Some(Self {
            vectors,
            rows,
            header,
            model_name,
        }))
    }

    /// Brute-force cosine top-k. Vectors are pre-normalized so this is a dot
    /// product (PRD §7.3).
    pub fn search(&self, query_vec: &[f32; VECTOR_DIM], top_k: usize) -> Vec<ScoredRow> {
        if self.rows.is_empty() || top_k == 0 {
            return Vec::new();
        }
        let n = self.rows.len();
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);
        for i in 0..n {
            let row = &self.vectors[i * VECTOR_DIM..(i + 1) * VECTOR_DIM];
            let mut dot: f32 = 0.0;
            for j in 0..VECTOR_DIM {
                dot += row[j] * query_vec[j];
            }
            scores.push((i, dot));
        }
        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scores.truncate(top_k);
        scores
            .into_iter()
            .map(|(i, s)| ScoredRow {
                row: self.rows[i].clone(),
                score: s,
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn header(&self) -> &IndexHeader {
        &self.header
    }

    pub fn rows(&self) -> &[RowMeta] {
        &self.rows
    }

    /// Read-only view of the underlying f32 vector buffer. Length = N * 768.
    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }
}

/// NFC + trim + collapse internal whitespace runs to a single space.
fn normalize_body(body: &str) -> String {
    let nfc: String = body.nfc().collect();
    let mut out = String::with_capacity(nfc.len());
    let mut prev_ws = false;
    for ch in nfc.trim().chars() {
        if ch.is_whitespace() {
            if !prev_ws {
                out.push(' ');
                prev_ws = true;
            }
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    out
}

// Silence unused-import warning for io::Read on platforms where mmap is direct.
#[allow(dead_code)]
fn _suppress_read_warning(mut r: impl Read) -> io::Result<usize> {
    let mut buf = [0u8; 0];
    r.read(&mut buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::chunker::Chunk;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;
    use tempfile::TempDir;

    /// Deterministic stub embedder. Vector = repeating SHA-256(text) bytes
    /// scaled into f32, then L2-normalized by the embedder contract.
    struct StubEmbedder {
        digest: [u8; 32],
        calls: AtomicUsize,
    }
    impl StubEmbedder {
        fn new() -> Self {
            Self {
                digest: [0xab; 32],
                calls: AtomicUsize::new(0),
            }
        }
        fn vec_for(text: &str) -> Vec<f32> {
            let mut hasher = Sha256::new();
            hasher.update(text.as_bytes());
            let h: [u8; 32] = hasher.finalize().into();
            let mut v: Vec<f32> = Vec::with_capacity(VECTOR_DIM);
            for i in 0..VECTOR_DIM {
                let b = h[i % 32] as f32;
                // Mix in i so vectors aren't constant-repeating (would yield
                // identical normalized outputs).
                v.push(b + (i as f32) * 0.001);
            }
            l2_normalize(&mut v);
            v
        }
    }
    impl Embedder for StubEmbedder {
        fn probe_digest(&self) -> io::Result<[u8; 32]> {
            Ok(self.digest)
        }
        fn embed_batch(
            &self,
            texts: &[String],
            _timeout: Duration,
        ) -> io::Result<Vec<Option<Vec<f32>>>> {
            self.calls.fetch_add(texts.len(), Ordering::SeqCst);
            Ok(texts.iter().map(|t| Some(Self::vec_for(t))).collect())
        }
        fn embed_query(&self, text: &str) -> Option<[f32; 768]> {
            let v = Self::vec_for(text);
            let mut out = [0f32; 768];
            out.copy_from_slice(&v);
            Some(out)
        }
        fn model_name(&self) -> &str {
            "nomic-embed-text"
        }
    }

    fn mk_chunk(i: usize, body: &str) -> Chunk {
        Chunk {
            id: format!("file{}.md#{}", i, i),
            file_path: PathBuf::from(format!("user/file{}.md", i)),
            heading: format!("h{}", i),
            body: body.to_string(),
            tokens: vec![],
        }
    }

    #[test]
    fn header_size_is_64() {
        assert_eq!(std::mem::size_of::<IndexHeader>(), 64);
    }

    #[test]
    fn build_persist_load_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let chunks: Vec<Chunk> = (0..10)
            .map(|i| mk_chunk(i, &format!("body content number {}", i)))
            .collect();
        let embedder = StubEmbedder::new();
        let mut cache = EmbeddingCache::open(tmp.path(), "nomic:abab:768".to_string()).unwrap();
        let idx = EmbeddingIndex::build(&chunks, &embedder, &mut cache, |_, _| {}).unwrap();
        assert_eq!(idx.len(), 10);
        idx.persist(tmp.path()).unwrap();

        let loaded = EmbeddingIndex::load(tmp.path(), &embedder.digest)
            .unwrap()
            .expect("expected Some");
        assert_eq!(loaded.len(), 10);
        for (a, b) in idx.rows.iter().zip(loaded.rows.iter()) {
            assert_eq!(a.chunk_id, b.chunk_id);
            assert_eq!(a.file_path, b.file_path);
            assert_eq!(a.heading, b.heading);
            assert_eq!(a.content_hash, b.content_hash);
        }
        assert_eq!(idx.vectors, loaded.vectors);
    }

    #[test]
    fn load_magic_mismatch_returns_none() {
        let tmp = TempDir::new().unwrap();
        let chunks: Vec<Chunk> = (0..3).map(|i| mk_chunk(i, "body")).collect();
        let embedder = StubEmbedder::new();
        let mut cache = EmbeddingCache::open(tmp.path(), "nomic:a:768".to_string()).unwrap();
        let idx = EmbeddingIndex::build(&chunks, &embedder, &mut cache, |_, _| {}).unwrap();
        idx.persist(tmp.path()).unwrap();

        // Corrupt magic bytes.
        let p = tmp.path().join("index.bin");
        let mut data = std::fs::read(&p).unwrap();
        data[0] = b'X';
        std::fs::write(&p, &data).unwrap();

        let loaded = EmbeddingIndex::load(tmp.path(), &embedder.digest).unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn load_digest_mismatch_returns_none() {
        let tmp = TempDir::new().unwrap();
        let chunks: Vec<Chunk> = (0..3).map(|i| mk_chunk(i, "body")).collect();
        let embedder = StubEmbedder::new();
        let mut cache = EmbeddingCache::open(tmp.path(), "nomic:a:768".to_string()).unwrap();
        let idx = EmbeddingIndex::build(&chunks, &embedder, &mut cache, |_, _| {}).unwrap();
        idx.persist(tmp.path()).unwrap();

        let wrong = [0xff; 32];
        let loaded = EmbeddingIndex::load(tmp.path(), &wrong).unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn load_count_sidecar_mismatch_returns_none() {
        let tmp = TempDir::new().unwrap();
        let chunks: Vec<Chunk> = (0..3).map(|i| mk_chunk(i, "body")).collect();
        let embedder = StubEmbedder::new();
        let mut cache = EmbeddingCache::open(tmp.path(), "nomic:a:768".to_string()).unwrap();
        let idx = EmbeddingIndex::build(&chunks, &embedder, &mut cache, |_, _| {}).unwrap();
        idx.persist(tmp.path()).unwrap();

        // Truncate chunks.jsonl to one line — header.count says 3.
        let p = tmp.path().join("chunks.jsonl");
        let content = std::fs::read_to_string(&p).unwrap();
        let first = content.lines().next().unwrap();
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "{}", first).unwrap();
        drop(f);

        let loaded = EmbeddingIndex::load(tmp.path(), &embedder.digest).unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn search_returns_top_k_by_descending_score() {
        // Hand-build: 4 rows of normalized vectors aligned with one query.
        let mut vectors = Vec::new();
        // Row 0: aligned with [1,0,0,...] → score = 1.
        let mut r0 = vec![0f32; VECTOR_DIM];
        r0[0] = 1.0;
        vectors.extend_from_slice(&r0);
        // Row 1: 0.5 alignment.
        let mut r1 = vec![0f32; VECTOR_DIM];
        r1[0] = 1.0;
        r1[1] = 1.0;
        l2_normalize(&mut r1);
        vectors.extend_from_slice(&r1);
        // Row 2: orthogonal.
        let mut r2 = vec![0f32; VECTOR_DIM];
        r2[1] = 1.0;
        vectors.extend_from_slice(&r2);
        // Row 3: anti-aligned.
        let mut r3 = vec![0f32; VECTOR_DIM];
        r3[0] = -1.0;
        vectors.extend_from_slice(&r3);

        let rows: Vec<RowMeta> = (0..4)
            .map(|i| RowMeta {
                chunk_id: format!("c{}", i),
                file_path: format!("f{}.md", i),
                heading: format!("h{}", i),
                content_hash: [i as u8; 32],
            })
            .collect();
        let header = IndexHeader {
            magic: INDEX_MAGIC,
            version: INDEX_VERSION,
            dim: VECTOR_DIM as u32,
            count: 4,
            model_name_len: 0,
            model_digest: [0; 32],
            _reserved: [0; 14],
        };
        let idx = EmbeddingIndex {
            vectors,
            rows,
            header,
            model_name: String::new(),
        };
        let mut q = [0f32; VECTOR_DIM];
        q[0] = 1.0;
        let res = idx.search(&q, 3);
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].row.chunk_id, "c0");
        assert_eq!(res[1].row.chunk_id, "c1");
        assert_eq!(res[2].row.chunk_id, "c2");
        assert!(res[0].score > res[1].score);
    }

    #[test]
    fn perf_load_1000_rows_under_50ms() {
        // PRD §H7 budgets <10ms cold load in release. Debug builds run 2-4x slower —
        // we test 50ms here to stay non-flaky and still catch real regressions.
        let tmp = TempDir::new().unwrap();
        let chunks: Vec<Chunk> = (0..1000)
            .map(|i| mk_chunk(i, &format!("content body unique {}", i)))
            .collect();
        let embedder = StubEmbedder::new();
        let mut cache = EmbeddingCache::open(tmp.path(), "nomic:a:768".to_string()).unwrap();
        let idx = EmbeddingIndex::build(&chunks, &embedder, &mut cache, |_, _| {}).unwrap();
        idx.persist(tmp.path()).unwrap();

        let start = Instant::now();
        let loaded = EmbeddingIndex::load(tmp.path(), &embedder.digest)
            .unwrap()
            .expect("expected Some");
        let elapsed = start.elapsed();
        assert_eq!(loaded.len(), 1000);
        assert!(
            elapsed < Duration::from_millis(50),
            "load took {:?}, expected <50ms (debug); release budget is <10ms",
            elapsed
        );
    }

    #[test]
    fn build_skips_oversized_chunks() {
        let mut huge = "word ".repeat(40000); // ~200k chars → ~50k tokens
        huge.push('.');
        let chunks = vec![
            mk_chunk(0, "small"),
            mk_chunk(1, &huge),
            mk_chunk(2, "tiny"),
        ];
        let embedder = StubEmbedder::new();
        let tmp = TempDir::new().unwrap();
        let mut cache = EmbeddingCache::open(tmp.path(), "nomic:a:768".to_string()).unwrap();
        let idx = EmbeddingIndex::build(&chunks, &embedder, &mut cache, |_, _| {}).unwrap();
        assert_eq!(idx.len(), 2);
        let ids: Vec<_> = idx.rows.iter().map(|r| r.chunk_id.clone()).collect();
        assert!(ids.contains(&"file0.md#0".to_string()));
        assert!(ids.contains(&"file2.md#2".to_string()));
    }

    #[test]
    fn build_uses_cache_on_second_run() {
        let chunks: Vec<Chunk> = (0..5).map(|i| mk_chunk(i, "same body")).collect();
        let embedder = StubEmbedder::new();
        let tmp = TempDir::new().unwrap();
        let mut cache = EmbeddingCache::open(tmp.path(), "nomic:a:768".to_string()).unwrap();
        let _idx = EmbeddingIndex::build(&chunks, &embedder, &mut cache, |_, _| {}).unwrap();
        // All 5 chunks have identical body → identical hash → 1 unique embed call.
        let after_first = embedder.calls.load(Ordering::SeqCst);
        assert!(after_first >= 1);

        let _idx2 = EmbeddingIndex::build(&chunks, &embedder, &mut cache, |_, _| {}).unwrap();
        let after_second = embedder.calls.load(Ordering::SeqCst);
        // Second run should hit cache for everything; no new calls.
        assert_eq!(
            after_second, after_first,
            "second build should fully hit cache"
        );
    }
}
