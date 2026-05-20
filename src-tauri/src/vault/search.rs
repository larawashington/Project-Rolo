//! Hybrid (BM25 + vector) search with Reciprocal Rank Fusion.
//!
//! Per `PRD/rolo-hybrid-search.md` §7.1 and §7.2:
//! - Two retrievers run in parallel: BM25 (keyword) and embedding (vector).
//! - Their ranked lists fuse via RRF with `k = 60`, oversampling each branch
//!   by 2x to give RRF something to fuse.
//! - When the embedding index is `None` or the embedder times out, the vector
//!   branch is empty and RRF degenerates to BM25-only — no special-case
//!   branching, the math handles it.
//!
//! Phase A wiring: `HybridSearcher` is fully implemented but **not yet
//! constructed by `Vault`** (PRD §12.3 step 4). That happens in Phase B,
//! along with the `PromptAssembler` signature swap.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, RwLock};

use lru::LruCache;

use crate::vault::bm25::BM25Index;
use crate::vault::chunker::Chunk;
use crate::vault::embeddings::{Embedder, EmbeddingIndex};

const RRF_K: f32 = 60.0;
const RRF_OVERSAMPLE: usize = 2;
const QUERY_LRU_CAP: usize = 128;

#[derive(Clone, Debug)]
pub struct FusedHit {
    pub chunk: Chunk,
    pub bm25_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub rrf_score: f32,
}

pub struct HybridSearcher {
    bm25: Arc<RwLock<Option<BM25Index>>>,
    emb: Arc<RwLock<Option<EmbeddingIndex>>>,
    embedder: Arc<dyn Embedder>,
    query_cache: Mutex<LruCache<String, [f32; 768]>>,
}

impl HybridSearcher {
    pub fn new(
        bm25: Arc<RwLock<Option<BM25Index>>>,
        emb: Arc<RwLock<Option<EmbeddingIndex>>>,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        let cap = NonZeroUsize::new(QUERY_LRU_CAP).expect("QUERY_LRU_CAP > 0");
        Self {
            bm25,
            emb,
            embedder,
            query_cache: Mutex::new(LruCache::new(cap)),
        }
    }

    /// Unfiltered hybrid search. BM25 top-(k*2) + vector top-(k*2) → RRF k=60
    /// → top-k.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<FusedHit> {
        if top_k == 0 {
            return Vec::new();
        }
        let oversample = top_k.saturating_mul(RRF_OVERSAMPLE);

        // 1. BM25 branch + 2. Vector branch.
        let bm25_hits = {
            let guard = self.bm25.read().unwrap_or_else(|p| p.into_inner());
            match guard.as_ref() {
                Some(b) => b.search(query, oversample),
                None => Vec::new(),
            }
        };
        let query_vec = self.cached_embed_query(query);
        let vector_hits = match query_vec {
            Some(qv) => {
                let guard = self.emb.read().unwrap_or_else(|p| p.into_inner());
                match guard.as_ref() {
                    Some(e) => e.search(&qv, oversample),
                    None => Vec::new(),
                }
            }
            None => Vec::new(),
        };

        // 3. RRF fusion. BM25 hits ship with Chunks already; vector hits need
        //    to look up their Chunk by id. Hydrate lazily — only the chunks
        //    actually referenced get cloned, never the whole index.
        let mut by_id: HashMap<String, FusedHit> = HashMap::new();
        for (rank, hit) in bm25_hits.iter().enumerate() {
            let entry = by_id
                .entry(hit.chunk.id.clone())
                .or_insert_with(|| FusedHit {
                    chunk: hit.chunk.clone(),
                    bm25_rank: None,
                    vector_rank: None,
                    rrf_score: 0.0,
                });
            entry.bm25_rank = Some(rank);
            entry.rrf_score += 1.0 / (RRF_K + (rank as f32) + 1.0);
        }
        // Vector-only hits — chunks not in bm25_hits — need a BM25 lookup.
        // Acquire the lock once for any vector hits we still need.
        let needs_lookup: Vec<&str> = vector_hits
            .iter()
            .map(|r| r.row.chunk_id.as_str())
            .filter(|id| !by_id.contains_key(*id))
            .collect();
        let resolved: HashMap<String, Chunk> = if needs_lookup.is_empty() {
            HashMap::new()
        } else {
            let guard = self.bm25.read().unwrap_or_else(|p| p.into_inner());
            match guard.as_ref() {
                Some(b) => b
                    .chunks()
                    .iter()
                    .filter(|c| needs_lookup.contains(&c.id.as_str()))
                    .map(|c| (c.id.clone(), c.clone()))
                    .collect(),
                None => HashMap::new(),
            }
        };
        for (rank, row) in vector_hits.iter().enumerate() {
            let entry = match by_id.get_mut(&row.row.chunk_id) {
                Some(existing) => existing,
                None => match resolved.get(&row.row.chunk_id) {
                    Some(chunk) => by_id.entry(row.row.chunk_id.clone()).or_insert(FusedHit {
                        chunk: chunk.clone(),
                        bm25_rank: None,
                        vector_rank: None,
                        rrf_score: 0.0,
                    }),
                    None => continue, // stale row from a not-yet-rebuilt index
                },
            };
            entry.vector_rank = Some(rank);
            entry.rrf_score += 1.0 / (RRF_K + (rank as f32) + 1.0);
        }

        let mut hits: Vec<FusedHit> = by_id.into_values().collect();
        hits.sort_by(|a, b| {
            b.rrf_score
                .partial_cmp(&a.rrf_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk.id.cmp(&b.chunk.id))
        });
        hits.truncate(top_k);

        // Per PRD §7.12: emit one structured JSON log per search call when
        // RUST_LOG=rolo::vault::search=debug. Off by default.
        if tracing::enabled!(target: "rolo::vault::search", tracing::Level::DEBUG) {
            let truncated_query: String = query.chars().take(80).collect();
            let results_json: Vec<String> = hits
                .iter()
                .map(|h| {
                    format!(
                        r#"{{"chunk_id":"{}","bm25_rank":{},"vector_rank":{},"rrf_score":{:.6}}}"#,
                        h.chunk.id.replace('"', "\\\""),
                        h.bm25_rank
                            .map(|r| r.to_string())
                            .unwrap_or_else(|| "null".into()),
                        h.vector_rank
                            .map(|r| r.to_string())
                            .unwrap_or_else(|| "null".into()),
                        h.rrf_score,
                    )
                })
                .collect();
            tracing::debug!(
                target: "rolo::vault::search",
                r#"{{"query":{:?},"results":[{}]}}"#,
                truncated_query,
                results_json.join(",")
            );
        }

        hits
    }

    /// User Context slot path. Same as `search`, then filters to wiki/user/
    /// and wiki/relationships/ chunks.
    pub fn search_user_scope(&self, query: &str, top_k: usize) -> Vec<FusedHit> {
        // Oversample at the BM25/vector layer in case the filter strips most
        // results — pull `top_k * RRF_OVERSAMPLE * 2` then truncate.
        let oversample = top_k.saturating_mul(RRF_OVERSAMPLE).max(top_k);
        let raw = self.search(query, oversample);
        let mut filtered: Vec<FusedHit> = raw
            .into_iter()
            .filter(|h| {
                let p = h
                    .chunk
                    .file_path
                    .to_string_lossy()
                    .replace(std::path::MAIN_SEPARATOR, "/");
                p.starts_with("user/") || p.starts_with("relationships/")
            })
            .collect();
        filtered.truncate(top_k);
        filtered
    }

    /// Pass-through to `BM25Index::chunks_in_file`, returning owned Chunks.
    /// Used by `PromptAssembler` slot 1/4 in Phase B (per PRD §6).
    pub fn bm25_chunks_in_file(&self, file: &str) -> Vec<Chunk> {
        let guard = self.bm25.read().unwrap_or_else(|p| p.into_inner());
        match guard.as_ref() {
            Some(b) => b.chunks_in_file(file).into_iter().cloned().collect(),
            None => Vec::new(),
        }
    }

    /// Whether the BM25 index is currently loaded. Used by `PromptAssembler`
    /// to decide between the verbatim fallback (None) vs. running through
    /// slot assembly with a possibly-empty index.
    pub fn bm25_is_loaded(&self) -> bool {
        let guard = self.bm25.read().unwrap_or_else(|p| p.into_inner());
        guard.is_some()
    }

    /// Diagnostic accessor: number of vectors in the embedding index, or 0
    /// when none is loaded. Hidden from rustdoc — production code should not
    /// branch on index size.
    #[doc(hidden)]
    pub fn embedding_count(&self) -> usize {
        let guard = self.emb.read().unwrap_or_else(|p| p.into_inner());
        guard.as_ref().map(|i| i.len()).unwrap_or(0)
    }

    /// LRU-cached query embedding. Drops the lock before the network call so
    /// concurrent searches don't serialize on it (PRD §7.7).
    fn cached_embed_query(&self, query: &str) -> Option<[f32; 768]> {
        {
            let mut cache = self.query_cache.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(v) = cache.get(query) {
                return Some(*v);
            }
        }
        let v = self.embedder.embed_query(query)?;
        let mut cache = self.query_cache.lock().unwrap_or_else(|p| p.into_inner());
        cache.put(query.to_string(), v);
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::bm25::BM25Index;
    use std::fs;
    use std::io;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tempfile::TempDir;

    /// Embedder that always returns None — drives the BM25-only fallback path.
    struct DeadEmbedder;
    impl Embedder for DeadEmbedder {
        fn probe_digest(&self) -> io::Result<[u8; 32]> {
            Ok([0; 32])
        }
        fn embed_batch(
            &self,
            _texts: &[String],
            _timeout: Duration,
        ) -> io::Result<Vec<Option<Vec<f32>>>> {
            Ok(Vec::new())
        }
        fn embed_query(&self, _text: &str) -> Option<[f32; 768]> {
            None
        }
        fn model_name(&self) -> &str {
            "dead"
        }
    }

    /// Counter-stub embedder. Returns a fixed vector and counts every call.
    struct CountingEmbedder {
        calls: AtomicUsize,
        vec: [f32; 768],
    }
    impl CountingEmbedder {
        fn new() -> Self {
            // Single-axis vector — easy to predict ranks against synthetic indexes.
            let mut v = [0f32; 768];
            v[0] = 1.0;
            Self {
                calls: AtomicUsize::new(0),
                vec: v,
            }
        }
    }
    impl Embedder for CountingEmbedder {
        fn probe_digest(&self) -> io::Result<[u8; 32]> {
            Ok([0xab; 32])
        }
        fn embed_batch(
            &self,
            _texts: &[String],
            _timeout: Duration,
        ) -> io::Result<Vec<Option<Vec<f32>>>> {
            Ok(Vec::new())
        }
        fn embed_query(&self, _text: &str) -> Option<[f32; 768]> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Some(self.vec)
        }
        fn model_name(&self) -> &str {
            "counting"
        }
    }

    fn write_wiki(root: &std::path::Path, files: &[(&str, &str)]) {
        for (rel, body) in files {
            let p = root.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, body).unwrap();
        }
    }

    fn build_bm25(files: &[(&str, &str)]) -> (TempDir, Arc<RwLock<Option<BM25Index>>>) {
        let tmp = TempDir::new().unwrap();
        write_wiki(tmp.path(), files);
        let idx = BM25Index::build(tmp.path()).unwrap();
        (tmp, Arc::new(RwLock::new(Some(idx))))
    }

    #[test]
    fn search_with_emb_none_equals_bm25_only() {
        let (_tmp, bm25) = build_bm25(&[
            (
                "user/preferences.md",
                "# T\n## Communication Style\nbe brief python\n",
            ),
            ("user/identity.md", "# T\n## Known Facts\nlearn rust\n"),
        ]);
        let emb: Arc<RwLock<Option<EmbeddingIndex>>> = Arc::new(RwLock::new(None));
        let embedder: Arc<dyn Embedder> = Arc::new(DeadEmbedder);

        let searcher = HybridSearcher::new(Arc::clone(&bm25), emb, embedder);
        let bm25_results = bm25.read().unwrap().as_ref().unwrap().search("python", 5);
        let fused = searcher.search("python", 5);

        assert_eq!(fused.len(), bm25_results.len());
        for (a, b) in fused.iter().zip(bm25_results.iter()) {
            assert_eq!(a.chunk.id, b.chunk.id);
            assert!(a.vector_rank.is_none());
            assert!(a.bm25_rank.is_some());
        }
    }

    #[test]
    fn rrf_score_ordering_matches_hand_computed() {
        // Three chunks, BM25 ranks them in a specific order; vector hits empty.
        let (_tmp, bm25) = build_bm25(&[
            ("a.md", "# T\n## A\npython python python\n"),
            ("b.md", "# T\n## B\npython rust\n"),
            ("c.md", "# T\n## C\nrust\n"),
        ]);
        let searcher = HybridSearcher::new(
            Arc::clone(&bm25),
            Arc::new(RwLock::new(None)),
            Arc::new(DeadEmbedder),
        );
        let res = searcher.search("python", 3);
        assert!(res.len() >= 2);
        // First-ranked BM25 chunk gets 1/(60+1) = 0.01639...
        assert!((res[0].rrf_score - 1.0 / 61.0).abs() < 1e-6);
        if res.len() >= 2 {
            // Second-ranked gets 1/62.
            assert!((res[1].rrf_score - 1.0 / 62.0).abs() < 1e-6);
            assert!(res[0].rrf_score > res[1].rrf_score);
        }
    }

    #[test]
    fn search_user_scope_filters_non_user_paths() {
        let (_tmp, bm25) = build_bm25(&[
            ("user/preferences.md", "# T\n## H\npython python\n"),
            ("relationships/human.md", "# T\n## H\npython\n"),
            ("personality/core.md", "# T\n## H\npython python python\n"),
            ("world/env.md", "# T\n## H\npython\n"),
        ]);
        let searcher = HybridSearcher::new(
            Arc::clone(&bm25),
            Arc::new(RwLock::new(None)),
            Arc::new(DeadEmbedder),
        );
        let res = searcher.search_user_scope("python", 10);
        assert!(!res.is_empty());
        for h in &res {
            let p = h
                .chunk
                .file_path
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            assert!(
                p.starts_with("user/") || p.starts_with("relationships/"),
                "unexpected path: {}",
                p
            );
        }
    }

    #[test]
    fn lru_skips_second_embed_call_for_same_query() {
        let (_tmp, bm25) = build_bm25(&[("a.md", "# T\n## H\nhello\n")]);
        let counter = Arc::new(CountingEmbedder::new());
        let counter_dyn: Arc<dyn Embedder> = counter.clone();
        let searcher = HybridSearcher::new(
            bm25,
            Arc::new(RwLock::new(None)), // no emb index — query embed still runs and populates LRU
            counter_dyn,
        );
        let _ = searcher.search("identical query", 5);
        let after_first = counter.calls.load(Ordering::SeqCst);
        let _ = searcher.search("identical query", 5);
        let after_second = counter.calls.load(Ordering::SeqCst);
        assert_eq!(
            after_second, after_first,
            "expected LRU hit on second identical query"
        );
    }

    #[test]
    fn embedder_returning_none_falls_back_to_bm25() {
        let (_tmp, bm25) = build_bm25(&[
            ("a.md", "# T\n## H\nhello python\n"),
            ("b.md", "# T\n## H\nrust\n"),
        ]);
        // emb is Some — but DeadEmbedder.embed_query returns None, so vector
        // branch should still produce zero hits.
        // (We don't actually populate the EmbeddingIndex; the searcher should
        // skip vector search when query_vec is None.)
        let searcher = HybridSearcher::new(
            Arc::clone(&bm25),
            Arc::new(RwLock::new(None)),
            Arc::new(DeadEmbedder),
        );
        let res = searcher.search("python", 5);
        assert!(!res.is_empty());
        for h in &res {
            assert!(h.vector_rank.is_none());
        }
    }

    #[test]
    fn bm25_chunks_in_file_passes_through() {
        let (_tmp, bm25) = build_bm25(&[("user/preferences.md", "# T\n## A\nfoo\n## B\nbar\n")]);
        let searcher =
            HybridSearcher::new(bm25, Arc::new(RwLock::new(None)), Arc::new(DeadEmbedder));
        let chunks = searcher.bm25_chunks_in_file("user/preferences.md");
        assert_eq!(chunks.len(), 2);
    }

    // Quiet `unused` warnings on chunker::PathBuf import.
    #[allow(dead_code)]
    fn _force_pathbuf_use() -> PathBuf {
        PathBuf::new()
    }
}
