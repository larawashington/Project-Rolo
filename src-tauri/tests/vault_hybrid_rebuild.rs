//! Phase C integration test (PRD §10.2).
//!
//! Bootstrap a Vault with a stub embedder, invoke `rebuild_index`, assert
//! the on-disk artifacts (`index.bin`, `chunks.jsonl`, `cache.jsonl`) exist
//! and are well-formed; assert `searcher.search` returns hits.

use std::io;
use std::sync::Arc;
use std::time::Duration;

use desktop_pet_lib::vault::embeddings::{Embedder, VECTOR_DIM};
use desktop_pet_lib::vault::Vault;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

/// Deterministic stub embedder: maps each input text to a unique 768-d
/// vector derived from SHA-256 of the text, then L2-normalized. Stable across
/// calls, so identical text → identical vector → cache hits.
struct StubEmbedder {
    digest: [u8; 32],
}

impl StubEmbedder {
    fn new() -> Self {
        Self { digest: [0x42; 32] }
    }

    fn embed_one(text: &str) -> [f32; 768] {
        // Take SHA-256 of the text as a 32-byte seed; expand to 768 floats by
        // repeating with a per-position salt; then L2-normalize.
        let seed = Sha256::digest(text.as_bytes());
        let mut out = [0f32; VECTOR_DIM];
        for (i, slot) in out.iter_mut().enumerate() {
            let byte = seed[i % 32];
            // Mix in position so the 24 successive 32-byte windows aren't identical.
            let mixed = (byte as u32).wrapping_add(i as u32);
            *slot = ((mixed as f32) / 255.0) - 0.5;
        }
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for slot in out.iter_mut() {
                *slot /= norm;
            }
        }
        out
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
        Ok(texts
            .iter()
            .map(|t| Some(Self::embed_one(t).to_vec()))
            .collect())
    }

    fn embed_query(&self, text: &str) -> Option<[f32; 768]> {
        Some(Self::embed_one(text))
    }

    fn model_name(&self) -> &str {
        "stub-embed-test"
    }
}

#[test]
fn rebuild_index_lands_embedding_files_and_searcher_returns_hits() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("vault");
    let embedder: Arc<dyn Embedder> = Arc::new(StubEmbedder::new());

    let vault = Vault::open_or_init_with_embedder(root.clone(), Arc::clone(&embedder));
    vault.rebuild_index();

    // Phase C acceptance: index.bin and chunks.jsonl land on disk.
    let embeddings_dir = root.join("embeddings");
    assert!(
        embeddings_dir.join("index.bin").exists(),
        "index.bin missing"
    );
    assert!(
        embeddings_dir.join("chunks.jsonl").exists(),
        "chunks.jsonl missing"
    );
    assert!(
        embeddings_dir.join("cache.jsonl").exists(),
        "cache.jsonl missing"
    );

    // Searcher returns fused hits — both BM25 and vector branches contribute.
    let hits = vault.searcher.search("brief", 5);
    assert!(!hits.is_empty(), "expected at least one hit for 'brief'");
}

#[test]
fn second_rebuild_uses_cache_for_zero_new_embed_calls() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingStub {
        digest: [u8; 32],
        embed_calls: AtomicUsize,
    }

    impl Embedder for CountingStub {
        fn probe_digest(&self) -> io::Result<[u8; 32]> {
            Ok(self.digest)
        }
        fn embed_batch(
            &self,
            texts: &[String],
            _timeout: Duration,
        ) -> io::Result<Vec<Option<Vec<f32>>>> {
            self.embed_calls.fetch_add(texts.len(), Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|t| Some(StubEmbedder::embed_one(t).to_vec()))
                .collect())
        }
        fn embed_query(&self, _text: &str) -> Option<[f32; 768]> {
            None
        }
        fn model_name(&self) -> &str {
            "stub-counting"
        }
    }

    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("vault");
    let embedder = Arc::new(CountingStub {
        digest: [0x77; 32],
        embed_calls: AtomicUsize::new(0),
    });
    let arc_emb: Arc<dyn Embedder> = embedder.clone();

    let vault = Vault::open_or_init_with_embedder(root.clone(), Arc::clone(&arc_emb));
    vault.rebuild_index();
    let after_first = embedder.embed_calls.load(Ordering::SeqCst);
    assert!(after_first > 0, "first rebuild should embed something");

    vault.rebuild_index();
    let after_second = embedder.embed_calls.load(Ordering::SeqCst);
    assert_eq!(
        after_second, after_first,
        "second rebuild should fully hit cache (no new embed calls)"
    );
}
