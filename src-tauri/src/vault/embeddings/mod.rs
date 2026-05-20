//! Embeddings subsystem for the Rolo vault.
//!
//! Phase A wiring — the modules compile and are unit-tested but are not yet
//! constructed by `Vault::open_or_init`. That happens in Phase B per
//! `PRD/rolo-hybrid-search.md` §12.3.

pub mod cache;
pub mod embedder;
pub mod hex;
pub mod index;

pub use cache::EmbeddingCache;
pub use embedder::{Embedder, OllamaEmbedder};
pub use index::{EmbeddingIndex, IndexHeader, RowMeta, ScoredRow, VECTOR_DIM};
