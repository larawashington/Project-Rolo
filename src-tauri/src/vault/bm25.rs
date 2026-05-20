use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use walkdir::WalkDir;

use crate::vault::chunker::{chunk_file, tokenize, Chunk};

pub struct BM25Index {
    chunks: Vec<Chunk>,
    /// term -> count of chunks containing the term
    doc_freq: HashMap<String, usize>,
    /// per-chunk: term -> count
    term_freq: Vec<HashMap<String, u32>>,
    /// per-chunk length in tokens
    doc_len: Vec<u32>,
    avg_doc_len: f64,
    n_docs: usize,
    k1: f64,
    b: f64,
}

#[derive(Debug, Clone)]
pub struct ScoredChunk {
    pub chunk: Chunk,
    pub score: f64,
}

impl BM25Index {
    /// Build an index from `wiki_dir/**/*.md`. Empty wiki -> valid empty index.
    /// Per-file UTF-8 / read errors are logged and the file is skipped (PRD §6.4 / §10).
    pub fn build(wiki_dir: &Path) -> std::io::Result<Self> {
        let mut paths: Vec<std::path::PathBuf> = Vec::new();
        if wiki_dir.exists() {
            for entry in WalkDir::new(wiki_dir).into_iter().filter_map(|e| e.ok()) {
                if !entry.file_type().is_file() {
                    continue;
                }
                let p = entry.path();
                if p.extension().and_then(|s| s.to_str()) == Some("md") {
                    paths.push(p.to_path_buf());
                }
            }
        }
        // Determinism: sort lexicographically before chunking.
        paths.sort();

        let mut chunks: Vec<Chunk> = Vec::new();
        for p in &paths {
            let rel = match p.strip_prefix(wiki_dir) {
                Ok(r) => r,
                Err(_) => p.as_path(),
            };
            // Use forward-slash relative path so chunk IDs match across platforms.
            let rel_str = rel
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");

            let content = match fs::read_to_string(p) {
                Ok(s) => s,
                Err(e) => {
                    log::warn!("vault: skipping {}: {}", p.display(), e);
                    continue;
                }
            };
            let mut file_chunks = chunk_file(&rel_str, &content);
            chunks.append(&mut file_chunks);
        }

        let n_docs = chunks.len();
        let mut term_freq: Vec<HashMap<String, u32>> = Vec::with_capacity(n_docs);
        let mut doc_len: Vec<u32> = Vec::with_capacity(n_docs);
        let mut doc_freq: HashMap<String, usize> = HashMap::new();

        for chunk in &chunks {
            let mut tf: HashMap<String, u32> = HashMap::new();
            for tok in &chunk.tokens {
                *tf.entry(tok.clone()).or_insert(0) += 1;
            }
            for term in tf.keys() {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
            doc_len.push(chunk.tokens.len() as u32);
            term_freq.push(tf);
        }

        let total_len: u64 = doc_len.iter().map(|&n| n as u64).sum();
        let avg_doc_len = if n_docs == 0 {
            0.0
        } else {
            total_len as f64 / n_docs as f64
        };

        Ok(Self {
            chunks,
            doc_freq,
            term_freq,
            doc_len,
            avg_doc_len,
            n_docs,
            k1: 1.2,
            b: 0.75,
        })
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<ScoredChunk> {
        if self.n_docs == 0 || top_k == 0 {
            return Vec::new();
        }
        let q_terms: Vec<String> = tokenize(query);
        if q_terms.is_empty() {
            return Vec::new();
        }

        let mut scores: Vec<(usize, f64)> = (0..self.n_docs)
            .map(|i| (i, self.score_doc(i, &q_terms)))
            .filter(|(_, s)| *s > 0.0)
            .collect();

        scores.sort_by(
            |a, b| match b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal) {
                Ordering::Equal => a.0.cmp(&b.0),
                ord => ord,
            },
        );
        scores.truncate(top_k);

        scores
            .into_iter()
            .map(|(i, s)| ScoredChunk {
                chunk: self.chunks[i].clone(),
                score: s,
            })
            .collect()
    }

    fn score_doc(&self, doc_i: usize, q_terms: &[String]) -> f64 {
        let mut score = 0.0;
        let dl = self.doc_len[doc_i] as f64;
        // Guard against divide-by-zero on a doc that tokenized to zero terms;
        // such chunks are usually filtered earlier but be defensive.
        let avgdl = if self.avg_doc_len == 0.0 {
            1.0
        } else {
            self.avg_doc_len
        };
        for term in q_terms {
            let n_term = *self.doc_freq.get(term).unwrap_or(&0);
            if n_term == 0 {
                continue;
            }
            let idf =
                ((self.n_docs as f64 - n_term as f64 + 0.5) / (n_term as f64 + 0.5) + 1.0).ln();
            let tf = *self.term_freq[doc_i].get(term).unwrap_or(&0) as f64;
            let numer = tf * (self.k1 + 1.0);
            let denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl);
            score += idf * numer / denom;
        }
        score
    }

    pub fn len(&self) -> usize {
        self.n_docs
    }

    pub fn is_empty(&self) -> bool {
        self.n_docs == 0
    }

    /// Sum of per-chunk lengths in tokens. Used for `meta.json::stats.total_wiki_tokens`.
    pub fn total_tokens(&self) -> u64 {
        self.doc_len.iter().map(|&n| n as u64).sum()
    }

    /// Used by HybridSearcher to map row IDs back to full chunks during RRF fusion.
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// Direct chunk lookup by file path string (e.g. "personality/core-identity.md").
    /// Used by PromptAssembler to pull specific slots without going through BM25 search.
    pub fn chunks_in_file(&self, file_path: &str) -> Vec<&Chunk> {
        self.chunks
            .iter()
            .filter(|c| {
                c.file_path
                    .to_string_lossy()
                    .replace(std::path::MAIN_SEPARATOR, "/")
                    == file_path
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_wiki(root: &Path, files: &[(&str, &str)]) {
        for (rel, body) in files {
            let p = root.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, body).unwrap();
        }
    }

    #[test]
    fn empty_dir_yields_empty_index() {
        let tmp = TempDir::new().unwrap();
        let idx = BM25Index::build(tmp.path()).unwrap();
        assert!(idx.is_empty());
        assert_eq!(idx.search("anything", 10).len(), 0);
    }

    #[test]
    fn nonexistent_dir_yields_empty_index() {
        let tmp = TempDir::new().unwrap();
        let idx = BM25Index::build(&tmp.path().join("nope")).unwrap();
        assert!(idx.is_empty());
    }

    #[test]
    fn single_doc_ranks_matching_term_first() {
        let tmp = TempDir::new().unwrap();
        write_wiki(
            tmp.path(),
            &[("a.md", "# T\n## H\nI love python programming.\n")],
        );
        let idx = BM25Index::build(tmp.path()).unwrap();
        let r = idx.search("python", 10);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].chunk.id, "a.md#H");
        assert!(r[0].score > 0.0);
    }

    #[test]
    fn idf_downweights_common_terms() {
        let tmp = TempDir::new().unwrap();
        write_wiki(
            tmp.path(),
            &[
                ("a.md", "# T\n## H\nthe quick brown fox\n"),
                ("b.md", "# T\n## H\nthe lazy dog\n"),
                ("c.md", "# T\n## H\nthe python the python\n"),
            ],
        );
        let idx = BM25Index::build(tmp.path()).unwrap();
        let r = idx.search("the python", 5);
        assert!(!r.is_empty());
        assert_eq!(r[0].chunk.id, "c.md#H");
    }

    #[test]
    fn determinism_two_builds_match_scores() {
        let tmp = TempDir::new().unwrap();
        write_wiki(
            tmp.path(),
            &[
                ("a.md", "# T\n## H\napple banana cherry\n"),
                ("b.md", "# T\n## H\napple apple\n"),
            ],
        );
        let i1 = BM25Index::build(tmp.path()).unwrap();
        let i2 = BM25Index::build(tmp.path()).unwrap();
        let r1 = i1.search("apple", 5);
        let r2 = i2.search("apple", 5);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.chunk.id, b.chunk.id);
            assert!((a.score - b.score).abs() < 1e-12);
        }
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let tmp = TempDir::new().unwrap();
        write_wiki(tmp.path(), &[("a.md", "# T\n## H\npython\n")]);
        let idx = BM25Index::build(tmp.path()).unwrap();
        assert!(idx.search("python", 0).is_empty());
    }

    #[test]
    fn unknown_query_terms_return_empty() {
        let tmp = TempDir::new().unwrap();
        write_wiki(tmp.path(), &[("a.md", "# T\n## H\npython\n")]);
        let idx = BM25Index::build(tmp.path()).unwrap();
        assert!(idx.search("zzz_never_seen_qqq", 5).is_empty());
    }

    #[test]
    fn total_tokens_matches_sum_of_doc_len() {
        let tmp = TempDir::new().unwrap();
        write_wiki(
            tmp.path(),
            &[
                ("a.md", "# T\n## H\none two three\n"),
                ("b.md", "# T\n## H\nfour five\n"),
            ],
        );
        let idx = BM25Index::build(tmp.path()).unwrap();
        let summed: u64 = idx.doc_len.iter().map(|&n| n as u64).sum();
        assert_eq!(idx.total_tokens(), summed);
    }
}
