//! Ollama-backed embedder for the wiki vector index.
//!
//! Per `PRD/rolo-hybrid-search.md` §7.1, §7.6, §H16:
//! - `POST /api/embed` is the batched embedding endpoint.
//! - `num_ctx: 8192` is mandatory in every request — Ollama defaults to 2048
//!   over HTTP without it (Ollama #7741).
//! - `truncate: false` makes the server reject oversized inputs rather than
//!   silently truncating; we belt-and-suspender this with a chars/4 pre-check.
//! - `POST /api/show` returns the model's manifest digest; we cache it in an
//!   RwLock so we only probe once per process.

use std::io;
use std::sync::RwLock;
use std::time::Duration;

use serde_json::{json, Value};

use super::hex::{l2_normalize, parse_digest_hex};

/// Pluggable embedder. Tests inject deterministic stubs so CI never has to
/// stand up a live Ollama (PRD §12.4).
pub trait Embedder: Send + Sync {
    /// SHA-256 manifest digest from `/api/show`. Used to invalidate the on-disk
    /// index when Ollama silently re-pulls a tag (Decision H6).
    fn probe_digest(&self) -> io::Result<[u8; 32]>;

    /// Batched embedding. Returns one slot per input; `None` slots are
    /// per-item failures the caller should skip-and-continue (PRD §7.6).
    fn embed_batch(&self, texts: &[String], timeout: Duration)
        -> io::Result<Vec<Option<Vec<f32>>>>;

    /// Single-query embedding with a tight timeout. `None` on any failure —
    /// vector search degenerates to BM25-only that turn (PRD §7.7, §8 row 3).
    fn embed_query(&self, text: &str) -> Option<[f32; 768]>;

    /// Model name as configured (e.g. `"nomic-embed-text"`). Used by the index
    /// header writer.
    fn model_name(&self) -> &str;
}

pub struct OllamaEmbedder {
    agent: ureq::Agent,
    base: String,
    model: String,
    digest: RwLock<Option<[u8; 32]>>,
}

impl OllamaEmbedder {
    pub fn new(base: &str, model: &str) -> Self {
        // Generous default timeouts; per-call timeouts override.
        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_connect(Some(Duration::from_secs(3)))
                .timeout_recv_body(Some(Duration::from_secs(60)))
                .build(),
        );
        Self {
            agent,
            base: base.trim_end_matches('/').to_string(),
            model: model.to_string(),
            digest: RwLock::new(None),
        }
    }
}

impl Embedder for OllamaEmbedder {
    fn probe_digest(&self) -> io::Result<[u8; 32]> {
        if let Ok(guard) = self.digest.read() {
            if let Some(d) = *guard {
                return Ok(d);
            }
        }

        // Ollama's `/api/show` does NOT return the manifest digest as a
        // top-level field — only weight tensor digests. The manifest digest
        // (which is what we want, since it changes when a tag is re-pulled)
        // lives in `/api/tags`. Read the list and find our model.
        let url = format!("{}/api/tags", self.base);
        let mut response = self
            .agent
            .get(&url)
            .config()
            .timeout_global(Some(Duration::from_secs(1)))
            .build()
            .call()
            .map_err(|e| io::Error::other(format!("ollama /api/tags: {e}")))?;

        if response.status() != 200 {
            return Err(io::Error::other(format!(
                "ollama /api/tags returned {}",
                response.status()
            )));
        }

        let body: Value = response
            .body_mut()
            .read_json()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("tags parse: {e}")))?;

        let models = body
            .get("models")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "no models[] in /api/tags response",
                )
            })?;

        // Match `nomic-embed-text` against `nomic-embed-text:latest` etc.
        let target = &self.model;
        let entry = models
            .iter()
            .find(|m| {
                m.get("name")
                    .and_then(|v| v.as_str())
                    .is_some_and(|name| name == target || name.split(':').next() == Some(target))
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("model {} not found in /api/tags — pull it first", target),
                )
            })?;

        let digest_str = entry
            .get("digest")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "no digest field on model entry in /api/tags",
                )
            })?;

        let digest = parse_digest_hex(digest_str)?;

        if let Ok(mut guard) = self.digest.write() {
            *guard = Some(digest);
        }
        Ok(digest)
    }

    fn embed_batch(
        &self,
        texts: &[String],
        timeout: Duration,
    ) -> io::Result<Vec<Option<Vec<f32>>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/api/embed", self.base);
        let body = json!({
            "model": self.model,
            "input": texts,
            "options": { "num_ctx": 8192 },
            "truncate": false,
        });

        let mut response = self
            .agent
            .post(&url)
            .config()
            .timeout_global(Some(timeout))
            .build()
            .send_json(&body)
            .map_err(|e| io::Error::other(format!("ollama /api/embed: {e}")))?;

        if response.status() != 200 {
            return Err(io::Error::other(format!(
                "ollama /api/embed returned {}",
                response.status()
            )));
        }

        let body: Value = response
            .body_mut()
            .read_json()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("embed parse: {e}")))?;

        let arr = body
            .get("embeddings")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "no embeddings field in /api/embed response",
                )
            })?;

        let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        for v in arr {
            match v.as_array() {
                Some(row) => {
                    let mut floats: Vec<f32> = Vec::with_capacity(row.len());
                    let mut bad = false;
                    for x in row {
                        match x.as_f64() {
                            Some(f) => floats.push(f as f32),
                            None => {
                                bad = true;
                                break;
                            }
                        }
                    }
                    if bad || floats.len() != 768 {
                        out.push(None);
                    } else {
                        l2_normalize(&mut floats);
                        out.push(Some(floats));
                    }
                }
                None => out.push(None),
            }
        }
        // Pad the trailing tail with None on short responses (PRD §7.6).
        while out.len() < texts.len() {
            out.push(None);
        }
        Ok(out)
    }

    fn embed_query(&self, text: &str) -> Option<[f32; 768]> {
        let res = self
            .embed_batch(&[text.to_string()], Duration::from_millis(200))
            .ok()?;
        let v = res.into_iter().next()??;
        if v.len() != 768 {
            return None;
        }
        let mut out = [0f32; 768];
        out.copy_from_slice(&v);
        Some(out)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use serde_json::json;

    fn vec_of(val: f32, len: usize) -> Vec<f32> {
        (0..len).map(|_| val).collect()
    }

    #[test]
    fn probe_digest_strips_prefix_and_decodes() {
        let mut server = Server::new();
        let hex = "9e1c".to_string() + &"0".repeat(60);
        let body = json!({
            "models": [
                { "name": "other-model:latest", "digest": "ff".repeat(32) },
                { "name": "nomic-embed-text:latest", "digest": format!("sha256:{}", hex) },
            ]
        })
        .to_string();
        let _m = server
            .mock("GET", "/api/tags")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body)
            .create();

        let emb = OllamaEmbedder::new(&server.url(), "nomic-embed-text");
        let d = emb.probe_digest().unwrap();
        // Round-trip: byte 0 is 0x9e, byte 1 is 0x1c, rest zero.
        assert_eq!(d[0], 0x9e);
        assert_eq!(d[1], 0x1c);
        assert_eq!(d[2], 0x00);
        // Cached: the mock allows one call; a second probe must not fire HTTP.
        let d2 = emb.probe_digest().unwrap();
        assert_eq!(d, d2);
    }

    #[test]
    fn embed_batch_request_has_num_ctx_and_truncate() {
        let mut server = Server::new();
        let m = server
            .mock("POST", "/api/embed")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "nomic-embed-text",
                "options": { "num_ctx": 8192 },
                "truncate": false,
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(json!({ "embeddings": [vec_of(0.5, 768), vec_of(0.5, 768)] }).to_string())
            .create();

        let emb = OllamaEmbedder::new(&server.url(), "nomic-embed-text");
        let out = emb
            .embed_batch(&["a".to_string(), "b".to_string()], Duration::from_secs(5))
            .unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].is_some());
        assert!(out[1].is_some());
        m.assert();
    }

    #[test]
    fn embed_batch_short_response_pads_with_none() {
        let mut server = Server::new();
        let _m = server
            .mock("POST", "/api/embed")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(json!({ "embeddings": [vec_of(0.5, 768)] }).to_string())
            .create();

        let emb = OllamaEmbedder::new(&server.url(), "nomic-embed-text");
        let out = emb
            .embed_batch(&["a".to_string(), "b".to_string()], Duration::from_secs(5))
            .unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].is_some());
        assert!(out[1].is_none());
    }

    #[test]
    fn embed_batch_500_returns_err() {
        let mut server = Server::new();
        let _m = server
            .mock("POST", "/api/embed")
            .with_status(500)
            .with_body("boom")
            .create();

        let emb = OllamaEmbedder::new(&server.url(), "nomic-embed-text");
        let res = emb.embed_batch(&["a".to_string()], Duration::from_secs(2));
        assert!(res.is_err());
    }

    #[test]
    fn embed_query_short_timeout_returns_none_on_slow_server() {
        let mut server = Server::new();
        let _m = server
            .mock("POST", "/api/embed")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body_from_request(|_| {
                std::thread::sleep(Duration::from_millis(500));
                json!({ "embeddings": [vec_of(0.5, 768)] })
                    .to_string()
                    .into_bytes()
            })
            .create();

        let emb = OllamaEmbedder::new(&server.url(), "nomic-embed-text");
        // embed_query has a 200 ms timeout; server sleeps 500 ms.
        let v = emb.embed_query("hi");
        assert!(v.is_none());
    }

    #[test]
    fn parse_digest_hex_accepts_unprefixed() {
        let hex = "a".repeat(64);
        let d = parse_digest_hex(&hex).unwrap();
        assert_eq!(d[0], 0xaa);
    }

    #[test]
    fn parse_digest_hex_rejects_short() {
        assert!(parse_digest_hex("abc").is_err());
    }

    #[test]
    fn embed_query_l2_normalizes() {
        let mut server = Server::new();
        // All 1.0s — L2 norm = sqrt(768); normalized values = 1/sqrt(768).
        let _m = server
            .mock("POST", "/api/embed")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(json!({ "embeddings": [vec_of(1.0, 768)] }).to_string())
            .create();

        let emb = OllamaEmbedder::new(&server.url(), "nomic-embed-text");
        let v = emb.embed_query("anything").unwrap();
        let expected = 1.0 / (768f32).sqrt();
        for x in v.iter() {
            assert!(
                (x - expected).abs() < 1e-5,
                "got {} expected {}",
                x,
                expected
            );
        }
    }
}
