pub mod atomic;
pub mod bm25;
pub mod bootstrap;
pub mod chunker;
pub mod compiler;
pub mod config;
pub mod dreaming;
pub mod dreams_log;
pub mod embeddings;
pub mod events;
pub mod idle;
pub mod linter;
pub mod logger;
pub mod meta;
pub mod prompt;
pub mod search;
pub mod user_profile;

use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use chrono::Local;

use crate::vault::bm25::BM25Index;
use crate::vault::config::SOFT_TOKEN_CAP;
use crate::vault::embeddings::{Embedder, EmbeddingIndex, OllamaEmbedder};
use crate::vault::logger::ExperienceLogger;
use crate::vault::meta::VaultMeta;
use crate::vault::search::HybridSearcher;

/// Default Ollama base for the embedder. Mirrors the chat client's `OLLAMA_BASE`
/// in `ollama.rs` — kept separate so embedding sickness can't reach chat.
const EMBED_OLLAMA_BASE: &str = "http://127.0.0.1:11434";
const EMBED_MODEL: &str = "nomic-embed-text";

/// Rolo's persistent memory. Owned as a single `Arc<Vault>` in Tauri state;
/// holds the experience logger, the BM25 index over wiki chunks, the
/// `meta.json` snapshot, and the PromptAssembler. See PRD §9.1.
pub struct Vault {
    root: PathBuf,
    pub logger: Arc<ExperienceLogger>,
    bm25: Arc<RwLock<Option<BM25Index>>>,
    embedding_index: Arc<RwLock<Option<EmbeddingIndex>>>,
    embedder: Arc<dyn Embedder>,
    pub searcher: Arc<HybridSearcher>,
    meta: Arc<Mutex<VaultMeta>>,
    pub assembler: Arc<crate::vault::prompt::PromptAssembler>,
}

impl Vault {
    /// Bootstrap the vault tree if missing, sweep retention, build BM25, and
    /// return a ready-to-use Vault. All sub-steps are best-effort: a failure
    /// in any one logs and continues so Rolo always boots (PRD §2.4).
    pub fn open_or_init(root: PathBuf) -> Arc<Self> {
        let embedder: Arc<dyn Embedder> =
            Arc::new(OllamaEmbedder::new(EMBED_OLLAMA_BASE, EMBED_MODEL));
        Self::open_or_init_with_embedder(root, embedder)
    }

    /// Test-only constructor that lets integration tests inject a stub
    /// embedder. Production code uses `open_or_init`.
    pub fn open_or_init_with_embedder(root: PathBuf, embedder: Arc<dyn Embedder>) -> Arc<Self> {
        if let Err(e) = bootstrap::bootstrap_vault(&root) {
            log::error!(
                "[Rolo vault] bootstrap failed at {:?}: {} — vault may be partial",
                root,
                e
            );
        }

        let meta_path = root.join("meta.json");
        let mut meta = VaultMeta::load_or_default(&meta_path);

        let events_dir = root.join("events");
        let logger = ExperienceLogger::new(events_dir);

        let swept = logger.retention_sweep();
        log::info!("[Rolo vault] retention sweep removed {} files", swept);
        meta.stats.last_retention_sweep = Some(Local::now());

        let bm25_opt = match BM25Index::build(&root.join("wiki")) {
            Ok(idx) => Some(idx),
            Err(e) => {
                log::error!(
                    "[Rolo vault] BM25 build failed: {} — falling back to const prompt",
                    e
                );
                None
            }
        };

        if let Some(ref idx) = bm25_opt {
            let total = idx.total_tokens();
            meta.stats.total_wiki_tokens = total;
            if total as usize > SOFT_TOKEN_CAP {
                log::warn!(
                    "vault wiki size {} tokens exceeds soft cap {}; dreaming compaction overdue",
                    total,
                    SOFT_TOKEN_CAP
                );
            }
        }

        // Persist all meta mutations once at the end — single fsync covers the
        // initial-create, retention-sweep, and BM25-stats updates above.
        if let Err(e) = meta.save(&meta_path) {
            log::warn!(
                "[Rolo vault] could not persist meta.json at {:?}: {}",
                meta_path,
                e
            );
        }

        let bm25 = Arc::new(RwLock::new(bm25_opt));

        // Probe the digest with a 1 s timeout (best-effort); on failure,
        // embedding_index stays None and search degrades to BM25-only
        // (PRD §7.9 step 4).
        let embeddings_dir = root.join("embeddings");
        let _ = std::fs::create_dir_all(&embeddings_dir);

        let probed_digest = match embedder.probe_digest() {
            Ok(d) => Some(d),
            Err(e) => {
                log::warn!(
                    "[Rolo vault] embedder probe failed: {} — BM25-only mode until Ollama is reachable",
                    e
                );
                None
            }
        };

        // Try to load an existing index — only when probe succeeded so we can
        // validate the digest. On mismatch or absent file, mark dirty and
        // wipe stale files.
        let embedding_index_opt: Option<EmbeddingIndex> = match probed_digest {
            Some(digest) => match EmbeddingIndex::load(&embeddings_dir, &digest) {
                Ok(Some(idx)) => {
                    log::info!("[Rolo vault] embedding index loaded ({} rows)", idx.len());
                    Some(idx)
                }
                Ok(None) => {
                    log::info!(
                        "[Rolo vault] no usable embedding index on disk — will rebuild on next dream"
                    );
                    // Wipe partial/stale files so a future load attempt is clean.
                    let _ = std::fs::remove_file(embeddings_dir.join("index.bin"));
                    let _ = std::fs::remove_file(embeddings_dir.join("chunks.jsonl"));
                    meta.embedding_dirty = true;
                    None
                }
                Err(e) => {
                    log::warn!(
                        "[Rolo vault] embedding index load errored: {} — wiping and marking dirty",
                        e
                    );
                    let _ = std::fs::remove_file(embeddings_dir.join("index.bin"));
                    let _ = std::fs::remove_file(embeddings_dir.join("chunks.jsonl"));
                    meta.embedding_dirty = true;
                    None
                }
            },
            None => None,
        };

        let embedding_index: Arc<RwLock<Option<EmbeddingIndex>>> =
            Arc::new(RwLock::new(embedding_index_opt));

        let meta = Arc::new(Mutex::new(meta));

        // Pre-warm the embedder in a background thread so the first real query
        // doesn't pay the cold-start penalty (PRD §13 R1). Best-effort: if it
        // fails, the 200 ms timeout in `embed_query` is the safety valve.
        if probed_digest.is_some() {
            let warm_embedder = Arc::clone(&embedder);
            std::thread::spawn(move || {
                let _ = warm_embedder.embed_query(".");
            });
        }

        let searcher = Arc::new(HybridSearcher::new(
            Arc::clone(&bm25),
            Arc::clone(&embedding_index),
            Arc::clone(&embedder),
        ));

        // 8. Build the PromptAssembler over the HybridSearcher. Falls back to
        //    the const safety-net prompt when BM25 is None (PRD §8.6 row 1).
        //    Production reads slot 3 from the wiki via hybrid search — the
        //    SqliteInbox path is dead in production (set_inbox is never wired)
        //    and would leave slot 3 empty, so user/* and relationships/* would
        //    never reach the prompt. Override the default to Wiki here.
        let prompt_config = crate::vault::prompt::PromptConfig {
            source: crate::vault::prompt::ContextSource::Wiki,
            ..crate::vault::prompt::PromptConfig::default()
        };
        let assembler = Arc::new(crate::vault::prompt::PromptAssembler::new(
            Arc::clone(&searcher),
            prompt_config,
            crate::ollama::FALLBACK_SYSTEM_PROMPT.to_string(),
        ));

        Arc::new(Self {
            root,
            logger,
            bm25,
            embedding_index,
            embedder,
            searcher,
            meta,
            assembler,
        })
    }

    /// Build a fresh BM25Index off-lock, then swap it in under the write lock.
    /// Then (per PRD §7.8) rebuild the embedding index sequentially against a
    /// snapshot of the BM25 chunks. On any embedding failure, BM25 stays
    /// swapped — Rolo continues with BM25-only search.
    ///
    /// Reserved for the Phase 5 dreaming compiler; the API is stable.
    pub fn rebuild_index(&self) {
        let wiki_dir = self.root.join("wiki");
        let chunks_snapshot: Option<Vec<crate::vault::chunker::Chunk>> =
            match BM25Index::build(&wiki_dir) {
                Ok(new_idx) => {
                    let total = new_idx.total_tokens();
                    let chunks = new_idx.chunks().to_vec();
                    {
                        let mut guard = self.bm25.write().unwrap_or_else(|p| p.into_inner());
                        *guard = Some(new_idx);
                    }
                    let meta_path = self.root.join("meta.json");
                    let mut meta = self.meta.lock().unwrap_or_else(|p| p.into_inner());
                    meta.stats.total_wiki_tokens = total;
                    if let Err(e) = meta.save(&meta_path) {
                        log::warn!(
                            "[Rolo vault] could not persist meta after rebuild_index: {}",
                            e
                        );
                    }
                    Some(chunks)
                }
                Err(e) => {
                    log::error!(
                        "[Rolo vault] rebuild_index failed: {} — keeping previous index",
                        e
                    );
                    None
                }
            };

        if let Some(chunks) = chunks_snapshot {
            self.rebuild_embeddings(&chunks);
        }
    }

    /// Rebuild the embedding index from a snapshot of chunks. Skipped when
    /// `meta.embedding_optout == true`. Best-effort throughout — any failure
    /// (Ollama down, digest mismatch, network error) leaves Rolo on the
    /// BM25-only path and logs a warning. (PRD §7.8)
    fn rebuild_embeddings(&self, chunks: &[crate::vault::chunker::Chunk]) {
        // Optout flag short-circuit (PRD §13 row 13).
        {
            let meta = self.meta.lock().unwrap_or_else(|p| p.into_inner());
            if meta.embedding_optout {
                log::info!("[Rolo vault] embedding rebuild skipped (meta.embedding_optout = true)");
                return;
            }
        }

        let embeddings_dir = self.root.join("embeddings");
        if let Err(e) = std::fs::create_dir_all(&embeddings_dir) {
            log::warn!(
                "[Rolo vault] could not create embeddings dir {:?}: {} — skipping rebuild",
                embeddings_dir,
                e
            );
            return;
        }

        // 1. Probe digest. On failure → abort embedding rebuild gracefully.
        let digest = match self.embedder.probe_digest() {
            Ok(d) => d,
            Err(e) => {
                log::warn!(
                    "[Rolo vault] embedder probe failed during rebuild: {} — keeping previous embedding index",
                    e
                );
                return;
            }
        };

        // 2. If existing index has a different digest, wipe sidecar files and
        //    invalidate the cache namespace before rebuilding (PRD §7.8 step b).
        {
            let existing = self
                .embedding_index
                .read()
                .unwrap_or_else(|p| p.into_inner());
            if let Some(idx) = existing.as_ref() {
                if idx.header().model_digest != digest {
                    log::info!("[Rolo vault] embedding model digest changed — wiping stale index");
                    drop(existing);
                    let mut guard = self
                        .embedding_index
                        .write()
                        .unwrap_or_else(|p| p.into_inner());
                    *guard = None;
                    let _ = std::fs::remove_file(embeddings_dir.join("index.bin"));
                    let _ = std::fs::remove_file(embeddings_dir.join("chunks.jsonl"));
                }
            }
        }

        // 3. Open the embedding cache (namespace = "{model}:{digest_hex}:768").
        let digest_hex = crate::vault::embeddings::hex::encode_hex_32(&digest);
        let namespace = format!(
            "{}:{}:{}",
            self.embedder.model_name(),
            digest_hex,
            crate::vault::embeddings::index::VECTOR_DIM
        );
        let mut cache =
            match crate::vault::embeddings::EmbeddingCache::open(&embeddings_dir, namespace) {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("[Rolo vault] could not open embedding cache: {}", e);
                    return;
                }
            };

        // Periodic compaction guard against unbounded cache.jsonl growth.
        if let Err(e) = cache.compact_if_oversized() {
            log::warn!("[Rolo vault] embedding cache compaction errored: {}", e);
        }

        // 4. Build the new index. Progress callback logs every 64 chunks.
        let total = chunks.len();
        let new_idx =
            match EmbeddingIndex::build(chunks, &*self.embedder, &mut cache, |done, total| {
                if done % 64 == 0 || done == total {
                    log::info!(
                        "[Rolo vault] embedding rebuild progress: {}/{}",
                        done,
                        total
                    );
                }
            }) {
                Ok(idx) => idx,
                Err(e) => {
                    log::warn!(
                        "[Rolo vault] embedding rebuild failed: {} — keeping previous index",
                        e
                    );
                    return;
                }
            };

        log::info!(
            "[Rolo vault] embedding rebuild complete: {} chunks indexed (of {} total)",
            new_idx.len(),
            total
        );

        // 5. Persist atomically (chunks.jsonl first, then index.bin per §7.4).
        if let Err(e) = new_idx.persist(&embeddings_dir) {
            log::warn!(
                "[Rolo vault] embedding index persist failed: {} — in-memory index kept, on-disk stale",
                e
            );
            // Still swap into RwLock — in-memory > none.
        }

        // 6. Swap into RwLock; clear dirty flag.
        {
            let mut guard = self
                .embedding_index
                .write()
                .unwrap_or_else(|p| p.into_inner());
            *guard = Some(new_idx);
        }
        {
            let meta_path = self.root.join("meta.json");
            let mut meta = self.meta.lock().unwrap_or_else(|p| p.into_inner());
            meta.embedding_dirty = false;
            if let Err(e) = meta.save(&meta_path) {
                log::warn!(
                    "[Rolo vault] could not persist meta after embedding rebuild: {}",
                    e
                );
            }
        }
    }

    // ----------------------------------------------------------------
    // Dreaming-task helpers (D4) — small surface so the poll loop in
    // `vault::dreaming` can avoid a `pub` field on the meta or root.
    // ----------------------------------------------------------------

    /// Clone the meta snapshot out from under the mutex. Cheap-ish (one
    /// allocation per BTree). The poll loop reads this once per 30s tick.
    pub fn meta_snapshot(&self) -> VaultMeta {
        let m = self.meta.lock().unwrap_or_else(|p| p.into_inner());
        m.clone()
    }

    /// Path to the wiki tree under this vault. Stable across the vault's
    /// lifetime. Returned by value (cheap `PathBuf` clone) so callers can
    /// hold it across `await` points without lifetime gymnastics.
    pub fn wiki_root(&self) -> PathBuf {
        self.root.join("wiki")
    }

    /// Path to the dreams-replay artifacts directory. Created on demand by
    /// the compiler — `wiki_root` and this path are siblings.
    pub fn dreams_artifacts_dir(&self) -> PathBuf {
        self.root.join("dreams_artifacts")
    }

    /// A fresh `DreamsLog` rooted at this vault. Cheap to construct (no
    /// I/O, no FS create) and one-per-cycle is the right granularity since
    /// the lock is per-instance.
    pub fn dreams_log_handle(&self) -> crate::vault::dreams_log::DreamsLog {
        crate::vault::dreams_log::DreamsLog::new(&self.root)
    }

    /// Count raw events in `<root>/events/*.jsonl` whose timestamps land
    /// after `meta.last_compile_time`. When `last_compile_time` is None,
    /// every line counts. Best-effort: unparseable lines are ignored, not
    /// rejected — the dreaming compiler treats ground truth as advisory.
    pub fn count_unprocessed_events(&self) -> usize {
        let last = self.meta_snapshot().last_compile_time;
        let events_dir = self.root.join("events");
        let mut count = 0usize;
        let entries = match std::fs::read_dir(&events_dir) {
            Ok(rd) => rd,
            Err(_) => return 0,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .extension()
                .and_then(|s| s.to_str())
                .is_none_or(|s| s != "jsonl")
            {
                continue;
            }
            let contents = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            for line in contents.lines() {
                if line.is_empty() {
                    continue;
                }
                if let Some(last_ts) = last {
                    let v: serde_json::Value = match serde_json::from_str(line) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let ts_str = match v.get("ts").and_then(|t| t.as_str()) {
                        Some(s) => s,
                        None => continue,
                    };
                    let ts = match chrono::DateTime::parse_from_rfc3339(ts_str) {
                        Ok(t) => t.with_timezone(&Local),
                        Err(_) => continue,
                    };
                    if ts > last_ts {
                        count += 1;
                    }
                } else {
                    count += 1;
                }
            }
        }
        count
    }

    /// Build a `CompileBatch` of up to `max` events newer than
    /// `meta.last_compile_time`. Events are emitted in encountered order
    /// across daily JSONL files (sorted by filename — date-stamped, so
    /// chronological). Returns `Ok(empty_batch)` rather than an error when
    /// no events qualify; the poll loop treats empty as "skip and retry".
    pub fn build_compile_batch(
        &self,
        max: usize,
    ) -> std::io::Result<crate::vault::compiler::CompileBatch> {
        use crate::vault::compiler::{CompileBatch, CompiledEvent};
        let last = self.meta_snapshot().last_compile_time;
        let events_dir = self.root.join("events");
        let mut events: Vec<CompiledEvent> = Vec::new();
        let entries = match std::fs::read_dir(&events_dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(CompileBatch { events });
            }
            Err(e) => return Err(e),
        };
        // Collect file paths first so we can sort deterministically (by
        // filename, which matches chronological since names are dated).
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s == "jsonl")
            })
            .collect();
        paths.sort();

        // Filenames are `%Y-%m-%d.jsonl`. Skip whole files whose date is
        // strictly before `last_compile_time`'s date — none of their events
        // can survive the per-line `ts <= last_ts` filter, so the read +
        // parse is wasted work that grows linearly with vault age.
        let last_date = last.map(|t| t.date_naive());

        'outer: for path in paths {
            if let Some(cutoff) = last_date {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(file_date) = chrono::NaiveDate::parse_from_str(stem, "%Y-%m-%d") {
                        if file_date < cutoff {
                            continue;
                        }
                    }
                }
            }
            let contents = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            for line in contents.lines() {
                if line.is_empty() {
                    continue;
                }
                let v: serde_json::Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(last_ts) = last {
                    let ts_str = match v.get("ts").and_then(|t| t.as_str()) {
                        Some(s) => s,
                        None => continue,
                    };
                    let ts = match chrono::DateTime::parse_from_rfc3339(ts_str) {
                        Ok(t) => t.with_timezone(&Local),
                        Err(_) => continue,
                    };
                    if ts <= last_ts {
                        continue;
                    }
                }
                let event_id = v
                    .get("event_id")
                    .and_then(|s| s.as_str())
                    .unwrap_or("legacy_unknown")
                    .to_string();
                let kind_raw = v.get("type").and_then(|s| s.as_str()).unwrap_or("unknown");
                let kind = crate::vault::compiler::prompt_kind_label(kind_raw).to_string();
                events.push(CompiledEvent {
                    event_id,
                    raw_json_line: line.to_string(),
                    kind,
                });
                if events.len() >= max {
                    break 'outer;
                }
            }
        }
        Ok(CompileBatch { events })
    }

    /// Add the given source IDs (event IDs or wiki paths) to
    /// `meta.revert_blocklist` and persist meta.json. Called by the
    /// `revert_fact` Tauri command when the user reverts a learned fact —
    /// future compile runs read this set so they can lower confidence on the
    /// originating events. Persisted under the meta lock to keep the on-disk
    /// state consistent with the in-memory snapshot.
    pub fn add_to_revert_blocklist(&self, ids: &[String]) -> std::io::Result<()> {
        let mut meta = self.meta.lock().unwrap_or_else(|p| p.into_inner());
        for id in ids {
            meta.revert_blocklist.insert(id.clone());
        }
        let path = self.root.join("meta.json");
        meta.save(&path)
    }

    // ----------------------------------------------------------------
    // User-profile API (Phase 6 of the Command Center PRD §6).
    //
    // `user_profile.json` lives at `<root>/user_profile.json` and is owned
    // entirely by the Vault — Command Center commands must go through these
    // four methods (no raw filesystem access). See PRD line 120.
    // ----------------------------------------------------------------

    /// Path to the user-profile sidecar file. Private helper used by all four
    /// public methods so they share one source of truth for placement.
    fn user_profile_path(&self) -> PathBuf {
        self.root.join("user_profile.json")
    }

    /// Load the user profile if it exists. Returns `None` when the file is
    /// missing — distinct from "saved but all sections empty", which returns
    /// `Some(UserProfile { all empty })`. A parse error is treated the same
    /// as missing (`None`); the Phase-7 UI can re-save to recover from
    /// corruption. The warn-log line is the audit trail.
    pub fn load_user_profile(&self) -> Option<crate::vault::user_profile::UserProfile> {
        let path = self.user_profile_path();
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
            Err(e) => {
                log::warn!(
                    "[Rolo vault] could not read user_profile.json at {:?}: {} — treating as missing",
                    path,
                    e
                );
                return None;
            }
        };
        match serde_json::from_slice::<crate::vault::user_profile::UserProfile>(&bytes) {
            Ok(p) => Some(p),
            Err(e) => {
                log::warn!(
                    "[Rolo vault] user_profile.json at {:?} failed to parse: {} — treating as missing",
                    path,
                    e
                );
                None
            }
        }
    }

    /// Atomically write a new user profile to `<vault_root>/user_profile.json`.
    /// `updated_at` is stamped HERE to `Local::now()` (the field on the input
    /// is ignored) so the timestamp reflects the actual save, not whatever the
    /// caller happened to have in memory. Returns the actually-persisted
    /// profile so callers can hand it back to the frontend without a second
    /// load.
    pub fn save_user_profile(
        &self,
        profile: crate::vault::user_profile::UserProfile,
    ) -> std::io::Result<crate::vault::user_profile::UserProfile> {
        let persisted = crate::vault::user_profile::UserProfile {
            version: profile.version,
            about_you: profile.about_you,
            people_and_context: profile.people_and_context,
            how_rolo_should_respond: profile.how_rolo_should_respond,
            updated_at: Local::now(),
        };
        let bytes = serde_json::to_vec_pretty(&persisted).map_err(std::io::Error::other)?;
        crate::vault::atomic::atomic_write(&self.user_profile_path(), &bytes)?;
        Ok(persisted)
    }

    /// Delete `<vault_root>/user_profile.json`. Idempotent — returns `Ok`
    /// even if the file was already missing. The Memory tab's "Clear User
    /// Profile" button wires here (Phase 7).
    pub fn clear_user_profile(&self) -> std::io::Result<()> {
        let path = self.user_profile_path();
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Walk the writable wiki and remove any lines whose `<!-- src: ... -->`
    /// provenance markers cite event IDs that resolve (in the events store)
    /// to `UserProfileUpdate` events. Re-indexes BM25 and embeddings after
    /// removal. Returns the count of lines removed.
    ///
    /// **Resolution strategy:** the dreaming compiler tags each appended fact
    /// line with `<!-- src: evt_id1,evt_id2 -->` (see
    /// `compiler.rs::apply_with_date`). We parse those IDs back out per line,
    /// look them up in `<root>/events/*.jsonl`, and remove the line iff
    /// **all** cited IDs belong to `UserProfileUpdate` events. Mixed-source
    /// lines (e.g. a fact grounded in both a chat and a profile update) are
    /// preserved — removing them would lose chat-derived knowledge the user
    /// did not ask to clear.
    pub fn clear_user_profile_dreams(&self) -> std::io::Result<usize> {
        // 1. Build the set of all UserProfileUpdate event IDs from the events
        //    store. Cheap-ish: events files are typically <1 MB per day and
        //    we only care about lines with `"type":"user_profile_update"`.
        let profile_ids = self.collect_user_profile_event_ids()?;
        if profile_ids.is_empty() {
            // No profile events on disk → nothing to prune. Skip the wiki
            // walk and the rebuild; saves a noticeable chunk of work on
            // first-run vaults.
            return Ok(0);
        }

        // 2. Walk the wiki and rewrite each file with profile-only lines
        //    removed.
        let wiki_root = self.root.join("wiki");
        if !wiki_root.exists() {
            return Ok(0);
        }
        let mut total_removed = 0usize;
        let mut files_changed = false;
        for entry in walkdir::WalkDir::new(&wiki_root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path().to_path_buf();
            // Only touch markdown files — we never wrote `<!-- src: ... -->`
            // markers into JSON sidecars and don't want surprises.
            if path
                .extension()
                .and_then(|s| s.to_str())
                .is_none_or(|s| s != "md")
            {
                continue;
            }
            let original = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    log::warn!(
                        "[Rolo vault] clear_user_profile_dreams: could not read {:?}: {}",
                        path,
                        e
                    );
                    continue;
                }
            };
            let (rewritten, removed) = strip_user_profile_lines(&original, &profile_ids);
            if removed == 0 {
                continue;
            }
            total_removed += removed;
            files_changed = true;
            if let Err(e) = crate::vault::atomic::atomic_write(&path, rewritten.as_bytes()) {
                log::warn!(
                    "[Rolo vault] clear_user_profile_dreams: could not rewrite {:?}: {}",
                    path,
                    e
                );
            }
        }

        // 3. If anything changed, rebuild BM25 + embeddings so search no
        //    longer surfaces the removed content. `rebuild_index` already
        //    handles "embedder offline" / "optout" gracefully.
        if files_changed {
            self.rebuild_index();
        }

        Ok(total_removed)
    }

    /// Scan `<root>/events/*.jsonl` and collect every `event_id` whose line
    /// carries `"type":"user_profile_update"`. Cheap-string match first
    /// (avoids parsing every chat line), then full JSON parse to extract the
    /// id. Returns an empty set on a missing events dir.
    fn collect_user_profile_event_ids(&self) -> std::io::Result<std::collections::HashSet<String>> {
        use std::collections::HashSet;
        let events_dir = self.root.join("events");
        let mut ids: HashSet<String> = HashSet::new();
        let entries = match std::fs::read_dir(&events_dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ids),
            Err(e) => return Err(e),
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .extension()
                .and_then(|s| s.to_str())
                .is_none_or(|s| s != "jsonl")
            {
                continue;
            }
            let contents = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            for line in contents.lines() {
                if !line.contains("\"user_profile_update\"") {
                    continue;
                }
                let v: serde_json::Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let is_profile =
                    v.get("type").and_then(|t| t.as_str()) == Some("user_profile_update");
                if !is_profile {
                    continue;
                }
                if let Some(id) = v.get("event_id").and_then(|s| s.as_str()) {
                    ids.insert(id.to_string());
                }
            }
        }
        Ok(ids)
    }

    /// Stamp `meta.last_compile_time` with `when` and persist meta. Called
    /// by the dreaming poll loop after a successful compile so subsequent
    /// `count_unprocessed_events`/`build_compile_batch` calls skip the
    /// freshly-compiled events. Failures are logged and swallowed —
    /// in-memory meta still moves forward.
    pub fn record_compile_complete(&self, when: chrono::DateTime<Local>) {
        let meta_path = self.root.join("meta.json");
        let mut m = self.meta.lock().unwrap_or_else(|p| p.into_inner());
        m.last_compile_time = Some(when);
        m.stats.total_dreams = m.stats.total_dreams.saturating_add(1);
        if let Err(e) = m.save(&meta_path) {
            log::warn!(
                "[Rolo vault] could not persist meta after compile complete: {}",
                e
            );
        }
    }
}

/// Rewrite `original` with any line whose `<!-- src: id1,id2 -->` marker
/// cites ONLY ids in `profile_ids` removed. Returns the new content and the
/// count of lines stripped.
///
/// "Cites only" is intentional: a line grounded in both a chat event and a
/// profile-update event is preserved — clearing the profile must not destroy
/// chat-derived knowledge. Lines without a `src:` marker are also preserved
/// (e.g. headers, manually-edited content, prose).
fn strip_user_profile_lines(
    original: &str,
    profile_ids: &std::collections::HashSet<String>,
) -> (String, usize) {
    let had_trailing_newline = original.ends_with('\n');
    let mut removed = 0usize;
    let mut kept: Vec<&str> = Vec::new();
    let body = original.strip_suffix('\n').unwrap_or(original);
    for line in body.split('\n') {
        if line_cites_only_profile_ids(line, profile_ids) {
            removed += 1;
            continue;
        }
        kept.push(line);
    }
    if removed == 0 {
        // Cheap fast path — return the original to avoid an allocation when
        // a wiki file is entirely chat-derived (the common case).
        return (original.to_string(), 0);
    }
    let mut out = kept.join("\n");
    if had_trailing_newline && !out.is_empty() {
        out.push('\n');
    }
    (out, removed)
}

/// True iff `line` carries a `<!-- src: ... -->` marker AND every id inside
/// that marker is in `profile_ids`. Lines without a marker return false
/// (they're not profile dreams).
fn line_cites_only_profile_ids(
    line: &str,
    profile_ids: &std::collections::HashSet<String>,
) -> bool {
    // Match the exact format `compiler.rs::apply_with_date` emits.
    const PREFIX: &str = "<!-- src: ";
    const SUFFIX: &str = " -->";
    let start = match line.find(PREFIX) {
        Some(i) => i + PREFIX.len(),
        None => return false,
    };
    // Take the first closing ` -->` after the prefix so we don't get fooled
    // by superseded-line markers (`<!-- superseded_by: ... --><!-- old line
    // ... -->`) — those don't carry our `src:` token at the start of the
    // payload anyway, so the `find(PREFIX)` already filtered them out.
    let rest = &line[start..];
    let end = match rest.find(SUFFIX) {
        Some(i) => i,
        None => return false,
    };
    let ids_blob = &rest[..end];
    let ids: Vec<&str> = ids_blob.split(',').map(|s| s.trim()).collect();
    if ids.is_empty() {
        return false;
    }
    ids.iter().all(|id| profile_ids.contains(*id))
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod user_profile_tests {
    use super::user_profile::UserProfile;
    use super::*;
    use crate::vault::embeddings::Embedder;
    use std::io;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::TempDir;

    /// Embedder that fails every operation — drives the BM25-only path so
    /// these tests don't need a live Ollama. Mirrors the `DeadEmbedder` in
    /// other modules' tests.
    struct DeadEmbedder;
    impl Embedder for DeadEmbedder {
        fn probe_digest(&self) -> io::Result<[u8; 32]> {
            Err(io::Error::other("dead embedder"))
        }
        fn embed_batch(
            &self,
            _texts: &[String],
            _timeout: Duration,
        ) -> io::Result<Vec<Option<Vec<f32>>>> {
            Err(io::Error::other("dead embedder"))
        }
        fn embed_query(&self, _text: &str) -> Option<[f32; 768]> {
            None
        }
        fn model_name(&self) -> &str {
            "dead"
        }
    }

    fn fresh_vault() -> (TempDir, Arc<Vault>) {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path().join("vault");
        let embedder: Arc<dyn Embedder> = Arc::new(DeadEmbedder);
        let vault = Vault::open_or_init_with_embedder(root, embedder);
        (tmp, vault)
    }

    fn sample_profile() -> UserProfile {
        UserProfile {
            version: 1,
            about_you: "ML researcher in Brooklyn.".into(),
            people_and_context: "Husband Sam (eng). Cat Rolo (digital).".into(),
            how_rolo_should_respond: "Be brief. Don't apologize.".into(),
            // Stamped over by save_user_profile, but Default needs a value.
            updated_at: chrono::Local::now(),
        }
    }

    #[test]
    fn save_then_load_user_profile_round_trips() {
        let (_tmp, vault) = fresh_vault();
        let profile = sample_profile();
        let saved = vault.save_user_profile(profile.clone()).expect("save");
        assert_eq!(saved.about_you, profile.about_you);
        assert_eq!(saved.people_and_context, profile.people_and_context);
        assert_eq!(
            saved.how_rolo_should_respond,
            profile.how_rolo_should_respond
        );

        let loaded = vault.load_user_profile().expect("loaded some");
        assert_eq!(loaded.about_you, profile.about_you);
        assert_eq!(loaded.people_and_context, profile.people_and_context);
        assert_eq!(
            loaded.how_rolo_should_respond,
            profile.how_rolo_should_respond
        );
    }

    #[test]
    fn load_user_profile_returns_none_when_missing() {
        let (_tmp, vault) = fresh_vault();
        assert!(vault.load_user_profile().is_none());
    }

    #[test]
    fn clear_user_profile_deletes_file() {
        let (_tmp, vault) = fresh_vault();
        vault.save_user_profile(sample_profile()).expect("save");
        assert!(vault.load_user_profile().is_some());
        vault.clear_user_profile().expect("clear");
        assert!(
            vault.load_user_profile().is_none(),
            "load must return None after clear"
        );
    }

    #[test]
    fn clear_user_profile_is_idempotent() {
        let (_tmp, vault) = fresh_vault();
        // Never saved — clear must still succeed.
        vault.clear_user_profile().expect("clear on missing");
        // Calling twice in a row must also succeed.
        vault.clear_user_profile().expect("second clear on missing");
    }

    #[test]
    fn save_user_profile_stamps_fresh_updated_at() {
        let (_tmp, vault) = fresh_vault();
        let old = chrono::Local.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let profile = UserProfile {
            updated_at: old,
            ..sample_profile()
        };
        let before_save = chrono::Local::now();
        let saved = vault.save_user_profile(profile).expect("save");
        assert!(
            saved.updated_at >= before_save,
            "save_user_profile must stamp a fresh updated_at; got {:?} < {:?}",
            saved.updated_at,
            before_save
        );
        assert!(
            saved.updated_at > old,
            "updated_at must be newer than the caller-supplied value"
        );
    }

    #[test]
    fn save_user_profile_persisted_value_round_trips_via_load() {
        // Extra coverage: the value returned by save() must equal the value
        // returned by a subsequent load() — i.e. the in-memory and on-disk
        // states agree.
        let (_tmp, vault) = fresh_vault();
        let saved = vault.save_user_profile(sample_profile()).expect("save");
        let loaded = vault.load_user_profile().expect("load");
        assert_eq!(saved.about_you, loaded.about_you);
        assert_eq!(saved.updated_at, loaded.updated_at);
    }

    // ---- strip_user_profile_lines unit tests ---------------------------

    #[test]
    fn strip_removes_lines_citing_only_profile_ids() {
        use std::collections::HashSet;
        let mut ids = HashSet::new();
        ids.insert("evt_profile_1".to_string());

        let input = "# Header\nFact A. <!-- src: evt_chat_1 -->\nFact B. <!-- src: evt_profile_1 -->\nFact C. <!-- src: evt_profile_1,evt_chat_1 -->\n";
        let (out, removed) = strip_user_profile_lines(input, &ids);
        assert_eq!(removed, 1, "only Fact B is profile-only");
        assert!(out.contains("Fact A."), "chat-only line preserved");
        assert!(!out.contains("Fact B."), "profile-only line removed");
        assert!(
            out.contains("Fact C."),
            "mixed-source line preserved (would destroy chat knowledge)"
        );
        assert!(out.ends_with('\n'), "trailing newline preserved");
    }

    #[test]
    fn strip_returns_zero_when_no_profile_ids_present() {
        use std::collections::HashSet;
        let ids = HashSet::new();
        let input = "Fact. <!-- src: evt_chat_1 -->\n";
        let (out, removed) = strip_user_profile_lines(input, &ids);
        assert_eq!(removed, 0);
        assert_eq!(out, input);
    }

    #[test]
    fn strip_preserves_lines_without_src_marker() {
        use std::collections::HashSet;
        let mut ids = HashSet::new();
        ids.insert("evt_profile_1".to_string());
        let input = "# Preferences\n\nProse paragraph.\n";
        let (out, removed) = strip_user_profile_lines(input, &ids);
        assert_eq!(removed, 0);
        assert_eq!(out, input);
    }

    // ---- clear_user_profile_dreams integration -------------------------

    /// Helper: write a UserProfileUpdate event to today's events JSONL so
    /// `collect_user_profile_event_ids` can find it.
    fn write_profile_event(root: &std::path::Path, event_id: &str) {
        let events_dir = root.join("events");
        std::fs::create_dir_all(&events_dir).unwrap();
        let today = chrono::Local::now().date_naive();
        let path = events_dir.join(format!("{}.jsonl", today.format("%Y-%m-%d")));
        let evt = crate::vault::events::ExperienceEvent::UserProfileUpdate {
            event_id: event_id.to_string(),
            ts: chrono::Local::now(),
            category: crate::vault::events::UserProfileCategory::About,
            text: "test".into(),
        };
        let line = serde_json::to_string(&evt).unwrap();
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(f, "{}", line).unwrap();
    }

    #[test]
    fn clear_user_profile_dreams_removes_profile_lines_from_wiki() {
        let (_tmp, vault) = fresh_vault();
        let root = vault.root.clone();

        // Seed an event so collect_user_profile_event_ids finds it.
        write_profile_event(&root, "evt_profile_xyz");

        // Seed the wiki with a mix of profile-derived and chat-derived facts.
        let wiki_user = root.join("wiki").join("user");
        std::fs::create_dir_all(&wiki_user).unwrap();
        let pref_path = wiki_user.join("preferences.md");
        let body = "# Preferences\nChat fact. <!-- src: evt_chat_aaa -->\nProfile fact. <!-- src: evt_profile_xyz -->\nMixed fact. <!-- src: evt_profile_xyz,evt_chat_aaa -->\n";
        std::fs::write(&pref_path, body).unwrap();

        let removed = vault.clear_user_profile_dreams().expect("clear dreams");
        assert_eq!(removed, 1, "only the profile-only line should be removed");

        let after = std::fs::read_to_string(&pref_path).unwrap();
        assert!(after.contains("Chat fact."), "chat-derived line preserved");
        assert!(
            !after.contains("Profile fact."),
            "profile-only line removed; got:\n{after}"
        );
        assert!(after.contains("Mixed fact."), "mixed-source line preserved");
    }

    #[test]
    fn clear_user_profile_dreams_returns_zero_when_no_profile_events() {
        let (_tmp, vault) = fresh_vault();
        // No UserProfileUpdate events on disk → nothing to strip.
        let removed = vault.clear_user_profile_dreams().expect("clear dreams");
        assert_eq!(removed, 0);
    }
}

#[cfg(test)]
use chrono::TimeZone;
