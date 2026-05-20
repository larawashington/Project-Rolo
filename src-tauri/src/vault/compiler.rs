//! Dreaming compiler — transforms a batch of raw experience events plus the
//! current writable wiki into a JSON list of facts via an LLM, validates the
//! response, and applies the accepted facts back to the wiki with provenance
//! markers.
//!
//! The pipeline is:
//!   build_prompt → provider.generate → parse → validate → apply
//!
//! Every stage is independently testable; the public `compile` method threads
//! them together and persists telemetry to the dreams ledger plus a per-run
//! artifact for offline replay (PRD §4, §7A, Testing-Strategy R6).
//!
//! The LLM never sees `personality/core-identity.md` or `mood-state.json` —
//! only files matching `WIKI_WRITABLE_PREFIXES` are exposed in the prompt and
//! accepted as fact targets. Defense in depth: the file allowlist is enforced
//! again in `validate` and a final content-string check rejects any fact
//! whose body smuggles forbidden paths.

use std::collections::BTreeMap;
use std::path::Path;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::chat::provider::{GenerationConfig, InferenceProvider, ProviderChatMessage};
use crate::vault::config::WIKI_WRITABLE_PREFIXES;
use crate::vault::dreams_log::DreamsLog;

/// What the LLM was asked to do with this fact relative to its target file.
/// `Add` appends a new line, `Update` and `Supersede` both reference an
/// existing line via `supersedes_line` — `Supersede` additionally comments the
/// old line out so the audit trail survives.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FactOperation {
    Add,
    Update,
    Supersede,
}

/// One LLM-proposed knowledge change. The fields below mirror the schema in
/// PRD §4 verbatim — any rename here is a wire-format break.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Fact {
    /// Wiki-relative path, e.g. `"user/preferences.md"`.
    pub file: String,
    /// Free-text section header the fact belongs under (informational; the
    /// applier doesn't actually move lines under headers in Phase 5).
    pub section: String,
    pub operation: FactOperation,
    pub content: String,
    /// 1-based line number this fact replaces or updates. `Add` ops leave it
    /// `None`. `Supersede` requires it; `Update` may use it.
    #[serde(default)]
    pub supersedes_line: Option<usize>,
    /// Event IDs that grounded this fact. Must be ≥ 1 entry and every entry
    /// must appear in the input batch (validated downstream).
    pub source_event_ids: Vec<String>,
}

/// The exact JSON shape we ask the LLM to emit (PRD §4 schema). `facts` is
/// non-defaulted on purpose — `{}` (no `facts` key) is treated as a parse
/// failure rather than silently coerced to an empty list. This keeps the
/// "the model emitted nothing" path distinct from "the model emitted prose".
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct CompileResponse {
    pub facts: Vec<Fact>,
}

/// One event prepared for the prompt: the verbatim JSONL line plus the
/// already-extracted `event_id` and `kind` for the human-readable header.
#[derive(Debug, Clone)]
pub struct CompiledEvent {
    pub event_id: String,
    /// Verbatim JSONL line as fed to the LLM. Newline-stripped.
    pub raw_json_line: String,
    /// Variant tag, e.g. `"chat"`, `"eat"` — drives the `[kind]` label.
    pub kind: String,
}

/// The set of events the compiler is offering to the model in one call. Held
/// in input order so the prompt's NEW OBSERVATIONS section is chronological.
#[derive(Debug, Clone)]
pub struct CompileBatch {
    pub events: Vec<CompiledEvent>,
}

/// Map the raw serde `type` string to the human-facing `[kind]` label used
/// in the compile prompt. Identity for most variants; collapses
/// `user_profile_update` to `user_profile` per PRD §6 spec (the wire format
/// keeps the verbose past-tense name; the prompt label reads as a topic).
pub fn prompt_kind_label(serde_type: &str) -> &str {
    match serde_type {
        "user_profile_update" => "user_profile",
        other => other,
    }
}

/// Strip a surrounding markdown code fence from an LLM response so the JSON
/// parser sees the bare object. Many Ollama-served models (Gemma especially)
/// wrap structured output in ```json ... ``` despite the prompt asking for
/// raw JSON. Returns the input verbatim when no fence is present.
///
/// Handles: ```json\n{...}\n```, ```\n{...}\n```, and either form without a
/// trailing fence (we strip whatever fence we find and trust serde to fail
/// on truly malformed payloads). Surrounding whitespace is also trimmed.
pub fn strip_json_code_fences(raw: &str) -> &str {
    let trimmed = raw.trim();
    let after_open = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(str::trim_start)
        .unwrap_or(trimmed);
    let after_close = after_open
        .strip_suffix("```")
        .map(str::trim_end)
        .unwrap_or(after_open);
    after_close
}

/// A snapshot of the writable wiki at compile time. The caller is responsible
/// for filtering to writable files only — passing in `core-identity.md` would
/// be a security bug, not a compiler responsibility.
pub struct WikiSnapshot {
    /// Wiki-relative path → file contents.
    pub files: BTreeMap<String, String>,
}

/// Reasons we drop a single fact during validation. The variants intentionally
/// match the dreams-log `rejected_reasons` keys (PRD §7A).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// `source_event_ids` cited an ID not present in the input batch.
    MissingSourceId(String),
    /// Target file lies outside `WIKI_WRITABLE_PREFIXES`.
    FileNotAllowed(String),
    /// Fact content tried to smuggle a forbidden path or filename.
    PathTraversalOrForbiddenContent,
}

/// The two batch-level outcomes of validation. `RejectedTooManyDrops` carries
/// the counts so the dreams-log entry can record the exact ratio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchStatus {
    Accepted,
    RejectedTooManyDrops { rejected: usize, proposed: usize },
}

/// Aggregate validation result. Even when `batch_status` is rejected, the
/// `accepted` and `rejected` lists are populated so the artifact can be
/// inspected after the fact — the caller decides not to apply.
#[derive(Debug, Clone)]
pub struct ValidationOutcome {
    pub accepted: Vec<Fact>,
    pub rejected: Vec<(Fact, RejectReason)>,
    pub batch_status: BatchStatus,
}

/// Summary of files written during `apply`. Returned to the caller so the
/// dreams-log entry can record `facts_accepted` etc. without a second pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppliedSummary {
    pub files_changed: Vec<String>,
    pub lines_added: usize,
    pub lines_superseded: usize,
}

/// End-to-end outcome of a single compile run. The status string mirrors the
/// `status` field of the dreams-log entry — kept as a `String` (rather than an
/// enum) so the dreaming orchestrator can stamp other lifecycle states like
/// `"skipped"` and `"cancelled"` onto the same shape later.
#[derive(Debug, Clone)]
pub struct CompileOutcome {
    pub run_id: String,
    pub status: String,
    pub facts_proposed: usize,
    pub facts_accepted: usize,
    pub facts_rejected: usize,
    pub latency_ms: u64,
    pub applied: AppliedSummary,
    pub reject_reason: Option<String>,
}

/// The compiler is stateless — keeping it as a struct lets tests stub future
/// fields (e.g. the prompt template version) without changing call sites.
pub struct Compiler;

impl Compiler {
    /// Construct a fresh compiler. Trivial — exists for symmetry with future
    /// configurable knobs.
    pub fn new() -> Self {
        Self
    }

    /// Render the full prompt sent to the LLM. The exact wording is locked
    /// against PRD §4; bumping it is a `prompt_template_version` bump.
    pub fn build_prompt(&self, batch: &CompileBatch, current_wiki: &WikiSnapshot) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("You are maintaining a knowledge base from observed user interactions.\n\n");

        out.push_str("CURRENT WIKI:\n");
        // BTreeMap iterates in sorted key order — gives a stable prompt across
        // runs which matters for the input-hash field in the dreams ledger.
        for (file, content) in &current_wiki.files {
            out.push_str("=== FILE: ");
            out.push_str(file);
            out.push_str(" ===\n");
            out.push_str(content);
            if !content.ends_with('\n') {
                out.push('\n');
            }
        }
        out.push('\n');

        out.push_str("NEW OBSERVATIONS:\n");
        for ev in &batch.events {
            out.push_str(&ev.event_id);
            out.push_str(" [");
            out.push_str(&ev.kind);
            out.push_str("] ");
            out.push_str(&ev.raw_json_line);
            out.push('\n');
        }
        out.push('\n');

        // Output instructions block — verbatim from PRD §4.
        out.push_str(
            "Extract NEW or UPDATED facts grounded in these observations. Output JSON only:\n\n",
        );
        out.push_str("{\n");
        out.push_str("  \"facts\": [\n");
        out.push_str("    {\n");
        out.push_str("      \"file\": \"user/preferences.md\",\n");
        out.push_str("      \"section\": \"Communication style\",\n");
        out.push_str("      \"operation\": \"add\" | \"update\" | \"supersede\",\n");
        out.push_str("      \"content\": \"User prefers short responses.\",\n");
        out.push_str("      \"supersedes_line\": null,\n");
        out.push_str("      \"source_event_ids\": [\"evt_abc123\", \"evt_def456\"]\n");
        out.push_str("    }\n");
        out.push_str("  ]\n");
        out.push_str("}\n\n");
        out.push_str("Rules:\n");
        out.push_str("- Every fact MUST cite >=1 source_event_id from the observations above.\n");
        out.push_str("- Do NOT infer facts not directly supported by an observation.\n");
        out.push_str(
            "- Do NOT modify personality/core-identity.md or personality/mood-state.json.\n",
        );
        out.push_str("- If an observation contradicts existing wiki, use \"supersede\" with supersedes_line.\n");
        out.push_str("- If no new facts: output {\"facts\": []}.\n");

        out
    }

    /// Apply the per-fact validators (PRD §4 steps 2–4) plus the batch-level
    /// drop-ratio gate (step 5). Pure function — the only side effect is
    /// `warn!` log lines naming each dropped fact.
    pub fn validate(&self, response: &CompileResponse, batch: &CompileBatch) -> ValidationOutcome {
        let proposed = response.facts.len();
        let mut accepted: Vec<Fact> = Vec::with_capacity(proposed);
        let mut rejected: Vec<(Fact, RejectReason)> = Vec::new();

        for fact in &response.facts {
            // Step 2: every cited event_id must appear in the batch.
            let mut citation_miss: Option<String> = None;
            for cited in &fact.source_event_ids {
                if !batch.events.iter().any(|e| &e.event_id == cited) {
                    citation_miss = Some(cited.clone());
                    break;
                }
            }
            if let Some(missing) = citation_miss {
                log::warn!(
                    "[Rolo dreaming] dropped fact rule=missing_source_id file={} missing_id={}",
                    fact.file,
                    missing
                );
                rejected.push((fact.clone(), RejectReason::MissingSourceId(missing)));
                continue;
            }

            // Step 3: file allowlist.
            if !file_is_writable(&fact.file) {
                log::warn!(
                    "[Rolo dreaming] dropped fact rule=file_not_allowed file={}",
                    fact.file
                );
                rejected.push((
                    fact.clone(),
                    RejectReason::FileNotAllowed(fact.file.clone()),
                ));
                continue;
            }

            // Step 4: content-string check. A literal `.contains()` is enough —
            // we are not parsing markdown, just refusing to write strings that
            // could be interpreted as targeting protected files / paths.
            let c = &fact.content;
            if c.contains("core-identity")
                || c.contains("mood-state.json")
                || c.contains("../")
                || c.contains("~/")
            {
                log::warn!(
                    "[Rolo dreaming] dropped fact rule=content_check file={}",
                    fact.file
                );
                rejected.push((fact.clone(), RejectReason::PathTraversalOrForbiddenContent));
                continue;
            }

            accepted.push(fact.clone());
        }

        // Step 5: batch-level drop ratio. Strictly greater than 25% triggers
        // a full-batch rejection (boundary value 25% is accepted) per PRD §4.
        let batch_status = if proposed == 0 {
            BatchStatus::Accepted
        } else {
            let ratio = rejected.len() as f64 / proposed as f64;
            if ratio > 0.25 {
                BatchStatus::RejectedTooManyDrops {
                    rejected: rejected.len(),
                    proposed,
                }
            } else {
                BatchStatus::Accepted
            }
        };

        ValidationOutcome {
            accepted,
            rejected,
            batch_status,
        }
    }

    /// Apply accepted facts to the wiki under `wiki_root`. Today's date is
    /// stamped into supersede markers; production callers use `apply` which
    /// reads `chrono::Local::now()` — tests inject a fixed date via
    /// `apply_with_date` to keep golden-file outputs deterministic.
    pub fn apply(&self, accepted: &[Fact], wiki_root: &Path) -> std::io::Result<AppliedSummary> {
        self.apply_with_date(accepted, wiki_root, chrono::Local::now().date_naive())
    }

    /// Test-injectable apply. Same behavior as `apply` but the supersede
    /// marker date is provided by the caller.
    pub fn apply_with_date(
        &self,
        accepted: &[Fact],
        wiki_root: &Path,
        today: NaiveDate,
    ) -> std::io::Result<AppliedSummary> {
        // Group by target file so we read each file once even if multiple
        // facts touch it. Insertion order across files is preserved by
        // BTreeMap iterator, which matters for the deterministic
        // files_changed list returned to the caller.
        let mut by_file: BTreeMap<String, Vec<&Fact>> = BTreeMap::new();
        for fact in accepted {
            by_file.entry(fact.file.clone()).or_default().push(fact);
        }

        let mut summary = AppliedSummary::default();
        let date_str = today.format("%Y-%m-%d").to_string();

        for (file, facts) in by_file {
            let path = wiki_root.join(&file);
            let existing = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
                Err(e) => return Err(e),
            };

            // Keep the trailing-newline state explicit so we can preserve it
            // on the way out. Files without trailing newlines stay that way
            // unless we add a line.
            let had_trailing_newline = existing.ends_with('\n');
            let mut lines: Vec<String> = if existing.is_empty() {
                Vec::new()
            } else {
                existing
                    .strip_suffix('\n')
                    .unwrap_or(&existing)
                    .split('\n')
                    .map(|s| s.to_string())
                    .collect()
            };

            for fact in &facts {
                // Provenance suffix shared by Add/Update/Supersede append paths.
                let src_marker = format!(" <!-- src: {} -->", fact.source_event_ids.join(","));
                let new_line = format!("{}{}", fact.content, src_marker);

                if fact.operation == FactOperation::Supersede {
                    if let Some(idx) = fact.supersedes_line {
                        // 1-based → 0-based.
                        if idx >= 1 && idx <= lines.len() {
                            let old = std::mem::take(&mut lines[idx - 1]);
                            let cited = fact.source_event_ids.join(",");
                            lines[idx - 1] = format!(
                                "<!-- superseded_by: {} on {} --><!-- {} -->",
                                cited, date_str, old
                            );
                            summary.lines_superseded += 1;
                        }
                    }
                }

                lines.push(new_line);
                summary.lines_added += 1;
            }

            // Reassemble. Always end with a newline if we added any line —
            // markdown convention plus it makes future `read+append` paths
            // safe.
            let mut content = lines.join("\n");
            if had_trailing_newline || !lines.is_empty() {
                content.push('\n');
            }

            crate::vault::atomic::atomic_write(&path, content.as_bytes())?;
            summary.files_changed.push(file);
        }

        Ok(summary)
    }

    /// One full compile run: build prompt, call the provider, parse, validate,
    /// apply (when accepted), persist a dreams_log entry plus a sibling
    /// artifact under `artifacts_dir`, and return a structured outcome.
    ///
    /// Failure modes are captured as outcome fields rather than `Result`s — a
    /// failed compile is a normal, expected event in Rolo's life and the
    /// caller (the dreaming orchestrator) shouldn't have to thread `Result`s
    /// through tokio::select! arms. The function only short-circuits when an
    /// I/O error makes telemetry impossible, in which case we still return a
    /// best-effort outcome.
    pub async fn compile(
        &self,
        batch: &CompileBatch,
        wiki_root: &Path,
        provider: &dyn InferenceProvider,
        dreams: &DreamsLog,
        artifacts_dir: &Path,
    ) -> CompileOutcome {
        let started_at = chrono::Local::now();
        let started_instant = std::time::Instant::now();
        let run_id = make_run_id(started_at);

        // 1. Build the writable-wiki snapshot for the prompt.
        let snapshot = read_wiki_snapshot(wiki_root);

        // 2. Build prompt and ship it to the provider.
        let prompt = self.build_prompt(batch, &snapshot);
        log::trace!(
            "[Rolo dreaming] run_id={} rendered_prompt_bytes={}",
            run_id,
            prompt.len()
        );

        let messages = vec![ProviderChatMessage {
            role: "user".to_string(),
            content: prompt.clone(),
        }];
        let config = GenerationConfig {
            temperature: 0.3,
            ..GenerationConfig::default()
        };

        // The streaming channel is created and the receiver dropped — the
        // compiler doesn't surface partial tokens. MockProvider's
        // Cancelled-on-drop check only fires when the channel buffer fills,
        // so we use a generous capacity here. Real Ollama streaming tokens
        // out fast enough that this never matters in production.
        let (tx, _rx) = tokio::sync::mpsc::channel::<String>(1024);

        let raw_response = match provider.generate(messages, config, tx).await {
            Ok(s) => s,
            Err(e) => {
                let latency_ms = started_instant.elapsed().as_millis() as u64;
                log::warn!("[Rolo dreaming] run_id={} llm_error={}", run_id, e);
                return finalize_failed(
                    &run_id,
                    "llm_error",
                    &prompt,
                    "",
                    None,
                    batch,
                    provider,
                    dreams,
                    artifacts_dir,
                    started_at,
                    latency_ms,
                );
            }
        };

        // 3. Parse the response. Strip surrounding markdown fences first —
        // Gemma loves to wrap its JSON in ```json ... ``` even when the prompt
        // asks for raw output, and serde refuses to parse the backtick prefix.
        let parsed: CompileResponse =
            match serde_json::from_str(strip_json_code_fences(&raw_response)) {
                Ok(p) => p,
                Err(e) => {
                    let latency_ms = started_instant.elapsed().as_millis() as u64;
                    log::warn!(
                        "[Rolo dreaming] run_id={} parse_failure={} raw={}",
                        run_id,
                        e,
                        raw_response
                    );
                    return finalize_failed(
                        &run_id,
                        "parse_failure",
                        &prompt,
                        &raw_response,
                        None,
                        batch,
                        provider,
                        dreams,
                        artifacts_dir,
                        started_at,
                        latency_ms,
                    );
                }
            };

        // 4. Validate.
        let validation = self.validate(&parsed, batch);

        // 5. Apply iff the batch was accepted by step 5 of the validator.
        // When rejected, we still log and persist artifact for inspection but
        // never touch the wiki.
        let (status, reject_reason, applied) = match &validation.batch_status {
            BatchStatus::Accepted => match self.apply(&validation.accepted, wiki_root) {
                Ok(s) => ("success".to_string(), None, s),
                Err(e) => {
                    log::error!("[Rolo dreaming] run_id={} apply_io_error={}", run_id, e);
                    (
                        "failed".to_string(),
                        Some("apply_io_error".to_string()),
                        AppliedSummary::default(),
                    )
                }
            },
            BatchStatus::RejectedTooManyDrops { .. } => (
                "rejected".to_string(),
                Some("high_rejection_rate".to_string()),
                AppliedSummary::default(),
            ),
        };

        let latency_ms = started_instant.elapsed().as_millis() as u64;
        let ended_at = chrono::Local::now();

        // 6. Build dreams_log entry per PRD §7A.
        let entry = build_dreams_entry(
            &run_id,
            &status,
            reject_reason.as_deref(),
            provider,
            batch,
            &prompt,
            &raw_response,
            &validation,
            started_at,
            ended_at,
            latency_ms,
        );
        if let Err(e) = dreams.append(entry.clone()) {
            log::error!(
                "[Rolo dreaming] run_id={} dreams_log_append_failed={}",
                run_id,
                e
            );
        }

        // 7. Persist the replay artifact (best-effort).
        if let Err(e) = persist_artifact(
            artifacts_dir,
            &run_id,
            &prompt,
            &raw_response,
            &parsed,
            &validation,
        ) {
            log::warn!(
                "[Rolo dreaming] run_id={} artifact_persist_failed={}",
                run_id,
                e
            );
        }

        log::info!(
            "[Rolo dreaming] run_id={} status={} facts_accepted={} facts_rejected={} latency_ms={}",
            run_id,
            status,
            validation.accepted.len(),
            validation.rejected.len(),
            latency_ms
        );

        CompileOutcome {
            run_id,
            status,
            facts_proposed: validation.accepted.len() + validation.rejected.len(),
            facts_accepted: validation.accepted.len(),
            facts_rejected: validation.rejected.len(),
            latency_ms,
            applied,
            reject_reason,
        }
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a `run_id` of the form `drm_YYYY-MM-DDTHH:MM:SS_xxxx` where the
/// trailing four chars come from a fresh UUID v7. Colon separators in the ISO
/// timestamp are kept on purpose — they match the PRD §7A example and make
/// the ID grep-friendly when scanning dreams.jsonl by timestamp.
fn make_run_id(now: chrono::DateTime<chrono::Local>) -> String {
    let ts = now.format("%Y-%m-%dT%H:%M:%S");
    let suffix: String = uuid::Uuid::now_v7().to_string().chars().take(4).collect();
    format!("drm_{}_{}", ts, suffix)
}

/// Walk `wiki_root` and return all writable files keyed by wiki-relative path.
/// Non-existent root or non-text files are returned as an empty snapshot —
/// the compiler tolerates a brand-new vault.
fn read_wiki_snapshot(wiki_root: &Path) -> WikiSnapshot {
    let mut files: BTreeMap<String, String> = BTreeMap::new();
    if !wiki_root.exists() {
        return WikiSnapshot { files };
    }
    for entry in walkdir::WalkDir::new(wiki_root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        let rel = match entry.path().strip_prefix(wiki_root) {
            Ok(r) => r,
            Err(_) => continue,
        };
        // Normalize to forward-slash regardless of platform — wiki paths are
        // first-class identifiers in the prompt and must match the allowlist
        // entries which use `/`.
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if !file_is_writable(&rel_str) {
            continue;
        }
        let contents = match std::fs::read_to_string(entry.path()) {
            Ok(s) => s,
            Err(e) => {
                log::warn!(
                    "[Rolo dreaming] could not read wiki file {}: {}",
                    rel_str,
                    e
                );
                continue;
            }
        };
        files.insert(rel_str, contents);
    }
    WikiSnapshot { files }
}

/// Hash the concatenated raw_json_lines of a batch — feeds the dreams-log
/// `input_hash` field. Stable regardless of map ordering since JSONL lines
/// are concatenated as opaque bytes.
fn hash_batch_input(batch: &CompileBatch) -> String {
    let mut hasher = Sha256::new();
    for ev in &batch.events {
        hasher.update(ev.raw_json_line.as_bytes());
        hasher.update(b"\n");
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(7 + 64);
    hex.push_str("sha256:");
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Aggregate per-fact rejection reasons into the compact map shape the PRD
/// specifies for `rejected_reasons`.
fn rejected_reasons_map(rejected: &[(Fact, RejectReason)]) -> BTreeMap<String, usize> {
    let mut map: BTreeMap<String, usize> = BTreeMap::new();
    for (_fact, reason) in rejected {
        let key = match reason {
            RejectReason::MissingSourceId(_) => "missing_source_id",
            RejectReason::FileNotAllowed(_) => "file_not_allowed",
            RejectReason::PathTraversalOrForbiddenContent => "content_check",
        };
        *map.entry(key.to_string()).or_insert(0) += 1;
    }
    map
}

/// Build the dreams-log entry. Field names mirror PRD §7A exactly; consumers
/// (the Dream Log UI in F2) parse this shape directly.
#[allow(clippy::too_many_arguments)]
fn build_dreams_entry(
    run_id: &str,
    status: &str,
    reject_reason: Option<&str>,
    provider: &dyn InferenceProvider,
    batch: &CompileBatch,
    prompt: &str,
    raw_response: &str,
    validation: &ValidationOutcome,
    started_at: chrono::DateTime<chrono::Local>,
    ended_at: chrono::DateTime<chrono::Local>,
    latency_ms: u64,
) -> serde_json::Value {
    let input_event_ids: Vec<String> = batch.events.iter().map(|e| e.event_id.clone()).collect();
    let mut entry = serde_json::json!({
        "run_id": run_id,
        "status": status,
        "model": provider.provider_name(),
        "model_digest": provider.model_name(),
        "temperature": 0.3,
        "input_event_ids": input_event_ids,
        "input_hash": hash_batch_input(batch),
        "prompt_template_version": "compile-v1",
        "tokens_in": prompt.len(),
        "tokens_out": raw_response.len(),
        "latency_ms": latency_ms,
        "facts_proposed": validation.accepted.len() + validation.rejected.len(),
        "facts_accepted": validation.accepted.len(),
        "facts_rejected": validation.rejected.len(),
        "rejected_reasons": rejected_reasons_map(&validation.rejected),
        "embedding_rebuild_ms": 0,
        "started_at": started_at.to_rfc3339(),
        "ended_at": ended_at.to_rfc3339(),
    });
    if let Some(reason) = reject_reason {
        if let Some(map) = entry.as_object_mut() {
            map.insert(
                "reject_reason".to_string(),
                serde_json::Value::String(reason.to_string()),
            );
        }
    }
    entry
}

/// Persist the per-run replay artifact at `<artifacts_dir>/<run_id>.json` and
/// FIFO-evict to keep the directory bounded at 100 most-recent runs.
fn persist_artifact(
    artifacts_dir: &Path,
    run_id: &str,
    prompt: &str,
    raw_response: &str,
    parsed: &CompileResponse,
    validation: &ValidationOutcome,
) -> std::io::Result<()> {
    std::fs::create_dir_all(artifacts_dir)?;
    let payload = serde_json::json!({
        "run_id": run_id,
        "rendered_prompt": prompt,
        "raw_response": raw_response,
        "parsed_response": parsed,
        "accepted_facts": validation.accepted,
        "rejected_facts": validation
            .rejected
            .iter()
            .map(|(fact, reason)| {
                serde_json::json!({
                    "fact": fact,
                    "reason": format!("{:?}", reason),
                })
            })
            .collect::<Vec<_>>(),
    });
    let path = artifacts_dir.join(format!("{}.json", run_id));
    let bytes = serde_json::to_vec_pretty(&payload).map_err(std::io::Error::other)?;
    crate::vault::atomic::atomic_write(&path, &bytes)?;
    enforce_artifact_cap(artifacts_dir, 100);
    Ok(())
}

/// FIFO-evict artifacts older than the most recent `max` files. Best-effort —
/// failures are logged but never propagated.
fn enforce_artifact_cap(dir: &Path, max: usize) {
    let mut entries: Vec<(std::path::PathBuf, std::time::SystemTime)> = match std::fs::read_dir(dir)
    {
        Ok(rd) => rd
            .filter_map(Result::ok)
            .filter_map(|e| {
                let p = e.path();
                if p.extension()
                    .and_then(|s| s.to_str())
                    .is_none_or(|s| s != "json")
                {
                    return None;
                }
                let mt = e.metadata().ok()?.modified().ok()?;
                Some((p, mt))
            })
            .collect(),
        Err(_) => return,
    };
    if entries.len() <= max {
        return;
    }
    entries.sort_by_key(|e| std::cmp::Reverse(e.1));
    for (p, _) in entries.into_iter().skip(max) {
        if let Err(e) = std::fs::remove_file(&p) {
            log::warn!(
                "[Rolo dreaming] artifact eviction could not remove {:?}: {}",
                p,
                e
            );
        }
    }
}

/// Failure path: write a dreams_log entry, persist a partial artifact, and
/// return a `CompileOutcome` with status="failed" and the given reason. Used
/// by the LLM-error and parse-failure branches.
#[allow(clippy::too_many_arguments)]
fn finalize_failed(
    run_id: &str,
    reason: &str,
    prompt: &str,
    raw_response: &str,
    parsed: Option<&CompileResponse>,
    batch: &CompileBatch,
    provider: &dyn InferenceProvider,
    dreams: &DreamsLog,
    artifacts_dir: &Path,
    started_at: chrono::DateTime<chrono::Local>,
    latency_ms: u64,
) -> CompileOutcome {
    let ended_at = chrono::Local::now();
    let empty_validation = ValidationOutcome {
        accepted: Vec::new(),
        rejected: Vec::new(),
        batch_status: BatchStatus::Accepted,
    };
    let entry = build_dreams_entry(
        run_id,
        "failed",
        Some(reason),
        provider,
        batch,
        prompt,
        raw_response,
        &empty_validation,
        started_at,
        ended_at,
        latency_ms,
    );
    if let Err(e) = dreams.append(entry) {
        log::error!(
            "[Rolo dreaming] run_id={} dreams_log_append_failed={}",
            run_id,
            e
        );
    }
    // Persist artifact even on failure so the replay binary can read whatever
    // raw bytes the LLM did produce. `parsed` is None on parse failure; use a
    // synthetic empty response so the artifact JSON stays well-formed.
    let synthetic = CompileResponse { facts: Vec::new() };
    let parsed_for_artifact = parsed.unwrap_or(&synthetic);
    if let Err(e) = persist_artifact(
        artifacts_dir,
        run_id,
        prompt,
        raw_response,
        parsed_for_artifact,
        &empty_validation,
    ) {
        log::warn!(
            "[Rolo dreaming] run_id={} artifact_persist_failed={}",
            run_id,
            e
        );
    }
    CompileOutcome {
        run_id: run_id.to_string(),
        status: "failed".to_string(),
        facts_proposed: 0,
        facts_accepted: 0,
        facts_rejected: 0,
        latency_ms,
        applied: AppliedSummary::default(),
        reject_reason: Some(reason.to_string()),
    }
}

/// Returns true iff `path` (wiki-relative) is targetable by the compiler. A
/// prefix entry ending in `/` matches any descendant; an entry without a
/// trailing slash is a literal-equals match. See `WIKI_WRITABLE_PREFIXES`.
fn file_is_writable(path: &str) -> bool {
    for entry in WIKI_WRITABLE_PREFIXES {
        if entry.ends_with('/') {
            if path.starts_with(entry) {
                return true;
            }
        } else if path == *entry {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    fn batch_with_two_events() -> CompileBatch {
        CompileBatch {
            events: vec![
                CompiledEvent {
                    event_id: "evt_aaa".into(),
                    raw_json_line: r#"{"type":"chat","user":"hi","rolo":"hello"}"#.into(),
                    kind: "chat".into(),
                },
                CompiledEvent {
                    event_id: "evt_bbb".into(),
                    raw_json_line: r#"{"type":"eat","file":"foo.py"}"#.into(),
                    kind: "eat".into(),
                },
            ],
        }
    }

    // ---- prompt_kind_label ----------------------------------------------

    #[test]
    fn prompt_kind_label_collapses_user_profile_update() {
        assert_eq!(prompt_kind_label("user_profile_update"), "user_profile");
    }

    #[test]
    fn prompt_kind_label_is_identity_for_other_kinds() {
        assert_eq!(prompt_kind_label("chat"), "chat");
        assert_eq!(prompt_kind_label("eat"), "eat");
        assert_eq!(prompt_kind_label("unknown"), "unknown");
    }

    // ---- strip_json_code_fences ------------------------------------------

    #[test]
    fn strip_fences_leaves_clean_json_unchanged() {
        let raw = r#"{"facts":[]}"#;
        assert_eq!(strip_json_code_fences(raw), r#"{"facts":[]}"#);
    }

    #[test]
    fn strip_fences_removes_json_fence() {
        let raw = "```json\n{\"facts\":[]}\n```";
        assert_eq!(strip_json_code_fences(raw), r#"{"facts":[]}"#);
    }

    #[test]
    fn strip_fences_removes_bare_fence() {
        let raw = "```\n{\"facts\":[]}\n```";
        assert_eq!(strip_json_code_fences(raw), r#"{"facts":[]}"#);
    }

    #[test]
    fn strip_fences_tolerates_missing_close() {
        let raw = "```json\n{\"facts\":[]}";
        assert_eq!(strip_json_code_fences(raw), r#"{"facts":[]}"#);
    }

    #[test]
    fn strip_fences_trims_surrounding_whitespace() {
        let raw = "   \n```json\n{\"facts\":[]}\n```\n   ";
        assert_eq!(strip_json_code_fences(raw), r#"{"facts":[]}"#);
    }

    #[test]
    fn strip_fences_then_parse_round_trips_fenced_response() {
        let raw = "```json\n{\"facts\":[]}\n```";
        let parsed: CompileResponse =
            serde_json::from_str(strip_json_code_fences(raw)).expect("fenced JSON should parse");
        assert!(parsed.facts.is_empty());
    }

    // ---- C1 tests --------------------------------------------------------

    #[test]
    fn prompt_includes_event_ids() {
        let compiler = Compiler::new();
        let batch = batch_with_two_events();
        let snapshot = WikiSnapshot {
            files: BTreeMap::new(),
        };
        let prompt = compiler.build_prompt(&batch, &snapshot);
        assert!(prompt.contains("evt_aaa"), "prompt must list evt_aaa");
        assert!(prompt.contains("evt_bbb"), "prompt must list evt_bbb");
        assert!(prompt.contains("[chat]"));
        assert!(prompt.contains("[eat]"));
    }

    #[test]
    fn prompt_lists_writable_files_only() {
        // Caller is responsible for the filter — the compiler trusts the
        // snapshot. Verify the writable file content shows up.
        let compiler = Compiler::new();
        let mut files = BTreeMap::new();
        files.insert(
            "user/identity.md".to_string(),
            "I am a software engineer.".to_string(),
        );
        let snapshot = WikiSnapshot { files };
        let prompt = compiler.build_prompt(&CompileBatch { events: Vec::new() }, &snapshot);
        assert!(prompt.contains("=== FILE: user/identity.md ==="));
        assert!(prompt.contains("I am a software engineer."));
        // core-identity does appear in the Rules block by name (the LLM is
        // told not to modify it). What must NOT appear is a CURRENT WIKI
        // entry for it, since the snapshot here only has writable files.
        assert!(
            !prompt.contains("=== FILE: personality/core-identity.md ==="),
            "core-identity must never be exposed as a wiki file in the prompt"
        );
    }

    #[test]
    fn parse_valid_response_round_trips() {
        let json = r#"{
            "facts": [
                {
                    "file": "user/preferences.md",
                    "section": "Style",
                    "operation": "add",
                    "content": "User prefers short responses.",
                    "supersedes_line": null,
                    "source_event_ids": ["evt_aaa"]
                }
            ]
        }"#;
        let parsed: CompileResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.facts.len(), 1);
        assert_eq!(parsed.facts[0].operation, FactOperation::Add);
        assert_eq!(parsed.facts[0].source_event_ids, vec!["evt_aaa"]);
        assert_eq!(parsed.facts[0].supersedes_line, None);

        // Round-trip: reserialize and reparse to make sure we kept the shape.
        let s = serde_json::to_string(&parsed).unwrap();
        let again: CompileResponse = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed, again);
    }

    #[test]
    fn parse_response_with_extra_field_succeeds_via_serde_default() {
        // Unknown top-level fields are tolerated (no `deny_unknown_fields`).
        let json = r#"{
            "facts": [],
            "model_extra_chatter": "I hope this is helpful!"
        }"#;
        let parsed: CompileResponse =
            serde_json::from_str(json).expect("extra field should be ignored");
        assert!(parsed.facts.is_empty());
    }

    #[test]
    fn parse_response_missing_facts_array_returns_err() {
        // `{}` lacks the required `facts` field → parse error.
        let json = "{}";
        let result: Result<CompileResponse, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "missing facts array must fail parse, got: {:?}",
            result
        );
    }

    // ---- C2 tests --------------------------------------------------------

    fn make_fact(file: &str, content: &str, ids: &[&str]) -> Fact {
        Fact {
            file: file.into(),
            section: "Test".into(),
            operation: FactOperation::Add,
            content: content.into(),
            supersedes_line: None,
            source_event_ids: ids.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn rejects_fact_with_unknown_event_id() {
        let compiler = Compiler::new();
        let batch = batch_with_two_events();
        // Mix one good citation in so this test exercises the per-fact
        // citation rule without crossing the 25% batch threshold.
        let response = CompileResponse {
            facts: vec![
                make_fact("user/preferences.md", "User likes tea.", &["evt_aaa"]),
                make_fact("user/preferences.md", "User likes coffee.", &["evt_aaa"]),
                make_fact("user/preferences.md", "User likes water.", &["evt_aaa"]),
                make_fact("user/preferences.md", "User likes juice.", &["evt_ghost"]),
            ],
        };
        let outcome = compiler.validate(&response, &batch);
        // 1/4 rejected = 25% (boundary, accepted).
        assert_eq!(outcome.accepted.len(), 3);
        assert_eq!(outcome.rejected.len(), 1);
        match &outcome.rejected[0].1 {
            RejectReason::MissingSourceId(id) => assert_eq!(id, "evt_ghost"),
            other => panic!("expected MissingSourceId, got {:?}", other),
        }
    }

    #[test]
    fn rejects_fact_targeting_core_identity() {
        let compiler = Compiler::new();
        let batch = batch_with_two_events();
        let response = CompileResponse {
            facts: vec![make_fact(
                "personality/core-identity.md",
                "Rolo is now evil.",
                &["evt_aaa"],
            )],
        };
        let outcome = compiler.validate(&response, &batch);
        assert_eq!(outcome.accepted.len(), 0);
        assert_eq!(outcome.rejected.len(), 1);
        match &outcome.rejected[0].1 {
            RejectReason::FileNotAllowed(f) => {
                assert_eq!(f, "personality/core-identity.md")
            }
            other => panic!("expected FileNotAllowed, got {:?}", other),
        }
    }

    #[test]
    fn rejects_fact_with_path_traversal() {
        let compiler = Compiler::new();
        let batch = batch_with_two_events();
        let response = CompileResponse {
            facts: vec![make_fact(
                "user/preferences.md",
                "User reads ../etc/passwd often.",
                &["evt_aaa"],
            )],
        };
        let outcome = compiler.validate(&response, &batch);
        assert_eq!(outcome.accepted.len(), 0);
        assert_eq!(outcome.rejected.len(), 1);
        assert!(matches!(
            outcome.rejected[0].1,
            RejectReason::PathTraversalOrForbiddenContent
        ));
    }

    #[test]
    fn accepts_fact_targeting_learned_behaviors() {
        let compiler = Compiler::new();
        let batch = batch_with_two_events();
        let response = CompileResponse {
            facts: vec![make_fact(
                "personality/learned-behaviors.md",
                "Rolo has learned to greet by name.",
                &["evt_aaa"],
            )],
        };
        let outcome = compiler.validate(&response, &batch);
        assert_eq!(outcome.accepted.len(), 1);
        assert_eq!(outcome.rejected.len(), 0);
        assert_eq!(outcome.batch_status, BatchStatus::Accepted);
    }

    #[test]
    fn rejects_batch_when_drop_ratio_above_25_percent() {
        let compiler = Compiler::new();
        let batch = batch_with_two_events();
        // 10 facts, 4 with bad event_ids → 40% rejection.
        let mut facts = Vec::new();
        for _ in 0..6 {
            facts.push(make_fact("user/preferences.md", "Good fact.", &["evt_aaa"]));
        }
        for _ in 0..4 {
            facts.push(make_fact(
                "user/preferences.md",
                "Bad fact.",
                &["evt_ghost"],
            ));
        }
        let response = CompileResponse { facts };
        let outcome = compiler.validate(&response, &batch);
        match outcome.batch_status {
            BatchStatus::RejectedTooManyDrops { rejected, proposed } => {
                assert_eq!(rejected, 4);
                assert_eq!(proposed, 10);
            }
            other => panic!("expected RejectedTooManyDrops, got {:?}", other),
        }
        // accepted list is still populated for artifact inspection — the
        // caller decides not to apply based on batch_status.
        assert_eq!(outcome.accepted.len(), 6);
    }

    #[test]
    fn accepts_batch_when_drop_ratio_at_25_percent() {
        // Boundary: exactly 25% rejected → still Accepted (strict > 0.25).
        let compiler = Compiler::new();
        let batch = batch_with_two_events();
        let mut facts = Vec::new();
        for _ in 0..3 {
            facts.push(make_fact("user/preferences.md", "Good.", &["evt_aaa"]));
        }
        // 1 of 4 is rejected = 25% exactly.
        facts.push(make_fact("user/preferences.md", "Bad.", &["evt_ghost"]));
        let response = CompileResponse { facts };
        let outcome = compiler.validate(&response, &batch);
        assert_eq!(outcome.batch_status, BatchStatus::Accepted);
        assert_eq!(outcome.accepted.len(), 3);
        assert_eq!(outcome.rejected.len(), 1);
    }

    // ---- C3 tests --------------------------------------------------------

    /// Walk every file under `expected_dir` and assert byte-equal content
    /// exists at the same relative path under `actual_dir`. Uses
    /// `pretty_assertions::assert_eq` for legible diffs.
    fn assert_dir_matches(expected_dir: &Path, actual_dir: &Path) {
        for entry in walkdir::WalkDir::new(expected_dir)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
        {
            let rel = entry.path().strip_prefix(expected_dir).unwrap();
            let expected = std::fs::read_to_string(entry.path())
                .unwrap_or_else(|e| panic!("read expected {:?}: {}", entry.path(), e));
            let actual_path = actual_dir.join(rel);
            let actual = std::fs::read_to_string(&actual_path)
                .unwrap_or_else(|e| panic!("read actual {:?}: {}", actual_path, e));
            assert_eq!(actual, expected, "diff in {:?}", rel);
        }
    }

    /// Recursively copy `src` into `dst` (must already exist).
    fn copy_dir(src: &Path, dst: &Path) {
        for entry in walkdir::WalkDir::new(src)
            .into_iter()
            .filter_map(Result::ok)
        {
            let rel = entry.path().strip_prefix(src).unwrap();
            let target = dst.join(rel);
            if entry.file_type().is_dir() {
                std::fs::create_dir_all(&target).unwrap();
            } else if entry.file_type().is_file() {
                if let Some(p) = target.parent() {
                    std::fs::create_dir_all(p).unwrap();
                }
                std::fs::copy(entry.path(), &target).unwrap();
            }
        }
    }

    fn fixtures_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("dreaming")
    }

    #[test]
    fn apply_writes_simple_add_with_provenance() {
        let case = fixtures_dir().join("case_01_simple_chat");
        let tmp = TempDir::new().unwrap();
        copy_dir(&case.join("input_wiki"), tmp.path());

        let compiler = Compiler::new();
        let accepted = vec![Fact {
            file: "user/preferences.md".into(),
            section: "Communication".into(),
            operation: FactOperation::Add,
            content: "User prefers short responses.".into(),
            supersedes_line: None,
            source_event_ids: vec!["evt_FIXTURE_1".into()],
        }];

        let date = NaiveDate::from_ymd_opt(2026, 5, 4).unwrap();
        let summary = compiler
            .apply_with_date(&accepted, tmp.path(), date)
            .expect("apply must succeed");
        assert_eq!(
            summary.files_changed,
            vec!["user/preferences.md".to_string()]
        );
        assert_eq!(summary.lines_added, 1);
        assert_eq!(summary.lines_superseded, 0);

        assert_dir_matches(&case.join("expected_wiki"), tmp.path());
    }

    #[test]
    fn apply_supersedes_existing_line() {
        let case = fixtures_dir().join("case_02_supersede");
        let tmp = TempDir::new().unwrap();
        copy_dir(&case.join("input_wiki"), tmp.path());

        let compiler = Compiler::new();
        let accepted = vec![Fact {
            file: "user/identity.md".into(),
            section: "Schedule".into(),
            operation: FactOperation::Supersede,
            content: "User works afternoons.".into(),
            supersedes_line: Some(2),
            source_event_ids: vec!["evt_FIXTURE_2".into()],
        }];

        let date = NaiveDate::from_ymd_opt(2026, 5, 4).unwrap();
        let summary = compiler
            .apply_with_date(&accepted, tmp.path(), date)
            .expect("apply must succeed");
        assert_eq!(summary.lines_added, 1);
        assert_eq!(summary.lines_superseded, 1);

        assert_dir_matches(&case.join("expected_wiki"), tmp.path());
    }

    // ---- C4 tests --------------------------------------------------------

    use crate::chat::mock_provider::MockProvider;
    use crate::vault::dreams_log::DreamsLog;

    /// Layout helper — every C4 test wants a wiki dir, an artifacts dir, and
    /// a fresh DreamsLog rooted somewhere stable.
    fn setup_compile_env() -> (TempDir, std::path::PathBuf, std::path::PathBuf, DreamsLog) {
        let tmp = TempDir::new().unwrap();
        let wiki = tmp.path().join("wiki");
        std::fs::create_dir_all(wiki.join("user")).unwrap();
        // Seed a writable file so the snapshot is non-empty (mirrors the
        // bootstrapped wiki layout).
        std::fs::write(wiki.join("user").join("preferences.md"), "# Preferences\n").unwrap();
        let artifacts = tmp.path().join("dreams_artifacts");
        let dreams = DreamsLog::new(tmp.path());
        (tmp, wiki, artifacts, dreams)
    }

    #[tokio::test]
    async fn compile_writes_dreams_log_success_on_clean_response() {
        let (_tmp, wiki, artifacts, dreams) = setup_compile_env();
        let batch = batch_with_two_events();

        // Single accepted fact citing evt_aaa from the batch.
        let response = r#"{"facts":[{"file":"user/preferences.md","section":"Style","operation":"add","content":"User prefers short responses.","supersedes_line":null,"source_event_ids":["evt_aaa"]}]}"#;
        let provider = MockProvider {
            response: response.to_string(),
            delay_ms: 0,
            should_fail: false,
        };

        let compiler = Compiler::new();
        let outcome = compiler
            .compile(&batch, &wiki, &provider, &dreams, &artifacts)
            .await;

        assert_eq!(outcome.status, "success");
        assert_eq!(outcome.facts_accepted, 1);
        assert_eq!(outcome.facts_rejected, 0);
        assert_eq!(outcome.applied.lines_added, 1);
        assert!(outcome.run_id.starts_with("drm_"));

        // Dreams log has one entry with status=success.
        let recent = dreams.read_recent(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0]["status"], "success");
        assert_eq!(recent[0]["facts_accepted"], 1);

        // Artifact file exists.
        let artifact_path = artifacts.join(format!("{}.json", outcome.run_id));
        assert!(artifact_path.exists(), "artifact must be persisted");

        // Wiki actually got the new line with provenance.
        let wiki_text = std::fs::read_to_string(wiki.join("user").join("preferences.md")).unwrap();
        assert!(wiki_text.contains("User prefers short responses."));
        assert!(wiki_text.contains("<!-- src: evt_aaa -->"));
    }

    #[tokio::test]
    async fn compile_writes_dreams_log_failure_on_parse_error() {
        let (_tmp, wiki, artifacts, dreams) = setup_compile_env();
        let batch = batch_with_two_events();

        let provider = MockProvider {
            response: "I'm not sure what to do here.".to_string(),
            delay_ms: 0,
            should_fail: false,
        };

        let compiler = Compiler::new();
        let outcome = compiler
            .compile(&batch, &wiki, &provider, &dreams, &artifacts)
            .await;

        assert_eq!(outcome.status, "failed");
        assert_eq!(outcome.reject_reason.as_deref(), Some("parse_failure"));
        assert_eq!(outcome.facts_accepted, 0);

        // Wiki untouched (the seeded file still has only the header).
        let wiki_text = std::fs::read_to_string(wiki.join("user").join("preferences.md")).unwrap();
        assert_eq!(wiki_text, "# Preferences\n");

        let recent = dreams.read_recent(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0]["status"], "failed");
        assert_eq!(recent[0]["reject_reason"], "parse_failure");
    }

    #[tokio::test]
    async fn compile_writes_dreams_log_rejection_on_high_drop_ratio() {
        let (_tmp, wiki, artifacts, dreams) = setup_compile_env();
        let batch = batch_with_two_events();

        // 10 facts, 4 cite a fake event_id → 40% drop → batch rejection.
        let mut facts = Vec::new();
        for _ in 0..6 {
            facts.push(serde_json::json!({
                "file": "user/preferences.md",
                "section": "Style",
                "operation": "add",
                "content": "Good.",
                "supersedes_line": null,
                "source_event_ids": ["evt_aaa"],
            }));
        }
        for _ in 0..4 {
            facts.push(serde_json::json!({
                "file": "user/preferences.md",
                "section": "Style",
                "operation": "add",
                "content": "Bad.",
                "supersedes_line": null,
                "source_event_ids": ["evt_FAKE"],
            }));
        }
        let response = serde_json::to_string(&serde_json::json!({ "facts": facts })).unwrap();
        let provider = MockProvider {
            response,
            delay_ms: 0,
            should_fail: false,
        };

        let compiler = Compiler::new();
        let outcome = compiler
            .compile(&batch, &wiki, &provider, &dreams, &artifacts)
            .await;

        assert_eq!(outcome.status, "rejected");
        assert_eq!(
            outcome.reject_reason.as_deref(),
            Some("high_rejection_rate")
        );
        assert_eq!(outcome.applied.lines_added, 0);

        // Wiki untouched.
        let wiki_text = std::fs::read_to_string(wiki.join("user").join("preferences.md")).unwrap();
        assert_eq!(wiki_text, "# Preferences\n");

        let recent = dreams.read_recent(10);
        assert_eq!(recent[0]["status"], "rejected");
    }

    #[tokio::test]
    async fn compile_persists_artifact_for_replay() {
        let (_tmp, wiki, artifacts, dreams) = setup_compile_env();
        let batch = batch_with_two_events();

        let response = r#"{"facts":[{"file":"user/preferences.md","section":"Style","operation":"add","content":"User prefers short responses.","supersedes_line":null,"source_event_ids":["evt_aaa"]}]}"#;
        let provider = MockProvider::new(response.to_string());
        let mut provider = provider;
        provider.delay_ms = 0;

        let compiler = Compiler::new();
        let outcome = compiler
            .compile(&batch, &wiki, &provider, &dreams, &artifacts)
            .await;

        let artifact_path = artifacts.join(format!("{}.json", outcome.run_id));
        let bytes = std::fs::read(&artifact_path).expect("artifact readable");
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        for key in [
            "rendered_prompt",
            "raw_response",
            "parsed_response",
            "accepted_facts",
            "rejected_facts",
        ] {
            assert!(parsed.get(key).is_some(), "artifact missing key {}", key);
        }
        // Sanity: the rendered prompt must contain the event ids it was built
        // from — that's the whole point of replay.
        assert!(parsed["rendered_prompt"]
            .as_str()
            .unwrap()
            .contains("evt_aaa"));
    }
}
