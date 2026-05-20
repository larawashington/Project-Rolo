//! Advisory wiki linter — runs immediately after a successful compile to look
//! for contradictions, duplicates, and stale entries across all wiki files.
//! Findings are appended to `vault/dreams.jsonl` under the same `run_id`
//! family as the parent compile (linked via `parent_run_id`).
//!
//! The linter is **strictly read-only** in Phase 5 (PRD §5). It must never
//! call `atomic_write`, never invoke `Compiler::apply`, and never mutate
//! `meta.json`. Tests assert this with a pre/post SHA-256 of the wiki tree.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::chat::provider::{GenerationConfig, InferenceProvider, ProviderChatMessage};
use crate::vault::dreams_log::DreamsLog;

/// One lint finding. Mirrors the JSON shape from PRD §5 — `type` is renamed
/// because it's a Rust keyword.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LintFinding {
    #[serde(rename = "type")]
    pub type_: String,
    pub files: Vec<String>,
    pub summary: String,
    pub suggested_resolution: String,
}

/// LLM response shape. Empty `findings` is the clean-wiki path.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct LintResponse {
    #[serde(default)]
    pub findings: Vec<LintFinding>,
}

/// Outcome surfaced to the dreaming orchestrator. `status="success"` means
/// the lint LLM call returned and parsed cleanly — it does NOT mean the wiki
/// was clean (`findings_count` may be > 0).
#[derive(Debug, Clone)]
pub struct LintOutcome {
    pub findings_count: usize,
    pub status: String,
}

/// Stateless advisory linter. Symmetric with `Compiler` so D-section can
/// hold a `(Compiler, Linter)` pair.
pub struct Linter;

impl Linter {
    pub fn new() -> Self {
        Self
    }

    /// Read the entire wiki, ask the LLM for findings, append the results to
    /// the dreams ledger under `parent_run_id`. NEVER writes wiki files.
    ///
    /// All failure modes (LLM error, parse error, IO error reading the wiki)
    /// produce a `failed` dreams entry and return a `LintOutcome` with
    /// `status="failed"` — they do NOT panic and do NOT abort the dream chain.
    pub async fn lint(
        &self,
        wiki_root: &Path,
        parent_run_id: &str,
        provider: &dyn InferenceProvider,
        dreams: &DreamsLog,
    ) -> LintOutcome {
        // 1. Snapshot the wiki — every file, regardless of writable status.
        // The linter is allowed to *see* core-identity.md; it's just not
        // allowed to fix anything.
        let wiki_text = match render_wiki_for_lint(wiki_root) {
            Ok(s) => s,
            Err(e) => {
                log::warn!(
                    "[Rolo dreaming] lint parent_run_id={} read_failed={}",
                    parent_run_id,
                    e
                );
                append_failed_lint(dreams, parent_run_id, "read_failed");
                return LintOutcome {
                    findings_count: 0,
                    status: "failed".to_string(),
                };
            }
        };

        // 2. Build prompt per PRD §5 (verbatim).
        let prompt = build_lint_prompt(&wiki_text);
        log::trace!(
            "[Rolo dreaming] lint parent_run_id={} prompt_bytes={}",
            parent_run_id,
            prompt.len()
        );

        // 3. Call provider.
        let messages = vec![ProviderChatMessage {
            role: "user".to_string(),
            content: prompt,
        }];
        let config = GenerationConfig {
            temperature: 0.3,
            ..GenerationConfig::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::channel::<String>(1024);
        let raw = match provider.generate(messages, config, tx).await {
            Ok(s) => s,
            Err(e) => {
                log::warn!(
                    "[Rolo dreaming] lint parent_run_id={} llm_error={}",
                    parent_run_id,
                    e
                );
                append_failed_lint(dreams, parent_run_id, "llm_error");
                return LintOutcome {
                    findings_count: 0,
                    status: "failed".to_string(),
                };
            }
        };

        // 4. Parse — leniently. `{}` (no findings key) parses as empty via
        // `#[serde(default)]` because lint, unlike compile, treats absent
        // findings the same as `[]`. The linter is advisory — there is no
        // semantic difference between "the model said nothing wrong" and
        // "the model didn't think the question made sense".
        // Strip ```json fences first — Gemma wraps structured output even
        // when the prompt asks for raw JSON. See compiler::strip_json_code_fences.
        let parsed: LintResponse =
            match serde_json::from_str(crate::vault::compiler::strip_json_code_fences(&raw)) {
                Ok(p) => p,
                Err(e) => {
                    log::warn!(
                        "[Rolo dreaming] lint parent_run_id={} parse_failure={} raw={}",
                        parent_run_id,
                        e,
                        raw
                    );
                    append_failed_lint(dreams, parent_run_id, "parse_failure");
                    return LintOutcome {
                        findings_count: 0,
                        status: "failed".to_string(),
                    };
                }
            };

        let count = parsed.findings.len();

        // 5. Append the success entry to dreams.jsonl.
        let entry = serde_json::json!({
            "run_type": "lint",
            "parent_run_id": parent_run_id,
            "status": "success",
            "findings": parsed.findings,
        });
        if let Err(e) = dreams.append(entry) {
            log::error!(
                "[Rolo dreaming] lint parent_run_id={} dreams_append_failed={}",
                parent_run_id,
                e
            );
        }

        log::info!(
            "[Rolo dreaming] lint parent_run_id={} findings={}",
            parent_run_id,
            count
        );

        LintOutcome {
            findings_count: count,
            status: "success".to_string(),
        }
    }
}

impl Default for Linter {
    fn default() -> Self {
        Self::new()
    }
}

/// Render every wiki file with a header and 1-based line numbers, the format
/// the §5 lint prompt expects. Recursively walks `wiki_root`.
fn render_wiki_for_lint(wiki_root: &Path) -> std::io::Result<String> {
    if !wiki_root.exists() {
        return Ok(String::new());
    }
    // Sort entries for deterministic prompts — relevant for the input hash
    // when D-section adds it.
    let mut paths: Vec<std::path::PathBuf> = walkdir::WalkDir::new(wiki_root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();
    paths.sort();

    let mut out = String::with_capacity(8192);
    for path in paths {
        let rel = match path.strip_prefix(wiki_root) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let contents = std::fs::read_to_string(&path)?;
        out.push_str("=== FILE: ");
        out.push_str(&rel_str);
        out.push_str(" ===\n");
        for (i, line) in contents.lines().enumerate() {
            out.push_str(&format!("{}: {}\n", i + 1, line));
        }
        out.push('\n');
    }
    Ok(out)
}

/// Build the §5 lint prompt — wording locked against PRD; bumping is a
/// `prompt_template_version` event.
fn build_lint_prompt(wiki_text: &str) -> String {
    let mut out = String::with_capacity(wiki_text.len() + 1024);
    out.push_str(
        "Review these knowledge files for contradictions, duplicates, and stale entries.\n\n",
    );
    out.push_str(wiki_text);
    out.push('\n');
    out.push_str("Output JSON only:\n\n");
    out.push_str("{\n");
    out.push_str("  \"findings\": [\n");
    out.push_str("    {\n");
    out.push_str("      \"type\": \"contradiction\" | \"duplicate\" | \"stale\",\n");
    out.push_str("      \"files\": [\"user/preferences.md:12\", \"user/identity.md:4\"],\n");
    out.push_str("      \"summary\": \"preferences.md:12 says 'works mornings' but identity.md:4 says 'night owl'.\",\n");
    out.push_str(
        "      \"suggested_resolution\": \"keep_newer\" | \"merge\" | \"flag_for_user\"\n",
    );
    out.push_str("    }\n");
    out.push_str("  ]\n");
    out.push_str("}\n\n");
    out.push_str("If clean: {\"findings\": []}.\n");
    out
}

/// Append a `status="failed"` lint entry to the dreams log. The dreaming
/// orchestrator never reads this beyond the `status` field, but the audit
/// trail wants the parent_run_id and reason on file.
fn append_failed_lint(dreams: &DreamsLog, parent_run_id: &str, reason: &str) {
    let entry = serde_json::json!({
        "run_type": "lint",
        "parent_run_id": parent_run_id,
        "status": "failed",
        "reason": reason,
    });
    if let Err(e) = dreams.append(entry) {
        log::error!(
            "[Rolo dreaming] lint append_failed_entry parent_run_id={} err={}",
            parent_run_id,
            e
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::mock_provider::MockProvider;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    /// Stand up a wiki dir with one file, plus a fresh DreamsLog.
    fn setup_lint_env() -> (TempDir, std::path::PathBuf, DreamsLog) {
        let tmp = TempDir::new().unwrap();
        let wiki = tmp.path().join("wiki");
        std::fs::create_dir_all(wiki.join("user")).unwrap();
        std::fs::write(
            wiki.join("user").join("preferences.md"),
            "# Preferences\nUser likes tea.\n",
        )
        .unwrap();
        let dreams = DreamsLog::new(tmp.path());
        (tmp, wiki, dreams)
    }

    /// Walk the wiki and SHA-256 every file's bytes into a stable map. Used to
    /// prove the linter touched no file.
    fn hash_wiki(wiki_root: &Path) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        for entry in walkdir::WalkDir::new(wiki_root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
        {
            let rel = entry
                .path()
                .strip_prefix(wiki_root)
                .unwrap()
                .to_string_lossy()
                .into_owned();
            let bytes = std::fs::read(entry.path()).unwrap();
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            out.insert(rel, format!("{:x}", hasher.finalize()));
        }
        out
    }

    #[tokio::test]
    async fn lint_appends_findings_under_parent_run_id() {
        let (_tmp, wiki, dreams) = setup_lint_env();
        let response = serde_json::json!({
            "findings": [{
                "type": "contradiction",
                "files": ["user/preferences.md:2", "user/identity.md:4"],
                "summary": "tea vs coffee.",
                "suggested_resolution": "keep_newer"
            }]
        })
        .to_string();
        let provider = MockProvider {
            response,
            delay_ms: 0,
            should_fail: false,
        };

        let linter = Linter::new();
        let outcome = linter.lint(&wiki, "drm_test", &provider, &dreams).await;
        assert_eq!(outcome.status, "success");
        assert_eq!(outcome.findings_count, 1);

        let recent = dreams.read_recent(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0]["run_type"], "lint");
        assert_eq!(recent[0]["parent_run_id"], "drm_test");
        assert_eq!(recent[0]["status"], "success");
        assert_eq!(recent[0]["findings"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn lint_failure_does_not_abort_compile_chain() {
        let (_tmp, wiki, dreams) = setup_lint_env();
        let provider = MockProvider::failing("oops");

        let linter = Linter::new();
        // Must NOT panic. Returns a structured failure outcome.
        let outcome = linter.lint(&wiki, "drm_test", &provider, &dreams).await;
        assert_eq!(outcome.status, "failed");
        assert_eq!(outcome.findings_count, 0);

        let recent = dreams.read_recent(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0]["status"], "failed");
        assert_eq!(recent[0]["run_type"], "lint");
    }

    #[tokio::test]
    async fn lint_with_clean_wiki_writes_zero_findings_entry() {
        let (_tmp, wiki, dreams) = setup_lint_env();
        let provider = MockProvider {
            response: r#"{"findings":[]}"#.to_string(),
            delay_ms: 0,
            should_fail: false,
        };

        let linter = Linter::new();
        let outcome = linter.lint(&wiki, "drm_test", &provider, &dreams).await;
        assert_eq!(outcome.status, "success");
        assert_eq!(outcome.findings_count, 0);

        let recent = dreams.read_recent(10);
        assert_eq!(recent[0]["findings"].as_array().unwrap().len(), 0);
        assert_eq!(recent[0]["status"], "success");
    }

    #[tokio::test]
    async fn lint_does_not_modify_wiki() {
        let (_tmp, wiki, dreams) = setup_lint_env();
        let before = hash_wiki(&wiki);

        let provider = MockProvider {
            response: r#"{"findings":[{"type":"duplicate","files":["user/preferences.md:1"],"summary":"d","suggested_resolution":"merge"}]}"#.to_string(),
            delay_ms: 0,
            should_fail: false,
        };
        let linter = Linter::new();
        let _ = linter.lint(&wiki, "drm_test", &provider, &dreams).await;

        let after = hash_wiki(&wiki);
        assert_eq!(before, after, "linter must not modify wiki files");
    }
}
