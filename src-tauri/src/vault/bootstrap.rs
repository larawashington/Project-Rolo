use std::fs;
use std::path::Path;

use crate::vault::atomic::atomic_write;
use crate::vault::config::{WIKI_CORE_IDENTITY, WIKI_LEARNED_BEHAVIORS};

/// Each entry: (relative path under wiki/, file content).
/// `include_str!` means the crate fails to compile if a content file is missing
/// — that's the belt-and-suspenders guarantee from PRD §5.10.
const BOOTSTRAP_FILES: &[(&str, &str)] = &[
    (
        WIKI_CORE_IDENTITY,
        include_str!("bootstrap_content/core_identity.md"),
    ),
    (
        WIKI_LEARNED_BEHAVIORS,
        include_str!("bootstrap_content/learned_behaviors.md"),
    ),
    (
        "personality/mood-state.json",
        include_str!("bootstrap_content/mood_state.json"),
    ),
    (
        "user/identity.md",
        include_str!("bootstrap_content/user_identity.md"),
    ),
    (
        "user/preferences.md",
        include_str!("bootstrap_content/user_preferences.md"),
    ),
    (
        "user/routines.md",
        include_str!("bootstrap_content/user_routines.md"),
    ),
    (
        "world/environment.md",
        include_str!("bootstrap_content/world_environment.md"),
    ),
    (
        "world/calendar.md",
        include_str!("bootstrap_content/world_calendar.md"),
    ),
    (
        "relationships/human.md",
        include_str!("bootstrap_content/relationships_human.md"),
    ),
];

/// Create vault layout (events/, wiki/{user,personality,world,relationships}/, embeddings/)
/// and write any missing bootstrap files. EXISTING wiki files are preserved
/// (never overwritten) — that's the contract that lets the user hand-edit her vault.
pub fn bootstrap_vault(vault_root: &Path) -> std::io::Result<()> {
    fs::create_dir_all(vault_root)?;
    fs::create_dir_all(vault_root.join("events"))?;
    fs::create_dir_all(vault_root.join("wiki"))?;
    fs::create_dir_all(vault_root.join("wiki/user"))?;
    fs::create_dir_all(vault_root.join("wiki/personality"))?;
    fs::create_dir_all(vault_root.join("wiki/world"))?;
    fs::create_dir_all(vault_root.join("wiki/relationships"))?;
    // embeddings/ stays empty: index.bin and chunks.jsonl are reserved for Phase 6.
    fs::create_dir_all(vault_root.join("embeddings"))?;

    for (rel_path, content) in BOOTSTRAP_FILES {
        let target = vault_root.join("wiki").join(rel_path);
        if target.exists() {
            continue;
        }
        atomic_write(&target, content.as_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::chunker::chunk_file;
    use crate::vault::config::BOOTSTRAP_TOKEN_BUDGET;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn bootstrap_writes_all_files_into_empty_dir() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("vault");
        bootstrap_vault(&root).unwrap();
        for (rel, _) in BOOTSTRAP_FILES {
            let p = root.join("wiki").join(rel);
            assert!(p.exists(), "missing: {:?}", p);
            let bytes = fs::read(&p).unwrap();
            assert!(!bytes.is_empty(), "empty: {:?}", p);
        }
        // embeddings dir exists but is empty
        assert!(root.join("embeddings").is_dir());
        assert!(!root.join("embeddings/index.bin").exists());
        assert!(!root.join("embeddings/chunks.jsonl").exists());
        // events dir exists
        assert!(root.join("events").is_dir());
    }

    #[test]
    fn bootstrap_is_idempotent_and_preserves_user_edits() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("vault");
        bootstrap_vault(&root).unwrap();

        // the user hand-edits a file.
        let edited = root.join("wiki/user/preferences.md");
        fs::write(&edited, b"USER EDIT").unwrap();

        // Re-bootstrap should not touch it.
        bootstrap_vault(&root).unwrap();
        assert_eq!(fs::read(&edited).unwrap(), b"USER EDIT");
    }

    #[test]
    fn every_md_file_starts_with_provenance_comment() {
        for (rel, content) in BOOTSTRAP_FILES {
            if !rel.ends_with(".md") {
                continue;
            }
            assert!(
                content.starts_with("<!-- last_compiled_event_ts: null -->\n"),
                "{} missing provenance comment on line 1",
                rel
            );
        }
    }

    #[test]
    fn every_md_has_exactly_one_h1_and_at_least_one_h2() {
        for (rel, content) in BOOTSTRAP_FILES {
            if !rel.ends_with(".md") {
                continue;
            }
            let h1_count = content
                .lines()
                .filter(|l| l.starts_with("# ") && !l.starts_with("## "))
                .count();
            let h2_count = content.lines().filter(|l| l.starts_with("## ")).count();
            assert_eq!(h1_count, 1, "{} expected 1 H1, got {}", rel, h1_count);
            assert!(h2_count >= 1, "{} expected >=1 H2, got {}", rel, h2_count);
        }
    }

    #[test]
    fn mood_state_json_is_valid_json() {
        let content = BOOTSTRAP_FILES
            .iter()
            .find(|(rel, _)| *rel == "personality/mood-state.json")
            .map(|(_, c)| *c)
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(content).unwrap();
        assert!(v.is_object());
    }

    #[test]
    fn per_page_byte_size_under_1500() {
        for (rel, content) in BOOTSTRAP_FILES {
            assert!(
                content.len() < 1500,
                "{} is {} bytes, exceeds 1500-byte sanity cap",
                rel,
                content.len()
            );
        }
    }

    #[test]
    fn total_bootstrap_tokens_under_budget() {
        // Exercises chunker §6.3 tokenizer over every bootstrap .md and asserts
        // the post-chunking token total stays under BOOTSTRAP_TOKEN_BUDGET (4000).
        let mut total: usize = 0;
        for (rel, content) in BOOTSTRAP_FILES {
            if !rel.ends_with(".md") {
                continue;
            }
            for chunk in chunk_file(rel, content) {
                total += chunk.tokens.len();
            }
        }
        assert!(
            total < BOOTSTRAP_TOKEN_BUDGET,
            "bootstrap totals {} tokens, exceeds budget {}",
            total,
            BOOTSTRAP_TOKEN_BUDGET
        );
        // surface the count in test output for diagnostic visibility
        eprintln!("bootstrap total tokens (post-chunk): {}", total);
    }
}
