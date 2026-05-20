use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};

use super::atomic::atomic_write;
use super::config::SCHEMA_VERSION;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultMeta {
    pub schema_version: u32,
    pub created_at: DateTime<Local>,
    pub last_compile_time: Option<DateTime<Local>>,
    pub last_compiled_event_timestamps: BTreeMap<String, Option<DateTime<Local>>>,
    pub stats: VaultStats,
    /// True when the embedding index is missing or out-of-date with the
    /// current embedder model digest. Set on boot after a digest mismatch or
    /// missing index; cleared after a successful rebuild. (PRD §15)
    #[serde(default)]
    pub embedding_dirty: bool,
    /// User declined the embed model pull during setup wizard. When true,
    /// `rebuild_index` skips the embedding rebuild thread entirely. (PRD §13 row 13)
    #[serde(default)]
    pub embedding_optout: bool,
    /// Wiki paths the user has manually reverted; the dreaming compiler must
    /// not propose patches to anything in this set on subsequent runs.
    /// Stored as `BTreeSet` so ordering is stable across save/load (PRD §A3).
    #[serde(default)]
    pub revert_blocklist: BTreeSet<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VaultStats {
    pub total_events_logged: u64,
    pub total_dreams: u64,
    pub last_retention_sweep: Option<DateTime<Local>>,
    pub total_wiki_tokens: u64,
}

impl VaultMeta {
    pub fn defaults_now() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            created_at: Local::now(),
            last_compile_time: None,
            last_compiled_event_timestamps: BTreeMap::new(),
            stats: VaultStats::default(),
            embedding_dirty: false,
            embedding_optout: false,
            revert_blocklist: BTreeSet::new(),
        }
    }

    pub fn load_or_default(path: &Path) -> Self {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Self::defaults_now();
            }
            Err(e) => {
                log::warn!(
                    "[Rolo vault] meta.json read failed at {:?} ({}), using defaults",
                    path,
                    e
                );
                return Self::defaults_now();
            }
        };

        // Probe schema_version first so we can preserve created_at across a
        // best-effort migration (PRD §2.3 / Decision #14).
        let probe: Option<SchemaProbe> = serde_json::from_slice(&bytes).ok();
        match probe {
            Some(p) if p.schema_version == SCHEMA_VERSION => {
                match serde_json::from_slice::<VaultMeta>(&bytes) {
                    Ok(meta) => meta,
                    Err(e) => {
                        log::warn!(
                            "[Rolo vault] meta.json parse failed ({}), using defaults",
                            e
                        );
                        Self::defaults_now()
                    }
                }
            }
            Some(p) => {
                log::warn!(
                    "[Rolo vault] meta.json schema_version mismatch ({} != {}), \
                     migrating to defaults but preserving created_at",
                    p.schema_version,
                    SCHEMA_VERSION
                );
                let mut migrated = Self::defaults_now();
                if let Some(ts) = p.created_at {
                    migrated.created_at = ts;
                }
                migrated
            }
            None => {
                log::warn!("[Rolo vault] meta.json corrupt or unparseable, using defaults");
                Self::defaults_now()
            }
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        atomic_write(path, &json)
    }
}

/// Minimal probe used to detect schema version without failing on unknown
/// future fields. `serde(default)` lets either field be missing.
#[derive(Deserialize)]
struct SchemaProbe {
    #[serde(default)]
    schema_version: u32,
    #[serde(default)]
    created_at: Option<DateTime<Local>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn defaults_now_uses_current_schema_version() {
        let m = VaultMeta::defaults_now();
        assert_eq!(m.schema_version, SCHEMA_VERSION);
        assert!(m.last_compile_time.is_none());
        assert!(m.last_compiled_event_timestamps.is_empty());
        assert_eq!(m.stats.total_events_logged, 0);
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("meta.json");
        let mut original = VaultMeta::defaults_now();
        original.stats.total_events_logged = 42;
        original
            .last_compiled_event_timestamps
            .insert("2026-04-26.jsonl".into(), Some(Local::now()));
        original
            .last_compiled_event_timestamps
            .insert("2026-04-27.jsonl".into(), None);

        original.save(&path).unwrap();
        let loaded = VaultMeta::load_or_default(&path);

        assert_eq!(loaded.schema_version, original.schema_version);
        assert_eq!(loaded.stats.total_events_logged, 42);
        assert_eq!(
            loaded.last_compiled_event_timestamps.len(),
            original.last_compiled_event_timestamps.len()
        );
        // Timestamp serialization is RFC3339 — round trip preserves the instant
        // (formatting may shift TZ offset string but to_rfc3339 == to_rfc3339).
        assert_eq!(
            loaded.created_at.to_rfc3339(),
            original.created_at.to_rfc3339()
        );
    }

    #[test]
    fn missing_file_returns_defaults_silently() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("missing.json");
        let m = VaultMeta::load_or_default(&path);
        assert_eq!(m.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn corrupt_file_returns_defaults() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("meta.json");
        fs::write(&path, b"{not valid json").unwrap();
        let m = VaultMeta::load_or_default(&path);
        assert_eq!(m.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn schema_version_mismatch_returns_defaults_preserving_created_at() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("meta.json");
        // Future schema_version with a created_at we want preserved.
        let earlier = Local::now();
        let body = serde_json::json!({
            "schema_version": 999,
            "created_at": earlier.to_rfc3339(),
            "future_field": "we don't know what this is",
        });
        fs::write(&path, serde_json::to_vec(&body).unwrap()).unwrap();

        let m = VaultMeta::load_or_default(&path);
        assert_eq!(m.schema_version, SCHEMA_VERSION);
        assert_eq!(m.created_at.to_rfc3339(), earlier.to_rfc3339());
        assert!(m.last_compile_time.is_none());
    }

    #[test]
    fn revert_blocklist_round_trips() {
        // The dreaming compiler reads this set on every run to skip already-
        // reverted paths; it must survive a save/load cycle byte-for-byte.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("meta.json");
        let mut original = VaultMeta::defaults_now();
        original
            .revert_blocklist
            .insert("user/preferences.md".into());
        original
            .revert_blocklist
            .insert("relationships/lara.md".into());

        original.save(&path).unwrap();
        let loaded = VaultMeta::load_or_default(&path);

        assert_eq!(loaded.revert_blocklist.len(), 2);
        assert!(loaded.revert_blocklist.contains("user/preferences.md"));
        assert!(loaded.revert_blocklist.contains("relationships/lara.md"));
    }

    #[test]
    fn save_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("a").join("b").join("meta.json");
        VaultMeta::defaults_now().save(&nested).unwrap();
        assert!(nested.exists());
    }
}
