//! Perception-tab diagnostic probes for Rolo's Command Center.
//!
//! Each probe is a small, dependency-light async function returning a
//! `ProbeOutcome`. `run_all` orchestrates the four live probes in parallel
//! (`tokio::join!`) and appends the three "Coming soon" rows so the frontend
//! gets a single, ordered list. The probes themselves never panic — every
//! failure is converted into a Red `ProbeOutcome` with a fix hint.
//!
//! Dependency policy: probes only reach into `ChatConfig` (for the Ollama
//! base URL) and the filesystem. They do NOT touch the chat engine, the
//! vault, the dreaming compiler, or any other heavy subsystem. This keeps
//! them cheap to invoke on every panel open and easy to unit-test.

use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

use crate::chat::config::ChatConfig;
use crate::http::http_client;

/// Heartbeat shared between `tick.rs` (writer) and `probe_drag_drop`
/// (reader). Holds the unix-millis timestamp of the most recent tick
/// iteration. A stale value means the heartbeat thread is wedged.
pub type DragWatchHeartbeat = Arc<AtomicI64>;

/// Traffic-light status for a single probe. Serializes to lowercase strings
/// (`"green"`, `"amber"`, `"red"`, `"grey"`) which the frontend keys CSS
/// classes off of.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeStatus {
    Green,
    Amber,
    Red,
    Grey,
}

/// Result of a single probe. `id` is a stable machine key used by the
/// frontend for keyed rendering; `label` is the human-readable row title.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeOutcome {
    pub id: String,
    pub label: String,
    pub status: ProbeStatus,
    pub message: String,
    pub fix_hint: Option<String>,
    pub latency_ms: Option<u32>,
}

/// Full report rendered by the Perception tab. `generated_at_ms` is unix
/// millis so the UI can show "Last refreshed Xs ago" without extra
/// conversion work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsReport {
    pub probes: Vec<ProbeOutcome>,
    pub generated_at_ms: i64,
}

// ---------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------

/// Drag-drop perception probe.
///
/// macOS-only: reads the `DragWatchHeartbeat` written by the tick loop and
/// compares its timestamp to `now`. Three thresholds:
///   - `< 1500ms` → Green (healthy)
///   - `1500..3000ms` → Amber (lagging, but might recover)
///   - `>= 3000ms` → Red (wedged; recommend restart)
///
/// On non-macOS platforms returns Grey — the underlying `DragWatcher` is
/// macOS-only, but the heartbeat itself is updated unconditionally (the
/// tick loop runs everywhere), so the platform gate lives here.
pub fn probe_drag_drop(heartbeat: &DragWatchHeartbeat) -> ProbeOutcome {
    let id = "drag_drop".to_string();
    let label = "Drag-drop perception".to_string();

    #[cfg(not(target_os = "macos"))]
    {
        let _ = heartbeat; // suppress unused warning on non-macOS
        return ProbeOutcome {
            id,
            label,
            status: ProbeStatus::Grey,
            message: "Drag-drop watcher is macOS-only.".to_string(),
            fix_hint: None,
            latency_ms: None,
        };
    }

    #[cfg(target_os = "macos")]
    {
        // Match the writer in tick.rs — UNIX_EPOCH millis, no TZ lookup.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let last_ms = heartbeat.load(Ordering::Relaxed);
        let lag_ms = now_ms.saturating_sub(last_ms);

        if lag_ms < 1500 {
            ProbeOutcome {
                id,
                label,
                status: ProbeStatus::Green,
                message: format!("Last tick {}ms ago.", lag_ms),
                fix_hint: None,
                latency_ms: None,
            }
        } else if lag_ms < 3000 {
            ProbeOutcome {
                id,
                label,
                status: ProbeStatus::Amber,
                message: format!("Drag-drop tick lagging — last beat {}ms ago", lag_ms),
                fix_hint: Some("Restart Rolo to re-arm drag detection.".to_string()),
                latency_ms: None,
            }
        } else {
            ProbeOutcome {
                id,
                label,
                status: ProbeStatus::Red,
                message: format!("Drag-drop watcher silent for {}ms", lag_ms),
                fix_hint: Some("Restart Rolo to re-arm drag detection.".to_string()),
                latency_ms: None,
            }
        }
    }
}

/// Ollama reachability probe. `GET {base_url}/api/tags` with a 2s timeout.
/// Any 2xx → Green; non-2xx → Red with the status code; transport failure
/// → Red with the underlying error string. Latency captured for green only.
pub async fn probe_ollama(base_url: String) -> ProbeOutcome {
    let id = "ollama".to_string();
    let label = "Ollama (local brain)".to_string();
    let fix_hint = Some(
        "Start Ollama: `ollama serve` in Terminal, or switch your Brain to a cloud provider."
            .to_string(),
    );

    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let client = match http_client(Duration::from_secs(2)) {
        Ok(c) => c,
        Err(e) => {
            return ProbeOutcome {
                id,
                label,
                status: ProbeStatus::Red,
                message: format!("Cannot build HTTP client: {}", e),
                fix_hint,
                latency_ms: None,
            };
        }
    };

    let start = Instant::now();
    match client.get(&url).send().await {
        Ok(resp) => {
            let elapsed_ms = start.elapsed().as_millis() as u32;
            let status = resp.status();
            if status.is_success() {
                ProbeOutcome {
                    id,
                    label,
                    status: ProbeStatus::Green,
                    message: format!("Reachable at {}", base_url),
                    fix_hint: None,
                    latency_ms: Some(elapsed_ms),
                }
            } else {
                ProbeOutcome {
                    id,
                    label,
                    status: ProbeStatus::Red,
                    message: format!("Ollama returned HTTP {}", status.as_u16()),
                    fix_hint,
                    latency_ms: Some(elapsed_ms),
                }
            }
        }
        Err(e) => ProbeOutcome {
            id,
            label,
            status: ProbeStatus::Red,
            message: format!("Cannot reach Ollama at {}: {}", base_url, e),
            fix_hint,
            latency_ms: None,
        },
    }
}

/// Embedder probe. `POST {base_url}/api/embed` with the smallest possible
/// payload and a 3s timeout (slightly longer than the bare `/api/tags`
/// because cold-starting the embedder model can take a beat).
///
/// Shares the Ollama base URL — the PRD explicitly notes there's no
/// separate embedder config.
pub async fn probe_embedder(base_url: String) -> ProbeOutcome {
    let id = "embedder".to_string();
    let label = "Embedder (nomic-embed-text)".to_string();
    let fix_hint = Some("Run `ollama pull nomic-embed-text` and try again.".to_string());

    let url = format!("{}/api/embed", base_url.trim_end_matches('/'));
    let client = match http_client(Duration::from_secs(3)) {
        Ok(c) => c,
        Err(e) => {
            return ProbeOutcome {
                id,
                label,
                status: ProbeStatus::Red,
                message: format!("Cannot build HTTP client: {}", e),
                fix_hint,
                latency_ms: None,
            };
        }
    };

    let body = serde_json::json!({
        "model": "nomic-embed-text",
        "input": "x",
    });

    let start = Instant::now();
    match client.post(&url).json(&body).send().await {
        Ok(resp) => {
            let elapsed_ms = start.elapsed().as_millis() as u32;
            let status = resp.status();
            if status.is_success() {
                ProbeOutcome {
                    id,
                    label,
                    status: ProbeStatus::Green,
                    message: "Embedder responded.".to_string(),
                    fix_hint: None,
                    latency_ms: Some(elapsed_ms),
                }
            } else {
                ProbeOutcome {
                    id,
                    label,
                    status: ProbeStatus::Red,
                    message: format!("Embedder returned HTTP {}", status.as_u16()),
                    fix_hint,
                    latency_ms: Some(elapsed_ms),
                }
            }
        }
        Err(e) => ProbeOutcome {
            id,
            label,
            status: ProbeStatus::Red,
            message: format!("Cannot reach embedder at {}: {}", base_url, e),
            fix_hint,
            latency_ms: None,
        },
    }
}

/// Vault writability probe. Tries to write a 1-byte `.heartbeat` file at
/// `vault_root` and immediately delete it. Green on success; Red on any
/// filesystem error. The file is best-effort cleaned up — if delete fails
/// after write, we still report Green (write was the actual probe) and
/// log the leftover at warn level.
pub async fn probe_vault(vault_root: PathBuf) -> ProbeOutcome {
    let id = "vault".to_string();
    let label = "Vault writability".to_string();
    let fix_hint = Some(
        "Check disk space and that `~/Library/Application Support/com.larawashington.rolo` exists and is writable."
            .to_string(),
    );

    let heartbeat_path = vault_root.join(".heartbeat");

    // Ensure the parent dir exists; if it doesn't, the write will fail with
    // a clear NotFound error and we'll surface it.
    let parent_exists = vault_root.exists();
    if !parent_exists {
        if let Err(e) = std::fs::create_dir_all(&vault_root) {
            return ProbeOutcome {
                id,
                label,
                status: ProbeStatus::Red,
                message: format!("Cannot write to vault: {}", e),
                fix_hint,
                latency_ms: None,
            };
        }
    }

    match std::fs::write(&heartbeat_path, b"x") {
        Ok(()) => {
            if let Err(e) = std::fs::remove_file(&heartbeat_path) {
                log::warn!(
                    "[Rolo] Command Center: vault heartbeat write succeeded but cleanup failed: {} \
                     (leftover at {})",
                    e,
                    heartbeat_path.display()
                );
            }
            ProbeOutcome {
                id,
                label,
                status: ProbeStatus::Green,
                message: "Vault is writable.".to_string(),
                fix_hint: None,
                latency_ms: None,
            }
        }
        Err(e) => ProbeOutcome {
            id,
            label,
            status: ProbeStatus::Red,
            message: format!("Cannot write to vault: {}", e),
            fix_hint,
            latency_ms: None,
        },
    }
}

/// Liveness probe for one of the three passive-perception producers
/// (foreground app, trash watcher, downloads watcher). Reads the
/// `Arc<AtomicI64>` heartbeat stamped at the top of the producer's loop and
/// compares its unix-millis to `now`:
///   - heartbeat == 0           → Red (never stamped — producer failed to spawn or bailed)
///   - lag < `green_within_ms`  → Green
///   - lag < `red_after_ms`     → Amber
///   - lag >= `red_after_ms`    → Red
///
/// On non-macOS the producers are not started (gated in `lib.rs`), so the
/// probe returns Grey with a platform-only message — matching the
/// `drag_drop` convention.
pub fn probe_perception_producer(
    heartbeat: Option<&Arc<AtomicI64>>,
    id: &str,
    label: &str,
    fix_hint: &str,
    green_within_ms: i64,
    red_after_ms: i64,
) -> ProbeOutcome {
    let id = id.to_string();
    let label = label.to_string();

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (heartbeat, fix_hint, green_within_ms, red_after_ms);
        return ProbeOutcome {
            id,
            label,
            status: ProbeStatus::Grey,
            message: "Passive perception is macOS-only.".to_string(),
            fix_hint: None,
            latency_ms: None,
        };
    }

    #[cfg(target_os = "macos")]
    {
        let hint = Some(fix_hint.to_string());
        let beat = match heartbeat {
            Some(b) => b,
            None => {
                return ProbeOutcome {
                    id,
                    label,
                    status: ProbeStatus::Red,
                    message: "Perception heartbeats not registered — watcher disabled.".to_string(),
                    fix_hint: hint,
                    latency_ms: None,
                };
            }
        };
        let last_ms = beat.load(Ordering::Relaxed);
        if last_ms == 0 {
            return ProbeOutcome {
                id,
                label,
                status: ProbeStatus::Red,
                message: "Watcher never started — likely TCC denial or missing folder.".to_string(),
                fix_hint: hint,
                latency_ms: None,
            };
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let lag_ms = now_ms.saturating_sub(last_ms);

        if lag_ms < green_within_ms {
            ProbeOutcome {
                id,
                label,
                status: ProbeStatus::Green,
                message: format!("Last beat {}ms ago.", lag_ms),
                fix_hint: None,
                latency_ms: None,
            }
        } else if lag_ms < red_after_ms {
            ProbeOutcome {
                id,
                label,
                status: ProbeStatus::Amber,
                message: format!("Watcher lagging — last beat {}ms ago", lag_ms),
                fix_hint: hint,
                latency_ms: None,
            }
        } else {
            ProbeOutcome {
                id,
                label,
                status: ProbeStatus::Red,
                message: format!("Watcher silent for {}ms", lag_ms),
                fix_hint: hint,
                latency_ms: None,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

/// Run every probe and assemble the report.
///
/// The four live probes run in parallel via `tokio::join!`. Each returns a
/// `ProbeOutcome` directly (not a `Result`), so failures are first-class
/// values and `tokio::try_join!` is intentionally NOT used — we never want
/// one probe's transport failure to short-circuit the others.
///
/// Final ordering (load-bearing — see PRD Phase 3 acceptance):
/// drag_drop, ollama, embedder, vault, window_detection, trash_monitoring,
/// downloads_monitoring.
pub async fn run_all(app: &AppHandle) -> DiagnosticsReport {
    // Snapshot the heartbeat Arc up front so the drag probe doesn't hold a
    // borrow on `app` across the await points.
    let heartbeat: DragWatchHeartbeat = match app.try_state::<DragWatchHeartbeat>() {
        Some(s) => Arc::clone(&*s),
        None => {
            // Should be impossible after Phase 3 lib.rs wiring lands, but if
            // someone calls run_all before setup completes, fall back to a
            // fresh heartbeat (which will read "now" and report Green). A
            // missing state is a programmer error, not user-facing.
            log::warn!(
                "[Rolo] Command Center: DragWatchHeartbeat state not registered — drag probe will read a synthetic heartbeat."
            );
            Arc::new(AtomicI64::new(chrono::Local::now().timestamp_millis()))
        }
    };

    // Load the brain config once and reuse the base URL for both Ollama
    // probes. `ChatConfig::load` reads from `com.rolo.desktop-pet/config.json`
    // (the historic bundle id — pre-existing tech debt the PRD calls out).
    let chat_config = ChatConfig::load();
    let base_url = chat_config.inference.ollama.base_url.clone();

    let vault_root: PathBuf = app
        .path()
        .app_data_dir()
        .map(|d| d.join("vault"))
        .unwrap_or_else(|_| PathBuf::from("/tmp/rolo-vault-fallback"));

    let drag = probe_drag_drop(&heartbeat);

    // Snapshot the perception heartbeats. State is only registered on macOS
    // (lib.rs gates the `spawn_all()` call), so `try_state` returns None on
    // other platforms — the probes detect that and return Grey.
    #[cfg(target_os = "macos")]
    let perception_hb = app
        .try_state::<crate::perception::SharedPerceptionHeartbeats>()
        .map(|s| Arc::clone(&*s));

    let (ollama, embedder, vault) = tokio::join!(
        probe_ollama(base_url.clone()),
        probe_embedder(base_url),
        probe_vault(vault_root),
    );

    // The foreground poller sleeps 2s between iterations; trash + downloads
    // both wake at 500ms via `recv_timeout`. Pick green-thresholds well clear
    // of those cadences so a normal idle loop never trips Amber.
    #[cfg(target_os = "macos")]
    let (window_probe, trash_probe, downloads_probe) = {
        let foreground_hb = perception_hb.as_ref().map(|h| &h.foreground);
        let trash_hb = perception_hb.as_ref().map(|h| &h.trash);
        let downloads_hb = perception_hb.as_ref().map(|h| &h.downloads);
        (
            probe_perception_producer(
                foreground_hb,
                "window_detection",
                "Window detection",
                "Grant Accessibility / Screen Recording to Rolo in System Settings → Privacy & Security, then restart.",
                5_000,
                30_000,
            ),
            probe_perception_producer(
                trash_hb,
                "trash_monitoring",
                "Trash monitoring",
                "Check that ~/.Trash exists and Rolo has Full Disk Access, then restart.",
                2_500,
                15_000,
            ),
            probe_perception_producer(
                downloads_hb,
                "downloads_monitoring",
                "Downloads monitoring",
                "Grant Downloads folder access to Rolo in System Settings → Privacy & Security → Files and Folders, then restart.",
                2_500,
                15_000,
            ),
        )
    };
    #[cfg(not(target_os = "macos"))]
    let (window_probe, trash_probe, downloads_probe) = (
        probe_perception_producer(None, "window_detection", "Window detection", "", 0, 0),
        probe_perception_producer(None, "trash_monitoring", "Trash monitoring", "", 0, 0),
        probe_perception_producer(
            None,
            "downloads_monitoring",
            "Downloads monitoring",
            "",
            0,
            0,
        ),
    );

    let probes = vec![
        drag,
        ollama,
        embedder,
        vault,
        window_probe,
        trash_probe,
        downloads_probe,
    ];

    DiagnosticsReport {
        probes,
        generated_at_ms: chrono::Local::now().timestamp_millis(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_status_serializes_as_lowercase_snake_case() {
        // Frontend keys CSS off this exact string — if serde's rename rule
        // is ever changed, this test breaks loudly rather than the UI
        // silently going colorless.
        assert_eq!(
            serde_json::to_string(&ProbeStatus::Green).expect("serialize"),
            "\"green\""
        );
        assert_eq!(
            serde_json::to_string(&ProbeStatus::Amber).expect("serialize"),
            "\"amber\""
        );
        assert_eq!(
            serde_json::to_string(&ProbeStatus::Red).expect("serialize"),
            "\"red\""
        );
        assert_eq!(
            serde_json::to_string(&ProbeStatus::Grey).expect("serialize"),
            "\"grey\""
        );
    }

    #[test]
    fn perception_probe_red_when_heartbeat_unregistered() {
        let outcome =
            probe_perception_producer(None, "window_detection", "Window detection", "fix", 5, 30);
        #[cfg(target_os = "macos")]
        {
            assert_eq!(outcome.status, ProbeStatus::Red);
            assert!(outcome.fix_hint.is_some());
        }
        #[cfg(not(target_os = "macos"))]
        {
            assert_eq!(outcome.status, ProbeStatus::Grey);
        }
        assert_eq!(outcome.id, "window_detection");
        assert_eq!(outcome.label, "Window detection");
    }

    #[test]
    fn perception_probe_red_when_heartbeat_never_stamped() {
        let beat: Arc<AtomicI64> = Arc::new(AtomicI64::new(0));
        let outcome = probe_perception_producer(
            Some(&beat),
            "trash_monitoring",
            "Trash monitoring",
            "fix",
            2_500,
            15_000,
        );
        #[cfg(target_os = "macos")]
        {
            assert_eq!(outcome.status, ProbeStatus::Red);
            assert!(outcome.message.contains("never started"));
        }
    }

    #[test]
    fn perception_probe_green_when_heartbeat_fresh() {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let beat: Arc<AtomicI64> = Arc::new(AtomicI64::new(now_ms));
        let outcome = probe_perception_producer(
            Some(&beat),
            "downloads_monitoring",
            "Downloads monitoring",
            "fix",
            2_500,
            15_000,
        );
        #[cfg(target_os = "macos")]
        {
            assert_eq!(outcome.status, ProbeStatus::Green);
        }
    }

    #[tokio::test]
    async fn probe_vault_green_for_temp_dir() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let outcome = probe_vault(tmp.path().to_path_buf()).await;
        assert_eq!(
            outcome.status,
            ProbeStatus::Green,
            "msg: {}",
            outcome.message
        );
        assert_eq!(outcome.id, "vault");
        // Cleanup must have removed the heartbeat file.
        assert!(
            !tmp.path().join(".heartbeat").exists(),
            "heartbeat file should be cleaned up after probe"
        );
    }

    #[tokio::test]
    async fn probe_vault_red_for_unwritable_path() {
        // Skip if running as root — root can write almost anywhere, which
        // would defeat the point of the test. The CI macOS runner runs as
        // a non-root user, so this branch is exercised there. We detect
        // root via $USER rather than adding a libc dep just for getuid.
        if std::env::var("USER").as_deref() == Ok("root")
            || std::env::var("USERNAME").as_deref() == Ok("root")
        {
            eprintln!("Skipping probe_vault_red_for_unwritable_path: running as root");
            return;
        }

        // A nested path under /nonexistent — `create_dir_all` will fail
        // because `/nonexistent` itself isn't creatable by a normal user.
        let bad = PathBuf::from("/nonexistent/rolo-vault-test/sub");
        let outcome = probe_vault(bad).await;
        assert_eq!(
            outcome.status,
            ProbeStatus::Red,
            "expected red, got {:?} with message: {}",
            outcome.status,
            outcome.message
        );
        assert!(outcome.fix_hint.is_some());
        assert!(outcome.message.contains("Cannot write to vault"));
    }

    #[test]
    fn probe_drag_drop_red_when_heartbeat_stale_on_macos() {
        // Stale heartbeat: pretend the last tick was an hour ago. On macOS
        // the probe must return Red with the "silent for Xms" copy.
        let one_hour_ago = chrono::Local::now().timestamp_millis() - 3_600_000;
        let heartbeat: DragWatchHeartbeat = Arc::new(AtomicI64::new(one_hour_ago));
        let outcome = probe_drag_drop(&heartbeat);

        #[cfg(target_os = "macos")]
        {
            assert_eq!(outcome.status, ProbeStatus::Red, "msg: {}", outcome.message);
            assert!(outcome.message.contains("silent for"));
            assert!(outcome.fix_hint.is_some());
        }
        #[cfg(not(target_os = "macos"))]
        {
            // On other platforms the probe always returns Grey regardless
            // of heartbeat freshness.
            assert_eq!(outcome.status, ProbeStatus::Grey);
        }
    }

    #[test]
    fn probe_drag_drop_green_when_heartbeat_fresh_on_macos() {
        let now = chrono::Local::now().timestamp_millis();
        let heartbeat: DragWatchHeartbeat = Arc::new(AtomicI64::new(now));
        let outcome = probe_drag_drop(&heartbeat);

        #[cfg(target_os = "macos")]
        {
            assert_eq!(outcome.status, ProbeStatus::Green);
        }
        #[cfg(not(target_os = "macos"))]
        {
            assert_eq!(outcome.status, ProbeStatus::Grey);
        }
    }
}
