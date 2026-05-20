//! Passive perception — three event-driven producers that quietly shape
//! Rolo's idle-speech prompt.
//!
//! Architecture: three producer threads (Trash, Downloads, Foreground app)
//! emit `PerceptionEvent`s on a single mpsc channel. The receiver lives in
//! the tick thread, which drains the channel into a single-slot buffer with
//! TTL + cooldown. The buffer's content is appended to the idle-speech
//! prompt's `context` string, then cleared. See `PRD/rolo-passive-perception.md`.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

pub mod buffer;
pub mod downloads;
pub mod filters;
pub mod foreground;
pub mod trash;

pub use buffer::PerceptionBuffer;

pub type SharedPerceptionBuffer = Arc<Mutex<PerceptionBuffer>>;

/// Per-producer liveness heartbeat. Each producer thread stamps the current
/// unix-millis at the top of its main loop; the Command Center's Perception
/// probes read these atomics to decide Green/Amber/Red. A heartbeat that
/// stays at 0 means the producer failed to spawn or bailed before its first
/// loop iteration (e.g., TCC denial on `~/Downloads`).
///
/// Relaxed ordering matches the drag-watch heartbeat — the probe only cares
/// about a recent value, not a memory-ordering relationship.
pub type PerceptionHeartbeat = Arc<AtomicI64>;

#[derive(Default)]
pub struct PerceptionHeartbeats {
    pub trash: PerceptionHeartbeat,
    pub downloads: PerceptionHeartbeat,
    pub foreground: PerceptionHeartbeat,
}

pub type SharedPerceptionHeartbeats = Arc<PerceptionHeartbeats>;

/// Stamp the current unix-millis into the heartbeat. Producers call this once
/// per loop iteration. `SystemTime` (not `chrono::Local::now()`) is used to
/// avoid the per-tick libc timezone lookup the foreground poller would
/// otherwise rack up every 2 seconds.
pub fn stamp_heartbeat(beat: &PerceptionHeartbeat) {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    beat.store(now_ms, Ordering::Relaxed);
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // `ext` and `size_bytes` are captured for future consumers.
pub enum PerceptionEvent {
    /// File added to ~/Downloads (debounced 2s, .crdownload/.part filtered).
    DownloadAdded {
        filename: String,
        ext: String,
        size_bytes: u64,
    },

    /// File moved to ~/.Trash by someone other than Rolo. `original_parent`
    /// is best-effort: Finder doesn't set `kMDItemWhereFroms` on its own
    /// trash operations, so it's `None` for most user-initiated trashes.
    TrashAdded {
        filename: String,
        original_parent: Option<String>,
    },

    /// Trash transitioned from N>=1 items to 0 (user-emptied).
    TrashEmptied { approximate_count: u32 },

    /// Frontmost app changed and was stable for 10 seconds.
    ForegroundChanged {
        app_name: String,
        window_title: Option<String>,
    },
}

impl PerceptionEvent {
    /// Render to the one-line context string for the prompt.
    /// Returns `None` if a privacy filter trips — the event is silently dropped.
    pub fn render(&self) -> Option<String> {
        match self {
            PerceptionEvent::DownloadAdded { filename, .. } => {
                if !filters::is_filename_safe(filename) {
                    return None;
                }
                Some(format!("Event: file added to Downloads — {}", filename))
            }
            PerceptionEvent::TrashAdded {
                filename,
                original_parent,
            } => {
                if !filters::is_filename_safe(filename) {
                    return None;
                }
                match original_parent {
                    Some(parent) => Some(format!(
                        "Event: file moved to Recycling Bin from {} — {}",
                        parent, filename
                    )),
                    None => Some(format!("Event: file moved to Recycling Bin — {}", filename)),
                }
            }
            PerceptionEvent::TrashEmptied { approximate_count } => Some(format!(
                "Event: trash emptied (about {} items) — user organizing",
                approximate_count
            )),
            PerceptionEvent::ForegroundChanged {
                app_name,
                window_title,
            } => {
                if filters::is_app_sensitive(app_name) {
                    return None;
                }
                match window_title {
                    Some(title) if !title.is_empty() => {
                        Some(format!("Event: foreground app — {} — {}", app_name, title))
                    }
                    _ => Some(format!("Event: foreground app — {}", app_name)),
                }
            }
        }
    }
}

/// Holds the receiver, producer-thread join handles, and the per-producer
/// heartbeat atomics. The receiver is moved into the tick thread closure (it
/// cannot be `app.manage`'d — `mpsc::Receiver` is `!Sync`). The join handles
/// are held only to keep the threads from being considered detached; they are
/// never joined explicitly. `heartbeats` is shared with Tauri-managed state so
/// the Command Center's Perception probes can read each producer's liveness.
pub struct PerceptionHandles {
    pub rx: mpsc::Receiver<PerceptionEvent>,
    pub heartbeats: SharedPerceptionHeartbeats,
    _trash: Option<thread::JoinHandle<()>>,
    _downloads: Option<thread::JoinHandle<()>>,
    _foreground: Option<thread::JoinHandle<()>>,
}

/// Construct the channel and spawn all three producers. Producers that fail
/// to start (TCC denial, NSWorkspace error, etc.) leave the corresponding
/// `Option<JoinHandle>` as `None`; other producers and the receiver still work.
/// Producers that fail before their first loop iteration leave their heartbeat
/// at 0 — the Perception tab surfaces this as a Red probe.
pub fn spawn_all() -> PerceptionHandles {
    let (tx, rx) = mpsc::channel::<PerceptionEvent>();
    let heartbeats: SharedPerceptionHeartbeats = Arc::new(PerceptionHeartbeats::default());

    let trash_handle = trash::spawn(tx.clone(), Arc::clone(&heartbeats.trash));
    let downloads_handle = downloads::spawn(tx.clone(), Arc::clone(&heartbeats.downloads));
    let foreground_handle = foreground::spawn(tx, Arc::clone(&heartbeats.foreground));

    PerceptionHandles {
        rx,
        heartbeats,
        _trash: trash_handle,
        _downloads: downloads_handle,
        _foreground: foreground_handle,
    }
}

/// Lock-and-take helper used at both idle-speech LLM-call sites in `tick.rs`.
/// Appends a single rendered perception line to `context` if the buffer has a
/// fresh, renderable event. Privacy-filtered events still start the cooldown
/// clock — they count as a "recent observation Rolo chose not to share."
pub fn append_perception_to_context(context: &mut String, buffer: &SharedPerceptionBuffer) {
    let mut buf = match buffer.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(ev) = buf.take_for_prompt(std::time::Instant::now()) {
        if let Some(line) = ev.render() {
            log::info!("[Rolo perception] append to prompt: {}", line);
            // v2 contract: perception lands as its own bracketed line between
            // the [State: ...] header and the blank-line/<idle> body. Matches
            // the training distribution's bracketed-annotation grammar (see
            // data/finetune/sft-v2/runtime_contract.md §2).
            if !context.is_empty() {
                context.push('\n');
            }
            context.push_str("[Observation: ");
            context.push_str(&line);
            context.push(']');
        } else {
            log::info!("[Rolo perception] event taken but rendered None (privacy-filtered)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_download() {
        let ev = PerceptionEvent::DownloadAdded {
            filename: "report.pdf".into(),
            ext: "pdf".into(),
            size_bytes: 524288,
        };
        assert_eq!(
            ev.render().unwrap(),
            "Event: file added to Downloads — report.pdf"
        );
    }

    #[test]
    fn render_download_filtered() {
        let ev = PerceptionEvent::DownloadAdded {
            filename: "my_passwords.txt".into(),
            ext: "txt".into(),
            size_bytes: 0,
        };
        assert!(ev.render().is_none());
    }

    #[test]
    fn render_trash_add_with_parent() {
        let ev = PerceptionEvent::TrashAdded {
            filename: "notes.md".into(),
            original_parent: Some("Documents".into()),
        };
        assert_eq!(
            ev.render().unwrap(),
            "Event: file moved to Recycling Bin from Documents — notes.md"
        );
    }

    #[test]
    fn render_trash_add_without_parent() {
        let ev = PerceptionEvent::TrashAdded {
            filename: "notes.md".into(),
            original_parent: None,
        };
        assert_eq!(
            ev.render().unwrap(),
            "Event: file moved to Recycling Bin — notes.md"
        );
    }

    #[test]
    fn render_trash_emptied() {
        let ev = PerceptionEvent::TrashEmptied {
            approximate_count: 8,
        };
        assert_eq!(
            ev.render().unwrap(),
            "Event: trash emptied (about 8 items) — user organizing"
        );
    }

    #[test]
    fn render_foreground_with_title() {
        let ev = PerceptionEvent::ForegroundChanged {
            app_name: "Xcode".into(),
            window_title: Some("MyApp.xcodeproj".into()),
        };
        assert_eq!(
            ev.render().unwrap(),
            "Event: foreground app — Xcode — MyApp.xcodeproj"
        );
    }

    #[test]
    fn render_foreground_no_title() {
        let ev = PerceptionEvent::ForegroundChanged {
            app_name: "Safari".into(),
            window_title: None,
        };
        assert_eq!(ev.render().unwrap(), "Event: foreground app — Safari");
    }

    #[test]
    fn render_foreground_sensitive_app() {
        let ev = PerceptionEvent::ForegroundChanged {
            app_name: "1Password".into(),
            window_title: None,
        };
        assert!(ev.render().is_none());
    }

    #[test]
    fn append_perception_appends_one_line_and_consumes() {
        let buf: SharedPerceptionBuffer = Arc::new(Mutex::new(PerceptionBuffer::default()));
        buf.lock().unwrap().push(
            PerceptionEvent::DownloadAdded {
                filename: "report.pdf".into(),
                ext: "pdf".into(),
                size_bytes: 1,
            },
            std::time::Instant::now(),
        );

        let mut ctx = String::from("baseline context");
        append_perception_to_context(&mut ctx, &buf);
        // v2 contract: rendered as its own bracketed line on a new row.
        assert!(
            ctx.ends_with("\n[Observation: Event: file added to Downloads — report.pdf]"),
            "got: {ctx}"
        );
        assert!(ctx.contains("baseline context"));

        // Buffer is consumed — second call should be a no-op.
        let before = ctx.clone();
        append_perception_to_context(&mut ctx, &buf);
        assert_eq!(ctx, before);
    }

    #[test]
    fn append_perception_no_op_on_empty_buffer() {
        let buf: SharedPerceptionBuffer = Arc::new(Mutex::new(PerceptionBuffer::default()));
        let mut ctx = String::from("baseline");
        append_perception_to_context(&mut ctx, &buf);
        assert_eq!(ctx, "baseline");
    }

    #[test]
    fn append_perception_filtered_event_still_starts_cooldown() {
        let buf: SharedPerceptionBuffer = Arc::new(Mutex::new(PerceptionBuffer::default()));
        // A privacy-filtered event still consumes the slot when taken.
        buf.lock().unwrap().push(
            PerceptionEvent::DownloadAdded {
                filename: "my_passwords.txt".into(),
                ext: "txt".into(),
                size_bytes: 0,
            },
            std::time::Instant::now(),
        );
        let mut ctx = String::from("base");
        append_perception_to_context(&mut ctx, &buf);
        // Filtered → context unchanged.
        assert_eq!(ctx, "base");
        // But buffer is now empty (slot taken) AND cooldown is active.
        let g = buf.lock().unwrap();
        // Pushing again while in cooldown should be a no-op.
        drop(g);
        buf.lock().unwrap().push(
            PerceptionEvent::DownloadAdded {
                filename: "safe.pdf".into(),
                ext: "pdf".into(),
                size_bytes: 1,
            },
            std::time::Instant::now(),
        );
        // Cooldown blocks this push.
        let mut ctx2 = String::from("");
        append_perception_to_context(&mut ctx2, &buf);
        assert_eq!(ctx2, "");
    }
}
