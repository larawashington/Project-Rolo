use std::fs;

use chrono::Local;
use desktop_pet_lib::vault::logger::ExperienceLogger;
use tempfile::TempDir;

#[test]
fn retention_sweep_removes_old_files_and_preserves_today() {
    let tmp = TempDir::new().unwrap();
    let events_dir = tmp.path().join("events");
    fs::create_dir_all(&events_dir).unwrap();

    // Plant a file far older than the 30-day retention window.
    let old_path = events_dir.join("2026-03-15.jsonl");
    fs::write(&old_path, b"{}\n").unwrap();

    let logger = ExperienceLogger::new(events_dir.clone());

    // Logger init opens today's file.
    let today = Local::now().date_naive();
    let today_path = events_dir.join(format!("{today}.jsonl"));
    assert!(today_path.exists(), "logger init should open today's file");

    let deleted = logger.retention_sweep();

    assert!(!old_path.exists(), "2026-03-15 file should be swept");
    assert_eq!(deleted, 1, "exactly one file should have been deleted");
    assert!(today_path.exists(), "today's file must survive the sweep");
}
