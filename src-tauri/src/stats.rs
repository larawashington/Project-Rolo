//! Rolo's eating statistics — tracking his appetite over time.
//!
//! Persists to ~/.rolo/stats.json. This data powers the future hunger system.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub type SharedStats = Arc<Mutex<EatingStats>>;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EatingStats {
    pub total_bytes_eaten: u64,
    pub total_files_eaten: u64,
    pub last_fed: Option<String>,
}

impl EatingStats {
    pub fn load() -> Self {
        let path = Self::stats_path();
        match fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_else(|e| {
                eprintln!(
                    "[Rolo] Stats file corrupted — resetting to zero. \
                     He may have lost count of his meals: {}",
                    e
                );
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) {
        let path = Self::stats_path();
        if let Ok(json) = serde_json::to_string_pretty(self) {
            if let Err(e) = crate::vault::atomic::atomic_write(&path, json.as_bytes()) {
                eprintln!("[Rolo] Failed to save stats — his memory is failing: {}", e);
            }
        }
    }

    pub fn record_meal(&mut self, bytes: u64, files: usize) {
        self.total_bytes_eaten += bytes;
        self.total_files_eaten += files as u64;
        self.last_fed = Some(chrono::Utc::now().to_rfc3339());
        self.save();
    }

    fn stats_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".rolo")
            .join("stats.json")
    }
}
