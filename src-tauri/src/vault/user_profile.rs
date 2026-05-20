//! User-profile schema for the Memory panel (PRD §6, Command Center).
//!
//! `user_profile.json` lives at `<vault_root>/user_profile.json` and captures
//! the three Memory textareas:
//!   - `about_you` — free-form "who the user is"
//!   - `people_and_context` — names, roles, relationships
//!   - `how_rolo_should_respond` — voice / behavior preferences
//!
//! Empty strings mean "section not set"; a *missing* file means "no profile
//! yet" — `Vault::load_user_profile` returns `None` in that case so the UI
//! can distinguish "fresh install" from "user cleared all sections".
//!
//! Phase 6 only defines the schema and Vault API; the Memory tab UI lands in
//! Phase 7 (`PRD/rolo-command-center.md` Order of Operations step 12).

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};

/// The Memory panel's saved state. `version` lets future schema migrations
/// detect older files; bump it whenever the on-disk shape changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    #[serde(default = "default_user_profile_version")]
    pub version: u32,
    #[serde(default)]
    pub about_you: String,
    #[serde(default)]
    pub people_and_context: String,
    #[serde(default)]
    pub how_rolo_should_respond: String,
    #[serde(default = "default_user_profile_updated_at")]
    pub updated_at: DateTime<Local>,
}

fn default_user_profile_version() -> u32 {
    1
}

fn default_user_profile_updated_at() -> DateTime<Local> {
    Local::now()
}

impl Default for UserProfile {
    fn default() -> Self {
        Self {
            version: 1,
            about_you: String::new(),
            people_and_context: String::new(),
            how_rolo_should_respond: String::new(),
            updated_at: Local::now(),
        }
    }
}
