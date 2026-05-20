//! Gemma-callable tool layer (PRD/rolo-tool-layer.md, T1).
//!
//! Defines the `RoloTool` trait, the three v1 tools, and a `ToolRegistry`
//! that the dispatcher (T3) and the dev surface (`dev_invoke_tool`) use.
//!
//! Why this exists: today every bubble pre-stuffs five system-prompt slots
//! whether or not the input needs them. Tool calling lets Gemma decide
//! *when* to look something up — vault search runs only when the input is
//! memory-shaped, mood reads happen only on mood probes, etc. T1 lays the
//! foundation; T3 wires it into the dispatcher; the registry is otherwise
//! unused by production code in T1.
//!
//! The trait holds `pet: &'a Pet` (locked guard reference) rather than the
//! `Arc<Mutex<Pet>>` directly. This matches `StateSnapshot::capture`'s
//! signature and lets call sites lock once on the outside instead of having
//! every tool re-enter the mutex.

pub mod dispatcher;
pub mod get_mood_state;
pub mod get_pet_state;
pub mod get_weather;
pub mod registry;
pub mod router;
pub mod search_vault;
pub mod weather_endpoint;

use crate::commands::SharedMood;
use crate::state_machine::Pet;
use crate::state_snapshot::Clock;
use crate::vault::Vault;

pub use registry::ToolRegistry;

/// A pointer into a vault chunk that backed the natural-language summary.
/// Line numbers are reserved for a future chunker upgrade — `Chunk` does
/// not currently expose them, so we emit `0..0` and fix the schema later
/// when the vault stores line ranges.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Citation {
    pub path: String,
    pub line_start: usize,
    pub line_end: usize,
}

/// What a tool returns. `natural_language` is the form fed back into Pass 2's
/// prompt, never raw JSON. Showing a small model JSON in its own context
/// teaches it to mimic JSON in conversation; we never do that.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolOutput {
    pub natural_language: String,
    pub citations: Vec<Citation>,
}

/// Tool failures. `Unavailable` is non-fatal — the caller should fall back
/// to the legacy prompt assembler (or in dev, surface the message).
#[derive(Debug)]
pub enum ToolError {
    InvalidArgs(String),
    Unavailable(String),
    Internal(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::InvalidArgs(s) => write!(f, "invalid arguments: {}", s),
            ToolError::Unavailable(s) => write!(f, "tool unavailable: {}", s),
            ToolError::Internal(s) => write!(f, "internal error: {}", s),
        }
    }
}

impl std::error::Error for ToolError {}

/// Bundle of references every tool needs. Caller locks the pet mutex
/// outside and passes the guard ref in; tools therefore never block on
/// pet contention.
pub struct ToolContext<'a> {
    pub vault: &'a Vault,
    pub mood: &'a SharedMood,
    pub pet: &'a Pet,
    pub clock: &'a dyn Clock,
}

#[async_trait::async_trait]
pub trait RoloTool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn schema(&self) -> &'static serde_json::Value;
    async fn invoke(
        &self,
        args: &serde_json::Value,
        ctx: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError>;
}
