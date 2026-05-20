//! Registry of every tool Rolo can call.
//!
//! `ToolRegistry::standard()` wires the v1 four (`search_vault`,
//! `get_mood_state`, `get_pet_state`, `get_weather`). The registry is
//! constructed once in `lib.rs::run` and stored as managed Tauri state
//! (`Arc<ToolRegistry>`). T1 only uses this from the dev `dev_invoke_tool`
//! command; the dispatcher (T3) and chat path (T5) wire it into production
//! paths later.
//!
//! `render_for_system_prompt` builds the textual block T4 will inject into
//! the Pass-1 router prompt. T1 doesn't *use* the rendering, but the tests
//! pin its shape so T4 can rely on it.
//!
//! **GetWeather is held as a concrete `Arc<GetWeather>` field in addition
//! to the trait-object map.** That second handle is what Phase 8 needs:
//! `cc_weather_status` reads the cache; `lib.rs::setup` plumbs the
//! `AppHandle` in via `set_app_handle`; and the `rolo://weather-config-changed`
//! listener calls `invalidate_cache`. None of those callers can reach the
//! tool through the `dyn RoloTool` map, so the typed handle stays alongside.

use std::collections::HashMap;
use std::sync::Arc;

use super::get_mood_state::GetMoodState;
use super::get_pet_state::GetPetState;
use super::get_weather::GetWeather;
use super::search_vault::SearchVault;
use super::RoloTool;

pub struct ToolRegistry {
    tools: HashMap<&'static str, Arc<dyn RoloTool>>,
    /// Concrete handle to the weather tool. Stored alongside the trait
    /// object so Phase-8 callers (cache invalidator, status reader,
    /// `set_app_handle` plumbing) can reach typed methods that aren't on
    /// the `RoloTool` trait.
    get_weather: Arc<GetWeather>,
}

impl ToolRegistry {
    /// Build the v1 registry: the four tools the tool-layer PRD ships.
    pub fn standard() -> Self {
        let mut tools: HashMap<&'static str, Arc<dyn RoloTool>> = HashMap::new();
        let sv: Arc<dyn RoloTool> = Arc::new(SearchVault::new());
        tools.insert(sv.name(), sv);
        let gm: Arc<dyn RoloTool> = Arc::new(GetMoodState::new());
        tools.insert(gm.name(), gm);
        let gp: Arc<dyn RoloTool> = Arc::new(GetPetState::new());
        tools.insert(gp.name(), gp);
        let gw = Arc::new(GetWeather::new());
        tools.insert("get_weather", Arc::clone(&gw) as Arc<dyn RoloTool>);
        Self {
            tools,
            get_weather: gw,
        }
    }

    /// Typed handle to the weather tool. Lets the Command Center plumb in
    /// the `AppHandle` post-construction and lets the cache-invalidation
    /// listener call `invalidate_cache` when settings change.
    pub fn get_weather(&self) -> &Arc<GetWeather> {
        &self.get_weather
    }

    /// Return the registry as a compact text block (~250–350 tokens) suitable
    /// for inlining into the Pass-1 router system prompt. Order is alphabetical
    /// so the rendered string is stable across runs.
    pub fn render_for_system_prompt(&self) -> String {
        let mut names: Vec<&&'static str> = self.tools.keys().collect();
        names.sort();
        let mut out = String::from("Available tools:\n");
        for name in names {
            let tool = self
                .tools
                .get(name)
                .expect("name came from this map's keys");
            // Schema rendered as one compact line so the whole block stays
            // small. Pretty-printing here would blow the token budget.
            let schema_compact = tool.schema().to_string();
            out.push_str("- ");
            out.push_str(tool.name());
            out.push_str(": ");
            out.push_str(tool.description());
            out.push_str("\n  args: ");
            out.push_str(&schema_compact);
            out.push('\n');
        }
        out
    }

    pub fn get(&self, name: &str) -> Option<&dyn RoloTool> {
        self.tools.get(name).map(|a| a.as_ref())
    }

    /// Names of every registered tool, sorted alphabetically. Used by the dev
    /// surface to enumerate options.
    pub fn names(&self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = self.tools.keys().copied().collect();
        v.sort();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_registers_exactly_four_tools_with_expected_names() {
        let reg = ToolRegistry::standard();
        let mut names = reg.names();
        names.sort();
        assert_eq!(
            names,
            vec![
                "get_mood_state",
                "get_pet_state",
                "get_weather",
                "search_vault"
            ]
        );
    }

    #[test]
    fn render_for_system_prompt_includes_every_tool_and_stays_compact() {
        let reg = ToolRegistry::standard();
        let rendered = reg.render_for_system_prompt();
        for name in reg.names() {
            assert!(
                rendered.contains(name),
                "rendered prompt missing tool name `{}`: {}",
                name,
                rendered
            );
        }
        // Compact-budget check — tighter than the PRD's 250–350 token target,
        // measured in chars (≈4 chars/token rule of thumb gives ~1500 char cap).
        assert!(
            rendered.len() < 1500,
            "rendered prompt grew past 1500 chars ({}), would blow the router's token budget",
            rendered.len()
        );
    }

    #[test]
    fn get_returns_none_for_unknown_name() {
        let reg = ToolRegistry::standard();
        assert!(reg.get("does_not_exist").is_none());
    }

    #[test]
    fn get_weather_returns_same_arc_each_call() {
        let reg = ToolRegistry::standard();
        let a = Arc::clone(reg.get_weather());
        let b = Arc::clone(reg.get_weather());
        assert!(Arc::ptr_eq(&a, &b));
    }
}
