//! Pre-router (PRD/rolo-tool-layer.md §5.4 / T2).
//!
//! Cheap, deterministic regex-only first pass over user input. Catches the
//! patterns we don't want to spend a Gemma call on — short utterances,
//! greetings, classic mood/memory/state probes — and routes them straight
//! to a Speak or a specific tool. The LLM router (T4) only runs on
//! `Unknown` so we save ~150 ms on the common path.
//!
//! Why not LLM-only: the smallest greeting prompt costs ~200 ms even with
//! `format` constrained. A regex test is microseconds. With `ROLO_DISPATCHER_
//! ENABLED=true` Rolo answers "hi" without ever entering Ollama.
//!
//! Rule order matters — first match wins. Rule 5 ("single short word →
//! Speak") is intentionally placed AFTER 1–4 so a bare `hi` flows through
//! rule 1 (which keeps the door open for that branch to grow tool calls
//! later). Rule 5 is the catchall for one-syllable utterances that rule 1's
//! whitelist misses (`oh`, `ok`, `ya`).
//!
//! `state` is unused in v1 but kept in the signature so T3/T4 can read it
//! without churning every callsite. State-aware rules (e.g. "if Rolo is
//! Sleeping and the input is short → Speak the sleepy variant") are an
//! easy follow-up.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::{json, Value};

use crate::state_snapshot::StateSnapshot;

/// What the pre-router decides to do with a given input.
///
/// `Tool` carries the resolved tool name and a JSON args object so the
/// dispatcher (T3) can call `registry.get(name).invoke(args, ctx)` without
/// having to know which rule fired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouterDecision {
    Tool { name: &'static str, args: Value },
    Speak,
    Unknown,
}

// ---------------------------------------------------------------------------
// Compiled regex singletons. Each pattern compiles once on first call and is
// reused for the lifetime of the process. Matches the OnceLock style used
// by the T1 tools (see tools/get_mood_state.rs::schema_static).
// ---------------------------------------------------------------------------

fn re_greeting() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)^(hi|hey|hello|yo|sup)\b").expect("greeting regex compiles"))
}

fn re_mood_probe() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)how are you|how's it going|you ok").expect("mood-probe regex compiles")
    })
}

fn re_memory_probe() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        // The `what (did|do) (i|we)` arm is what catches phrases like
        // "what did we do last summer" — note this overlaps with the state
        // probe's "what are you doing", but rule 3 fires first so memory
        // wins, which is correct: the user is asking about *us*, not Rolo.
        Regex::new(r"(?i)remember|do you know|what (did|do) (i|we)|last time")
            .expect("memory-probe regex compiles")
    })
}

fn re_state_probe() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)what are you doing|what's up").expect("state-probe regex compiles")
    })
}

fn re_weather() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)\b(weather|raining|snowing|sunny|cloudy|forecast|temperature|how (hot|cold|warm))\b")
            .expect("weather regex compiles")
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the pre-router rules in order; first match wins.
///
/// `state` is reserved for future state-aware rules (T3/T4 may use it to
/// gate certain branches on PetState or mood). v1 ignores it.
pub fn pre_route(input: &str, _state: &StateSnapshot) -> RouterDecision {
    let trimmed = input.trim();

    // Rule 1: Greeting.
    if re_greeting().is_match(trimmed) {
        return RouterDecision::Speak;
    }

    // Rule 2: Mood probe.
    if re_mood_probe().is_match(trimmed) {
        return RouterDecision::Tool {
            name: "get_mood_state",
            args: json!({}),
        };
    }

    // Rule 3: Memory probe. We pass the original `input` (post-trim) as
    // the query so the vault searcher gets the user's exact phrasing —
    // BM25 is sensitive to it.
    if re_memory_probe().is_match(trimmed) {
        return RouterDecision::Tool {
            name: "search_vault",
            args: json!({ "query": trimmed }),
        };
    }

    // Rule 4: State probe.
    if re_state_probe().is_match(trimmed) {
        return RouterDecision::Tool {
            name: "get_pet_state",
            args: json!({}),
        };
    }

    // Rule 4.5: Weather utterance.
    if re_weather().is_match(trimmed) {
        return RouterDecision::Tool {
            name: "get_weather",
            args: json!({}),
        };
    }

    // Rule 5: Single short word (≤3 chars, no whitespace) → Speak.
    // Catchall for greetings/acks rule 1 missed: "ok", "ya", "oh", "no".
    if !trimmed.is_empty()
        && trimmed.chars().count() <= 3
        && !trimmed.chars().any(|c| c.is_whitespace())
    {
        return RouterDecision::Speak;
    }

    // Rule 6: Default — let the LLM router (T4) take it.
    RouterDecision::Unknown
}

// ---------------------------------------------------------------------------
// LLM router (Pass 1) — envelope, schema, system prompt, projection.
//
// PRD §5.5: Ollama's `format` parameter takes a JSON Schema; the assistant
// must produce a JSON object matching it. We then deserialize that object
// into `RouterEnvelope` and project it onto the same `RouterDecision` enum
// the pre-router uses, so the dispatcher's downstream code path is uniform
// for both routers.
//
// Why a fixed `tool` enum in the schema: small models (Gemma 3 4B) under
// `format` are markedly less likely to hallucinate a tool name when the
// schema enumerates the valid set up front. The envelope-to-decision
// projection re-checks the name against the dynamic registry anyway, so
// adding a tool requires touching both the schema and the projection — that
// coupling is intentional.
// ---------------------------------------------------------------------------

/// Parsed Pass-1 router envelope — produced by the LLM under grammar
/// constraint (Ollama `format`). The dispatcher passes this through
/// `envelope_to_decision` to get a `RouterDecision` it already knows how to
/// dispatch.
#[derive(Debug, serde::Deserialize)]
pub struct RouterEnvelope {
    pub action: String,
    pub tool: Option<String>,
    pub args: Option<serde_json::Value>,
    pub reason: Option<String>,
}

/// Why an envelope failed the projection step. Each variant maps cleanly
/// to a `dev_log` outcome label so telemetry can distinguish "model said
/// something we can't act on" from "model said something invalid".
#[derive(Debug)]
pub enum EnvelopeError {
    /// `action` was something other than `tool` or `speak`.
    UnknownAction(String),
    /// `action == "tool"` but no tool name was supplied.
    MissingToolName,
    /// `action == "tool"` but the tool name isn't one of the v1 three.
    UnknownTool(String),
    /// `action == "tool"`, `tool == "search_vault"`, but `args.query` was
    /// missing/empty/non-string.
    MissingQuery,
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvelopeError::UnknownAction(a) => write!(f, "unknown action: {a}"),
            EnvelopeError::MissingToolName => write!(f, "tool action missing tool name"),
            EnvelopeError::UnknownTool(t) => write!(f, "unknown tool: {t}"),
            EnvelopeError::MissingQuery => write!(f, "search_vault missing args.query"),
        }
    }
}

impl std::error::Error for EnvelopeError {}

/// Returns the JSON Schema fed to Ollama's `format` parameter. Cached behind
/// `OnceLock` because Ollama copies the schema on every call and we don't
/// want to re-parse it every dispatch.
pub fn router_format_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["tool", "speak"] },
                "tool":   { "type": "string", "enum": ["search_vault", "get_mood_state", "get_pet_state", "get_weather"] },
                "args":   { "type": "object" },
                "reason": { "type": "string", "maxLength": 80 }
            },
            "required": ["action"]
        })
    })
}

/// The system prompt fed to the LLM router. Inlines the registry's tool
/// catalog so adding a tool only requires touching `tools/mod.rs` +
/// `router_format_schema` (the schema's enum), never this prompt.
///
/// Token budget: ~250–400 tokens. The catalog block already runs ~150–250
/// tokens for the v1 three; the rule list and examples consume the rest.
pub fn router_system_prompt(registry: &super::ToolRegistry) -> String {
    let catalog = registry.render_for_system_prompt();
    format!(
        "You are Rolo's input router. Decide whether the user's message needs a \
tool call before Rolo replies, or whether Rolo can speak directly.\n\
\n\
{catalog}\n\
Rules:\n\
- Return JSON matching the supplied schema. No prose, no extra keys.\n\
- action=\"tool\" only when one of the listed tools clearly applies. If you \
are not sure, return action=\"speak\".\n\
- search_vault is for questions about the user's own past notes, conversations, \
or events — anything that requires looking back at history. Always supply \
args.query as a non-empty string copied from the user's message.\n\
- get_mood_state is for questions about Rolo's current feelings, energy, \
hunger, social need.\n\
- get_pet_state is for questions about what Rolo is doing right now \
(activity, position, screen, battery).\n\
- Never invent a tool name. Never put free-form text under any field.\n\
- If multiple tools could apply, prefer search_vault for memory questions; \
otherwise prefer the more specific tool over the more general.\n\
\n\
<example>\n\
user: do you remember the wedding\n\
output: {{\"action\":\"tool\",\"tool\":\"search_vault\",\"args\":{{\"query\":\"the wedding\"}}}}\n\
</example>\n\
\n\
<example>\n\
user: tell me a joke\n\
output: {{\"action\":\"speak\"}}\n\
</example>\n\
\n\
<example>\n\
user: are you hungry right now\n\
output: {{\"action\":\"tool\",\"tool\":\"get_mood_state\"}}\n\
</example>\n"
    )
}

/// Validates a parsed envelope and projects it onto the dispatcher's
/// `RouterDecision`. The dispatcher treats `Err(_)` here as "fall through to
/// legacy" — the envelope was syntactically valid (passed the `format`
/// schema) but didn't map to anything actionable.
pub fn envelope_to_decision(env: &RouterEnvelope) -> Result<RouterDecision, EnvelopeError> {
    match env.action.as_str() {
        "speak" => Ok(RouterDecision::Speak),
        "tool" => {
            let tool_name = env.tool.as_deref().ok_or(EnvelopeError::MissingToolName)?;
            match tool_name {
                "search_vault" => {
                    // search_vault requires a non-empty `query` string.
                    let query = env
                        .args
                        .as_ref()
                        .and_then(|a| a.get("query"))
                        .and_then(|q| q.as_str())
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .ok_or(EnvelopeError::MissingQuery)?;
                    Ok(RouterDecision::Tool {
                        name: "search_vault",
                        args: json!({ "query": query }),
                    })
                }
                "get_mood_state" => Ok(RouterDecision::Tool {
                    name: "get_mood_state",
                    args: env.args.clone().unwrap_or_else(|| json!({})),
                }),
                "get_pet_state" => Ok(RouterDecision::Tool {
                    name: "get_pet_state",
                    args: env.args.clone().unwrap_or_else(|| json!({})),
                }),
                "get_weather" => Ok(RouterDecision::Tool {
                    name: "get_weather",
                    args: env.args.clone().unwrap_or_else(|| json!({})),
                }),
                other => Err(EnvelopeError::UnknownTool(other.to_string())),
            }
        }
        other => Err(EnvelopeError::UnknownAction(other.to_string())),
    }
}

/// Telemetry label for a decision. The dispatcher (T3) and `dev_log` use
/// this to record which branch a given input took without leaking arg
/// payloads (which can contain user PII) into logs.
pub fn pre_route_decision_name(d: &RouterDecision) -> &'static str {
    match d {
        RouterDecision::Speak => "speak",
        RouterDecision::Tool { name, .. } => match *name {
            "search_vault" => "tool:search_vault",
            "get_mood_state" => "tool:get_mood_state",
            "get_pet_state" => "tool:get_pet_state",
            "get_weather" => "tool:get_weather",
            // Future tools will need a new arm; pre_route only emits the
            // three v1 tools so this is unreachable in v1.
            _ => "tool:unknown",
        },
        RouterDecision::Unknown => "unknown",
    }
}

// ---------------------------------------------------------------------------
// Smoke test (PRD T2 verification block).
//
// Lives inside the module so it can build a `StateSnapshot` via
// `from_parts`, which is `pub` but parameterised — staying in-tree means
// we don't have to add a public `for_test()` constructor on `StateSnapshot`
// just for this test. The fixture file is located via `CARGO_MANIFEST_DIR`
// so the path is robust to where `cargo test` is invoked from.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mood::MoodState;
    use crate::state_machine::PetState;
    use chrono::TimeZone;

    /// Build a stable snapshot for tests. v1 pre_route ignores it, so any
    /// well-formed snapshot is fine — we pin to a fixed wall-clock so this
    /// stays deterministic if a future rule starts reading time-of-day.
    fn fixture_snapshot() -> StateSnapshot {
        let now = chrono::Local
            .with_ymd_and_hms(2026, 5, 6, 14, 0, 0)
            .single()
            .expect("fixture timestamp is unambiguous");
        let mood = MoodState {
            energy_cached: MoodState::compute_energy_from_clock(now),
            social: 0.5,
            happiness: 0.5,
            sass_level: 0.0,
            last_fed_secs: 3600,
            ..MoodState::default()
        };
        StateSnapshot::from_parts(&mood, PetState::Idle, 0, false, now)
    }

    // ---- Targeted unit tests for each rule ------------------------------

    #[test]
    fn rule1_greeting_routes_to_speak() {
        let s = fixture_snapshot();
        for input in ["hi", "Hey there", "HELLO Rolo", "yo", "sup buddy"] {
            assert_eq!(
                pre_route(input, &s),
                RouterDecision::Speak,
                "input: {input}"
            );
        }
    }

    #[test]
    fn rule2_mood_probe_routes_to_get_mood_state() {
        let s = fixture_snapshot();
        let d = pre_route("how are you doing today", &s);
        assert!(matches!(
            d,
            RouterDecision::Tool {
                name: "get_mood_state",
                ..
            }
        ));
    }

    #[test]
    fn rule3_memory_probe_routes_to_search_vault_with_query() {
        let s = fixture_snapshot();
        let input = "remember when we went hiking";
        match pre_route(input, &s) {
            RouterDecision::Tool { name, args } => {
                assert_eq!(name, "search_vault");
                assert_eq!(args["query"], input);
            }
            other => panic!("expected search_vault tool, got {other:?}"),
        }
    }

    #[test]
    fn rule3_memory_probe_what_did_we_overlaps_state_but_memory_wins() {
        // "do you know what we did last summer" hits both rule 3
        // ("do you know") and the "what (did|do) (i|we)" arm. Either way
        // it should land on search_vault, NOT on the state probe (rule 4).
        let s = fixture_snapshot();
        let d = pre_route("do you know what we did last summer", &s);
        assert!(
            matches!(
                d,
                RouterDecision::Tool {
                    name: "search_vault",
                    ..
                }
            ),
            "expected search_vault, got {d:?}"
        );
    }

    #[test]
    fn rule4_state_probe_routes_to_get_pet_state() {
        let s = fixture_snapshot();
        let d = pre_route("what are you doing", &s);
        assert!(matches!(
            d,
            RouterDecision::Tool {
                name: "get_pet_state",
                ..
            }
        ));
    }

    #[test]
    fn rule5_short_utterance_routes_to_speak() {
        let s = fixture_snapshot();
        for input in ["ok", "oh", "ya", "no"] {
            assert_eq!(
                pre_route(input, &s),
                RouterDecision::Speak,
                "input: {input}"
            );
        }
    }

    #[test]
    fn rule5_does_not_eat_short_with_whitespace() {
        // "i am" is 4 chars but contains whitespace — rule 5's no-whitespace
        // clause keeps it from matching, so it falls to Unknown.
        let s = fixture_snapshot();
        assert_eq!(pre_route("i am", &s), RouterDecision::Unknown);
    }

    #[test]
    fn rule6_default_is_unknown() {
        let s = fixture_snapshot();
        assert_eq!(
            pre_route("tell me a story about dragons", &s),
            RouterDecision::Unknown
        );
    }

    #[test]
    fn decision_name_labels_are_stable() {
        assert_eq!(pre_route_decision_name(&RouterDecision::Speak), "speak");
        assert_eq!(pre_route_decision_name(&RouterDecision::Unknown), "unknown");
        assert_eq!(
            pre_route_decision_name(&RouterDecision::Tool {
                name: "search_vault",
                args: json!({})
            }),
            "tool:search_vault"
        );
        assert_eq!(
            pre_route_decision_name(&RouterDecision::Tool {
                name: "get_mood_state",
                args: json!({})
            }),
            "tool:get_mood_state"
        );
        assert_eq!(
            pre_route_decision_name(&RouterDecision::Tool {
                name: "get_pet_state",
                args: json!({})
            }),
            "tool:get_pet_state"
        );
        assert_eq!(
            pre_route_decision_name(&RouterDecision::Tool {
                name: "get_weather",
                args: json!({})
            }),
            "tool:get_weather"
        );
    }

    #[test]
    fn rule4_5_weather_routes_to_get_weather() {
        let s = fixture_snapshot();
        for input in [
            "what's the weather?",
            "is it raining?",
            "how hot is it outside?",
            "is it sunny today",
        ] {
            assert!(
                matches!(
                    pre_route(input, &s),
                    RouterDecision::Tool {
                        name: "get_weather",
                        ..
                    }
                ),
                "input: {input}"
            );
        }
    }

    #[test]
    fn envelope_round_trip_get_weather() {
        let env = RouterEnvelope {
            action: "tool".into(),
            tool: Some("get_weather".into()),
            args: None,
            reason: Some("user asked about weather".into()),
        };
        match envelope_to_decision(&env).unwrap() {
            RouterDecision::Tool { name, .. } => assert_eq!(name, "get_weather"),
            other => panic!("expected get_weather tool, got {other:?}"),
        }
    }

    // ---- T4 envelope round-trip (no live Ollama) ------------------------

    #[test]
    fn envelope_round_trip_speak() {
        let env = RouterEnvelope {
            action: "speak".into(),
            tool: None,
            args: None,
            reason: Some("just chatting".into()),
        };
        assert_eq!(envelope_to_decision(&env).unwrap(), RouterDecision::Speak);
    }

    #[test]
    fn envelope_round_trip_search_vault_with_query() {
        let env = RouterEnvelope {
            action: "tool".into(),
            tool: Some("search_vault".into()),
            args: Some(json!({ "query": "the wedding" })),
            reason: None,
        };
        match envelope_to_decision(&env).unwrap() {
            RouterDecision::Tool { name, args } => {
                assert_eq!(name, "search_vault");
                assert_eq!(args["query"], "the wedding");
            }
            other => panic!("expected search_vault tool, got {other:?}"),
        }
    }

    #[test]
    fn envelope_round_trip_search_vault_missing_query_errors() {
        let env = RouterEnvelope {
            action: "tool".into(),
            tool: Some("search_vault".into()),
            args: Some(json!({})),
            reason: None,
        };
        assert!(matches!(
            envelope_to_decision(&env),
            Err(EnvelopeError::MissingQuery)
        ));

        // Empty/whitespace-only also fails.
        let env = RouterEnvelope {
            action: "tool".into(),
            tool: Some("search_vault".into()),
            args: Some(json!({ "query": "   " })),
            reason: None,
        };
        assert!(matches!(
            envelope_to_decision(&env),
            Err(EnvelopeError::MissingQuery)
        ));
    }

    #[test]
    fn envelope_round_trip_get_mood_state_no_args() {
        let env = RouterEnvelope {
            action: "tool".into(),
            tool: Some("get_mood_state".into()),
            args: None,
            reason: None,
        };
        match envelope_to_decision(&env).unwrap() {
            RouterDecision::Tool { name, args } => {
                assert_eq!(name, "get_mood_state");
                assert_eq!(args, json!({}));
            }
            other => panic!("expected get_mood_state tool, got {other:?}"),
        }
    }

    #[test]
    fn envelope_round_trip_get_pet_state_with_empty_args() {
        let env = RouterEnvelope {
            action: "tool".into(),
            tool: Some("get_pet_state".into()),
            args: Some(json!({})),
            reason: Some("status check".into()),
        };
        match envelope_to_decision(&env).unwrap() {
            RouterDecision::Tool { name, .. } => assert_eq!(name, "get_pet_state"),
            other => panic!("expected get_pet_state tool, got {other:?}"),
        }
    }

    #[test]
    fn envelope_round_trip_unknown_action_errors() {
        let env = RouterEnvelope {
            action: "shrug".into(),
            tool: None,
            args: None,
            reason: None,
        };
        assert!(matches!(
            envelope_to_decision(&env),
            Err(EnvelopeError::UnknownAction(_))
        ));
    }

    #[test]
    fn envelope_round_trip_tool_action_missing_name_errors() {
        let env = RouterEnvelope {
            action: "tool".into(),
            tool: None,
            args: None,
            reason: None,
        };
        assert!(matches!(
            envelope_to_decision(&env),
            Err(EnvelopeError::MissingToolName)
        ));
    }

    #[test]
    fn envelope_round_trip_unknown_tool_errors() {
        let env = RouterEnvelope {
            action: "tool".into(),
            tool: Some("hallucinated_tool".into()),
            args: None,
            reason: None,
        };
        assert!(matches!(
            envelope_to_decision(&env),
            Err(EnvelopeError::UnknownTool(_))
        ));
    }

    #[test]
    fn envelope_deserializes_from_canonical_json() {
        // Mirrors what Ollama returns under `format`: a JSON string parsed
        // into a Value. Make sure serde sees it cleanly.
        let raw =
            r#"{"action":"tool","tool":"search_vault","args":{"query":"x"},"reason":"memory"}"#;
        let env: RouterEnvelope = serde_json::from_str(raw).expect("envelope parses");
        assert_eq!(env.action, "tool");
        assert_eq!(env.tool.as_deref(), Some("search_vault"));
    }

    #[test]
    fn router_format_schema_is_stable_singleton() {
        let a = router_format_schema();
        let b = router_format_schema();
        assert!(std::ptr::eq(a, b), "schema should be cached");
        assert_eq!(a["required"], json!(["action"]));
    }

    #[test]
    fn router_system_prompt_contains_each_tool_and_examples() {
        let registry = crate::tools::ToolRegistry::standard();
        let prompt = router_system_prompt(&registry);
        for name in [
            "search_vault",
            "get_mood_state",
            "get_pet_state",
            "get_weather",
        ] {
            assert!(
                prompt.contains(name),
                "system prompt missing tool name `{name}`"
            );
        }
        assert!(
            prompt.contains("<example>"),
            "system prompt should include at least one example block"
        );
        // Loose token budget: the PRD asks for ~250–400 tokens. At ~4
        // chars/token that's ≤1600 chars; we leave headroom by requiring
        // ≤2600 to allow the registry catalog to grow modestly. Bumped
        // from 2200 to 2600 when get_weather joined the v1 catalog — its
        // description carries the longest usage hint of any tool.
        assert!(
            prompt.len() <= 2600,
            "system prompt grew past 2600 chars ({}), would blow router budget",
            prompt.len()
        );
    }

    // ---- Smoke fixture (PRD T2: ≥15 inputs, ≥90% accuracy) --------------

    /// One row of the smoke fixture.
    #[derive(serde::Deserialize)]
    struct SmokeRow {
        input: String,
        expected: String,
    }

    /// PRD T2 smoke set: each fixture line carries the expected
    /// `pre_route_decision_name` output. We require ≥90% accuracy (target
    /// 100%, allow 1-line slack at 15 lines).
    #[test]
    fn pre_route_smoke() {
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("prerouter_smoke.jsonl");

        let raw = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", fixture_path.display()));

        let rows: Vec<SmokeRow> = raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .enumerate()
            .map(|(i, line)| {
                serde_json::from_str(line).unwrap_or_else(|e| {
                    panic!("fixture line {} is not valid JSON: {line:?} — {e}", i + 1)
                })
            })
            .collect();

        assert!(
            rows.len() >= 15,
            "fixture must have ≥15 rows, found {}",
            rows.len()
        );

        let snap = fixture_snapshot();
        let mut hits = 0usize;
        let mut misses: Vec<String> = Vec::new();
        for row in &rows {
            let decision = pre_route(&row.input, &snap);
            let got = pre_route_decision_name(&decision);
            if got == row.expected {
                hits += 1;
            } else {
                misses.push(format!(
                    "  - input={:?} expected={} got={}",
                    row.input, row.expected, got
                ));
            }
        }

        let total = rows.len();
        let accuracy = (hits as f64) / (total as f64);
        assert!(
            accuracy >= 0.90,
            "pre_route accuracy {:.0}% < 90% on smoke set ({}/{}). misses:\n{}",
            accuracy * 100.0,
            hits,
            total,
            misses.join("\n")
        );
    }
}
