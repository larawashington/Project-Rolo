//! `get_weather` — return current local weather as a one-line summary.
//!
//! Mirrors the structure of `get_mood_state.rs`: empty schema, no args, a
//! single natural-language line out. The tool resolves the user's lat/lon
//! via ipwho.is (cached for the process lifetime) then queries Open-Meteo
//! for `current=temperature_2m,weather_code` in Fahrenheit. Result is
//! cached in-tool for 10 minutes; subsequent calls within that window
//! short-circuit to memory.
//!
//! Critically, `invoke` always returns `Ok(ToolOutput { ... })`. On any
//! network or parse failure we return `"weather unavailable."` as a
//! successful tool output so the dispatcher emits `tool:get_weather`
//! (rather than `tool_failed:get_weather`) and the LLM can adapt the
//! conversation gracefully without hallucinating data.
//!
//! See PRD/rolo-weather-tool.md for the full design.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::{RoloTool, ToolContext, ToolError, ToolOutput};
use crate::http::http_client;

use std::sync::Arc;

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

pub(crate) const GEOIP_URL: &str = "https://ipwho.is/";
pub(crate) const OPEN_METEO_BASE: &str = "https://api.open-meteo.com/v1/forecast";
pub(crate) const HTTP_TIMEOUT: Duration = Duration::from_secs(3);
pub(crate) const CACHE_TTL: Duration = Duration::from_secs(600);

// ---------------------------------------------------------------------------
// HTTP abstraction (mirrors Embedder pattern — trait-object boundary so tests
// can swap in a MockFetcher without touching the production wiring).
// ---------------------------------------------------------------------------

/// Why a fetch failed. Variants exist mainly so tests can simulate each
/// failure mode in isolation; the production code path collapses all of
/// them into the same `"weather unavailable."` user-facing string.
#[derive(Debug)]
pub(crate) enum FetchError {
    Timeout,
    /// DNS, connection refused, TLS handshake, etc.
    Network(String),
    /// Non-2xx response.
    Status(u16),
    /// Body received but `serde_json` couldn't parse it.
    Parse(String),
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FetchError::Timeout => write!(f, "timeout"),
            FetchError::Network(s) => write!(f, "network: {s}"),
            FetchError::Status(c) => write!(f, "http status {c}"),
            FetchError::Parse(s) => write!(f, "parse: {s}"),
        }
    }
}

#[async_trait::async_trait]
pub(crate) trait HttpFetcher: Send + Sync {
    async fn get_json(&self, url: &str) -> Result<Value, FetchError>;
}

/// Production fetcher: a single `reqwest::Client` with a 3s timeout
/// configured at construction. Built once per `GetWeather` and reused.
struct ReqwestFetcher {
    client: reqwest::Client,
}

#[async_trait::async_trait]
impl HttpFetcher for ReqwestFetcher {
    async fn get_json(&self, url: &str) -> Result<Value, FetchError> {
        let resp = self.client.get(url).send().await.map_err(|e| {
            if e.is_timeout() {
                FetchError::Timeout
            } else if let Some(code) = e.status() {
                FetchError::Status(code.as_u16())
            } else {
                FetchError::Network(e.to_string())
            }
        })?;

        let status = resp.status();
        if !status.is_success() {
            return Err(FetchError::Status(status.as_u16()));
        }

        resp.json::<Value>()
            .await
            .map_err(|e| FetchError::Parse(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// WMO weather code table (PRD §"WMO weather code table").
// Hardcoded match: no external dependency, total ~30 LOC, easy to audit.
// ---------------------------------------------------------------------------

fn weather_code_to_string(code: u32) -> &'static str {
    match code {
        0 => "clear sky",
        1..=3 => "partly cloudy",
        45 | 48 => "foggy",
        51 | 53 | 55 | 56 | 57 => "drizzling",
        61 | 63 | 65 | 66 | 67 | 80 | 81 | 82 => "raining",
        71 | 73 | 75 | 77 | 85 | 86 => "snowing",
        95 | 96 | 99 => "thunderstorms",
        _ => "overcast",
    }
}

// ---------------------------------------------------------------------------
// Tool implementation
// ---------------------------------------------------------------------------

pub struct GetWeather {
    /// Resolved (lat, lon). Set on first successful ipwho.is lookup; held
    /// for the lifetime of the process. Restart Rolo to refresh.
    location: Mutex<Option<(f64, f64)>>,
    /// Last rendered NL line plus the instant we fetched it. Treated as
    /// stale at `>= CACHE_TTL` so the boundary case (exactly 10 min) refetches.
    cache: Mutex<Option<(String, Instant)>>,
    http: Arc<dyn HttpFetcher>,
    /// Set once at app startup so the tool can read
    /// `CommandCenterSettings` per call without being constructed inside the
    /// Tauri setup closure (the registry is built before all handles are
    /// available). Tests leave this None and exercise the legacy Open-Meteo
    /// path only — the custom-endpoint branch is unit-tested separately in
    /// `weather_endpoint::tests`.
    app: Mutex<Option<tauri::AppHandle>>,
}

impl GetWeather {
    pub fn new() -> Self {
        let client = http_client(HTTP_TIMEOUT).expect("reqwest client builds with default config");
        Self {
            location: Mutex::new(None),
            cache: Mutex::new(None),
            http: Arc::new(ReqwestFetcher { client }),
            app: Mutex::new(None),
        }
    }

    /// Test-only constructor — wires a mock fetcher in place of the real
    /// `reqwest` client. Kept `pub(crate)` because no consumer outside this
    /// module needs it.
    #[cfg(test)]
    pub(crate) fn with_fetcher(http: Arc<dyn HttpFetcher>) -> Self {
        Self {
            location: Mutex::new(None),
            cache: Mutex::new(None),
            http,
            app: Mutex::new(None),
        }
    }

    /// Wire the Tauri app handle in after construction. The registry is
    /// built inside the setup closure before `app.manage()` calls finish, so
    /// the tool can't grab a handle in its constructor — we set it once,
    /// here, before `start_tick_loop` arms anything that might call the tool.
    pub fn set_app_handle(&self, app: tauri::AppHandle) {
        let mut guard = self
            .app
            .lock()
            .expect("weather app handle mutex not poisoned");
        *guard = Some(app);
    }

    /// Drop the cached NL line so the next `invoke` refetches. Wired up to
    /// the `rolo://weather-config-changed` event in `lib.rs::setup` so a
    /// successful `cc_save_settings` immediately invalidates whatever was
    /// fetched against the old endpoint.
    pub fn invalidate_cache(&self) {
        let mut guard = self.cache.lock().expect("weather cache mutex not poisoned");
        *guard = None;
    }

    /// Read the (line, fetched_at) pair without disturbing it. Used by
    /// `cc_weather_status` to render the Weather tab's status indicator.
    /// Returns `None` when no fetch has succeeded yet this session.
    pub fn cache_snapshot(&self) -> Option<(String, Instant)> {
        self.cache
            .lock()
            .expect("weather cache mutex not poisoned")
            .as_ref()
            .map(|(line, when)| (line.clone(), *when))
    }

    /// Try the user-configured custom weather endpoint. Returns `Some(line)`
    /// on success (already formatted in the canonical "72°F, partly cloudy."
    /// shape so the cache stores a string identical to what the Open-Meteo
    /// path would produce). Returns `None` on ANY error — the caller falls
    /// through to Open-Meteo silently per PRD AC #18.
    ///
    /// Coordinates come from the manual override when set; otherwise we use
    /// the cached IP-geo result. If neither is available we run the IP-geo
    /// lookup here on a best-effort basis (we'd do it again in the Open-Meteo
    /// branch, but the lookup itself is also cached process-wide so the
    /// second call is a memory read).
    async fn try_custom_endpoint(
        &self,
        settings: &crate::command_center::settings::CommandCenterSettings,
    ) -> Option<String> {
        use crate::tools::weather_endpoint::{parse_with_preset, substitute_template};

        let template = settings.weather.custom_url_template.as_str();
        if template.is_empty() {
            log::warn!(
                "[Rolo WEATHER] custom endpoint mode enabled but URL template is empty — falling back to Open-Meteo"
            );
            return None;
        }
        if let Err(e) = crate::command_center::settings::validate_url_template(template) {
            log::warn!(
                "[Rolo WEATHER] custom endpoint failed — falling back to Open-Meteo: {}",
                e
            );
            return None;
        }

        // Resolve coords for substitution. Manual location wins; otherwise
        // we re-use the process-wide IP-geo cache (and refresh it on miss).
        let (lat, lon) = if let Some(m) = settings.weather.manual_location.as_ref() {
            (m.lat, m.lon)
        } else {
            let cached = {
                let g = self
                    .location
                    .lock()
                    .expect("weather location mutex not poisoned");
                *g
            };
            match cached {
                Some(c) => c,
                None => match self.http.get_json(GEOIP_URL).await {
                    Ok(body) => {
                        let la = body.get("latitude").and_then(|v| v.as_f64());
                        let lo = body.get("longitude").and_then(|v| v.as_f64());
                        match (la, lo) {
                            (Some(la), Some(lo)) if !(la == 0.0 && lo == 0.0) => {
                                let mut g = self
                                    .location
                                    .lock()
                                    .expect("weather location mutex not poisoned");
                                *g = Some((la, lo));
                                (la, lo)
                            }
                            _ => {
                                log::warn!(
                                    "[Rolo WEATHER] custom endpoint failed — falling back to Open-Meteo: ipwho.is returned bad coords"
                                );
                                return None;
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "[Rolo WEATHER] custom endpoint failed — falling back to Open-Meteo: ipwho.is {}",
                            e
                        );
                        return None;
                    }
                },
            }
        };

        let url = substitute_template(template, lat, lon, &settings.weather.api_key);
        let body = match self.http.get_json(&url).await {
            Ok(v) => v,
            Err(e) => {
                log::warn!(
                    "[Rolo WEATHER] custom endpoint failed — falling back to Open-Meteo: {}",
                    e
                );
                return None;
            }
        };

        let jt = if settings.weather.custom_jsonpath_temp.is_empty() {
            None
        } else {
            Some(settings.weather.custom_jsonpath_temp.as_str())
        };
        let jc = if settings.weather.custom_jsonpath_condition.is_empty() {
            None
        } else {
            Some(settings.weather.custom_jsonpath_condition.as_str())
        };

        let parsed = match parse_with_preset(
            settings.weather.response_preset.as_wire_str(),
            &body,
            jt,
            jc,
        ) {
            Ok(p) => p,
            Err(e) => {
                log::warn!(
                    "[Rolo WEATHER] custom endpoint failed — falling back to Open-Meteo: {}",
                    e
                );
                return None;
            }
        };

        // Render with the same shape Open-Meteo produces so callers (and the
        // cache) can't tell them apart. Half-up rounding matches the existing
        // logic; range check rejects nonsense.
        let temp_int = (parsed.temp_f + 0.5).floor() as i32;
        if !(-100..=140).contains(&temp_int) {
            log::warn!(
                "[Rolo WEATHER] custom endpoint failed — falling back to Open-Meteo: temperature {}°F outside [-100, 140]",
                temp_int
            );
            return None;
        }
        // Normalize the condition: lowercased, trimmed. OpenWeatherMap is
        // already lowercase, WeatherAPI and Visual Crossing are title-case;
        // lowercasing keeps the rendered NL line consistent with the
        // Open-Meteo path ("partly cloudy", not "Partly cloudy").
        let condition = parsed.condition_text.trim().to_lowercase();
        Some(format!("{}°F, {}.", temp_int, condition))
    }
}

impl Default for GetWeather {
    fn default() -> Self {
        Self::new()
    }
}

fn schema_static() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true
        })
    })
}

const UNAVAILABLE: &str = "weather unavailable.";

fn unavailable_output() -> ToolOutput {
    ToolOutput {
        natural_language: UNAVAILABLE.to_string(),
        citations: Vec::new(),
    }
}

#[async_trait::async_trait]
impl RoloTool for GetWeather {
    fn name(&self) -> &'static str {
        "get_weather"
    }

    fn description(&self) -> &'static str {
        "Get current weather for the user's location. Use when the user mentions weather, asks what it's like outside, or when small talk could naturally include weather (greetings, mood check-ins, idle comments)."
    }

    fn schema(&self) -> &'static Value {
        schema_static()
    }

    async fn invoke(&self, _args: &Value, _ctx: &ToolContext<'_>) -> Result<ToolOutput, ToolError> {
        // 1. Cache hit?  `>= CACHE_TTL` means stale, so a cache exactly 10
        //    minutes old refetches — matches the PRD's edge case row.
        {
            let guard = self.cache.lock().expect("weather cache mutex not poisoned");
            if let Some((line, fetched_at)) = guard.as_ref() {
                if fetched_at.elapsed() < CACHE_TTL {
                    return Ok(ToolOutput {
                        natural_language: line.clone(),
                        citations: Vec::new(),
                    });
                }
            }
        }

        // 1b. Read the Command Center settings up front. The file is tiny and
        //     this only runs after a cache miss (at most once every 10 min),
        //     so the syscall cost is negligible. Settings drive two
        //     branches below: the manual-location override and the
        //     custom-endpoint try-then-fall-back behavior.
        let settings = {
            let guard = self
                .app
                .lock()
                .expect("weather app handle mutex not poisoned");
            guard
                .as_ref()
                .map(crate::command_center::settings::CommandCenterSettings::load)
        };

        // 1c. Try the custom endpoint first if configured. Per PRD AC #18,
        //     ANY failure here (template invalid, network, parse, schema)
        //     falls through silently to the Open-Meteo path — the user only
        //     sees error strings via `cc_test_weather`, never the runtime.
        if let Some(s) = settings.as_ref() {
            if s.weather.mode == crate::command_center::settings::WeatherMode::Custom {
                if let Some(line) = self.try_custom_endpoint(s).await {
                    let mut guard = self.cache.lock().expect("weather cache mutex not poisoned");
                    *guard = Some((line.clone(), Instant::now()));
                    return Ok(ToolOutput {
                        natural_language: line,
                        citations: Vec::new(),
                    });
                }
                // try_custom_endpoint already logged the reason; fall through.
            }
        }

        // 2. Resolve location. Manual override beats ipwho.is when set;
        //    otherwise we look up once per process and cache the result.
        //    We don't cache lookup failures so a flaky ipwho.is at startup
        //    self-heals on the next call.
        let manual_override = settings
            .as_ref()
            .and_then(|s| s.weather.manual_location.as_ref())
            .map(|m| (m.lat, m.lon));

        let location = {
            let guard = self
                .location
                .lock()
                .expect("weather location mutex not poisoned");
            *guard
        };

        let (lat, lon) = match (manual_override, location) {
            (Some(coords), _) => coords,
            (None, Some(coords)) => coords,
            (None, None) => match self.http.get_json(GEOIP_URL).await {
                Ok(body) => {
                    let lat = body.get("latitude").and_then(|v| v.as_f64());
                    let lon = body.get("longitude").and_then(|v| v.as_f64());
                    match (lat, lon) {
                        (Some(la), Some(lo)) => {
                            // Null-island short-circuit: broken IP-geo
                            // services return (0, 0). Don't poison the
                            // cache and don't burn an Open-Meteo call.
                            if la == 0.0 && lo == 0.0 {
                                log::warn!("[Rolo WEATHER] ipwho.is: returned null-island coords");
                                return Ok(unavailable_output());
                            }
                            let mut guard = self
                                .location
                                .lock()
                                .expect("weather location mutex not poisoned");
                            *guard = Some((la, lo));
                            (la, lo)
                        }
                        _ => {
                            log::warn!("[Rolo WEATHER] ipwho.is: malformed body (missing lat/lon)");
                            return Ok(unavailable_output());
                        }
                    }
                }
                Err(e) => {
                    log::warn!("[Rolo WEATHER] ipwho.is: {e}");
                    return Ok(unavailable_output());
                }
            },
        };

        // 3. Fetch weather. Open-Meteo accepts lat/lon as URL params.
        let url = format!(
            "{OPEN_METEO_BASE}?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code&temperature_unit=fahrenheit"
        );
        let body = match self.http.get_json(&url).await {
            Ok(v) => v,
            Err(e) => {
                log::warn!("[Rolo WEATHER] open-meteo.com: {e}");
                return Ok(unavailable_output());
            }
        };

        let current = match body.get("current") {
            Some(c) => c,
            None => {
                log::warn!("[Rolo WEATHER] open-meteo.com: missing `current` field");
                return Ok(unavailable_output());
            }
        };

        let temp_f = match current.get("temperature_2m").and_then(|v| v.as_f64()) {
            Some(t) => t,
            None => {
                log::warn!("[Rolo WEATHER] open-meteo.com: missing/invalid temperature_2m");
                return Ok(unavailable_output());
            }
        };
        // Open-Meteo sometimes returns the code as a float (e.g. 2.0) — pull
        // it through `as_f64()` first then cast.
        let code_raw = match current.get("weather_code").and_then(|v| v.as_f64()) {
            Some(c) => c,
            None => {
                log::warn!("[Rolo WEATHER] open-meteo.com: missing/invalid weather_code");
                return Ok(unavailable_output());
            }
        };

        // 4. Round-half-up via `(t + 0.5).floor()` so -2.4 → -2 (not -3).
        //    Verify mentally: (-2.4 + 0.5).floor() = (-1.9).floor() = -2. ✓
        let temp_int = (temp_f + 0.5).floor() as i32;

        // 5. Range-check. Out-of-range temps imply a busted upstream
        //    station; the LLM should hear "unavailable" rather than the
        //    nonsense reading.
        if !(-100..=140).contains(&temp_int) {
            log::warn!(
                "[Rolo WEATHER] open-meteo.com: temperature {temp_int}°F outside [-100, 140]"
            );
            return Ok(unavailable_output());
        }

        let code = code_raw as u32;
        let condition = weather_code_to_string(code);
        let line = format!("{}°F, {}.", temp_int, condition);

        // 6. Cache and return. A second invocation while we're computing
        //    here is fine — the dispatcher serialises tool calls.
        {
            let mut guard = self.cache.lock().expect("weather cache mutex not poisoned");
            *guard = Some((line.clone(), Instant::now()));
        }

        Ok(ToolOutput {
            natural_language: line,
            citations: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::SharedMood;
    use crate::mood::MoodState;
    use crate::state_machine::Pet;
    use crate::state_snapshot::Clock;
    use crate::vault::embeddings::Embedder;
    use crate::vault::Vault;
    use chrono::TimeZone;
    use std::collections::HashMap;
    use std::io;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // Test helpers (cloned from get_mood_state.rs::tests so this file owns
    // a self-contained ToolContext fixture).
    // -----------------------------------------------------------------------

    struct FixedClock(chrono::DateTime<chrono::Local>);
    impl Clock for FixedClock {
        fn now(&self) -> chrono::DateTime<chrono::Local> {
            self.0
        }
    }

    struct DeadEmbedder;
    impl Embedder for DeadEmbedder {
        fn probe_digest(&self) -> io::Result<[u8; 32]> {
            Ok([0; 32])
        }
        fn embed_batch(
            &self,
            _texts: &[String],
            _timeout: Duration,
        ) -> io::Result<Vec<Option<Vec<f32>>>> {
            Ok(Vec::new())
        }
        fn embed_query(&self, _text: &str) -> Option<[f32; 768]> {
            None
        }
        fn model_name(&self) -> &str {
            "dead"
        }
    }

    fn fresh_pet() -> Pet {
        Pet::new(0, 0, 1920, 1080, 100, 2)
    }

    fn fresh_vault() -> (TempDir, Arc<Vault>) {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("vault");
        let embedder: Arc<dyn Embedder> = Arc::new(DeadEmbedder);
        let vault = Vault::open_or_init_with_embedder(root, embedder);
        (tmp, vault)
    }

    fn make_clock() -> FixedClock {
        FixedClock(
            chrono::Local
                .with_ymd_and_hms(2026, 5, 8, 14, 0, 0)
                .single()
                .unwrap(),
        )
    }

    // -----------------------------------------------------------------------
    // MockFetcher — keyed by URL substring so tests don't have to match the
    // exact Open-Meteo query string. Each key carries a queue of responses
    // (for tests that want different bodies on successive calls) plus a hit
    // counter for cache-miss assertions.
    //
    // FetchError doesn't derive Clone, so we store responses as Options and
    // `take()` them on lookup. If the queue empties, we fall back to a
    // generic Network error so test-author mistakes show up loudly.
    // -----------------------------------------------------------------------

    type MockResponses = HashMap<&'static str, Vec<Option<Result<Value, FetchError>>>>;

    struct MockFetcher {
        responses: Mutex<MockResponses>,
        hits: Mutex<HashMap<&'static str, usize>>,
    }

    impl MockFetcher {
        fn new() -> Self {
            Self {
                responses: Mutex::new(HashMap::new()),
                hits: Mutex::new(HashMap::new()),
            }
        }

        fn set_response(&self, key: &'static str, resp: Result<Value, FetchError>) {
            let mut map = self.responses.lock().unwrap();
            map.entry(key).or_default().push(Some(resp));
        }

        fn hits_for(&self, key: &'static str) -> usize {
            *self.hits.lock().unwrap().get(key).unwrap_or(&0)
        }
    }

    #[async_trait::async_trait]
    impl HttpFetcher for MockFetcher {
        async fn get_json(&self, url: &str) -> Result<Value, FetchError> {
            let key = {
                let map = self.responses.lock().unwrap();
                map.keys().find(|k| url.contains(*k)).copied()
            };

            match key {
                Some(k) => {
                    {
                        let mut h = self.hits.lock().unwrap();
                        *h.entry(k).or_insert(0) += 1;
                    }
                    let mut map = self.responses.lock().unwrap();
                    let queue = map.get_mut(k).unwrap();
                    // Pop the next response; if empty, repeat the most
                    // recently-set value via... actually, surface a clear
                    // error so test bugs stand out.
                    if queue.is_empty() {
                        return Err(FetchError::Network(format!(
                            "MockFetcher: no responses queued for key `{k}`"
                        )));
                    }
                    let next = queue.remove(0);
                    next.unwrap_or_else(|| {
                        Err(FetchError::Network(
                            "MockFetcher: response taken twice".into(),
                        ))
                    })
                }
                None => Err(FetchError::Network(format!(
                    "MockFetcher: no key matched url `{url}`"
                ))),
            }
        }
    }

    // -----------------------------------------------------------------------
    // ToolContext builder — every test needs the same fixture.
    // -----------------------------------------------------------------------

    fn make_ctx<'a>(
        vault: &'a Vault,
        mood: &'a SharedMood,
        pet: &'a Pet,
        clock: &'a dyn Clock,
    ) -> ToolContext<'a> {
        ToolContext {
            vault,
            mood,
            pet,
            clock,
        }
    }

    // -----------------------------------------------------------------------
    // Happy-path tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn first_invocation_returns_formatted_string() {
        let mock = Arc::new(MockFetcher::new());
        mock.set_response(
            "ipwho",
            Ok(json!({"latitude": 40.6782, "longitude": -73.9442})),
        );
        mock.set_response(
            "open-meteo",
            Ok(json!({"current": {"temperature_2m": 62, "weather_code": 2}})),
        );

        let tool = GetWeather::with_fetcher(mock.clone());
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = make_clock();
        let ctx = make_ctx(&vault, &mood, &pet, &clock);

        let out = tool.invoke(&json!({}), &ctx).await.unwrap();
        assert_eq!(out.natural_language, "62°F, partly cloudy.");
        assert!(out.citations.is_empty());
    }

    #[tokio::test]
    async fn cache_hit_within_ttl_skips_network() {
        let mock = Arc::new(MockFetcher::new());
        mock.set_response(
            "ipwho",
            Ok(json!({"latitude": 40.6782, "longitude": -73.9442})),
        );
        mock.set_response(
            "open-meteo",
            Ok(json!({"current": {"temperature_2m": 62, "weather_code": 2}})),
        );

        let tool = GetWeather::with_fetcher(mock.clone());
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = make_clock();
        let ctx = make_ctx(&vault, &mood, &pet, &clock);

        let first = tool.invoke(&json!({}), &ctx).await.unwrap();
        let second = tool.invoke(&json!({}), &ctx).await.unwrap();

        assert_eq!(first.natural_language, second.natural_language);
        assert_eq!(
            mock.hits_for("open-meteo"),
            1,
            "second invocation should hit cache, not open-meteo"
        );
        assert_eq!(mock.hits_for("ipwho"), 1);
    }

    // -----------------------------------------------------------------------
    // Error / edge-case tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn ipwho_failure_returns_unavailable() {
        let mock = Arc::new(MockFetcher::new());
        mock.set_response("ipwho", Err(FetchError::Timeout));

        let tool = GetWeather::with_fetcher(mock.clone());
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = make_clock();
        let ctx = make_ctx(&vault, &mood, &pet, &clock);

        let out = tool.invoke(&json!({}), &ctx).await.unwrap();
        assert_eq!(out.natural_language, "weather unavailable.");
        // Location stays None — next call retries the lookup.
        let loc = *tool.location.lock().unwrap();
        assert!(loc.is_none(), "location should not be cached on failure");
    }

    #[tokio::test]
    async fn open_meteo_500_returns_unavailable() {
        let mock = Arc::new(MockFetcher::new());
        mock.set_response(
            "ipwho",
            Ok(json!({"latitude": 40.6782, "longitude": -73.9442})),
        );
        mock.set_response("open-meteo", Err(FetchError::Status(500)));

        let tool = GetWeather::with_fetcher(mock.clone());
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = make_clock();
        let ctx = make_ctx(&vault, &mood, &pet, &clock);

        let out = tool.invoke(&json!({}), &ctx).await.unwrap();
        assert_eq!(out.natural_language, "weather unavailable.");
    }

    #[tokio::test]
    async fn malformed_json_returns_unavailable() {
        let mock = Arc::new(MockFetcher::new());
        mock.set_response(
            "ipwho",
            Ok(json!({"latitude": 40.6782, "longitude": -73.9442})),
        );
        mock.set_response(
            "open-meteo",
            Err(FetchError::Parse("expected object".into())),
        );

        let tool = GetWeather::with_fetcher(mock.clone());
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = make_clock();
        let ctx = make_ctx(&vault, &mood, &pet, &clock);

        let out = tool.invoke(&json!({}), &ctx).await.unwrap();
        assert_eq!(out.natural_language, "weather unavailable.");
    }

    #[tokio::test]
    async fn unknown_weather_code_falls_back_to_overcast() {
        let mock = Arc::new(MockFetcher::new());
        mock.set_response(
            "ipwho",
            Ok(json!({"latitude": 40.6782, "longitude": -73.9442})),
        );
        mock.set_response(
            "open-meteo",
            Ok(json!({"current": {"temperature_2m": 62, "weather_code": 200}})),
        );

        let tool = GetWeather::with_fetcher(mock.clone());
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = make_clock();
        let ctx = make_ctx(&vault, &mood, &pet, &clock);

        let out = tool.invoke(&json!({}), &ctx).await.unwrap();
        assert_eq!(out.natural_language, "62°F, overcast.");
    }

    #[tokio::test]
    async fn positive_float_temp_rounds_correctly() {
        let mock = Arc::new(MockFetcher::new());
        mock.set_response(
            "ipwho",
            Ok(json!({"latitude": 40.6782, "longitude": -73.9442})),
        );
        mock.set_response(
            "open-meteo",
            Ok(json!({"current": {"temperature_2m": 61.7, "weather_code": 0}})),
        );

        let tool = GetWeather::with_fetcher(mock.clone());
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = make_clock();
        let ctx = make_ctx(&vault, &mood, &pet, &clock);

        let out = tool.invoke(&json!({}), &ctx).await.unwrap();
        assert_eq!(out.natural_language, "62°F, clear sky.");
    }

    #[tokio::test]
    async fn negative_float_temp_rounds_toward_zero() {
        let mock = Arc::new(MockFetcher::new());
        mock.set_response(
            "ipwho",
            Ok(json!({"latitude": 40.6782, "longitude": -73.9442})),
        );
        mock.set_response(
            "open-meteo",
            Ok(json!({"current": {"temperature_2m": -2.4, "weather_code": 0}})),
        );

        let tool = GetWeather::with_fetcher(mock.clone());
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = make_clock();
        let ctx = make_ctx(&vault, &mood, &pet, &clock);

        let out = tool.invoke(&json!({}), &ctx).await.unwrap();
        assert_eq!(out.natural_language, "-2°F, clear sky.");
    }

    #[tokio::test]
    async fn null_island_short_circuits() {
        let mock = Arc::new(MockFetcher::new());
        mock.set_response("ipwho", Ok(json!({"latitude": 0, "longitude": 0})));
        // Queue an open-meteo response so we can prove it was *not* consumed.
        mock.set_response(
            "open-meteo",
            Ok(json!({"current": {"temperature_2m": 99, "weather_code": 0}})),
        );

        let tool = GetWeather::with_fetcher(mock.clone());
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = make_clock();
        let ctx = make_ctx(&vault, &mood, &pet, &clock);

        let out = tool.invoke(&json!({}), &ctx).await.unwrap();
        assert_eq!(out.natural_language, "weather unavailable.");
        assert_eq!(
            mock.hits_for("open-meteo"),
            0,
            "open-meteo must not be called when ipwho returns null-island"
        );
    }

    #[tokio::test]
    async fn out_of_range_temp_rejected() {
        let mock = Arc::new(MockFetcher::new());
        mock.set_response(
            "ipwho",
            Ok(json!({"latitude": 40.6782, "longitude": -73.9442})),
        );
        mock.set_response(
            "open-meteo",
            Ok(json!({"current": {"temperature_2m": 999.9, "weather_code": 0}})),
        );

        let tool = GetWeather::with_fetcher(mock.clone());
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = make_clock();
        let ctx = make_ctx(&vault, &mood, &pet, &clock);

        let out = tool.invoke(&json!({}), &ctx).await.unwrap();
        assert_eq!(out.natural_language, "weather unavailable.");
    }

    #[tokio::test]
    async fn tool_always_returns_ok_never_err() {
        let mock = Arc::new(MockFetcher::new());
        mock.set_response("ipwho", Err(FetchError::Network("dns".into())));
        mock.set_response("open-meteo", Err(FetchError::Network("dns".into())));

        let tool = GetWeather::with_fetcher(mock.clone());
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = make_clock();
        let ctx = make_ctx(&vault, &mood, &pet, &clock);

        let result = tool.invoke(&json!({}), &ctx).await;
        assert!(result.is_ok(), "invoke must never return Err");
        assert_eq!(result.unwrap().natural_language, "weather unavailable.");
    }

    #[tokio::test]
    async fn stray_args_ignored() {
        let mock = Arc::new(MockFetcher::new());
        mock.set_response(
            "ipwho",
            Ok(json!({"latitude": 40.6782, "longitude": -73.9442})),
        );
        mock.set_response(
            "open-meteo",
            Ok(json!({"current": {"temperature_2m": 62, "weather_code": 2}})),
        );

        let tool = GetWeather::with_fetcher(mock.clone());
        let (_tmp, vault) = fresh_vault();
        let mood: SharedMood = Arc::new(Mutex::new(MoodState::default()));
        let pet = fresh_pet();
        let clock = make_clock();
        let ctx = make_ctx(&vault, &mood, &pet, &clock);

        let result = tool
            .invoke(&json!({"city": "Tokyo", "garbage": 1}), &ctx)
            .await;
        assert!(result.is_ok());
    }
}
