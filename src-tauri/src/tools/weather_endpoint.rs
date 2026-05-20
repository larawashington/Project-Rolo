//! Custom weather endpoint support (PRD/rolo-command-center.md Phase 8).
//!
//! The default weather path in `get_weather.rs` queries Open-Meteo with a
//! WMO weather code. When the user configures a custom endpoint via the
//! Weather tab, the response shape becomes whatever the chosen preset's API
//! returns (OpenWeatherMap, WeatherAPI.com, Visual Crossing) or whatever a
//! pair of flat JSONPath expressions can pull out.
//!
//! This module owns:
//!   1. `ParsedWeather` — the cross-preset normalized output that
//!      `get_weather` knows how to render into the same "72°F, partly cloudy"
//!      one-liner the LLM sees today.
//!   2. Hardcoded parsers for each known preset.
//!   3. A deliberately tiny flat-JSONPath walker for the "Custom" preset.
//!   4. `test_endpoint` — the function the Brain-style "Test endpoint"
//!      button on the Weather tab invokes (via `cc_test_weather`). It
//!      validates the URL template, substitutes placeholders, fetches once
//!      with a 5s timeout, parses the body, and returns a `ParsedWeather`
//!      (or a human-readable error the UI can render verbatim).
//!
//! **Silent fallback contract.** Runtime fetches in `get_weather.rs` never
//! surface these error strings to the user — they log and fall through to
//! Open-Meteo. The error path is reserved for `cc_test_weather`, where the
//! user is explicitly asking "what's wrong with my config?" (PRD AC #18).

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::command_center::settings::validate_url_template;
use crate::http::http_client;

/// Cross-preset normalized weather result. The "Custom JSONPath" preset
/// can't always supply a numeric `weather_code` (no shared code system
/// across third-party APIs), hence the `Option`. Open-Meteo's WMO code is
/// the reference shape — known presets that ship a numeric code at all
/// (OpenWeatherMap, WeatherAPI.com) populate it raw without remapping;
/// `get_weather.rs` does not currently consume the field, but it's
/// captured so future condition-aware logic doesn't need a re-parse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedWeather {
    pub temp_f: f64,
    pub condition_text: String,
    pub weather_code: Option<i32>,
}

/// HTTP/parse failure timeline. Surfaced verbatim by `cc_test_weather`;
/// folded into a single `log::warn!` line by the runtime path.
#[derive(Debug)]
pub enum EndpointError {
    InvalidTemplate(String),
    InvalidJsonPath(String),
    MissingArg(&'static str),
    Network(String),
    Status(u16),
    Parse(String),
    Schema(String),
    UnknownPreset(String),
}

impl std::fmt::Display for EndpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EndpointError::InvalidTemplate(s) => write!(f, "{}", s),
            EndpointError::InvalidJsonPath(s) => write!(f, "Invalid JSONPath: {}", s),
            EndpointError::MissingArg(s) => write!(f, "Missing required argument: {}", s),
            EndpointError::Network(s) => write!(f, "Network error: {}", s),
            EndpointError::Status(c) => write!(f, "HTTP {}", c),
            EndpointError::Parse(s) => write!(f, "Response was not valid JSON: {}", s),
            EndpointError::Schema(s) => write!(f, "Response did not match preset shape: {}", s),
            EndpointError::UnknownPreset(s) => write!(f, "Unknown preset: {}", s),
        }
    }
}

// ---------------------------------------------------------------------------
// Preset parsers
// ---------------------------------------------------------------------------

/// OpenWeatherMap `/data/2.5/weather` (with `units=imperial`):
///   { "main": {"temp": 72.4}, "weather": [{"id": 800, "description": "clear sky"}] }
pub fn parse_openweathermap(body: &Value) -> Result<ParsedWeather, EndpointError> {
    let temp_f = body
        .get("main")
        .and_then(|m| m.get("temp"))
        .and_then(|t| t.as_f64())
        .ok_or_else(|| EndpointError::Schema("missing `main.temp`".into()))?;
    let first = body
        .get("weather")
        .and_then(|w| w.as_array())
        .and_then(|arr| arr.first())
        .ok_or_else(|| EndpointError::Schema("missing `weather[0]`".into()))?;
    let condition_text = first
        .get("description")
        .and_then(|d| d.as_str())
        .ok_or_else(|| EndpointError::Schema("missing `weather[0].description`".into()))?
        .to_string();
    let weather_code = first.get("id").and_then(|i| i.as_i64()).map(|i| i as i32);
    Ok(ParsedWeather {
        temp_f,
        condition_text,
        weather_code,
    })
}

/// WeatherAPI.com `/v1/current.json?q={lat},{lon}`:
///   { "current": {"temp_f": 72.4, "condition": {"code": 1003, "text": "Partly cloudy"}} }
pub fn parse_weatherapi(body: &Value) -> Result<ParsedWeather, EndpointError> {
    let current = body
        .get("current")
        .ok_or_else(|| EndpointError::Schema("missing `current`".into()))?;
    let temp_f = current
        .get("temp_f")
        .and_then(|t| t.as_f64())
        .ok_or_else(|| EndpointError::Schema("missing `current.temp_f`".into()))?;
    let condition = current
        .get("condition")
        .ok_or_else(|| EndpointError::Schema("missing `current.condition`".into()))?;
    let condition_text = condition
        .get("text")
        .and_then(|s| s.as_str())
        .ok_or_else(|| EndpointError::Schema("missing `current.condition.text`".into()))?
        .to_string();
    let weather_code = condition
        .get("code")
        .and_then(|c| c.as_i64())
        .map(|c| c as i32);
    Ok(ParsedWeather {
        temp_f,
        condition_text,
        weather_code,
    })
}

/// Visual Crossing `/timeline/{lat},{lon}/today?unitGroup=us&include=current`:
///   { "currentConditions": {"temp": 72.4, "conditions": "Partly cloudy", "icon": "..."} }
///
/// Visual Crossing publishes a textual `icon` slug (e.g. "partly-cloudy-day")
/// but no numeric WMO-equivalent code, so `weather_code` is left None.
pub fn parse_visualcrossing(body: &Value) -> Result<ParsedWeather, EndpointError> {
    let current = body
        .get("currentConditions")
        .ok_or_else(|| EndpointError::Schema("missing `currentConditions`".into()))?;
    let temp_f = current
        .get("temp")
        .and_then(|t| t.as_f64())
        .ok_or_else(|| EndpointError::Schema("missing `currentConditions.temp`".into()))?;
    let condition_text = current
        .get("conditions")
        .and_then(|s| s.as_str())
        .ok_or_else(|| EndpointError::Schema("missing `currentConditions.conditions`".into()))?
        .to_string();
    Ok(ParsedWeather {
        temp_f,
        condition_text,
        weather_code: None,
    })
}

// ---------------------------------------------------------------------------
// Flat JSONPath walker
// ---------------------------------------------------------------------------

/// Walk a flat JSONPath like `$.main.temp` against `body`. v1 is intentionally
/// minimal — no arrays, no filters, no wildcards. Anything other than a
/// dot-separated chain of keys is rejected up front so the user gets a clear
/// "this kind of path isn't supported yet" message rather than a silent
/// mis-extraction.
pub fn walk_flat_jsonpath<'a>(body: &'a Value, path: &str) -> Result<&'a Value, EndpointError> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(EndpointError::InvalidJsonPath("path is empty".into()));
    }
    if trimmed.contains('[') || trimmed.contains('*') || trimmed.contains('?') {
        return Err(EndpointError::InvalidJsonPath(format!(
            "v1 supports only flat paths (no arrays, wildcards, or filters): `{}`",
            trimmed
        )));
    }
    // Strip a leading `$` if present, then strip a leading `.`.
    let mut rest = trimmed.strip_prefix('$').unwrap_or(trimmed);
    rest = rest.strip_prefix('.').unwrap_or(rest);

    let mut cursor = body;
    if rest.is_empty() {
        return Ok(cursor);
    }
    for key in rest.split('.') {
        if key.is_empty() {
            return Err(EndpointError::InvalidJsonPath(format!(
                "empty segment in path `{}`",
                trimmed
            )));
        }
        cursor = cursor.get(key).ok_or_else(|| {
            EndpointError::Schema(format!("path `{}` missed at key `{}`", trimmed, key))
        })?;
    }
    Ok(cursor)
}

/// Custom JSONPath preset — pluck `temp` and `condition_text` via two
/// user-supplied paths. Temp is coerced from number or string; condition is
/// coerced from string. `weather_code` is always None for this preset.
pub fn parse_custom_jsonpath(
    body: &Value,
    temp_path: &str,
    condition_path: &str,
) -> Result<ParsedWeather, EndpointError> {
    let temp_val = walk_flat_jsonpath(body, temp_path)?;
    let temp_f = match temp_val {
        Value::Number(n) => n.as_f64().ok_or_else(|| {
            EndpointError::Schema(format!("temp at `{}` is not numeric", temp_path))
        })?,
        Value::String(s) => s.parse::<f64>().map_err(|_| {
            EndpointError::Schema(format!(
                "temp at `{}` is a string but not a parseable number (got `{}`)",
                temp_path, s
            ))
        })?,
        _ => {
            return Err(EndpointError::Schema(format!(
                "temp at `{}` must be a number or string",
                temp_path
            )))
        }
    };
    let cond_val = walk_flat_jsonpath(body, condition_path)?;
    let condition_text = match cond_val {
        Value::String(s) => s.clone(),
        // Allow numbers/booleans too — fall back to a display form rather
        // than failing. Real users might have an integer status code their
        // app already stringifies elsewhere.
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => {
            return Err(EndpointError::Schema(format!(
                "condition at `{}` must be a scalar (string/number/bool)",
                condition_path
            )))
        }
    };
    Ok(ParsedWeather {
        temp_f,
        condition_text,
        weather_code: None,
    })
}

// ---------------------------------------------------------------------------
// Dispatch by preset name (mirrors the strings in `WeatherSettings::response_preset`)
// ---------------------------------------------------------------------------

pub fn parse_with_preset(
    preset: &str,
    body: &Value,
    jsonpath_temp: Option<&str>,
    jsonpath_condition: Option<&str>,
) -> Result<ParsedWeather, EndpointError> {
    match preset {
        "openweathermap" => parse_openweathermap(body),
        "weatherapi" => parse_weatherapi(body),
        "visualcrossing" => parse_visualcrossing(body),
        "custom_jsonpath" => {
            let t = jsonpath_temp.ok_or(EndpointError::MissingArg("jsonpath_temp"))?;
            let c = jsonpath_condition.ok_or(EndpointError::MissingArg("jsonpath_condition"))?;
            parse_custom_jsonpath(body, t, c)
        }
        other => Err(EndpointError::UnknownPreset(other.to_string())),
    }
}

// ---------------------------------------------------------------------------
// URL substitution
// ---------------------------------------------------------------------------

/// Substitute `{lat}`, `{lon}`, `{key}` placeholders. Lat/lon are formatted to
/// 4 decimal places — every preset the PRD names accepts that precision (and
/// usually more) without complaint, and 4dp keeps URLs short for logs.
pub fn substitute_template(template: &str, lat: f64, lon: f64, api_key: &str) -> String {
    template
        .replace("{lat}", &format!("{:.4}", lat))
        .replace("{lon}", &format!("{:.4}", lon))
        .replace("{key}", api_key)
}

// ---------------------------------------------------------------------------
// test_endpoint — the Tauri command's worker
// ---------------------------------------------------------------------------

/// Probe a fully-configured custom weather endpoint with a 5s timeout, parse
/// the body per `preset`, and return the normalized `ParsedWeather`.
///
/// On any failure (template invalid, network error, non-2xx status, body not
/// JSON, body shape wrong, JSONPath miss) returns `Err(human-readable string)`
/// the UI can render verbatim under the Test button.
pub async fn test_endpoint(
    template: &str,
    api_key: &str,
    preset: &str,
    lat: f64,
    lon: f64,
    jsonpath_temp: Option<&str>,
    jsonpath_condition: Option<&str>,
) -> Result<ParsedWeather, String> {
    // 1. Template validation — uses the same helper Phase 1 ships, so the
    //    Save button's gating logic and the Test button share one truth.
    validate_url_template(template).map_err(|e| e.to_string())?;

    // 2. Placeholder substitution. Plain replacement — every preset the PRD
    //    lists accepts unencoded lat/lon and api keys (which are alphanumeric).
    let url = substitute_template(template, lat, lon, api_key);

    // 3. Fetch with a 5s wall budget. Build a fresh client per call so the
    //    test path doesn't share state with the runtime tool's client and
    //    tests can't flake on a shared connection pool.
    let client = http_client(Duration::from_secs(5))
        .map_err(|e| format!("Could not build HTTP client: {}", e))?;

    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            if e.is_timeout() {
                return Err("Test timed out after 5 seconds.".to_string());
            }
            return Err(EndpointError::Network(e.to_string()).to_string());
        }
    };

    let status = resp.status();
    if !status.is_success() {
        return Err(EndpointError::Status(status.as_u16()).to_string());
    }

    let body: Value = resp
        .json()
        .await
        .map_err(|e| EndpointError::Parse(e.to_string()).to_string())?;

    parse_with_preset(preset, &body, jsonpath_temp, jsonpath_condition).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ------------------------------------------------------------------
    // Preset parser tests — fixture JSON pulled straight from the PRD's
    // "reference shapes" section. Each test exercises one parser only.
    // ------------------------------------------------------------------

    #[test]
    fn openweathermap_extracts_temp_description_id() {
        let body = json!({
            "main": {"temp": 72.4},
            "weather": [{"id": 800, "description": "clear sky"}]
        });
        let out = parse_openweathermap(&body).expect("parse");
        assert_eq!(out.temp_f, 72.4);
        assert_eq!(out.condition_text, "clear sky");
        assert_eq!(out.weather_code, Some(800));
    }

    #[test]
    fn openweathermap_missing_main_temp_errors() {
        let body = json!({"weather": [{"id": 800, "description": "x"}]});
        let err = parse_openweathermap(&body).expect_err("must error");
        let msg = err.to_string();
        assert!(msg.contains("main.temp"), "got: {}", msg);
    }

    #[test]
    fn openweathermap_empty_weather_array_errors() {
        let body = json!({"main": {"temp": 70.0}, "weather": []});
        let err = parse_openweathermap(&body).expect_err("must error");
        let msg = err.to_string();
        assert!(msg.contains("weather[0]"), "got: {}", msg);
    }

    #[test]
    fn weatherapi_extracts_temp_text_code() {
        let body = json!({
            "current": {"temp_f": 72.4, "condition": {"code": 1003, "text": "Partly cloudy"}}
        });
        let out = parse_weatherapi(&body).expect("parse");
        assert_eq!(out.temp_f, 72.4);
        assert_eq!(out.condition_text, "Partly cloudy");
        assert_eq!(out.weather_code, Some(1003));
    }

    #[test]
    fn weatherapi_missing_current_errors() {
        let body = json!({});
        let err = parse_weatherapi(&body).expect_err("must error");
        assert!(err.to_string().contains("current"), "{}", err);
    }

    #[test]
    fn visualcrossing_extracts_temp_conditions_leaves_code_none() {
        let body = json!({
            "currentConditions": {"temp": 72.4, "conditions": "Partly cloudy", "icon": "partly-cloudy-day"}
        });
        let out = parse_visualcrossing(&body).expect("parse");
        assert_eq!(out.temp_f, 72.4);
        assert_eq!(out.condition_text, "Partly cloudy");
        assert_eq!(out.weather_code, None);
    }

    #[test]
    fn visualcrossing_missing_temp_errors() {
        let body = json!({"currentConditions": {"conditions": "Partly cloudy"}});
        let err = parse_visualcrossing(&body).expect_err("must error");
        assert!(err.to_string().contains("temp"), "{}", err);
    }

    // ------------------------------------------------------------------
    // Flat JSONPath walker tests
    // ------------------------------------------------------------------

    #[test]
    fn jsonpath_walks_nested_object() {
        let body = json!({"main": {"temp": 72.4}});
        let v = walk_flat_jsonpath(&body, "$.main.temp").expect("walk");
        assert_eq!(v.as_f64(), Some(72.4));
    }

    #[test]
    fn jsonpath_strips_leading_dollar_and_dot() {
        let body = json!({"foo": {"bar": 1}});
        let v = walk_flat_jsonpath(&body, "foo.bar").expect("walk");
        assert_eq!(v.as_i64(), Some(1));
    }

    #[test]
    fn jsonpath_rejects_bracket() {
        let err = walk_flat_jsonpath(&json!({}), "$.weather[0].description").expect_err("reject");
        assert!(
            matches!(err, EndpointError::InvalidJsonPath(_)),
            "got: {:?}",
            err
        );
        assert!(err.to_string().contains("flat paths"), "{}", err);
    }

    #[test]
    fn jsonpath_rejects_wildcard() {
        let err = walk_flat_jsonpath(&json!({}), "$.*.temp").expect_err("reject");
        assert!(matches!(err, EndpointError::InvalidJsonPath(_)));
    }

    #[test]
    fn jsonpath_rejects_filter() {
        let err = walk_flat_jsonpath(&json!({}), "$.weather[?(@.id)]").expect_err("reject");
        assert!(matches!(err, EndpointError::InvalidJsonPath(_)));
    }

    #[test]
    fn jsonpath_rejects_empty_path() {
        let err = walk_flat_jsonpath(&json!({}), "").expect_err("reject");
        assert!(matches!(err, EndpointError::InvalidJsonPath(_)));
    }

    #[test]
    fn jsonpath_rejects_empty_segment() {
        let err = walk_flat_jsonpath(&json!({"a": {"b": 1}}), "$.a..b").expect_err("reject");
        assert!(matches!(err, EndpointError::InvalidJsonPath(_)));
    }

    #[test]
    fn jsonpath_miss_returns_schema_error() {
        let body = json!({"main": {"temp": 72.4}});
        let err = walk_flat_jsonpath(&body, "$.main.humidity").expect_err("miss");
        assert!(matches!(err, EndpointError::Schema(_)));
        assert!(err.to_string().contains("humidity"), "{}", err);
    }

    #[test]
    fn custom_jsonpath_happy_path() {
        let body = json!({"main": {"temp": 65.0}, "summary": "sunny"});
        let out = parse_custom_jsonpath(&body, "$.main.temp", "$.summary").expect("parse");
        assert_eq!(out.temp_f, 65.0);
        assert_eq!(out.condition_text, "sunny");
        assert!(out.weather_code.is_none());
    }

    #[test]
    fn custom_jsonpath_temp_as_string_parses() {
        // Some APIs return temp wrapped in quotes.
        let body = json!({"t": "70.5", "c": "cloudy"});
        let out = parse_custom_jsonpath(&body, "$.t", "$.c").expect("parse");
        assert_eq!(out.temp_f, 70.5);
    }

    #[test]
    fn custom_jsonpath_temp_non_numeric_string_errors() {
        let body = json!({"t": "warm-ish", "c": "cloudy"});
        let err = parse_custom_jsonpath(&body, "$.t", "$.c").expect_err("reject");
        assert!(matches!(err, EndpointError::Schema(_)));
    }

    // ------------------------------------------------------------------
    // Dispatch
    // ------------------------------------------------------------------

    #[test]
    fn parse_with_preset_unknown_errors() {
        let err = parse_with_preset("not-a-preset", &json!({}), None, None).expect_err("reject");
        assert!(matches!(err, EndpointError::UnknownPreset(_)));
    }

    #[test]
    fn parse_with_preset_custom_requires_paths() {
        let err = parse_with_preset("custom_jsonpath", &json!({}), None, None).expect_err("reject");
        assert!(matches!(err, EndpointError::MissingArg(_)));
    }

    // ------------------------------------------------------------------
    // URL substitution
    // ------------------------------------------------------------------

    #[test]
    fn substitute_template_replaces_all_tokens() {
        let tpl = "https://x/wx?lat={lat}&lon={lon}&appid={key}";
        let out = substitute_template(tpl, 40.6782, -73.9442, "ABC123");
        assert_eq!(out, "https://x/wx?lat=40.6782&lon=-73.9442&appid=ABC123");
    }

    #[test]
    fn substitute_template_truncates_to_four_decimals() {
        let tpl = "lat={lat}&lon={lon}&key={key}";
        let out = substitute_template(tpl, 40.123456789, -73.987654321, "k");
        assert_eq!(out, "lat=40.1235&lon=-73.9877&key=k");
    }

    // ------------------------------------------------------------------
    // test_endpoint — only the cases that don't need a real network.
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_endpoint_rejects_template_missing_token() {
        // No {key} token.
        let result = test_endpoint(
            "https://api.example.com/wx?lat={lat}&lon={lon}",
            "key",
            "openweathermap",
            40.6782,
            -73.9442,
            None,
            None,
        )
        .await;
        let err = result.expect_err("must error");
        assert!(err.contains("{key}"), "got: {}", err);
    }

    #[tokio::test]
    async fn test_endpoint_rejects_template_missing_lat() {
        let result = test_endpoint(
            "https://api.example.com/wx?lon={lon}&appid={key}",
            "key",
            "openweathermap",
            40.6782,
            -73.9442,
            None,
            None,
        )
        .await;
        let err = result.expect_err("must error");
        assert!(err.contains("{lat}"), "got: {}", err);
    }
}
