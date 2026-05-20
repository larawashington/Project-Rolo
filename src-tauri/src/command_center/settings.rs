//! Persistence + validation for the Rolo Command Center settings file.
//!
//! Lives at `app_data_dir/command_center.json` (the `com.larawashington.rolo`
//! bundle — same directory as the vault). The schema is intentionally
//! forward-compatible: every struct flattens an `extra` `serde_json::Map` so
//! a future Rolo writing a newer version doesn't drop fields when this
//! version round-trips the file. All known fields default sensibly so a
//! missing or empty file yields a working struct rather than an error.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::vault::atomic::atomic_write;

/// Top-level Command Center settings, matching `command_center.json` in the
/// PRD's Schemas section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandCenterSettings {
    #[serde(default = "default_version")]
    pub version: u32,

    #[serde(default)]
    pub weather: WeatherSettings,

    /// Preserves unknown top-level keys across a load/save round-trip so a
    /// downgrade (older Rolo reading a file written by newer Rolo) doesn't
    /// silently drop future panel state.
    #[serde(default, flatten, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for CommandCenterSettings {
    fn default() -> Self {
        Self {
            version: default_version(),
            weather: WeatherSettings::default(),
            extra: serde_json::Map::new(),
        }
    }
}

fn default_version() -> u32 {
    1
}

/// Whether the weather tool uses Rolo's default endpoint or a user-supplied one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeatherMode {
    Default,
    Custom,
}

/// Response-shape preset for parsing a custom weather endpoint. `OpenMeteo` is
/// the default-mode reading; the rest are the user-visible custom-endpoint
/// dropdown choices. Keep variant names + the snake_case rename in lockstep
/// with `src/types/commandCenter.ts::ResponsePreset` — the TS union pins the
/// expected wire strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsePreset {
    OpenMeteo,
    Openweathermap,
    Weatherapi,
    Visualcrossing,
    CustomJsonpath,
}

impl ResponsePreset {
    /// Wire string used by `parse_with_preset` and surfaced in tests/logs.
    /// Mirrors the snake_case rename so this stays the single source of
    /// truth for the spelling.
    pub fn as_wire_str(self) -> &'static str {
        match self {
            Self::OpenMeteo => "open_meteo",
            Self::Openweathermap => "openweathermap",
            Self::Weatherapi => "weatherapi",
            Self::Visualcrossing => "visualcrossing",
            Self::CustomJsonpath => "custom_jsonpath",
        }
    }
}

/// Weather-panel configuration. The custom-endpoint fields stay empty when
/// `mode == Default`; the manual-location override applies in either mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeatherSettings {
    #[serde(default = "default_weather_mode")]
    pub mode: WeatherMode,

    #[serde(default)]
    pub custom_url_template: String,

    #[serde(default)]
    pub api_key: String,

    #[serde(default = "default_response_preset")]
    pub response_preset: ResponsePreset,

    #[serde(default)]
    pub custom_jsonpath_temp: String,

    #[serde(default)]
    pub custom_jsonpath_condition: String,

    #[serde(default)]
    pub manual_location: Option<ManualLocation>,

    /// Preserves unknown keys inside the weather object too — see top-level
    /// `extra` for rationale.
    #[serde(default, flatten, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for WeatherSettings {
    fn default() -> Self {
        Self {
            mode: default_weather_mode(),
            custom_url_template: String::new(),
            api_key: String::new(),
            response_preset: default_response_preset(),
            custom_jsonpath_temp: String::new(),
            custom_jsonpath_condition: String::new(),
            manual_location: None,
            extra: serde_json::Map::new(),
        }
    }
}

fn default_weather_mode() -> WeatherMode {
    WeatherMode::Default
}

fn default_response_preset() -> ResponsePreset {
    ResponsePreset::OpenMeteo
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ManualLocation {
    pub lat: f64,
    pub lon: f64,
}

impl CommandCenterSettings {
    /// Resolve the on-disk path. Lives in `app_data_dir/command_center.json`
    /// alongside the vault (`com.larawashington.rolo` bundle). Bubbles the
    /// Tauri path-resolution error up as a string so callers can decide
    /// whether the failure is fatal — `load` swallows it; `save` propagates.
    pub fn path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
        use tauri::Manager;
        app.path()
            .app_data_dir()
            .map(|d| d.join("command_center.json"))
            .map_err(|e| format!("Cannot resolve app_data_dir: {}", e))
    }

    /// Load from disk. A missing file or a parse error both yield `default()`
    /// — Rolo must keep running even if his settings file is corrupt. Parse
    /// errors are logged at warn level so the failure isn't completely
    /// silent.
    pub fn load(app: &tauri::AppHandle) -> Self {
        let path = match Self::path(app) {
            Ok(p) => p,
            Err(e) => {
                log::warn!(
                    "[Rolo] Command Center: cannot resolve settings path ({}). \
                     Falling back to defaults.",
                    e
                );
                return Self::default();
            }
        };
        let data = match std::fs::read_to_string(&path) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Self::default();
            }
            Err(e) => {
                log::warn!(
                    "[Rolo] Command Center: cannot read settings file at {}: {}. \
                     Falling back to defaults.",
                    path.display(),
                    e
                );
                return Self::default();
            }
        };
        match serde_json::from_str(&data) {
            Ok(s) => s,
            Err(e) => {
                log::warn!(
                    "[Rolo] Command Center: settings file at {} failed to parse: {}. \
                     Falling back to defaults (file left in place).",
                    path.display(),
                    e
                );
                Self::default()
            }
        }
    }

    /// Atomic write — write to `command_center.json.tmp`, fsync, rename. The
    /// rename is atomic on the same volume; `atomic_write` falls back to
    /// copy+remove on cross-device errors with a warn log. Returns an error
    /// string the Tauri command can surface to the frontend.
    pub fn save(&self, app: &tauri::AppHandle) -> Result<(), String> {
        let path = Self::path(app)?;
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Cannot serialize Command Center settings: {}", e))?;
        atomic_write(&path, json.as_bytes())
            .map_err(|e| format!("Cannot write Command Center settings to disk: {}", e))?;
        Ok(())
    }
}

/// Confirms a custom-endpoint URL template contains every required token.
/// Returns an error naming the FIRST missing token so the UI can highlight
/// it directly. Used by the Weather panel's Save validation (Phase 8).
#[allow(dead_code)] // Wired up in Phase 8 (weather-tab backend).
pub fn validate_url_template(template: &str) -> Result<(), String> {
    for token in ["{lat}", "{lon}", "{key}"] {
        if !template.contains(token) {
            return Err(format!("Missing required token: {}", token));
        }
    }
    Ok(())
}

/// Confirms a manual-location override falls inside Earth's coordinate
/// range. Lat is the more common failure (people enter degrees beyond 90),
/// so it's checked first.
#[allow(dead_code)] // Wired up in Phase 8 (weather-tab backend).
pub fn validate_lat_lon(lat: f64, lon: f64) -> Result<(), String> {
    if !(-90.0..=90.0).contains(&lat) {
        return Err(format!("Latitude must be between -90 and 90 (got {})", lat));
    }
    if !(-180.0..=180.0).contains(&lon) {
        return Err(format!(
            "Longitude must be between -180 and 180 (got {})",
            lon
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip_through_json() {
        let original = CommandCenterSettings::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: CommandCenterSettings = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, original);
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.weather.mode, WeatherMode::Default);
        assert_eq!(parsed.weather.response_preset, ResponsePreset::OpenMeteo);
        assert!(parsed.weather.manual_location.is_none());
    }

    #[test]
    fn unknown_top_level_keys_are_preserved() {
        // A future Rolo writes a key we don't know about — round-tripping
        // through this version must NOT drop it.
        let raw = r#"{
            "version": 1,
            "weather": { "mode": "default" },
            "future_panel": { "enabled": true, "threshold": 7 }
        }"#;
        let parsed: CommandCenterSettings = serde_json::from_str(raw).expect("parse");
        let reserialized = serde_json::to_string(&parsed).expect("reserialize");
        let reparsed: serde_json::Value = serde_json::from_str(&reserialized).expect("reparse");
        assert_eq!(
            reparsed["future_panel"]["enabled"],
            serde_json::Value::Bool(true)
        );
        assert_eq!(
            reparsed["future_panel"]["threshold"],
            serde_json::Value::from(7)
        );
    }

    #[test]
    fn unknown_weather_keys_are_preserved() {
        // Same contract but one level deeper — unknown keys inside the
        // `weather` object must survive too.
        let raw = r#"{
            "version": 1,
            "weather": {
                "mode": "default",
                "experimental_radar_url": "https://example.com/radar"
            }
        }"#;
        let parsed: CommandCenterSettings = serde_json::from_str(raw).expect("parse");
        let reserialized = serde_json::to_string(&parsed).expect("reserialize");
        let reparsed: serde_json::Value = serde_json::from_str(&reserialized).expect("reparse");
        assert_eq!(
            reparsed["weather"]["experimental_radar_url"],
            serde_json::Value::String("https://example.com/radar".to_string())
        );
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        // A minimal file written by an even older version should still parse.
        let raw = "{}";
        let parsed: CommandCenterSettings = serde_json::from_str(raw).expect("parse");
        assert_eq!(parsed, CommandCenterSettings::default());
    }

    #[test]
    fn url_template_accepts_all_tokens() {
        let ok = "https://api.example.com/wx?lat={lat}&lon={lon}&appid={key}";
        assert!(validate_url_template(ok).is_ok());
    }

    #[test]
    fn url_template_rejects_missing_lat() {
        let err = validate_url_template("https://api.example.com/wx?lon={lon}&appid={key}")
            .expect_err("must error");
        assert!(
            err.contains("{lat}"),
            "error should name the missing token: {}",
            err
        );
    }

    #[test]
    fn url_template_rejects_missing_lon() {
        let err = validate_url_template("https://api.example.com/wx?lat={lat}&appid={key}")
            .expect_err("must error");
        assert!(
            err.contains("{lon}"),
            "error should name the missing token: {}",
            err
        );
    }

    #[test]
    fn url_template_rejects_missing_key() {
        let err = validate_url_template("https://api.example.com/wx?lat={lat}&lon={lon}")
            .expect_err("must error");
        assert!(
            err.contains("{key}"),
            "error should name the missing token: {}",
            err
        );
    }

    #[test]
    fn lat_lon_accepts_in_range() {
        assert!(validate_lat_lon(40.6782, -73.9442).is_ok()); // Brooklyn-ish
        assert!(validate_lat_lon(-90.0, 180.0).is_ok());
        assert!(validate_lat_lon(90.0, -180.0).is_ok());
    }

    #[test]
    fn lat_lon_rejects_lat_out_of_range() {
        let err = validate_lat_lon(200.0, 0.0).expect_err("must error");
        assert!(err.contains("Latitude"), "got: {}", err);
    }

    #[test]
    fn lat_lon_rejects_lon_out_of_range() {
        let err = validate_lat_lon(0.0, -500.0).expect_err("must error");
        assert!(err.contains("Longitude"), "got: {}", err);
    }
}
