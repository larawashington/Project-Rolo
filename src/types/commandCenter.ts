/**
 * TypeScript mirror of `CommandCenterSettings` in
 * `src-tauri/src/command_center/settings.rs`.
 *
 * Field names use snake_case to match Rust's serde defaults — Tauri's invoke
 * bridge round-trips JSON literally, so any renaming here would silently
 * desync the frontend from the backend schema.
 *
 * The Rust structs also carry a flattened `extra: serde_json::Map<...>` to
 * preserve unknown keys across downgrades. We intentionally omit it from
 * these types: the frontend never edits or reads it, and TypeScript's
 * structural typing allows it to pass through state untouched.
 */

export type ProviderName =
  | "ollama"
  | "openai_compat"
  | "anthropic"
  | "gemini"
  | "huggingface"
  | "deepinfra";

export type WeatherMode = "default" | "custom";

export type ResponsePreset =
  | "open_meteo"
  | "openweathermap"
  | "weatherapi"
  | "visualcrossing"
  | "custom_jsonpath";

export interface ManualLocation {
  lat: number;
  lon: number;
}

export interface WeatherSettings {
  mode: WeatherMode;
  custom_url_template: string;
  api_key: string;
  response_preset: ResponsePreset;
  custom_jsonpath_temp: string;
  custom_jsonpath_condition: string;
  manual_location: ManualLocation | null;
}

export interface CommandCenterSettings {
  version: number;
  weather: WeatherSettings;
}

export type CommandCenterTab = "brain" | "memory" | "perception" | "weather";

// ---------------------------------------------------------------------------
// Diagnostics — Perception tab (Phase 3)
// ---------------------------------------------------------------------------

/**
 * Mirrors the Rust `ProbeStatus` enum. Rust serializes with
 * `#[serde(rename_all = "snake_case")]`, so the wire values are lowercase.
 * Keep these in lockstep — `index.css` keys color classes off the string.
 */
export type ProbeStatus = "green" | "amber" | "red" | "grey";

/**
 * One row in the Perception tab. `fix_hint` is `null` when the status is
 * Green (or for the "Coming soon" greys). `latency_ms` is populated only
 * for the live HTTP probes (Ollama, embedder) when they actually fired.
 */
export interface ProbeOutcome {
  id: string;
  label: string;
  status: ProbeStatus;
  message: string;
  fix_hint: string | null;
  latency_ms: number | null;
}

/**
 * Full output of `cc_run_diagnostics`. `probes` is ordered — the frontend
 * renders it in the order received without re-sorting.
 */
export interface DiagnosticsReport {
  probes: ProbeOutcome[];
  generated_at_ms: number;
}

// ---------------------------------------------------------------------------
// Brain tab — Phase 5 (mirrors src-tauri/src/chat/config.rs)
// ---------------------------------------------------------------------------

/**
 * The six values the Brain dropdown can take. HF and DeepInfra are stored
 * with `provider: "openai_compat"` on disk; `ui_provider` is what restores
 * the dropdown position after a reload.
 */
export type UiProvider =
  | "ollama"
  | "openai_compat"
  | "anthropic"
  | "gemini"
  | "huggingface"
  | "deepinfra";

export interface OllamaConfig {
  base_url: string;
  model: string;
  context_window: number;
}

export interface OpenAICompatConfig {
  base_url: string;
  api_key: string;
  model: string;
  context_window: number;
}

export interface AnthropicConfig {
  api_key: string;
  model: string;
  context_window: number;
}

export interface GeminiConfig {
  api_key: string;
  model: string;
  context_window: number;
}

export interface InferenceConfig {
  provider: ProviderName;
  ui_provider: UiProvider | null;
  ollama: OllamaConfig;
  openai_compat: OpenAICompatConfig;
  anthropic: AnthropicConfig;
  gemini: GeminiConfig;
}

export interface ChatConfig {
  inference: InferenceConfig;
}

/** Mirror of the Rust `TestBrainResult`. */
export interface TestBrainResult {
  ok: boolean;
  message: string;
  latency_ms: number;
}

// ---------------------------------------------------------------------------
// Memory tab — Phase 7 (mirrors src-tauri/src/vault/user_profile.rs and
// src-tauri/src/commands.rs Memory commands)
// ---------------------------------------------------------------------------

/** Mirrors Rust `UserProfileCategory` (serde snake_case). */
export type UserProfileCategory = "about" | "people" | "response_style";

/** On-disk shape of `<vault_root>/user_profile.json`. */
export interface UserProfile {
  version: number;
  about_you: string;
  people_and_context: string;
  how_rolo_should_respond: string;
  /** RFC 3339 timestamp string — chrono::DateTime<Local> on the Rust side. */
  updated_at: string;
}

/** Request body for `cc_save_memory_and_sleep`. */
export interface SaveMemoryRequest {
  about_you: string;
  people_and_context: string;
  how_rolo_should_respond: string;
}

/** Mirrors Rust `SaveMemoryStatus` (serde snake_case). */
export type SaveMemoryStatus = "success" | "skipped" | "failed" | "cancelled";

/** Result of `cc_save_memory_and_sleep`. */
export interface SaveMemoryResult {
  status: SaveMemoryStatus;
  reason: string | null;
  profile: UserProfile;
}

/** Result of `cc_load_user_profile`. */
export interface LoadProfileResult {
  profile: UserProfile | null;
  learned_dream_titles: string[];
}

/** Mirrors Rust `ClearProfileStatus` (serde snake_case). */
export type ClearProfileStatus = "ok" | "cancel_then_clear";

/** Result of `cc_clear_user_profile`. */
export interface ClearProfileResult {
  status: ClearProfileStatus;
  dreams_removed: number;
}

/**
 * Payload of the `rolo://command-center-dream-progress` event. Fired by
 * `cc_save_memory_and_sleep` at each stage boundary so the SleepOverlay can
 * update without polling.
 */
export interface DreamProgressEvent {
  stage: "writing_memory" | "indexing" | "dreaming" | "waking_up";
  label: string;
}

// ---------------------------------------------------------------------------
// Weather tab — Phase 8 (mirrors src-tauri/src/tools/weather_endpoint.rs and
// the cc_test_weather / cc_weather_status commands)
// ---------------------------------------------------------------------------

/** Mirror of the Rust `ParsedWeather` struct. */
export interface ParsedWeather {
  temp_f: number;
  condition_text: string;
  weather_code: number | null;
}

/** Request body for `cc_test_weather`. */
export interface TestWeatherRequest {
  url_template: string;
  api_key: string;
  preset: string;
  lat: number;
  lon: number;
  jsonpath_temp: string | null;
  jsonpath_condition: string | null;
}

/** Result of `cc_test_weather` — never throws; failures surface as `ok=false`. */
export interface TestWeatherResult {
  ok: boolean;
  parsed: ParsedWeather | null;
  error: string | null;
}

/** Result of `cc_weather_status` — drives the top-of-tab status dot. */
export interface WeatherStatus {
  status: "green" | "amber" | "red";
  last_summary: string | null;
  last_fetched_seconds_ago: number | null;
}
