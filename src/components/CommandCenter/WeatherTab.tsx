/**
 * WeatherTab — weather-source configuration for Rolo's get_weather tool.
 *
 * Two-mode form:
 *  - "Use default" — Open-Meteo + ipwho.is, no fields. This is what every
 *    existing install gets and what Rolo falls back to silently if the
 *    custom path fails (PRD AC #18).
 *  - "Custom endpoint" — user provides a URL template with `{lat}`, `{lon}`,
 *    `{key}` placeholders, an API key, a response-shape preset, and
 *    optionally two JSONPath strings (for the "Custom (JSONPath)" preset).
 *
 * Either mode supports an optional manual location override (lat/lon
 * numeric inputs) so users whose IP geolocation is wrong can pin Rolo to
 * the right coords.
 *
 * Save is gated like the Brain tab: a custom-endpoint Save requires a
 * successful Test endpoint against the current form state, and any edit
 * invalidates that test. Switching back to "Use default" releases the gate.
 */

import { invoke } from "@tauri-apps/api/core";
import { useCallback, useEffect, useState } from "react";
import type { CommandCenterTabProps } from "./CommandCenter";
import type {
  CommandCenterSettings,
  ResponsePreset,
  TestWeatherRequest,
  TestWeatherResult,
  WeatherMode,
  WeatherStatus,
} from "../../types/commandCenter";
import { FormRow } from "./FormRow";

/**
 * Update a single `WeatherSettings` field, immutably, threading the
 * change through `props.setSettings`. The hook's snapshot-diff dirty
 * computation picks the edit up on the next render — no per-field
 * `markDirty("weather")` call is needed.
 */
type WeatherSettings = CommandCenterSettings["weather"];
function updateWeather<K extends keyof WeatherSettings>(
  setSettings: CommandCenterTabProps["setSettings"],
  key: K,
  value: WeatherSettings[K],
) {
  setSettings((prev) =>
    prev === null
      ? prev
      : { ...prev, weather: { ...prev.weather, [key]: value } },
  );
}

// Fallback coords used by the Test endpoint button when the user has NOT
// turned on manual location override. A test needs concrete lat/lon to
// substitute into the template — without IP geolocation in the panel we
// hardcode NYC and surface a notice so the user knows what's happening.
const TEST_FALLBACK_LAT = 40.7128;
const TEST_FALLBACK_LON = -74.006;

// Required placeholder tokens for a custom URL template. Kept in sync with
// `validate_url_template` in src-tauri/src/command_center/settings.rs —
// changes here must change both sides.
const REQUIRED_TOKENS = ["{lat}", "{lon}", "{key}"] as const;

interface PresetOption {
  value: ResponsePreset;
  label: string;
}

// The Open-Meteo preset isn't picked by the user from this dropdown — it's
// the implicit preset for "default" mode. Custom-endpoint mode picks one of
// the four below.
const PRESETS: ReadonlyArray<PresetOption> = [
  { value: "openweathermap", label: "OpenWeatherMap" },
  { value: "weatherapi", label: "WeatherAPI.com" },
  { value: "visualcrossing", label: "Visual Crossing" },
  { value: "custom_jsonpath", label: "Custom (JSONPath)" },
];

interface TestState {
  status: "idle" | "running" | "ok" | "fail";
  message: string;
  // Cached summary line ("72°F, partly cloudy") when status === "ok".
  summary: string;
}

/** Find the first required token missing from `template`, if any. */
function firstMissingToken(template: string): string | null {
  for (const token of REQUIRED_TOKENS) {
    if (!template.includes(token)) {
      return token;
    }
  }
  return null;
}

export default function WeatherTab(props: CommandCenterTabProps) {
  const { settings, markClean, markDirty, setSettings } = props;
  const w = settings.weather;

  // The Weather tab is driven directly from `settings.weather` — there is
  // no local form mirror. The hook's snapshot-diff dirty computation
  // notices any edit, and `discardAll` reloads the canonical settings
  // from disk, which naturally rolls the form back via the same path.
  const mode: WeatherMode = w.mode;
  const urlTemplate: string = w.custom_url_template;
  const apiKey: string = w.api_key;
  // The on-disk value of `response_preset` is "open_meteo" while in
  // "default" mode (the dropdown is hidden then), so coerce to the first
  // visible preset for display purposes when the dropdown is shown.
  const preset: ResponsePreset =
    w.response_preset === "open_meteo" ? "openweathermap" : w.response_preset;
  const jsonpathTemp: string = w.custom_jsonpath_temp;
  const jsonpathCondition: string = w.custom_jsonpath_condition;
  // Manual location stays as local form state for two reasons:
  //   (a) the lat/lon strings are mid-edit values that can't always be
  //       parsed to numbers (e.g., the user has typed "40." with no decimal
  //       yet) — settings.weather.manual_location requires real numbers.
  //   (b) toggling the checkbox alone (no coords typed) should still
  //       count as a dirty edit, which is hard to express via snapshot-
  //       diff without committing a placeholder `{lat:0, lon:0}`.
  // The sync effect below seeds the local strings whenever the canonical
  // settings reload (mount, discardAll, external open) so the "discard"
  // path is handled automatically — no separate listener needed.
  const [manualOn, setManualOn] = useState<boolean>(w.manual_location !== null);
  const [manualLat, setManualLat] = useState<string>(
    w.manual_location ? String(w.manual_location.lat) : "",
  );
  const [manualLon, setManualLon] = useState<string>(
    w.manual_location ? String(w.manual_location.lon) : "",
  );
  useEffect(() => {
    setManualOn(w.manual_location !== null);
    setManualLat(w.manual_location ? String(w.manual_location.lat) : "");
    setManualLon(w.manual_location ? String(w.manual_location.lon) : "");
  }, [w.manual_location]);

  // Wrap `updateWeather` once so each onChange site is a single call.
  const setW = useCallback(
    <K extends keyof WeatherSettings>(key: K, value: WeatherSettings[K]) =>
      updateWeather(setSettings, key, value),
    [setSettings],
  );

  const [test, setTest] = useState<TestState>({
    status: "idle",
    message: "",
    summary: "",
  });
  const [saving, setSaving] = useState(false);
  const [savedBanner, setSavedBanner] = useState<string | null>(null);
  const [statusRow, setStatusRow] = useState<WeatherStatus | null>(null);

  // Fade the saved banner after 3 seconds.
  useEffect(() => {
    if (!savedBanner) return;
    const t = setTimeout(() => setSavedBanner(null), 3000);
    return () => clearTimeout(t);
  }, [savedBanner]);

  // Pull the status indicator once on mount. Refresh after a successful Save
  // so the dot reflects the post-invalidation state.
  const refreshStatus = useCallback(() => {
    invoke<WeatherStatus>("cc_weather_status")
      .then((s) => setStatusRow(s))
      .catch(() => setStatusRow(null));
  }, []);
  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  // Any edit invalidates a passing test — mirrors Brain tab pattern.
  const invalidateTest = useCallback(() => {
    setTest((prev) =>
      prev.status === "idle" ? prev : { status: "idle", message: "", summary: "" },
    );
  }, []);

  // -----------------------------------------------------------------------
  // Validation — drives Save button disabled state and inline error text.
  // -----------------------------------------------------------------------

  const missingToken = mode === "custom" ? firstMissingToken(urlTemplate) : null;
  const urlTemplateError =
    mode === "custom" && urlTemplate.length > 0 && missingToken !== null
      ? `Missing required token: ${missingToken}`
      : null;

  const manualLatNum = manualLat === "" ? NaN : Number(manualLat);
  const manualLonNum = manualLon === "" ? NaN : Number(manualLon);
  const manualLatError =
    manualOn && manualLat !== "" && (Number.isNaN(manualLatNum) || manualLatNum < -90 || manualLatNum > 90)
      ? "Latitude must be between -90 and 90."
      : null;
  const manualLonError =
    manualOn && manualLon !== "" && (Number.isNaN(manualLonNum) || manualLonNum < -180 || manualLonNum > 180)
      ? "Longitude must be between -180 and 180."
      : null;
  const manualEmpty =
    manualOn && (manualLat === "" || manualLon === "");

  // Test button: disabled until the template has all tokens, API key present,
  // and either manual location is on with valid values (test uses those) OR
  // manual location is off (test uses NYC fallback).
  const testCoordsValid =
    !manualOn || (manualLat !== "" && manualLon !== "" && !manualLatError && !manualLonError);
  const canTest =
    mode === "custom" &&
    test.status !== "running" &&
    urlTemplate.length > 0 &&
    missingToken === null &&
    apiKey.length > 0 &&
    (preset !== "custom_jsonpath" ||
      (jsonpathTemp.length > 0 && jsonpathCondition.length > 0)) &&
    testCoordsValid;

  // Save button: in default mode, only manual-location validation matters.
  // In custom mode, the test must have succeeded for the current form state.
  let canSave = !saving;
  if (mode === "custom" && test.status !== "ok") canSave = false;
  if (manualOn && (manualLatError !== null || manualLonError !== null || manualEmpty)) {
    canSave = false;
  }
  if (mode === "custom" && (urlTemplate.length === 0 || apiKey.length === 0)) {
    canSave = false;
  }
  if (
    mode === "custom" &&
    preset === "custom_jsonpath" &&
    (jsonpathTemp.length === 0 || jsonpathCondition.length === 0)
  ) {
    canSave = false;
  }

  // -----------------------------------------------------------------------
  // Test endpoint
  // -----------------------------------------------------------------------

  const onTest = useCallback(async () => {
    setTest({ status: "running", message: "", summary: "" });
    const lat = manualOn && !Number.isNaN(manualLatNum) ? manualLatNum : TEST_FALLBACK_LAT;
    const lon = manualOn && !Number.isNaN(manualLonNum) ? manualLonNum : TEST_FALLBACK_LON;
    const req: TestWeatherRequest = {
      url_template: urlTemplate,
      api_key: apiKey,
      preset,
      lat,
      lon,
      jsonpath_temp: preset === "custom_jsonpath" ? jsonpathTemp : null,
      jsonpath_condition: preset === "custom_jsonpath" ? jsonpathCondition : null,
    };
    try {
      const result = await invoke<TestWeatherResult>("cc_test_weather", { req });
      if (result.ok && result.parsed) {
        const tempInt = Math.round(result.parsed.temp_f);
        const summary = `${tempInt}°F, ${result.parsed.condition_text}`;
        setTest({ status: "ok", message: "", summary });
      } else {
        setTest({
          status: "fail",
          message: result.error ?? "Test failed.",
          summary: "",
        });
      }
    } catch (err) {
      setTest({
        status: "fail",
        message: err instanceof Error ? err.message : String(err),
        summary: "",
      });
    }
  }, [
    apiKey,
    jsonpathCondition,
    jsonpathTemp,
    manualLatNum,
    manualLonNum,
    manualOn,
    preset,
    urlTemplate,
  ]);

  // -----------------------------------------------------------------------
  // Save
  // -----------------------------------------------------------------------

  const onSave = useCallback(async () => {
    if (!canSave) return;
    setSaving(true);
    try {
      // Build the full CommandCenterSettings to send: preserve top-level
      // shape (version + any unknown future keys handled server-side) and
      // overwrite only the `weather` slot. The shared `settings` came from
      // `cc_load_settings` so non-weather top-level keys flow through.
      //
      // The form fields other than manual lat/lon are read directly from
      // `settings.weather`, so the only "merge" work here is normalizing
      // defaulted-out fields (`custom_url_template` etc. blank out in
      // default mode) and packing the manual lat/lon strings back into a
      // `{lat, lon}` numeric pair.
      const next: CommandCenterSettings = {
        ...settings,
        weather: {
          mode,
          custom_url_template: mode === "custom" ? urlTemplate : "",
          api_key: mode === "custom" ? apiKey : "",
          response_preset: mode === "custom" ? preset : "open_meteo",
          custom_jsonpath_temp:
            mode === "custom" && preset === "custom_jsonpath" ? jsonpathTemp : "",
          custom_jsonpath_condition:
            mode === "custom" && preset === "custom_jsonpath"
              ? jsonpathCondition
              : "",
          manual_location: manualOn
            ? { lat: manualLatNum, lon: manualLonNum }
            : null,
        },
      };
      // Pass `next` explicitly so the hook saves and snapshots the
      // freshly-built shape in one shot — no React async race between
      // setSettings and saveSettings. The hook also writes `next` back
      // into its own settings state.
      await props.saveSettings(next);
      markClean("weather");
      setSavedBanner("Saved — Rolo will use this on his next weather check.");
      // The save emits `rolo://weather-config-changed`, which invalidates
      // the cache. Refresh the status row so the UI reflects the change.
      // (The cache is now empty → status returns amber.)
      refreshStatus();
    } catch (err) {
      setTest({
        status: "fail",
        message: `Save failed: ${err instanceof Error ? err.message : String(err)}`,
        summary: "",
      });
    } finally {
      setSaving(false);
    }
  }, [
    apiKey,
    canSave,
    jsonpathCondition,
    jsonpathTemp,
    manualLatNum,
    manualLonNum,
    manualOn,
    markClean,
    mode,
    preset,
    props,
    refreshStatus,
    settings,
    urlTemplate,
  ]);

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  return (
    <div className="cc-tab cc-tab-weather">
      <h2 className="cc-tab-title">Weather</h2>

      {statusRow && (
        <div className={`cc-weather-status cc-weather-status-${statusRow.status}`}>
          <span className={`cc-status-dot cc-status-dot-${statusRow.status}`} />
          <span className="cc-weather-status-label">
            {statusRow.status === "green" &&
              statusRow.last_summary &&
              statusRow.last_fetched_seconds_ago !== null && (
                <>
                  Weather endpoint is healthy. Last fetched{" "}
                  {Math.max(1, Math.round(statusRow.last_fetched_seconds_ago / 60))}{" "}
                  min ago: {statusRow.last_summary}.
                </>
              )}
            {statusRow.status === "amber" &&
              "No weather fetched this session yet."}
            {statusRow.status === "red" && "Last weather fetch is stale."}
          </span>
        </div>
      )}

      {savedBanner && <div className="cc-success-banner">{savedBanner}</div>}

      <div className="cc-form-row cc-radio-row">
        <label className="cc-radio-option">
          <input
            type="radio"
            name="weather-mode"
            value="default"
            checked={mode === "default"}
            onChange={() => {
              setW("mode", "default");
              invalidateTest();
            }}
          />
          <span>Use default (Open-Meteo + IP geolocation)</span>
        </label>
        <label className="cc-radio-option">
          <input
            type="radio"
            name="weather-mode"
            value="custom"
            checked={mode === "custom"}
            onChange={() => {
              setW("mode", "custom");
              invalidateTest();
            }}
          />
          <span>Custom endpoint</span>
        </label>
      </div>

      {mode === "custom" && (
        <>
          <FormRow
            label="URL template"
            htmlFor="weather-url-template"
            help={
              <>
                Use <code>{"{lat}"}</code>, <code>{"{lon}"}</code>,{" "}
                <code>{"{key}"}</code> placeholders.
              </>
            }
            error={urlTemplateError || undefined}
          >
            <input
              id="weather-url-template"
              type="text"
              className="cc-form-input"
              value={urlTemplate}
              onChange={(e) => {
                setW("custom_url_template", e.target.value);
                invalidateTest();
              }}
              placeholder="https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=imperial"
              autoComplete="off"
            />
          </FormRow>

          <FormRow label="API key" htmlFor="weather-api-key">
            <input
              id="weather-api-key"
              type="password"
              className="cc-form-input"
              value={apiKey}
              onChange={(e) => {
                setW("api_key", e.target.value);
                invalidateTest();
              }}
              autoComplete="off"
            />
          </FormRow>

          <FormRow label="Response shape preset" htmlFor="weather-preset">
            <select
              id="weather-preset"
              className="cc-form-select"
              value={preset}
              onChange={(e) => {
                setW("response_preset", e.target.value as ResponsePreset);
                invalidateTest();
              }}
            >
              {PRESETS.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
            </select>
          </FormRow>

          {preset === "custom_jsonpath" && (
            <>
              <FormRow
                label="JSONPath to temperature"
                htmlFor="weather-jsonpath-temp"
              >
                <input
                  id="weather-jsonpath-temp"
                  type="text"
                  className="cc-form-input"
                  value={jsonpathTemp}
                  onChange={(e) => {
                    setW("custom_jsonpath_temp", e.target.value);
                    invalidateTest();
                  }}
                  placeholder="$.main.temp"
                  autoComplete="off"
                />
              </FormRow>
              <FormRow
                label="JSONPath to condition text"
                htmlFor="weather-jsonpath-condition"
                help="v1 supports flat paths only — no arrays, wildcards, or filters."
              >
                <input
                  id="weather-jsonpath-condition"
                  type="text"
                  className="cc-form-input"
                  value={jsonpathCondition}
                  onChange={(e) => {
                    setW("custom_jsonpath_condition", e.target.value);
                    invalidateTest();
                  }}
                  placeholder="$.weather[0].description (arrays not supported in v1)"
                  autoComplete="off"
                />
              </FormRow>
            </>
          )}
        </>
      )}

      <div className="cc-form-row">
        <label className="cc-checkbox-option">
          <input
            type="checkbox"
            checked={manualOn}
            onChange={(e) => {
              setManualOn(e.target.checked);
              markDirty("weather");
              invalidateTest();
            }}
          />
          <span>Set my location manually</span>
        </label>
      </div>

      {manualOn && (
        <>
          <FormRow
            label="Latitude"
            htmlFor="weather-manual-lat"
            className="cc-form-row-inline"
          >
            <input
              id="weather-manual-lat"
              type="number"
              step="0.0001"
              className="cc-form-input cc-form-input-narrow"
              value={manualLat}
              onChange={(e) => {
                setManualLat(e.target.value);
                markDirty("weather");
                invalidateTest();
              }}
              placeholder="40.7128"
            />
          </FormRow>
          {manualLatError && <p className="cc-form-error">{manualLatError}</p>}
          <FormRow
            label="Longitude"
            htmlFor="weather-manual-lon"
            className="cc-form-row-inline"
          >
            <input
              id="weather-manual-lon"
              type="number"
              step="0.0001"
              className="cc-form-input cc-form-input-narrow"
              value={manualLon}
              onChange={(e) => {
                setManualLon(e.target.value);
                markDirty("weather");
                invalidateTest();
              }}
              placeholder="-74.0060"
            />
          </FormRow>
          {manualLonError && <p className="cc-form-error">{manualLonError}</p>}
        </>
      )}

      <div className="cc-form-actions">
        {mode === "custom" && (
          <button
            type="button"
            className="cc-button cc-button-test"
            onClick={onTest}
            disabled={!canTest}
            title={
              canTest
                ? manualOn
                  ? "Probe the endpoint with the manual coordinates"
                  : "Probe the endpoint with NYC fallback coords (40.7, -74.0)"
                : "Fill in the URL template (with all three placeholders) and API key first"
            }
          >
            {test.status === "running" ? "Testing…" : "Test endpoint"}
          </button>
        )}
        <button
          type="button"
          className="cc-button cc-button-primary"
          onClick={onSave}
          disabled={!canSave}
          title={
            canSave
              ? "Persist these weather settings"
              : mode === "custom" && test.status !== "ok"
                ? "Run Test endpoint successfully before saving"
                : "Resolve the errors above before saving"
          }
        >
          {saving ? "Saving…" : "Save"}
        </button>
      </div>

      {mode === "custom" && test.status !== "idle" && (
        <div className="cc-test-status-row">
          {test.status === "running" && (
            <span className="cc-test-status-pending">Testing endpoint…</span>
          )}
          {test.status === "ok" && (
            <span className="cc-test-status-ok">✓ {test.summary}</span>
          )}
          {test.status === "fail" && (
            <span className="cc-test-status-fail">✗ {test.message}</span>
          )}
        </div>
      )}

      {mode === "custom" && !manualOn && (
        <p className="cc-form-helper cc-form-helper-muted">
          Test endpoint will use New York coords ({TEST_FALLBACK_LAT},{" "}
          {TEST_FALLBACK_LON}) since manual location is off.
        </p>
      )}
    </div>
  );
}
