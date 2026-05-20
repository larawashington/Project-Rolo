/**
 * BrainTab — provider configuration for Rolo's LLM brain.
 *
 * Lets the user pick Rolo's LLM brain. Six provider families are supported:
 * Ollama (local), OpenAI-compatible, Anthropic, Google Gemini, Hugging Face
 * Inference, and DeepInfra. HF and DeepInfra are presented as their own
 * dropdown choices but are stored on disk as `provider: "openai_compat"`
 * with a backend-resolved base URL — the `ui_provider` field round-trips
 * the dropdown choice so reloads restore the right form.
 *
 * Save is gated on a successful Test Connection in the current session.
 * Any form edit after a passing test resets that gate so credentials are
 * always tested against the values that will actually be saved.
 *
 * Tool routing and idle speech are Ollama-only today — those notices live
 * at the bottom of the form and are always visible.
 */

import { invoke } from "@tauri-apps/api/core";
import { openUrl } from "@tauri-apps/plugin-opener";
import { useTauriEvent } from "../../hooks/useTauriEvent";
import { useCallback, useEffect, useRef, useState } from "react";
import type { CommandCenterTabProps } from "./CommandCenter";
import type {
  ChatConfig,
  ProviderName,
  TestBrainResult,
  UiProvider,
} from "../../types/commandCenter";
import { FormRow } from "./FormRow";

// The Rolo Brain tag (HF-hosted fine-tune) is the identifier we pull via
// Ollama. Friendly name is "Rolo Brain"; the full tag only appears in code
// blocks and the persisted model field. Must stay in sync with
// `PRIMARY_MODEL` in src-tauri/src/ollama.rs.
const ROLO_BRAIN_MODEL = "hf.co/larawashington/rolo-brain";

/** Wizard state for the first-run pull walkthrough. */
type PullState =
  | { kind: "checking" }
  | { kind: "not_reachable"; error: string }
  | { kind: "ready_to_pull"; alreadyInstalled: boolean }
  | {
      kind: "pulling";
      percent: number;
      completedBytes: number;
      totalBytes: number;
      statusText: string;
    }
  | { kind: "pull_failed"; error: string }
  | { kind: "pulled" };

interface PullProgressEvent {
  status: string;
  digest?: string;
  total?: number;
  completed?: number;
}

interface ModelStatus {
  ollama_reachable: boolean;
  model_present: boolean;
  error?: string;
}

function formatMb(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(0);
}

// Backend presets — these strings must match the constants in
// src-tauri/src/commands.rs (HF_INFERENCE_BASE_URL, DEEPINFRA_BASE_URL). We
// echo them into the saved config's `openai_compat.base_url` so the
// dispatcher and chat engine pick up the right endpoint at runtime.
const HF_BASE_URL = "https://api-inference.huggingface.co/v1";
const DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai";

interface ProviderOption {
  value: UiProvider;
  label: string;
}

const PROVIDER_OPTIONS: ReadonlyArray<ProviderOption> = [
  { value: "ollama", label: "Ollama (local)" },
  { value: "openai_compat", label: "OpenAI-compatible" },
  { value: "anthropic", label: "Anthropic" },
  { value: "gemini", label: "Google Gemini" },
  { value: "huggingface", label: "Hugging Face Inference" },
  { value: "deepinfra", label: "DeepInfra" },
];

// Form state — one shape for every provider; conditional rendering picks
// which fields are visible. Defaults match the Rust struct defaults so a
// fresh launch produces the same starting point on either side.
interface BrainFormState {
  uiProvider: UiProvider;
  // Ollama
  ollamaBaseUrl: string;
  ollamaModel: string;
  // OpenAI-compatible
  openaiBaseUrl: string;
  openaiApiKey: string;
  openaiModel: string;
  // Anthropic
  anthropicApiKey: string;
  anthropicModel: string;
  // Gemini
  geminiApiKey: string;
  geminiModel: string;
  // Hugging Face Inference (stored as openai_compat at save time)
  hfApiKey: string;
  hfModel: string;
  // DeepInfra (stored as openai_compat at save time)
  deepinfraApiKey: string;
  deepinfraModel: string;
}

function defaultFormState(): BrainFormState {
  return {
    uiProvider: "ollama",
    ollamaBaseUrl: "http://localhost:11434",
    ollamaModel: ROLO_BRAIN_MODEL,
    openaiBaseUrl: "",
    openaiApiKey: "",
    openaiModel: "",
    anthropicApiKey: "",
    anthropicModel: "claude-haiku-4-5-20251001",
    geminiApiKey: "",
    geminiModel: "gemini-2.0-flash",
    hfApiKey: "",
    hfModel: "",
    deepinfraApiKey: "",
    deepinfraModel: "",
  };
}

/** Convert a loaded ChatConfig into the form state. */
function formFromConfig(cfg: ChatConfig): BrainFormState {
  const base = defaultFormState();
  const inf = cfg.inference;
  // Restore the dropdown from ui_provider if set; otherwise fall back to the
  // runtime provider field (legacy configs from before Phase 5).
  const ui: UiProvider = inf.ui_provider ?? inf.provider;
  base.uiProvider = PROVIDER_OPTIONS.some((o) => o.value === ui)
    ? ui
    : "ollama";
  base.ollamaBaseUrl = inf.ollama.base_url || base.ollamaBaseUrl;
  base.ollamaModel = inf.ollama.model || base.ollamaModel;
  base.openaiBaseUrl = inf.openai_compat.base_url;
  base.openaiApiKey = inf.openai_compat.api_key;
  base.openaiModel = inf.openai_compat.model;
  base.anthropicApiKey = inf.anthropic.api_key;
  base.anthropicModel = inf.anthropic.model || base.anthropicModel;
  base.geminiApiKey = inf.gemini.api_key;
  base.geminiModel = inf.gemini.model || base.geminiModel;
  // HF and DeepInfra share openai_compat fields on disk — the ui_provider
  // tells us which preset the user was on, and we lift their key/model
  // back into their dedicated form fields.
  if (ui === "huggingface") {
    base.hfApiKey = inf.openai_compat.api_key;
    base.hfModel = inf.openai_compat.model;
  } else if (ui === "deepinfra") {
    base.deepinfraApiKey = inf.openai_compat.api_key;
    base.deepinfraModel = inf.openai_compat.model;
  }
  return base;
}

/** Marshal the form back into a ChatConfig the backend can persist. */
function formToConfig(form: BrainFormState): ChatConfig {
  // Resolve UI choice → runtime provider string. HF / DeepInfra collapse
  // into openai_compat with the preset base URL. The ui_provider field
  // round-trips the original choice so the dropdown restores on reload.
  let runtimeProvider: ProviderName;
  let openaiBaseUrl = form.openaiBaseUrl;
  let openaiApiKey = form.openaiApiKey;
  let openaiModel = form.openaiModel;

  switch (form.uiProvider) {
    case "ollama":
      runtimeProvider = "ollama";
      break;
    case "anthropic":
      runtimeProvider = "anthropic";
      break;
    case "gemini":
      runtimeProvider = "gemini";
      break;
    case "openai_compat":
      runtimeProvider = "openai_compat";
      break;
    case "huggingface":
      runtimeProvider = "openai_compat";
      openaiBaseUrl = HF_BASE_URL;
      openaiApiKey = form.hfApiKey;
      openaiModel = form.hfModel;
      break;
    case "deepinfra":
      runtimeProvider = "openai_compat";
      openaiBaseUrl = DEEPINFRA_BASE_URL;
      openaiApiKey = form.deepinfraApiKey;
      openaiModel = form.deepinfraModel;
      break;
  }

  return {
    inference: {
      provider: runtimeProvider,
      ui_provider: form.uiProvider,
      ollama: {
        base_url: form.ollamaBaseUrl,
        model: form.ollamaModel,
        context_window: 4096,
      },
      openai_compat: {
        base_url: openaiBaseUrl,
        api_key: openaiApiKey,
        model: openaiModel,
        context_window: 4096,
      },
      anthropic: {
        api_key: form.anthropicApiKey,
        model: form.anthropicModel,
        context_window: 4096,
      },
      gemini: {
        api_key: form.geminiApiKey,
        model: form.geminiModel,
        context_window: 4096,
      },
    },
  };
}

/** Pull the (provider, base_url, api_key, model) tuple needed by cc_test_brain. */
function testRequestFromForm(form: BrainFormState): {
  provider: string;
  base_url: string | null;
  api_key: string | null;
  model: string;
} {
  switch (form.uiProvider) {
    case "ollama":
      return {
        provider: "ollama",
        base_url: form.ollamaBaseUrl,
        api_key: null,
        model: form.ollamaModel,
      };
    case "openai_compat":
      return {
        provider: "openai_compat",
        base_url: form.openaiBaseUrl,
        api_key: form.openaiApiKey,
        model: form.openaiModel,
      };
    case "anthropic":
      return {
        provider: "anthropic",
        base_url: null,
        api_key: form.anthropicApiKey,
        model: form.anthropicModel,
      };
    case "gemini":
      return {
        provider: "gemini",
        base_url: null,
        api_key: form.geminiApiKey,
        model: form.geminiModel,
      };
    case "huggingface":
      return {
        provider: "huggingface",
        base_url: null,
        api_key: form.hfApiKey,
        model: form.hfModel,
      };
    case "deepinfra":
      return {
        provider: "deepinfra",
        base_url: null,
        api_key: form.deepinfraApiKey,
        model: form.deepinfraModel,
      };
  }
}

interface TestState {
  status: "idle" | "running" | "ok" | "fail";
  message: string;
  latencyMs: number;
}

export default function BrainTab(props: CommandCenterTabProps) {
  const { markDirty, markClean, refreshBrainNeedsSetup, brainNeedsSetup } =
    props;
  const [form, setForm] = useState<BrainFormState>(defaultFormState());
  const [loaded, setLoaded] = useState(false);
  const [test, setTest] = useState<TestState>({
    status: "idle",
    message: "",
    latencyMs: 0,
  });
  const [savedBanner, setSavedBanner] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  // In first-run setup mode the walkthrough is auto-expanded and promoted;
  // for already-configured users it remains a collapsible accordion.
  const [showSetup, setShowSetup] = useState(brainNeedsSetup);
  const [pullState, setPullState] = useState<PullState>({ kind: "checking" });
  // Tracks the per-layer byte totals so the aggregate progress bar can sum
  // `completed` and `total` across layers. Reset on each new pull attempt.
  const layerProgressRef = useRef<
    Map<string, { completed: number; total: number }>
  >(new Map());
  // AbortController-style cancel for the in-flight pull. Dropping the
  // promise on the React side doesn't cancel the Rust future, so we
  // signal via a boolean ref the listener consults to ignore late events
  // and a Tauri command would do nothing for us here — the actual cancel
  // is performed by letting Rust's future drop. We achieve that by
  // re-invoking with a sentinel? Simpler: we just tear down the listener
  // and ignore subsequent events. Ollama keeps partial layers, so retry
  // resumes. See PRD Cancel section.
  const pullCancelledRef = useRef(false);

  // One-way: open on fail, never auto-close on success — user controls collapse.
  useEffect(() => {
    if (test.status === "fail") {
      setShowSetup(true);
    }
  }, [test.status]);

  // Setup-mode flips: auto-expand when entering, leave it alone after the
  // user exits (no auto-collapse).
  useEffect(() => {
    if (brainNeedsSetup) {
      setShowSetup(true);
    }
  }, [brainNeedsSetup]);

  // Initial load — pull existing config so the dropdown and fields restore.
  const reloadFromDisk = useCallback(() => {
    let cancelled = false;
    invoke<ChatConfig>("cc_load_chat_config")
      .then((cfg) => {
        if (!cancelled) {
          setForm(formFromConfig(cfg));
          setLoaded(true);
          // Reloading from disk = freshly-canonical state. Reset the
          // test gate so the user must re-test before saving.
          setTest({ status: "idle", message: "", latencyMs: 0 });
        }
      })
      .catch((err) => {
        // Defaults are still usable; surface the error in the test slot
        // (the closest visible failure channel).
        if (!cancelled) {
          setTest({
            status: "fail",
            message: `Failed to load config: ${
              err instanceof Error ? err.message : String(err)
            }`,
            latencyMs: 0,
          });
          setLoaded(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const cancel = reloadFromDisk();
    return cancel;
  }, [reloadFromDisk]);

  // Wipes in-memory edits and reloads the saved config. Fires from
  // useCommandCenter.discardAll() on the close-modal "Discard" button. We
  // listen here (not on `command-center-opened`) so the external re-open
  // path keeps its "don't clobber unsaved edits" semantics.
  useTauriEvent("rolo://command-center-discard", () => {
    reloadFromDisk();
  });

  // Fade the saved banner after 3 seconds.
  useEffect(() => {
    if (!savedBanner) return;
    const t = setTimeout(() => setSavedBanner(null), 3000);
    return () => clearTimeout(t);
  }, [savedBanner]);

  // Any field edit invalidates a previous successful test — the user must
  // re-test against the new values before Save is enabled again. Each edit
  // also flips the tab's dirty flag so the close-handler raises a discard
  // prompt if the user tries to close without saving.
  const updateForm = useCallback(
    <K extends keyof BrainFormState>(key: K, value: BrainFormState[K]) => {
      setForm((prev) => ({ ...prev, [key]: value }));
      setTest((prev) =>
        prev.status === "idle" ? prev : { status: "idle", message: "", latencyMs: 0 },
      );
      markDirty("brain");
    },
    [markDirty],
  );

  const onTest = useCallback(async () => {
    setTest({ status: "running", message: "", latencyMs: 0 });
    try {
      const req = testRequestFromForm(form);
      const result = await invoke<TestBrainResult>("cc_test_brain", { req });
      if (result.ok) {
        setTest({
          status: "ok",
          message: result.message,
          latencyMs: result.latency_ms,
        });
      } else {
        setTest({
          status: "fail",
          message: result.message,
          latencyMs: result.latency_ms,
        });
      }
    } catch (err) {
      setTest({
        status: "fail",
        message: err instanceof Error ? err.message : String(err),
        latencyMs: 0,
      });
    }
  }, [form]);

  const onSave = useCallback(async () => {
    if (test.status !== "ok") return;
    setSaving(true);
    try {
      const config = formToConfig(form);
      await invoke("cc_apply_brain", { config });
      setSavedBanner("Saved — Rolo's brain has been swapped.");
      // Clear the brain tab's dirty flag on a successful save.
      markClean("brain");
      // A successful Brain save may have released the modal
      // setup gate. Re-query so the sidebar tabs re-enable immediately.
      // Fire-and-forget — failures degrade gracefully (next CC open will
      // re-check anyway), and we don't want to block the saved-banner.
      void refreshBrainNeedsSetup();
    } catch (err) {
      setTest({
        status: "fail",
        message: `Save failed: ${
          err instanceof Error ? err.message : String(err)
        }`,
        latencyMs: 0,
      });
    } finally {
      setSaving(false);
    }
  }, [form, markClean, refreshBrainNeedsSetup, test.status]);

  // ---------------------------------------------------------------------
  // Rolo Brain first-run pull wizard
  // ---------------------------------------------------------------------

  // Probe `/api/tags` to learn whether Ollama is up and whether Rolo Brain
  // is already installed. Re-runs when the user edits the Ollama base URL
  // or switches into the Ollama provider, and polls every 3s while we're
  // in `not_reachable` so a freshly-started Ollama lifts the wizard out
  // of the waiting state without a manual refresh.
  const probeModelStatus = useCallback(async () => {
    try {
      const status = await invoke<ModelStatus>("cc_check_brain_model", {
        baseUrl: form.ollamaBaseUrl,
        model: ROLO_BRAIN_MODEL,
      });
      if (!status.ollama_reachable) {
        setPullState({
          kind: "not_reachable",
          error: status.error ?? "Ollama is not reachable.",
        });
        return;
      }
      if (status.model_present) {
        setPullState({ kind: "ready_to_pull", alreadyInstalled: true });
      } else {
        setPullState({ kind: "ready_to_pull", alreadyInstalled: false });
      }
    } catch (err) {
      setPullState({
        kind: "not_reachable",
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }, [form.ollamaBaseUrl]);

  // Initial probe + base-URL change. We don't probe while a pull is in
  // flight (that would clobber the progress state).
  useEffect(() => {
    if (form.uiProvider !== "ollama") return;
    if (pullState.kind === "pulling" || pullState.kind === "pulled") return;
    void probeModelStatus();
    // We want this to re-run on URL changes too; probeModelStatus already
    // depends on it via useCallback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form.uiProvider, form.ollamaBaseUrl, probeModelStatus]);

  // Background poll while waiting for Ollama to come up.
  useEffect(() => {
    if (pullState.kind !== "not_reachable") return;
    const id = setInterval(() => {
      void probeModelStatus();
    }, 3000);
    return () => clearInterval(id);
  }, [pullState.kind, probeModelStatus]);

  // Listen for streamed pull progress. We aggregate `completed` and `total`
  // across layers so the bar reflects overall download progress, not
  // per-layer. Late events arriving after a Cancel are ignored.
  useTauriEvent<PullProgressEvent>("rolo://brain-pull-progress", (payload) => {
    if (pullCancelledRef.current) return;
    if (payload.digest && typeof payload.completed === "number") {
      const total =
        typeof payload.total === "number"
          ? payload.total
          : layerProgressRef.current.get(payload.digest)?.total ?? 0;
      layerProgressRef.current.set(payload.digest, {
        completed: payload.completed,
        total,
      });
    }
    let completedSum = 0;
    let totalSum = 0;
    for (const v of layerProgressRef.current.values()) {
      completedSum += v.completed;
      totalSum += v.total;
    }
    const percent = totalSum > 0 ? Math.min(100, (completedSum / totalSum) * 100) : 0;
    setPullState((prev) => {
      if (prev.kind !== "pulling") return prev;
      return {
        kind: "pulling",
        percent,
        completedBytes: completedSum,
        totalBytes: totalSum,
        statusText: payload.status || prev.statusText,
      };
    });
  });

  const onPullClick = useCallback(async () => {
    pullCancelledRef.current = false;
    layerProgressRef.current.clear();
    setPullState({
      kind: "pulling",
      percent: 0,
      completedBytes: 0,
      totalBytes: 0,
      statusText: "Starting…",
    });
    try {
      await invoke("cc_pull_brain_model", {
        baseUrl: form.ollamaBaseUrl,
        model: ROLO_BRAIN_MODEL,
      });
      if (pullCancelledRef.current) {
        // Cancel path already handled by onCancelClick; do nothing here.
        return;
      }
      setPullState({ kind: "pulled" });
    } catch (err) {
      if (pullCancelledRef.current) {
        return;
      }
      setPullState({
        kind: "pull_failed",
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }, [form.ollamaBaseUrl]);

  const onCancelClick = useCallback(async () => {
    pullCancelledRef.current = true;
    try {
      await invoke("cc_cancel_brain_pull");
    } catch (err) {
      // Swallow — the cancel command is best-effort. The UI flips back
      // regardless; Ollama keeps any partial layers it had written.
      // eslint-disable-next-line no-console
      console.warn("[Rolo] cc_cancel_brain_pull failed:", err);
    }
    layerProgressRef.current.clear();
    setPullState({ kind: "ready_to_pull", alreadyInstalled: false });
  }, []);

  // Pull completed → during first-run setup mode, auto-run the test and
  // (on pass) auto-apply the brain config. For already-configured users
  // this still runs the test (helpful confirmation) but does NOT
  // auto-save — they keep their explicit Save click via the existing UI.
  useEffect(() => {
    if (pullState.kind !== "pulled") return;

    let cancelled = false;
    const runAutoFlow = async () => {
      // Make sure the Ollama form fields point at Rolo Brain so the test
      // and (potential) save target the right model. We mutate form
      // in-place via updateForm so the dirty-flag semantics are honoured
      // for non-setup users (they'll see Rolo Brain pre-filled in the
      // Model field and a clear Save button).
      if (form.ollamaModel !== ROLO_BRAIN_MODEL) {
        setForm((prev) => ({ ...prev, ollamaModel: ROLO_BRAIN_MODEL }));
      }
      if (form.uiProvider !== "ollama") {
        setForm((prev) => ({ ...prev, uiProvider: "ollama" }));
      }

      // Run the same probe Save normally requires.
      setTest({ status: "running", message: "", latencyMs: 0 });
      try {
        const result = await invoke<TestBrainResult>("cc_test_brain", {
          req: {
            provider: "ollama",
            base_url: form.ollamaBaseUrl,
            api_key: null,
            model: ROLO_BRAIN_MODEL,
          },
        });
        if (cancelled) return;
        if (!result.ok) {
          setTest({
            status: "fail",
            message: result.message,
            latencyMs: result.latency_ms,
          });
          return;
        }
        setTest({
          status: "ok",
          message: result.message,
          latencyMs: result.latency_ms,
        });
        if (!brainNeedsSetup) return;
        // First-run: auto-save so the setup gate releases without a
        // further click.
        setSaving(true);
        try {
          const cfg: ChatConfig = {
            inference: {
              provider: "ollama",
              ui_provider: "ollama",
              ollama: {
                base_url: form.ollamaBaseUrl,
                model: ROLO_BRAIN_MODEL,
                context_window: 4096,
              },
              openai_compat: {
                base_url: "",
                api_key: "",
                model: "",
                context_window: 4096,
              },
              anthropic: {
                api_key: "",
                model: "claude-haiku-4-5-20251001",
                context_window: 4096,
              },
              gemini: {
                api_key: "",
                model: "gemini-2.0-flash",
                context_window: 4096,
              },
            },
          };
          await invoke("cc_apply_brain", { config: cfg });
          if (cancelled) return;
          setSavedBanner("Rolo Brain is ready.");
          markClean("brain");
          void refreshBrainNeedsSetup();
        } catch (saveErr) {
          if (cancelled) return;
          setTest({
            status: "fail",
            message: `Save failed: ${
              saveErr instanceof Error ? saveErr.message : String(saveErr)
            }`,
            latencyMs: 0,
          });
        } finally {
          if (!cancelled) setSaving(false);
        }
      } catch (err) {
        if (cancelled) return;
        setTest({
          status: "fail",
          message: err instanceof Error ? err.message : String(err),
          latencyMs: 0,
        });
      }
    };

    void runAutoFlow();
    return () => {
      cancelled = true;
    };
    // We intentionally don't depend on `form.*` here — we want this to run
    // exactly once when pullState becomes "pulled". The form values are
    // read inside via closure on the latest render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pullState.kind]);

  if (!loaded) {
    return (
      <div className="cc-tab cc-tab-brain">
        <h2 className="cc-tab-title">Brain</h2>
        <p className="cc-tab-placeholder">Loading…</p>
      </div>
    );
  }

  const isOllama = form.uiProvider === "ollama";
  const canSave = test.status === "ok" && !saving;
  // During first-run setup mode we hide the advanced provider form and
  // show only the wizard. Once the user has a working brain saved, the
  // full provider/dropdown UI returns. Non-Ollama providers can still
  // configure freely after setup — the wizard is Ollama-only.
  const showAdvancedForm = !brainNeedsSetup;

  return (
    <div className="cc-tab cc-tab-brain">
      <h2 className="cc-tab-title">Brain</h2>

      {savedBanner && <div className="cc-success-banner">{savedBanner}</div>}

      {brainNeedsSetup && (
        <div className="cc-setup-banner" role="status">
          Welcome! Let's get Rolo Brain on your machine.
        </div>
      )}

      {showAdvancedForm && (
        <>
      <FormRow label="Provider" htmlFor="brain-provider">
        <select
          id="brain-provider"
          className="cc-form-select"
          value={form.uiProvider}
          onChange={(e) =>
            updateForm("uiProvider", e.target.value as UiProvider)
          }
        >
          {PROVIDER_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </FormRow>

      {form.uiProvider === "ollama" && (
        <>
          <FormRow label="Base URL" htmlFor="ollama-base-url">
            <input
              id="ollama-base-url"
              type="text"
              className="cc-form-input"
              value={form.ollamaBaseUrl}
              onChange={(e) => updateForm("ollamaBaseUrl", e.target.value)}
              placeholder="http://localhost:11434"
            />
          </FormRow>
          <FormRow label="Model" htmlFor="ollama-model">
            <input
              id="ollama-model"
              type="text"
              className="cc-form-input"
              value={form.ollamaModel}
              onChange={(e) => updateForm("ollamaModel", e.target.value)}
              placeholder={ROLO_BRAIN_MODEL}
            />
          </FormRow>
        </>
      )}

      {form.uiProvider === "openai_compat" && (
        <>
          <FormRow label="Base URL" htmlFor="openai-base-url">
            <input
              id="openai-base-url"
              type="text"
              className="cc-form-input"
              value={form.openaiBaseUrl}
              onChange={(e) => updateForm("openaiBaseUrl", e.target.value)}
              placeholder="https://api.openai.com/v1"
            />
          </FormRow>
          <FormRow label="API key" htmlFor="openai-api-key">
            <input
              id="openai-api-key"
              type="password"
              className="cc-form-input"
              value={form.openaiApiKey}
              onChange={(e) => updateForm("openaiApiKey", e.target.value)}
              autoComplete="off"
            />
          </FormRow>
          <FormRow label="Model" htmlFor="openai-model">
            <input
              id="openai-model"
              type="text"
              className="cc-form-input"
              value={form.openaiModel}
              onChange={(e) => updateForm("openaiModel", e.target.value)}
            />
          </FormRow>
        </>
      )}

      {form.uiProvider === "anthropic" && (
        <>
          <FormRow label="API key" htmlFor="anthropic-api-key">
            <input
              id="anthropic-api-key"
              type="password"
              className="cc-form-input"
              value={form.anthropicApiKey}
              onChange={(e) => updateForm("anthropicApiKey", e.target.value)}
              autoComplete="off"
            />
          </FormRow>
          <FormRow label="Model" htmlFor="anthropic-model">
            <input
              id="anthropic-model"
              type="text"
              className="cc-form-input"
              value={form.anthropicModel}
              onChange={(e) => updateForm("anthropicModel", e.target.value)}
              placeholder="claude-haiku-4-5-20251001"
            />
          </FormRow>
        </>
      )}

      {form.uiProvider === "gemini" && (
        <>
          <FormRow label="API key" htmlFor="gemini-api-key">
            <input
              id="gemini-api-key"
              type="password"
              className="cc-form-input"
              value={form.geminiApiKey}
              onChange={(e) => updateForm("geminiApiKey", e.target.value)}
              autoComplete="off"
            />
          </FormRow>
          <FormRow label="Model" htmlFor="gemini-model">
            <input
              id="gemini-model"
              type="text"
              className="cc-form-input"
              value={form.geminiModel}
              onChange={(e) => updateForm("geminiModel", e.target.value)}
              placeholder="gemini-2.0-flash"
            />
          </FormRow>
        </>
      )}

      {form.uiProvider === "huggingface" && (
        <>
          <FormRow label="API key" htmlFor="hf-api-key">
            <input
              id="hf-api-key"
              type="password"
              className="cc-form-input"
              value={form.hfApiKey}
              onChange={(e) => updateForm("hfApiKey", e.target.value)}
              autoComplete="off"
            />
          </FormRow>
          <FormRow label="Model" htmlFor="hf-model">
            <input
              id="hf-model"
              type="text"
              className="cc-form-input"
              value={form.hfModel}
              onChange={(e) => updateForm("hfModel", e.target.value)}
              placeholder="meta-llama/Llama-3.2-3B-Instruct"
            />
          </FormRow>
        </>
      )}

      {form.uiProvider === "deepinfra" && (
        <>
          <FormRow label="API key" htmlFor="deepinfra-api-key">
            <input
              id="deepinfra-api-key"
              type="password"
              className="cc-form-input"
              value={form.deepinfraApiKey}
              onChange={(e) =>
                updateForm("deepinfraApiKey", e.target.value)
              }
              autoComplete="off"
            />
          </FormRow>
          <FormRow label="Model" htmlFor="deepinfra-model">
            <input
              id="deepinfra-model"
              type="text"
              className="cc-form-input"
              value={form.deepinfraModel}
              onChange={(e) => updateForm("deepinfraModel", e.target.value)}
              placeholder="meta-llama/Meta-Llama-3.1-8B-Instruct"
            />
          </FormRow>
        </>
      )}

      <div className="cc-form-actions">
        <button
          type="button"
          className="cc-button cc-button-test"
          onClick={onTest}
          disabled={test.status === "running"}
        >
          {test.status === "running" ? "Testing…" : "Test Connection"}
        </button>
        <button
          type="button"
          className="cc-button cc-button-primary"
          onClick={onSave}
          disabled={!canSave}
          title={
            canSave
              ? "Persist and hot-swap Rolo's brain"
              : "Run Test Connection successfully before saving"
          }
        >
          {saving ? "Saving…" : "Save"}
        </button>
      </div>

      {test.status !== "idle" && (
        <div className="cc-test-status-row">
          {test.status === "running" && (
            <span className="cc-test-status-pending">Testing connection…</span>
          )}
          {test.status === "ok" && (
            <span className="cc-test-status-ok">
              ✓ Connected ({test.latencyMs} ms)
            </span>
          )}
          {test.status === "fail" && (
            <span className="cc-test-status-fail">✗ {test.message}</span>
          )}
        </div>
      )}
        </>
      )}

      {(isOllama || brainNeedsSetup) && (
        <section
          className={
            "cc-collapsible" +
            (brainNeedsSetup ? " cc-collapsible-promoted" : "")
          }
        >
          {!brainNeedsSetup && (
            <button
              type="button"
              className="cc-collapsible-toggle"
              onClick={() => setShowSetup((v) => !v)}
              aria-expanded={showSetup}
            >
              <span
                className={
                  "cc-collapsible-caret" +
                  (showSetup ? " cc-collapsible-caret-open" : "")
                }
              >
                ›
              </span>
              First time? Walk me through installing Rolo Brain
            </button>
          )}
          {(showSetup || brainNeedsSetup) && (
            <div className="cc-collapsible-body cc-setup-guide">
              <ol className="cc-setup-steps">
                <li>
                  <div className="cc-setup-step-title">Install Ollama.</div>
                  <button
                    type="button"
                    className="cc-button cc-button-secondary"
                    onClick={() => {
                      openUrl("https://ollama.com/download").catch((err) =>
                        console.error("[Rolo] open Ollama download:", err),
                      );
                    }}
                  >
                    Download Ollama
                  </button>
                  <div className="cc-setup-step-note">
                    Install it, then make sure the llama icon appears in your
                    menu bar.
                  </div>
                </li>
                <li>
                  <div className="cc-setup-step-title">
                    Pull Rolo Brain.
                    {pullState.kind === "ready_to_pull" &&
                      pullState.alreadyInstalled && (
                        <span className="cc-step-checkmark"> ✓ Installed</span>
                      )}
                    {pullState.kind === "pulled" && (
                      <span className="cc-step-checkmark"> ✓ Installed</span>
                    )}
                  </div>
                  <div className="cc-setup-step-note">
                    Downloads directly from Hugging Face via Ollama. One-time
                    download.
                  </div>

                  {pullState.kind === "checking" && (
                    <div className="cc-setup-step-note">
                      Checking if Rolo Brain is installed…
                    </div>
                  )}

                  {pullState.kind === "not_reachable" && (
                    <div className="cc-pull-waiting">
                      <span className="cc-test-status-pending">
                        Waiting for Ollama…
                      </span>
                      <div className="cc-setup-step-note">
                        Start Ollama from your menu bar. We'll detect it
                        automatically.
                      </div>
                    </div>
                  )}

                  {pullState.kind === "ready_to_pull" && (
                    <div className="cc-pull-actions">
                      <button
                        type="button"
                        className="cc-button cc-button-primary"
                        onClick={() => void onPullClick()}
                      >
                        {pullState.alreadyInstalled
                          ? "Re-pull to update"
                          : "Pull Rolo Brain"}
                      </button>
                    </div>
                  )}

                  {pullState.kind === "pulling" && (
                    <div className="cc-pull-progress">
                      <div
                        className="cc-pull-progress-bar"
                        role="progressbar"
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={Math.round(pullState.percent)}
                      >
                        <div
                          className="cc-pull-progress-bar-fill"
                          style={{ width: `${pullState.percent}%` }}
                        />
                      </div>
                      <div className="cc-pull-progress-meta">
                        <span>{Math.round(pullState.percent)}%</span>
                        {pullState.totalBytes > 0 && (
                          <span>
                            {formatMb(pullState.completedBytes)} /{" "}
                            {formatMb(pullState.totalBytes)} MB
                          </span>
                        )}
                      </div>
                      <div className="cc-setup-step-note">
                        {pullState.statusText}
                      </div>
                      <button
                        type="button"
                        className="cc-button cc-button-secondary"
                        onClick={() => void onCancelClick()}
                      >
                        Cancel
                      </button>
                    </div>
                  )}

                  {pullState.kind === "pull_failed" && (
                    <div className="cc-pull-error">
                      <div className="cc-test-status-fail">
                        ✗ {pullState.error}
                      </div>
                      <div className="cc-pull-actions">
                        <button
                          type="button"
                          className="cc-button cc-button-primary"
                          onClick={() => void onPullClick()}
                        >
                          Retry
                        </button>
                      </div>
                      <details className="cc-pull-manual">
                        <summary>Run manually in Terminal</summary>
                        <pre className="cc-setup-code">
                          <code>ollama pull {ROLO_BRAIN_MODEL}</code>
                        </pre>
                      </details>
                    </div>
                  )}

                  {pullState.kind === "pulled" && (
                    <div className="cc-setup-step-note">
                      Rolo Brain installed. Testing connection…
                    </div>
                  )}
                </li>
                <li>
                  <div className="cc-setup-step-title">
                    Test the connection.
                    {test.status === "ok" && (
                      <span className="cc-step-checkmark"> ✓ Connected</span>
                    )}
                  </div>
                  {brainNeedsSetup ? (
                    <div className="cc-setup-step-note">
                      {test.status === "ok"
                        ? "Rolo Brain is ready. Other tabs are now unlocked."
                        : "We'll test automatically once Rolo Brain is installed."}
                    </div>
                  ) : (
                    <div className="cc-setup-step-note">
                      Scroll up and click Test Connection. Once it goes green,
                      Rolo is ready.
                    </div>
                  )}
                </li>
              </ol>
            </div>
          )}
        </section>
      )}

      {showAdvancedForm && (
        <>
          <div className="cc-tool-notice">
            {isOllama ? (
              <span className="cc-tool-notice-neutral">
                Tool routing: enabled (Ollama only)
              </span>
            ) : (
              <span className="cc-tool-notice-amber">
                Tool routing: currently always runs on Ollama. Other providers
                fall back to a no-tool reply.
              </span>
            )}
          </div>
          <div className="cc-tool-notice">
            <span className="cc-tool-notice-neutral">
              Idle speech bubbles currently always use Ollama.
            </span>
          </div>
        </>
      )}
    </div>
  );
}
