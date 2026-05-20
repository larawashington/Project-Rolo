/**
 * CommandCenter — root view for the `command-center` Tauri window.
 *
 * Renders a custom pixel-art titlebar (drag region + min/close), a left
 * sidebar with four tab buttons (Brain, Memory, Perception, Weather), and
 * a content pane showing the active tab. Owns the unsaved-changes guard
 * (close modal) and the modal "set up Brain first" gate that fresh
 * installs hit before any other tab becomes interactive.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTauriEvent } from "../../hooks/useTauriEvent";
import { getCurrentWindow } from "@tauri-apps/api/window";

import { useCommandCenter } from "../../hooks/useCommandCenter";
import type {
  CommandCenterSettings,
  CommandCenterTab,
} from "../../types/commandCenter";
import BrainTab from "./BrainTab";
import MemoryTab from "./MemoryTab";
import PerceptionTab from "./PerceptionTab";
import WeatherTab from "./WeatherTab";
import "./index.css";

export interface CommandCenterTabProps {
  settings: CommandCenterSettings;
  setSettings: React.Dispatch<
    React.SetStateAction<CommandCenterSettings | null>
  >;
  /**
   * Persist the current `settings` via `cc_save_settings` and update the
   * hook's clean-snapshot. WeatherTab uses this so its snapshot-diff
   * dirty flag flips back to false after a successful save without an
   * extra `markClean` call. Accepts an optional `explicit` shape so
   * callers can save a freshly-built settings object without waiting a
   * tick for React's async setSettings to land.
   */
  saveSettings: (explicit?: CommandCenterSettings) => Promise<void>;
  /** Mark the calling tab dirty so the close-handler raises a discard prompt. */
  markDirty: (tab: CommandCenterTab) => void;
  /** Mark the calling tab clean — typically after a successful save. */
  markClean: (tab: CommandCenterTab) => void;
  /**
   * Re-query `cc_brain_needs_setup`. Brain tab calls this after a successful
   * save so the modal gate releases. Other tabs ignore it.
   */
  refreshBrainNeedsSetup: () => Promise<void>;
  /**
   * Latest value of `cc_brain_needs_setup`. Brain tab uses this to enter
   * first-run wizard mode (auto-expand the walkthrough, hide the advanced
   * provider form, auto-save once the pull-and-test succeeds).
   */
  brainNeedsSetup: boolean;
}

interface TabSpec {
  id: CommandCenterTab;
  label: string;
}

// Order is load-bearing: PRD acceptance criterion #2 requires this exact
// sequence. Update with care.
const TABS: ReadonlyArray<TabSpec> = [
  { id: "brain", label: "Brain" },
  { id: "memory", label: "Memory" },
  { id: "perception", label: "Perception" },
  { id: "weather", label: "Weather" },
];

export default function CommandCenter() {
  const {
    settings,
    setSettings,
    saveSettings,
    loading,
    activeTab,
    setActiveTab,
    dirty,
    markDirty,
    markClean,
    discardAll,
    brainNeedsSetup,
    refreshBrainNeedsSetup,
  } = useCommandCenter();

  // Local state for the "Discard changes?" modal — driven by the close
  // listener below. Setup-mode close attempts surface a different message
  // (modalMessage) since the modal doubles as the "must configure" notice.
  const [showDiscardModal, setShowDiscardModal] = useState(false);
  const [setupCloseNotice, setSetupCloseNotice] = useState<string | null>(null);

  // We need the freshest `dirty` and `brainNeedsSetup` inside the close-
  // listener callback, but Tauri's `listen` snapshots the closure at attach
  // time. A ref keeps the callback's view of state current without
  // re-attaching the listener (which is expensive and racy).
  const dirtyRef = useRef(dirty);
  const setupRef = useRef(brainNeedsSetup);
  useEffect(() => {
    dirtyRef.current = dirty;
  }, [dirty]);
  useEffect(() => {
    setupRef.current = brainNeedsSetup;
  }, [brainNeedsSetup]);

  // Single source of truth for close-attempts. Both the Rust-side close
  // listener AND the custom-titlebar close button call this — so the
  // discard-modal + setup-gate logic only lives in one place. The function
  // reads refs (not state) so it stays stable across renders.
  const requestClose = useCallback(() => {
    // Setup mode wins: the window must NOT close until Brain is configured.
    // The backend's cc_force_close will also reject, but the frontend
    // shows the friendly explanation rather than waiting for the err.
    if (setupRef.current) {
      setSetupCloseNotice("Pick a brain for Rolo before continuing.");
      return;
    }
    if (dirtyRef.current) {
      setShowDiscardModal(true);
    } else {
      // No unsaved edits — just hide the window via the helper command.
      // The cc_force_close path lets the backend reject close in setup
      // mode (defense in depth — the frontend gate above is the primary
      // check, the backend gate prevents bypass via direct invoke).
      void invoke("cc_force_close").catch((err) => {
        // eslint-disable-next-line no-console
        console.warn("cc_force_close failed:", err);
        // If the backend rejected (we just transitioned into setup
        // mode), surface that. Otherwise, the close attempt silently
        // dropped — the window stays open, which is the safer fallback.
        setSetupCloseNotice("Pick a brain for Rolo before continuing.");
      });
    }
  }, []);

  // Listen for the Rust-side close interception (native OS close gestures
  // we can't catch in JS — Cmd-Q, dock-quit, etc.). With the custom
  // titlebar the X button takes the direct path via requestClose() so this
  // listener mostly handles app-shutdown and edge cases.
  useTauriEvent("rolo://command-center-close-requested", () => {
    requestClose();
  });

  const handleMinimize = useCallback(async () => {
    try {
      await getCurrentWindow().minimize();
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("[Rolo] Failed to minimize Command Center:", err);
    }
  }, []);

  const handleDiscard = useCallback(async () => {
    setShowDiscardModal(false);
    await discardAll();
    try {
      await invoke("cc_force_close");
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("cc_force_close after discard failed:", err);
    }
  }, [discardAll]);

  const handleGoBack = useCallback(() => {
    setShowDiscardModal(false);
  }, []);

  if (loading || !settings) {
    return (
      <div className="cc-frame">
        <div className="cc-header" data-tauri-drag-region>
          <span className="cc-header-title" data-tauri-drag-region>
            Rolo Command Center
          </span>
          <div className="cc-header-controls">
            <button
              className="cc-header-min"
              onClick={() => void handleMinimize()}
              aria-label="Minimize"
            >
              _
            </button>
            <button
              className="cc-header-close"
              onClick={requestClose}
              aria-label="Close"
            >
              X
            </button>
          </div>
        </div>
        <div className="cc-body cc-loading">
          <span className="cc-loading-label">Loading…</span>
        </div>
      </div>
    );
  }

  const tabProps: CommandCenterTabProps = {
    settings,
    setSettings,
    saveSettings,
    markDirty,
    markClean,
    refreshBrainNeedsSetup,
    brainNeedsSetup,
  };

  // In setup mode, non-Brain tabs are disabled. The disabled buttons stay
  // in the DOM so the layout doesn't reflow when the mode flips — they
  // just refuse clicks and dim visually.
  const inSetupMode = brainNeedsSetup;

  return (
    <div className="cc-frame">
      <div className="cc-header" data-tauri-drag-region>
        <span className="cc-header-title" data-tauri-drag-region>
          Rolo Command Center
        </span>
        <div className="cc-header-controls">
          <button
            className="cc-header-min"
            onClick={() => void handleMinimize()}
            aria-label="Minimize"
          >
            _
          </button>
          <button
            className="cc-header-close"
            onClick={requestClose}
            aria-label="Close"
          >
            X
          </button>
        </div>
      </div>
      <div className="cc-body">
      <nav className="cc-sidebar" aria-label="Command Center sections">
        <ul className="cc-tab-list">
          {TABS.map((tab) => {
            const selected = tab.id === activeTab;
            const tabDisabled = inSetupMode && tab.id !== "brain";
            return (
              <li key={tab.id} className="cc-tab-list-item">
                <button
                  type="button"
                  className={
                    "cc-tab-button" +
                    (selected ? " cc-tab-button-active" : "") +
                    (tabDisabled ? " cc-tab-button-disabled" : "")
                  }
                  aria-current={selected ? "page" : undefined}
                  aria-disabled={tabDisabled || undefined}
                  disabled={tabDisabled}
                  onClick={() => {
                    if (tabDisabled) return;
                    setActiveTab(tab.id);
                  }}
                >
                  {tab.label}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
      <main className="cc-content">
        {inSetupMode && (
          <div className="cc-setup-banner" role="status">
            Welcome! Pick a brain for Rolo before continuing — other tabs
            unlock once a Brain is saved.
          </div>
        )}
        {activeTab === "brain" && <BrainTab {...tabProps} />}
        {activeTab === "memory" && <MemoryTab {...tabProps} />}
        {activeTab === "perception" && <PerceptionTab {...tabProps} />}
        {activeTab === "weather" && <WeatherTab {...tabProps} />}
      </main>
      </div>

      {showDiscardModal && (
        <div className="cc-modal-backdrop" role="dialog" aria-modal="true">
          <div className="cc-modal-card">
            <h3 className="cc-modal-title">Discard changes?</h3>
            <p className="cc-modal-body">
              You have unsaved edits in Rolo Command Center. Closing now
              will lose them.
            </p>
            <div className="cc-modal-actions">
              <button
                type="button"
                className="cc-button cc-button-secondary"
                onClick={handleGoBack}
                autoFocus
              >
                Go back
              </button>
              <button
                type="button"
                className="cc-button cc-button-danger"
                onClick={() => void handleDiscard()}
              >
                Discard
              </button>
            </div>
          </div>
        </div>
      )}

      {setupCloseNotice && (
        <div className="cc-modal-backdrop" role="dialog" aria-modal="true">
          <div className="cc-modal-card">
            <h3 className="cc-modal-title">One thing first</h3>
            <p className="cc-modal-body">{setupCloseNotice}</p>
            <div className="cc-modal-actions">
              <button
                type="button"
                className="cc-button cc-button-primary"
                onClick={() => setSetupCloseNotice(null)}
                autoFocus
              >
                OK
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
