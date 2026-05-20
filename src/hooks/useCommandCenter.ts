/**
 * useCommandCenter — state owner for the Command Center window.
 *
 * Loads settings on mount via `cc_load_settings`, exposes a `setSettings`
 * setter for tab components, and persists via `cc_save_settings`. Also
 * listens for `rolo://command-center-opened` so an external open (e.g. tray
 * menu) refreshes from disk in case another process edited the file.
 *
 * Per-tab dirty tracking: `markDirty(tab)` adds the tab to a set,
 * `markClean(tab)` removes it, `dirty` is `size > 0`. `discardAll()`
 * empties the set and reloads from disk. `brainNeedsSetup` is derived
 * from `cc_brain_needs_setup` and flips back to false once Rolo has a
 * working brain.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTauriEvent } from "./useTauriEvent";
import type {
  CommandCenterSettings,
  CommandCenterTab,
} from "../types/commandCenter";

/**
 * Structural deep-equal for the Weather tab's snapshot-diff dirty
 * computation. Handles plain objects, arrays, Dates, Maps, and Sets so
 * that adding a non-JSON-serializable field to `WeatherSettings` later
 * doesn't silently degrade into a key-order-sensitive string compare.
 */
function deepEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) {
    return false;
  }
  if (a instanceof Date || b instanceof Date) {
    return (
      a instanceof Date &&
      b instanceof Date &&
      a.getTime() === b.getTime()
    );
  }
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      return false;
    }
    for (let i = 0; i < a.length; i++) {
      if (!deepEqual(a[i], b[i])) return false;
    }
    return true;
  }
  if (a instanceof Map || b instanceof Map) {
    if (!(a instanceof Map) || !(b instanceof Map) || a.size !== b.size) {
      return false;
    }
    for (const [k, v] of a) {
      if (!b.has(k) || !deepEqual(v, b.get(k))) return false;
    }
    return true;
  }
  if (a instanceof Set || b instanceof Set) {
    if (!(a instanceof Set) || !(b instanceof Set) || a.size !== b.size) {
      return false;
    }
    for (const v of a) {
      if (!b.has(v)) return false;
    }
    return true;
  }
  const aObj = a as Record<string, unknown>;
  const bObj = b as Record<string, unknown>;
  const aKeys = Object.keys(aObj);
  const bKeys = Object.keys(bObj);
  if (aKeys.length !== bKeys.length) return false;
  for (const k of aKeys) {
    if (!Object.prototype.hasOwnProperty.call(bObj, k)) return false;
    if (!deepEqual(aObj[k], bObj[k])) return false;
  }
  return true;
}

export interface UseCommandCenterResult {
  settings: CommandCenterSettings | null;
  setSettings: React.Dispatch<
    React.SetStateAction<CommandCenterSettings | null>
  >;
  /**
   * Persist settings to disk via `cc_save_settings` and update the clean
   * snapshot so the snapshot-diff dirty flag flips back to false. The
   * optional `explicit` argument lets callers (e.g., WeatherTab) save a
   * freshly-built shape without waiting a tick for React's async
   * setSettings to land.
   */
  saveSettings: (explicit?: CommandCenterSettings) => Promise<void>;
  loading: boolean;
  saveError: string | null;
  dirty: boolean;
  markDirty: (tab: CommandCenterTab) => void;
  markClean: (tab: CommandCenterTab) => void;
  /** Drop all dirty marks AND reload settings from disk so unsaved edits revert. */
  discardAll: () => Promise<void>;
  activeTab: CommandCenterTab;
  setActiveTab: (tab: CommandCenterTab) => void;
  brainNeedsSetup: boolean;
  /** Re-query `cc_brain_needs_setup`. Call after a successful Brain save. */
  refreshBrainNeedsSetup: () => Promise<void>;
}

export function useCommandCenter(): UseCommandCenterResult {
  const [settings, setSettings] = useState<CommandCenterSettings | null>(null);
  /**
   * Snapshot of the last-known-clean settings. The Weather tab now drives
   * edits straight through `setSettings`, so its dirty status is derived
   * from a deep-equal comparison between `settings.weather` and
   * `lastSavedSnapshot.weather` instead of being toggled by hand. Brain
   * and Memory still own local form mirrors and use `markDirty` /
   * `markClean` against `dirtyTabs`.
   */
  const [lastSavedSnapshot, setLastSavedSnapshot] =
    useState<CommandCenterSettings | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<CommandCenterTab>("brain");

  // Per-tab dirty tracking for the tabs that still own local form mirrors
  // (Brain, Memory). Weather is computed below via snapshot-diff, then
  // merged into the public `dirty` flag.
  const [dirtyTabs, setDirtyTabs] = useState<Set<CommandCenterTab>>(
    () => new Set(),
  );
  const weatherDirty = useMemo(
    () =>
      settings !== null &&
      lastSavedSnapshot !== null &&
      !deepEqual(settings.weather, lastSavedSnapshot.weather),
    [settings, lastSavedSnapshot],
  );
  const dirty = dirtyTabs.size > 0 || weatherDirty;

  // Brain configuration gate. Initialized true so the first paint
  // doesn't briefly show non-Brain tabs as interactive; the mount effect
  // resolves the actual value within a tick.
  const [brainNeedsSetup, setBrainNeedsSetup] = useState<boolean>(true);

  // Used by the load effect and the external-open listener so both paths
  // share one fetch implementation.
  const loadFromDisk = useCallback(async () => {
    try {
      const loaded = await invoke<CommandCenterSettings>("cc_load_settings");
      setSettings(loaded);
      // Reset the saved-snapshot too — what's on disk is, by definition, clean.
      setLastSavedSnapshot(loaded);
    } catch (err) {
      // Rust's loader falls back to defaults on any read/parse failure, so
      // an invoke-level error here is unusual (IPC issue). Surface it via
      // saveError so the user at least sees something rather than a
      // permanently-spinning window.
      const msg = err instanceof Error ? err.message : String(err);
      setSaveError(`Failed to load settings: ${msg}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshBrainNeedsSetup = useCallback(async () => {
    try {
      const needs = await invoke<boolean>("cc_brain_needs_setup");
      setBrainNeedsSetup(needs);
    } catch (err) {
      // If the probe itself fails (IPC), err on the side of letting the
      // user out — a stuck-modal CC is worse than a CC that didn't realize
      // it needed setup. Log + treat as configured.
      // eslint-disable-next-line no-console
      console.warn("cc_brain_needs_setup failed:", err);
      setBrainNeedsSetup(false);
    }
  }, []);

  // Initial load.
  useEffect(() => {
    void loadFromDisk();
    void refreshBrainNeedsSetup();
  }, [loadFromDisk, refreshBrainNeedsSetup]);

  // If we're in setup mode, force the active tab to Brain. The user can't
  // navigate to other tabs anyway, but this also handles the case where the
  // user opened CC, navigated to Memory, then deleted their config off-disk
  // and re-opened CC — first paint after `refreshBrainNeedsSetup` resolves
  // would otherwise sit on a disabled tab.
  useEffect(() => {
    if (brainNeedsSetup && activeTab !== "brain") {
      setActiveTab("brain");
    }
  }, [brainNeedsSetup, activeTab]);

  // Refresh when an external open fires (tray menu, another window, etc.).
  useTauriEvent("rolo://command-center-opened", () => {
    void loadFromDisk();
    void refreshBrainNeedsSetup();
  });

  const saveSettings = useCallback(
    async (explicit?: CommandCenterSettings) => {
      const toSave = explicit ?? settings;
      if (!toSave) return;
      setSaveError(null);
      try {
        await invoke("cc_save_settings", { settings: toSave });
        // A successful save makes `toSave` the new "clean" snapshot — and
        // also the live settings, in case the caller bypassed React's
        // async setSettings cycle (WeatherTab does this so the snapshot
        // and the live state both flip on the same tick).
        setSettings(toSave);
        setLastSavedSnapshot(toSave);
        setDirtyTabs((prev) => {
          if (prev.size === 0) return prev;
          const next = new Set(prev);
          next.delete("weather");
          return next;
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setSaveError(msg);
        throw err;
      }
    },
    [settings],
  );

  const markDirty = useCallback((tab: CommandCenterTab) => {
    setDirtyTabs((prev) => {
      if (prev.has(tab)) return prev;
      const next = new Set(prev);
      next.add(tab);
      return next;
    });
  }, []);

  const markClean = useCallback((tab: CommandCenterTab) => {
    setDirtyTabs((prev) => {
      if (!prev.has(tab)) return prev;
      const next = new Set(prev);
      next.delete(tab);
      return next;
    });
  }, []);

  const discardAll = useCallback(async () => {
    setDirtyTabs((prev) => (prev.size === 0 ? prev : new Set()));
    // Reload the shared settings (Weather tab's source of truth) from
    // disk, and emit `rolo://command-center-discard` so Brain and Memory
    // — which own their own local form state — force-reload their
    // canonical values too. They use a dedicated event (not
    // `command-center-opened`) so the external-open path keeps its
    // existing "don't clobber unsaved edits" guard.
    await loadFromDisk();
    try {
      const { emit } = await import("@tauri-apps/api/event");
      await emit("rolo://command-center-discard");
    } catch (err) {
      // Failing to emit just means Brain/Memory keep their in-memory
      // form state — the dirty marks are gone, so the next save will
      // still ask the user to confirm. Not fatal.
      // eslint-disable-next-line no-console
      console.warn("discardAll: emit failed:", err);
    }
  }, [loadFromDisk]);

  return {
    settings,
    setSettings,
    saveSettings,
    loading,
    saveError,
    dirty,
    markDirty,
    markClean,
    discardAll,
    activeTab,
    setActiveTab,
    brainNeedsSetup,
    refreshBrainNeedsSetup,
  };
}
