/**
 * SleepOverlay — modal overlay rendered over the Memory panel while Rolo is
 * absorbing a fresh User Profile update.
 *
 * Subscribes to `rolo://command-center-dream-progress` on mount and updates
 * the stage label as each event arrives. If no event lands within 30 seconds
 * the label switches to "Still working…" — but we never auto-dismiss; the
 * parent owns the visible/hidden state and only un-mounts us when the backing
 * `cc_save_memory_and_sleep` call resolves (or when Cancel fires).
 *
 * The overlay is scoped to the Memory panel rather than the whole window: the
 * surrounding sidebar and tab chrome stay visible so the user can tell at a
 * glance which surface is busy.
 */

import { invoke } from "@tauri-apps/api/core";
import { useEffect, useRef, useState } from "react";
import { useTauriEvent } from "../../hooks/useTauriEvent";
import type { DreamProgressEvent } from "../../types/commandCenter";

interface SleepOverlayProps {
  /** Initial stage label shown before any event arrives. */
  initialLabel?: string;
  /** Called when the user clicks Cancel. */
  onCancel: () => void;
}

/** ms before we flip the label to "Still working…" if no event has arrived. */
const STILL_WORKING_TIMEOUT_MS = 30_000;

export default function SleepOverlay({
  initialLabel = "Writing memory…",
  onCancel,
}: SleepOverlayProps) {
  const [label, setLabel] = useState<string>(initialLabel);
  const [cancelling, setCancelling] = useState<boolean>(false);
  const stillWorkingTimerRef = useRef<number | null>(null);

  const armStillWorking = () => {
    if (stillWorkingTimerRef.current !== null) {
      window.clearTimeout(stillWorkingTimerRef.current);
    }
    stillWorkingTimerRef.current = window.setTimeout(() => {
      setLabel("Still working…");
    }, STILL_WORKING_TIMEOUT_MS);
  };

  // Arm the timer once at mount; clear it on unmount.
  useEffect(() => {
    armStillWorking();
    return () => {
      if (stillWorkingTimerRef.current !== null) {
        window.clearTimeout(stillWorkingTimerRef.current);
        stillWorkingTimerRef.current = null;
      }
    };
  }, []);

  // Listen for stage events. We accept any payload that has a string `label`
  // so a future schema bump doesn't break the visible text.
  useTauriEvent<DreamProgressEvent>(
    "rolo://command-center-dream-progress",
    (payload) => {
      if (payload && typeof payload.label === "string") {
        setLabel(payload.label);
        armStillWorking();
      }
    },
  );

  const handleCancel = async () => {
    if (cancelling) return;
    setCancelling(true);
    try {
      await invoke("cc_cancel_sleep");
    } catch (err) {
      // The dream may have already finished between click and invoke. We
      // surface nothing — the parent's save promise will resolve normally
      // and tear us down.
      console.warn("[Rolo] cc_cancel_sleep failed:", err);
    } finally {
      onCancel();
    }
  };

  return (
    <div
      className="cc-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Rolo is sleeping"
    >
      <div className="cc-overlay-card">
        <h3 className="cc-overlay-title">
          Rolo is sleeping while he absorbs this…
        </h3>
        <p className="cc-overlay-stage">{label}</p>
        <div className="cc-overlay-actions">
          <button
            type="button"
            className="cc-button cc-button-secondary"
            onClick={() => void handleCancel()}
            disabled={cancelling}
          >
            {cancelling ? "Cancelling…" : "Cancel"}
          </button>
        </div>
      </div>
    </div>
  );
}
