import { useEffect, useState, useRef } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

/**
 * Subscribes to file drop events from the Rust backend.
 *
 * When files are dropped on Rolo and he starts sniffing, the backend emits
 * a "rolo://files-pending" event with the file paths. This hook captures
 * those paths so the ConfirmationBubble can display them.
 *
 * The pending files are cleared when the state leaves "sniffing" (the
 * backend clears them on confirm/decline, and a new event arrives for
 * queued batches).
 */
export function useFileDrop() {
  const [pendingFiles, setPendingFiles] = useState<string[] | null>(null);
  const unlistenRef = useRef<UnlistenFn | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function setup() {
      try {
        const unlisten = await listen<string[]>(
          "rolo://files-pending",
          (event) => {
            if (!cancelled) {
              setPendingFiles(event.payload);
            }
          },
        );

        if (cancelled) {
          unlisten();
        } else {
          unlistenRef.current = unlisten;
        }
      } catch (err) {
        console.error(
          "[Rolo] Failed to subscribe to file drop events:",
          err,
        );
      }

      // Also listen for state changes to clear pending files when leaving sniffing
      try {
        const unlistenState = await listen<{ state: string }>(
          "rolo://state-changed",
          (event) => {
            if (!cancelled && event.payload.state !== "sniffing") {
              setPendingFiles(null);
            }
          },
        );

        if (cancelled) {
          unlistenState();
        } else {
          const prevUnlisten = unlistenRef.current;
          unlistenRef.current = () => {
            prevUnlisten?.();
            unlistenState();
          };
        }
      } catch (err) {
        console.error(
          "[Rolo] Failed to subscribe to state events for file drop:",
          err,
        );
      }
    }

    setup();

    return () => {
      cancelled = true;
      if (unlistenRef.current) {
        unlistenRef.current();
        unlistenRef.current = null;
      }
    };
  }, []);

  return { pendingFiles };
}
