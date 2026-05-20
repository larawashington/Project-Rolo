import { useEffect, useState, useRef } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import type { InteractionPrompt } from "../types/interaction";

/**
 * Subscribes to interaction events from the Rust backend.
 *
 * When the interaction engine decides it's time for Rolo to ask the user
 * something (e.g., a mood check-in), it emits `rolo://show-interaction`
 * with an InteractionPrompt payload. When the interaction ends (response
 * submitted, dismissed, or timed out on the Rust side), it emits
 * `rolo://hide-interaction`.
 *
 * This hook follows the exact same pattern as useRoloState and useFileDrop:
 * async listener setup, cancellation guard, ref-based cleanup.
 *
 * Returns the currently active prompt, or null if Rolo isn't asking anything.
 */
export function useInteraction() {
  const [activePrompt, setActivePrompt] = useState<InteractionPrompt | null>(
    null,
  );

  // Track unlisteners for cleanup — combined into a single ref
  const unlistenRef = useRef<UnlistenFn | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function setup() {
      let unlistenShow: UnlistenFn | undefined;
      let unlistenHide: UnlistenFn | undefined;

      // Listen for show-interaction events (new prompt from the engine)
      try {
        unlistenShow = await listen<InteractionPrompt>(
          "rolo://show-interaction",
          (event) => {
            if (!cancelled) {
              setActivePrompt(event.payload);
            }
          },
        );
      } catch (err) {
        console.error(
          "[Rolo] Failed to subscribe to show-interaction — he won't be able to ask you anything:",
          err,
        );
      }

      // Listen for hide-interaction events (backend says interaction is over)
      try {
        unlistenHide = await listen<void>(
          "rolo://hide-interaction",
          () => {
            if (!cancelled) {
              setActivePrompt(null);
            }
          },
        );
      } catch (err) {
        console.error(
          "[Rolo] Failed to subscribe to hide-interaction — bubbles may get stuck:",
          err,
        );
      }

      if (cancelled) {
        // Component unmounted while we were setting up listeners
        unlistenShow?.();
        unlistenHide?.();
      } else {
        // Store a combined cleanup function
        unlistenRef.current = () => {
          unlistenShow?.();
          unlistenHide?.();
        };
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

  return { activePrompt };
}
