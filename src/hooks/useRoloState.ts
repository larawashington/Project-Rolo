import { useEffect, useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import type { PetState, Position, StatePayload } from "../types/rolo";

/**
 * Subscribes to Rolo's state changes from the Rust backend.
 *
 * On mount, fetches the initial state via `get_pet_state` command.
 * Then listens for `rolo://state-changed` events emitted by the tick loop.
 *
 * Returns the current state and position so the UI can render accordingly.
 * This is the primary bridge between Rolo's Rust soul and his React body.
 */
export function useRoloState() {
  const [state, setState] = useState<PetState>("idle");
  const [position, setPosition] = useState<Position>({ x: 0, y: 0 });
  const [ready, setReady] = useState(false);

  // Track whether we've already cleaned up to avoid double-unlisten
  const unlistenRef = useRef<UnlistenFn | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      // 1. Fetch initial state from Rust
      try {
        const payload = await invoke<StatePayload>("get_pet_state");
        if (!cancelled) {
          setState(payload.state);
          setPosition(payload.position);
          setReady(true);
        }
      } catch (err) {
        console.error(
          "[Rolo] Failed to fetch initial state — he may appear lifeless:",
          err,
        );
        // Still mark as ready so the UI renders something
        if (!cancelled) {
          setReady(true);
        }
      }

      // 2. Subscribe to state change events from the tick loop
      try {
        const unlisten = await listen<StatePayload>(
          "rolo://state-changed",
          (event) => {
            if (!cancelled) {
              setState(event.payload.state);
              setPosition(event.payload.position);
            }
          },
        );
        if (cancelled) {
          // Component unmounted while we were setting up
          unlisten();
        } else {
          unlistenRef.current = unlisten;
        }
      } catch (err) {
        console.error(
          "[Rolo] Failed to subscribe to state events — he'll be frozen:",
          err,
        );
      }
    }

    init();

    return () => {
      cancelled = true;
      if (unlistenRef.current) {
        unlistenRef.current();
        unlistenRef.current = null;
      }
    };
  }, []);

  return { state, position, ready };
}
