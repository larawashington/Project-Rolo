import { useEffect, useRef } from "react";
import { listen, type UnlistenFn, type Event } from "@tauri-apps/api/event";

/**
 * Subscribe to a Tauri event for the component's lifetime.
 *
 * Handles the cancel-if-unmounted race: if the component unmounts before
 * `listen()`'s promise resolves, the returned unlisten function is invoked
 * immediately so we never end up with a registered listener pointing at a
 * dead component.
 *
 * The handler is read through a ref so call sites can pass inline arrows
 * without re-registering the listener on every render. Re-registration
 * only happens when `event` changes.
 */
export function useTauriEvent<T = unknown>(
  event: string,
  handler: (payload: T, e: Event<T>) => void,
): void {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    let cancelled = false;
    let unlisten: UnlistenFn | null = null;

    void listen<T>(event, (e) => {
      handlerRef.current(e.payload, e);
    }).then((un) => {
      if (cancelled) un();
      else unlisten = un;
    });

    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, [event]);
}
