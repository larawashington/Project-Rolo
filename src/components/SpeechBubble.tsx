/**
 * SpeechBubble — Rolo's voice rendered as a pixel-art speech bubble.
 *
 * This component runs in the "speech-bubble" Tauri window. It listens for
 * `rolo://show-speech`, `rolo://hide-speech`, and `rolo://speech-token`
 * events from the Rust backend. Tokens stream in with a typewriter effect
 * when Rolo's LLM brain is active. Falls back to instant display for
 * phrases.json fallback text.
 */

import { useEffect, useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import "./SpeechBubble.css";

/** Payload from the Rust `rolo://show-speech` event. */
interface ShowSpeechPayload {
  text: string;
  flipped: boolean;
}

/** Payload from the Rust `rolo://speech-token` event. */
interface SpeechTokenPayload {
  token: string;
}

/** Payload from the Rust `rolo://bubble-reposition` event. */
interface RepositionPayload {
  flipped: boolean;
}

export default function SpeechBubble() {
  const [visible, setVisible] = useState(false);
  const [text, setText] = useState("");
  const [flipped, setFlipped] = useState(false);

  // Use a ref for the stream buffer so the event listener closure
  // always sees the latest value without re-registering.
  const streamBufferRef = useRef("");

  // Track unlisteners for cleanup
  const unlistenShowRef = useRef<UnlistenFn | null>(null);
  const unlistenHideRef = useRef<UnlistenFn | null>(null);
  const unlistenReposRef = useRef<UnlistenFn | null>(null);
  const unlistenTokenRef = useRef<UnlistenFn | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function setup() {
      // Listen for speech-token events (LLM streaming)
      try {
        const unlistenToken = await listen<SpeechTokenPayload>(
          "rolo://speech-token",
          (event) => {
            if (cancelled) return;
            streamBufferRef.current += event.payload.token;
            setText(streamBufferRef.current);
            setVisible(true);
          },
        );
        if (cancelled) {
          unlistenToken();
        } else {
          unlistenTokenRef.current = unlistenToken;
        }
      } catch (err) {
        console.error(
          "[Rolo] Failed to listen for speech-token:",
          err,
        );
      }

      // Listen for show-speech events (final text or fallback)
      try {
        const unlistenShow = await listen<ShowSpeechPayload>(
          "rolo://show-speech",
          (event) => {
            if (!cancelled) {
              streamBufferRef.current = "";
              setText(event.payload.text);
              setFlipped(event.payload.flipped);
              setVisible(true);
            }
          },
        );
        if (cancelled) {
          unlistenShow();
        } else {
          unlistenShowRef.current = unlistenShow;
        }
      } catch (err) {
        console.error(
          "[Rolo] Failed to listen for show-speech — bubble will be silent:",
          err,
        );
      }

      // Listen for hide-speech events
      try {
        const unlistenHide = await listen<void>(
          "rolo://hide-speech",
          () => {
            if (!cancelled) {
              setVisible(false);
              streamBufferRef.current = "";
              setText("");
            }
          },
        );
        if (cancelled) {
          unlistenHide();
        } else {
          unlistenHideRef.current = unlistenHide;
        }
      } catch (err) {
        console.error(
          "[Rolo] Failed to listen for hide-speech:",
          err,
        );
      }

      // Listen for bubble-reposition events
      try {
        const unlistenRepos = await listen<RepositionPayload>(
          "rolo://bubble-reposition",
          (event) => {
            if (!cancelled) {
              setFlipped(event.payload.flipped);
            }
          },
        );
        if (cancelled) {
          unlistenRepos();
        } else {
          unlistenReposRef.current = unlistenRepos;
        }
      } catch (err) {
        console.error(
          "[Rolo] Failed to listen for bubble-reposition:",
          err,
        );
      }
    }

    setup();

    return () => {
      cancelled = true;
      if (unlistenShowRef.current) {
        unlistenShowRef.current();
        unlistenShowRef.current = null;
      }
      if (unlistenHideRef.current) {
        unlistenHideRef.current();
        unlistenHideRef.current = null;
      }
      if (unlistenReposRef.current) {
        unlistenReposRef.current();
        unlistenReposRef.current = null;
      }
      if (unlistenTokenRef.current) {
        unlistenTokenRef.current();
        unlistenTokenRef.current = null;
      }
    };
  }, []);

  /** Click the bubble to open chat with this text as the trigger. */
  async function handleDismiss() {
    setVisible(false);
    streamBufferRef.current = "";
    setText("");
    try {
      await invoke("open_chat", {
        triggerText: text,
        source: "speech",
        initialUserMessage: null,
      });
    } catch (err) {
      console.error("[Rolo] Failed to open chat:", err);
      try {
        await invoke("dismiss_speech");
      } catch (dismissErr) {
        console.error("[Rolo] Failed to dismiss speech:", dismissErr);
      }
    }
  }

  if (!visible) {
    return null;
  }

  const containerClass = [
    "speech-bubble-container",
    flipped ? "flipped" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={containerClass} onClick={handleDismiss}>
      <div className="speech-bubble">{text}</div>
      <div className="pixel-tail speech-tail" />
    </div>
  );
}
