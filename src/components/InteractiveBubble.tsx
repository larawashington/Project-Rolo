/**
 * InteractiveBubble — Rolo's interactive conversation renderer.
 *
 * Takes a generic InteractionPrompt and renders it as a pixel-art bubble
 * above Rolo's sprite. Handles all three element types (text, button rows,
 * text input) and all response paths (button click, text submit, dismiss).
 */

import { useState, useRef, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
  InteractionPrompt,
  InteractionResponse,
  InteractionElement,
  ResponseValue,
} from "../types/interaction";
import "./InteractiveBubble.css";

interface InteractiveBubbleProps {
  prompt: InteractionPrompt;
  onResponse: (response: InteractionResponse) => void;
  onDismiss: () => void;
}

export function InteractiveBubble({
  prompt,
  onResponse,
  onDismiss,
}: InteractiveBubbleProps) {
  const [textValue, setTextValue] = useState("");
  const [visible, setVisible] = useState(true);
  const submittedRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Find max_length from the text_input element if one exists
  const textInputElement = prompt.elements.find(
    (el) => el.type === "text_input",
  );
  const maxLength =
    textInputElement?.type === "text_input"
      ? textInputElement.max_length ?? 280
      : 280;

  /**
   * Build and submit a response to the Rust backend.
   * This is the single path through which all responses flow —
   * every button click, text submit, and dismiss goes through here.
   */
  const submitResponse = useCallback(
    async (responseValue: ResponseValue) => {
      if (submittedRef.current) return;
      submittedRef.current = true;
      setVisible(false);

      const response: InteractionResponse = {
        instance_id: prompt.instance_id,
        interaction_id: prompt.interaction_id,
        response: responseValue,
        timestamp: new Date().toISOString(),
      };

      try {
        await invoke<{ reaction: string | null }>(
          "submit_interaction_response",
          { response },
        );
      } catch (err) {
        console.error(
          "[Rolo] Failed to submit interaction response — his feelings may not be recorded:",
          err,
        );
      }

      if (responseValue.type === "dismissed") {
        onDismiss();
      } else {
        onResponse(response);
      }
    },
    [prompt.instance_id, prompt.interaction_id, onResponse, onDismiss],
  );

  // --- Timeout auto-dismiss ---
  useEffect(() => {
    if (prompt.timeout_ms != null && prompt.timeout_ms > 0) {
      timerRef.current = setTimeout(() => {
        submitResponse({ type: "dismissed" });
      }, prompt.timeout_ms);
    }

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [prompt.timeout_ms, submitResponse]);

  // --- Click-outside dismiss ---
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        submitResponse({ type: "dismissed" });
      }
    }

    // Delay listener attachment by a frame to avoid the click that
    // might have triggered the bubble from immediately dismissing it
    const frameId = requestAnimationFrame(() => {
      document.addEventListener("mousedown", handleClickOutside);
    });

    return () => {
      cancelAnimationFrame(frameId);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [submitResponse]);

  // --- Button click handler ---
  function handleButtonClick(value: string) {
    if (timerRef.current) clearTimeout(timerRef.current);
    submitResponse({ type: "button_press", value });
  }

  // --- Dismiss X button ---
  function handleDismissClick(e: React.MouseEvent) {
    e.stopPropagation();
    if (timerRef.current) clearTimeout(timerRef.current);
    submitResponse({ type: "dismissed" });
  }

  // --- Text input handlers ---
  function handleTextChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    const newValue = e.target.value;
    if (newValue.length <= maxLength) {
      setTextValue(newValue);
    }
  }

  function handleTextKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const trimmed = textValue.trim();
      if (trimmed.length === 0) return; // Empty submit ignored
      if (timerRef.current) clearTimeout(timerRef.current);
      // First record the check-in via the existing pathway (MoodEvent +
      // vault Checkin event). Then open chat with the user's text as the
      // first user message — Rolo replies via the normal streaming pipeline.
      submitResponse({ type: "text_submit", text: trimmed });
      invoke("open_chat", {
        triggerText: "",
        source: "checkin_text",
        initialUserMessage: trimmed,
      }).catch((err) => {
        console.error(
          "[Rolo] Failed to open chat from check-in text submit:",
          err,
        );
      });
    }
  }

  // --- Auto-resize textarea ---
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      // Reset height to measure scrollHeight accurately
      textarea.style.height = "auto";
      // Clamp to max 2 lines (approx 2 * lineHeight).
      // With font-size 8px and line-height 1.5, one line is ~12px.
      // Two lines plus padding = ~32px. We cap at 32px.
      const maxHeight = 32;
      textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
    }
  }, [textValue]);

  // --- Render each element ---
  function renderElement(element: InteractionElement, index: number) {
    switch (element.type) {
      case "text":
        return (
          <div key={index} className="ib-text">
            {element.content}
          </div>
        );

      case "button_row":
        return (
          <div key={index} className="ib-button-column">
            {element.buttons.map((btn, btnIndex) => (
              <button
                key={btnIndex}
                className={`ib-btn ${btn.style === "secondary" ? "ib-btn-secondary" : "ib-btn-primary"}`}
                onClick={() => handleButtonClick(btn.value)}
                disabled={!visible}
              >
                {btn.label}
              </button>
            ))}
          </div>
        );

      case "text_input":
        return (
          <div key={index} className="ib-text-input-wrapper">
            <textarea
              ref={textareaRef}
              className="ib-text-input"
              placeholder={element.placeholder ?? ""}
              value={textValue}
              onChange={handleTextChange}
              onKeyDown={handleTextKeyDown}
              maxLength={maxLength}
              rows={1}
              disabled={!visible}
            />
            <div className="ib-char-count">
              {maxLength - textValue.length}
            </div>
          </div>
        );

      default:
        return null;
    }
  }

  if (!visible) {
    return null;
  }

  return (
    <div className="ib-container" ref={containerRef}>
      <div className="ib-bubble">
        <button
          className="ib-dismiss"
          onClick={handleDismissClick}
          aria-label="Dismiss"
        >
          x
        </button>

        {prompt.elements.map((element, index) => renderElement(element, index))}
      </div>
      <div className="pixel-tail bubble-tail" />
    </div>
  );
}
