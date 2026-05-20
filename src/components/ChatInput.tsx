import { useState, useRef, useEffect } from "react";

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled?: boolean;
  maxLength?: number;
}

export function ChatInput({
  onSend,
  disabled = false,
  maxLength = 2000,
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = `${Math.min(ta.scrollHeight, 80)}px`;
    }
  }, [value]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleSend() {
    const trimmed = value.trim();
    if (trimmed.length === 0 || disabled) return;
    onSend(trimmed);
    setValue("");
  }

  const remaining = maxLength - value.length;
  const showCount = value.length > 1500;
  const countClass =
    remaining < 100 ? "danger" : remaining < 500 ? "warning" : "";

  return (
    <div className="chat-input-area">
      <div className="chat-input-wrapper">
        <textarea
          ref={textareaRef}
          className="chat-input"
          placeholder="Type something..."
          value={value}
          onChange={(e) => {
            if (e.target.value.length <= maxLength) {
              setValue(e.target.value);
            }
          }}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
        />
        {showCount && (
          <span className={`chat-input-char-count ${countClass}`}>
            {remaining}
          </span>
        )}
      </div>
      <button
        className="chat-send-btn"
        onClick={handleSend}
        disabled={disabled || value.trim().length === 0}
        aria-label="Send message"
      >
        ▶
      </button>
    </div>
  );
}
