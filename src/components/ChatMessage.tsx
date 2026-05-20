import { useState } from "react";

interface ChatMessageProps {
  role: "user" | "assistant" | "system";
  content: string;
  messageId?: number;
  onReport?: (messageId: number) => void;
}

export function ChatMessage({
  role,
  content,
  messageId,
  onReport,
}: ChatMessageProps) {
  const [reported, setReported] = useState(false);

  function handleReport() {
    if (reported || messageId == null || !onReport) return;
    setReported(true);
    onReport(messageId);
  }

  return (
    <div className={`chat-message ${role}`}>
      <div className="chat-message-bubble">
        {content}
        {role === "assistant" && messageId != null && onReport && (
          <button
            className={`chat-message-report ${reported ? "reported" : ""}`}
            onClick={handleReport}
            disabled={reported}
            aria-label="Report message"
          >
            ⚑
          </button>
        )}
      </div>
    </div>
  );
}
