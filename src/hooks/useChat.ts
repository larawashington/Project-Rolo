import { useState, useCallback, useRef } from "react";
import { invoke, Channel } from "@tauri-apps/api/core";

export interface ChatMsg {
  id?: number;
  role: "user" | "assistant" | "system";
  content: string;
  isStreaming?: boolean;
}

interface ChatStreamEvent {
  type: "Token" | "Done" | "Error";
  text?: string;
  full_text?: string;
  message_id?: number;
  message?: string;
}

const FUZZY_BRAIN_MSG =
  "*blinks* my brain feels fuzzy — open Command Center to wake me up.";

// Replace the trailing message if it's still streaming, otherwise append.
// Both Done/Error happen mid-stream (replace) and pre-stream (append, when no
// Token ever arrived) — this collapses both cases.
function replaceStreamingOrAppend(prev: ChatMsg[], msg: ChatMsg): ChatMsg[] {
  const last = prev[prev.length - 1];
  if (last && last.isStreaming) {
    const updated = prev.slice();
    updated[updated.length - 1] = msg;
    return updated;
  }
  return [...prev, msg];
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const streamingRef = useRef(false);

  const initSession = useCallback((id: string, triggerText: string) => {
    setSessionId(id);
    if (triggerText.trim().length > 0) {
      setMessages([{ role: "assistant", content: triggerText }]);
    } else {
      setMessages([]);
    }
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!sessionId || streamingRef.current) return;

      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setIsTyping(true);
      streamingRef.current = true;

      const onEvent = new Channel<ChatStreamEvent>();
      onEvent.onmessage = (event: ChatStreamEvent) => {
        if (event.type === "Token" && event.text) {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.isStreaming) {
              updated[updated.length - 1] = {
                ...last,
                content: last.content + event.text,
              };
              return updated;
            }
            return [
              ...updated,
              { role: "assistant", content: event.text!, isStreaming: true },
            ];
          });
          setIsTyping(false); // Hide typing indicator once tokens arrive
        } else if (event.type === "Done" && event.full_text != null) {
          setMessages((prev) =>
            replaceStreamingOrAppend(prev, {
              role: "assistant",
              content: event.full_text!,
              id: event.message_id,
              isStreaming: false,
            }),
          );
          setIsTyping(false);
          streamingRef.current = false;
        } else if (event.type === "Error") {
          setMessages((prev) =>
            replaceStreamingOrAppend(prev, {
              role: "assistant",
              content: FUZZY_BRAIN_MSG,
              isStreaming: false,
            }),
          );
          setIsTyping(false);
          streamingRef.current = false;
          console.error("[Rolo] Chat error:", event.message);
        }
      };

      try {
        await invoke("send_chat_message", {
          text,
          sessionId,
          onEvent,
        });
      } catch (err) {
        // Backend emits ChatStreamEvent::Error *and* returns Err; if the
        // onmessage handler already cleared streamingRef, the error is
        // already on screen — skip to avoid a duplicate append.
        if (!streamingRef.current) {
          console.error("[Rolo] send_chat_message failed (already shown):", err);
          return;
        }
        setMessages((prev) =>
          replaceStreamingOrAppend(prev, {
            role: "assistant",
            content: FUZZY_BRAIN_MSG,
            isStreaming: false,
          }),
        );
        setIsTyping(false);
        streamingRef.current = false;
        console.error("[Rolo] send_chat_message failed:", err);
      }
    },
    [sessionId, messages.length],
  );

  const reportMessage = useCallback(async (messageId: number) => {
    try {
      await invoke("report_chat_message", { messageId });
    } catch (err) {
      console.error("[Rolo] Failed to report message:", err);
    }
  }, []);

  return {
    messages,
    isTyping,
    sendMessage,
    reportMessage,
    sessionId,
    initSession,
  };
}
