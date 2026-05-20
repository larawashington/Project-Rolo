import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useChat } from "../hooks/useChat";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { TypingIndicator } from "./TypingIndicator";
import "../styles/chat.css";

export default function ChatWindow() {
  const { messages, isTyping, sendMessage, reportMessage, sessionId, initSession } =
    useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sourceRef = useRef<string>("speech");
  const [pendingInitialMessage, setPendingInitialMessage] = useState<
    string | null
  >(null);

  // Read session data from URL params (set by Rust open_chat command)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sid = params.get("session_id");
    const triggerText = params.get("trigger_text") || "";
    const source = params.get("source") || "speech";
    const initialUserMessage = params.get("initial_user_message") || "";

    if (sid) {
      sourceRef.current = source;
      initSession(sid, triggerText);
      if (initialUserMessage.trim().length > 0) {
        setPendingInitialMessage(initialUserMessage);
      }
    }
  }, [initSession]);

  // Once the session is wired up, fire any pending initial user message.
  // This runs after initSession has set sessionId (which is async re state),
  // so sendMessage's sessionId guard will pass.
  useEffect(() => {
    if (sessionId && pendingInitialMessage) {
      const text = pendingInitialMessage;
      setPendingInitialMessage(null);
      sendMessage(text);
    }
  }, [sessionId, pendingInitialMessage, sendMessage]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // Escape key closes chat
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        handleClose();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  async function handleClose() {
    try {
      await invoke("close_chat", {
        sessionId: sessionId,
        source: sourceRef.current,
      });
    } catch (err) {
      console.error("[Rolo] Failed to close chat:", err);
    }
  }

  async function handleMinimize() {
    try {
      await getCurrentWindow().minimize();
    } catch (err) {
      console.error("[Rolo] Failed to minimize chat:", err);
    }
  }

  return (
    <div className="chat-frame">
      <div className="chat-header" data-tauri-drag-region>
        <span className="chat-header-title" data-tauri-drag-region>
          Chatting with Rolo
        </span>
        <div className="chat-header-controls">
          <button
            className="chat-header-min"
            onClick={handleMinimize}
            aria-label="Minimize chat"
          >
            _
          </button>
          <button
            className="chat-header-close"
            onClick={handleClose}
            aria-label="Close chat"
          >
            X
          </button>
        </div>
      </div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <ChatMessage
            key={i}
            role={msg.role}
            content={msg.content}
            messageId={msg.id}
            onReport={reportMessage}
          />
        ))}
        {isTyping && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      <ChatInput onSend={sendMessage} disabled={isTyping} />
    </div>
  );
}
