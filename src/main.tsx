import React from "react";
import ReactDOM from "react-dom/client";
import { getCurrentWindow } from "@tauri-apps/api/window";
import App from "./App";
import SpeechBubble from "./components/SpeechBubble";
import StatusPanel from "./components/StatusPanel";
import ChatWindow from "./components/ChatWindow";
import CommandCenter from "./components/CommandCenter/CommandCenter";

/**
 * Window-label routing: the same React build serves every Tauri window.
 * We check the window label to decide which component tree to render.
 *
 * - "speech-bubble"  → renders the pixel-art speech bubble UI
 * - "status-panel"   → renders Rolo's mood/bars character-sheet panel
 * - "chat"           → renders the chat window
 * - "command-center" → renders the settings/diagnostics surface (fresh
 *                      installs land here in modal setup mode — replaced
 *                      the legacy first-run wizard in PRD Phase 10)
 * - anything else    → renders Rolo (the pet sprite), safe fallback
 */
const windowLabel = getCurrentWindow().label;

// Suppress the WebView's native context menu in the main pet window so it
// never races with the Rust-side popup_menu. Capture-phase + preventDefault
// stops the browser default before any subtree handler runs. The
// command-center window is a decorated settings surface where the user
// should retain normal right-click affordances.
if (windowLabel !== "command-center") {
  window.addEventListener(
    "contextmenu",
    (e) => e.preventDefault(),
    { capture: true },
  );
}

const root = document.getElementById("root") as HTMLElement;

if (windowLabel === "speech-bubble") {
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <SpeechBubble />
    </React.StrictMode>,
  );
} else if (windowLabel === "status-panel") {
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <StatusPanel />
    </React.StrictMode>,
  );
} else if (windowLabel === "chat") {
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <ChatWindow />
    </React.StrictMode>,
  );
} else if (windowLabel === "command-center") {
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <CommandCenter />
    </React.StrictMode>,
  );
} else {
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
}
