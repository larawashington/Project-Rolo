import "./App.css";
import { useRoloState } from "./hooks/useRoloState";
import { useAnimationLoop } from "./hooks/useAnimationLoop";
import { useFileDrop } from "./hooks/useFileDrop";
import { useInteraction } from "./hooks/useInteraction";
import { STATE_TO_ANIMATION } from "./types/rolo";
import { ConfirmationBubble } from "./components/ConfirmationBubble";
import { InteractiveBubble } from "./components/InteractiveBubble";
import type { AnimationName } from "./animations";
import { invoke } from "@tauri-apps/api/core";

function App() {
  const { state, ready } = useRoloState();
  const { pendingFiles } = useFileDrop();
  const { activePrompt } = useInteraction();

  const animationName = (STATE_TO_ANIMATION[state] ?? "idle") as AnimationName;
  const frameSrc = useAnimationLoop(animationName);

  if (!ready || !frameSrc) {
    return null;
  }

  const isSniffing = state === "sniffing" && pendingFiles !== null;

  const handleConfirm = async () => {
    try {
      await invoke("confirm_eat");
    } catch (err) {
      console.error("[Rolo] confirm_eat failed:", err);
    }
  };

  const handleDecline = async () => {
    try {
      await invoke("decline_eat");
    } catch (err) {
      console.error("[Rolo] decline_eat failed:", err);
    }
  };

  return (
    <div className="rolo-world" onContextMenu={(e) => e.preventDefault()}>
      {isSniffing && (
        <ConfirmationBubble
          files={pendingFiles}
          onConfirm={handleConfirm}
          onDecline={handleDecline}
          timeoutMs={10000}
        />
      )}

      {activePrompt && !isSniffing && (
        <InteractiveBubble
          prompt={activePrompt}
          onResponse={() => {
            // Response submitted — Rust handles pet state transition
            // (Happy/Disappointed) and emits hide-interaction to clear
            // activePrompt via the useInteraction hook.
          }}
          onDismiss={() => {
            // Dismissed — Rust emits hide-interaction to clear activePrompt.
          }}
        />
      )}

      <div className="pet-container">
        <img
          src={frameSrc}
          alt="Rolo"
          className="pet-sprite"
          draggable={false}
          width={128}
          height={128}
        />
      </div>
    </div>
  );
}

export default App;
