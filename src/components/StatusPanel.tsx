/**
 * StatusPanel — Rolo's pixel-art "character sheet".
 *
 * Renders four mood bars (Hunger, Social, Energy, Happiness) and the
 * derived mood word. Lives in the "status-panel" Tauri window. Receives
 * `mood-tick` events from the Rust tick loop while open, and dismisses
 * itself on Escape, blur, or 30s of mouse-idle (the macOS blur fallback
 * — PRD §6).
 */

import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import DreamLog from "./DreamLog";
import "./StatusPanel.css";

type StatusTab = "mood" | "dreams";

interface MoodSnapshot {
  hunger: number;
  social: number;
  energy: number;
  happiness: number;
  mood_word: string;
}

const SEGMENTS = 10;
const IDLE_TIMEOUT_MS = 30_000;

const DEFAULT_SNAPSHOT: MoodSnapshot = {
  hunger: 0.5,
  social: 0.5,
  energy: 0.5,
  happiness: 0.5,
  mood_word: "neutral",
};

function colorClass(value: number): string {
  if (value >= 0.6) return "bar-green";
  if (value >= 0.25) return "bar-yellow";
  return "bar-red";
}

function filledSegments(value: number): number {
  const clamped = Math.max(0, Math.min(1, value));
  return Math.round(clamped * SEGMENTS);
}

interface BarRowProps {
  label: string;
  value: number;
}

function BarRow({ label, value }: BarRowProps) {
  const filled = filledSegments(value);
  const cls = colorClass(value);
  return (
    <div className="bar-row">
      <span className="bar-label">{label}</span>
      <div className="bar-track" role="progressbar" aria-valuenow={filled} aria-valuemin={0} aria-valuemax={SEGMENTS}>
        {Array.from({ length: SEGMENTS }).map((_, i) => (
          <div
            key={i}
            className={`bar-segment ${i < filled ? cls : "bar-empty"}`}
          />
        ))}
      </div>
    </div>
  );
}

export default function StatusPanel() {
  const [snapshot, setSnapshot] = useState<MoodSnapshot>(DEFAULT_SNAPSHOT);
  const [activeTab, setActiveTab] = useState<StatusTab>("mood");
  const idleTimerRef = useRef<number | null>(null);

  const resetIdleTimer = () => {
    if (idleTimerRef.current !== null) {
      window.clearTimeout(idleTimerRef.current);
    }
    idleTimerRef.current = window.setTimeout(() => {
      void invoke("close_status_panel").catch(() => {});
    }, IDLE_TIMEOUT_MS);
  };

  useEffect(() => {
    let unlistenTick: UnlistenFn | null = null;
    let unlistenBlur: UnlistenFn | null = null;
    let cancelled = false;

    async function setup() {
      // Fetch initial snapshot as a fallback if the open-time emit was missed.
      try {
        const initial = await invoke<MoodSnapshot>("get_mood_snapshot");
        if (!cancelled) setSnapshot(initial);
      } catch (e) {
        console.warn("[StatusPanel] get_mood_snapshot failed:", e);
      }

      unlistenTick = await listen<MoodSnapshot>("mood-tick", (event) => {
        if (!cancelled) setSnapshot(event.payload);
      });

      // Blur dismissal — clicking outside the panel hides it.
      const win = getCurrentWindow();
      unlistenBlur = await win.onFocusChanged(({ payload: focused }) => {
        if (!focused) {
          void invoke("close_status_panel").catch(() => {});
        }
      });
    }

    void setup();

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        void invoke("close_status_panel").catch(() => {});
      }
    };
    const handleMouseMove = () => resetIdleTimer();

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("mousemove", handleMouseMove);
    resetIdleTimer();

    return () => {
      cancelled = true;
      if (unlistenTick) unlistenTick();
      if (unlistenBlur) unlistenBlur();
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("mousemove", handleMouseMove);
      if (idleTimerRef.current !== null) {
        window.clearTimeout(idleTimerRef.current);
      }
    };
  }, []);

  return (
    <div className="status-panel">
      <div className="statuspanel-tabs">
        <button
          type="button"
          className={`statuspanel-tab ${
            activeTab === "mood" ? "statuspanel-tab-active" : ""
          }`}
          onClick={() => setActiveTab("mood")}
        >
          Mood
        </button>
        <button
          type="button"
          className={`statuspanel-tab ${
            activeTab === "dreams" ? "statuspanel-tab-active" : ""
          }`}
          onClick={() => setActiveTab("dreams")}
        >
          Dreams
        </button>
      </div>
      {activeTab === "mood" && (
        <>
          <div className="status-header">
            Rolo's current mood:{" "}
            <span className="mood-word">{snapshot.mood_word}</span>
          </div>
          <div className="bars">
            <BarRow label="Hunger" value={snapshot.hunger} />
            <BarRow label="Social" value={snapshot.social} />
            <BarRow label="Energy" value={snapshot.energy} />
            <BarRow label="Happy" value={snapshot.happiness} />
          </div>
        </>
      )}
      {activeTab === "dreams" && <DreamLog />}
    </div>
  );
}
