/**
 * Shared types that mirror the Rust backend types exactly.
 *
 * These MUST stay in sync with `src-tauri/src/state_machine.rs`.
 * If Rust changes a type, update it here too — mismatches between
 * the backend and frontend are a direct threat to Rolo's health.
 */

/** Rolo's behavioral states. Maps 1:1 with Rust `PetState` enum (snake_case). */
export type PetState =
  | "idle"
  | "walk_left"
  | "walk_right"
  | "happy"
  | "drag_hover"
  | "sniffing"
  | "eating"
  | "satisfied"
  | "disappointed"
  | "sleeping";

/** Animation loop type — how the animation cycles. */
export type LoopType = "forward" | "ping-pong";

/** Metadata from animation_meta.json files in ASSETS/ folders. */
export interface AnimationMeta {
  animation: string;
  frames: number;
  fps: number;
  loop_type: LoopType;
  source: string;
}

/** Rolo's position in physical screen pixels. Mirrors Rust `Position`. */
export interface Position {
  x: number;
  y: number;
}

/** Full state snapshot emitted on state changes. Mirrors Rust `StatePayload`. */
export interface StatePayload {
  state: PetState;
  position: Position;
}

/**
 * Maps each PetState to its AnimationName for the animation system.
 * This bridges the state machine (which uses PetState) with the animation
 * registry (which uses AnimationName).
 */
export const STATE_TO_ANIMATION: Record<PetState, string> = {
  idle: "idle",
  walk_left: "walk_left",
  walk_right: "walk_right",
  happy: "happy",
  drag_hover: "drag_hover",
  sniffing: "sniffing",
  eating: "eating",
  satisfied: "satisfied",
  disappointed: "disappointed",
  sleeping: "sleep",
};
