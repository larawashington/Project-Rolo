/**
 * Pure frame-advancement logic extracted from useAnimationLoop for testability.
 *
 * These functions compute the next frame index given the current state.
 * They have no side effects — they just return the next index and direction.
 * The hook calls these on every animation tick.
 */

import type { LoopType } from "./types";

export interface FrameState {
  index: number;
  direction: 1 | -1;
}

/**
 * Advance a frame index by one step for a "forward" loop.
 * Wraps back to 0 when it reaches the end.
 */
export function advanceForward(index: number, totalFrames: number): number {
  return (index + 1) % totalFrames;
}

/**
 * Advance a frame index by one step for a "ping-pong" loop.
 *
 * Mirrors the logic in useAnimationLoop tick():
 *   - When going forward (direction=1) and we reach the end, we bounce:
 *       direction becomes -1, index becomes totalFrames - 2
 *   - When going backward (direction=-1) and we reach the start (<0), we bounce:
 *       direction becomes 1, index becomes 1
 *   - Single-frame edge case: index stays at 0 regardless of direction
 */
export function advancePingPong(
  index: number,
  direction: 1 | -1,
  totalFrames: number,
): FrameState {
  let idx = index + direction;

  if (idx >= totalFrames) {
    // Hit the end — bounce back
    const newDir: 1 | -1 = -1;
    idx = totalFrames - 2;
    // Single-frame animations can't bounce
    if (idx < 0) idx = 0;
    return { index: idx, direction: newDir };
  }

  if (idx < 0) {
    // Hit the start — bounce forward
    const newDir: 1 | -1 = 1;
    idx = 1;
    if (idx >= totalFrames) idx = 0;
    return { index: idx, direction: newDir };
  }

  return { index: idx, direction };
}

/**
 * Convenience wrapper: advance frame state by one step for any loop type.
 */
export function advanceFrame(
  state: FrameState,
  loopType: LoopType,
  totalFrames: number,
): FrameState {
  if (loopType === "ping-pong") {
    return advancePingPong(state.index, state.direction, totalFrames);
  }
  if (loopType === "once") {
    return { index: Math.min(state.index + 1, totalFrames - 1), direction: 1 };
  }
  return { index: advanceForward(state.index, totalFrames), direction: 1 };
}

/**
 * State for an animation that plays a one-shot intro then enters a sustained loop.
 *
 * - `phase` indicates whether we're still in the intro or have entered the loop.
 * - `introIndex` is a cursor while in the intro: 0..introFrames.length (advanced
 *   by one each tick; once it reaches introFrames.length we transition to loop).
 * - `loopIndex` is an index into `loopFrames` (NOT into the global frameSrcs).
 * - `direction` drives ping-pong inside the loop phase.
 */
export interface IntroLoopState {
  phase: "intro" | "loop";
  introIndex: number;
  loopIndex: number;
  direction: 1 | -1;
}

/**
 * Advance an IntroLoopState by one tick.
 *
 * - Intro phase: walks through `introFrames` indices in order (one per tick).
 *   When the cursor walks past the end, transitions to the loop phase starting
 *   at loopFrames[0].
 * - Loop phase: advances over the `loopFrames` subset using the requested
 *   loopType (forward or ping-pong).
 *
 * The returned state's `loopIndex` is an index into `loopFrames` (use
 * `frameSrcIndex` to translate back to a global frameSrcs index).
 */
export function advanceIntroLoop(
  state: IntroLoopState,
  introFrames: number[],
  loopFrames: number[],
  loopType: LoopType,
): IntroLoopState {
  if (state.phase === "intro") {
    const next = state.introIndex + 1;
    if (next >= introFrames.length) {
      // Intro just finished — first loop tick lands on loopFrames[0].
      return {
        phase: "loop",
        introIndex: introFrames.length,
        loopIndex: 0,
        direction: 1,
      };
    }
    return { ...state, introIndex: next };
  }

  // Loop phase: delegate to existing helpers, but operate over loopFrames.length.
  if (loopFrames.length === 0) {
    return state;
  }
  if (loopType === "ping-pong") {
    const r = advancePingPong(state.loopIndex, state.direction, loopFrames.length);
    return { ...state, loopIndex: r.index, direction: r.direction };
  }
  return {
    ...state,
    loopIndex: advanceForward(state.loopIndex, loopFrames.length),
    direction: 1,
  };
}

/**
 * Resolve the current global frameSrcs index from an IntroLoopState.
 *
 * In the intro phase we pull from `introFrames[introIndex]`; in the loop phase
 * we pull from `loopFrames[loopIndex]`. Falls back to 0 if the underlying
 * arrays are empty (defensive — Rolo should never be frameless).
 */
export function frameSrcIndex(
  state: IntroLoopState,
  introFrames: number[],
  loopFrames: number[],
): number {
  if (state.phase === "intro") {
    if (introFrames.length === 0) return 0;
    const i = Math.min(state.introIndex, introFrames.length - 1);
    return introFrames[i] ?? 0;
  }
  if (loopFrames.length === 0) return 0;
  return loopFrames[state.loopIndex] ?? 0;
}

/**
 * Simulate N steps of a loop and return the sequence of frame indices.
 * Used in tests to verify the full pattern.
 */
export function simulateFrameSequence(
  steps: number,
  loopType: LoopType,
  totalFrames: number,
): number[] {
  const result: number[] = [0]; // always starts at frame 0
  let state: FrameState = { index: 0, direction: 1 };

  for (let i = 0; i < steps; i++) {
    state = advanceFrame(state, loopType, totalFrames);
    result.push(state.index);
  }

  return result;
}
