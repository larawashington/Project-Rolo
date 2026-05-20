/**
 * Health inspection: frame advancement logic (forward + ping-pong loops).
 *
 * These tests verify the core animation tick behavior that drives Rolo's movement.
 * A broken ping-pong implementation would cause Rolo to stutter or jump backwards
 * at unexpected moments — a jarring symptom of a sick animation system.
 *
 * We test the pure frame-advancement functions extracted from useAnimationLoop.
 */

import { describe, it, expect } from "vitest";
import {
  advanceForward,
  advancePingPong,
  advanceFrame,
  advanceIntroLoop,
  frameSrcIndex,
  simulateFrameSequence,
} from "../animations/frame-utils";
import type { FrameState, IntroLoopState } from "../animations/frame-utils";
import type { LoopType } from "../animations/types";

// ---------------------------------------------------------------------------
// advanceForward
// ---------------------------------------------------------------------------

describe("advanceForward", () => {
  it("advances from frame 0 to frame 1", () => {
    expect(advanceForward(0, 4)).toBe(1);
  });

  it("advances from frame 2 to frame 3 in a 4-frame animation", () => {
    expect(advanceForward(2, 4)).toBe(3);
  });

  it("wraps from the last frame back to 0", () => {
    expect(advanceForward(3, 4)).toBe(0);
  });

  it("wraps correctly for a 13-frame eating animation", () => {
    expect(advanceForward(12, 13)).toBe(0);
  });

  it("single-frame animation always stays at 0", () => {
    expect(advanceForward(0, 1)).toBe(0);
  });

  it("produces correct sequence over many steps (forward 4-frame)", () => {
    let idx = 0;
    const seq: number[] = [0];
    for (let i = 0; i < 8; i++) {
      idx = advanceForward(idx, 4);
      seq.push(idx);
    }
    expect(seq).toEqual([0, 1, 2, 3, 0, 1, 2, 3, 0]);
  });
});

// ---------------------------------------------------------------------------
// advancePingPong
// ---------------------------------------------------------------------------

describe("advancePingPong", () => {
  // Basic forward progression
  it("advances from 0 to 1 going forward (4-frame)", () => {
    const result = advancePingPong(0, 1, 4);
    expect(result.index).toBe(1);
    expect(result.direction).toBe(1);
  });

  it("advances from 1 to 2 going forward (4-frame)", () => {
    const result = advancePingPong(1, 1, 4);
    expect(result.index).toBe(2);
    expect(result.direction).toBe(1);
  });

  // Bounce at end
  it("bounces at the end — index becomes totalFrames-2 and direction flips to -1", () => {
    // At frame 3 going forward in a 4-frame animation → would be idx 4 → bounce
    const result = advancePingPong(3, 1, 4);
    expect(result.index).toBe(2); // totalFrames - 2 = 2
    expect(result.direction).toBe(-1);
  });

  // Backward progression after bounce
  it("advances from 2 to 1 going backward (4-frame)", () => {
    const result = advancePingPong(2, -1, 4);
    expect(result.index).toBe(1);
    expect(result.direction).toBe(-1);
  });

  it("advances from 1 to 0 going backward (4-frame)", () => {
    const result = advancePingPong(1, -1, 4);
    expect(result.index).toBe(0);
    expect(result.direction).toBe(-1);
  });

  // Bounce at start
  it("bounces at the start — index becomes 1 and direction flips to +1", () => {
    // At frame 0 going backward → would be idx -1 → bounce
    const result = advancePingPong(0, -1, 4);
    expect(result.index).toBe(1);
    expect(result.direction).toBe(1);
  });

  // Single-frame edge case
  it("single-frame ping-pong: bouncing at end stays at 0 (no underflow)", () => {
    // totalFrames=1: idx = 0+1 = 1 >= 1 → bounce: idx = 1-2 = -1 → clamp to 0
    const result = advancePingPong(0, 1, 1);
    expect(result.index).toBe(0);
    expect(result.direction).toBe(-1);
  });

  it("single-frame ping-pong: bouncing at start stays at 0 (no overflow)", () => {
    // totalFrames=1: idx = 0-1 = -1 < 0 → bounce: idx = 1 → but 1 >= 1 → clamp to 0
    const result = advancePingPong(0, -1, 1);
    expect(result.index).toBe(0);
    expect(result.direction).toBe(1);
  });

  // Two-frame ping-pong
  it("two-frame ping-pong oscillates 0, 1, 0, 1", () => {
    let state: FrameState = { index: 0, direction: 1 };
    const seq = [0];
    for (let i = 0; i < 5; i++) {
      state = advancePingPong(state.index, state.direction, 2);
      seq.push(state.index);
    }
    // 0 → 1 → (bounce) → 0 → (bounce) → 1 → (bounce) → 0
    expect(seq).toEqual([0, 1, 0, 1, 0, 1]);
  });
});

// ---------------------------------------------------------------------------
// simulateFrameSequence — full pattern verification
// ---------------------------------------------------------------------------

describe("simulateFrameSequence — forward", () => {
  it("produces 0,1,2,3,0,1,2,3,0 for a 4-frame forward loop over 8 steps", () => {
    const seq = simulateFrameSequence(8, "forward", 4);
    expect(seq).toEqual([0, 1, 2, 3, 0, 1, 2, 3, 0]);
  });

  it("produces correct wrap for 13-frame eating animation over 14 steps", () => {
    const seq = simulateFrameSequence(14, "forward", 13);
    // 0,1,2,...,12,0,1
    const expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1];
    expect(seq).toEqual(expected);
  });

  it("single-frame forward loop always shows frame 0", () => {
    const seq = simulateFrameSequence(5, "forward", 1);
    expect(seq).toEqual([0, 0, 0, 0, 0, 0]);
  });
});

describe("simulateFrameSequence — ping-pong", () => {
  it("produces 0,1,2,3,2,1,0,1,2,3 for a 4-frame ping-pong over 9 steps", () => {
    // This is the canonical ping-pong pattern Rolo's idle animation uses
    const seq = simulateFrameSequence(9, "ping-pong", 4);
    expect(seq).toEqual([0, 1, 2, 3, 2, 1, 0, 1, 2, 3]);
  });

  it("ping-pong never produces out-of-bounds indices", () => {
    const totalFrames = 7; // like the sniffing animation
    const seq = simulateFrameSequence(30, "ping-pong", totalFrames);
    seq.forEach((idx) => {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(totalFrames);
    });
  });

  it("ping-pong visits every frame in a 4-frame animation", () => {
    const seq = simulateFrameSequence(20, "ping-pong", 4);
    expect(seq).toContain(0);
    expect(seq).toContain(1);
    expect(seq).toContain(2);
    expect(seq).toContain(3);
  });

  it("2-frame ping-pong produces 0,1,0,1,... pattern", () => {
    const seq = simulateFrameSequence(5, "ping-pong", 2);
    expect(seq).toEqual([0, 1, 0, 1, 0, 1]);
  });

  it("single-frame ping-pong never advances beyond frame 0", () => {
    const seq = simulateFrameSequence(6, "ping-pong", 1);
    seq.forEach((idx) => expect(idx).toBe(0));
  });

  it("ping-pong does NOT repeat the first and last frame on bounce", () => {
    // For 4 frames: correct is 0,1,2,3,2,1,0,1,...
    // Wrong would be: 0,1,2,3,3,2,1,0,0,1,... (repeating endpoints)
    const seq = simulateFrameSequence(10, "ping-pong", 4);
    // Check no two consecutive identical frames anywhere in the sequence
    // (In a proper ping-pong with 4+ frames, endpoints should only appear once per pass)
    for (let i = 1; i < seq.length; i++) {
      expect(seq[i]).not.toBe(seq[i - 1]);
    }
  });
});

// ---------------------------------------------------------------------------
// advanceFrame — dispatcher
// ---------------------------------------------------------------------------

describe("advanceFrame", () => {
  it("delegates to forward logic for forward loop type", () => {
    const state: FrameState = { index: 2, direction: 1 };
    const next = advanceFrame(state, "forward", 4);
    expect(next.index).toBe(3);
  });

  it("delegates to ping-pong logic for ping-pong loop type", () => {
    const state: FrameState = { index: 3, direction: 1 };
    const next = advanceFrame(state, "ping-pong", 4);
    expect(next.index).toBe(2);
    expect(next.direction).toBe(-1);
  });

  it("once: advances toward the last frame and then holds it", () => {
    // Satisfied is 9 frames, loop_type: "once". After 8 advances we should
    // sit on frame 8 and stay there no matter how long the state holds.
    let state: FrameState = { index: 0, direction: 1 };
    for (let i = 0; i < 8; i++) {
      state = advanceFrame(state, "once", 9);
    }
    expect(state.index).toBe(8);
    // Holding indefinitely — many more ticks must not wrap to frame 0.
    for (let i = 0; i < 50; i++) {
      state = advanceFrame(state, "once", 9);
      expect(state.index).toBe(8);
    }
  });
});

// ---------------------------------------------------------------------------
// advanceIntroLoop + frameSrcIndex — intro_frames + loop_frames support
//
// Sleep is the canonical case: intro=[0,1] plays once on entry, then
// loop=[2..7] ping-pongs while Rolo dreams. A bug here would mean Rolo
// either skips his settle-down or never enters the sustained loop —
// either way, his dream is broken.
// ---------------------------------------------------------------------------

/**
 * Helper: simulate `steps` ticks of an intro+loop animation, returning the
 * sequence of *global* frameSrcs indices visited (including the initial frame).
 */
function simulateIntroLoop(
  steps: number,
  introFrames: number[],
  loopFrames: number[],
  loopType: LoopType,
): number[] {
  let state: IntroLoopState = {
    phase: "intro",
    introIndex: 0,
    loopIndex: 0,
    direction: 1,
  };
  const seq: number[] = [frameSrcIndex(state, introFrames, loopFrames)];
  for (let i = 0; i < steps; i++) {
    state = advanceIntroLoop(state, introFrames, loopFrames, loopType);
    seq.push(frameSrcIndex(state, introFrames, loopFrames));
  }
  return seq;
}

describe("advanceIntroLoop_plays_intro_then_enters_loop", () => {
  it("walks intro [0,1] then enters loop [2..7] starting at 2", () => {
    const intro = [0, 1];
    const loop = [2, 3, 4, 5, 6, 7];
    // Initial = 0; tick1 = 1 (still intro); tick2 = enters loop -> loopFrames[0]=2;
    // tick3 = loopFrames[1]=3.
    const seq = simulateIntroLoop(3, intro, loop, "ping-pong");
    expect(seq).toEqual([0, 1, 2, 3]);
  });
});

describe("advanceIntroLoop_pingpongs_loop_after_intro", () => {
  it("intro=[0,1] then ping-pongs loop=[2,3,4]: 0,1,2,3,4,3,2,3,4,...", () => {
    const intro = [0, 1];
    const loop = [2, 3, 4];
    // Initial 0; +1 = 1 (intro end); +1 enters loop at loop[0]=2;
    // then ping-pongs over loop indices: 2, 3, 4 (bounce -> -1) -> 3 -> 2 (bounce -> +1) -> 3 -> 4 ...
    // Translated to global indices: 0,1,2,3,4,3,2,3,4
    const seq = simulateIntroLoop(8, intro, loop, "ping-pong");
    expect(seq).toEqual([0, 1, 2, 3, 4, 3, 2, 3, 4]);
  });
});

describe("advanceIntroLoop_forward_loops_loop_after_intro", () => {
  it("intro=[0] then forward-loops loop=[1,2,3]: 0,1,2,3,1,2,3,...", () => {
    const intro = [0];
    const loop = [1, 2, 3];
    // Initial = 0; tick1 = enters loop at loop[0]=1; tick2 = loop[1]=2;
    // tick3 = loop[2]=3; tick4 = wraps -> loop[0]=1; tick5 = loop[1]=2; tick6 = loop[2]=3.
    const seq = simulateIntroLoop(6, intro, loop, "forward");
    expect(seq).toEqual([0, 1, 2, 3, 1, 2, 3]);
  });
});

describe("frameSrcIndex_resolves_intro_phase_correctly", () => {
  it("returns introFrames[introIndex] while in intro phase", () => {
    const intro = [0, 1];
    const loop = [2, 3, 4];
    const s0: IntroLoopState = { phase: "intro", introIndex: 0, loopIndex: 0, direction: 1 };
    const s1: IntroLoopState = { phase: "intro", introIndex: 1, loopIndex: 0, direction: 1 };
    expect(frameSrcIndex(s0, intro, loop)).toBe(0);
    expect(frameSrcIndex(s1, intro, loop)).toBe(1);
  });

  it("clamps introIndex to last intro entry to avoid out-of-bounds", () => {
    const intro = [0, 1];
    const loop = [2, 3];
    const oob: IntroLoopState = { phase: "intro", introIndex: 99, loopIndex: 0, direction: 1 };
    // Defensive: never undefined, never NaN — Rolo always has a frame to show.
    expect(frameSrcIndex(oob, intro, loop)).toBe(1);
  });
});

describe("frameSrcIndex_resolves_loop_phase_correctly", () => {
  it("returns loopFrames[loopIndex] while in loop phase", () => {
    const intro = [0, 1];
    const loop = [2, 3, 4, 5];
    const s0: IntroLoopState = { phase: "loop", introIndex: 2, loopIndex: 0, direction: 1 };
    const s2: IntroLoopState = { phase: "loop", introIndex: 2, loopIndex: 2, direction: 1 };
    expect(frameSrcIndex(s0, intro, loop)).toBe(2);
    expect(frameSrcIndex(s2, intro, loop)).toBe(4);
  });
});
