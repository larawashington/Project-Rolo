/**
 * Health inspection: useAnimationLoop hook — FPS throttling and animation switching.
 *
 * These tests verify that Rolo advances frames at the correct rate and resets
 * properly when switching animations. A broken FPS throttle would make Rolo
 * move too fast (seizure-inducing) or too slow (nearly comatose).
 *
 * Because useAnimationLoop uses requestAnimationFrame, we mock rAF and manually
 * drive the animation clock.
 *
 * We also mock `getAnimation` so the tests are independent of import.meta.glob
 * and the actual ASSETS on disk.
 *
 * TIMING MODEL NOTE:
 * The hook checks `lastFrameTimeRef.current === 0` to detect the very first tick.
 * If we pass timestamp=0, it sets lastFrameTime=0 and the NEXT tick ALSO sees
 * lastFrameTimeRef === 0, causing frame 0 to show twice. We therefore start
 * at T0 = 1000ms (a realistic browser timestamp) so the first-frame condition
 * only fires once and subsequent ticks advance correctly.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useAnimationLoop } from "../hooks/useAnimationLoop";

// ---------------------------------------------------------------------------
// Mock getAnimation so we control the animation data completely.
// ---------------------------------------------------------------------------

vi.mock("../animations", () => ({
  getAnimation: vi.fn(),
}));

import { getAnimation } from "../animations";
const mockGetAnimation = vi.mocked(getAnimation);

// ---------------------------------------------------------------------------
// rAF mock helpers
// ---------------------------------------------------------------------------

/**
 * Installs a manual requestAnimationFrame implementation.
 * Calling triggerFrame(timestamp) fires all pending rAF callbacks.
 */
function setupRafMock() {
  const callbacks = new Map<number, FrameRequestCallback>();
  let idCounter = 1;

  const requestAnimationFrame = vi.fn((cb: FrameRequestCallback): number => {
    const id = idCounter++;
    callbacks.set(id, cb);
    return id;
  });

  const cancelAnimationFrame = vi.fn((id: number) => {
    callbacks.delete(id);
  });

  vi.stubGlobal("requestAnimationFrame", requestAnimationFrame);
  vi.stubGlobal("cancelAnimationFrame", cancelAnimationFrame);

  function triggerFrame(timestamp: number) {
    const pending = [...callbacks.entries()];
    callbacks.clear();
    for (const [, cb] of pending) {
      cb(timestamp);
    }
  }

  function pendingCount() {
    return callbacks.size;
  }

  return { triggerFrame, pendingCount };
}

// ---------------------------------------------------------------------------
// Test animation fixtures
// ---------------------------------------------------------------------------

const FORWARD_ANIM = {
  name: "eating",
  fps: 8,                  // 125ms per frame
  loopType: "forward" as const,
  frameSrcs: ["eat0.png", "eat1.png", "eat2.png", "eat3.png"],
};

const PINGPONG_ANIM = {
  name: "idle",
  fps: 4,                  // 250ms per frame (clean integer for easier math)
  loopType: "ping-pong" as const,
  frameSrcs: ["idle0.png", "idle1.png", "idle2.png", "idle3.png"],
};

const SINGLE_FRAME_ANIM = {
  name: "happy",
  fps: 10,
  loopType: "forward" as const,
  frameSrcs: ["happy0.png"],
};

// Use a realistic non-zero starting timestamp so the hook's first-frame
// detection (lastFrameTimeRef.current === 0) only fires once.
const T0 = 1000;                                    // ms
const MSPF_FORWARD = 1000 / FORWARD_ANIM.fps;       // 125ms
const MSPF_PINGPONG = 1000 / PINGPONG_ANIM.fps;     // 250ms

// ---------------------------------------------------------------------------
// Tests: initial state
// ---------------------------------------------------------------------------

describe("useAnimationLoop — first frame", () => {
  let triggerFrame: ReturnType<typeof setupRafMock>["triggerFrame"];

  beforeEach(() => {
    ({ triggerFrame } = setupRafMock());
    mockGetAnimation.mockReturnValue(FORWARD_ANIM);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("returns null before the first rAF tick", () => {
    const { result } = renderHook(() => useAnimationLoop("eating"));
    expect(result.current).toBeNull();
  });

  it("shows the first frame (index 0) on the very first tick", () => {
    const { result } = renderHook(() => useAnimationLoop("eating"));
    act(() => triggerFrame(T0));
    expect(result.current).toBe("eat0.png");
  });

  it("still shows frame 0 on second tick with insufficient elapsed time", () => {
    const { result } = renderHook(() => useAnimationLoop("eating"));
    act(() => triggerFrame(T0));
    act(() => triggerFrame(T0 + 50));
    expect(result.current).toBe("eat0.png");
  });
});

// ---------------------------------------------------------------------------
// Tests: FPS throttling
// ---------------------------------------------------------------------------

describe("useAnimationLoop — FPS throttling (forward, 8fps = 125ms/frame)", () => {
  let triggerFrame: ReturnType<typeof setupRafMock>["triggerFrame"];

  beforeEach(() => {
    ({ triggerFrame } = setupRafMock());
    mockGetAnimation.mockReturnValue(FORWARD_ANIM);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("does NOT advance before one full frame duration has elapsed", () => {
    const { result } = renderHook(() => useAnimationLoop("eating"));
    act(() => triggerFrame(T0));
    act(() => triggerFrame(T0 + MSPF_FORWARD - 1));   // 124ms — just short
    expect(result.current).toBe("eat0.png");
  });

  it("advances to frame 1 after exactly one frame duration", () => {
    const { result } = renderHook(() => useAnimationLoop("eating"));
    act(() => triggerFrame(T0));
    act(() => triggerFrame(T0 + MSPF_FORWARD));        // exactly 125ms
    expect(result.current).toBe("eat1.png");
  });

  it("advances through frames 0->1->2->3 at each 125ms interval", () => {
    const { result } = renderHook(() => useAnimationLoop("eating"));
    act(() => triggerFrame(T0));
    act(() => triggerFrame(T0 + MSPF_FORWARD * 1)); expect(result.current).toBe("eat1.png");
    act(() => triggerFrame(T0 + MSPF_FORWARD * 2)); expect(result.current).toBe("eat2.png");
    act(() => triggerFrame(T0 + MSPF_FORWARD * 3)); expect(result.current).toBe("eat3.png");
  });

  it("wraps from frame 3 back to frame 0 (forward loop)", () => {
    const { result } = renderHook(() => useAnimationLoop("eating"));
    act(() => triggerFrame(T0));
    act(() => triggerFrame(T0 + MSPF_FORWARD * 1));
    act(() => triggerFrame(T0 + MSPF_FORWARD * 2));
    act(() => triggerFrame(T0 + MSPF_FORWARD * 3));
    act(() => triggerFrame(T0 + MSPF_FORWARD * 4)); // wrap
    expect(result.current).toBe("eat0.png");
  });

  it("continues wrapping on a second cycle (8 advances of 4 frames = frame 0)", () => {
    const { result } = renderHook(() => useAnimationLoop("eating"));
    act(() => triggerFrame(T0));
    for (let i = 1; i <= 8; i++) {
      act(() => triggerFrame(T0 + MSPF_FORWARD * i));
    }
    expect(result.current).toBe("eat0.png");
  });
});

// ---------------------------------------------------------------------------
// Tests: ping-pong loop
// ---------------------------------------------------------------------------

describe("useAnimationLoop — ping-pong loop (idle, 4fps = 250ms/frame)", () => {
  let triggerFrame: ReturnType<typeof setupRafMock>["triggerFrame"];

  beforeEach(() => {
    ({ triggerFrame } = setupRafMock());
    mockGetAnimation.mockReturnValue(PINGPONG_ANIM);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("advances forward: frame 0 then 1 then 2 then 3", () => {
    const { result } = renderHook(() => useAnimationLoop("idle"));
    act(() => triggerFrame(T0));
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 1)); expect(result.current).toBe("idle1.png");
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 2)); expect(result.current).toBe("idle2.png");
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 3)); expect(result.current).toBe("idle3.png");
  });

  it("bounces at the end: 3 -> 2 -> 1", () => {
    const { result } = renderHook(() => useAnimationLoop("idle"));
    act(() => triggerFrame(T0));
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 1));  // 1
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 2));  // 2
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 3));  // 3
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 4));  // bounce -> 2
    expect(result.current).toBe("idle2.png");
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 5));  // 1
    expect(result.current).toBe("idle1.png");
  });

  it("bounces at the start: 0 -> 1 (forward again)", () => {
    const { result } = renderHook(() => useAnimationLoop("idle"));
    act(() => triggerFrame(T0));
    for (let i = 1; i <= 6; i++) {
      act(() => triggerFrame(T0 + MSPF_PINGPONG * i)); // 0->1->2->3->2->1->0
    }
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 7)); // bounce from 0 -> 1
    expect(result.current).toBe("idle1.png");
  });

  it("produces the canonical pattern 0,1,2,3,2,1,0,1,2,3 over 9 steps", () => {
    const { result } = renderHook(() => useAnimationLoop("idle"));
    const seq: string[] = [];

    // Read result.current AFTER act() so React has flushed state updates.
    act(() => { triggerFrame(T0); });
    seq.push(result.current!);
    for (let i = 1; i <= 9; i++) {
      act(() => { triggerFrame(T0 + MSPF_PINGPONG * i); });
      seq.push(result.current!);
    }

    expect(seq).toEqual([
      "idle0.png",
      "idle1.png",
      "idle2.png",
      "idle3.png",
      "idle2.png",
      "idle1.png",
      "idle0.png",
      "idle1.png",
      "idle2.png",
      "idle3.png",
    ]);
  });

  it("never produces an out-of-bounds frame over 50 ticks", () => {
    const { result } = renderHook(() => useAnimationLoop("idle"));
    act(() => triggerFrame(T0));
    for (let i = 1; i <= 50; i++) {
      act(() => triggerFrame(T0 + MSPF_PINGPONG * i));
      expect(PINGPONG_ANIM.frameSrcs).toContain(result.current);
    }
  });
});

// ---------------------------------------------------------------------------
// Tests: animation switching
// ---------------------------------------------------------------------------

describe("useAnimationLoop — animation switching", () => {
  let triggerFrame: ReturnType<typeof setupRafMock>["triggerFrame"];

  beforeEach(() => {
    ({ triggerFrame } = setupRafMock());
    mockGetAnimation.mockImplementation((name) => {
      if (name === "eating") return FORWARD_ANIM;
      if (name === "idle") return PINGPONG_ANIM;
      return undefined;
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("resets to frame 0 immediately on the first tick after switching", () => {
    const { result, rerender } = renderHook(
      ({ name }: { name: Parameters<typeof useAnimationLoop>[0] }) =>
        useAnimationLoop(name),
      { initialProps: { name: "eating" as Parameters<typeof useAnimationLoop>[0] } }
    );

    // Advance eating to frame 3
    act(() => triggerFrame(T0));
    act(() => triggerFrame(T0 + MSPF_FORWARD * 1));
    act(() => triggerFrame(T0 + MSPF_FORWARD * 2));
    act(() => triggerFrame(T0 + MSPF_FORWARD * 3));
    expect(result.current).toBe("eat3.png");

    // Switch to idle — hook's useEffect resets lastFrameTime to 0
    rerender({ name: "idle" as Parameters<typeof useAnimationLoop>[0] });

    // First tick after switch -> first-frame path -> idle frame 0
    act(() => triggerFrame(T0 + MSPF_FORWARD * 3 + 50));
    expect(result.current).toBe("idle0.png");
  });

  it("advances from frame 0 (not from previous frame index) after switch", () => {
    const { result, rerender } = renderHook(
      ({ name }: { name: Parameters<typeof useAnimationLoop>[0] }) =>
        useAnimationLoop(name),
      { initialProps: { name: "eating" as Parameters<typeof useAnimationLoop>[0] } }
    );

    // Advance eating to frame 2
    act(() => triggerFrame(T0));
    act(() => triggerFrame(T0 + MSPF_FORWARD * 1));
    act(() => triggerFrame(T0 + MSPF_FORWARD * 2));
    expect(result.current).toBe("eat2.png");

    rerender({ name: "idle" as Parameters<typeof useAnimationLoop>[0] });
    const switchTime = T0 + MSPF_FORWARD * 2 + 10;
    act(() => triggerFrame(switchTime));                        // idle frame 0
    act(() => triggerFrame(switchTime + MSPF_PINGPONG));        // idle frame 1
    expect(result.current).toBe("idle1.png");
  });

  it("resets direction to forward when leaving a reversing ping-pong", () => {
    const { result, rerender } = renderHook(
      ({ name }: { name: Parameters<typeof useAnimationLoop>[0] }) =>
        useAnimationLoop(name),
      { initialProps: { name: "idle" as Parameters<typeof useAnimationLoop>[0] } }
    );

    // Advance idle into reverse phase: 0->1->2->3->2
    act(() => triggerFrame(T0));
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 1));
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 2));
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 3));
    act(() => triggerFrame(T0 + MSPF_PINGPONG * 4)); // idle2, going backward
    expect(result.current).toBe("idle2.png");

    // Switch to eating — should restart at frame 0 going forward
    rerender({ name: "eating" as Parameters<typeof useAnimationLoop>[0] });
    const switchTime = T0 + MSPF_PINGPONG * 4 + 10;
    act(() => triggerFrame(switchTime));                        // eat0
    expect(result.current).toBe("eat0.png");

    act(() => triggerFrame(switchTime + MSPF_FORWARD));         // eat1 (forward)
    expect(result.current).toBe("eat1.png");

    act(() => triggerFrame(switchTime + MSPF_FORWARD * 2));     // eat2 (still forward)
    expect(result.current).toBe("eat2.png");
  });
});

// ---------------------------------------------------------------------------
// Tests: single-frame animation
// ---------------------------------------------------------------------------

describe("useAnimationLoop — single frame animation", () => {
  let triggerFrame: ReturnType<typeof setupRafMock>["triggerFrame"];

  beforeEach(() => {
    ({ triggerFrame } = setupRafMock());
    mockGetAnimation.mockReturnValue(SINGLE_FRAME_ANIM);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("always shows frame 0 regardless of elapsed time", () => {
    const { result } = renderHook(() => useAnimationLoop("happy"));
    act(() => triggerFrame(T0));
    act(() => triggerFrame(T0 + 1000));
    act(() => triggerFrame(T0 + 10000));
    expect(result.current).toBe("happy0.png");
  });
});

// ---------------------------------------------------------------------------
// Tests: cleanup / memory safety
// ---------------------------------------------------------------------------

describe("useAnimationLoop — cleanup", () => {
  let triggerFrame: ReturnType<typeof setupRafMock>["triggerFrame"];
  let pendingCount: ReturnType<typeof setupRafMock>["pendingCount"];

  beforeEach(() => {
    ({ triggerFrame, pendingCount } = setupRafMock());
    mockGetAnimation.mockReturnValue(FORWARD_ANIM);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("cancels the pending rAF on unmount (no memory leak)", () => {
    const { unmount } = renderHook(() => useAnimationLoop("eating"));
    act(() => triggerFrame(T0)); // start the loop
    expect(pendingCount()).toBeGreaterThan(0);

    unmount();
    expect(vi.mocked(cancelAnimationFrame)).toHaveBeenCalled();
  });
});
