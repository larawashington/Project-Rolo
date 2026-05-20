import { useCallback, useEffect, useRef, useState } from "react";
import type { AnimationName } from "../animations";
import { getAnimation } from "../animations";
import {
  advanceIntroLoop,
  frameSrcIndex,
  type IntroLoopState,
} from "../animations/frame-utils";

/**
 * Drives Rolo's sprite animation using requestAnimationFrame, respecting
 * per-animation FPS from animation_meta.json.
 *
 * Supports both "forward" and "ping-pong" loop types:
 *   forward   — 0, 1, 2, …, N, 0, 1, …
 *   ping-pong — 0, 1, 2, …, N, …, 2, 1, 0, 1, …
 *
 * Also supports animations with separate `intro_frames` + `loop_frames`
 * subsets in animation_meta.json — the intro plays once on entry, then the
 * loop subset cycles per loop_type. Sleep is the canonical example: a brief
 * settle-down intro followed by a sustained ping-pong while Rolo dreams.
 *
 * Returns the image source URL for the current frame. Switching animation
 * names resets the frame index so Rolo starts each new expression from the
 * beginning.
 */
export function useAnimationLoop(animationName: AnimationName): string | null {
  const [frameSrc, setFrameSrc] = useState<string | null>(null);

  // Mutable refs to avoid re-creating the rAF callback on every state change.
  const frameIndexRef = useRef(0);
  const directionRef = useRef<1 | -1>(1); // 1 = forward, -1 = reversing
  const lastFrameTimeRef = useRef(-1); // -1 = uninitialized; 0 is a valid rAF timestamp
  const rafIdRef = useRef(0);
  const animNameRef = useRef(animationName);

  // Parallel state for animations that declare intro_frames + loop_frames.
  // null means the current animation uses the simple single-list path.
  const introLoopStateRef = useRef<IntroLoopState | null>(null);

  // Reset frame state whenever the animation changes.
  if (animationName !== animNameRef.current) {
    animNameRef.current = animationName;
    frameIndexRef.current = 0;
    directionRef.current = 1;
    lastFrameTimeRef.current = -1;
    introLoopStateRef.current = null;
  }

  const tick = useCallback((timestamp: number) => {
    const anim = getAnimation(animNameRef.current);
    if (!anim || anim.frameSrcs.length === 0) {
      rafIdRef.current = requestAnimationFrame(tick);
      return;
    }

    // "once" animations freeze on the last frame. Once we land there, stop
    // rescheduling — the useEffect on animationName will restart the loop
    // when Rolo transitions out. Without this, rAF keeps firing at 60Hz
    // and re-setting the same frameSrc for the entire state duration.
    if (
      anim.loopType === "once" &&
      lastFrameTimeRef.current !== -1 &&
      frameIndexRef.current === anim.frameSrcs.length - 1
    ) {
      return;
    }

    const hasIntroLoop =
      anim.introFrames !== undefined &&
      anim.loopFrames !== undefined &&
      anim.introFrames.length > 0 &&
      anim.loopFrames.length > 0;

    const msPerFrame = 1000 / anim.fps;

    // First frame — show immediately.
    if (lastFrameTimeRef.current === -1) {
      lastFrameTimeRef.current = timestamp;
      if (hasIntroLoop) {
        // Initialise the intro/loop cursor and show frameSrcs[introFrames[0]].
        const initial: IntroLoopState = {
          phase: "intro",
          introIndex: 0,
          loopIndex: 0,
          direction: 1,
        };
        introLoopStateRef.current = initial;
        const idx = frameSrcIndex(initial, anim.introFrames!, anim.loopFrames!);
        setFrameSrc(anim.frameSrcs[idx] ?? anim.frameSrcs[0]);
      } else {
        setFrameSrc(anim.frameSrcs[0]);
      }
      rafIdRef.current = requestAnimationFrame(tick);
      return;
    }

    const elapsed = timestamp - lastFrameTimeRef.current;

    if (elapsed >= msPerFrame) {
      lastFrameTimeRef.current = timestamp - (elapsed % msPerFrame);

      if (hasIntroLoop) {
        const introFrames = anim.introFrames!;
        const loopFrames = anim.loopFrames!;
        const prev =
          introLoopStateRef.current ?? {
            phase: "intro" as const,
            introIndex: 0,
            loopIndex: 0,
            direction: 1 as 1 | -1,
          };
        const next = advanceIntroLoop(prev, introFrames, loopFrames, anim.loopType);
        introLoopStateRef.current = next;
        const idx = frameSrcIndex(next, introFrames, loopFrames);
        setFrameSrc(anim.frameSrcs[idx] ?? anim.frameSrcs[0]);
      } else {
        const totalFrames = anim.frameSrcs.length;
        let idx = frameIndexRef.current;

        if (anim.loopType === "ping-pong") {
          idx += directionRef.current;
          if (idx >= totalFrames) {
            // Hit the end — bounce back.
            directionRef.current = -1;
            idx = totalFrames - 2;
            // Single-frame animations can't bounce.
            if (idx < 0) idx = 0;
          } else if (idx < 0) {
            // Hit the start — bounce forward.
            directionRef.current = 1;
            idx = 1;
            if (idx >= totalFrames) idx = 0;
          }
        } else if (anim.loopType === "once") {
          idx = Math.min(idx + 1, totalFrames - 1);
        } else {
          // forward loop
          idx = (idx + 1) % totalFrames;
        }

        frameIndexRef.current = idx;
        setFrameSrc(anim.frameSrcs[idx]);
      }
    }

    rafIdRef.current = requestAnimationFrame(tick);
  }, []);

  useEffect(() => {
    // Reset timing state on mount / animation change.
    lastFrameTimeRef.current = -1;
    introLoopStateRef.current = null;
    rafIdRef.current = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(rafIdRef.current);
    };
  }, [animationName, tick]);

  return frameSrc;
}
