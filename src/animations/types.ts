export type LoopType = "forward" | "ping-pong" | "once";

export interface AnimationMeta {
  animation: string;
  frames: number;
  fps: number;
  loop_type: LoopType;
  source: string;
  /** Indices into the natural-sorted frame list to play exactly once on entry. */
  intro_frames?: number[];
  /** Indices into the natural-sorted frame list to loop after the intro. */
  loop_frames?: number[];
}

export interface Animation {
  name: string;
  fps: number;
  loopType: LoopType;
  /** Resolved image URLs in natural-sorted frame order. */
  frameSrcs: string[];
  /** Optional intro indices (into frameSrcs), played once on animation entry. */
  introFrames?: number[];
  /** Optional loop indices (into frameSrcs), looped after the intro per loopType. */
  loopFrames?: number[];
}

/** All available animation names. Must match the `animation` field in each
 *  ASSETS folder's animation_meta.json. */
export type AnimationName =
  | "idle"
  | "walk_left"
  | "walk_right"
  | "happy"
  | "drag_hover"
  | "eating"
  | "sniffing"
  | "satisfied"
  | "disappointed"
  | "rolo360"
  | "sleep";
