import type { Animation, AnimationMeta, AnimationName } from "./types";
import { compareNatural, hasDigitInStem } from "./sort-utils";

// ---------------------------------------------------------------------------
// Eager glob imports — Vite resolves these at build time.
// Each entry maps a relative path to its resolved URL string.
// ---------------------------------------------------------------------------
const allFrames: Record<string, string> = import.meta.glob(
  "../../ASSETS/**/*.png",
  { eager: true, import: "default", query: "?url" },
) as Record<string, string>;

const allMeta: Record<string, AnimationMeta> = import.meta.glob(
  "../../ASSETS/**/animation_meta.json",
  { eager: true, import: "default" },
) as Record<string, AnimationMeta>;

// Map from animation name → folder path prefix (e.g. "../ASSETS/IDLE/")
// and metadata, built once at module load.
interface FolderEntry {
  prefix: string;
  meta: AnimationMeta;
}

const foldersByName = new Map<string, FolderEntry>();

for (const [metaPath, meta] of Object.entries(allMeta)) {
  // metaPath looks like "../ASSETS/IDLE/animation_meta.json"
  const prefix = metaPath.replace("animation_meta.json", "");
  foldersByName.set(meta.animation, { prefix, meta });
}

// ---------------------------------------------------------------------------
// Build the registry — one Animation per folder.
// ---------------------------------------------------------------------------
const registry = new Map<AnimationName, Animation>();

for (const [name, { prefix, meta }] of foldersByName) {
  // Collect all PNG paths that belong to this folder.
  const framePaths = Object.keys(allFrames).filter(
    (p) => p.startsWith(prefix) && p.endsWith(".png"),
  );

  // Prefer files with digits in the stem (filters sprite sheets).
  const numbered = framePaths.filter(hasDigitInStem);
  const candidates = numbered.length > 0 ? numbered : framePaths;

  // Natural sort to get correct frame order regardless of naming scheme.
  candidates.sort(compareNatural);

  const frameSrcs = candidates.map((p) => allFrames[p]);

  if (frameSrcs.length === 0) {
    console.warn(`[animation-registry] No frames found for "${name}" in ${prefix}`);
    continue;
  }

  registry.set(name as AnimationName, {
    name,
    fps: meta.fps,
    loopType: meta.loop_type,
    frameSrcs,
    introFrames: meta.intro_frames,
    loopFrames: meta.loop_frames,
  });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function getAnimation(name: AnimationName): Animation | undefined {
  return registry.get(name);
}

export function getAllAnimationNames(): AnimationName[] {
  return [...registry.keys()];
}
