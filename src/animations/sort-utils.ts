/**
 * Natural sort utilities extracted from registry.ts so they can be unit-tested
 * independently of import.meta.glob (which is a Vite build-time feature and
 * not available in vitest's Node/jsdom environment).
 *
 * These are the pure functions that handle frame ordering for Rolo's sprites.
 */

/**
 * Natural sort key — splits a path into alternating text / numeric segments so
 * that "frame2.png" sorts before "frame10.png". Mirrors the Python reference
 * `_natural_sort_key()` in `animator.py`.
 */
export function naturalSortKey(path: string): (string | number)[] {
  const basename = path.split("/").pop() ?? path;
  return basename.split(/(\d+)/).map((part) =>
    /^\d+$/.test(part) ? parseInt(part, 10) : part.toLowerCase(),
  );
}

/**
 * Comparator that imposes natural (human) sort order on file paths.
 * Returns negative, zero, or positive — the standard JS comparator contract.
 */
export function compareNatural(a: string, b: string): number {
  const ka = naturalSortKey(a);
  const kb = naturalSortKey(b);
  const len = Math.max(ka.length, kb.length);
  for (let i = 0; i < len; i++) {
    const va = ka[i] ?? "";
    const vb = kb[i] ?? "";
    if (typeof va === "number" && typeof vb === "number") {
      if (va !== vb) return va - vb;
    } else {
      const sa = String(va);
      const sb = String(vb);
      if (sa !== sb) return sa < sb ? -1 : 1;
    }
  }
  return 0;
}

/**
 * Returns true if the filename stem (without extension) contains at least one
 * digit. Used to filter out stray sprite sheets (e.g. "WalkLeft.png") that
 * live alongside numbered frame files.
 */
export function hasDigitInStem(path: string): boolean {
  const basename = path.split("/").pop() ?? path;
  const stem = basename.replace(/\.[^.]+$/, "");
  return /\d/.test(stem);
}
