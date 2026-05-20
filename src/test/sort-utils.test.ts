/**
 * Health inspection: natural sort utilities for Rolo's animation frame ordering.
 *
 * These tests verify that sprite frames are always loaded in the correct order,
 * which is critical to Rolo's visual coherence. An out-of-order frame sequence
 * would cause Rolo to twitch erratically — a deeply concerning symptom.
 *
 * Test data uses real filename patterns from the ASSETS directory:
 *   - idle0..3.png          (IDLE/)
 *   - rolo_eat00..12.png   (Eating/) — zero-padded two-digit index
 *   - walk_left0..3.png         (WalkLeft/) — underscore-separated prefix + index
 *   - Rolo_Sniff0..6.png   (DragHover/)
 *   - Sniffing0..5.png + sniffing6.png  (Sniffing/) — mixed case
 */

import { describe, it, expect } from "vitest";
import {
  naturalSortKey,
  compareNatural,
  hasDigitInStem,
} from "../animations/sort-utils";

// ---------------------------------------------------------------------------
// naturalSortKey
// ---------------------------------------------------------------------------

describe("naturalSortKey", () => {
  it("splits a simple numbered filename into text and numeric segments", () => {
    const key = naturalSortKey("idle0.png");
    // Expected: ["idle", 0, ".png"]
    expect(key).toEqual(["idle", 0, ".png"]);
  });

  it("handles zero-padded two-digit index correctly — numeric not string", () => {
    // "rolo_eat00.png" → the '00' part should become the number 0
    const key = naturalSortKey("rolo_eat00.png");
    expect(typeof key[1]).toBe("number");
    expect(key[1]).toBe(0);
  });

  it("produces lowercase text segments for case-insensitive comparison", () => {
    const key = naturalSortKey("Rolo_Sniff0.png");
    expect(key[0]).toBe("rolo_sniff");
  });

  it("handles complex filename with multiple embedded digit groups", () => {
    // image__1_-removebg-preview.png → segments: ["image__", 1, "_-removebg-preview.png"]
    // (general edge case — exercises digit extraction from messy stems)
    const key = naturalSortKey("image__1_-removebg-preview.png");
    expect(key[1]).toBe(1);
  });

  it("extracts only the basename from a full path", () => {
    const key = naturalSortKey("../../ASSETS/IDLE/idle2.png");
    // Should behave the same as naturalSortKey("idle2.png")
    const keyDirect = naturalSortKey("idle2.png");
    expect(key).toEqual(keyDirect);
  });

  it("handles a filename with no digits — produces only text segments", () => {
    const key = naturalSortKey("SpriteSheet.png");
    // No numeric segments — every element is a string
    key.forEach((segment) => expect(typeof segment).toBe("string"));
  });

  it("mixed-case filename with digit: Sniffing6.png lowercases text", () => {
    const key = naturalSortKey("sniffing6.png");
    expect(key[0]).toBe("sniffing");
    expect(key[1]).toBe(6);
  });
});

// ---------------------------------------------------------------------------
// compareNatural — ordering verification
// ---------------------------------------------------------------------------

describe("compareNatural", () => {
  // --- IDLE sprite frames ---

  it("sorts idle0 before idle1", () => {
    expect(compareNatural("idle0.png", "idle1.png")).toBeLessThan(0);
  });

  it("sorts idle3 after idle2", () => {
    expect(compareNatural("idle3.png", "idle2.png")).toBeGreaterThan(0);
  });

  it("sorts all IDLE frames in correct numeric order", () => {
    const files = [
      "idle3.png",
      "idle1.png",
      "idle0.png",
      "idle2.png",
    ];
    files.sort(compareNatural);
    expect(files).toEqual([
      "idle0.png",
      "idle1.png",
      "idle2.png",
      "idle3.png",
    ]);
  });

  // --- Eating frames (zero-padded two-digit) ---

  it("sorts rolo_eat00 through rolo_eat09 before rolo_eat10", () => {
    // The key difference from lexicographic: "09" < "10" numerically but
    // lexicographically "10" < "09" is FALSE — both work here, but the
    // real edge case is "rolo_eat9" vs "rolo_eat10" which lexicographic gets wrong.
    const files = ["rolo_eat10.png", "rolo_eat09.png", "rolo_eat02.png"];
    files.sort(compareNatural);
    expect(files).toEqual(["rolo_eat02.png", "rolo_eat09.png", "rolo_eat10.png"]);
  });

  it("sorts all 13 eating frames in correct numeric order", () => {
    const frames = Array.from({ length: 13 }, (_, i) =>
      `rolo_eat${String(i).padStart(2, "0")}.png`
    );
    // Shuffle deliberately
    const shuffled = [...frames].reverse();
    shuffled.sort(compareNatural);
    expect(shuffled).toEqual(frames);
  });

  // --- WalkLeft frames ---

  it("sorts walk_left0 before walk_left3", () => {
    expect(compareNatural("walk_left0.png", "walk_left3.png")).toBeLessThan(0);
  });

  it("sorts all 4 WalkLeft frames correctly", () => {
    const files = [
      "walk_left3.png",
      "walk_left1.png",
      "walk_left0.png",
      "walk_left2.png",
    ];
    files.sort(compareNatural);
    expect(files).toEqual([
      "walk_left0.png",
      "walk_left1.png",
      "walk_left2.png",
      "walk_left3.png",
    ]);
  });

  // --- WalkRight frames ---

  it("sorts all 4 WalkRight frames correctly", () => {
    const files = [
      "walk_right2.png",
      "walk_right0.png",
      "walk_right3.png",
      "walk_right1.png",
    ];
    files.sort(compareNatural);
    expect(files).toEqual([
      "walk_right0.png",
      "walk_right1.png",
      "walk_right2.png",
      "walk_right3.png",
    ]);
  });

  // --- DragHover Rolo_Sniff frames ---

  it("sorts Rolo_Sniff0 through Rolo_Sniff6 correctly (mixed stem case)", () => {
    const files = [
      "Rolo_Sniff6.png",
      "Rolo_Sniff3.png",
      "Rolo_Sniff0.png",
      "Rolo_Sniff5.png",
      "Rolo_Sniff1.png",
      "Rolo_Sniff4.png",
      "Rolo_Sniff2.png",
    ];
    files.sort(compareNatural);
    expect(files).toEqual([
      "Rolo_Sniff0.png",
      "Rolo_Sniff1.png",
      "Rolo_Sniff2.png",
      "Rolo_Sniff3.png",
      "Rolo_Sniff4.png",
      "Rolo_Sniff5.png",
      "Rolo_Sniff6.png",
    ]);
  });

  // --- Sniffing frames (mixed case: Sniffing0..5 + sniffing6) ---

  it("treats Sniffing0 and sniffing0 as equal (case-insensitive)", () => {
    expect(compareNatural("Sniffing0.png", "sniffing0.png")).toBe(0);
  });

  it("sorts mixed-case Sniffing0..5 + sniffing6 in correct frame order", () => {
    const files = [
      "sniffing6.png",
      "Sniffing3.png",
      "Sniffing0.png",
      "Sniffing5.png",
      "Sniffing1.png",
      "Sniffing4.png",
      "Sniffing2.png",
    ];
    files.sort(compareNatural);
    expect(files).toEqual([
      "Sniffing0.png",
      "Sniffing1.png",
      "Sniffing2.png",
      "Sniffing3.png",
      "Sniffing4.png",
      "Sniffing5.png",
      "sniffing6.png",
    ]);
  });

  // --- Edge cases ---

  it("returns 0 for two identical filenames", () => {
    expect(compareNatural("idle0.png", "idle0.png")).toBe(0);
  });

  it("correctly orders numeric > lexicographic (the classic natural sort proof)", () => {
    // Lexicographic: "frame10" < "frame2" (because "1" < "2")
    // Natural sort: "frame2" < "frame10" (because 2 < 10)
    const files = ["frame10.png", "frame2.png", "frame1.png"];
    files.sort(compareNatural);
    expect(files).toEqual(["frame1.png", "frame2.png", "frame10.png"]);
  });

  it("sorts with full paths — only the basename matters", () => {
    const a = "../../ASSETS/IDLE/idle0.png";
    const b = "../../ASSETS/IDLE/idle3.png";
    expect(compareNatural(a, b)).toBeLessThan(0);
    expect(compareNatural(b, a)).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// hasDigitInStem
// ---------------------------------------------------------------------------

describe("hasDigitInStem", () => {
  // Should accept (return true)

  it("accepts idle0.png — digit in stem", () => {
    expect(hasDigitInStem("idle0.png")).toBe(true);
  });

  it("accepts rolo_eat00.png — zero-padded digit", () => {
    expect(hasDigitInStem("rolo_eat00.png")).toBe(true);
  });

  it("accepts Rolo_Sniff0.png — digit at end of mixed-case stem", () => {
    expect(hasDigitInStem("Rolo_Sniff0.png")).toBe(true);
  });

  it("accepts Sniffing0.png — digit at end of stem", () => {
    expect(hasDigitInStem("Sniffing0.png")).toBe(true);
  });

  it("accepts sniffing6.png — lowercase with digit", () => {
    expect(hasDigitInStem("sniffing6.png")).toBe(true);
  });

  it("accepts walk_left0.png — digit in stem", () => {
    expect(hasDigitInStem("walk_left0.png")).toBe(true);
  });

  it("accepts file with digit in middle of stem", () => {
    expect(hasDigitInStem("frame_2_loop.png")).toBe(true);
  });

  // Should reject (return false)

  it("rejects SpriteSheet.png — no digit in stem", () => {
    expect(hasDigitInStem("SpriteSheet.png")).toBe(false);
  });

  it("rejects sprite.png — no digit in stem", () => {
    expect(hasDigitInStem("sprite.png")).toBe(false);
  });

  it("rejects Rolo.png — no digit in stem", () => {
    expect(hasDigitInStem("Rolo.png")).toBe(false);
  });

  it("rejects animation_meta.json — no digit, different extension", () => {
    expect(hasDigitInStem("animation_meta.json")).toBe(false);
  });

  it("digit in the extension does not count — only stem matters", () => {
    // "sprite.1png" — hypothetical; digit is in extension, not stem
    // This is an edge case but shows the function is stem-only
    expect(hasDigitInStem("sprite.1png")).toBe(false);
  });

  it("works with full path — only basename stem is checked", () => {
    expect(hasDigitInStem("../../ASSETS/WalkLeft/walk_left0.png")).toBe(true);
    expect(hasDigitInStem("../../ASSETS/IDLE/SpriteSheet.png")).toBe(false);
  });
});
