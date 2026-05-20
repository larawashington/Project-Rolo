import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

/**
 * Vitest configuration for Rolo's health inspection tests.
 *
 * jsdom is required because useAnimationLoop tests use renderHook from
 * @testing-library/react, which needs a browser-like DOM environment.
 */
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
  },
});
