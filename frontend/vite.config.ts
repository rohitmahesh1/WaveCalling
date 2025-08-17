import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiTarget = env.VITE_API_URL || "http://localhost:8000";

  return {
    plugins: [react()],
    root: process.cwd(),
    base: "/ui/", // app is served from /ui by FastAPI
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "src")
      }
    },
    server: {
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": {
          target: apiTarget,
          changeOrigin: true
        },
        "/runs": {
          target: apiTarget,
          changeOrigin: true
        }
      }
    },
    build: {
      outDir: "../web", // FastAPI serves this at /ui
      emptyOutDir: true,
      sourcemap: true
    }
  };
});
