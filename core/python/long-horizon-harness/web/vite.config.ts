// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { tanstackRouter } from "@tanstack/router-plugin/vite";

export default defineConfig({
  plugins: [
    tanstackRouter({
      target: "react",
      routesDirectory: "src/routes",
      generatedRouteTree: "src/routeTree.gen.ts",
      autoCodeSplitting: true,
    }),
    react(),
  ],
  // Mirror the @/ → web/ alias from tsconfig so shared lib/components resolve.
  resolve: { alias: { "@": new URL(".", import.meta.url).pathname } },
  build: { outDir: "dist" },
  server: {
    // NOTE: /a2a must stream SSE unbuffered. Vite's dev proxy (http-proxy)
    // does not gzip responses, so Server-Sent Events pass through unbuffered.
    proxy: Object.fromEntries(
      ["/lha", "/a2a", "/feedback", "/.well-known"].map((p) => [
        p,
        {
          target: process.env.NEXT_PUBLIC_LHA_URL ?? "http://127.0.0.1:8001",
          changeOrigin: true,
          ws: true,
        },
      ]),
    ),
  },
});
