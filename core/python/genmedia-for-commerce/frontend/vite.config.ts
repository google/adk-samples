/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  publicDir: '../assets/frontend_assets/public',
  build: {
    rollupOptions: {
      output: {
        // Vite 8 / Rolldown requires a function (object form was removed).
        manualChunks: (id) => {
          if (!id.includes('node_modules')) return
          if (id.includes('/react-router-dom/') || id.includes('/react-dom/') || id.includes('/react/')) {
            return 'vendor'
          }
          if (id.includes('@mediapipe/face_mesh') || id.includes('@mediapipe/camera_utils')) {
            return 'mediapipe'
          }
        }
      }
    },
    commonjsOptions: {
      include: [/node_modules/],
      transformMixedEsModules: true
    }
  },
  optimizeDeps: {
    include: ['@mediapipe/face_mesh', '@mediapipe/camera_utils']
  },
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/api/shoes/spinning': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy, _options) => {
          proxy.on('proxyReq', (proxyReq, _req, _res) => {
            proxyReq.setTimeout(600000); // 10 minutes for video generation
          });
          proxy.on('proxyRes', (proxyRes, _req, _res) => {
            proxyRes.setTimeout(600000); // 10 minutes for video generation
          });
        }
      },
      '/api/glasses': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy, _options) => {
          proxy.on('proxyReq', (proxyReq, _req, _res) => {
            proxyReq.setTimeout(600000); // 10 minutes for video generation
          });
          proxy.on('proxyRes', (proxyRes, _req, _res) => {
            proxyRes.setTimeout(600000); // 10 minutes for video generation
          });
        }
      },
      '/api/clothes': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy, _options) => {
          proxy.on('proxyReq', (proxyReq, _req, _res) => {
            proxyReq.setTimeout(600000); // 10 minutes for VTO generation
          });
          proxy.on('proxyRes', (proxyRes, _req, _res) => {
            proxyRes.setTimeout(600000); // 10 minutes for VTO generation
          });
        }
      },
      '/api/spinning/interpolation/other': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy, _options) => {
          proxy.on('proxyReq', (proxyReq, _req, _res) => {
            proxyReq.setTimeout(600000); // 10 minutes for interpolation video generation
          });
          proxy.on('proxyRes', (proxyRes, _req, _res) => {
            proxyRes.setTimeout(600000); // 10 minutes for interpolation video generation
          });
        }
      },
      '/api/spinning/r2v/other': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy, _options) => {
          proxy.on('proxyReq', (proxyReq, _req, _res) => {
            proxyReq.setTimeout(600000); // 10 minutes for R2V video generation
          });
          proxy.on('proxyRes', (proxyRes, _req, _res) => {
            proxyRes.setTimeout(600000); // 10 minutes for R2V video generation
          });
        }
      },
      '/api/other/background-changer': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy, _options) => {
          proxy.on('proxyReq', (proxyReq, _req, _res) => {
            proxyReq.setTimeout(600000); // 10 minutes for background changing
          });
          proxy.on('proxyRes', (proxyRes, _req, _res) => {
            proxyRes.setTimeout(600000); // 10 minutes for background changing
          });
        }
      },
      '/api/product-enrichment/product-fitting': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy, _options) => {
          proxy.on('proxyReq', (proxyReq, _req, _res) => {
            proxyReq.setTimeout(600000); // 10 minutes for product fitting
          });
          proxy.on('proxyRes', (proxyRes, _req, _res) => {
            proxyRes.setTimeout(600000); // 10 minutes for product fitting
          });
        }
      },
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy, _options) => {
          proxy.on('proxyReq', (proxyReq, _req, _res) => {
            proxyReq.setTimeout(600000);
          });
          proxy.on('proxyRes', (proxyRes, _req, _res) => {
            proxyRes.setTimeout(600000);
          });
        }
      },
      '/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy, _options) => {
          proxy.on('proxyReq', (proxyReq, _req, _res) => {
            proxyReq.setTimeout(600000); // 10 minutes for chat
          });
          proxy.on('proxyRes', (proxyRes, _req, _res) => {
            proxyRes.setTimeout(600000); // 10 minutes for chat
          });
        }
      },
    },
  },
})
