# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

# Build context is the repo root (so we can copy from web/ and web/server/).
# Build with:
#   docker build -f web/server/Dockerfile -t lha-web .

# ---- Stage 1: build the Vite static bundle -------------------------------
FROM node:22-slim AS web-builder

WORKDIR /web

COPY web/package.json web/package-lock.json* ./
RUN npm ci

COPY web/ ./
RUN npm run build

# ---- Stage 2: Express proxy runtime --------------------------------------
FROM node:22-slim

WORKDIR /app

COPY web/server/package.json web/server/package-lock.json* ./
RUN npm install --omit=dev --no-audit --no-fund

COPY web/server/server.js ./server.js
COPY --from=web-builder /web/dist ./web/dist

ARG COMMIT_SHA=""
ENV COMMIT_SHA=${COMMIT_SHA}
ENV NODE_ENV=production
ENV PORT=8080

EXPOSE 8080

CMD ["node", "server.js"]
