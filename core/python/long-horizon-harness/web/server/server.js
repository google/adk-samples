// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import { createProxyMiddleware } from "http-proxy-middleware";
import { GoogleAuth } from "google-auth-library";
import { createRemoteJWKSet, jwtVerify } from "jose";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const BACKEND_URL = required("LHA_BACKEND_URL");
const IAP_AUDIENCE = required("LHA_IAP_AUDIENCE");
const PORT = parseInt(process.env.PORT || "8080", 10);
const STATIC_DIR = path.join(__dirname, "web", "dist");
const IAP_JWT_HEADER = "x-goog-iap-jwt-assertion";

const IAP_JWKS = createRemoteJWKSet(
  new URL("https://www.gstatic.com/iap/verify/public_key-jwk"),
);

// BACKEND_URL must be the bare Cloud Run service origin (e.g.
// https://lha-...run.app). Cloud Run rejects ID tokens whose `aud` includes
// a path. The Terraform sets this from `google_cloud_run_v2_service.app.uri`,
// which is origin-only — do not append a path here.
const auth = new GoogleAuth();
const backendClient = await auth.getIdTokenClient(BACKEND_URL);

const PROXY_PREFIXES = [
  "/lha",
  "/a2a",
  "/feedback",
  "/.well-known",
];

function required(name) {
  const value = process.env[name];
  if (!value) {
    console.error(`missing required env var: ${name}`);
    process.exit(1);
  }
  return value;
}

function isProxiedPath(url) {
  return PROXY_PREFIXES.some(
    (p) => url === p || url.startsWith(`${p}/`) || url.startsWith(`${p}?`),
  );
}

async function verifyIap(jwt) {
  const { payload } = await jwtVerify(jwt, IAP_JWKS, {
    issuer: "https://cloud.google.com/iap",
    audience: IAP_AUDIENCE,
  });
  if (!payload.email || typeof payload.email !== "string") {
    throw new Error("IAP JWT missing email claim");
  }
  return payload.email;
}

const app = express();
app.disable("x-powered-by");

// IAP gates the front door; we re-verify the JWT here as defense-in-depth (so a
// bad token is rejected before proxying) and forward it to the backend under a
// custom header, where iap mode verifies it again and derives the user from its
// email claim — no plaintext identity is trusted.
async function requireIap(req, res, next) {
  if (!isProxiedPath(req.url)) return next();
  const jwt = req.header(IAP_JWT_HEADER);
  if (!jwt) return res.status(401).send("missing IAP JWT");
  try {
    await verifyIap(jwt);
    res.locals.iapJwt = jwt;
    next();
  } catch (err) {
    console.warn(`IAP JWT verify failed: ${err.message}`);
    res.status(401).send("invalid IAP JWT");
  }
}

// ID tokens are valid ~1h. Refresh proactively in the background so per-request
// middleware stays synchronous.
let cachedBackendToken = null;
async function refreshBackendToken() {
  try {
    cachedBackendToken = await backendClient.idTokenProvider.fetchIdToken(
      BACKEND_URL,
    );
  } catch (err) {
    console.error("backend ID token refresh failed:", err);
  }
}
await refreshBackendToken();
setInterval(refreshBackendToken, 30 * 60 * 1000).unref();

const proxy = createProxyMiddleware({
  target: BACKEND_URL,
  changeOrigin: true,
  xfwd: true,
  proxyTimeout: 60_000,
  // pathFilter at root avoids Express's path-mount stripping. Mounting via
  // `app.use(prefix, asyncMid, proxy)` silently breaks: http-proxy-middleware
  // v3 is not invoked after any async middleware on a mounted prefix.
  pathFilter: (pathname) => isProxiedPath(pathname),
  on: {
    proxyReq: (proxyReq, req, res) => {
      // IAP injects x-serverless-authorization (a bearer token whose audience
      // is lha-web, the frontend), x-goog-iap-jwt-assertion, and the
      // x-goog-authenticated-user-* headers. Cloud Run on the backend
      // prefers x-serverless-authorization over Authorization, so leaving it
      // in place causes a 401 ("access token could not be verified") because
      // its audience is lha-web, not the backend service.
      proxyReq.removeHeader("x-serverless-authorization");
      // Google strips X-Goog-* IAP headers inbound to a non-IAP backend, so the
      // native x-goog-iap-jwt-assertion can't be forwarded. We send the JWT under
      // a custom header (X-LHA-IAP-Assertion) below instead.
      proxyReq.removeHeader("x-goog-authenticated-user-email");
      proxyReq.removeHeader("x-goog-authenticated-user-id");
      proxyReq.removeHeader("cookie");
      // LHA_AUTH_MODE=trusted_header reads this verbatim as the identity, so a
      // client-supplied value would be impersonation. Identity is carried only
      // by the verified assertion set below.
      proxyReq.removeHeader("x-lha-user-id");
      // The browser Origin (lha-web) never matches the backend host, so ADK's
      // same-origin check 403s POST /a2a streams. Strip it on every proxied
      // route; this proxy is the trusted IAP boundary (absent Origin is allowed
      // server-to-server), so the CSRF protection the check incidentally gave
      // all POSTs is already covered by IAP upstream.
      proxyReq.removeHeader("origin");
      if (cachedBackendToken) {
        proxyReq.setHeader("Authorization", `Bearer ${cachedBackendToken}`);
      }
      // The backend (iap mode) verifies this forwarded IAP JWT and derives the
      // user from its email claim.
      if (res.locals.iapJwt) {
        proxyReq.setHeader("X-LHA-IAP-Assertion", res.locals.iapJwt);
      }
    },
    error: (err, req, res) => {
      // Log the failing method+path so a 502/504 can be triaged (which route,
      // which failure mode). ETIMEDOUT / ESOCKETTIMEDOUT => upstream too slow
      // (504, Gateway Timeout); everything else => upstream unreachable (502).
      const where = req ? `${req.method} ${req.url}` : "unknown";
      const code = err && err.code ? err.code : "ERR";
      console.error(`proxy error [${code}] on ${where}:`, err && err.message);
      if (res && !res.headersSent) {
        const timedOut = code === "ETIMEDOUT" || code === "ESOCKETTIMEDOUT";
        res
          .status(timedOut ? 504 : 502)
          .send(timedOut ? "upstream timeout" : "upstream error");
      }
    },
  },
});

app.use(requireIap);
app.use(proxy);

// Vite SPA bundle. express.static serves the hashed assets and index.html;
// the catch-all below falls back to index.html for client-side routes.
// Anything served from here is gated by IAP itself at the perimeter; we don't
// require the JWT for static assets.
app.use(express.static(STATIC_DIR, { extensions: ["html"] }));
app.use((_req, res) => {
  res.sendFile(path.join(STATIC_DIR, "index.html"));
});

app.listen(PORT, () => {
  console.log(`lha-web listening on :${PORT} → ${BACKEND_URL}`);
});
