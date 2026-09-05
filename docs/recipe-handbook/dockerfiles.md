<!-- word count: 903 (target 700, cap 1000) -->

# Dockerfile Standard

The rules a recipe's `Dockerfile` must follow before its image is
built and pushed to the public registry.

This is a higher bar than "it builds on my machine", and the reason
is the registry. A published image carries Google's name, is
world-readable, and stays pullable long after the run that produced
it. Every rule below exists because a real Dockerfile in this repo
broke it.

**Not every recipe needs this.** Most recipes are read, copied and
adapted — they never ship an image. Deployability is opt-in. If your
recipe has no `Dockerfile`, nothing here applies to you.

## Getting an image published

Publishing is an **allowlist**, not a scan. Adding a `Dockerfile`
does not start publishing it; an entry in
[`.github/policy.yml`](../../.github/policy.yml) under
`deployability.publish.images` does, and only a repository admin can
add one.

That is deliberate. `docker build` runs every `RUN` instruction it is
handed, and the result is published under Google's name — so which
Dockerfiles earn that is a decision someone makes, not a side effect
of adding a file.

Bring your Dockerfile up to the standard below, then ask a
maintainer to add the entry. Each entry declares four things: the
Dockerfile path, the build context, the published image name, and
whether the image serves HTTP.

## The nine rules

### 1. Reproducible installs

Install from a committed lockfile. `uv sync --frozen`, never bare
`uv sync`. `npm ci`, never `npm install`. A pinned
`requirements.txt` if you use neither.

Never delete a lockfile during the build. One recipe here runs
`rm -f uv.lock && uv sync`, which means two builds of the same
commit can produce different dependency sets — nobody can reproduce
or explain the resulting image.

### 2. Pinned base image

Pin by digest, or at minimum an immutable tag. A floating tag like
`ghcr.io/astral-sh/uv:python3.11-bookworm-slim` moves underneath you,
so the image you tested is not the image you shipped.

### 3. Pinned build tools

`RUN pip install --no-cache-dir uv==0.8.13`, with the version. This
repo already uses that exact pin; match it unless you have a reason.

### 4. Honor `$PORT`, with a default

Use `${PORT:-8080}`. Cloud Run injects `PORT` and expects your
container to listen on it; a hardcoded port breaks the moment it
differs. The default matters just as much — `--port $PORT` with no
fallback fails under a plain `docker run`.

### 5. Run as a non-root user

Create an unprivileged user and `USER` to it before the entrypoint.
Switch before installing dependencies if your runtime re-validates
the environment at startup, so the virtualenv is owned by the user
that runs it.

### 6. Explicit `COPY`, plus a `.dockerignore`

Name what goes in. Never `COPY . .` — it puts whatever happens to be
in the directory into a public layer, including files added later by
someone who never thought about your image.

Ship a `.dockerignore` covering at least `.venv/`, `.env`, `.git/`
and `gha-creds-*.json`. That last one is not hypothetical: the build
workflow authenticates to Google Cloud in the same job, and a
credential file landing in the build context is exactly the accident
this line prevents.

### 7. No configuration or secrets in `ENV`

`ENV` is for build metadata — `COMMIT_SHA`, `AGENT_VERSION` — and
nothing else.

Configuration arrives at runtime, through `docker run -e`,
`--env-file`, or Cloud Run environment variables. Baking it in does
not save anyone a step: `-e` works whether or not the Dockerfile
declares the variable. What it does do is freeze a value into a
world-readable layer and go stale — an `ENV MODEL_NAME=...` is wrong
for every operator the day that model is deprecated.

### 8. No unpinned network fetches at build time

Downloading model weights, remote installers or anything else
unpinned during the build makes it depend on a third-party host
staying up and serving the same bytes. It also means you may be
redistributing someone else's artifact under your image's licence.

### 9. Declared, not inferred

The build context is declared per image, not guessed from the
Dockerfile's location. Several Dockerfiles here live in a
subdirectory and `COPY` from the recipe root, so
`dirname(Dockerfile)` is the wrong answer often enough that guessing
it would generate bugs.

The same goes for whether the image serves HTTP. Not every
publishable image is a server — one here is a base image for
pipeline components and has no `CMD` at all — and probing it like a
web service would report a healthy image as broken.

## Checking your work

The declaration is validated on every pull request. To run it
yourself:

```bash
uv run --no-project --with pyyaml python3 \
  .github/scripts/publish_matrix.py --validate
```

That checks the declared paths resolve, the image names are unique
and registry-safe, and each Dockerfile sits inside its declared
context. It does not check the nine rules above — those are a
review conversation, not a linter.

## When a published image breaks

A build failure on `main` is classified before anyone is told about
it. Infrastructure trouble — a registry 5xx, a rate limit, a runner
out of disk — is never reported to a recipe owner, because a channel
that blames you for a registry outage is a channel you learn to
ignore.

A genuine failure comments on the pull request that caused it. If
the same image fails the next time it is built, a tracking issue is
opened. Until it builds, the image is not published.

---

← [Checklist](../recipe-checklist.md) · [Handbook](./README.md)
