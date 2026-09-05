<!-- word count: 998 (target 700, cap 1000) -->

# Dockerfile Standard

The rules a recipe's `Dockerfile` must follow before its image is
built and pushed to the public registry.

The bar is high because the registry is public: an image carries
Google's name and stays pullable long after the run that built it.
Almost every rule below is here because a real Dockerfile in this
repo broke it.

**Not every recipe needs this.** Most recipes are read and adapted,
never shipped as an image. Deployability is opt-in: if your recipe
has no `Dockerfile`, nothing here applies to you.

> **Publishing is not switched on yet.** Declared images are built on
> every pull request and on merge, which is what verifies them, but
> nothing is pushed: the push step waits on repository variables that
> arrive with the hosting project. Meet the standard now and your
> image ships when that lands — but do not tell anyone to pull it
> yet, because there is nothing there.

## Getting an image published

Publishing is an **allowlist**, not a scan. Adding a `Dockerfile`
does not start publishing it; an entry in
[`.github/policy.yml`](../../.github/policy.yml) under
`deployability.publish.images` does, and that file needs a
maintainer's approval to change.

That is deliberate. `docker build` runs every `RUN` instruction it is
handed and the result carries Google's name, so which Dockerfiles
earn that is a decision someone makes, not a side effect of adding a
file.

Bring your Dockerfile up to the standard below, then ask a
maintainer to add the entry. Each entry declares five things: the
recipe directory, the Dockerfile path, the build context, the
published image name, and whether the image serves HTTP.

**The images declared today do not all meet this yet.** All three
use a floating base tag and hardcode their port, and one runs as
root. They were allowlisted on reproducibility — the property that
makes an image explainable — and are being brought up to the rest.
New entries are held to the whole standard.

## The nine rules

### 1. Reproducible installs

Install from a committed lockfile. `uv sync --frozen`, never bare
`uv sync`. `npm ci`, never `npm install`. A pinned
`requirements.txt` if you use neither.

Never delete a lockfile during the build. One recipe here runs
`rm -f uv.lock && uv sync`, so two builds of the same commit can
produce different dependency sets — nobody can reproduce or explain
the resulting image.

### 2. Pinned base image

Pin by digest, or at minimum an immutable tag. A floating tag like
`python:3.12-slim` moves underneath you, so the image you tested is
not the image you shipped.

### 3. Pinned build tools

`RUN pip install --no-cache-dir uv==0.8.13`, with the version. This
repo already uses that exact pin; match it unless you have a reason.

### 4. Honor `$PORT`, with a default

Use `${PORT:-8080}`. Cloud Run injects `PORT` and expects your
container to listen on it; a hardcoded port breaks the moment it
differs. The default matters too — `--port $PORT` with no fallback
fails under a plain `docker run`.

### 5. Run as a non-root user

Create an unprivileged user and `USER` to it before the entrypoint.
Switch before installing dependencies if your runtime re-validates
the environment at startup, so the virtualenv is owned by whoever
runs it.

### 6. Explicit `COPY`, plus a `.dockerignore`

Name what goes in. Never `COPY . .` — it puts whatever happens to be
in the directory into a public layer, including files added later by
someone who never thought about your image.

Ship a `.dockerignore` covering at least `.venv/`, `.env`, `.git/`
and `gha-creds-*.json`. The build workflow authenticates to Google
Cloud in the same job. It is configured to keep that credential out
of the workspace entirely — but that is one setting away from not
being true, and a `COPY` that cannot reach the file does not depend
on the setting.

### 7. No configuration or secrets in `ENV`

`ENV` is for build metadata — `COMMIT_SHA`, `AGENT_VERSION` — and
nothing else.

Configuration arrives at runtime, via `docker run -e`, `--env-file`
or Cloud Run. Baking it in saves nobody a step — `-e` works whether
or not the Dockerfile declares the variable — and it freezes a value
into a world-readable layer that goes stale: an
`ENV MODEL_NAME=...` is wrong for every operator the day that model
is deprecated.

### 8. No unpinned network fetches at build time

Fetching model weights or remote installers during the build ties it
to a third-party host serving the same bytes tomorrow. It can also
mean redistributing someone else's artifact under your license.

### 9. Declared, not inferred

The build context is declared per image, not guessed from the
Dockerfile's location. Several Dockerfiles here sit in a
subdirectory and `COPY` from the recipe root, so
`dirname(Dockerfile)` is wrong often enough that guessing would
generate bugs.

Whether the image serves HTTP is declared too. Not every publishable
image is a server — one here is a pipeline base image with no `CMD`
— and probing it like a web service would report it broken.

## Checking your work

The declaration is validated on every pull request. To run it
yourself:

```bash
uv run --no-project --with pyyaml python3 \
  .github/scripts/publish_matrix.py --validate
```

That checks the declared paths resolve, the image names are unique
and registry-safe, and each Dockerfile sits inside its declared
context. It does not check the nine rules above — those are a review
conversation, not a linter.

## When a build breaks

A build failure on `main` is classified before anyone is told.
Infrastructure trouble — a registry 5xx, a rate limit, a runner out
of disk — is never reported to an owner, because a channel that
blames you for a registry outage is one you learn to ignore.

A genuine failure comments on the pull request that caused it. If
the same image fails the next time it is built, a tracking issue is
opened. Until it builds, the image is not published.

---

← [Checklist](../recipe-checklist.md) · [Handbook](./README.md)
