# Design & Naming

← Back to the [Recipe Guidelines hub](README.md)

---

## 💡 Core Philosophy & Design Constraints

Recipes in the `adk-samples` repository must follow these core principles:

### 1. Clear Intent & Community Value
Recipes must demonstrate a specific agent use-case, integration, or design pattern that helps other ADK developers learn. Avoid arbitrary code dumps.

### 2. Lightweight Constraints (Size & Scope)
To keep the repository clean and easy to clone, we enforce size and file-count
limits that depend on both the recipe's top-level location (`core/` vs
`contrib/`) and whether `manifest.yaml` sets `large: true`.

| Tier | Location | Max files | Max size |
|------|----------|-----------|----------|
| Default | `core/<language>/…` | 500 | 50 MB |
| Large   | `core/<language>/…` | (defensive cap) | (defensive cap) |
| Default | `contrib/<language>/…` | 70 | 2 MB |
| Large   | `contrib/<language>/…` | 200 | 10 MB |

Auto-generated content and language tool caches (e.g. `uv.lock`,
`__pycache__/`, `.venv/`, `node_modules/`, `target/`, `build/`, `.gradle/`,
`vendor/`, `.next/`, …) never count toward either limit. The full exclusion
list — organised per language — lives in
[`.github/policy.yml`](../../.github/policy.yml) under `excluded_paths`, and
that file is also the single source of truth for every number in the table
above.

Both limits are enforced by the `python-validate-recipe` CI workflow — see
[Tooling & CI](tooling-and-ci.md#continuous-integration).

Files and directories NOT on the exclusion list count normally — including
`README.md`, `manifest.yaml`, `AGENTS.md`, and everything under `tests/`.
Plan accordingly.

---

## Naming Conventions

Recipe directory names must be:
*   **Lowercase letters and hyphens only** — no digits, underscores, or uppercase. CI enforces the exact pattern `^[a-z][a-z-]*$`.
*   **Start with a lowercase letter** (e.g., `rag-agent-search`).
*   **30 characters or less**.

> **Note:** Digits are not allowed, so version-style suffixes like `-v2` or
> `-1` will fail CI.

---

← Back to the [Recipe Guidelines hub](README.md)
