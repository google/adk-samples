# Design & Naming

← Back to the [Recipe Guidelines hub](README.md)

---

## 💡 Core Philosophy & Design Constraints

Recipes in the `adk-samples` repository must follow these core principles:

### 1. Clear Intent & Community Value
Recipes must demonstrate a specific agent use-case, integration, or design pattern that helps other ADK developers learn. Avoid arbitrary code dumps.

### 2. Lightweight Constraints (Size & Scope)
To keep the repository clean and easy to clone, we enforce the following boundaries:
*   **Size Limit**: Under **1MB** total directory size, **excluding `uv.lock`** (no large assets or datasets).
*   **File Limit**: Maximum of **50 main files** (excludes `README.md`, `manifest.yaml`, `AGENTS.md`, and the `tests/` directory).

Both limits are enforced by the `validate-python-recipe` CI workflow — see [Tooling & CI](tooling-and-ci.md#continuous-integration).

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
