# `manifest.yaml`

← Back to the [Recipe Guidelines hub](README.md)

---

Every recipe must contain a `manifest.yaml` at its root. This metadata file is automatically validated against the repository's schema [manifest-schema.json](../../.github/schemas/manifest-schema.json).

## Required Fields
*   **`type`**: Either `standalone` (runnable) or `module` (importable sub-agent).
*   **`status`**: Active maintenance status (e.g., `"active"`).
*   **`language`**: Primary language (`python`, `java`, `go`, `kotlin`, `typescript`).
*   **`description`**: Summary of the recipe (minimum 10 characters).
*   **`ownership`**: Contains ownership details:
    *   `team` (Required): The name of the team owning the recipe (do **not** leave as placeholder).
    *   `poc` (Required): The GitHub ID of the primary point of contact (do **not** leave as placeholder).

## Optional Fields
*   **`deployable`**: Can it be deployed with a single command? (default `false`).
*   **`ownership.contributors`**: List of other contributing developers' GitHub IDs.
*   **`architecture.agent`**: Architecture type: `single` or `multi`.
*   **`architecture.stateful`**: `true` if it maintains persistent state/memory across sessions, or performs transactional writes to external systems.
*   **`architecture.datasources`**: List of data sources:
    *   `hardcoded`: Data or mock responses hardcoded directly within the source code.
    *   `local`: Static data loaded from files bundled with the recipe (e.g., local PDF, JSON, or CSV files).
    *   `external`: Live data retrieved at runtime from external systems (e.g., APIs, live databases, or third-party web services).
*   **`dependencies`**: Lists of imported `libraries` (e.g. `pandas`) and runtime cloud `services` (e.g. `BigQuery`).
*   **`tags`**: Category tags (e.g., `rag`, `gemini`).

## Example `manifest.yaml`
```yaml
type: "standalone"
deployable: false
status: "active"
language: "python"
description: "A premium search assistant utilizing RAG pattern for document querying."
architecture:
  agent: "single"
  stateful: false
  datasources:
    - "local"
ownership:
  team: "Adk DevRel Team"
  poc: "johndoe"
  contributors:
    - "alice"
```

> **Tip:** The `generate-manifest` skill can inspect your recipe and produce a
> populated `manifest.yaml` for you — see [Developer Agent Skills](tooling-and-ci.md#developer-agent-skills-agentsskills).

---

← Back to the [Recipe Guidelines hub](README.md)
