---
name: generate-manifest
description: Scan an ADK recipe directory and generate a manifest.yaml for it based on the schema at .github/schemas/manifest-schema.json. Use when the user wants to create or generate a manifest.yaml for a recipe under core/ or contrib/.
---

# Generate Manifest

Scan a recipe directory and produce a valid `manifest.yaml` for it, then validate it against the schema.

## Process

### 1. Identify the target recipe

The user will provide a path to a recipe directory (e.g. `core/rag-agent-search`). If they don't, ask for it before proceeding.

### 2. Read the schema

Read `.github/schemas/manifest-schema.json` from the repo root to understand the current required and optional fields, allowed enum values, and constraints. Do not rely on memory — always read the live schema file so changes to it are automatically picked up.

### 3. Scan the recipe directory and infer only what the code proves

Read the following files if they exist:

- `pyproject.toml` / `requirements.txt` — primary language (for `language` field)
- `app/agent.py` or equivalent entry point — single vs multi-agent, agent count
- `Makefile` — deploy targets (determines `deployable`)
- `app/` source files — stateful writes, datasource patterns
- `AGENTS.md` — architecture notes if present

**Field-by-field rules — read carefully before filling any field:**

| Field | Rule |
|---|---|
| `type` | Infer from code: `standalone` if it has its own runnable entry point; `module` if it is only importable. |
| `deployable` | `true` only if a single `make` target or script deploys everything with no manual steps. Default `false`. |
| `status` | Always `"active"` unless there is explicit evidence of abandonment. |
| `language` | Read from file extensions or `pyproject.toml`. Never guess. |
| `description` | Read `README.md` and `AGENTS.md` only (author-written intent, not code). Write a draft of at most 15 words summarising what the recipe does. Append the comment `# TODO: review and expand this draft description`. If neither file exists or the intent is unclear, fall back to `"DESCRIPTION"` with the same TODO comment. |
| `architecture.agent` | Infer from code only: count `Agent(` or equivalent constructor calls. `single` or `multi`. Omit the whole `architecture` block if uncertain. |
| `architecture.stateful` | Infer from code: `true` only if the recipe writes persistently to an external system (DB, vector store, GCS) during normal operation — not just one-time setup. |
| `architecture.datasources` | Infer from code: `hardcoded` = data literals in source; `local` = files bundled in repo; `external` = live systems queried at runtime. Can be multiple. |
| `dependencies` | Include the block but comment it out entirely. Do NOT infer library or service names — GCP product names change and guesses will be wrong. Use the commented-out sample shown in the template. |
| `ownership.team` | Always use the placeholder `"YOUR TEAM NAME"`. Never invent or infer a team name. |
| `ownership.poc` | Always use the placeholder `"your-github-id"`. Never invent or infer a GitHub ID from email addresses, file authors, or any other source. |
| `tags` | Include but comment out entirely, with a sample entry showing the expected style. Do NOT generate real tags — tag choices are the author's call. |

### 4. Make judgment calls explicitly

Before writing the manifest, briefly state what you found for each inferred field (`type`, `deployable`, `architecture.*`) and cite the specific file/line that supports your conclusion. This lets the user catch errors before they land in the file.

### 5. Write the manifest

Write `manifest.yaml` to the root of the recipe directory. Follow this format exactly — include inline comments for every field documenting the allowed values, matching the style of the reference manifest in `tools/` or `core/`:

```yaml
# REQUIRED — Recipe type.
#   standalone : complete, runnable recipe with its own entry point
#   module     : importable sub-agent meant to be orchestrated by another workflow
type: "..."

# REQUIRED — One-click/one-command deployable with no manual steps?
deployable: false

# REQUIRED — Maintenance status.
#   active | inactive
status: "active"

# REQUIRED — Primary programming language.
#   python | java | go | kotlin | typescript
language: "..."

# REQUIRED — Short description of what this recipe does and the value it provides.
description: "..."  # TODO: review and expand this draft description

# OPTIONAL — Agent architecture details. Omit the whole block if unknown.
architecture:
  # single | multi
  agent: "..."

  # true  : recipe writes persistently to an external system during normal operation
  # false : read-only at runtime
  stateful: false

  # One or more of: hardcoded | local | external
  #   hardcoded : data literals embedded in source code
  #   local     : files bundled in the repo
  #   external  : live system queried at runtime (DB, API, etc.)
  datasources:
    - "..."

# OPTIONAL — Library and service dependencies.
# dependencies:
#   libraries:
#     - "ADK"
#     - "pandas"     # replace with actual libraries used
#   services:
#     - "GCP Project"
#     - "Cloud Run"  # replace with actual GCP/external services used

# REQUIRED — Ownership. Both sub-fields are required.
ownership:
  team: "YOUR TEAM NAME"  # TODO: replace with your team name
  poc: "your-github-id"   # TODO: replace with the GitHub ID of the primary contact

# OPTIONAL — Classification tags (technology names, patterns, use-case keywords).
# tags:
#   - "rag"
#   - "gemini"
```

Omit `architecture` entirely if none of its sub-fields can be determined from the code.

### 6. Validate

After writing the manifest, run the validation tool from the `tools/` directory:

```bash
cd tools && uv run --active validate manifest <recipe-path>
```

If validation fails, fix the errors and re-validate until it passes. Report the final `[PASS]` output to the user. Do not change or modify any other files in the repository. Also do not commit the changes.

### 7. Report back

Tell the user:
1. What was inferred from the code and which file/line supports each inference.
2. Which fields need human input: `description` (draft generated — must be reviewed and expanded), `ownership.team`, `ownership.poc` (placeholders), and `dependencies`/`tags` (commented out, ready to uncomment and fill in).
