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

### 3. Scan the recipe directory thoroughly

Explore the recipe directory and gather the information needed to fill every schema field. Read the following files if they exist:

- `README.md` — description, services, deployment story
- `AGENTS.md` — agent architecture, tools, entry point
- `pyproject.toml` / `requirements.txt` — library dependencies, Python version
- `Makefile` — how the recipe is run and deployed
- `infra/` or `terraform/` — GCP services provisioned, deployment automation
- `app/agent.py` or equivalent entry point — single vs multi-agent, tools used
- `data_ingestion/` or equivalent — ingestion pipeline, data sources
- `.env.example` — environment variables, external services required
- Any `cloudbuild.yaml`, `deploy/`, or CI/CD config

Answer the following questions from what you find:

| Field | Question to answer |
|---|---|
| `type` | Does the recipe have its own entry point and can run independently (`standalone`), or is it a sub-agent meant to be imported by another workflow (`module`)? |
| `deployable` | Can it be deployed with a single command or click, with no manual steps required? |
| `status` | Is it actively maintained? Default to `active` unless there is evidence otherwise. |
| `language` | What is the primary programming language? |
| `description` | What does this recipe do and what value does it provide? Write 2-3 sentences. |
| `architecture.agent` | Is there one agent or multiple agents? |
| `architecture.stateful` | Does it write persistently to external systems (DB, vector store, etc.) or maintain memory across sessions? |
| `architecture.datasources` | Does it use hardcoded data, local files bundled in the repo, and/or live external systems at runtime? Select all that apply. |
| `architecture.rag` | Does it use a RAG pipeline (retrieval + augmentation + generation)? |
| `dependencies.libraries` | What libraries does it depend on? Always include `ADK`. |
| `dependencies.services` | What GCP or external services does it require? Always include `GCP Project`. |
| `team` | Always use the placeholder values `YOUR TEAM NAME` / `team@email.com`. |
| `poc` | Always use the placeholder values `POINT OF CONTACT NAME` / `poc@email.com`. |
| `tags` | What free-form labels best describe this recipe? Include technology names, patterns, and use case. |

### 4. Make judgment calls explicitly

Some fields require interpretation — document your reasoning briefly before writing the manifest so the user can correct you if needed:

- **`deployable`**: Only `true` if there is a one-click/one-command deploy path with no manual config steps required.
- **`stateful`**: `true` if the recipe writes to any external system (vector store, database, GCS) as part of its normal operation — not just one-time setup.
- **`datasources`**: Can be multiple values. `hardcoded` = data embedded in source code. `local` = files bundled in the repo. `external` = live systems queried at runtime.

### 5. Write the manifest

Write `manifest.yaml` to the root of the recipe directory. Follow this format exactly — include inline comments for every field documenting the allowed values, matching the style of the reference manifest in `tools/` or `core/`:

```yaml
type: "..."         # Options: [standalone | module]
deployable: false   # Options: [true | false]. Default: false
status: "active"    # Options: [active | inactive]
language: "..."     # Options: [python | java | go | kotlin | typescript]
description: "..."

architecture:
  agent: "..."          # Options: [single | multi]
  stateful: false       # Options: [true | false]
  datasources:          # Options: [hardcoded | local | external]
    - "..."
  rag: false            # Options: [true | false]

dependencies:
  libraries:
    - "ADK"
  services:
    - "GCP Project"

team:
  name: "YOUR TEAM NAME"
  email: "team@email.com"
poc:
  name: "POINT OF CONTACT NAME"
  email: "poc@email.com"

tags:
  - "..."
```

Omit `architecture` entirely if none of its sub-fields can be determined.

### 6. Validate

After writing the manifest, run the validation tool from the `tools/` directory:

```bash
cd tools && uv run --active validate manifest <recipe-path>
```

If validation fails, fix the errors and re-validate until it passes. Report the final `[PASS]` output to the user. Do not change or modify any other files in the repository. Also do not commit the changes.

### 7. Report back

Briefly summarize the key judgment calls you made (e.g. why `stateful: true`, why certain datasources were chosen) so the user can review and correct anything that doesn't look right.
