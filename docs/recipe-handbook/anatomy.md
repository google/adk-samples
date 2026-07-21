<!-- word count: 378 (target 700, cap 1000) -->

# Anatomy of a Recipe

The universal shape of a `contrib/` recipe. Language-specific
detail lives in [languages/](./languages/).

## Where a recipe lives

| Location | Use for |
|---|---|
| `contrib/<lang>/<name>` | Community-contributed recipes, nested by language |

## Naming

- Max 30 characters.
- Lowercase letters and hyphens only, starts with a letter
  (`^[a-z][a-z-]*$`).

## Size limits

| Root | Max files | Max size |
|---|---|---|
| `contrib/` | 70 | 2 MB |

**Excluded from the count:** generated files and caches. Common
exclusions:

- `uv.lock`, `__pycache__/`, `.venv/` (Python)
- `node_modules/`, lockfiles, `dist/` (TypeScript)
- `target/`, `build/`, `.gradle/` (Java, Kotlin)
- `vendor/`, `go.sum` (Go)

## `manifest.yaml`

Every recipe has one. Schema:
[`.github/schemas/manifest-schema.json`](../../.github/schemas/manifest-schema.json).
Generate with the `generate-manifest` AI skill.

**Required fields:**

| Field | Values |
|---|---|
| `type` | `standalone` (runnable) or `module` (importable sub-agent) |
| `status` | `active` or `inactive` |
| `language` | `python`, `java`, `go`, `kotlin`, `typescript` |
| `description` | Prose, minimum 10 characters |
| `ownership.team` | Team name |
| `ownership.poc` | GitHub user ID of the accountable owner |

**Common optional fields:**

| Field | Purpose |
|---|---|
| `deployable` | `true` if the recipe supports one-click deployment. Useful for surfacing the recipe in Agent Garden. Defaults to `false`. |
| `ownership.contributors` | Additional GitHub user IDs |
| `tags` | Classification strings |
| `architecture.agent` | `single` or `multi` |
| `architecture.stateful` | Whether the agent persists state |
| `architecture.datasources` | `hardcoded`, `local`, `external` |
| `dependencies.libraries` | e.g. `["adk", "langgraph"]` |
| `dependencies.services` | e.g. `["vertex-ai", "bigquery"]` |

For the exact set of valid values for each enumerated field, see
the [schema](../../.github/schemas/manifest-schema.json).

Example minimum:

```yaml
type: standalone
status: active
language: python
description: A retrieval-augmented search agent over public docs.
ownership:
  team: your-team-name
  poc: your-github-username
```

## `README.md`

Every recipe has one. Cover:

1. What the recipe does (one paragraph).
2. Setup — prerequisites, credentials, environment variables.
3. Run — the exact command to start the agent.
4. Optional: architecture diagram, example prompts, screenshots.

No enforced template. Write for a contributor who has never
seen the recipe before.

## See also

- **Language-specific files** (Python's `pyproject.toml`,
  `uv.lock`, `.env.example`, `tests/test_runnability.py`) — see
  [languages/](./languages/).

---

← [Checklist](../recipe-checklist.md) · [Handbook](./README.md)
