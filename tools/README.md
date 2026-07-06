# Tools

Local developer tools for validating recipes before submitting a PR.

## Setup

Run once from the **repo root** to install dependencies and register the `validate` command:

```bash
uv sync
```

## Usage

All commands are run from the **repo root**:

```bash
uv run validate <subcommand> [recipe]
```

| Argument | Description |
|---|---|
| `subcommand` | The check to run (see below) |
| `recipe` | Optional. Path to a single recipe relative to the repo root (e.g. `core/rag-agent-search`). If omitted, all recipes are checked. |

## Subcommands

### `manifest` — validate manifest.yaml

Checks that a recipe directory has a `manifest.yaml` file and that it conforms to the schema in `.github/schemas/manifest-schema.json`.

```bash
# Check all recipes
uv run validate manifest

# Check a single recipe
uv run validate manifest core/rag-agent-search
```

### `all` — run all checks

Runs every available check in sequence and prints a combined summary.

```bash
# Check all recipes
uv run validate all

# Check a single recipe
uv run validate all core/rag-agent-search
```

---

> Adding a new tool? Create a new `validate_<name>.py` file in this directory with a `main(recipe: str | None) -> int` function, then register it in `validate.py` under `SUBCOMMANDS`.
