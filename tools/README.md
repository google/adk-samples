# Tools

Local developer tools for validating recipes before submitting a PR.

## Setup

Run once from the **repo root** to install dependencies and register the `validate` command:

```bash
uv sync
```

## Usage

Both arguments are optional:

```bash
uv run validate [subcommand] [scope]
```

| Argument | Description |
|---|---|
| `subcommand` | Optional. Which check(s) to run: `manifest` or `all` (default: `all`). |
| `scope` | Optional. `all` (default), `core`, `contrib`, or a path to a single recipe (e.g. `core/rag-agent-search`). |

```bash
uv run validate                                  # run all checks on all recipes
uv run validate all core                         # run all checks on core/ only
uv run validate manifest core/rag-agent-search   # run manifest check on one recipe
```

## Subcommands

### `manifest` — validate manifest.yaml

Checks that a recipe directory has a `manifest.yaml` file and that it conforms to the schema in `.github/schemas/manifest-schema.json`.

### `all` — run all checks

Runs every available check in sequence and prints a combined summary.

---

> Adding a new tool? Create a new `validate_<name>.py` file in this directory with a `main(scope: str | None) -> int` function, then register it in `validate.py` under `SUBCOMMANDS`. The `scope` argument follows the same convention as existing validators: `None`/`"all"` for everything, `"core"` or `"contrib"` for a single root, or `"core/<recipe>"` for a single recipe.
