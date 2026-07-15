# Quick Recipe Update & Preparation Guide

This guide outlines the steps to update and prepare a Python recipe (under `core/python/` or `contrib/`) for a Pull Request.

---

## 1. Manual Preparation
Perform these steps manually before invoking any AI assistant tools:

1. **Deactivate** any active Python virtual environment in your terminal.
2. **Update the repository**:
   ```bash
   git pull
   uv sync
   ```
3. **Copy the recipe** folder into its target location (e.g., `core/python/my-recipe` or `contrib/my-recipe`).
4. **Rename the recipe folder** if needed to match the repository naming conventions.
5. **Create a git snapshot**: Commit your copied recipe folder first. This allows you to easily diff and review the changes made by the automated skills.

---

## 2. Automated Preparation (Recommended)
The easiest way to prepare a recipe is using the **`prepare-python-recipe`** master skill. It runs all sub-steps and checks end-to-end.

**Example Prompts:**
* `"Prepare core/python/my-recipe end to end."`
* `"Run all the checks and fixes on contrib/my-agent — make it PR-ready."`

---

## 3. Step-by-Step Skill Execution
If you prefer to run the preparation phases individually, follow this sequence:

### Phase 1: Generate `manifest.yaml`
Generate the metadata manifest file:
* **Prompt:** `"Create a manifest file to core/python/my-recipe recipe"`
* **Action:** Review the generated `manifest.yaml` and verify the values. Update fields like `team` and `poc` (Point of Contact) if they are missing or incorrect.

### Phase 2: Extract Environment Variables
Scan the recipe for environment variables and hardcoded model names:
* **Prompt:** `"Extract the environment variables for the core/python/my-recipe recipe"`
* **Action:** This updates/creates `.env.example` with placeholders and bootstraps `load_dotenv()` in the package initialization.

### Phase 3: Clean up `pyproject.toml`
Align the recipe's configuration with repository standards:
* **Prompt:** `"Review pyproject.toml in the core/python/my-recipe recipe and tell me what needs updating"`

### Phase 4: Format and Lint
From the repository root, run `ruff` to format and fix auto-fixable issues:
```bash
uv run ruff format core/python/my-recipe
uv run ruff check --fix core/python/my-recipe
```

### Phase 5: Generate Runnability Test
Generate a smoke test to verify the recipe's imports and entry points:
* **Prompt:** `"create the runnability test for core/python/my-recipe"`

### Phase 6: Lock Dependencies
Lock dependencies locally inside the recipe folder:
1. Navigate to the recipe directory:
   ```bash
   cd core/python/my-recipe
   ```
2. Run `uv lock`:
   ```bash
   uv lock
   ```
