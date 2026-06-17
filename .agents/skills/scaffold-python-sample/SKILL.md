---
name: scaffold-python-sample
description: >
  This skill should be used when the user wants to "create a new Python ADK sample",
  "scaffold a new Python sample recipe", "generate a new Python sample in contrib",
  "add a new Python sample to the adk-samples repository", or "create a Python adk sample".
  It utilizes an automated script to copy template files and resolve basic placeholders.
metadata:
  author: Google
  license: Apache-2.0
  version: 1.0.0
---

# Scaffolding a New Python ADK Sample

Use this skill to scaffold a new Python ADK sample recipe inside this repository using the automated script at `scripts/scaffold.py`.

---

## Rules

1. **No Manual Boilerplate Writing**: Always use `scripts/scaffold.py` to create the recipe. Never write the boilerplate files manually.
2. **Do Not Modify After Scaffolding**: Once the script runs successfully, do not make any further changes to the recipe in this turn.
3. **Use the Correct Terminology (Recipe)**: Always refer to what is being created as a **recipe**, never as a "project". Ensure all output messages, instructions, and explanations to the user use "recipe" exclusively.

---

## Inputs

Gather the following from the user before running the script:
- **Recipe Name** (Required): e.g., `weather-assistant`. Must be 26 characters or less, lowercase letters, numbers, and hyphens only.
- **Output Directory** (Optional): Defaults to `contrib/`.

---

## Run

Execute the scaffold script:
```bash
python3 .agents/skills/scaffold-python-sample/scripts/scaffold.py --name <RECIPE_NAME> --output-dir <OUTPUT_DIRECTORY>
```
*The script accepts exactly two flags: `--name` (required) and `--output-dir` (optional, defaults to `contrib/`). Do not pass any other flags.*

---
## Respond

Once the script succeeds, inform the user the recipe is ready. Remind them to update `RECIPE.md` (specifically the Title, Description, Owner, and Point of Contact, which are used to track ownership and support). Then, provide these quick-start commands:
```bash
cd <OUTPUT_DIRECTORY>/<RECIPE_NAME>
uv sync                  # install dependencies
uv run pytest            # run the test suite
uv run adk run app       # run the agent interactively
```
Do not make any further changes. End your turn.
