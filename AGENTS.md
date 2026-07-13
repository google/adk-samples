# Agent Guidelines for adk-samples
All agents must follow the guidelines below without being reminded.

## General
- Use the term **recipe** instead of **sample**.

## Models
- Do NOT use `gemini-2.0-flash` or `gemini-2.5-flash` — both are deprecated. Use `gemini-3.5-flash` instead.

## Python
- Python recipes go under `contrib/` or `core/python`
- Use the `scaffold-python-recipe` skill to create a new python recipe
- Minimum python version: 3.11
- Package manager: Use `uv`, not `pip`
- Formatter: Use `ruff` (line length 80, double quotes)
