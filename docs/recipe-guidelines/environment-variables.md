# Environment Variables Configuration (`.env.example`)

← Back to the [Recipe Guidelines hub](README.md)

---

Define any environment variables (API keys, project IDs, model names) in `.env.example` with placeholder values. **Never commit `.env` files with active secrets.**

## Placeholder Convention
Use `<TODO: update-this-value>` as the placeholder for any variable that has no sensible default:
```
GOOGLE_CLOUD_PROJECT=<TODO: update-this-value>
GOOGLE_CLOUD_LOCATION=<TODO: update-this-value>
MODEL_NAME=gemini-2.5-flash
```

## Required Variables
Every recipe's `.env.example` must declare at least one variable starting with each of the following prefixes:
*   `GOOGLE_CLOUD_PROJECT`
*   `GOOGLE_CLOUD_LOCATION`
*   `MODEL_NAME`

> **Note:** CI currently flags a missing prefix as a **warning** (it does not
> fail the build), but declaring all three is still required by these guidelines.

## Completeness (enforced by CI — fails the build)
Every environment variable your Python code reads — via `os.getenv`,
`os.environ.get`, or `os.environ[...]` in non-test files — **must** be declared
in `.env.example`, or the `validate-python-sample` CI check **fails the pull
request**. The `extract-python-environment-variables` skill can populate these
for you automatically.

> **Known limitation:** the CI scan does not detect `from os import getenv`
> (direct import) style reads, so prefer `os.getenv(...)` / `os.environ[...]`.

## Local Setup for Users
Users must copy `.env.example` to a git-ignored `.env` file and fill in their actual settings:
```bash
cp .env.example .env
```

## Loading Environment Variables (Python)
1. Declare `python-dotenv` in your dependencies (included in the [`pyproject.toml` template](required-files.md#dependency-management-pyprojecttoml)).
2. Bootstrap `load_dotenv()` in `app/__init__.py` — **not** in `agent.py`. Placing it here ensures the environment is populated once, before any module-level code in other files runs. Use the following exact snippet:

```python
from dotenv import load_dotenv

# Load the .env file
if not load_dotenv():
    raise FileNotFoundError(
        "Critical Error: No .env file found. "
        "Make sure to copy .env.example to .env and update the values."
    )

from .agent import app  # relative imports come AFTER load_dotenv()

__all__ = ["app"]
```

3. In `agent.py` and other modules, read variables normally via `os.getenv()`:

```python
import os

model_name = os.getenv("MODEL_NAME")
```

> **Note:** The `extract-python-environment-variables` skill can automate all
> of the above — scanning your code for env var reads, populating
> `.env.example`, and injecting the `load_dotenv()` snippet into `__init__.py`.
> See [Developer Agent Skills](tooling-and-ci.md#developer-agent-skills-agentsskills).

---

← Back to the [Recipe Guidelines hub](README.md)
