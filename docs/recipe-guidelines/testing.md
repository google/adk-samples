# Testing

← Back to the [Recipe Guidelines hub](README.md)

---

## 🧪 Runnability Tests (`tests/test_runnability.py`)

Recipes must contain a `tests/` folder with a `test_runnability.py` test (using `pytest`) to verify that the agent instantiates successfully without requiring active external API calls or real credentials.

> **Recommended: use the `generate-python-runnability-test` skill.** It
> parses your recipe's `agent.py` **and** its package `__init__.py` to detect
> what side effects fire at import time (`vertexai.init`, `google.auth.default`,
> `INTEGRATION_TEST` env-var reads), then emits the minimal test that will
> actually import cleanly without real credentials. Refuses to overwrite an
> existing file unless `--overwrite` is passed. See [Developer Agent Skills](tooling-and-ci.md#developer-agent-skills-agentsskills).

### Example — minimal shape (no import-time side effects detected)

```python
"""Runnability tests for the recipe."""

import app.agent


def test_agent_runnability() -> None:
    """Verify agent.py imports and defines the expected globals."""
    assert app.agent.root_agent is not None
    assert app.agent.app is not None
```

### Example — guarded shape (side effects detected)

For recipes whose `agent.py` or `__init__.py` calls `vertexai.init(...)`
and/or `google.auth.default()` at import time, the guarded shape sets
dummy env vars and patches the offending calls so the import succeeds
without valid credentials. The patches are only active during the
import; assertions run after the `with` block:

```python
"""Runnability tests for the recipe."""

import os
from unittest.mock import MagicMock, patch


def test_agent_runnability() -> None:
    """Verify agent.py imports and defines the expected globals."""
    # provide a dummy GCP project and patch google.auth.default() so import-time
    # credential lookups don't need ADC, and mock vertexai.init to avoid a real
    # GCP call — the setup must happen before the import.
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
    os.environ.setdefault("INTEGRATION_TEST", "TRUE")

    with (
        patch("vertexai.init"),
        patch(
            "google.auth.default", return_value=(MagicMock(), "test-project")
        ),
    ):
        import app.agent

    assert app.agent.root_agent is not None
    assert app.agent.app is not None
```

**Why patch `google.auth.default` and not just set `GOOGLE_CLOUD_PROJECT`?**
Some recipes gate the credential call (`if not project_id: _, project_id =
google.auth.default()`) — for those, the env var alone is enough. Others
call it unconditionally in `__init__.py` — for those, the env var runs
too late (the call has already fired) and only a patch prevents the
`DefaultCredentialsError`. The generated test handles both cases.

---

## 🚀 Running & Testing Recipes Locally

To interactively run an agent or run its tests locally, navigate to the recipe's subfolder:

### Running the Agent (Interactive Mode)
Start the agent locally in interactive mode to test prompts:
```bash
cd core/<recipe-name>
uv run adk run app
```

### Running Tests
Run unit tests or runnability checks using `pytest`:
```bash
cd core/<recipe-name>
uv run pytest
```

> **Note:** Tests under `tests/integration/` require live credentials and are excluded from CI automatically. To run them locally: `uv run pytest tests/integration`

---

← Back to the [Recipe Guidelines hub](README.md)
