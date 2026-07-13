# Testing

← Back to the [Recipe Guidelines hub](README.md)

---

## 🧪 Runnability Tests (`tests/test_runnability.py`)

Recipes must contain a `tests/` folder with a `test_runnability.py` test (using `pytest`) to verify that the agent instantiates successfully without requiring active external API calls or real credentials.

### Example of `test_runnability.py`
For Python recipes, you can adapt this example test. It sets dummy credentials to ensure the test can run successfully in isolated CI environments:

```python
"""Runnability tests for the recipe."""

import os
from unittest.mock import patch

from google.adk.agents import Agent
from google.adk.apps import App


def test_agent_runnability() -> None:
    """Verifies agent.py compiles and instantiates the agent successfully."""
    # Set dummy credentials to avoid real GCP calls during import/instantiation
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
    os.environ.setdefault("INTEGRATION_TEST", "TRUE")

    with patch("vertexai.init"):
        # Import your agent module containing your app and root agent
        import app.agent

    # Validate instantiation
    assert app.agent.root_agent is not None
    assert isinstance(app.agent.root_agent, Agent)

    assert app.agent.app is not None
    assert isinstance(app.agent.app, App)
```

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
