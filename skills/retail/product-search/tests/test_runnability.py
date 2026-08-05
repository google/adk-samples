"""Runnability test for the retail-product-search recipe."""


def test_agent_runnability() -> None:
    """Verify scripts/agent.py imports and defines root_agent."""
    import importlib

    module = importlib.import_module("scripts.agent")

    assert getattr(module, "root_agent", None) is not None
