"""Runnability test for the retail-virtual-tryon recipe."""


def test_agent_runnability() -> None:
    """Verify scripts/tryon_agent.py imports and defines root_agent."""
    import importlib

    module = importlib.import_module("scripts.tryon_agent")

    assert getattr(module, "root_agent", None) is not None
