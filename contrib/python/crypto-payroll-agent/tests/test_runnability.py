"""Runnability tests for the recipe."""


def test_agent_runnability() -> None:
    """Verify the package imports and defines root_agent."""
    import crypto_payroll_agent

    assert crypto_payroll_agent.root_agent is not None
