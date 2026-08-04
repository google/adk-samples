import pytest

from economic_research.sub_agents.agent import JudgeAgent


def test_judge_agent_instantiation():
    """Test that the JudgeAgent.get_agent() successfully instantiates an ADK Agent."""
    try:
        judge = JudgeAgent()
        agent_instance = judge.get_agent()

        # Verify it has the correct name and type
        assert agent_instance.name == "Auditor_Judge"
        assert len(agent_instance.tools) > 0
    except ImportError as e:
        pytest.skip(
            f"Skipping test because ADK library or dependency is missing: {e}"
        )
    except Exception as e:
        # If we get a real exception (like a TypeError or NameError), that's a failure
        raise AssertionError(
            f"JudgeAgent instantiation failed with unexpected error: {e}"
        ) from e
