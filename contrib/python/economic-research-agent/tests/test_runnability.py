#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""
Smoke test verifying that the Economic Research Agent initializes and imports correctly.
"""

from unittest.mock import patch


def test_agent_runnability():
    """Verify that ERAAgent and root_agent initialize correctly."""
    with patch("google.adk.models.Gemini"):
        from economic_research.agent import ERAAgent, root_agent

        assert root_agent is not None
        assert ERAAgent is not None
