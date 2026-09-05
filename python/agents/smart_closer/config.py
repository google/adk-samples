import os

# --- CENTRALIZED CONFIGURATIONS ---

# 1. Agent Metadata Configs
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3-flash-preview")
AGENT_NAME = os.environ.get("AGENT_NAME", "SmartCloserAgent")
