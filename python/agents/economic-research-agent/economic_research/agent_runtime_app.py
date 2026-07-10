# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

from vertexai.preview.reasoning_engines import AdkApp
from economic_research.agent import ERAAgent

# Instantiate the agent
era_instance = ERAAgent()
root_agent = era_instance.get_app().root_agent

# Expose agent_runtime for agents-cli introspection
agent_runtime = AdkApp(
    agent=root_agent,
    enable_tracing=True,
)
