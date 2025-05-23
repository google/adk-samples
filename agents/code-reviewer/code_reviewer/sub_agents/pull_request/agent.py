from google.adk.agents.llm_agent import Agent
from ...shared_libraries import constants
from ...tools import pull_request, get_diff
from . import prompt

pr_analyzer = Agent(
    model=constants.MODEL,
    name=constants.PR_AGENT_NAME,
    description=constants.PR_AGENT_DESCRIPTION,
    instruction=prompt.PR_AGENT,
    tools=[
        pull_request.get_pull_request,
        get_diff.get_diff_content,
    ],
)