from google.adk.agents.llm_agent import Agent
from ...shared_libraries import constants
from ...tools import commit
from . import prompt

get_commmit = Agent(
    model=constants.MODEL,
    name=constants.COMMITS_AGENT_NAME,
    description=constants.COMMITS_AGENT_DESCRIPTION,
    instruction=prompt.GET_COMMIT,
    tools=[
        commit.get_commit_tool,
    ],
)