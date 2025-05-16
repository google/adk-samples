from google.adk.agents.llm_agent import Agent
from .shared_libraries import constants
from .sub_agents.commit.agent import get_commmit
from .sub_agents.pull_request.agent import pr_analyzer
from . import prompt

root_agent = Agent(
    model=constants.MODEL,
    name=constants.ROOT_AGENT_NAME,
    description=constants.ROOT_AGENT_DESCRIPTION,
    instruction=prompt.ROOT_AGENT,
    sub_agents=[
        get_commmit,
        pr_analyzer,
    ],
)