from google.adk.agents import Agent

from utils.prompt import BASE_SYSTEM_INSTRUCTION

from utils.config import LIVE_AGENT_MODEL

root_agent = Agent(
    name="info_gather_agent",
    model=LIVE_AGENT_MODEL,
    instruction=BASE_SYSTEM_INSTRUCTION
)

