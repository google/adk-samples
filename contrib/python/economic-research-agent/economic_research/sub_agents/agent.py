from google.adk.agents import Agent
from google.adk.models import Gemini

from .prompt import JudgePrompts
from .tools.search_skill import web_search_skill

prompts = JudgePrompts()
JUDGE_INSTRUCTIONS = prompts.auditor_judge_instructions()


class JudgeAgent:
    def __init__(self):
        pass

    def get_agent(self) -> Agent:
        """
        Instantiates the Auditor Judge agent using ADK.
        """
        tools = [web_search_skill]

        # We use Gemini 2.5 flash as a lightweight, fast auditor
        return Agent(
            name="Auditor_Judge",
            model=Gemini(model_name="gemini-2.5-flash"),
            instruction=JUDGE_INSTRUCTIONS,
            tools=tools,
        )
