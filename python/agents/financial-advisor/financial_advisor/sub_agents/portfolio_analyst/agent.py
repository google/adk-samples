from google.adk import Agent
from google.adk.tools import google_search

import prompt

MODEL = "gemini-2.5-pro-preview-05-06"

portfolio_analyst_agent = Agent(
    model=MODEL,
    name="portfolio_analyst_agent",
    instruction=prompt.PORTFOLIO_ANALYST_PROMPT,
    output_key="portfolio_analysis_output",
    tools=[google_search],
)
