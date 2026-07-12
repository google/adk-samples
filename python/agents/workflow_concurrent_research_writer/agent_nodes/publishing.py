from google.adk.agents.llm_agent import LlmAgent
from ..prompts import GENERATE_BLOG_POST_PROMPT

generate_blog_post_agent = LlmAgent(
    name="generate_blog_post_agent",
    model="gemini-2.5-flash",
    instruction=GENERATE_BLOG_POST_PROMPT,
)
