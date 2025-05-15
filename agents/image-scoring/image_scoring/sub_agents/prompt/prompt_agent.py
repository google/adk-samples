from ... import config
from google.adk.agents import Agent


image_generation_prompt_agent = Agent(
    name="image_generation_prompt_agent",
    model=config.GENAI_MODEL,
    description=(
        "You are an expert in creating imagen3 prompts for image generation"
    ),
    instruction=(
        "You are provide with a news article. Your task is to understand the news article and generate a relvant prompt to be used in imagen3"
        " to generate an image of the news article. You should generate a total of 1 prompt" \
    
    ),
    output_key="imagen_prompts"#gets stored in session.state
    
)






