import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = os.getenv("MODEL_VERSION")

api_base_url = "http://localhost:8000/v1"
model_name_at_endpoint = f"hosted_vllm//gcs/{MODEL_NAME}/{MODEL_VERSION}"

root_agent = LlmAgent(
    name="root_agent",
    model=LiteLlm(
        model=model_name_at_endpoint,
        api_base=api_base_url,
    ),
    instruction=(
        """You are a helpful agent who can answer user questions."""
    ),
    description="Answers questions about math problems.",
)
