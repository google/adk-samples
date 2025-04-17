import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types
from sub_agent.planning_agent import create_planning
from sub_agent.weather_agent import create_weather
from sub_agent.tools.weather import get_mcp_tools, close_mcp_connection
import prompt

load_dotenv()

app = FastAPI()

# Define the request model for user input
class QueryRequest(BaseModel):
    query: str


session_service = InMemorySessionService()
artifacts_service = InMemoryArtifactService()

# init the root agent
root_agent = Agent(
    model=LiteLlm(
        model="azure/gpt-4o",
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION")
    ),
    name="root_agent",
    description="Travel concierge services using multiple sub-agents",
    instruction=prompt.ROOT_AGENT_INSTR,
    sub_agents=[
        create_planning,
        create_weather
    ]
)

# On app startup: load MCP tools and assign to the weather agent
@app.on_event("startup")
async def init_weather_tool():
    tools = await get_mcp_tools()
    create_weather.tools = tools

# On app shutdown: cleanly close the MCP tool connection
@app.on_event("shutdown")
async def shutdown_cleanup():
    await close_mcp_connection()

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    session = session_service.create_session(
        state={}, app_name="mcp_app", user_id="user"
    )

    content = types.Content(role="user", parts=[types.Part(text=request.query)])

    runner = Runner(
        app_name="mcp_app",
        agent=root_agent,
        artifact_service=artifacts_service,
        session_service=session_service,
    )

    events_async = runner.run_async(
        session_id=session.id, user_id="user", new_message=content
    )

    response = []
    async for event in events_async:
        if event.content:
            for part in event.content.parts:
                if part.text:
                    response.append(part.text)

    return {"response": "\n".join(response)}
