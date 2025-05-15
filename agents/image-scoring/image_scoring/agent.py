import datetime, uuid
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from .sub_agents.prompt import image_generation_prompt_agent as image_prompt
from .sub_agents.image import image_generation_agent as image_gen
from .sub_agents.scoring import scoring_images_prompt as scoring_prompt
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
#import checker_agent_instance
from .checker_agent import checker_agent_instance
from google.adk.agents import SequentialAgent, LoopAgent
from google.adk.agents.callback_context import CallbackContext


from vertexai.preview.reasoning_engines import AdkApp

def set_session(callback_context: CallbackContext):
    """
    Sets a unique ID and timestamp in the callback context's state.
    This function is called before the main_loop_agent executes.
    """
    
    callback_context.state['unique_id'] = str(uuid.uuid4())
    callback_context.state['timestamp'] = datetime.datetime.now(ZoneInfo("UTC")).isoformat()
    print(f"Callback set_session: unique_id='{callback_context.state['unique_id']}', timestamp='{callback_context.state['timestamp']}'")

llm_news_image_generation = SequentialAgent(
    name='image_generation_scoring_agent',
    description=(
        'Analyzes a news article and creates the image generation prompt, generates the relevant images with imagen3 and scores the images.' \
        'if the total_score is greater than 20 stop the execution' 
        
    ),
    sub_agents=[image_prompt, image_gen, scoring_prompt]
)
session_service = InMemorySessionService()

session = session_service.create_session(app_name="imagegen_news", 
                                        user_id='user1', 
                                        session_id='session11')


# --- 5. Define the Loop Agent ---
# The LoopAgent will repeatedly execute its sub_agents in the order they are listed.
# It will continue looping until one of its sub_agents (specifically, the checker_agent's tool)
# sets tool_context.actions.escalate = True.
main_loop_agent = LoopAgent(
    name="main_loop_agent",
    description="Repeatedly runs a sequential process and checks a termination condition.",
    sub_agents=[
        llm_news_image_generation, # First, run your sequential process [1]
        checker_agent_instance              # Second, check the condition and potentially stop the loop [1]
    ]
)
root_agent = main_loop_agent
app = AdkApp(agent=root_agent, artifact_service_builder= InMemoryArtifactService())
