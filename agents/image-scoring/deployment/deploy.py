import logging
import argparse
import sys, os
import vertexai

from image_scoring.agent import root_agent


from dotenv import load_dotenv
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp
from google.api_core.exceptions import NotFound

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


project_id=os.getenv("GOOGLE_CLOUD_PROJECT")
location=os.getenv("GOOGLE_CLOUD_LOCATION")
bucket=os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")





load_dotenv()
vertexai.init(
    project=project_id,
    location=location,
    staging_bucket=f"gs://{bucket}",
)

parser = argparse.ArgumentParser(description="Short sample app")

parser.add_argument(
    "--delete",
    action="store_true",
    dest="delete",
    required=False,
    help="Delete deployed agent",
)
parser.add_argument(
    "--resource_id",
    required="--delete" in sys.argv,
    action="store",
    dest="resource_id",
    help="The resource id of the agent to be deleted in the format projects/PROJECT_ID/locations/LOCATION/reasoningEngines/REASONING_ENGINE_ID",
)


args = parser.parse_args()

if args.delete:
    try:
        agent_engines.get(resource_name=args.resource_id)
        agent_engines.delete(resource_name=args.resource_id)
        print(f"Agent {args.resource_id} deleted successfully")
    except NotFound as e:
        print(e)
        print(f"Agent {args.resource_id} not found")

else:
    logger.info("deploying app...")
    app = AdkApp(agent=root_agent, enable_tracing=False)
    
    logging.debug("deploying agent to agent engine:")
    remote_app = agent_engines.create(
        app,
        requirements=[           
            "google-adk (>=0.0.2)",
            "google-cloud-aiplatform[agent_engines] (>=1.88.0,<2.0.0)",
            "google-genai (>=1.5.0,<2.0.0)",
            "google-cloud-storage(>=2.14.0,<=3.1.0)"
        ]
    )
    
    logging.debug("testing deployment:")
    session = remote_app.create_session(user_id="123")
    for event in remote_app.stream_query(
        user_id="123",
        session_id=session["id"],
        message="hello!",
    ):
        if event.get("content", None):
            print(
                f"Agent deployed successfully under resource name: {remote_app.resource_name}"
            )
