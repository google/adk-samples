"""
Delete the deployed bidi-demo agent from Agent Engine.

Usage:
    uv run agent_engine/cleanup.py
"""

import os

import vertexai
from dotenv import load_dotenv

load_dotenv(
    os.path.join(os.path.dirname(__file__), "..", "app", ".env"), override=True
)

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1")


def main():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

    resource_name = open("agent_resource_name.txt").read().strip()
    print(f"Deleting agent: {resource_name}")

    agent = client.agent_engines.get(name=resource_name)
    agent.delete(force=True)
    print("Agent deleted successfully.")

    os.remove("agent_resource_name.txt")


if __name__ == "__main__":
    main()
