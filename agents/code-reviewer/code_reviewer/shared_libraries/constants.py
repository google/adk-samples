import os
from dotenv import load_dotenv

load_dotenv()

ROOT_AGENT_NAME = "code_reviewer"
ROOT_AGENT_DESCRIPTION = "Review code from a especific repository"
COMMITS_AGENT_NAME = "get_commit"
COMMITS_AGENT_DESCRIPTION = "Get commit information from a specific repository"
PR_AGENT_NAME = "get_pr"
PR_AGENT_DESCRIPTION = "Get pull request information from a specific repository"

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "EMPTY")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
MODEL = os.getenv("MODEL", "gemini-2.0-flash-001")
GH_PAT = os.getenv("GH_PAT", "EMPTY")
