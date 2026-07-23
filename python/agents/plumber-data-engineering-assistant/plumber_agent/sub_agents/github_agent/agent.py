"""GitHub Agent module for repository and cloud storage management."""

from google.adk.agents import Agent
from google.genai import types

from .tools.git_ops import (
    add_files_to_git,
    commit_changes,
    get_git_status,
    initialize_git_repo,
    list_git_branches,
    switch_git_branch,
)
from .tools.github_api import (
    authenticate_github,
    download_repository,
    list_branches,
    search_repositories,
)
from .tools.github_prompts import AGENT_INSTRUCTIONS

root_agent = Agent(
    name="github_agent",
    model="gemini-2.5-flash",
    description=(
        "Advanced GitHub, Git, and Google Cloud Storage repository "
        "management agent with full version control and cloud storage "
        "capabilities."
    ),
    instruction=AGENT_INSTRUCTIONS,
    tools=[
        # GitHub API Tools
        authenticate_github,
        search_repositories,
        list_branches,
        download_repository,
        # Git Operations Tools
        initialize_git_repo,
        get_git_status,
        add_files_to_git,
        commit_changes,
        list_git_branches,
        switch_git_branch,
    ],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]
    ),
)
