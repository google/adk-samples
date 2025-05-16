from google.adk.tools import ToolContext
from github import GithubException
from .github_ops import GithubHandler

import re

def get_commit_tool(tool_context: ToolContext) -> dict[str, str]:

    user_input = tool_context.user_content.parts[0].text

    sha_match = re.search(r"\b([0-9a-f]{6,40})\b", user_input)
    if not sha_match:
        return {"status": "error", "message": "No valid SHA found in user input."}
    sha = sha_match.group(1)

    repo_match = re.search(r"\b([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+)\b", user_input)
    if not repo_match:
        return {"status": "error", "message": "No valid repository name (owner/repo) found in user input."}
    repo_name = repo_match.group(1)

    try:
        handler = GithubHandler(repo_name=repo_name)
        repo_object = handler.get_repository(lazy=True)

        if repo_object is None:
             return {"status": "error", "message": f"Could not retrieve repository '{repo_name}'. Check logs."}

        commit = repo_object.get_commit(sha=sha)

        return {
            "status": "ok",
            "author": commit.commit.author.name,
            "message": commit.commit.message,
            "date": commit.commit.author.date.isoformat(),
        }
    except GithubException as gh_error:
         return {"status": "error", "message": f"GitHub API Error: {gh_error.status} {gh_error.data.get('message', '')}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}