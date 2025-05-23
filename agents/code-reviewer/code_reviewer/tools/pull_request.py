from google.adk.tools import ToolContext
from github import GithubException, UnknownObjectException
from .github_ops import GithubHandler
import re
from typing import Any, Dict, List

def get_pull_request(tool_context: ToolContext) -> dict[str, str]:
    """
    Extracts pull request information from user input.
    """
    user_input = tool_context.user_content.parts[0].text

    pr_match = re.search(r"\b([0-9]+)\b", user_input)
    if not pr_match:
        return {"status": "error", "message": "No valid PR number found in user input."}
    pr_number = int(pr_match.group(1))

    repo_match = re.search(r"\b([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+)\b", user_input)
    if not repo_match:
        return {"status": "error", "message": "No valid repository name (owner/repo) found in user input."}
    repo_name = repo_match.group(1)

    try:
        handler = GithubHandler(repo_name=repo_name)
        repo_object = handler.get_repository(lazy=True)
        if repo_object is None:
            return {"status": "error", "message": f"Could not retrieve repository '{repo_name}'. Check authentication and repo existence. See logs for details."}

        pull_request = repo_object.get_pull(pr_number)

        changed_files_details: List[Dict[str, Any]] = []

        files = pull_request.get_files()
        for file in files:
            file_info = {
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "changes": file.changes,
            }
            changed_files_details.append(file_info)

        return {
            "status": "ok",
            "title": pull_request.title,
            "body": pull_request.body,
            "url": pull_request.html_url,
            "state": pull_request.state,
            "merged": pull_request.merged,
            "files_changed": changed_files_details,
            "path": pull_request.patch_url,
            "diff": pull_request.diff_url,
        }

    except UnknownObjectException as obj_error:
         return {"status": "error", "message": f"GitHub API Error: Pull Request #{pr_number} not found in repository '{repo_name}'. {obj_error.status} {obj_error.data.get('message', '')}"}
    except GithubException as gh_error:
         return {"status": "error", "message": f"GitHub API Error: {gh_error.status} {gh_error.data.get('message', '')}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}