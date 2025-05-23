import os
from dotenv import load_dotenv
import logging
import requests

from github import (
    Auth,
    Github,
    Repository,
    Commit,
    PullRequest
)

from github.GithubException import (
    BadCredentialsException,
    UnknownObjectException
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

class GithubHandler:
    def __init__(self, repo_name: str):
        """
        Initialize the GithubHandler with the repository name.
        """
        self.repo_name = repo_name
        self.pat = os.getenv("GH_PAT")
        if not self.pat:
            raise ValueError("GitHub token not found. Check the .env file")

    def authenticate(self) -> Github:
        """
        Authenticate with the GitHub API using the provided token.
        """
        try:
            auth = Auth.Token(self.pat)
            gh = Github(auth=auth)
            status = gh.get_user().login

            logging.info("Username authenticated: %s", status)
            print("---")
            return gh
        except BadCredentialsException as auth_error:
            logging.error(f"Authenticate error: {auth_error}")


    def get_repository(self, lazy = False) -> Repository:
        """
        Get the repository object for the specified repository name.
        """
        try:
            gh = self.authenticate()
            repo =  gh.get_repo(
                full_name_or_id=self.repo_name,
                lazy=lazy
            )
            logging.info(f"Repository: {self.repo_name}")
            return repo
        except UnknownObjectException as repo_error:
            logging.error(f"Repository error: {repo_error}")


    def get_commit(self, sha: str) -> Commit:
        """
        Get the commit object for the specified SHA.
        """
        try:
            repo = self.get_repository()
            commit = repo.get_commit(sha=sha)
            logging.info(f"Commit Author: {commit.commit.author}")
            return commit
        except UnknownObjectException as get_commit_error:
            logging.error(f"Get commit error: {get_commit_error}")

    def create_pr(self, title: str, body:str, base="main", head="preprod") -> PullRequest:
        """
        Create a pull request with the specified title and body.
        """
        try:
            repo = self.get_repository()

            pr = repo.create_pull(
                base=base,
                head=head,
                title=title,
                body=body,
            )
            logging.info(f"PR: {pr}")
            return pr.get_commits()
        except UnknownObjectException as create_pr_error:
            logging.error(f"Create PR Error: {create_pr_error}")
        
    def get_pr(self, pr_number: int) -> PullRequest:
        """
        Get the pull request object for the specified PR number.
        """
        try:
            repo = self.get_repository()
            pr = repo.get_pull(pr_number)
            logging.info(f"PR Number: {pr_number}")
            return pr
        except UnknownObjectException as get_pr_error:
            logging.error(f"PR number error: {get_pr_error}")
    
    def get_diff(self, pr_number: int) -> str:
        """
        Get the diff of the specified PR.
        """
        try:
            pr = self.get_pr(pr_number)
            diff_url = pr.diff_url
            logging.info(f"PR Diff URL: {diff_url}")
            
            headers = {
                "Accept": "application/vnd.github.v3.diff",
                "Authorization": f"Bearer {self.pat}"
            }

            response = requests.get(diff_url, headers=headers)

            if response.status_code == 200:
                return response.text
            else:
                logging.error(f"Failed to fetch diff. Status code: {response.status_code}")
                return None
        except UnknownObjectException as get_diff_error:
            logging.error(f"Get diff error: {get_diff_error}") 
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None