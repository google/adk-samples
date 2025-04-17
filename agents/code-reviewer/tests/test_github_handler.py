import pytest
from unittest.mock import MagicMock
from code_reviewer.tools.github_ops import GithubHandler

def test_authenticate_success(mock_env_token, mock_github, mock_user):
    handler = GithubHandler("owner/repo")
    gh = handler.authenticate()
    assert gh.get_user().login == "mockuser"

def test_get_repository_success(mock_env_token, mock_github_with_repo):
    handler = GithubHandler("owner/repo")
    repo = handler.get_repository()
    assert repo.full_name == "owner/repo"

def test_get_commit_success(mock_env_token, mock_github_with_repo_and_commit):
    handler = GithubHandler("owner/repo")
    commit = handler.get_commit("mocksha")
    assert commit.commit.author == "mockauthor"

def test_create_pr_success(mock_env_token, mock_github_with_repo_and_pr):
    handler = GithubHandler("owner/repo")
    commits = handler.create_pr("New Feature", "Adds new feature")
    assert len(commits) == 1

def test_get_pr_success(mock_env_token, mock_github_with_repo_and_pr):
    handler = GithubHandler("owner/repo")
    pr = handler.get_pr(42)
    assert pr.number == 42

def test_get_diff_success(mock_env_token, mock_github_with_repo_and_pr, mock_requests_get):
    handler = GithubHandler("owner/repo")
    diff = handler.get_diff(42)
    assert diff == "mock diff content"
