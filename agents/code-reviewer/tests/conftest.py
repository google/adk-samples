import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_env_token():
    with patch("os.getenv", return_value="mocktoken"):
        yield

@pytest.fixture
def mock_user():
    user = MagicMock()
    user.login = "mockuser"
    return user

@pytest.fixture
def mock_github(mock_user):
    with patch("code_reviewer.tools.github_ops.Github") as mock_gh_class:
        mock_gh = MagicMock()
        mock_gh.get_user.return_value = mock_user
        mock_gh_class.return_value = mock_gh
        yield mock_gh

@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.full_name = "owner/repo"
    return repo

@pytest.fixture
def mock_commit():
    commit = MagicMock()
    commit.commit.author = "mockauthor"
    return commit

@pytest.fixture
def mock_pull_request():
    pr = MagicMock()
    pr.number = 42
    pr.diff_url = "https://mock.diff.url"
    pr.get_commits.return_value = ["mock_commit"]
    return pr

@pytest.fixture
def mock_github_with_repo(mock_github, mock_repo):
    mock_github.get_repo.return_value = mock_repo
    return mock_github

@pytest.fixture
def mock_github_with_repo_and_commit(mock_github_with_repo, mock_commit):
    mock_github_with_repo.get_repo.return_value.get_commit.return_value = mock_commit
    return mock_github_with_repo

@pytest.fixture
def mock_github_with_repo_and_pr(mock_github_with_repo, mock_pull_request):
    mock_github_with_repo.get_repo.return_value.create_pull.return_value = mock_pull_request
    mock_github_with_repo.get_repo.return_value.get_pull.return_value = mock_pull_request
    return mock_github_with_repo

@pytest.fixture
def mock_requests_get():
    with patch("code_reviewer.tools.github_ops.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "mock diff content"
        mock_get.return_value = mock_response
        yield mock_get
