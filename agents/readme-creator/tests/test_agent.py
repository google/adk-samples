# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from google.adk.evaluation.agent_evaluator import AgentEvaluator
import dotenv
import pytest

# Import the agent components to test
from readme_creator import GenerativeModel, clone_repository, get_repo_structure, create_prompt_from_repo

@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables for testing."""
    dotenv.load_dotenv()
    # Setup test directory
    test_dir = tempfile.mkdtemp()
    os.environ["TEST_DIR"] = test_dir
    yield
    # Teardown
    shutil.rmtree(test_dir)

@pytest.fixture
def mock_git_repo():
    """Create a mock git repository structure for testing."""
    repo_path = os.path.join(os.environ["TEST_DIR"], "test_repo")
    os.makedirs(repo_path)
    
    # Create important files
    with open(os.path.join(repo_path, "README.md"), "w") as f:
        f.write("# Existing README\n")
    
    with open(os.path.join(repo_path, "requirements.txt"), "w") as f:
        f.write("google-generativeai\npython-dotenv\n")
    
    # Create some source files
    os.makedirs(os.path.join(repo_path, "src"))
    with open(os.path.join(repo_path, "src", "main.py"), "w") as f:
        f.write("def main():\n    pass\n")
    
    return repo_path

def test_generative_model_initialization():
    """Test that the GenerativeModel class initializes correctly."""
    model = GenerativeModel(model_name="gemini-2.0-flash")
    assert model.model_name == "gemini-2.0-flash"
    assert hasattr(model, "model")

@patch("subprocess.run")
def test_clone_repository(mock_subprocess, mock_git_repo):
    """Test repository cloning functionality."""
    repo_url = "https://github.com/testuser/testrepo.git"
    clone_dir = os.environ["TEST_DIR"]
    
    # Test cloning new repository
    result_path = clone_repository(repo_url, clone_dir)
    assert mock_subprocess.called
    assert result_path.endswith("testrepo")
    assert os.path.exists(result_path)

    # Test existing repository handling
    result_path = clone_repository(repo_url, clone_dir)
    assert not mock_subprocess.called  # Should skip cloning if exists

def test_get_repo_structure(mock_git_repo):
    """Test repository structure analysis."""
    repo_info = get_repo_structure(mock_git_repo)
    
    assert isinstance(repo_info, dict)
    assert "files" in repo_info
    assert "important_files_content" in repo_info
    assert "README.md" in repo_info["important_files_content"]
    assert "requirements.txt" in repo_info["important_files_content"]
    assert "src/main.py" in repo_info["files"]

def test_create_prompt_from_repo(mock_git_repo):
    """Test prompt generation from repository info."""
    repo_info = get_repo_structure(mock_git_repo)
    repo_url = "https://github.com/testuser/testrepo.git"
    prompt = create_prompt_from_repo(repo_info, repo_url)
    
    assert repo_url in prompt
    assert "README.md" in prompt
    assert "requirements.txt" in prompt
    assert "src/main.py" in prompt
    assert "Summarize what this repository is about" in prompt

@patch("readme_agent.GenativeModel.generate_content")
def test_readme_generation_agent(mock_generate):
    """Test the full agent workflow."""
    mock_generate.return_value = "# Generated README\nTest content"
    
    test_repo_url = "https://github.com/testuser/testrepo.git"
    clone_dir = os.environ["TEST_DIR"]
    
    # Mock the git clone to avoid actual network calls
    with patch("readme_agent.clone_repository") as mock_clone:
        mock_clone.return_value = os.path.join(clone_dir, "testrepo")
        
        # Mock the repo structure analysis
        with patch("readme_agent.get_repo_structure") as mock_structure:
            mock_structure.return_value = {
                "files": ["README.md", "requirements.txt", "src/main.py"],
                "important_files_content": {
                    "README.md": "# Existing README",
                    "requirements.txt": "google-generativeai"
                }
            }
            
            # Initialize and run the agent
            from readme_creator import root_agent
            root_agent.tools[0](test_repo_url, clone_dir)
            
            # Verify the generated README
            readme_path = os.path.join(clone_dir, "testrepo", "README.md")
            assert os.path.exists(readme_path)
            with open(readme_path, "r") as f:
                content = f.read()
                assert "Generated README" in content

def test_tools():
    """Test the agent's basic ability on a few examples."""
    AgentEvaluator.evaluate(
        "readme_generation_agent",
        os.path.join(os.path.dirname(__file__), "tools"),
        num_runs=1,
    )