from google.adk.agents import Agent
from typing import Dict
import os # Needed if you were to actually execute git commands
import subprocess # Needed for executing git commands
import google.generativeai as genai

API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

class GenerativeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
    
    def generate_content(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

model = GenerativeModel(model_name="gemini-2.0-flash")

def clone_repository(repo_url: str, clone_dir: str) -> str:
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(clone_dir, repo_name)
    
    if not os.path.exists(repo_path):
        subprocess.run(["git", "clone", repo_url], cwd=clone_dir, check=True)
    
    return repo_path

def get_repo_structure(repo_path: str) -> Dict:
    files_list = []
    important_files_content = {}
    important_files = ["README.md", "requirements.txt", "setup.py", "package.json", "pom.xml"]

    for root, dirs, files in os.walk(repo_path):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), repo_path)
            files_list.append(relative_path)

            if file in important_files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        important_files_content[relative_path] = f.read()
                except Exception as e:
                    important_files_content[relative_path] = f"Could not read file: {e}"

    return {
        "files": files_list,
        "important_files_content": important_files_content
    }

def create_prompt_from_repo(repo_info: Dict, repo_url: str) -> str:
    prompt = f"""
    You are an AI assistant helping to create a README.md file for a GitHub project.

    Repository URL: {repo_url}

    The project has the following structure:
    {repo_info["files"]}

    Important file contents are as follows:
    {repo_info["important_files_content"]}

    Based on this information:
    - Summarize what this repository is about.
    - Mention any installation steps if possible.
    - Explain how to use it.
    - Add any additional sections you find necessary (e.g., Contribution, License).

    Write it clearly, professionally, and make it look like a typical GitHub README file.
    """
    return prompt

def get_repo_with_git_command_and_generate_readme(repo_url: str, clone_dir: str = "./repositories"):
    repo_path = clone_repository(repo_url, clone_dir)
    repo_info = get_repo_structure(repo_path)
    prompt = create_prompt_from_repo(repo_info, repo_url)
    # Here you would call your AI model to generate the README content
    generated_readme = model.generate_content(prompt)
    print(f"Generated README content:\n{generated_readme}")
    readme_path = os.path.join(repo_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(generated_readme)

    print(f"✅ README generated at: {readme_path}")

def update_file_content(file_path: str, content: str) -> None:
    """Updates the content of a file at the specified path."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Successfully updated file: {file_path}")
    except Exception as e:
        print(f"Error updating file {file_path}: {e}")

# Define your agent as before
root_agent = Agent(
    name="readme_generation_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to get repositories using git command in console. Then generate a README file and fill it with the "
        "repository information. The README file will be in markdown format."
    ),
    instruction=(
        "You are a helpful assistant. You will be given a git URL. Your task is to retrieve the repository using a git "
        "command in console. Then generate a README file and fill it with the repository information. The README file "
        "will be in markdown format. Use the `get_repo_with_git_command` tool to get repository information."
    ),
    tools=[get_repo_with_git_command_and_generate_readme],
)