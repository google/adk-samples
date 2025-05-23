# Code Reviewer Agents

This project implements an AI-powered code review assistant using Google’s Agent Development Kit (ADK). The assistant is designed to analyze GitHub repositories, extract commit information, and review pull requests by leveraging a team of specialized agents.

## Overview

The Code Reviewer Agent is built to assist developers in reviewing code repositories with enhanced context awareness. The root agent delegates tasks to specialized sub-agents based on the user's request, enabling modular analysis of GitHub commits and pull requests. This allows the agent to provide streamlined insights into code changes and overall repository activity.

The project is powered by Gemini through the Google ADK, and interacts with GitHub using the PyGitHub library, authenticated via a GitHub Personal Access Token (PAT).

## Agent Details

| Feature            | Description             |
| ------------------ | ----------------------- |
| _Interaction Type_ | Conversational          |
| _Complexity_       | Intermediate            |
| _Agent Type_       | Multi-Agent             |
| _Components_       | Tools, Code Analysis    |
| _Vertical_         | Developer Productivity  |

### Agent Architecture

The architecture consists of one root agent and two specialized sub-agents:

```
├── code_reviewer
│   ├── agent.py
│   ├── __init__.py
│   ├── prompt.py
│   ├── sub_agents
│   │   ├── commit
│   │   ├── __init__.py
│   │   ├── pull_request
│   └── tools
│       ├── commit.py
│       ├── get_diff.py
│       ├── github_ops.py
│       ├── pull_request.py

```

Each sub-agent is responsible for performing a focused analysis task and returning relevant information to the root agent, which communicates with the user.

### Key Features

- **Commit Analysis:**
  - Extracts commit message, author, files changed, and lines added/deleted.
  - Provides summaries of changes.

- **Pull Request Review:**
  - Analyzes code diffs in PRs.
  - Offers AI-generated suggestions and review comments.
  - Detects potential issues or patterns in code changes.

- **Task Delegation:**
  - Routes user input to the appropriate sub-agent based on intent.

- **GitHub Integration:**
  - Authenticates via GitHub PAT.
  - Uses PyGitHub for API access.

## Setup and Installations

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- Google ADK SDK
- GitHub Personal Access Token (PAT)
- Google Cloud Project (for Vertex AI Gemini)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/google/adk-samples.git
   cd code-review
   ```

2. **Install dependencies using Poetry:**

   If Poetry is not installed:

   ```bash
   pip install poetry
   ```

   Then install the project dependencies:

   ```bash
   poetry install
   ```

   Activate the virtual environment:

   ```bash
   poetry shell
   ```

3. **Set up environment variables:**

   Create a `.env` file with the following contents:

   ```
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_LOCATION=your-location-on-gcloud
   MODEL=the-model-you-prefer
   GH_PAT=your-github-PAT
   GOOGLE_GENAI_USE_VERTEXAI=0
   ```

   Replace the placeholders with your actual project configuration and GitHub token.

4. **Enable Google Cloud APIs:**

   Authenticate and enable Vertex AI:

   ```bash
   gcloud auth login
   gcloud services enable aiplatform.googleapis.com
   ```

5. **(Optional) Configure project settings manually:**

   If needed, set environment variables in your shell:

   ```bash
   export GOOGLE_CLOUD_PROJECT=your-project-id
   export GOOGLE_CLOUD_LOCATION=your-location
   export GOOGLE_GENAI_USE_VERTEXAI=1
   ```

## Running the Agent

To run the agent using the ADK CLI:

```bash
adk run code_reviewer
```

To run using the ADK Web UI:

```bash
adk web
```

Then select `code_reviewer` from the available agents.

### Example Interaction

**User:**
Can you show me the latest commit on this repo?

**Agent:**
Sure, I’ll fetch the latest commit details from the repository you specify. Please provide the repo URL or name.

**User:**
Here’s the link: https://github.com/example/repo

**Agent:**
Got it. Here’s the latest commit:

- **Message:** Fix bug in auth logic
- **Author:** Jane Doe
- **Files Changed:** auth.py
- **Lines Added:** 12
- **Lines Deleted:** 4

Would you like a deeper analysis?

---

**User:**
Can you review this PR? https://github.com/example/repo/pull/123

**Agent:**
Analyzing pull request #123…

✅ Summary of changes:
- Refactors the authentication flow
- Adds unit tests for edge cases

🔍 Review comments:
- [auth.py: line 45] Consider renaming the function `check_token` to better reflect its behavior.
- [tests/test_auth.py: line 10] You might want to add a case for expired tokens.

Would you like me to summarize this review for posting?

## Notes

- The agents are currently configured for simple use cases.
- Authentication is based on a GitHub PAT with appropriate read access.
- Model selection is customizable via the `.env` file.
- You can expand the number of sub-agents to include other types of code insights (e.g., style checks, security scanning).