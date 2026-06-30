# Parallel Task Decomposition & Execution Agent

The Parallel Task Decomposition & Execution agent is designed to decompose a single complex goal into multiple, independent sub-actions that can be executed concurrently to maximize efficiency and speed.

## Overview

The goal of this agent is to show how the agent decomposes a single complex goal into multiple, independent sub-actions that can be executed concurrently to maximize efficiency and speed. This "one-to-many" broadcast is highly efficient for rapid, coordinated communication across different systems like Slack, Email, and Google Calendar. It uses Google Search grounding to provide the most up-to-date and accurate information to the user.

**Example Use Cases:**

* Sending an "all-hands" meeting announcement to multiple Slack channels, a company-wide email, and creating a corresponding calendar event.
* Broadcasting a critical incident report (e.g., "Service X outage, estimated fix in 2 hours") to relevant engineering channels and an internal status email.
* Pushing a product update notification to key team channels.

---
> **Note on Tools: For Inspiration, Not Production**
>
> The goal of this agent is to demonstrate an agentic workflow, not to be a drop-in, production-ready solution. By default, it uses **mock tools** that only simulate actions (like sending a Slack message).
>
> This approach allows you to quickly clone the repository and see the agent's flow in action without a complicated setup process. In a real-world scenario, you would replace these mock tools with connections to actual services like Discord, Notion, or Google Services, etc and you would also want to customize the prompts to fit your needs.
>
> The code for connecting to real services (Slack, Gmail, and Calendar) via MCP is included but commented out in `agent.py` and `tools.py` to serve as a starting point.
---

## Agent Details
| Feature | Description |
| --- | --- |
| *Interaction Type* | Workflow |
| *Complexity* | Intermediate |
| *Agent Type* | Multi-Agent |
| *Components* | Tools |
| *Vertical* | General |

## Setup and Installation
1.  **Prerequisites:**

    **Google Cloud SDK and GCP Project:**

    For the Agent Runtime deployment steps, you will need
    a Google Cloud Project. Once you have created your project,
    [install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install).
    Then run the following command to authenticate with your project:
    ```bash
    gcloud auth login
    ```
    You also need to enable certain APIs. Run the following command to enable
    the required APIs:
    ```bash
    gcloud services enable aiplatform.googleapis.com
    ```

2.  **Installation:**

    Clone this repository and change to the repo directory:
    ```
    git clone https://github.com/google/adk-samples.git
    cd adk-samples/python/agents/parallel_task_decomposition_execution
    ```

    Install [uv](https://docs.astral.sh/uv/)
    If you have not installed uv before, you can do so by running:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    Install the agent's requirements:

    This is a one-time setup.
    ```bash
    uv sync --dev
    ```

    This will also install the released version of 'google-adk', the Google Agent Development Kit.

3.  **Configuration:**

    **Environment:**

    There is a `.env-example` file included in the repository. Update this file
    with the values appropriate to your project, and save it as `.env`. The values
    in this file will be read into the environment of your application.

    Once you have created your `.env` file, if you're using the `bash` shell,
    run the following command to export the variables from the `.env` file into your
    local shell environment:
    ```bash
    set -o allexport
    . .env
    set +o allexport
    ```
    If you aren't using `bash`, you may need to export the variables manually.

## Running the Agent

**Using the ADK command line:**

From the `parallel_task_decomposition_execution` directory, run this command:
```bash
adk run parallel_task_decomposition_agent
```

**Using the ADK Dev UI:**

From the `parallel_task_decomposition_execution` directory, run this command:
```bash
adk web .
```
It will display a URL for the demo UI (the default is http://127.0.0.1:8000). Go to that URL.

The UI will be blank initially. In the dropdown at the top left, choose `parallel_task_decomposition_agent`
to load the agent.

The logs from the agent will display on the console in real time as it runs.

### Example Interaction

Begin the interaction by typing "Hello". The agent will then prompt you for a topic. Add in a topic that you'd like to research and then broadcast to your team across multiple communication channels.

## Disclaimer

This agent sample is provided for illustrative purposes only and is not intended for production use. It serves as a basic example of an agent and a foundational starting point for individuals or teams to develop their own agents.

This sample has not been rigorously tested, may contain bugs or limitations, and does not include features or optimizations typically required for a production environment (e.g., robust error handling, security measures, scalability, performance considerations, comprehensive logging, or advanced configuration options).

Users are solely responsible for any further development, testing, security hardening, and deployment of agents based on this sample. We recommend thorough review, testing, and the implementation of appropriate safeguards before using any derived agent in a live or critical system.

### Google Agents CLI (Recommended)

Use the [Google Agents CLI](https://github.com/google/agents-cli) to create a production-ready version of this agent with additional deployment options.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-parallel-task-decomposition` with your project name):

```bash
agents-cli create my-parallel-task-decomposition -a adk@parallel-task-decomposition-execution
```

The Google Agents CLI will prompt you to select deployment options and provides additional production-ready features including automated CI/CD deployment scripts.