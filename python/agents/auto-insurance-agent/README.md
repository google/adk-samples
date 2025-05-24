# Auto insurance Agent using Apigee API hub

## Overview

This agent sample acts as a helpful virtual assistant for an auto insurance company. It is able to register new members and perform several functions for existing members, including filing claims, providing roadside assistance, and returning information about reward offers from partner companies.  The agent uses several tools to fulfill these tasks. The tools are custom APIs registered in [Apigee API hub](https://cloud.google.com/apigee/docs/apihub/what-is-api-hub).

## Agent Details

| Attribute | Detail |
|---|---|
|   Interaction Type |   Conversational |
|   Complexity |   Easy |
|   Agent Type |   Multi Agent |
|   Components |   Tools, ApiHubToolset |
|   Vertical |   Financial Services |

### Agent Architecture

![architecture](arch.png)

### Key Features

#### Tools

The agent has access to the following tools:

- `membership`: Registers new members.
- `claims`: Handles claims.
- `roadsideAssistance`: Provides roadside assistance, including towing services.
- `rewards`: Finds nearby reward offers from partners.

The tools are provided by custom APIs. The specifications are imported to [API hub](https://cloud.google.com/apigee/docs/apihub/what-is-api-hub) and then referenced in the agent code using ADK's built-in [ApiHubToolset](https://google.github.io/adk-docs/tools/google-cloud-tools/#apigee-api-hub-tools). This lets agent developers easily turn any existing API from their organization's API catalog into a tool with just a few lines of code. For this sample, the APIs themselves are served using [Apigee](https://cloud.google.com/apigee).

## Setup and Installation

### Prerequisites

- Python 3.12+
-   Poetry for dependency management and packaging
    -   See the official
        [Poetry website](https://python-poetry.org/docs/) for more information. To install Poetry run:
    ```bash
    pip install poetry
    ```
- Google Cloud Project

Once you have created your project, [install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install). Then run the following command to authenticate:
```bash
gcloud auth login
```
You also need to enable certain APIs. Run the following command to enable:
```bash
gcloud services enable aiplatform.googleapis.com
```

### Apigee and API hub

For this sample you must also provision Apigee and API hub in your project, to serve the APIs that act as the tools for the agent.

The API assets and additional prerequisite instructions are available in the Apigee Samples repo here: [Auto Insurance Agent APIs](https://github.com/GoogleCloudPlatform/apigee-samples/tree/main/adk-auto-insurance-agent). 

If you already have Apigee and API hub provisioned in your project, you can simply  deploy the assets by following the quickstart below.

### Quickstart: Deploy the API assets using Cloud Shell

Follow the instructions in this GCP Cloud Shell tutorial.

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.png)](https://ssh.cloud.google.com/cloudshell/open?cloudshell_git_repo=https://github.com/GoogleCloudPlatform/apigee-samples&cloudshell_git_branch=main&cloudshell_workspace=.&cloudshell_tutorial=adk-auto-insurance-agent/docs/cloudshell-tutorial.md)


## Agent Setup

1.  Clone the repository:

    ```bash
    git clone https://github.com/google/adk-samples.git
    cd python/agents/auto-insurance-agent
    ```

    For the rest of this tutorial **ensure you remain in the `python/agents/auto-insurance-agent` directory**.

2.  Install the dependencies:

    ```bash
    poetry install
    ```

3.  Configure settings:

    - Set the following environment variables. You can set them in your `.env` file (modify and rename `.env.example` file to `.env`) or set them directly in your shell. For example:

    ```bash
    export GOOGLE_GENAI_USE_VERTEXAI=1
    export GOOGLE_CLOUD_PROJECT=my-project
    export GOOGLE_CLOUD_LOCATION=my-region
    export GOOGLE_CLOUD_STORAGE_BUCKET=my-storage-bucket  # Only required for deployment on Agent Engine
    ```

## Running the Agent Locally

You can run the agent locally using the `adk` command in your terminal:

1.  To run the agent from the CLI:

    ```bash
    adk run auto_insurance_agent
    ```

2.  To run the agent from the ADK web UI:

    ```bash
    adk web
    ```
    Then select the `auto-insurance-agent` from the dropdown.

## Deploying the Agent Remotely

### To Agent Engine

The agent can also be deployed to [Vertex AI Agent Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview) using the following
commands:

```bash
poetry install --with deployment
python3 deployment/deploy.py
```

When the deployment finishes, it will output the resource ID of the remote agent deployment, for example:
```
Created remote agent: projects/<PROJECT_NUMBER>/locations/<PROJECT_LOCATION>/reasoningEngines/<AGENT_ENGINE_ID>
```

For more information on deploying to Agent Engine, see [here](https://google.github.io/adk-docs/deploy/agent-engine/#install-vertex-ai-sdk).

To test the remote agent, first set the following variable using the ID returned above e.g.
```bash
export AGENT_ENGINE_ID=projects/<PROJECT_NUMBER>/locations/<PROJECT_LOCATION>/reasoningEngines/<AGENT_ENGINE_ID>
```
then run:
```bash
python3 deployment/test_deployment.py
```

You may then interact with the deployed agent from the shell. You can type `quit` at any point to exit.

## Example Interaction

Here's a quick example of how a user might interact with the agent:

**User:**
>hi

**Agent:**
>Welcome to Cymbal Auto Insurance! How can I help you today? Do you have a member ID?

**User:**
>37184324

**Agent:**
>Thanks, Tom! What can I help you with today?

**User:**
>My car broke down and I need help

**Agent:**
>What type of help do you need? Do you need a tow, a jump start, fuel, a tire change, or help unlocking your vehicle? Also, can you provide me with your location? You can give me an address or an approximate location by cross street.

**User:**
>I ran out of gas. I'm on the shoulder of eastbound I-70 near the Kipling exit

**Agent:**
>Okay, I'm creating a request for you now. I have found a company nearby who can help. They estimate they will be there in about 45 minutes. The company is called Roadside Rescue. They will give you a call shortly. Is there anything else I can help you with?

**User:**
>No that's all thanks

**Agent:**
>Thank you for contacting Cymbal Auto Insurance.