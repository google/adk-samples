# Order Processing Agent with human in the loop workflow using Application Integration

## Overview

This agent sample showcases the art of the possible using the ADK and [Application Integration](https://cloud.google.com/application-integration/docs/overview), Google Cloud’s low-to-no code integration platform for building Agentic workflows. Application Integration will be used in this sample to orchestrate a deterministic set of operations, including the following:
- Order processing/storage via BigQuery
- Orchestrating email confirmations once the order data has been written to BigQuery
- Invoking a “Human in the Loop” approval pattern if the order quantity exceeds 100


## Agent Details

| Attribute | Detail |
|---|---|
|   Interaction Type |   Conversational |
|   Complexity |   Easy |
|   Agent Type |   Single Agent |
|   Components |   Tools, ApplicationIntegrationToolset |
|   Vertical |   Order Management |

### Agent Architecture

![architecture](arch.png)

### Key Features

#### Tools

The agent will utilize the following tools:
- order_processing_tool: Deterministically processes order details depending on the size of the order, etc

This tool was created through an [Application Integration](https://cloud.google.com/application-integration/docs/overview) process and is referenced in the agent code using the ADK’s built in [ApplicationIntegrationToolset](https://google.github.io/adk-docs/tools/google-cloud-tools/#create-an-api-hub-toolset). This pattern/workflow lets an agent developer turn a deterministic set of operations into a tool through a few lines of code!

## Setup and Installation

### Prerequisites

- Python 3.12+
-   Poetry for dependency management and packaging
    -   See the official
        [Poetry website](https://python-poetry.org/docs/) for more information. To install Poetry run:
    ```bash
    pip install poetry
    ```
- Google Cloud Project with the following roles assigned
  - Application Integration Admin
  - Connector Admin
  - Secret Manager Admin
  - Storage Admin
  - Service Usage Consumer
  - Logs Viewer

Once you have created your project, [install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install). Then run the following command to authenticate:
```bash
gcloud auth login
```
You need to enable certain Google APIs. Run the following command to enable:
```bash
export PROJECT_ID=<project_id>
gcloud services enable integrations.googleapis.com connectors.googleapis.com secretmanager.googleapis.com aiplatform.googleapis.com compute.googleapis.com --project "$PROJECT_ID"
```

### Application Integration and Integration Connector Setup

For this sample you must also provision Application Integration and Integration Connectors in your project.

The assets and additional prerequisite instructions are available in the Application Integration Samples repo here: [Order Processing](https://github.com/GoogleCloudPlatform/application-integration-samples/tree/main/src/adk-order-processing).

If you already have Application Integration and Integration Connectors provisioned in your project, you can simply deploy the assets by following the quickstart below.

### Quickstart: Deploy the Application Integration assets using Cloud Shell

Follow the instructions in this GCP Cloud Shell tutorial.

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.png)](https://ssh.cloud.google.com/cloudshell/open?cloudshell_git_repo=https://github.com/GoogleCloudPlatform/application-integration-samples&cloudshell_git_branch=main&cloudshell_workspace=.&cloudshell_tutorial=src/adk-order-processing/docs/cloudshell-tutorial.md)

## Agent Setup

1.  Clone the repository:

    ```bash
    git clone https://github.com/google/adk-samples.git
    cd python/agents/order-processing
    ```

    For the rest of this tutorial **ensure you remain in the `python/agents/order-processing` directory**.

2.  Install the dependencies:

    **Note for Linux users:** If you get an error related to `keyring` during the installation, you can disable it by running the following command:
    ```bash
    poetry config keyring.enabled false
    ```
    This is a one-time setup.

    ```bash
    poetry install
    ```

3.  Configure settings:

    - Set the following environment variables. You can set them in your `.env` file (modify and rename `.env.example` file to `.env`) or set them directly in your shell.

## Running the Agent Locally

You can run the agent locally using the `adk` command in your terminal:

To run the agent from the ADK web UI:

```bash
adk web
```
Then select the `order-processing` from the dropdown.

## Deploying the Agent Remotely

### To Agent Runtime

Create a service account and assign it `Vertex AI User` and `Application Integration Invoker` roles

```bash
PROJECT_ID=<project_id>

gcloud iam service-accounts create "adk-order-processing" --project "$PROJECT_ID"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:adk-order-processing@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:adk-order-processing@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/integrations.integrationEditor"

```

Add the service account email in the `.env` file

The agent can also be deployed to [Agent Runtime](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime) using the following
commands:

```bash
poetry install --with deployment
python3 deployment/deploy.py
```

When the deployment finishes, it will output the resource ID of the remote agent deployment, for example:
```
Created remote agent: projects/<PROJECT_NUMBER>/locations/<PROJECT_LOCATION>/reasoningEngines/<AGENT_ENGINE_ID>
```

For more information on deploying to Agent Runtime, see [here](https://google.github.io/adk-docs/deploy/agent-engine/#install-vertex-ai-sdk).

The deployment script adds the `AGENT_ENGINE_ID` to your `.env` file. To test the remote agent, simply run:
```bash
python3 deployment/test_deployment.py
```

You may then interact with the deployed agent from the shell. You can type `quit` at any point to exit.

## Example Interaction

Here's a quick example of how a user might interact with the agent:

**User:**
>Hello

**Agent:**
>Hello! How can I assist you today? I can help with creating orders.

**User:**
>I would like to order 5 Google Pixel phones please

**Agent:**
>No problem! To finalize your order for 5 Google Pixel phones, I need a little more information. Could you please provide:
>
>Your Name:
>
>Shipping Address:
>

**User:**
>Sure, my name is Test User and my Shipping Address is 15 Testing Drive

**Agent:**
>Great! Your order for 5 Google Pixel phones has been placed successfully. Your order will be shipped to 15 Testing Drive.

***Note: At this point an email will be sent to the previously provided email address confirming your order details (order id, quantity/type of order, etc)***

**User:**
>Can I place another order?

**Agent:**
>Yes, absolutely! What would you like to order this time? Please let me know the product, quantity, your name, and the shipping address.

**User:**
>Please use the same name and shipping address as last time, I would like to order 150 Chromebooks please

**Agent:**
>Your order for 150 Chromebooks has been placed successfully using the same name (Test User) and shipping address (15 Testing Drive). The order status is currently "In Progress".

***Note: At this point the Human in the Loop process is invoked, where the email address previously provided has to approve a process run before the order is fulfilled (as the order quantity is >100)***

***Once approved, you will receive an email address confirming the order details***
