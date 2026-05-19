# Incident Management Agent using Integration Connector

## Overview

This agent sample showcases the utilization of dynamic identity propagation with ServiceNow and [Application Integration Connectors](https://cloud.google.com/application-integration/docs/using-integration-connectors), allowing your end user credentials to be passed at runtime to an underlying connection action (in this case, ServiceNow Incidents). This framework removes the previous requirement of hardcoded service account credentials, as your end users token/identity is now validated on a per request basis (typically recommended from a security perspective)

## Agent Details

| Attribute | Detail |
|---|---|
|   Interaction Type |   Conversational |
|   Complexity |   Easy |
|   Agent Type |   Single Agent |
|   Components |   Tools, ApplicationIntegrationToolset |
|   Vertical |   Customer Support |

### Agent Architecture

![architecture](arch.png)

### Key Features

#### Tools

The agent will utilize the following tools:
- Snow_connector_tool: Dynamically performs CRUD operations on the ServiceNow Incidents Table given end user credentials

This tool was created through a [ServiceNow Integration Connector](https://cloud.google.com/integration-connectors/docs/connectors/servicenow/configure) and is referenced in the agent code using the ADK’s built in [ApplicationIntegrationToolset](https://cloud.google.com/integration-connectors/docs/connectors/servicenow/configure). This pattern/workflow lets an agent developer turn a [connection for 150+ first/third party services](https://cloud.google.com/integration-connectors/docs/all-integration-connectors) into a tool through a few lines of code!

## Setup and Installation

### Prerequisites

- Python 3.12+
- uv for dependency management and packaging
    - See the official [uv website](https://docs.astral.sh/uv/) for more information. To install uv run:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
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
You also need to enable certain APIs. Run the following command to enable:
```bash
export PROJECT_ID=<project_id>
gcloud services enable aiplatform.googleapis.com compute.googleapis.com --project "$PROJECT_ID"
```

### Application Integration and Integration Connector Setup

For this sample you must also provision Application Integration and Integration Connectors in your project.

The assets and additional prerequisite instructions are available in the Application Integration Samples repo here: [Incident Management](https://github.com/GoogleCloudPlatform/application-integration-samples/tree/main/src/adk-incident-management).

If you already have Application Integration and Integration Connectors provisioned in your project, you can simply deploy the assets by following the quickstart below.

### Quickstart: Deploy the Application Integration assets using Cloud Shell

Follow the instructions in this GCP Cloud Shell tutorial.

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.png)](https://ssh.cloud.google.com/cloudshell/open?cloudshell_git_repo=https://github.com/GoogleCloudPlatform/application-integration-samples&cloudshell_git_branch=main&cloudshell_workspace=.&cloudshell_tutorial=src/adk-incident-management/docs/cloudshell-tutorial.md)

## Agent Setup

1.  Clone the repository:

    ```bash
    git clone https://github.com/google/adk-samples.git
    cd python/agents/incident-management
    ```

    For the rest of this tutorial **ensure you remain in the `python/agents/incident-management` directory**.

2.  Install the dependencies:

    ```bash
    uv sync --dev
    ```

3.  Configure settings:

    - Set the following environment variables. You can set them in your `.env` file (modify and rename `.env.example` file to `.env`) or set them directly in your shell.

### Alternative: Using Google Agents CLI

You can also use the [Google Agents CLI](https://github.com/google/agents-cli) to create a production-ready version of this agent with additional deployment options.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-incident-management` with your project name):

```bash
agents-cli create my-incident-management -a adk@incident-management
```

The Google Agents CLI will prompt you to select deployment options and provides additional production-ready features including automated CI/CD deployment scripts.

## Running the Agent Locally

You can run the agent locally using the `adk` command in your terminal:

NOTE: Modify your ServiceNow application `Redirect URL` to point to your local host http://localhost:8080/dev-ui/ (including the trailing "/")

To run the agent from the ADK web UI:

```bash
adk web
```
Then select the `incident-management` from the dropdown.

## Example Interaction

Here's a quick example of how a user might interact with the agent:

**User:**
>hi

**Agent:**
>Hello! I'm a ServiceNow agent and can help you with getting, listing, and creating incidents. How can I help you today?

**User:**
>I would like to create an incident, my laptop has crashed multiple times today and I need help. This is urgent and should be addressed ASAP

**Agent:**
>I can help with that. I've summarized the incident details below:
>
>Short Description: Laptop crashing multiple times
>Description: My laptop has crashed multiple times today and I need help. This is urgent and should be addressed ASAP.
>
>Impact: 1 - High
>
>Urgency: 1 - High

Do you want me to proceed with creating the incident?

**User:**
>Yes

**Agent:**
>Sign in/grant consent via pop up prompt

***Note: To showcase the dynamic identity in action, click on the top right section of the pop up and say “Not You” for system administrator. Instead, sign in using the user id/password created in the previous step***

**Agent:**
>I have created the incident for you. The incident ID is xxx. A support agent will be in touch with you shortly. Can I help you with anything else?

**User:**
>Yes thank you! Can you provide more details about Incident ID xxx?

**Agent:**
>Of course! Here are more details for Incident ID xxx
>
>Incident Number: xxxxx
>
>Description: My laptop has crashed multiple times today and I need help. This is urgent and should be addressed ASAP.
>
>Ticket Creator: **dynamic user identity**
>
>Time of Creation: xxxxx
