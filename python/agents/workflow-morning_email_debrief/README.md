# Timed Trigger Workflow Agent

This sample demonstrates a workflow that can be triggered on a schedule, for example, to run a task at a specific time every day.

## 1. Architecture

The architecture of this sample is a `WorkflowAgent` that retrieves and summarizes emails.

- **`retrieve_and_list_emails`**: A Python function that uses the Gmail API to fetch the latest emails from a user's inbox.
- **`format_news_agent`**: An `LlmAgent` that takes the fetched emails and summarizes them into bullet points.

The workflow is defined as a simple sequence:

```python
edges=[
    ("START", retrieve_and_list_emails, format_news_agent),
]
```

## 2. Feature: Timed Triggers

The key feature demonstrated here is the ability to deploy a workflow as an endpoint that can be triggered by an external scheduler, like Google Cloud Scheduler. This allows for automated, time-based execution of workflows. The sample is designed to be deployed to Agent Runtime, and then invoked via an HTTP request, which can be automated.

This is useful for tasks such as:
- Daily reports
- Morning briefings
- Data fetching and processing on a schedule

## 3. Deployment Guide

This guide provides step-by-step instructions to deploy your workflow agent to Agent Runtime and trigger it with Cloud Scheduler.

### Prerequisites

Ensure you have authenticated with Google Cloud:
```sh
gcloud auth application-default login
```

Your GCP `project` and `location` should be set in a `.env` file in the root of this project. The agent is already configured to load these values.

### Deployment Steps

1.  **Deploy the Agent using ADK CLI**

    From your terminal, run the following command to deploy your agent. This command will package your agent and deploy it to Agent Runtime.

    ```sh
    adk deploy workflow-triggers/agent.py:root_agent --display-name "Morning Email Agent"
    ```

    This process can take a few minutes. Upon completion, your agent will be deployed and ready to be triggered.

2.  **Triggering the Agent**

    Once deployed, you can trigger your agent. For a timed trigger, you would typically use a scheduler service like Cloud Scheduler to invoke the agent's endpoint at your desired time (e.g., every morning at 9:00 am).

### Setting up a Timed Trigger with Cloud Scheduler

To automate the execution of your deployed agent at a specific time (e.g., 9 AM every morning), you can leverage Google Cloud Scheduler.

#### Prerequisites for Cloud Scheduler

*   Your agent must already be deployed to Agent Runtime.
*   You need the HTTPS endpoint URL of your deployed agent. This URL is provided in the `adk deploy` command output or can be found in the Google Cloud Console under Vertex AI > Agent Runtime.
*   Ensure you have the necessary `gcloud` CLI components installed and authenticated.

#### Cloud Scheduler Configuration Steps

1.  **Create a Service Account for the Scheduler**

    It's a security best practice for Cloud Scheduler to use a dedicated service account to invoke your agent.
    ```sh
    gcloud iam service-accounts create scheduler-invoker --display-name "Scheduler Invoker"
    ```

2.  **Grant Invoker Permissions to the Service Account**

    Grant the service account permission to invoke your Agent Runtime service. Replace `[YOUR_AGENT_DISPLAY_NAME]` with the actual display name you used during deployment (e.g., "Morning Email Agent"), and `[YOUR_GCP_LOCATION]` with the GCP location where your agent is deployed.
    ```sh
    gcloud run services add-iam-policy-binding [YOUR_AGENT_DISPLAY_NAME] 
        --member="serviceAccount:scheduler-invoker@$(gcloud config get-value project).iam.gserviceaccount.com" 
        --role="roles/run.invoker" 
        --region="[YOUR_GCP_LOCATION]"
    ```

3.  **Create the Cloud Scheduler Job**

    This command creates the Cloud Scheduler job that will trigger your agent daily at 9:00 AM (0 9 * * *). Remember to replace `[YOUR_AGENT_ENDPOINT_URL]` with the actual HTTPS endpoint of your deployed agent and `[YOUR_GCP_LOCATION]` with your agent's deployment location.

    ```sh
    gcloud scheduler jobs create http daily-email-agent-trigger 
        --schedule="0 9 * * *" 
        --uri="[YOUR_AGENT_ENDPOINT_URL]" 
        --http-method="POST" 
        --oidc-service-account-email="scheduler-invoker@$(gcloud config get-value project).iam.gserviceaccount.com" 
        --oidc-token-audience="[YOUR_AGENT_ENDPOINT_URL]" 
        --location="[YOUR_GCP_LOCATION]"
    ```

After completing these steps, your Cloud Scheduler job will be active and will trigger your "Morning Email Agent" every day at 9:00 AM. You can manage and monitor this job from the Cloud Scheduler page in the Google Cloud Console.
