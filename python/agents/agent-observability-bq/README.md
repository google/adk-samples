# Agent Observability (BigQuery)

This sample demonstrates how to build an agent that interacts directly with Google BigQuery, while also leveraging the `BigQueryAgentAnalyticsPlugin` for observability and logging. 

## Features

- **BigQuery Tools**: The agent is equipped with native tools (e.g., `execute_sql`, `list_dataset_ids`, `get_table_info`, `ask_data_insights`) that allow it to explore available metadata, gain data insights, and execute SQL queries against BigQuery datasets.
- **Analytics Plugin**: Every interaction, tool call, and state transition is centrally logged to a BigQuery dataset (`agent_analytics`) for analysis, debugging, and auditing using the built-in ADK plugin.

## Prerequisites

1.  A Google Cloud Project with the BigQuery API enabled.
2.  Application Default Credentials configured (e.g., via `gcloud auth application-default login`).
3.  Python >3.10.

## Setup

1.  **Environment Variables**:
    Create a `.env` file in the root of the application based on the `.env.example`:
    ```bash
    cp .env.example .env
    ```
    Then, update `.env` with your actual Google Cloud Project ID and desired location.
    
    *Optional:* You can also configure the `BQ_ANALYTICS_DATASET_ID` variable if you wish to write logs to a dataset other than the default (`adk_agent_analytics`).

2.  **Provision the Analytics Dataset**:
    The ADK analytics plugin will automatically provision the underlying logging tables, but the dataset itself must be created first. Run the following command (substituting your dataset name and location if you changed them):
    ```bash
    bq mk --location=us-east1 --dataset "your-project-id:adk_agent_analytics"
    ```

3.  **Dependencies**:
    Ensure `uv` is installed, then run the application using the starter pack CLI or direct ADK commands.

## Usage

### Using the Starter Pack CLI

To test this agent interactively in a terminal UI:

```bash
adk run
```

Or, to launch the Web UI:

```bash
adk web
```

### Try it out

Ask the agent a question like:
> "List the datasets available in my project."
> "Write and run a query to get the top 10 most popular programming languages on GitHub using `bigquery-public-data.github_repos.languages`."

As you interact with the agent, notice that a dataset named `agent_analytics` is automatically created (if it didn't exist) in your project, populated with detailed agent execution logs.
