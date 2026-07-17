# Google Trends Agent

## Overview

The Google Trends Agent is an AI agent designed to surface the newest Google Trends in real-time. It can identify emerging topics, analyze their velocity, and provide insights into what is currently capturing the world's attention. This is useful for content creators, marketers, and analysts who need to stay ahead of the curve. For example, a marketer could use this agent as part of their workflow to design a marketing campaign based on a trend in a specific region or city that relates to the product or service they promote.

## Disclaimer

This agent has several important limitations to be aware of:

- **Dataset Constraints**: The agent can only access data that exists in the [public Google Trends BigQuery dataset](https://support.google.com/trends/answer/12764470?hl=en), which contains only the top trending terms by region and time period.
- **No Open-ended Searches**: You cannot ask for trends on specific topics (e.g., "trending terms about AI agents") if they aren't already ranked as top terms in the dataset.
- **Regional Limitations**: Only regions included in the BigQuery dataset are available.

## Agent Architecture

This diagram shows the detailed architecture of the agents and tools used to implement this workflow.
<img src="architecture/google_trends_architecture.png" alt="Google Trends Agent Architecture" width="800"/>

## Agent Details

This agent is a sequential agent composed of two sub-agents that work together to fetch and analyze Google Trends data. The first sub-agent, `TrendsQueryGeneratorAgent`, generates a BigQuery SQL query from the user's request. The second sub-agent, `TrendsQueryExecutorAgent`, executes this query to retrieve the trends data and present it to the user. This modular design can be extended with more complex, multi-agent workflows.

| Feature              | Description           |
| -------------------- | --------------------- |
| **Interaction Type** | Conversational        |
| **Complexity**       | Medium                |
| **Agent Type**       | Sequential Agent      |
| **Components**       | Tools: BigQuery       |
| **Vertical**         | Marketing & Analytics |

- **Core Logic:** The agent's main logic is defined in `google_trends_agent/agent.py`.
- **Tools:** It utilizes tools to query the public [Google Trends dataset on BigQuery](https://support.google.com/trends/answer/12764470?hl=en) to fetch trending data. You can explore querying the [Google Trends dataset](https://console.cloud.google.com/marketplace/product/bigquery-public-datasets/google-search-trends) in Google Cloud Console.
- **Dependencies:** Key dependencies include `google-cloud-aiplatform` for the ADK and agent engine deployment, `google-cloud-bigquery` and `pandas` for data handling.

## Setup and Installation

1.  **Prerequisites**

    - Python 3.11+
    - Install [uv](https://docs.astral.sh/uv/)
    - A Google Cloud Platform project.
    - The [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).

2.  **Installation**

    ```bash
    # Navigate to the agent's directory
    cd adk-samples/python/agents/google-trends-agent
    # Install the package and dependencies.
    uv sync --dev
    ```

3.  **Configuration**

    - Set up your Google Cloud credentials. You can set these in your shell or create a `.env` file in the agent's root directory (`google-trends-agent/`).

      ```bash
      # Authenticate your gcloud account
      gcloud auth application-default login
      gcloud auth application-default set-quota-project <your-project-id>

      # Create and populate .env file
      cp .env.example .env
      ```

    - Edit the `.env` file with your specific configuration:

      ```env
      GOOGLE_CLOUD_PROJECT="<your-project-id>"
      GOOGLE_CLOUD_LOCATION="<your-project-location>"
      GOOGLE_CLOUD_STORAGE_BUCKET="<your-storage-bucket>" # Required for deployment
      ```

    - Grant the Agent Runtime service account permission to run BigQuery jobs. This is required for deployment.

      ```bash
      # Set your project ID
      export PROJECT_ID="<your-project-id>"

      # Get your project number
      export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

      # Grant the Agent Engine service account permission to run BigQuery jobs
      gcloud projects add-iam-policy-binding ${PROJECT_ID} \
          --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-aiplatform-re.iam.gserviceaccount.com" \
          --role="roles/bigquery.user"
      ```

## Running the Agent Locally

You can run the agent locally using the `adk` command in your terminal.

1.  **To run the agent from the CLI:**

    Make sure your working directory is `adk-samples/python/agents/google-trends-agent/google_trends_agent`, then run:

    ```bash
    cd google_trends_agent
    adk run .
    ```

2.  **To run the agent from the ADK web UI:**

    From the `google_trends_agent` directory:

    ```bash
    adk web .
    ```
    Then select `google-trends-agent` from the dropdown menu.

## Deploying the Agent Remotely

### To Agent Runtime

The agent can also be deployed to [Agent Runtime](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime).

1.  **Ensure Prerequisites:** Make sure your `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `GOOGLE_CLOUD_STORAGE_BUCKET` environment variables are set correctly in your `.env` file.

2.  **Run the deployment script:**
    ```bash
    uv sync --group deployment
    uv run python deployment/deploy.py
    ```
    When the deployment finishes, it will output the resource ID of the remote agent deployment and update your `.env` file with the `AGENT_ENGINE_ID`. For example:
    ```
    Created remote agent: projects/<PROJECT_NUMBER>/locations/<PROJECT_LOCATION>/reasoningEngines/<AGENT_ENGINE_ID>
    ```
3.  **Test the remote agent:**
    Once deployed, you can interact with the remote agent by running:
    ```bash
    python deployment/test_deployment.py
    ```
    You can type `quit` at any point to exit.

### Example Interaction

**User:** List the top 10 terms in Canada during the past 3 weeks

**[TrendsQueryGeneratorAgent]:**

```sql
SELECT
  term,
  rank,
  week
FROM
  `bigquery-public-data.google_trends.international_top_terms`
WHERE
  refresh_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
  AND country_name = 'Canada'
  AND week IN (
    SELECT DISTINCT
      week
    FROM
      `bigquery-public-data.google_trends.international_top_terms`
    WHERE
      refresh_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
      AND country_name = 'Canada'
    ORDER BY
      week DESC
    LIMIT 3
  )
ORDER BY
  week DESC,
  rank
LIMIT 100
```

**[TrendsQueryExecutorAgent]:**
The top terms in Canada for the week of 2025-07-13 are:

1.  usyk vs dubois
2.  man united vs leeds united
3.  rashford
4.  election loser nyt crossword answers
5.  reading vs tottenham
6.  blake wheeler
7.  france vs germany
8.  усик дюбуа

## Customization

You can customize this agent to fit your specific needs:

- **Change Data Source:** Modify the agent's tools to pull trend data from different sources, such as social media APIs or other analytics platforms.
- **Enhance Analysis:** Add new tools to perform more in-depth analysis on the trends, deeper research (e.g. using Google Search as an additional tool) such as sentiment analysis or forecasting.
- **Add Notifications:** Integrate tools that send alerts via email or Slack when a new trend matching specific criteria is detected.

### Google Agents CLI (Recommended)

Use the [Google Agents CLI](https://github.com/google/agents-cli) to create a production-ready version of this agent with additional deployment options.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-google-trends-agent` with your project name):

```bash
agents-cli create my-google-trends-agent -a adk@google-trends
```

The Google Agents CLI will prompt you to select deployment options and provides additional production-ready features including automated CI/CD deployment scripts.
