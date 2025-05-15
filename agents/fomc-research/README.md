#  FOMC Research Agent

The FOMC Research Agent uses a multi-agent, multi-modal architecture, combined
with tool use, live web access and external database integration to generate a
detailed analysis report on the latest meeting of the Federal Open Market
Committee. This agent showcases a multi-stage, non-conversational agentic
workflow as opposed to a conversational user interaction.

## Overview

The Federal Open Market Committee (FOMC) is the body of the United States
government responsible for setting interest rate policy. Statements and
press releases from the FOMC meetings are closely watched and thoroughly
analyzed by financial market participants around the world.

This agent shows how a multi-agent architecture might be used to generate
detailed analysis reports on financial market events such as Fed meetings. The
FOMC Research Agent is slightly different from other agents in that it is
largely non-conversational -- most of the agent's work takes place through
back-and-forth interactions between individual sub-agents. When necessary,
it asks the user for a key piece of information, but in general it functions
without human interaction.

This is the high-level workflow the agent follows to generate its analysis (note
that step 3, "Review press conference video", is still in development).
![FOMC Research agent workflow](<FOMC_Research_Agent_Workflow.png>)

## Agent Details
The key features of the FOMC Research Agent include:

| Feature | Description |
| --- | --- |
| *Interaction Type* | Workflow |
| *Complexity* | Advanced |
| *Agent Type* | Multi Agent |
| *Components* | Tools, Multimodal, AgentTools |
| *Vertical* | Financial Services |

### Agent Architecture

This diagram shows the detailed architecture of the agents and tools used
to implement this workflow.
![FOMC Research agent architecture](<fomc-research.svg>)

### Key Features

##### Agents
* **root_agent:** Entry point for the agent workflow. Coordinates the activity of the other agents.
* **research_agent:** Coordinates the retrieval of individual research components.
* **analysis_agent:** Takes in the output of the `research_agent` and generates the analysis report.
* **retrieve_meeting_data_agent:** Fetches FOMC meeting data from the web.
* **extract_page_data_agent:** Extracts specific data from an HTML page.
* **summarize_meeting_agent:** Reads the meeting transcript and generates a summary.

##### Tools
* **fetch_page_tool**: Encapsulates an HTTP request for retrieving a web page.
* **store_state_tool**: Stores specific information in the ToolContext.
* **analyze_video_tool**: Processes and analyzes a YouTube video.
* **compute_probability_tool**: Computes the probability of rate changes from Fed Futures pricing.
* **compare_statements**: Compares the current and previous FOMC statements.
* **fetch_transcript**: Retrieves the FOMC meeting transcript.

##### Callbacks
* **rate_limit_callback**: Implements request rate limiting to minimize `429: Resource Exhausted` errors.

## Setup and Installation
1.  **Prerequisites:**

    **Google Cloud SDK and GCP Project:**

    For the BigQuery setup and the Agent Engine deployment steps, you will need
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
    gcloud services enable bigquery.googleapis.com
    ```

2.  **Installation:**

    Clone this repository and change to the repo directory:
    ```
    git clone https://github.com/google/adk-samples.git
    cd adk-samples/agents/fomc-research
    ```

    Install [Poetry](https://python-poetry.org)

    If you have not installed poetry before, you can do so by running:
    ```bash
    pip install poetry
    ```

    Install the FOMC Research agent requirements:
    ```bash
    poetry install
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

    **BigQuery Setup:**

    You need to create a BigQuery table containing the Fed Futures pricing data.

    The FOMC Research Agent repo contains a sample data file
    (`sample_timeseries_data.csv`) with data covering the FOMC meetings on Jan
    29 and Mar 19 2025. If you want to run the agent for other FOMC meetings you
    will need to get additional data.

    To install this data file in a BigQuery table in your project, run the following
    commands in the `fomc-research/deployment` directory:
    ```bash
    python bigquery_setup.py --project_id=$GOOGLE_CLOUD_PROJECT \
        --dataset_id=$GOOGLE_CLOUD_BQ_DATASET \
        --location=$GOOGLE_CLOUD_LOCATION \
        --data_file=sample_timeseries_data.csv
    ```

## Running the Agent

**Using the ADK command line:**

From the `fomc-research` directory, run this command:
```bash
adk run fomc_research
```
The initial output will include a command you can use to tail the agent log
file. The command will be something like this:
```bash
tail -F /tmp/agents_log/agent.latest.log
```

**Using the ADK Dev UI:**

From the `fomc-research` directory, run this command:
```bash
adk web .
```
It will display a URL for the demo UI. Point your browser to that URL.

The UI will be blank initially. In the dropdown at the top left, choose `fomc_research`
to load the agent.

The logs from the agent will display on the console in real time as it runs. However,
if you want to store a log of the interaction and also tail the interaction in real
time, use the following commands:

```bash
adk web . > fomc_research_log.txt 2>&1 &
tail -f fomc_research_log.txt
```

### Example Interaction

Begin the interaction by typing "Hello. What can you do for me?". After
the first prompt, give the date: "2025-01-29".

The interaction will look something like this:
```
$ adk run .
Log setup complete: /tmp/agents_log/agent.20250405_140937.log
To access latest log: tail -F /tmp/agents_log/agent.latest.log
Running agent root_agent, type exit to exit.
user: Hello. What can you do for me?
[root_agent]: I can help you analyze past Fed Open Market Committee (FOMC) meetings and provide you with a thorough analysis report. To start, please provide the date of the meeting you would like to analyze. If you have already provided it, please confirm the date. I need the date in ISO format (YYYY-MM-DD).

user: 2025-01-29
[analysis_agent]: Here is a summary and analysis of the January 29, 2025 FOMC meeting, based on the available information:
...
```
If the agent stops before completing the analysis, try asking it to continue.

## Deployment on Vertex AI Agent Engine

To deploy the agent to Google Agent Engine, first follow
[these steps](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/set-up)
to set up your Google Cloud project for Agent Engine.

You also need to give BigQuery User and BigQuery Data Viewer permissions to the
Reasoning Engine Service Agent. Run the following commands to grant the required
permissions:
```bash
export RE_SA="service-${GOOGLE_CLOUD_PROJECT_NUMBER}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT} \
    --member="serviceAccount:${RE_SA}" \
    --condition=None \
    --role="roles/bigquery.user"
gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT} \
    --member="serviceAccount:${RE_SA}" \
    --condition=None \
    --role="roles/bigquery.dataViewer"
```
Next, you need to create a `.whl` file for your agent. From the `fomc-research`
directory, run this command:
```bash
poetry build --format=wheel --output=deployment
```
This will create a file named `fomc_research-0.1-py3-none-any.whl` in the
`deployment` directory.

Then run the following command:
```bash
cd deployment
python3 deploy.py --create
```
When this command returns, if it succeeds it will print an AgentEngine resource
name that looks something like this:
```
projects/************/locations/us-central1/reasoningEngines/7737333693403889664
```
The last sequence of digits is the AgentEngine resource ID.

Once you have successfully deployed your agent, you can interact with it
using the `test_deployment.py` script in the `deployment` directory. Store the
agent's resource ID in an environment variable and run the following command:
```bash
export RESOURCE_ID=...
export USER_ID=<any string>
python test_deployment.py --resource_id=$RESOURCE_ID --user_id=$USER_ID
```
The session will look something like this:
```
Found agent with resource ID: ...
Created session for user ID: ...
Type 'quit' to exit.
Input: Hello. What can you do for me?
Response: I can create an analysis report on FOMC meetings. To start, please provide the date of the meeting you want to analyze. I need the date in YYYY-MM-DD format.

Input: 2025-01-29
Response: I have stored the date you provided. Now I will retrieve the meeting data.
...
```
Note that this is *not* a full-featured, production-ready CLI; it is just intended to
show how to use the Agent Engine API to interact with a deployed agent.

The main part of the `test_deploy.py` script is approximately this code:

```python
from vertexai import agent_engines
remote_agent = vertexai.agent_engines.get(RESOURCE_ID)
session = remote_agent.create_session(user_id=USER_ID)
while True:
    user_input = input("Input: ")
    if user_input == "quit":
      break

    for event in remote_agent.stream_query(
        user_id=USER_ID,
        session_id=session["id"],
        message=user_input,
    ):
        parts = event["content"]["parts"]
        for part in parts:
            if "text" in part:
                text_part = part["text"]
                print(f"Response: {text_part}")
```

To delete the agent, run the following command (using the resource ID returned previously):
```bash
python3 deployment/deploy.py --delete --resource_id=$RESOURCE_ID
```

## Troubleshooting

### "Malformed function call"

Occasionally the agent returns the error "Malformed function call". This is a
Gemini model error which should be addressed in future model versions. Simply
restart the UI and the agent will reset.

### Agent stops mid-workflow

Sometimes the agent will stop mid-workflow, after completing one of the
intermediate steps. When this happens, it frequently works just to tell the agent
to continue, or another instruction to continue its operation.


## Disclaimer

This agent sample is provided for illustrative purposes only and is not intended for production use. It serves as a basic example of an agent and a foundational starting point for individuals or teams to develop their own agents.

This sample has not been rigorously tested, may contain bugs or limitations, and does not include features or optimizations typically required for a production environment (e.g., robust error handling, security measures, scalability, performance considerations, comprehensive logging, or advanced configuration options).

Users are solely responsible for any further development, testing, security hardening, and deployment of agents based on this sample. We recommend thorough review, testing, and the implementation of appropriate safeguards before using any derived agent in a live or critical system.

## Disclaimer

This agent sample is provided for illustrative purposes only and is not intended for production use. It serves as a basic example of an agent and a foundational starting point for individuals or teams to develop their own agents.

This sample has not been rigorously tested, may contain bugs or limitations, and does not include features or optimizations typically required for a production environment (e.g., robust error handling, security measures, scalability, performance considerations, comprehensive logging, or advanced configuration options).

Users are solely responsible for any further development, testing, security hardening, and deployment of agents based on this sample. We recommend thorough review, testing, and the implementation of appropriate safeguards before using any derived agent in a live or critical system.