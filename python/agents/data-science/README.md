# Data Science with Multiple Agents

## :star: :star: MAJOR UPDATE :star: :star:

Note for previous users of the Data Science Agent: this newly released version
of the Data Science Agent introduces some significant changes. Please read this
document carefully.

### Significant Changes Introduced

1. *AlloyDB data source*: The updated agent includes support for a second data
    source in AlloyDB, including an AlloyDB sub-agent.
1. *MCP Toolbox for Databases*: The AlloyDB sub-agent uses the
    [MCP Toolbox for Databases][mcp-toolbox] to connect to AlloyDB.
1. *BigQuery Built-In Tools*: The BigQuery sub-agent now uses the
    [ADK Built-in BigQuery Tool][adk-builtin-tool-bq] to connect to BigQuery.
1. *New sample dataset*: The agent now includes a new sample dataset with simulated
    flight and ticket information for a fictitional airline. The new dataset is
    designed to be hosted in both AlloyDB and BigQuery, to demonstrate the
    cross-dataset capabiliites of the agent.
1. *Dataset configuration*: The agent uses a new configuration file format allowing
    users to configure the data sources used at runtime, including using only
    BigQuery or BigQuery and AlloyDB.
1. *Cross-dataset joins*: The new configuration format also includes support for
    specifying cross-dataset key relationships, allowing the agent to perform
    cross-dataset joins.


## Overview

This project demonstrates a multi-agent system designed for sophisticated data
analysis. It integrates several specialized agents to handle different aspects
of the data pipeline, from data retrieval to advanced analytics and machine
learning. The system is built to interact with BigQuery and AlloyDB, perform complex data
manipulations, generate data visualizations and execute machine learning tasks
using BigQuery ML (BQML). The agent can generate text response as well as
visuals, including plots and graphs for data analysis and exploration.

▶️ **Watch the Video Walkthrough:** [How to build a Data Science agent with
ADK](https://www.youtube.com/watch?v=efcUXoMX818)

## Agent Details
The key features of the Data Science Multi-Agent include:

| Feature | Description |
| --- | --- |
| **Interaction Type:** | Conversational |
| **Complexity:**  | Advanced |
| **Agent Type:**  | Multi Agent |
| **Components:**  | Tools, AgentTools, Session Memory, RAG, MCP Toolkit for Databases, ADK Builtin BigQuery Tools |
| **Vertical:**  | All (Applicable across industries needing advanced data analysis) |


### Architecture
![Data Science Architecture](data-science-architecture.svg)

### Key Features

*   **Multi-Agent Architecture:** Utilizes a top-level agent that orchestrates
    sub-agents, each specialized in a specific task.
*   **Database Interaction (NL2SQL):** Employs a Database Agent to interact with
    BigQuery and AlloyDB using natural language queries, translating them into SQL.
*   **Data Science Analysis (NL2Py):** Includes a Data Science Agent that
    performs data analysis and visualization using Python, based on natural
    language instructions.
*   **Machine Learning (BQML):** Features a BQML Agent that leverages BigQuery
    ML for training and evaluating machine learning models.
*   **Code Interpreter Integration:** Supports the use of a Code Interpreter
    extension in Vertex AI for executing Python code, enabling complex data
    analysis and manipulation.
*   **ADK Web GUI:** Offers a user-friendly GUI interface for interacting with
    the agents.
*   **Testability:** Includes a comprehensive test suite for ensuring the
    reliability of the agents.


## Agent Setup and Installation

### Prerequisites

*   **Google Cloud Account:** You need a Google Cloud account with BigQuery
    enabled.
*   **Python 3.12+:** Ensure you have Python 3.12 or a later version installed.
*   **uv:** Install uv by following the instructions on the official uv website:
    [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
*   **Git:** Ensure you have git installed. If not, you can download it from
    [https://git-scm.com/](https://git-scm.com/) and follow the [installation
    guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).


### ADK Project Setup with uv

First, you need to install and configure the core ADK agent. After this you'll
set up the data sources to be used with the agent.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/google/adk-samples.git
    cd adk-samples/python/agents/data-science
    ```

1.  **Install Dependencies with uv:**

    ```bash
    uv sync
    ```

    This command reads the `pyproject.toml` file and installs all the necessary
    dependencies into a virtual environment managed by uv. On the first run,
    this command will also create a new virtual environment.

    By default, the virtual environment will be created in a `.venv` directory
    inside `adk-samples/python/agents/data-science`. If you already have a virtual
    environment created, or you want to use a different location, you can use
    the `--active` flag for `uv` commands, and/or change the
    `UV_PROJECT_ENVIRONMENT` environment variable. See
    [How to customize uv's virtual environment location](https://pydevtools.com/handbook/how-to/how-to-customize-uvs-virtual-environment-location/)
    for more details.

1.  **Activate the uv Shell:**

    If you are using the `uv` default virtual environment, you now need
    to activate the environment.

    ```bash
    source .venv/bin/activate
    ```

1.  **Set up Environment Variables:**

    Rename the file ".env.example" to ".env"
    Fill the below values:

    ```bash
    # Choose Model Backend: 0 -> ML Dev, 1 -> Vertex
    GOOGLE_GENAI_USE_VERTEXAI=1

    # ML Dev backend config. Fill if using Ml Dev backend.
    GOOGLE_API_KEY='YOUR_VALUE_HERE'

    # Vertex backend config
    GOOGLE_CLOUD_PROJECT='YOUR_VALUE_HERE'
    GOOGLE_CLOUD_LOCATION='YOUR_VALUE_HERE'
    ```

1.  **BQML Setup:**

    The BQML Agent uses the Vertex AI RAG Engine to query the full BigQuery ML
    Reference Guide.

    Before running the setup, ensure your project ID is added in .env file:
    `"GOOGLE_CLOUD_PROJECT"`. Leave the corpus name empty in the .env file:
    `BQML_RAG_CORPUS_NAME = ''`. The corpus name will be added automatically
    once it's created.

    To set up the RAG Corpus for your project, run the methods
    `create_RAG_corpus()` and `ingest_files()` in
    `data-science/data_science/utils/reference_guide_RAG.py` by running the
    below command from the working directory:

    ```bash
    python3 data_science/utils/reference_guide_RAG.py
    ```

1.  **Code Interpreter Setup:**

    The Data Science Agent also relies on a Vertex AI Code Interpreter extension.

    If an extension has already been created, provide the full resource name of
    the pre-existing Code Interpreter extension
    (e.g., `projects/<YOUR_PROJECT_ID>/locations/<YOUR_LOCATION>/extensions/<YOUR_EXTENSION_ID>`).
    in the `CODE_INTERPRETER_EXTENSION_NAME` variable in the .env file.

    If an extension name is not provided, a new extension will be created.
    Check the logs for the Vertex Extension ID and provide the value in your
    environment variables for future runs to avoid creating multiple extensions.


1. **NL2SQL Configuration:**

    For BigQuery NL2SQL generation, the gent can use one of two methods: either
    querying Gemini directly, or [CHASE-SQL](https://arxiv.org/abs/2410.01943).
    Set the variable `NL2SQL_METHOD` to either `BASELINE` (to use Gemini) or
    `CHASE` to use CHASE-SQL.

    For AlloyDB NL2SQL generation the agent will always use  Gemini, so the
    value of `NL2SQL_METHOD` will not affect the AlloyDB sub-agent.

## Database Setup

This sample has two alternate data sets that can be used. The
`ticket_sales_history` dataset only uses BigQuery, so if you plan to use
that dataset, you can skip the AlloyDB setup steps below.

The `cymbal_flights` dataset uses both BigQuery and AlloyDb. If you plan to
use that dataset, you should follow the instructions below for both BigQuery
and AlloyDB.

### <a name="bigquery-setup">BigQuery Setup</a>

Set the BigQuery project IDs in the `.env` file. This can be the same GCP
Project you use for `GOOGLE_CLOUD_PROJECT`, but you can use other BigQuery
projects as well, as long as you have access permissions to that project.

In some cases you may want to separate the BigQuery compute consumption from
BigQuery data storage. You can set `BQ_DATA_PROJECT_ID` to the project you
use for data storage, and `BQ_COMPUTE_PROJECT_ID` to the project you want to
use for compute. Otherwise, you can set both `BQ_DATA_PROJECT_ID` and
`BQ_COMPUTE_PROJECT_ID` to the same project id.

If you have an existing BigQuery table you wish to connect, specify the
`BQ_DATASET_ID` in the `.env` file as well. Otherwise set this value
according to the choice of sample dataset (see above).

We recommend not adding any production critical datasets to this sample agent.

### <a name="alloydb-setup">AlloyDB Setup</a>

#### AlloyDB Cluster Configuration

For this demo, we will setup your AlloyDB cluster in the same project
as you will be using for the Vertex AI API calls. In a production
scenario, this would likely be in a different project; in that case,
you would need to set up some form of VPC peering between the projects to
allow your ADK Agent to access the AlloyDB cluster.

1. Enable APIs:

    ```bash
    gcloud services enable alloydb.googleapis.com \
                           compute.googleapis.com \
                           cloudresourcemanager.googleapis.com \
                           servicenetworking.googleapis.com \
                           vpcaccess.googleapis.com \
                           aiplatform.googleapis.com
    ```

1. Download and install [postgres-client cli (`psql`)][install-psql].

1. Install the [AlloyDB Auth Proxy][install-alloydb-auth-proxy].

1. Set environment variables. For security reasons, use a different password for
   `$DB_PASS` and note it for future use:

    ```bash
    export CLUSTER=my-alloydb-cluster
    export INSTANCE=my-alloydb-instance
    export REGION=us-east1
    export DB_USER=postgres
    export DB_PASS=my-alloydb-pass
    ```

1. Create an AlloyDB cluster:

    ```bash
    gcloud alloydb clusters create $CLUSTER \
        --password=$DB_PASS\
        --network=default \
        --region=$REGION \
        --project=$PROJECT_ID
    ```

1. Create a primary instance:

    ```bash
    gcloud alloydb instances create $INSTANCE \
        --instance-type=PRIMARY \
        --cpu-count=8 \
        --region=$REGION \
        --cluster=$CLUSTER \
        --project=$PROJECT_ID \
        --ssl-mode=ALLOW_UNENCRYPTED_AND_ENCRYPTED \
        --database-flags=password.enforce_complexity=on
    ```

1. Enable public IP on instance:

    ```bash
    gcloud alloydb instances update $INSTANCE \
        --cluster=$CLUSTER  \
        --region=$REGION  \
        --assign-inbound-public-ip=ASSIGN_IPV4
    ```

1. Connect to instance using AlloyDB auth proxy:

    ```bash
    ./alloydb-auth-proxy --public-ip \
        "projects/$PROJECT_ID/locations/$REGION/clusters/$CLUSTER/instances/$INSTANCE"
    ```

1. Verify you can connect to your instance with the `psql` tool. Enter
   password for AlloyDB (`$DB_PASS` environment variable set above) when prompted:

    ```bash
    psql -h 127.0.0.1 -p 5432 -U $DB_USER
    ```

[install-psql]: https://www.timescale.com/blog/how-to-install-psql-on-mac-ubuntu-debian-windows/
[install-alloydb-auth-proxy]: https://cloud.google.com/alloydb/docs/auth-proxy/connect#install

## Sample Datasets

The Data Science agent includes two different sample datasets that showcase
different aspects of its capabilities.

### Dataset Configuration
To configure the Data Science Agent to use the correct dataset, you
will need to give it the name of a dataset configuration file in
the environment variable `DATASET_CONFIG_FILE`. Two sample configuration
files are provided: `forecasting_sticker_sales_dataset_config.json`
and `flights_dataset_config.json`, corresponding to the two datasets
described below. In your `.env. file, set the environment variable
as either
```bash
DATASET_CONFIG_FILE='./forecasting_sticker_sales_dataset_config.json'
```
or
```bash
DATASET_CONFIG_FILE='./flights_dataset_config.json
```
#### Dataset Configuration File Format
The two provided configuration files give examples of how to specify
a dataset configuration. The file is a standard JSON format, with
two main sections: `datasets` and `cross_dataset_relations`.

Each entry in `datasets` must contain the following fields: `type`,
`name`, and `description`. The description field is passed to the
root data science agent to help it decide how to use that dataset,
so ensure the description is written to be useful to the agent.

The `cross_dataset_relations` element contains an array,
`foreign_keys`, with entries for each foreign key relation between the
datasets the agent has access to. A `foreign_key` entry has two
fields, `child` and `parent`, each with the same fields:
* `type`: The type of the dataset (must match the type in the
    datasets field)
* `dataset`: The name of the dataset (must be one of the datasets
    configured in the datasets field)
* `table`: The table in the dataset where the key field is found
* `column`: The column in the dataset containing the key entries.

### Forecasting Sticker Sales Dataset ###

***NOTE:** This dataset uses BigQuery only. Make sure to follow the steps in the
[BigQuery Setup](#bigquery-setup) section to setup and configure the
agent to access BigQuery.*

The dataset contains two tables, `train` and `test`, to enable forecasting
and BQML analytics queries.

You will find this sample dataset in the
`data-science/data_science/utils/data/` directory. To load this dataset into
BigQuery, make sure you are still in the working directory
(`agents/data-science`). Then run the following commands:
```bash
python3 data_science/utils/create_bq_table.py
```

Set the following environment variable in your `.env` file to use this dataset:
```
BQ_DATASET_ID='forecasting_sticker_sales'`
```
Dataset source: _Walter Reade and Elizabeth Park. Forecasting Sticker Sales.
https://kaggle.com/competitions/playground-series-s5e1, 2025. Kaggle._

### Cymbal Airlines Flights Dataset ###

***NOTE:** This dataset uses both BigQuery and AlloyDB (via the MCP Toolbox for
Databases). Follow the steps in the [BigQuery Setup](#bigquery-setup)
section and the [AlloyDB Setup](#alloydb-setup) section below to setup both
databases and configure the agent for database access.*

After setting up AlloyDB, follow these steps to initialize the Cymbal
Airlines flights dataset.

#### Load data into AlloyDB ####

1. Set up the correct environment variables (these should also be set in the
`.env` file):
    ```bash
    # If you change the name of the dataset, change it here.
    export ALLOYDB_DATABASE=flights_dataset
    export ALLOYDB_HOSTNAME=<your AlloyDB hostname>
    export ALLOYDB_PORT=<your AlloyDB port>
    export ALLOYDB_USER=<your AlloyDB user>
    ```

1.  Connect to your database using `psql:
    ```bash
    psql -h $ALLOYDB_HOSTNAME -p $ALLOYDB_PORT -U $ALLOYDB_USER -d $ALLOYDB_DATABASE
    ```

1. Run this command in `psql` (Note that if you changed the name of the database
above, you will also need to change it here).
    ```sql
    CREATE DATABASE flights_dataset;
    ```
    Then type `<CTRL>-D` to exit `psql`.

1. Run this command from the `data-science/flights_dataset` directory to
populate data into the database:

    ```bash
    psql -h $ALLOYDB_HOSTNAME -p $ALLOYDB_PORT -U $ALLOYDB_USER -d $ALLOYDB_DATABASE \
        -f flights_dataset_alloydb.sql
    ```
#### Load data into BigQuery ####

1. Configure the environment variables as directed in the BigQuery Setup section below.
Also export the BigQuery dataset id for this sample dataset:

    ```bash
    export BQ_DATASET_ID=flights_dataset
    ```

1. Run this command to create a new BigQuery dataset:
    ```bash
    bq mk --location $GOOGLE_CLOUD_LOCATION --dataset $BQ_DATA_PROJECT_ID:$BQ_DATASET_ID
    ```
1. Run these commands from the `data-science/flights_dataset` directory to load the
data into BigQuery:

    ```bash
    bq --project_id=$BQ_DATA_PROJECT_ID --location=$GOOGLE_CLOUD_LOCATION \
        load --source_format=CSV --autodetect --skip_leading_rows=1 --replace \
        $BQ_DATASET_ID.flight_history flight_history_table.csv
    bq --project_id=$BQ_DATA_PROJECT_ID --location=$GOOGLE_CLOUD_LOCATION \
        load --source_format=CSV --autodetect --skip_leading_rows=1 \
        --allow_quoted_newlines --replace \
        $BQ_DATASET_ID.cymbalair_policies cymbalair_policies_table.csv
    bq --project_id=$BQ_DATA_PROJECT_ID --location=$GOOGLE_CLOUD_LOCATION \
        load --source_format=CSV --autodetect --skip_leading_rows=1 --replace \
        $BQ_DATASET_ID.ticket_sales_history ticket_sales_history_table.csv
    ```



#### MCP Toolkit for Databases: Local Setup

To use this dataset, you also need to set up the [MCP Toolbox for Databases][mcp-toolbox].
For initial setup, you can run the toolbox locally by following these steps:

1. Download the latest version of Toolbox as a binary:

    ```bash
    export OS="linux/amd64" # one of linux/amd64, darwin/arm64, darwin/amd64, or windows/amd64
    curl -O https://storage.googleapis.com/genai-toolbox/v0.12.0/$OS/toolbox
    ```

1. Make the binary executable:

    ```bash
    chmod +x toolbox
    ```

1. Copy the `toolbox_env-example.sh` file to a file called `toolbox_env.sh` and
populate it with the appropriate values for your project. Then source the `toolbox_env.sh` file:

    ```bash
    . ./toolbox_env.sh
    ```

1. Run the Toolbox server, pointing to the `toolbox-alloydb-local.yaml` configuration file:

    ```bash
    ./toolbox --tools-file "toolbox-alloydb-local.yaml"
    ```


## Running the Agent

You can run the agent using the ADK command in your terminal.
from the working directory:

1.  Run agent in CLI:

    ```bash
    uv run adk run data_science
    ```

1.  Run agent with ADK Web UI:
    ```bash
    uv run adk web
    ```
    Select the data_science from the dropdown



### Example Agent Interaction

Here's a quick example of how a user might interact with the Data Science
Multi-Agent System using the _Forecasting Sticker Sales_ dataset.

> ´**User:** Hi, What data do you have access to?

> **Agent:**  I have access to two tables: `train` and `test`. Both tables
contain sticker sales data with the following columns: `id`, `date`, `country`,
`store`, `product`, and `num_sold`.

> **User:** I need more details on the train table. What countries exist? How
many stores are there?

> **Agent:** Result: The train table contains data from Canada, Finland, Italy,
Kenya, Norway, and Singapore. Each of these countries has 3 distinct stores.

>Explanation: The database agent executed a SQL query against the train table to
find the distinct countries and the number of distinct stores within each
country. The query grouped the results by country and counted the distinct store
values.

> **User:** Please generate a plot with total sales per country.

> **Agent:**  Result: A bar plot has been generated showing the total sales per
country. The x-axis represents the country, and the y-axis represents the total
sales.

> Explanation:
> 1. The database agent was first called to retrieve the total sales per country
     from the train table.
> 2. The data science agent was then called to generate a bar plot visualizing
     the results. The plot displays each country on the x-axis and its corresponding
     total sales on the y-axis.

> **User:** What kinds of forecasting models can I train in BQML?

> **Agent:** You can train several types of forecasting models in BQML,
including ARIMA, Exponential Smoothing, and Temporal Fusion Transformer (TFT)
models. Would you like to proceed with training a forecasting model?

> **User:** Yes

> **Agent:** Okay, before proceeding, let's confirm the project and dataset ID
for BQML. We'll be using the `PROJECT_ID` project and the
`forecasting_sticker_sales` dataset. Does that sound right?

> **User:** Yes

> **Agent:** The BQML supports `ARIMA_PLUS`, `ARIMA_PLUS_XREG`.

>To proceed, I need to know which model you'd like to use. I also need to know
the following:

>*   The name of the column that provides the time points (e.g., 'date').
>*   The name of the column that contains the data to forecast (e.g., 'num_sold').
>*   Do you want to fit and forecast multiple time series using a single query?
If so, what are the ID columns? (e.g., `country`, `store`, `product`)


### Google Agents CLI (Recommended)

Use the [Google Agents CLI](https://github.com/google/agents-cli) to create a production-ready version of this agent with additional deployment options.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-data-science` with your project name):

```bash
agents-cli create my-data-science -a adk@data-science
```

The Google Agents CLI will prompt you to select deployment options and provides additional production-ready features including automated CI/CD deployment scripts.

## Testing and Evaluation

To run the test and evaluation code, you need a few additional dependencies. Run
the following uv command from the `agents/data-science` directory to install them:
```bash
uv sync --dev
```

### Running Evaluations

Evaluation tests assess the overall performance and capabilities of the agent in
a holistic manner.

**Run Evaluation Tests:**

```bash
uv run pytest eval
```

- This command executes all test files within the `eval/` directory.
- `uv run` ensures that pytest runs within the project's virtual environment.

### Running Tests

Tests assess the overall executability of the agents.

**Test Categories:**

*   **Integration Tests:** These tests verify that the agents can interact
    correctly with each other and with external services like BigQuery. They
    ensure that the root agent can delegate tasks to the appropriate sub-agents
    and that the sub-agents can perform their intended tasks.
*    **Sub-Agent Functionality Tests:** These tests focus on the specific
     capabilities of each sub-agent (e.g., Database Agent, BQML Agent). They
     ensure that each sub-agent can perform its intended tasks, such as
     executing SQL queries or training BQML models.
*   **Environment Query Tests:** These tests verify that the agent can handle
    queries that are based on the environment.

**Run Tests:**

```bash
uv run pytest tests
```

- This command executes all test files within the `tests/` directory.
- `uv run` ensures that pytest runs within the project's virtual environment.


## Deployment on Agent Runtime

### Initial Setup

To deploy the agent to Agent Runtime, first follow
[these steps](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime)
to set up your Google Cloud project for Agent Runtime.

You also need to give BigQuery User, BigQuery Data Viewer, and Vertex AI User
permissions to the Reasoning Engine Service Agent. Run the following commands to
grant the required permissions:

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
gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT} \
    --member="serviceAccount:${RE_SA}" \
    --condition=None \
    --role="roles/aiplatform.user"
```

### Deployment Steps

#### MCP Toolbox Deployment

Follow these steps to deploy the MCP Toolbox for Databases on Cloud Run. For
more details on the process, see the official
[Cloud Run deployment instructions](deploy-mcp-toolbox).

1. Enable the required Google Cloud APIs:
    ```bash
    gcloud services enable run.googleapis.com \
                        cloudbuild.googleapis.com \
                        artifactregistry.googleapis.com \
                        iam.googleapis.com \
                        secretmanager.googleapis.com
    ```

1. Ensure the account used for administering your Google Cloud project has the
approriate IAM roles:
    * Create Service Account role (`roles/iam.serviceAccountCreator`)
    * Secret Manager Admin role (`roles/secretmanager.admin`)
    * Cloud Run Developer (`roles/run.developer`)
    * Service Account User role (`roles/iam.serviceAccountUser`)

    ```bash
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member user:$USER_ACCOUNT \
        --role roles/iam.serviceAccountCreator \
        --role roles/secretmanager.admin \
        --role roles/run.developer \
        --role roles/iam.serviceAccountUser
    ```

1. Create a service account for the MCP Toolbox:
    ```bash
    gcloud iam service-accounts create toolbox-identity
    ```
1. Grant permissions to use secret manager.
    ```bash
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member serviceAccount:toolbox-identity@$PROJECT_ID.iam.gserviceaccount.com \
        --role roles/secretmanager.secretAccessor
    ```

1. [Create a secret](create-a-secret) for the AlloyDB user password.
    ```bash
    export ALLOYDB_POSTGRES_PASSWORD=<your Postgres user password>
    echo -n $ALLOYDB_POSTGRES_PASSWORD | \
        gcloud secrets create ALLOYDB_POSTGRES_PASSWORD \
        --replication-policy="automatic" \
        --data-file=-
    ```
    Note that the previous command will expose the database password in plaintext
    in the list of processes on your machine and in your shell history. To prevent
    this, store the password in a data file (e.g. `db-pass.txt`) and use this
    command instead.
    ```bash
    gcloud secrets create ALLOYDB_POSTGRES_PASSWORD \
        --replication-policy="automatic" \
        --data-file="db-pass.txt"
    ```

1. Copy the `toolbox.env-example` file to a version called `toolbox.env` with
the appropriate values for your project and AlloyDB setup.

1. Add the `toolbox-alloydb-remote.yaml` configuration file to Secret Manager.
    ```bash
    gcloud secrets create tools --data-file=toolbox-alloydb-remote.yaml
    ```
1. Export an environment variable for the container image to use for Cloud Run:
    ```bash
    export IMAGE=us-east1-docker.pkg.dev/database-toolbox/toolbox/toolbox:latest
    ```

1. Deploy Toolbox to Cloud Run.
    ```bash
    # TODO(dev): update --network and --subnet to match your VPC if necessary
    gcloud run deploy toolbox \
        --image $IMAGE \
        --service-account toolbox-identity \
        --region us-east1 \
        --set-secrets "/app/tools.yaml=tools:latest,ALLOYDB_POSTGRES_PASSWORD=ALLOYDB_POSTGRES_PASSWORD:latest" \
        --env-vars-file="toolbox.env" \
        --args="--tools-file=/app/tools.yaml","--address=0.0.0.0","--port=8080" \
        --network default \
        --subnet default
        # --allow-unauthenticated # https://cloud.google.com/run/docs/authenticating/public#gcloud

    ```

1. When the MCP Toolbox is deployed, you should be able to run the following command
to get a URL for the deployed Toolbox instance:
    ```bash
    gcloud run services describe toolbox --format 'value(status.url)'
    ```

1. Set the value of `MCP_TOOLBOX_HOST` in your `.env` file to that hostname. NOTE: Do not include
the `https://` prefix.

[deploy-mcp-toolbox]: https://googleapis.github.io/genai-toolbox/how-to/deploy_toolbox/
[create-a-secret]: https://cloud.google.com/secret-manager/docs/creating-and-accessing-secrets

#### Agent Deployment

Next, you need to create a `.whl` file for your agent. From the `data-science`
directory, run this command:

```bash
uv build --wheel --out-dir deployment
```

This will create a file named `data_science-0.1-py3-none-any.whl` in the
`deployment` directory.

Then run the below command. This will create a staging bucket in your GCP
project and deploy the agent to Agent Runtime:

```bash
cd deployment/
python3 deploy.py --create
```

When this command returns, if it succeeds it will print an Agent Runtime resource
name that looks something like this:
```
projects/************/locations/us-east1/reasoningEngines/7737333693403889664
```
The last sequence of digits is the Agent Runtime resource ID.

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
Input: Hello. What data do you have?
Response: I have access to the train and test tables inside the
forecasting_sticker_sales dataset.
...
```

Note that this is *not* a full-featured, production-ready CLI; it is just
intended to show how to use the Agent Runtime API to interact with a deployed
agent.

The main part of the `test_deployment.py` script is approximately this code:

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

To delete the agent, run the following command (using the resource ID returned
previously):
```bash
python3 deployment/deploy.py --delete --resource_id=RESOURCE_ID
```



## Optimizing and Adjustment Tips

*   **Prompt Engineering:** Refine the prompts for `root_agent`, `bqml_agent`,
    `bigquery_agent`, `alloydb_agent`, and `ds_agent` to improve accuracy and
    guide the agents more effectively. Experiment with different phrasing and
    levels of detail.
*   **Extension:** Extend the multi-agent system with your own AgentTools or
    sub_agents. You can do so by adding additional tools and sub_agents to the
    root agent inside `agents/data-science/data_science/agent.py`.
*   **Partial imports:** If you only need certain capabilities inside the
    multi-agent system, e.g. just the data agent, you can import the data_agent
    as an AgentTool into your own root agent.
*   **Model Selection:** Try different language models for both the top-level
    agent and the sub-agents to find the best performance for your data and
    queries.


## Troubleshooting

*   If you face `500 Internal Server Errors` when running the agent, simply
    re-run your last command. That should fix the issue.
*   If you encounter issues with the code interpreter, review the logs to
    understand the errors. Make sure you're using base-64 encoding for
    files/images if interacting directly with a code interpreter extension
    instead of through the agent's helper functions.
*   If you see errors in the SQL generated, try the following:
    - including clear descriptions in your tables and columns help boost
      performance
    - if your database is large, try setting up a RAG pipeline for schema
      linking by storing your table schema details in a vector store

## Clean Up

Clean up after completing the demo.

1. Set environment variables:

    ```bash
    export CLUSTER=my-alloydb-cluster
    export REGION=us-east1
    ```

1. Delete AlloyDB cluster that contains instances:

    ```bash
    gcloud alloydb clusters delete $CLUSTER \
        --force \
        --region=$REGION \
        --project=$PROJECT_ID
    ```

## Deployment on Google Cloud Run

These instructions walk through the process of deploying the Data Science agent to Google Cloud Run, including Cloud SQL for session storage.

### Before you begin

Deploying to Google Cloud Run requires:

- A [Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) with billing enabled.
- `gcloud` CLI ([Installation instructions](https://cloud.google.com/sdk/docs/install))

### 1 - Authenticate the Google Cloud CLI, and enable Google Cloud APIs.

```
gcloud auth login
gcloud auth application-default login

export PROJECT_ID="<YOUR_PROJECT_ID>"
gcloud config set project $PROJECT_ID

gcloud services enable sqladmin.googleapis.com \
   compute.googleapis.com \
   cloudresourcemanager.googleapis.com \
   servicenetworking.googleapis.com \
   aiplatform.googleapis.com
```

### 2 - Create a Cloud SQL instance for the agent sessions service.

```bash
gcloud sql instances create ds-agent-session-service \
   --database-version=POSTGRES_17 \
   --tier=db-g1-small \
   --region=us-east1 \
   --edition=ENTERPRISE \
   --root-password=ds-agent-demo
```

Once created, you can view your instance in the Cloud Console [here](https://console.cloud.google.com/sql/instances/ds-agent-session-service/overview).

### 3 - Deploy the agent to Cloud Run
Now we are ready to deploy the Data Science agent to Cloud Run! :rocket:

```bash
gcloud run deploy data-science-agent \
  --source . \
  --port 8080 \
  --memory 2G \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --add-cloudsql-instances $PROJECT_ID:us-east1:ds-agent-session-service \
  --update-env-vars SERVE_WEB_INTERFACE=True,SESSION_SERVICE_URI="postgresql+pg8000://postgres:ds-agent-demo@postgres/?unix_sock=/cloudsql/$PROJECT_ID:us-east1:ds-agent-session-service/.s.PGSQL.5432",GOOGLE_CLOUD_PROJECT=$PROJECT_ID \
  --region us-east1
```

When this runs successfully, you should see:

```bash
Service [data-science-agent] revision [data-science-agent-00001-aaa] has been deployed and is serving 100 percent of traffic.
```

### 4 - Test the Cloud Run Deployment

Open the Cloud Run Service URL outputted by the previous step.
You should see the ADK Web UI for the Data Science Agent.

### Clean up

You can clean up this agent sample by:
- Deleting the [Cloud Run Services](https://console.cloud.google.com/run).
- Deleting the [Cloud SQL instance](https://console.cloud.google.com/sql/instances).


## Disclaimer

This agent sample is provided for illustrative purposes only and is not intended
for production use. It serves as a basic example of an agent and a foundational
starting point for individuals or teams to develop their own agents.

This sample has not been rigorously tested, may contain bugs or limitations, and
does not include features or optimizations typically required for a production
environment (e.g., robust error handling, security measures, scalability,
performance considerations, comprehensive logging, or advanced configuration
options).

Users are solely responsible for any further development, testing, security
hardening, and deployment of agents based on this sample. We recommend thorough
review, testing, and the implementation of appropriate safeguards before using
any derived agent in a live or critical system.


[mcp-toolbox]: https://googleapis.github.io/genai-toolbox/
[adk-builtin-tool-bq]: https://google.github.io/adk-docs/tools/built-in-tools/#bigquery
