# Airflow Version Upgrade Agent

This project provides an intelligent, two-stage agentic system built with the Google Cloud Agent Development Kit (ADK) to automate the migration of Apache Airflow DAGs from a source version to a target version.

The system leverages Gemini models, Vertex AI Search for Retrieval-Augmented Generation (RAG), and a custom-built knowledge base in BigQuery to provide accurate, context-aware code conversions.

## Architecture Overview

The solution is composed of two primary agents that orchestrate a complex workflow:

1.  **Knowledge Base Agent (`knowledge_base_updater`)**: This agent is responsible for the one-time, intensive process of building the migration knowledge base. It's a three-step pipeline:
    *   **Parse DAGs**: Scans a GCS folder of your existing DAGs to identify the unique set of Airflow operators being used.
    *   **Research & Scrape**: For each operator, it uses Vertex AI Search (with Web Search datastore) to find relevant migration guides and documentation, then scrapes the content from these web pages.
    *   **Structure & Store**: It uses a Gemini model to analyze the scraped content, structure it into a predefined JSON schema (e.g., parameter changes, deprecation status, code examples), and stores this structured data in a BigQuery table.

2.  **Migration Assistant Agent (`airflow_migration_assistant`)**: This is the main user-facing agent.
    *   It first asks the user if the knowledge base has been built.
    *   If **yes**, it proceeds to the migration step.
    *   If **no**, it invokes the `knowledge_base_updater` agent to build the knowledge base first.
    *   For migration, it reads each source DAG file from GCS and uses a Gemini model with RAG (grounded on the BigQuery knowledge base via Vertex AI Search) to generate the migrated code, which is then saved to a destination GCS folder.

### Architectural Diagram

```text
[User] -> [Migration Assistant Agent (Cloud Run)]
  |
  + - Q: KB Exists?
      |
      + - (No) -> [Knowledge Base Agent]
      |               |
      |               1. Parse DAGs from [GCS]
      |               2. Scrape Web Content
      |               3. Structure with [Gemini] -> Store in [BigQuery]
      |               |
      |               +-> [Vertex AI Search] indexes BigQuery table
      |
      + - (Yes) -> [DAG Converter Tool]
                      |
                      1. Read DAG from [GCS Source]
                      2. Generate migrated code with [Gemini + RAG via Vertex AI Search]
                      3. Write new DAG to [GCS Destination]
```
![alt text](agent_pattern.png)
---

## Prerequisites

Before you begin, ensure you have the following:

*   A Google Cloud Project with the Billing account enabled.
*   [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/install) installed and authenticated.
*   [Terraform CLI](https://developer.hashicorp.com/terraform/downloads) installed.
*   [Python 3.11+](https://www.python.org/downloads/) installed.
*   **Permissions**: We assume that all necessary Google Cloud APIs and Service Account permissions have been provisioned by an administrator. Specifically, the following APIs must be enabled:
    *   `run.googleapis.com`
    *   `iam.googleapis.com`
    *   `iamcredentials.googleapis.com`
    *   `cloudbuild.googleapis.com`
    *   `artifactregistry.googleapis.com`
    *   `bigquery.googleapis.com`
    *   `aiplatform.googleapis.com`
    *   `storage-component.googleapis.com`
    *   `discoveryengine.googleapis.com`
*   Additionally, ensure your deployment user/service account has the following IAM roles:
    *   Cloud Build Editor (`roles/cloudbuild.builds.editor`)
    *   Artifact Registry Admin (`roles/artifactregistry.admin`)
    *   Cloud Run Admin (`roles/run.admin`)
    *   Create Service Accounts (`roles/iam.serviceAccountCreator`)
    *   Project IAM Admin (`roles/resourcemanager.projectIamAdmin`)
    *   Service Account User (`roles/iam.serviceAccountUser`)
    *   Service Usage Consumer (`roles/serviceusage.serviceUsageConsumer`)
    *   Storage Admin (`roles/storage.admin`)
    *   BigQuery DataEditor (`roles/bigquery.dataEditor`)
    *   BigQuery JobUser (`roles/bigquery.jobUser`)
    *   Discovery Engine Editor (`roles/discoveryengine.user`)
    *   Service Usage Consumer (`roles/serviceusage.serviceUsageConsumer`)

Consider your user has permissions on all the APIs and resources. If not, the given setup should take care of required access for user and service account.

---

## Setup and Configuration

Follow these steps to set up your local environment and provision the required infrastructure.

### 1. Local Development Setup

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```
2.  **Install Dependencies**:
    We recommend using `uv` for dependency management. Run the following command to install dependencies:
    ```bash
    uv sync --dev
    ```
    *(Alternatively, you can use `pip` with a virtual environment.)*

3.  **Configure Environment**:
    Create a `.env` file in the project root based on `.env.example` and fill in the values with your project details:
    *   `PROJECT_ID`: Your Google Cloud Project ID.
    *   `USER_EMAIL`: Your Google Cloud user email (used for IAM bindings).
    *   `TF_STATE_BUCKET_NAME`: A unique name for the GCS bucket to store Terraform state.
    *   Other variables can be left as defaults or customized.

### 2. Automated Infrastructure Setup

We provide a `Makefile` to automate the provisioning of all required Google Cloud infrastructure and Vertex AI Search resources for local testing.

**Prerequisites:**
*   Ensure you have authenticated with Google Cloud:
    ```bash
    gcloud auth application-default login
    ```
*   Ensure you have the required permissions listed in the [Prerequisites](#prerequisites) section.

To deploy the infrastructure, run the following command from the project root:

```bash
make deploy
```

This command will:
1.  Source your `.env` file.
2.  Check if the Terraform state bucket exists, and create it if it does not.
3.  Initialize and apply Terraform to create:
    *   GCS buckets for source and destination DAGs.
    *   BigQuery dataset and table for the migration corpus.
    *   Sample Airflow 1.10 DAGs in the source bucket for testing.
    *   Necessary IAM role bindings for the default Compute service account and your user account.
4.  Run a Python script to automate the creation of Vertex AI Search resources:
    *   Create a Website Data Store with predefined Airflow documentation sources.
    *   Create a BigQuery Data Store pointing to the newly created table.
    *   Create a Search App and link it to the Website Data Store.

Upon completion, you will see a summary of the created resources.

*Note: This automated setup is intended for local development and testing. For production deployments, refer to the [Deployment to Cloud Run](#deployment-to-cloud-run) section.*

### Alternative: Using Google Agents CLI

You can also use the [Google Agents CLI](https://github.com/google/agents-cli) to create a production-ready version of this agent with additional deployment options.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-airflow-migration-agent` with your project name):

```bash
agents-cli create my-airflow-migration-agent -a adk@airflow_version_upgrade_agent
```

The Google Agents CLI will prompt you to select deployment options and provides additional production-ready features including automated CI/CD deployment scripts.

### 5. Running the Agent Locally

Authenticate your local environment and run the ADK server:

```bash
# Authenticate
gcloud auth application-default login

# Make a call to adk web from the **agents/airflow_agents** directory
adk web
```

You can now interact with your agent in the ADK web UI, typically at `http://127.0.0.1:8000`.

---

## Deployment to Cloud Run

This agent is designed to be deployed as a containerized service on Cloud Run via the Agent Development Kit (ADK).

### Required Permissions for Cloud Run Deployment

If you plan to deploy this agent to Cloud Run (e.g., using `adk deploy`), ensure the deploying user or service account has the following additional permissions:

*   **Cloud Build Editor** (`roles/cloudbuild.builds.editor`)
*   **Artifact Registry Admin** (`roles/artifactregistry.admin`)
*   **Cloud Run Admin** (`roles/run.admin`)
*   **Service Account User** (`roles/iam.serviceAccountUser`) on the runtime service account (e.g., the default Compute Engine service account if used).

These permissions are required because `adk deploy` typically builds a container image using Cloud Build, pushes it to Artifact Registry, and then creates a Cloud Run service.

> [!NOTE]
> These permissions were ignored in the core Terraform scripts as they are only needed for remote deployment, and the agent is assumed to be running in a Google Cloud environment (like Cloud Shell) with sufficient permissions for local execution/testing.

1. Set Deployment Variables:

The variables needed for the deployment are captured in the `.env.example` file. 

Copy the example file to `.env` and configure your variables:

```bash
cp .env.example .env
# Open .env and fill in your details (PROJECT_ID, GOOGLE_CLOUD_LOCATION, SERVICE_NAME, etc.)
# Make sure to source your env variables before proceeding, or let your deployment pipeline load them.
source .env
```

2. Deploy to Cloud Run: 

Deploy the container, passing configuration as environment variables.

```bash
# Run this command from your **`agents`** directory

adk deploy cloud_run \
--project=$PROJECT_ID \
--region=$GOOGLE_CLOUD_DEPLOY_LOCATION \
--service_name=$CLOUD_RUN_SERVICE_NAME \
--app_name=$CLOUD_RUN_APP_NAME \
--with_ui \
--port 8000 \
$AGENT_PATH
```

During the deployment process, you might be prompted: `Allow unauthenticated invocations to [your-service-name] (y/N)?`.
* Enter `y` to allow public access to your agent's API endpoint without authentication.
* Enter `N` (or press Enter for the default) to require authentication.

Upon successful execution, the command will deploy your agent to Cloud Run and provide the URL of the deployed service.

---

## UI Testing

### Get an identity token (if needed)
```bash
export APP_URL="YOUR_CLOUD_RUN_SERVICE_URL"
export TOKEN=$(gcloud auth print-identity-token)
```

### List Apps
```bash
curl -X GET -H "Authorization: Bearer $TOKEN" $APP_URL/list-apps
```

### Create or Update a Session
Initialize or update the state for a specific user and session. Replace $APP_NAME with your actual app name if different.

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
    $APP_URL/apps/$APP_NAME/users/user_123/sessions/session_abcd \
    -H "Content-Type: application/json" \
    -d '{"state": {"preferred_language": "English", "visit_count": 5}}'
```

### Run the Agent
Send a prompt to your agent. Adjust the user/session IDs and prompt as needed.

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
    $APP_URL/run_sse \
    -H "Content-Type: application/json" \
    -d '{
    "app_name": "airflow_version_upgrade_app",
    "user_id": "user_123",
    "session_id": "session_abcd",
    "new_message": {
        "role": "user",
        "parts": [{
        "text": "Help me migrate airflow"
        }]
    },
    "streaming": false
    }'
```

Set `"streaming": true` if you want to receive Server-Sent Events (SSE).


## Demo Run Example

### How to access the agent:
1. Navigate to your **Cloud Run Services** in the Google Cloud Console.
2. Click on the deployed application URL for the `airflow-migration-agent`.
3. This will open the ADK web interface and start a new chat session.
4. Begin a conversation with the agent using the chat interface!

### Example Conversation Flow

Here is a step-by-step example of a successful interactive migration session:

**🧑 User:** 
> Hi

**🤖 Agent:** 
> *[Introduces itself and asks for details to upgrade to the next version]*
> 1. Do you have a knowledge base created already? 
> 2. If not, I can create it for you with the required inputs.

**🧑 User:** 
> No, help me create one.

**🤖 Agent:** 
> *[Asks for the details required to create a knowledge base]*
> Please provide:
> 1. GCS URI where your input DAGs reside
> 2. Source Airflow version
> 3. Destination Airflow version
> 4. Your Google Cloud Project ID

**🧑 User:** 
> ```text
> source_airflow_version="1.10"
> target_airflow_version="2.10.5"
> source_gcs_uri="gs://<your-source-bucket-name>/dags/"
> project_id="<project-id>"
> ```

**🤖 Agent:** 
> *[Invokes the 3-stage knowledge-building process (detailed in the operational guide). Once completed, it verifies the outcome]*
> The knowledge base has been built! Would you like to proceed with the migration, or stop here?

**🧑 User:** 
> Yes, let's proceed with migration.
> ```text
> destination_gcs_uri="gs://<your-destination-bucket-name>/output/"
> project_id="<project-id>"
> collection_id="<collection-id>"
> data_store_id="<datastore-id>"
> ```

**🤖 Agent:** 
> *[Executes the DAG conversion using RAG and grounded responses]*
> Migration complete! The output GCS bucket now contains your migrated DAGs.
