# Brand Aligner Agent

The Brand Aligner Agent provides an end-to-end workflow for evaluating visual assets (images and videos) against a company's brand guidelines. It operates as a self-contained application where orchestration, brand guidelines extraction, asset evaluation and processing run within a single process.

The evaluation methodology relies on a state-of-the-art implementation that utilises [Gecko](https://cloud.google.com/blog/products/ai-machine-learning/evaluate-your-gen-media-models-on-vertex-ai) on the *Vertex AI Evaluation Service* for prompt adherence, along with additional custom rubrics that cover brand alignment and generation quality.

Refer to the defensive publication on this topic titled [Rubric-Based Evaluation for On-Brand Generative Media](https://www.tdcommons.org/dpubs_series/8935/) for more details.

## Architecture & Workflow

The agent utilizes the **Google Agent Developer Kit (ADK)** and follows a sequential chain of specialized agents:

1. **`brand_aligner_agent` (Root/Planner)**:
   * **Role**: Entry point for the user. It handles file uploads, searches for existing files in Google Cloud Storage (GCS), and formulates the execution plan.
   * **Tools**: `save_artifacts_to_gcs_tool`, `search_user_files_tool`, `save_plan_to_state_tool`.
   * **Next Step**: Transitions to the `guideline_processor_agent`.

2. **`guideline_processor_agent`**:
   * **Role**: Iterates through the selected guideline files one by one. It extracts structured criteria from documents (PDF, Markdown, Text).
   * **Tools**: `guideline_processor_tool` (calls `GuidelineService`).
   * **Next Step**: Transitions to the `asset_evaluator_agent`.

3. **`asset_evaluator_agent`**:
   * **Role**: Iterates through the selected asset files (images/videos) one by one. It evaluates each asset against the processed guidelines using a multimodal model.
   * **Tools**: `asset_evaluator_tool` (calls `EvalService`).
   * **Next Step**: Transitions to the `summarizer_agent`.

4. **`summarizer_agent`**:
   * **Role**: Aggregates the results from all evaluations and presents a final summary to the user.

## Services & Logic

* **In-Process Services**: The agent logic relies on `services.py` (`GuidelineService` and `EvalService`), which handle the interactions with Google's generative models (Gemini and Vertex AI Eval SDK).
* **Tools**: Defined in `tools.py`, these bridge the agents to the services and handle state management (reading/writing to GCS).

## Authentication

The agent implements a hybrid authentication strategy in `auth.py`:

* **Gemini Enterprise Mode**: When running within the Gemini Enterprise ecosystem, it utilizes the user's existing session token for seamless authentication.
* **Standalone/ADK Mode**: When running independently, it implements a standard OAuth2 flow, prompting the user to authenticate via Google to access GCS resources.

## State & Artifacts

* **Session State**: Used to pass the plan (file lists), processed guidelines, and evaluation reports between agents.
* **Google Cloud Storage (GCS)**: Acts as the persistent storage layer.
  * User uploads are saved as artifacts.
  * Processed guideline JSONs and Evaluation Report JSONs are stored for audit and retrieval.

## Getting Started

### 1. Prerequisites & Environment Setup

Before running or deploying the agent, you must configure the environment.

1. **Copy the environment template:**

   ```bash
   cp .env.template .env
   ```

2. **Configure `.env`:** Open the newly created `.env` file and fill in the required values (Project ID, Region, GCS Bucket, Model Name, etc.).

### 2. Google Agents CLI (recommended)

Use the [Google Agents CLI](https://github.com/google/agents-cli) to create a production-ready version of this agent with additional deployment options.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-brand-aligner` with your project name):

```bash
agents-cli create my-brand-aligner -a adk@brand-aligner
```

The Google Agents CLI guides you through setup options and generates a production-ready project structure.

<details>
<summary>Alternative: Local development from this sample repo (manual clone)</summary>

### 3. Local Testing (ADK Web Interface)

1. Clone and move into the sample:

   ```bash
   git clone https://github.com/google/adk-samples.git
   cd adk-samples/python/agents/brand-aligner
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Run tests:

   ```bash
   uv run pytest -v
   ```

4. Run the ADK web UI:

   ```bash
   uv run adk web --log_level DEBUG
   ```

   Access the UI at `http://localhost:8000`.

### 4. Deployment to Agent Runtime and Gemini Enterprise

To deploy the agent to Agent Runtime then register it with Gemini Enterprise:

1. **Deploy the Agent:**

   ```bash
   uv run deployment/deploy.py
   ```

2. **Register with Gemini Enterprise:**

The `deployment/` directory contains essential scripts for managing the agent's registration and configuration with Gemini Enterprise.

* **`config.sh`**: **CRITICAL**. This file contains all the variables required for Gemini Enterprise registration and deployment (for example, Agent Runtime ID and service account details). It relies on your local `.env` file for values. Ensure this is configured correctly before running the registration script.
* **`ge_register.sh`**: Registers the deployed agent with Gemini Enterprise.
* **`ge_unregister.sh`**: Unregisters the agent from Gemini Enterprise.

</details>

## Configuration Reference

The agent relies on the following key environment variables (defined in `.env`):

* `MODEL_NAME`: The name of the generative model to use for agent reasoning and evaluation (for example, `gemini-2.5-flash`).
* `GCS_BUCKET_NAME`: The GCS bucket for storing artifacts and processed files.
* `PROJECT_ID`: The Google Cloud Project ID.
* `LOCATION`: The Google Cloud region (for example, `us-east1`).
* `AUTH_ID`, `OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`: Authentication credentials for the OAuth2 flow (if running in standalone mode).
