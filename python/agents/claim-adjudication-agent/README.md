# Health Claim Adjudication Agent

## Overview

The **Health Claim Adjudication Agent** is a sophisticated, multi-agent workflow designed to automate the complex process of insurance claim processing. By coordinating specialized sub-agents, it evaluates claim admissibility, performs detailed financial adjudication against hospital bills and policy terms, and synthesizes the findings into a comprehensive summary report.

The agent streamlines the end-to-end lifecycle of a cashless health insurance claim:
1.  **Discovery**: Retrieves medical documents and invoices directly from Google Cloud Storage for a specific claim ID.
2.  **Admissibility**: Verifies policy coverage, identifies pre-existing diseases (PED), and checks waiting periods.
3.  **Adjudication**: Analyzes hospital bills against policy terms, co-payments, deductibles.
4.  **Synthesis**: Consolidates technical findings into a professional, structured report for final approval.

## Agent Details

| Feature | Description |
| --- | --- |
| **Interaction Type** | Workflow / Conversational |
| **Complexity**  | Advanced |
| **Agent Type**  | Multi-Agent (Parallel & Sequential) |
| **Components**  | Tools: GCS Retrieval; Callbacks: State Management |
| **Vertical**  | Healthcare / Insurance |

### Agent Architecture

This workflow utilizes a combination of parallel and sequential execution patterns:
- **Parallel Execution**: Admissibility and Amount Adjudication sub-agents run concurrently to maximize efficiency.
- **Sequential Execution**: The results from the core adjudication are passed to the Summary Writer to generate the final report.

## Setup and Installation

### 1. Prerequisites

*   **Python 3.11+**
*   **uv**: For dependency management.
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*   **Google Cloud Platform Project** with GCS and Vertex AI enabled.

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd claim-adjudication-agent

# Install dependencies
uv sync
```

### 3. Authentication

Before running the agent, you must authenticate with Google Cloud to provide Application Default Credentials (ADC):

```bash
gcloud auth application-default login
```

### 4. Configuration

Set up your `.env` file with the following credentials:

```bash
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=<your-project-id>
GOOGLE_CLOUD_LOCATION=<region>
GEMINI_FLASH=gemini-2.5-flash
CLAIM_DOCUMENTS_BUCKET=<your-gcs-bucket-name>
CLAIM_DOCUMENTS_BUCKET_FOLDER=<doc_folder>
```

## Running the Agent

### Using `adk`

ADK provides a convenient way to run the agent locally:

```bash
# Run via CLI
uv run adk run claim_adjudication_agent

# Run via Web Interface
uv run adk web
```

## Example Interaction

```text
* user: Adjudicate the claim ID CLAIMIDX0001

* agent: [Chief Health Claims Advisor]
I have retrieved the documentation for Claim ID CLAIMIDX0001 from our secure storage.
I am now initiating the parallel evaluation of admissibility and financial adjudication...

[...Processing...]

* agent: Adjudication Complete. Here is the summary:
- Status: Admissible
- Total Billed: ₹1,50,000
- Total Payable: ₹1,35,000
- Deductions: ₹15,000
- Conclusion: The claim is approved.
```

## Running Tests

To run the functional tests and evaluation suite, first ensure your development dependencies are synced:

```bash
# Sync development dependencies
uv sync --dev

# Run functional agent tests
uv run pytest tests/test_agents.py

# Run the automated evaluation suite
uv run pytest eval/test_eval.py
```

You can also run all tests at once with `uv run pytest`.

### Alternative: Using Google Agents CLI

You can also use the [Google Agents CLI](https://github.com/google/agents-cli) to create a production-ready version of this agent with additional deployment options.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-claim-adjudication-agent` with your project name):

```bash
agents-cli create my-claim-adjudication-agent -a adk@claim-adjudication-agent
```

The Google Agents CLI will prompt you to select deployment options and provides additional production-ready features including automated CI/CD deployment scripts.

## Customization

- **Modify Prompts**: Update `claim_adjudication_agent/prompt.py` or the sub-agent prompts.
- **Add Tools**: Extend the capability by adding new functions to `tools/tools.py`.
- **Callback Logic**: Adjust state management or logging in `tools/tools.py`.

## Areas of Enhancement

- **Document Validation Agent**: Implement an initial validation step to verify document completeness and legibility before the main adjudication flow begins.
- **On-Demand Retrieval**: Transition from block storage (GCS) to fetching documents via a backend API server on demand to improve scalability and security.
