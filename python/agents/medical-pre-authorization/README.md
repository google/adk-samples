# Medical-Pre-Authorization Agent


## Overview

The Medical Pre-Authorization Agent is an intelligent, automated workflow designed to streamline the medical pre-authorization process on Google Cloud. It leverages a multi-agent system to efficiently process a patient's request, from initial document submission to final report generation.

This sample is compatible with the Google Agents CLI and can be used as a base for creating production-ready agent deployments.

## How It Works
The workflow is orchestrated by a primary Insurance Agent, built with Vertex AI Agent Builder and the ADK framework, which manages the end-to-end logic.

1. Request Initiation: A patient submits a pre-authorization request, along with their medical records and insurance details, which triggers a Cloud Run service.

2. Orchestration: The Cloud Run service invokes the main Insurance Agent, which takes control of the workflow.

3. Task Delegation: The Insurance Agent delegates specialized tasks to two sub-agents:

information_extraction Agent: Parses the submitted documents using LLM capabilities to extract relevant medical and insurance data.

data_analyst Agent: Analyzes the extracted information, makes a decision on the pre-auth request and generates a comprehensive summary.

Final Output: The system consolidates the analysis into a final PDF report and stores it in Cloud Storage for user access.

This automated solution transforms the complex pre-authorization process, leveraging AI to deliver efficient and accurate results.


## Agent Details

The key features of the Medical Pre-Authorization Agent include:

| Feature | Description |
| --- | --- |
| **Interaction Type** | Conversational |
| **Complexity**  | Medium |
| **Agent Type**  | Multi Agent |
| **Components**  | Tools, AgentTools|
| **Vertical**  | Healthcare |


### Agent architecture:

This diagram shows the detailed architecture of the agents and tools used
to implement this workflow.
<img src="pre-auth-ai-agent-architrecture.png" alt="Medical Pre-Authurization Agent" width="800"/>

## Using Google Agents CLI

Use the **Google Agents CLI** to scaffold a production-ready version of this agent with additional deployment and configuration options.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

## Setup and Installation

1.  **Prerequisites**

    *   Python 3.12
    *   uv
        *   An extremely fast Python package installer and resolver for 
        dependency management. Please follow the instructions on the official 
        [uv website](https://docs.astral.sh/uv/getting-started/installation/) for installation.

    * A project on Google Cloud Platform
    * Google Cloud CLI
        *   For installation, please follow the instruction on the official
            [Google Cloud website](https://cloud.google.com/sdk/docs/install).

2.  **Installation**

    ```bash
    # Clone this repository.
    git clone https://github.com/google/adk-samples.git
    cd adk-samples/python/agents/medical-pre-authorization

    # Verify if uv is installed correctly
    uv --version

    # Sync dependencies.
    uv sync
    ```

3.  **Configuration**

    *   Set up Google Cloud credentials.

        *   Copy the `.env.example` file to `.env` and fill in the required
            environment variables. You may also set these environment
            variables in your shell instead.

        ```bash
        export GOOGLE_GENAI_USE_VERTEXAI=true
        export GOOGLE_CLOUD_PROJECT=<your-project-id>
        export GOOGLE_CLOUD_LOCATION=<your-project-location>
        export GOOGLE_CLOUD_STORAGE_BUCKET=<your-storage-bucket-for-deployment-artifacts>
        export REPORT_STORAGE_BUCKET=<your-storage-bucket-for-reports>
        ```

    *   Authenticate your GCloud account.

        ```bash
        gcloud auth application-default login
        gcloud auth application-default set-quota-project $GOOGLE_CLOUD_PROJECT
        ```

## Running the Agent

**Using `adk`**

ADK provides convenient ways to bring up agents locally and interact with them.
You may talk to the agent using the CLI:

```bash
uv run adk run medical_pre_authorization
```

Or on a web interface:

```bash
uv run adk web
```

The command `uv run adk web` will start a web server on your machine and print the URL.
You may open the URL, select "medical_pre_authorization" in the top-left drop-down menu, and
a chatbot interface will appear on the right. The conversation is initially
blank. Here are some example requests you may ask the Medical Pre-Authorization Agent to verify:

```
Hello, who are you?
```
Sampled responses of these requests are shown below in the [Example
Interaction](#example-interaction) section.

```
Hello! I'm a pre-authorization agent. I can help you process pre-authorization requests for medical treatments. I'll need some details about the treatment, your medical records, and your health insurance policy to do so.

```

## Example Interaction

Below is an example interaction with the Medical Pre-Authorization Agent. 

```
User: Hello, how can you help me?

Agent: Hello! I'm here to help you with your treatment pre-authorization requests. I can extract details from your medical records and insurance policy, analyze them, and then provide a report detailing the decision on your pre-authorization request.

How can I assist you today? Do you have a specific treatment in mind for pre-authorization? Please provide the treatment name, your medical records, and your health insurance policy.

User: I want to submit a pre authorization request for cataract surgery.

Agent: Thank you. To process your pre-authorization request for cataract surgery, I need two documents from you:

Your medical records related to the cataract surgery.
Your health insurance policy document.
Please provide these documents so I can proceed with your request.

User: Here is my medical report. #Attach a medical report. Sample report is available at tests/sample_data/patient_medical_diagnosis.pdf

Agent: Thank you for providing your medical report. I have received it.

Now, please provide your health insurance policy document so I can proceed with your pre-authorization request for cataract surgery.

User: Here is my insurance policy. #Attach a policy document. Sample doc is available at tests/sample_data/patient_health_insurance_policy.pdf

Agent: The pre-authorization request for Mr. Arjun Sharma's cataract surgery has been REJECTED.

The reason for the rejection is that the 48-month Pre-Existing Disease waiting period, which began on July 1, 2024, has not yet been completed.

A detailed report has been created and uploaded to the Cloud Storage Bucket. You can access it at: https://storage.cloud.google.com/<your-storage-bucket-for-report>/pre_authorization_report_<timestamp>.pdf

Please let me know if you have any other questions or need further assistance.

User: This is all. Thank you.

Agent: You're welcome! If you have any more questions or need assistance with future pre-authorization requests, feel free to ask. Have a good day!

```

## Running Tests

Then the tests and evaluation can be run from the `medical-pre-authorization` directory using
the `pytest` module:

```bash
uv run --extra dev pytest tests
uv run --extra dev pytest eval
```

`tests` runs the agent on a sample request, and makes sure that every component
is functional. `eval` is a demonstration of how to evaluate the agent, using the
`AgentEvaluator` in ADK. It sends a couple requests to the agent and expects
that the agent's responses match a pre-defined response reasonably well.


## Deployment

The Medical Pre-Authorization AI Agent can be deployed to Agent Runtime using the following
commands:

```bash
uv run --extra deployment deployment/deploy.py --create
```

When the deployment finishes, it will print a line like this:

```
Created remote agent: projects/<PROJECT_NUMBER>/locations/<PROJECT_LOCATION>/reasoningEngines/<AGENT_ENGINE_ID>
```

If you forgot the AGENT_ENGINE_ID, you can list existing agents using:

```bash
uv run --extra deployment deployment/deploy.py --list
```

The output will be like:

```
All remote agents:

123456789 ("medical_pre_authorization")
- Create time: 2025-05-12 12:35:34.245431+00:00
- Update time: 2025-05-12 12:36:01.421432+00:00
```

You may interact with the deployed agent using the `test_deployment.py` script
```bash
export USER_ID=<any string>
export AGENT_ENGINE_ID=<AGENT_ENGINE_ID>
uv run --extra deployment deployment/test_deployment.py --resource_id=${AGENT_ENGINE_ID} --user_id=${USER_ID}
```

The output will be like:
```bash
Found agent with resource ID: ...
Created session for user ID: ...
Type 'quit' to exit.
Input: Hello, what can you do for me?
Response: Hello! I'm a pre-authorization agent. I can help you process pre-authorization requests for medical treatments. I'll need some details about the treatment, your medical records, and your health insurance policy to do so.
```

To delete the deployed agent, you may run the following command:

```bash
uv run --extra deployment deployment/deploy.py --delete --resource_id=${AGENT_ENGINE_ID}
```
## Disclaimer

This agent sample is provided for illustrative purposes only and is not intended for production use. It serves as a basic example of an agent and a foundational starting point for individuals or teams to develop their own agents.

This sample has not been rigorously tested, may contain bugs or limitations, and does not include features or optimizations typically required for a production environment (e.g., robust error handling, security measures, scalability, performance considerations, comprehensive logging, or advanced configuration options).

Users are solely responsible for any further development, testing, security hardening, and deployment of agents based on this sample. We recommend thorough review, testing, and the implementation of appropriate safeguards before using any derived agent in a live or critical system.
