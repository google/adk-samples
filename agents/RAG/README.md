# Documentation Retrieval Agent

## Overview

This agent is designed to answer questions related to documents you uploaded to Vertex AI RAG Engine. It utilizes Retrieval-Augmented Generation (RAG) with the Vertex AI RAG Engine to fetch relevant documentation snippets and code references, which are then synthesized by an LLM (Gemini) to provide informative answers with citations.


![RAG Architecture](RAG_architecture.png)

This diagram outlines the agent's workflow, designed to provide informed and context-aware responses. User queries are processed by agent development kit. The LLM determines if external knowledge (RAG corpus) is required. If so, the `VertexAiRagRetrieval` tool fetches relevant information from the configured Vertex RAG Engine corpus. The LLM then synthesizes this retrieved information with its internal knowledge to generate an accurate answer, including citations pointing back to the source documentation URLs.

## Agent Details
| Attribute         | Details                                                                                                                                                                                             |
| :---------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Interaction Type** | Conversational                                                                                                                                                                                      |
| **Complexity**    | Intermediate 
| **Agent Type**    | Single Agent                                                                                                                                                                                        |
| **Components**    | Tools, RAG, Evaluation                                                                                                                                                                               |
| **Vertical**      | Horizontal                                                                                                                                                                               |
### Agent Architecture

![RAG](RAG_workflow.png)


### Key Features

*   **Retrieval-Augmented Generation (RAG):** Leverages [Vertex AI RAG
    Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview)
    to fetch relevant documentation.
*   **Citation Support:** Provides accurate citations for the retrieved content,
    formatted as URLs.
*   **Clear Instructions:** Adheres to strict guidelines for providing factual
    answers and proper citations.

## Setup and Installation Instructions
### Prerequisites

*   **Google Cloud Account:** You need a Google Cloud account.
*   **Python 3.9+:** Ensure you have Python 3.9 or a later version installed.
*   **Poetry:** Install Poetry by following the instructions on the official Poetry website: [https://python-poetry.org/docs/](https://python-poetry.org/docs/)
*   **Git:** Ensure you have git installed.

### Project Setup with Poetry

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/google/adk-samples.git
    cd adk-samples/agents/RAG
    ```

2.  **Install Dependencies with Poetry:**

    ```bash
    poetry install
    ```

    This command reads the `pyproject.toml` file and installs all the necessary dependencies into a virtual environment managed by Poetry.

3.  **Activate the Poetry Shell:**

    ```bash
    poetry env activate
    ```

    This activates the virtual environment, allowing you to run commands within the project's environment.
    Make sure the environment is active. If not, you can also activate it through 

     ```bash
    source .venv/bin/activate 
    ```   
4.  **Set up Environment Variables:**
    Rename the file ".env example" to ".env" 
    Follow the steps in the file to set up the environment variables.

5. **Setup Corpus:**
    If you have an existing corpus in Vertex AI RAG Engine, please set corpus information in your .env file. For example: RAG_CORPUS='projects/123/locations/us-central1/ragCorpora/456'. 

    If you don't have a corpus setup yet, please follow "How to upload my file to my RAG corpus" section. The `prepare_corpus_and_data.py` script will automatically create a corpus (if needed) and update the `RAG_CORPUS` variable in your `.env` file with the resource name of the created or retrieved corpus.

#### How to upload my file to my RAG corpus

The `rag/shared_libraries/prepare_corpus_and_data.py` script helps you set up a RAG corpus and upload an initial document. By default, it downloads Alphabet's 2024 10-K PDF and uploads it to a new corpus.

1.  **Authenticate with your Google Cloud account:**
    ```bash
    gcloud auth application-default login
    ```

2.  **Set up environment variables in your `.env` file:**
    Ensure your `.env` file (copied from `.env.example`) has the following variables set:
    ```
    GOOGLE_CLOUD_PROJECT=your-project-id
    GOOGLE_CLOUD_LOCATION=your-location  # e.g., us-central1
    ```

3.  **Configure and run the preparation script:**
    *   **To use the default behavior (upload Alphabet's 10K PDF):**
        Simply run the script:
        ```bash
        python rag/shared_libraries/prepare_corpus_and_data.py
        ```
        This will create a corpus named `Alphabet_10K_2024_corpus` (if it doesn't exist) and upload the PDF `goog-10-k-2024.pdf` downloaded from the URL specified in the script.

    *   **To upload a different PDF from a URL:**
        a. Open the `rag/shared_libraries/prepare_corpus_and_data.py` file.
        b. Modify the following variables at the top of the script:
           ```python
           # --- Please fill in your configurations ---
           # ... project and location are read from .env ...
           CORPUS_DISPLAY_NAME = "Your_Corpus_Name"  # Change as needed
           CORPUS_DESCRIPTION = "Description of your corpus" # Change as needed
           PDF_URL = "https://path/to/your/document.pdf"  # URL to YOUR PDF document
           PDF_FILENAME = "your_document.pdf"  # Name for the file in the corpus
           # --- Start of the script ---
           ```
        c. Run the script:
           ```bash
           python rag/shared_libraries/prepare_corpus_and_data.py
           ```

    *   **To upload a local PDF file:**
        a. Open the `rag/shared_libraries/prepare_corpus_and_data.py` file.
        b. Modify the `CORPUS_DISPLAY_NAME` and `CORPUS_DESCRIPTION` variables as needed (see above).
        c. Modify the `main()` function at the bottom of the script to directly call `upload_pdf_to_corpus` with your local file details:
           ```python
           def main():
             initialize_vertex_ai()
             corpus = create_or_get_corpus() # Uses CORPUS_DISPLAY_NAME & CORPUS_DESCRIPTION

             # Upload your local PDF to the corpus
             local_file_path = "/path/to/your/local/file.pdf" # Set the correct path
             display_name = "Your_File_Name.pdf" # Set the desired display name
             description = "Description of your file" # Set the description

             # Ensure the file exists before uploading
             if os.path.exists(local_file_path):
                 upload_pdf_to_corpus(
                     corpus_name=corpus.name,
                     pdf_path=local_file_path,
                     display_name=display_name,
                     description=description
                 )
             else:
                 print(f"Error: Local file not found at {local_file_path}")

             # List all files in the corpus
             list_corpus_files(corpus_name=corpus.name)
           ```
        d. Run the script:
           ```bash
           python rag/shared_libraries/prepare_corpus_and_data.py
           ```

More details about managing data in Vertex RAG Engine can be found in the
[official documentation page](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-quickstart).

## Running the Agent
You can run the agent using the ADK command in your terminal.
from the root project directory:

1.  Run agent in CLI:

    ```bash
    adk run rag
    ```

2.  Run agent with ADK Web UI:
    ```bash
    adk web
    ```
    Select the RAG from the dropdown


### Example Interaction
Here's a quick example of how a user might interact with the agent:

**Example 1: Document Information Retrieval**

User: What are the key business segments mentioned in Alphabet's 2024 10-K report?

Agent: According to Alphabet's 2024 10-K report, the key business segments are:
1. Google Services (including Google Search, YouTube, Google Maps, Play Store)
2. Google Cloud (offering cloud computing services, data analytics, and AI solutions)
3. Other Bets (including Waymo for autonomous driving technology)
[Source: goog-10-k-2024.pdf]

## Evaluating the Agent

The evaluation can be run from the `RAG` directory using
the `pytest` module:

```
poetry run pytest eval
```

### Evaluation Process

The evaluation framework consists of three key components:

1. **test_eval.py**: The main test script that orchestrates the evaluation process. It uses the `AgentEvaluator` from Google ADK to run the agent against a test dataset and assess its performance based on predefined criteria.

2. **conversation.test.json**: Contains a sequence of test cases structured as a conversation. Each test case includes:
   - A user query (e.g., questions about Alphabet's 10-K report)
   - Expected tool usage (which tools the agent should call and with what parameters)
   - Reference answers (ideal responses the agent should provide)

3. **test_config.json**: Defines evaluation criteria and thresholds:
   - `tool_trajectory_avg_score`: Measures how well the agent uses the appropriate tools
   - `response_match_score`: Measures how closely the agent's responses match the reference answers

When you run the evaluation, the system:
1. Loads the test cases from conversation.test.json
2. Sends each query to the agent
3. Compares the agent's tool usage against expected tool usage
4. Compares the agent's responses against reference answers
5. Calculates scores based on the criteria in test_config.json

This evaluation helps ensure the agent correctly leverages the RAG capabilities to retrieve relevant information and generates accurate responses with proper citations.

## Deploying the Agent

The Agent can be deployed to Vertex AI Agent Engine using the following
commands:

```
python deployment/deploy.py
```

After deploying the agent, you'll be able to read the following INFO log message:

```
Deployed agent to Vertex AI Agent Engine successfully, resource name: projects/<PROJECT_NUMBER>/locations/us-central1/reasoningEngines/<AGENT_ENGINE_ID>
```

Please note your Agent Engine resource name and update `.env` file accordingly as this is crucial for testing the remote agent.

You may also modify the deployment script for your use cases.

## Testing the deployed agent

After deploying the agent, follow these steps to test it:

1. **Update Environment Variables:**
   - Open your `.env` file.
   - The `AGENT_ENGINE_ID` should have been automatically updated by the `deployment/deploy.py` script when you deployed the agent. Verify that it is set correctly:
     ```
     AGENT_ENGINE_ID=projects/<PROJECT_NUMBER>/locations/us-central1/reasoningEngines/<AGENT_ENGINE_ID>
     ```

2. **Grant RAG Corpus Access Permissions:**
   - Ensure your `.env` file has the following variables set correctly:
     ```
     GOOGLE_CLOUD_PROJECT=your-project-id
     RAG_CORPUS=projects/<project-number>/locations/us-central1/ragCorpora/<corpus-id>
     ```
   - Run the permissions script:
     ```bash
     chmod +x deployment/grant_permissions.sh
     ./deployment/grant_permissions.sh
     ```
   This script will:
   - Read the environment variables from your `.env` file
   - Create a custom role with RAG Corpus query permissions
   - Grant the necessary permissions to the AI Platform Reasoning Engine Service Agent

3. **Test the Remote Agent:**
   - Run the test script:
     ```bash
     python deployment/run.py
     ```
   This script will:
   - Connect to your deployed agent
   - Send a series of test queries
   - Display the agent's responses with proper formatting

The test script includes example queries about Alphabet's 10-K report. You can modify the queries in `deployment/run.py` to test different aspects of your deployed agent.

## Customization

### Customize Agent
You can customize system instruction for the agent and add more tools to suit your need, for example, google search.

### Customize Vertex RAG Engine
You can read more about [official Vertex RAG Engine documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-quickstart) for more details on customizing corpora and data.


### Plug-in other retrieval sources
You can also integrate your preferred retrieval sources to enhance the agent's
capabilities. For instance, you can seamlessly replace or augment the existing
`VertexAiRagRetrieval` tool with a tool that utilizes Vertex AI Search or any
other retrieval mechanism. This flexibility allows you to tailor the agent to
your specific data sources and retrieval requirements.
