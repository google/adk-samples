# Nurse Handover Agent

## Overview

The Nurse Handover Agent is a generative AI-powered tool designed to streamline the critical process of shift handovers in a clinical setting. By automatically processing detailed, raw medical logs, it generates concise, structured, and informative handover summaries in the standardized ISBAR (Identification, Situation, Background, Assessment, Recommendation) format. This reduces the risk of miscommunication, saves valuable time for nursing staff, and ultimately enhances patient safety.

## Features

| Feature              | Description                                                                                                                                                                                                                                                                           |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Interaction Type** | Conversational                                                                                                                                                                                                                                                                        |
| **Complexity**       | Intermediate                                                                                                                                                                                                                                                                          |
| **Agent Type**       | Single Agent                                                                                                                                                                                                                                    |
| **Vertical**         | Healthcare                                                                                                                                                                                                                                                                            |
| **Key Features**     | **Tools**: `list_available_shifts`, `list_patients`, `generate_shift_endorsement`.<br>**Data Management**: Reads patient medical records from local text files.<br>**LLM Processing**: Utilizes multiple Gemini model calls to intelligently segment, analyze, and summarize raw medical data into the ISBAR format. |

### Example Interaction

A typical conversation with the agent looks like this:

**User:** `Hi!`

**Assistant:** `Hello! I'm your nurse shift handover assistant. My purpose is to help streamline your shift handover process by automatically generating comprehensive reports. I can generate a shift endorsement for patient MHID123456789 for the shift that started on 2024-06-07T07:30:00 and ended on 2024-06-07T19:30:00. Would you like me to go ahead and generate the report for you?`

**User:** `Yes, please!`

**Assistant:** `I have stored the following artifacts: (1) raw patient input file at MHID123456789-1717763400-1717806600-raw-inputs.txt (2) the endorsement report at MHID123456789-1717763400-1717806600-endorsement.md.`

## Architecture

The agent operates in a clear, multi-step process:

1.  **Initialization**: When the agent starts, it loads a predefined state including available patients and shifts from the `initialize_state` function in `nurse_handover/agent.py`.
2.  **User Interaction**: The agent converses with the user to confirm the patient and shift for which a handover report is needed.
3.  **Data Retrieval & Processing**: The `generate_shift_endorsement` tool reads the corresponding raw patient data from a local text file in the `nurse_handover/data/` directory.
4.  **Summarization with Gemini**: The `Summarizer` class in `nurse_handover/summary.py` orchestrates a series of calls to the Gemini model. It breaks down the summarization task into smaller, manageable chunks based on the ISBAR format, using specific prompts and configurations for each section (e.g., `id_and_background_template.txt`, `situation_template.txt`).
5.  **Report Generation**: The final, structured ISBAR report is assembled from the model's responses and saved as a Markdown file.

![Architecture Diagram](agent_pattern.png)

## Setup and Running

### Prerequisites

*   Python 3.11+
*   [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
*   A Google Cloud project with the Vertex AI API enabled, or a Gemini API key.

### Installation

1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    make install
    ```
3.  **Configure Environment Variables:**
    Create a `.env` file in the `python/agents/nurse-handover` directory (you can copy `.env.example` as a starting point) and populate it with your Google Cloud credentials.

    *   **For Vertex AI (Recommended):**
        ```env
        GOOGLE_CLOUD_PROJECT=your-gcp-project-id
        GOOGLE_CLOUD_LOCATION=your-gcp-region
        GOOGLE_GENAI_USE_VERTEXAI=true
        ```
    *   **For Gemini API Key:**
        ```env
        GOOGLE_API_KEY=your-api-key
        GOOGLE_GENAI_USE_VERTEXAI=false
        ```

### Running the Agent

You can interact with the agent using the ADK CLI or a web interface.

*   **Run with the ADK CLI:**
    ```bash
    make cli
    ```
*   **Run with the ADK Web UI:**
    ```bash
    make web
    ```
    This will start a local web server, and you can interact with the agent in your browser.

## Customization

This agent is designed to be a starting point. You can extend its functionality to fit your specific needs.

### Adding New Patient Data

Currently, the agent loads patient data from flat files. To add a new patient:

1.  **Create a Text File**: Add a new `.txt` file in the `python/agents/nurse-handover/nurse_handover/data/` directory. The filename must be the patient's ID (e.g., `MHID987654321.txt`).
2.  **Update Agent State**: Open `python/agents/nurse-handover/nurse_handover/agent.py` and add the new patient ID to the `patients` list within the `initialize_state` function.

    ```python
    def initialize_state(callback_context: CallbackContext) -> None:
        # ... (existing code)
        callback_context.state["patients"] = ["MHID123456789", "MHID987654321"] # Add new patient ID here
        # ... (existing code)
    ```

### Connecting to Real-time Data Sources

To make this agent production-ready, you can modify it to pull data from cloud-based sources like Google Cloud Storage (GCS) and BigQuery.

#### **1. Fetching Patient Files from Google Cloud Storage (GCS)**

Instead of reading files from a local directory, you can adapt the agent to fetch them from a GCS bucket.

**Modification Point**: `nurse_handover/tools.py` within the `generate_shift_endorsement` function.

**Steps**:

1.  Import the GCS client: `from google.cloud import storage`
2.  Initialize the client: `storage_client = storage.Client()`
3.  Replace the local file read logic with GCS blob download logic.

    ```python
    # In generate_shift_endorsement function in tools.py

    # Replace this:
    # patient_file = PATIENT_FILE_DIR / f"{patient}.txt"
    # document_text = patient_file.read_text()

    # With this:
    bucket_name = "your-gcs-bucket-name"
    blob_name = f"patient-data/{patient}.txt"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    document_text = blob.download_as_text()

    # The rest of the function can then use the `document_text` variable
    ```

#### **2. Fetching Patient & Shift Data from BigQuery**

For a truly dynamic system, patient lists, shifts, and medical records can be queried directly from BigQuery.

**Modification Points**:

*   `nurse_handover/agent.py`: Remove hardcoded state in `initialize_state`.
*   `nurse_handover/tools.py`: Update `list_patients`, `list_available_shifts`, and `generate_shift_endorsement` to query BigQuery.

**Steps**:

1.  **Update `list_patients` and `list_available_shifts`**:
    *   In `tools.py`, use the `google-cloud-bigquery` client to run queries against your database to fetch active patients and available shifts for a given nurse.
    *   This removes the need for hardcoded data in `agent.py`.

    ```python
    # Example for list_patients in tools.py
    from google.cloud import bigquery
    client = bigquery.Client()

    def list_patients(tool_context: ToolContext) -> dict[str, Any]:
        # Query BigQuery for a list of all active patients.
        query = f"""
            SELECT patient_id
            FROM `your-project.your_dataset.patients`
            WHERE is_active = TRUE
        """
        query_job = client.query(query)
        patients = [row.patient_id for row in query_job]
        return {"success": "Patients found.", "patients": patients}
    ```

2.  **Update `generate_shift_endorsement`**:
    *   Instead of reading a single text file, this function would query BigQuery for all relevant medical log entries for the given `patient` and `shift` time range.
    *   The collected rows can be concatenated into a single string, which then serves as the input document for the `Summarizer`.

## Evaluation

To validate the agent's performance, you can run the included integration and unit tests.

```bash
make test
```

This test suite performs end-to-end testing to ensure the agent correctly identifies shifts and patients, and verifies that the generated artifacts (`.txt` and `.md` files) are created successfully and adhere to the ISBAR format.

## Deployment

For production use, you can deploy this agent using the [Agent Starter Pack](https://goo.gle/agent-starter-pack), which provides automated CI/CD deployment scripts for services like Google Cloud Run.

```bash
# Install the starter pack and create a new project from this sample
uvx agent-starter-pack create my-nurse-handover -a adk@nurse-handover
```
