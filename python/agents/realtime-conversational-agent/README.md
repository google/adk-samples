# Real-Time Conversational Agent Template

## Overview

This repository provides a full-stack, reusable template for building real-time, multimodal conversational AI agents. It uses leverages the **Agent Development Kit (ADK)** on a FastAPI backend and a Next.js (React) frontend.

This template handles all the complex WebSocket and media streaming "plumbing," allowing you to focus on what matters: building your agent's logic and UI.

## Agent Details

| Feature | Description |
| --- | --- |
| **Interaction Type** | Conversational |
| **Complexity**  | Medium |
| **Agent Type**  | Single Agent |
| **Components**  | Live API (Native Audio) |
| **Vertical**  | Reusable across industry verticals |

## Features
<img src="realtime-conversational-agent.png" alt="realtime conversational agent" width="800"/>

- **Real-Time, Bidirectional Audio:** Streams user microphone input to the agent and streams the agent's synthesized voice back to the client.
- **Live Video Feed:** Supports streaming from a user's camera or screen share for the agent to analyze.
- **Live Transcriptions:** Displays a real-time transcript of both the user's speech (STT) and the agent's speech (TTS).
- **Configurable Persona:** The agent's identity and instructions are easily configured.

## Use Cases & Examples
This template is the foundation for any application that requires an AI to see, hear, and talk in real-time.
- **Real-Time Tutors:** An agent that can watch you solve a math problem (via screen share) and talk you through it. *This is the default behavior of this example agent.*
- **Live Customer Support:** An agent that can visually guide a user through a website or product setup.
- **Accessibility Tools:** A "be my eyes" agent that can describe a user's surroundings or the content of their screen.
- **Interactive Assistant:** An agent that pairs with you, watches you work, and provides real-time feedback or assistance.

## Setup and Installation

### Prerequisites

- Python 3.11+
- Node.js v22+
- uv for dependency management and packaging
  - See the official [uv website](https://docs.astral.sh/uv/) for installation.

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Google Agents CLI (recommended)

Use the [Google Agents CLI](https://github.com/google/agents-cli) to scaffold a production-ready project and choose your deployment target ([Agent Runtime](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime) or [Cloud Run](https://cloud.google.com/run)), with CI/CD and other production features.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-realtime-agent` with your project name):

```bash
agents-cli create my-realtime-agent -a adk@realtime-conversational-agent
```

The Google Agents CLI will prompt you to select deployment options and set up your Google Cloud project.

---

<details>
<summary>Alternative: Local development (run from this sample repo)</summary>

### Local Setup
Follow these instructions to get the client and server running on your local machine.

#### Prerequisites
1. Node.js: v22 or later
2. Python: 3.11 or later
3. `uv` (Recommended): Python package manager
    ```
    pip install uv
    ```
4. Platform setup

    Choose a platform from either **Google AI Studio** or **Google Cloud Vertex AI**:
    - Option 1: Google AI Studio (Default)
        - Get a `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/). This is the simplest way to get started.
    - Option 2: Google Cloud / Vertex AI
        - Create a Google Cloud Project and enable the [Vertex AI API](https://console.cloud.google.com/vertex-ai).
        - Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
        - Authenticate your local environment by running:
            ```
            gcloud auth login
            ```


#### Server Setup
The server handles the AI agent logic and WebSocket connections.

1. **Navigate to the server directory:**
    ```
    cd server
    ```

2. **Create and activate a virtual environment:**
    ```
    uv venv
    source .venv/bin/activate
    ```

3. **Install Python dependencies:**
    ```
    uv pip install .
    ```

4. **Set SSL Certificate File:**
    ```
    export SSL_CERT_FILE=$(python -m certifi)
    ```

5. **Create your environment file:**

    Create a new file named .env in the server/ directory. Use one of the two templates below based on your authentication method.

    `server/.env` **(Option 1: Google AI Studio)**

    ```
    # --- AI Studio ---
    GOOGLE_GENAI_USE_VERTEXAI=FALSE

    # Get this from Google AI Studio
    GOOGLE_API_KEY="PASTE_YOUR_ACTUAL_API_KEY_HERE"

    # Configuration for the agent's voice (Example: 'Puck' for gemini-live-2.5-flash)
    AGENT_VOICE="Puck"
    AGENT_LANGUAGE="en-US"
    ```

    `server/.env` **(Option 2: Google Cloud / Vertex AI)**

    ```
    # --- Vertex AI ---
    GOOGLE_GENAI_USE_VERTEXAI=TRUE

    # Your Google Cloud project ID
    GOOGLE_CLOUD_PROJECT="PASTE_YOUR_ACTUAL_PROJECT_ID"
    # Your Vertex AI location (e.g., us-east1)
    GOOGLE_CLOUD_LOCATION="us-east1"

    # Configuration for the agent's voice (Example: 'Puck' for gemini-live-2.5-flash)
    AGENT_VOICE="Puck"
    AGENT_LANGUAGE="en-US"
    ```
6. **Run the server:**

    ```
    uvicorn main:app --reload
    ```

    The server will be running at `http://127.0.0.1:8000`.

#### Client Setup
The client is the Next.js application that the user interacts with.

1. **Open a new terminal** and navigate to the client directory:
    ```
    cd client
    ```

2. **Install Node.js dependencies:**
    ```
    npm install
    ```

3. **Run the client:**
    ```
    npm run dev
    ```

    The client will be running at `http://localhost:3000`.

4. **Open the app:**

    Open `http://localhost:3000` in your browser. You can now click the microphone icon to start a session!

</details>

## How to Customize Your Agent
This repository is designed for easy reuse. You don't need to change any Python code to completely change your agent's persona.

**Simply edit the `AGENT_INSTRUCTION` in your `server/example_agent/prompts.py` file.**

### Example: Expert Math Tutor -> Generic Assistant
To turn your "Math Tutor" agent into a "Generic Assistant", stop your server, replace the AGENT_INSTRUCTION in `server/example_agent/prompts.py` with the following, and restart the server.

```
AGENT_INSTRUCTION="You are a helpful and friendly AI assistant. Keep your responses concise."
```


## Disclaimer

This agent sample is provided for illustrative purposes only and is not intended for production use. It serves as a basic example of an agent and a foundational starting point for individuals or teams to develop their own agents.

This sample has not been rigorously tested, may contain bugs or limitations, and does not include features or optimizations typically required for a production environment (e.g., robust error handling, security measures, scalability, performance considerations, comprehensive logging, or advanced configuration options).

Users are solely responsible for any further development, testing, security hardening, and deployment of agents based on this sample. We recommend thorough review, testing, and the implementation of appropriate safeguards before using any derived agent in a live or critical system.