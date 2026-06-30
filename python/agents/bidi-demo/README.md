# ADK Gemini Live API Toolkit Demo

A working demonstration of real-time bidirectional streaming with Google's Agent Development Kit (ADK). This FastAPI application showcases WebSocket-based communication with Gemini models, supporting multimodal requests (text, audio, and image/video input) and flexible responses (text or audio output).

![bidi-demo-screen](assets/bidi-demo-screen.png)

## Overview

This demo implements the complete ADK bidirectional streaming lifecycle:

1. **Application Initialization**: Creates `Agent`, `SessionService`, and `Runner` at startup
2. **Session Initialization**: Establishes `Session`, `RunConfig`, and `LiveRequestQueue` per connection
3. **Bidirectional Streaming**: Concurrent upstream (client → queue) and downstream (events → client) tasks
4. **Graceful Termination**: Proper cleanup of `LiveRequestQueue` and WebSocket connections

## Features

- **WebSocket Communication**: Real-time bidirectional streaming via `/ws/{user_id}/{session_id}`
- **Multimodal Requests**: Text, audio, and image/video input with automatic audio transcription
- **Flexible Responses**: Text or audio output, automatically determined based on model architecture
- **Session Resumption**: Reconnection support configured via `RunConfig`
- **Concurrent Tasks**: Separate upstream/downstream async tasks for optimal performance
- **Interactive UI**: Web interface with event console for monitoring Live API events
- **Google Search Integration**: Agent equipped with `google_search` tool

## Architecture

The application follows ADK's recommended concurrent task pattern:

```
┌─────────────┐         ┌──────────────────┐         ┌─────────────┐
│             │         │                  │         │             │
│  WebSocket  │────────▶│ LiveRequestQueue │────────▶│  Live API   │
│   Client    │         │                  │         │   Session   │
│             │◀────────│   run_live()     │◀────────│             │
└─────────────┘         └──────────────────┘         └─────────────┘
  Upstream Task              Queue              Downstream Task
```

- **Upstream Task**: Receives WebSocket messages and forwards to `LiveRequestQueue`
- **Downstream Task**: Processes `run_live()` events and sends to WebSocket client

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/)
- Google API key (for Gemini Live API) or Google Cloud project (for Vertex AI Live API)

**Installing uv (if not already installed):**

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

### Google Agents CLI (Recommended)

Use the [Google Agents CLI](https://github.com/google/agents-cli) to create a production-ready version of this agent with additional deployment options.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-bidi-demo` with your project name):

```bash
agents-cli create my-bidi-demo -a adk@bidi-demo
```

The Google Agents CLI will prompt you to select deployment options and provides additional production-ready features including automated CI/CD deployment scripts.

### Manual Setup

### 1. Navigate to Demo Directory

```bash
cd src/bidi-demo
```

### 2. Install Dependencies

**Using uv (recommended):**

```bash
uv sync
```

This automatically creates a virtual environment, installs all dependencies, and generates a lock file for reproducible builds.

**Using pip (alternative):**

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 3. Configure Environment Variables

Create or edit `app/.env` with your credentials:

```bash
# Choose your Live API platform
GOOGLE_GENAI_USE_VERTEXAI=FALSE

# For Gemini Live API (when GOOGLE_GENAI_USE_VERTEXAI=FALSE)
GOOGLE_API_KEY=your_api_key_here

# For Vertex AI Live API (when GOOGLE_GENAI_USE_VERTEXAI=TRUE)
# GOOGLE_CLOUD_PROJECT=your_project_id
# GOOGLE_CLOUD_LOCATION=us-east1

# Model selection (optional, defaults to native audio model)
# See "Supported Models" section below for available model names
DEMO_AGENT_MODEL=gemini-2.5-flash-native-audio-preview-12-2025
```

#### Getting API Credentials

**Gemini Live API:**
1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Create an API key
3. Set `GOOGLE_API_KEY` in `.env`

**Vertex AI Live API:**
1. Enable Vertex AI API in [Google Cloud Console](https://console.cloud.google.com)
2. Set up authentication using `gcloud auth application-default login`
3. Set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` in `.env`
4. Set `GOOGLE_GENAI_USE_VERTEXAI=TRUE`

### 4. Set SSL Certificate Path

Set the SSL certificate file path for secure connections:

```bash
# If using uv
export SSL_CERT_FILE=$(uv run python -m certifi)

# If using pip with activated venv
export SSL_CERT_FILE=$(python -m certifi)
```

## Running the Demo

### Start the Server

From the `src/bidi-demo` directory, first change to the `app` subdirectory:

```bash
cd app
```

> **Note:** You must run from inside the `app` directory so Python can find the `google_search_agent` module. Running from the parent directory will fail with `ModuleNotFoundError: No module named 'google_search_agent'`.

**Using uv (recommended):**

```bash
uv run --project .. uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Using pip (with activated venv):**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-restart on code changes during development.

#### Background Mode (Testing/Production)

To run in background with log output:

```bash
# Using uv (from app directory)
uv run --project .. uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# Using pip (from app directory)
uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

To check the server log:

```bash
tail -f server.log  # Follow log in real-time
```

To stop the background server:

```bash
kill $(lsof -ti:8000)
```

### Access the Application

Open your browser and navigate to:

```
http://localhost:8000
```

## Usage

### Text Mode

1. Type your message in the input field
2. Click "Send" or press Enter
3. Watch the event console for Live API events
4. Receive streamed responses in real-time

### Audio Mode

1. Click "Start Audio" to begin voice interaction
2. Speak into your microphone
3. Receive audio responses with real-time transcription
4. Click "Stop Audio" to end the audio session

## WebSocket API

### Endpoint

```
ws://localhost:8000/ws/{user_id}/{session_id}
```

**Path Parameters:**
- `user_id`: Unique identifier for the user
- `session_id`: Unique identifier for the session

**Response Modality:**
- Automatically determined based on model architecture
- Native audio models use AUDIO response modality
- Half-cascade models use TEXT response modality

### Message Format

**Client → Server (Text):**
```json
{
  "type": "text",
  "text": "Your message here"
}
```

**Client → Server (Image):**
```json
{
  "type": "image",
  "data": "base64_encoded_image_data",
  "mimeType": "image/jpeg"
}
```

**Client → Server (Audio):**
- Send raw binary frames (PCM audio, 16kHz, 16-bit)

**Server → Client:**
- JSON-encoded ADK `Event` objects
- See [ADK Events Documentation](https://google.github.io/adk-docs/) for event schemas

## Project Structure

```
bidi-demo/
├── app/
│   ├── google_search_agent/      # Agent definition module
│   │   ├── __init__.py           # Package exports
│   │   └── agent.py              # Agent configuration
│   ├── main.py                   # FastAPI application and WebSocket endpoint
│   ├── .env                      # Environment configuration (not in git)
│   └── static/                   # Frontend files
│       ├── index.html            # Main UI
│       ├── css/
│       │   └── style.css         # Styling
│       └── js/
│           ├── app.js                    # Main application logic
│           ├── audio-player.js           # Audio playback
│           ├── audio-recorder.js         # Audio recording
│           ├── pcm-player-processor.js   # Audio processing
│           └── pcm-recorder-processor.js # Audio processing
├── agent_engine/                 # Agent Engine deployment scripts
│   ├── deploy.py               # Deploy agent to Agent Engine
│   ├── test.py                 # Test deployed agent via bidi streaming
│   ├── cleanup.py              # Delete deployed agent
│   └── .env.template           # Environment template for Agent Engine
├── pyproject.toml               # Python project configuration
├── Dockerfile                   # Cloud Run container image
├── .dockerignore                # Docker build exclusions
└── README.md                    # This file
```

## Code Overview

### Agent Definition (app/google_search_agent/agent.py)

The agent is defined in a separate module following ADK best practices:

```python
agent = Agent(
    name="google_search_agent",
    model=os.getenv("DEMO_AGENT_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"),
    tools=[google_search],
    instruction="You are a helpful assistant that can search the web."
)
```

### Application Initialization (app/main.py:37-50)

```python
from google_search_agent.agent import agent

app = FastAPI()
session_service = InMemorySessionService()
runner = Runner(app_name="bidi-demo", agent=agent, session_service=session_service)
```

### WebSocket Handler (app/main.py:65-209)

The WebSocket endpoint implements the complete bidirectional streaming pattern:

1. **Accept Connection**: Establish WebSocket connection
2. **Configure Session**: Create `RunConfig` with automatic modality detection
3. **Initialize Queue**: Create `LiveRequestQueue` for message passing
4. **Start Concurrent Tasks**: Launch upstream and downstream tasks
5. **Handle Cleanup**: Close queue in `finally` block

### Concurrent Tasks

**Upstream Task** (app/main.py:125-172):
- Receives WebSocket messages (text, image, or audio binary)
- Converts to ADK format (`Content` or `Blob`)
- Sends to `LiveRequestQueue` via `send_content()` or `send_realtime()`

**Downstream Task** (app/main.py:174-187):
- Calls `runner.run_live()` with queue and config
- Receives `Event` stream from Live API
- Serializes events to JSON and sends to WebSocket

## Configuration

### Supported Models

The demo supports any Gemini model compatible with Live API:

**Native Audio Models** (recommended for voice):
- `gemini-2.5-flash-native-audio-preview-12-2025` (Gemini Live API)
- `gemini-live-2.5-flash-native-audio` (Vertex AI)

Set the model via `DEMO_AGENT_MODEL` in `.env` or modify `app/google_search_agent/agent.py`.

For the latest model availability and features:
- **Gemini Live API**: Check the [official Gemini API models documentation](https://ai.google.dev/gemini-api/docs/models)
- **Vertex AI Live API**: Check the [official Vertex AI models documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models)

### RunConfig Options

The demo automatically configures bidirectional streaming based on model architecture (app/main.py:76-104):

**For Native Audio Models** (containing "native-audio" in model name):
```python
run_config = RunConfig(
    streaming_mode=StreamingMode.BIDI,
    response_modalities=["AUDIO"],
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    session_resumption=types.SessionResumptionConfig()
)
```

**For Half-Cascade Models** (other models):
```python
run_config = RunConfig(
    streaming_mode=StreamingMode.BIDI,
    response_modalities=["TEXT"],
    input_audio_transcription=None,
    output_audio_transcription=None,
    session_resumption=types.SessionResumptionConfig()
)
```

The modality detection is automatic based on the model name. Native audio models use AUDIO response modality with transcription enabled, while half-cascade models use TEXT response modality for better performance.

## Troubleshooting

### Connection Issues

**Problem**: WebSocket fails to connect

**Solutions**:
- Verify API credentials in `app/.env`
- Check console for error messages
- Ensure uvicorn is running on correct port

### Audio Not Working

**Problem**: Audio input/output not functioning

**Solutions**:
- Grant microphone permissions in browser
- Verify browser supports Web Audio API
- Check that audio model is configured (native audio model required)
- Review browser console for errors

### Model Errors

**Problem**: "Model not found" or quota errors

**Solutions**:
- Verify model name matches your platform (Gemini vs Vertex AI)
- Check API quota limits in console
- Ensure billing is enabled (for Vertex AI)

## Development

### Code Formatting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration is in `pyproject.toml`.

```bash
# Lint and auto-fix
uvx ruff check --fix .

# Format
uvx ruff format .
```

To check without making changes:

```bash
uvx ruff check .
uvx ruff format --check .
```

## Deployment to Cloud Run

### 1. Prerequisites

- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`) installed and configured
- A Google Cloud project with Cloud Run API enabled
- Vertex AI API enabled (if using Vertex AI Live API)

### 2. Configure Environment

Use `app/.env` as the source of truth. Export the values before deploying:

```bash
set -a
source app/.env
set +a
```

### 3. Deploy

From the `src/bidi-demo` directory, deploy with `gcloud run deploy`:

**Vertex AI Live API mode** (`GOOGLE_GENAI_USE_VERTEXAI=TRUE`):

```bash
gcloud run deploy bidi-demo \
  --source . \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --region "${GOOGLE_CLOUD_LOCATION}" \
  --allow-unauthenticated \
  --timeout 3600 \
  --min-instances 1 \
  --max-instances 1 \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI}",GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT}",GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION}",DEMO_AGENT_MODEL="${DEMO_AGENT_MODEL}"
```

**Gemini Live API mode** (`GOOGLE_GENAI_USE_VERTEXAI=FALSE`):

```bash
gcloud run deploy bidi-demo \
  --source . \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --region "${GOOGLE_CLOUD_LOCATION}" \
  --allow-unauthenticated \
  --timeout 3600 \
  --min-instances 1 \
  --max-instances 1 \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI}",GOOGLE_API_KEY="${GOOGLE_API_KEY}",GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT}",GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION}",DEMO_AGENT_MODEL="${DEMO_AGENT_MODEL}"
```

Cloud Run deployments are not instant. A normal deploy can take a few minutes while source upload, Cloud Build, image rollout, and revision health checks complete.

### 4. Make the Service Public (If Needed)

In some projects, `--allow-unauthenticated` finishes with an IAM warning instead of granting public access. If that happens:

```bash
gcloud run services add-iam-policy-binding bidi-demo \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --region "${GOOGLE_CLOUD_LOCATION}" \
  --member=allUsers \
  --role=roles/run.invoker
```

### 5. Validate the Deployment

Get the service URL:

```bash
gcloud run services describe bidi-demo \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --region "${GOOGLE_CLOUD_LOCATION}" \
  --format='value(status.url)'
```

Open the app in your browser at the returned URL.

### 6. Clean Up Old Revisions

After each successful deploy, delete older revisions and keep only the newest five:

```bash
for rev in $(gcloud run revisions list \
  --service bidi-demo \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --region "${GOOGLE_CLOUD_LOCATION}" \
  --format='value(metadata.name)' | tail -n +6); do
  gcloud run revisions delete "$rev" \
    --project "${GOOGLE_CLOUD_PROJECT}" \
    --region "${GOOGLE_CLOUD_LOCATION}" \
    --quiet
done
```

### Deployment Checklist

- The root URL serves the HTML app
- Text chat works end-to-end
- Audio mode connects and streams (native audio model required)
- WebSocket connection stays connected during conversation
- Image input is accepted and processed

## Deployment to Agent Engine

You can deploy the agent to [Vertex AI Agent Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview) for managed bidirectional streaming. Agent Engine handles agent scaling, infrastructure, and lifecycle management on Google Cloud.

### Agent Engine vs Cloud Run

Agent Engine hosts **agents**, not web applications. It does not expose a public URL or serve static files — access is via authenticated Vertex AI SDK WebSocket connections only. This makes it well suited for backend agent hosting, but not for serving the full bidi-demo webapp directly.

| | Cloud Run | Agent Engine |
|---|---|---|
| **What's deployed** | Full webapp (FastAPI + frontend + agent) | Agent only |
| **Public URL** | Yes | No |
| **Static files** | Yes | No |
| **Access method** | Browser WebSocket | Vertex AI SDK (authenticated) |
| **Use case** | Self-contained demo, public apps | Managed agent backend, split architectures |

To combine Agent Engine with a browser-facing frontend, use a split architecture where Cloud Run serves the UI and proxies WebSocket connections to Agent Engine:

```
Browser ──ws──► Cloud Run (proxy) ──SDK──► Agent Engine (agent)
```

### How the Scripts Work

Three scripts in `agent_engine/` manage the Agent Engine lifecycle. All read configuration from `app/.env`.

**`agent_engine/deploy.py`** — Wraps the existing agent from `app/google_search_agent/agent.py` in an `AdkApp`, then deploys it to Agent Engine with `EXPERIMENTAL` server mode (required for bidi streaming). The deployed agent's resource name is saved to `agent_resource_name.txt` for use by the other scripts.

**`agent_engine/test.py`** — Connects to the deployed agent via `client.aio.live.agent_engines.connect()` and sends text queries as `LiveRequest` objects over a WebSocket. Supports two modes: automated (runs preset queries) and interactive (`--interactive` flag for a chat loop). Responses arrive as ADK `Event` objects containing audio or text.

**`agent_engine/cleanup.py`** — Reads the resource name from `agent_resource_name.txt`, deletes the deployed agent from Agent Engine, and removes the file.

### 1. Prerequisites

- Google Cloud project with Vertex AI API enabled
- Authenticated via `gcloud auth application-default login`
- A Cloud Storage bucket for staging

### 2. Install Agent Engine Dependencies

```bash
uv sync --extra agent-engine
```

This installs `google-cloud-aiplatform[agent_engines,adk]`, `numpy`, and `websockets` as optional dependencies.

### 3. Configure Environment

Copy the template and edit with your project details:

```bash
cp agent_engine/.env.template app/.env
```

Then edit `app/.env`:

```bash
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-east1
STAGING_BUCKET=gs://your-bucket-name

# Must use a Vertex AI Live API model
DEMO_AGENT_MODEL=gemini-live-2.5-flash-native-audio
```

> **Note:** Agent Engine requires `GOOGLE_GENAI_USE_VERTEXAI=TRUE` and a Vertex AI model. The `STAGING_BUCKET` is a Cloud Storage bucket used to stage deployment artifacts. The `GOOGLE_API_KEY` is not needed — Agent Engine uses Application Default Credentials.

### 4. Deploy

```bash
uv run agent_engine/deploy.py
```

The resource name is saved to `agent_resource_name.txt`.

### 5. Test

```bash
# Automated test
uv run agent_engine/test.py

# Interactive chat
uv run agent_engine/test.py --interactive
```

### 6. Cleanup

```bash
uv run agent_engine/cleanup.py
```

## Additional Resources

- **ADK Documentation**: https://google.github.io/adk-docs/
- **Gemini Live API**: https://ai.google.dev/gemini-api/docs/live
- **Vertex AI Live API**: https://cloud.google.com/vertex-ai/generative-ai/docs/live-api
- **ADK GitHub Repository**: https://github.com/google/adk-python

## License

Apache 2.0 - See repository LICENSE file for details.
