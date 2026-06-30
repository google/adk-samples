# Retail AI Location Strategy with Google ADK

A multi-agent AI pipeline for retail site selection, built with [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/) and Gemini.

<table>
  <thead>
    <tr>
      <th colspan="2">Key Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🔍</td>
      <td><strong>Multi-Agent Pipeline:</strong> 7 specialized agents for market research, competitor mapping, gap analysis, strategy synthesis, and report generation.</td>
    </tr>
    <tr>
      <td>🗺️</td>
      <td><strong>Real-World Data:</strong> Integrates Google Maps Places API for competitor mapping and Google Search for live market research.</td>
    </tr>
    <tr>
      <td>🐍</td>
      <td><strong>Code Execution:</strong> Python/pandas analysis for quantitative gap analysis with viability scoring.</td>
    </tr>
    <tr>
      <td>🎨</td>
      <td><strong>AI-Generated Outputs:</strong> Executive HTML reports and infographics via Gemini's native image generation.</td>
    </tr>
    <tr>
      <td>🖥️</td>
      <td><strong>AG-UI Frontend:</strong> Optional interactive dashboard with <a href="https://docs.ag-ui.com/">AG-UI Protocol</a> and <a href="https://docs.copilotkit.ai/">CopilotKit</a> for real-time pipeline visualization.</td>
    </tr>
    <tr>
      <td>🏗️</td>
      <td><strong>Production-Ready:</strong> Deploy to <a href="https://cloud.google.com/run">Cloud Run</a> or <a href="https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime">Agent Runtime</a> via <a href="https://github.com/google/agents-cli">Google Agents CLI</a>.</td>
    </tr>
  </tbody>
</table>

<p align="center">
  <img src="assets/images/main-intro-image.png" alt="Retail AI Location Strategy - System Architecture" width="800">
</p>

## What It Does

Given a location and business type, this pipeline automatically:

- Researches the market using live web search
- Maps competitors using Google Maps Places API
- Calculates viability scores with Python code execution
- Generates strategic recommendations with extended reasoning
- Produces an HTML executive report and visual infographic

## Run this agent

> [!IMPORTANT]
> This agent uses the Google Agents CLI, which is the fastest and easiest way to run and customize this agent.

From the `python` directory, run the following command:

```shell
uvx agent start retail-ai-location-strategy
```

This will start the agent and make it available to call from the command line.

In a separate terminal, you can chat with this agent by running:

```shell
uvx agent chat retail-ai-location-strategy -- -q "I want to open a coffee shop in Indiranagar, Bangalore"
```

<details>
<summary>Alternative: Set up and run manually</summary>

### A. Google AI Studio (Recommended)

You'll need a **[Google AI Studio API Key](https://aistudio.google.com/app/apikey)**.

#### Step 1: Clone Repository
Clone the repository and `cd` into the project directory.

```bash
git clone https://github.com/google/adk-samples.git
cd adk-samples/python/agents/retail-ai-location-strategy
```

#### Step 2: Set Environment Variables
Create a `.env` file in the `app` folder with your API keys (see `.env.example` for reference):

```bash
echo "GOOGLE_GENAI_USE_VERTEXAI=FALSE" >> app/.env
echo "GOOGLE_API_KEY=YOUR_AI_STUDIO_API_KEY" >> app/.env
echo "MAPS_API_KEY=YOUR_MAPS_API_KEY" >> app/.env
```

#### Step 3: Install & Run
From the `retail-ai-location-strategy` directory, install dependencies and start the server.

```bash
make install && make dev
```

#### What You'll See

1. Open `http://localhost:8501` in your browser
2. Select **"app"** from the agent dropdown
3. Type a query like: *"I want to open a coffee shop in Indiranagar, Bangalore"*
4. Watch the 7-stage pipeline execute:
   - **Intake** → Extract location and business type
   - **Market Research** → Web search for demographics and trends
   - **Competitor Mapping** → Google Maps Places API for competitors
   - **Gap Analysis** → Python code execution for viability scores
   - **Strategy Advisor** → Extended reasoning for recommendations
   - **Report Generator** → HTML executive report
   - **Infographic Generator** → Visual summary image

<p align="center">
  <img src="assets/gifs/adk-web-demo.gif" alt="ADK Web Demo" width="700">
</p>

Your agent is now running at `http://localhost:8501`.

---

### B. Google Cloud Vertex AI (via Google Agents CLI)

Use the [Google Agents CLI](https://github.com/google/agents-cli) to create a production-ready project with deployment scripts. This is ideal for cloud deployment scenarios.

You'll need: **[Google Cloud SDK](https://cloud.google.com/sdk/docs/install)** and a **Google Cloud Project** with the **Vertex AI API** enabled.

<details>
<summary>📁 Alternative: Using the cloned repository with Vertex AI</summary>

If you've already cloned the repository (as in Option A) and want to use Vertex AI instead of AI Studio, create a `.env` file in the `app` folder with:

```bash
echo "GOOGLE_GENAI_USE_VERTEXAI=TRUE" >> app/.env
echo "GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID" >> app/.env
echo "GOOGLE_CLOUD_LOCATION=us-central1" >> app/.env
echo "MAPS_API_KEY=YOUR_MAPS_API_KEY" >> app/.env
```

Make sure you're authenticated with Google Cloud:
```bash
gcloud auth application-default login
```

Then run `make install && make dev` to start the agent.
</details>

#### Step 1: Create Project from Template

This uses the [Google Agents CLI](https://github.com/google/agents-cli) to create a new directory (`my-retail-agent`) with all the necessary code.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-retail-agent` with your project name):

```bash
agents-cli create my-retail-agent -a adk@retail-ai-location-strategy
```

The Google Agents CLI will prompt you to select deployment options and provides additional production-ready features including automated CI/CD deployment scripts.

#### Step 2: Install & Run
Navigate into your **newly created project folder**, then install dependencies and start the server.
```bash
cd my-retail-agent && make install && make dev
```
Your agent is now running at `http://localhost:8501`.

## Cloud Deployment

> **Note:** Cloud deployment applies only to projects created with **google-agents-cli** (Option B).

**Prerequisites:**
```bash
gcloud components update
gcloud config set project YOUR_PROJECT_ID
```

Deploy with the built-in [adk-web](https://github.com/google/adk-web) interface:

```bash
make deploy IAP=true
```

After deployment, grant users access to your IAP-protected service by following the [Manage User Access](https://cloud.google.com/run/docs/securing/identity-aware-proxy-cloud-run#manage_user_or_group_access) documentation.

For production deployments with CI/CD, see the [Google Agents CLI Development Guide](https://github.com/google/agents-cli).

---

## Agent Details

| Attribute | Description |
| :--- | :--- |
| **Interaction Type** | Workflow |
| **Complexity** | Advanced |
| **Agent Type** | Multi Agent (Sequential Pipeline) |
| **Components** | Multi-agent, Function calling, Web search, Google Maps API, Code execution, Image generation |
| **Vertical** | Retail / Real Estate |

<p align="center">
  <img src="assets/images/agent-tools.png" alt="Agent Tools Integration" width="700">
</p>

## Model Configuration

This agent supports multiple Gemini model families. Edit `app/config.py` to switch models based on your access and quota:

| Model Option | Text Models | Image Model | Notes |
|--------------|-------------|-------------|-------|
| **Gemini 2.5 Pro** (default) | `gemini-2.5-pro` | `gemini-3-pro-image-preview` | **Recommended** - Stable, production-ready |
| **Gemini 3 Pro Preview** | `gemini-3-pro-preview` | `gemini-3-pro-image-preview` | Recently launched - may throw 503 "model overloaded" errors |
| **Gemini 2.5 Flash** | `gemini-2.5-flash` | `gemini-2.0-flash-exp` | Fastest, lowest cost |

**Gemini 3 Documentation:**
- [Vertex AI - Get started with Gemini 3](https://cloud.google.com/vertex-ai/generative-ai/docs/start/get-started-with-gemini-3)
- [Google AI - Gemini 3 API](https://ai.google.dev/gemini-api/docs/gemini-3)

To use Gemini 3 text models, uncomment Option 2 in `app/config.py`:

```python
# app/config.py

# Comment out Option 1 (2.5 Pro)
# FAST_MODEL = "gemini-2.5-pro"
# ...

# Uncomment Option 2 (3 Pro Preview)
FAST_MODEL = "gemini-3-pro-preview"
PRO_MODEL = "gemini-3-pro-preview"
CODE_EXEC_MODEL = "gemini-3-pro-preview"
IMAGE_MODEL = "gemini-3-pro-image-preview"
```

> **Note:** If you encounter `503 UNAVAILABLE - model overloaded` errors with Gemini 3, switch back to Gemini 2.5 Pro for better reliability.

---

## AG-UI Frontend (Optional)

Want a richer experience beyond the default ADK web UI? This agent includes an optional **[AG-UI Protocol](https://docs.ag-ui.com/)** frontend built with [CopilotKit](https://docs.copilotkit.ai/) that provides:

- **Real-time Pipeline Timeline**: Watch the 7-stage analysis unfold with collapsible steps
- **Generative UI**: Rich visualizations appear in the chat as the agent works
- **Interactive Dashboard**: Location scores, competitor stats, market characteristics
- **Bidirectional State Sync**: Frontend and ADK agent share state in real-time

<p align="center">
  <img src="assets/images/ag-ui-sync.png" alt="AG-UI Bidirectional State Sync" width="650">
</p>

### Quick Start

```bash
# First time: Install frontend dependencies
make ag-ui-install

# Run both backend and frontend servers
make ag-ui
```

This starts:
- **Backend** at `http://localhost:8000` (FastAPI + ADK agent)
- **Frontend** at `http://localhost:3000` (Next.js + CopilotKit)

Open `http://localhost:3000` to see the interactive dashboard.

<p align="center">
  <img src="assets/gifs/ag-ui-demo.gif" alt="AG-UI Frontend Demo" width="700">
</p>

<details>
<summary>Manual Setup (Alternative)</summary>

```bash
# Terminal 1: Start the backend
cd app/frontend/backend
uv pip install -r requirements.txt
uv run main.py
# Runs at http://localhost:8000

# Terminal 2: Start the frontend
cd app/frontend
npm install
cp .env.local.example .env.local
npm run dev
# Runs at http://localhost:3000
```
</details>

See [app/frontend/README.md](app/frontend/README.md) for detailed frontend documentation.

---

## Example Prompts

| Region | Location | Business | Example Prompt |
|--------|----------|----------|----------------|
| Asia | Bangalore, India | Coffee Shop | "I want to open a coffee shop in Indiranagar, Bangalore" |
| Asia | Tokyo, Japan | Ramen Restaurant | "Analyze Shibuya, Tokyo for opening a ramen restaurant" |
| Asia | Singapore | Bubble Tea | "Where should I open a bubble tea shop in Orchard Road, Singapore?" |
| Americas | Austin, Texas | Fitness Studio | "Where should I open a fitness studio in Austin, Texas?" |
| Americas | Mexico City | Taco Restaurant | "Analyze Roma Norte, Mexico City for a taco restaurant" |
| Americas | Toronto, Canada | Craft Brewery | "Help me find a location for a craft brewery in Toronto's Distillery District" |
| Europe | London, UK | Bookstore Cafe | "Help me find the best location for a bookstore cafe in Shoreditch, London" |
| Europe | Berlin, Germany | Vegan Restaurant | "Analyze Berlin's Kreuzberg for opening a vegan restaurant" |
| Middle East | Dubai, UAE | Bakery | "I'm planning to open a bakery in Dubai Marina" |
| Oceania | Sydney, Australia | Juice Bar | "Analyze the market for a juice bar in Bondi Beach, Sydney" |

---

## Architecture

<p align="center">
  <img src="assets/images/pipeline-architecture.png" alt="Pipeline Architecture" width="700">
</p>

The pipeline is built as a `SequentialAgent` that orchestrates 7 specialized sub-agents, each handling a specific phase of the analysis.

### State Flow

Each agent reads from and writes to a shared session state, enabling seamless data flow between stages:

<p align="center">
  <img src="assets/images/data-flow.png" alt="Data Flow Between Agents" width="650">
</p>

---

## Project Structure

```
retail-ai-location-strategy/
├── Makefile                          # Build and run commands
├── pyproject.toml                    # Dependencies and package config
├── .env.example                      # Environment template
├── README.md                         # This file
├── DEVELOPER_GUIDE.md                # Detailed developer documentation
│
├── notebook/                         # Original Gemini API notebook
│   └── retail_ai_location_strategy_gemini_3.ipynb
│
└── app/                              # Agent package (exported as root_agent)
    ├── __init__.py                   # Exports root_agent for ADK discovery
    ├── agent.py                      # SequentialAgent pipeline definition
    ├── config.py                     # Model selection and retry config
    ├── .env                          # Environment variables (from .env.example)
    │
    ├── sub_agents/                   # 7 specialized agents
    │   ├── competitor_mapping/
    │   │   ├── __init__.py
    │   │   └── agent.py
    │   ├── gap_analysis/
    │   │   ├── __init__.py
    │   │   └── agent.py
    │   ├── infographic_generator/
    │   │   ├── __init__.py
    │   │   └── agent.py
    │   ├── intake_agent/
    │   │   ├── __init__.py
    │   │   └── agent.py
    │   ├── market_research/
    │   │   ├── __init__.py
    │   │   └── agent.py
    │   ├── report_generator/
    │   │   ├── __init__.py
    │   │   └── agent.py
    │   └── strategy_advisor/
    │       ├── __init__.py
    │       └── agent.py
    │
    ├── tools/                        # Custom function tools
    │   ├── places_search.py          # Google Maps Places API
    │   ├── html_report_generator.py  # Executive report generation
    │   └── image_generator.py        # Infographic generation
    │
    ├── schemas/                      # Pydantic output schemas
    ├── callbacks/                    # Pipeline lifecycle callbacks
    └── frontend/                     # AG-UI interactive dashboard
```

---

## Learn More

For detailed documentation, see **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)**:

- [The Business Problem](DEVELOPER_GUIDE.md#the-business-problem) - Why this exists
- [Architecture Deep Dive](DEVELOPER_GUIDE.md#architecture-deep-dive) - State flow and agent communication
- [Agents and Tools](DEVELOPER_GUIDE.md#agents-and-tools) - Sub-agents, tools, callbacks, schemas
- [Configuration](DEVELOPER_GUIDE.md#configuration) - Model selection and retry options
- [Troubleshooting](DEVELOPER_GUIDE.md#troubleshooting) - Common issues and fixes

## Troubleshooting

If you encounter issues while setting up or running this agent, here are some resources to help you troubleshoot:
- [ADK Documentation](https://google.github.io/adk-docs/): Comprehensive documentation for the Agent Development Kit
- [Vertex AI Authentication Guide](https://cloud.google.com/vertex-ai/docs/authentication): Detailed instructions for setting up authentication
- [Google Agents CLI Troubleshooting](https://github.com/google/agents-cli): Common issues

---

## Authors

Based on the original [Retail AI Location Strategy notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retail/retail_ai_location_strategy_gemini_3.ipynb) by [Lavi Nigam](https://github.com/lavinigam-gcp) and [Deepak Moonat](https://github.com/dmoonat).

---

## Disclaimer

This agent sample is provided for illustrative purposes only. It serves as a basic example of an agent and a foundational starting point for individuals or teams to develop their own agents.

Users are solely responsible for any further development, testing, security hardening, and deployment of agents based on this sample. We recommend thorough review, testing, and the implementation of appropriate safeguards before using any derived agent in a live or critical system.

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
