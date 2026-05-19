# 🧠 Economic Research Agent (ERA)

[![Level 3 Maturity](https://img.shields.io/badge/Maturity-Level%203%20Structural-blueviolet)](https://github.com/google/agents-cli)
[![Framework-Atomic Agents](https://img.shields.io/badge/Framework-Atomic%20Agents-blue)](https://github.com/google/agents-cli)
[![ADK-Enabled](https://img.shields.io/badge/ADK-v2.0-green)](https://github.com/google/adk)
[![Live-API](https://img.shields.io/badge/Live--API-Grounded-orange)](#)

An enterprise-grade, **Multi-Agent intelligence** for high-fidelity regional economic analysis, labor market evaluation, and cross-industry site selection. Upgraded to **Agent Runtime (ADK 2.0 / AdkApp)** with a 100% Live-API grounded architecture and **Zero LangChain/LangGraph dependencies** for massive speed-up.

---

## A. Overview & Functionalities

The **Economic Research Agent (ERA)** is a production-grade site-selection and market intelligence consultant. Designed for corporate research, ERA automates comparative metropolitan cost-modeling by orchestrating live public APIs (FRED, BLS, CENSUS, HUD, EIA) into McKinsey-style strategic briefs—now upgraded with an automated **Auditor Judge Agent** (via Serper Live Search) for zero-hallucination verification.

### 📊 Agent Details

| Feature | Specifics |
| :--- | :--- |
| **Interaction Type** | Conversational / Strategic Consultant |
| **Complexity** | Advanced (Multi-Source Synthesis) |
| **Agent Type** | Single Agent (ADK Class-Based) |
| **Vertical** | Economic Development / Real Estate / Finance |
| **Grounding** | 100% Live-API (FRED, BLS, HUD, Census, BEA) |

### 💬 Example Interaction (Strategic Consulting)

**User**: "Compare Austin and Raleigh for a new data center HQ."

**ERA**: "I am analyzing technical and fiscal ROI for NAICS 518210 (Data Processing). 
1. **Utility Benchmarking**: Texas (ERCOT) industrial rates average 8.2c/kWh, while North Carolina (Duke Energy) is 7.5c/kWh.
2. **Talent Pipeline**: UT Austin graduated 1,200 CS majors in 2023 vs. 850 from NC State.
3. **Fiscal Climate**: NC has a scheduled corporate tax phasedown to 0% by 2030, offering a better longitudinal OPEX profile.
*Recommendation: Raleigh for long-term tax stability; Austin for immediate talent density.*"

### 💎 Consultant's Playbook: The "WOW" Matrix

| Source | Strategic "WOW" Query | Consultative Insight |
| :--- | :--- | :--- |
| **FRED** | "What is the 10-year unemployment trend for Austin vs. Nashville?" | Longitudinal Labor Resilience |
| **BEA** | "Compare the Real GDP growth rate for the San Francisco MSA vs. Dallas." | Macroeconomic Momentum |
| **Census** | "Show the educational attainment (Bachelor's+) pipeline for Seattle vs. Raleigh." | Talent Depth & Engineering Density |
| **HUD** | "Is Austin affordable for a 50% AMI workforce? Correlate rent vs income." | Workforce Retention & COLA Risk |
| **BLS** | "What is the 10-year wage trend vs. unionization in the Rust Belt?" | Labor Cost & Structural Risk |
| **FEC** | "Benchmark the political stability of site selection in Ohio using FEC data." | Political Volatility & Lobbying Exposure |
| **USITC** | "Analyze Arizona as a semiconductor hub. Show trade flows vs state tax rates." | Supply Chain Dependency (Chips) |
| **EIA** | "Compare industrial electricity rates in Texas vs. Ohio for a data center." | Operational Utility Benchmarking |
| **Register** | "Are there any recent regulatory notices regarding semiconductors in Texas?" | Live Regulatory Drift & Compliance |
| **Tax F.** | "What are the corporate income tax brackets for North Carolina in 2024?" | Fiscal Competitiveness |
| **Combined** | "Create a Metro Matrix comparing Denver and Seattle for a new Tech Hub." | 360-Degree Site Selection (Level 3) |

### 📡 Consultative Capabilities

#### 💼 Labor & Macro (FRED/BLS)
- **Live Wage Analysis**: Real-time median hourly wages fetched via live FRED search (No hardcoded mocks).
- **Unemployment Trends**: 10-year historical time-series sampling for MSA-level analysis.
- **Union Density**: Live state-level union membership percentages.

#### 🏢 Real Estate & Utilities (CoStar/EIA)
- **Energy Matrix**: Live Industrial electricity rates (per kWh) using compliant EIA `IND` sector codes.
- **ROI Modeling**: Real estate acquisition ROI based on live macro health indicators.

#### 🗳️ Policy & Political Risk (FEC/LDA/OpenSecrets)
- **Campaign Finance**: Correlate political stability with corporate and PAC contribution data.
- **Lobbying Hubs**: Identification of industry influence and regulatory engagement levels.
- **Regulatory Monitoring**: Live notices from the **Federal Register** regarding industry-specific policy shifts.

#### 🏠 Housing & Affordability (HUD/Census)
- **Workforce Burden Analysis**: Correlation of Fair Market Rents (FMR) against Area Median Income (AMI).
- **Relocation COLA**: Precise cost-of-living benchmarking for talent retention strategy.
- **Demographic Depth**: Hyper-localized education and age-bucket analysis (Census ACS).

---

## B. Architecture Visuals

![ERA Architecture](economic_research_agent_architecture.png)

```mermaid
graph TD
    User([User Query]) --> Planner["Researcher Agent (Planner)"]
    
    subgraph "Structured Live Grounding (ReAct)"
        Planner --> ToolRouter{"Tool Router"}
        ToolRouter --> Macro["Macro Hub (FRED, BEA, Census, Tax Foundation)"]
        ToolRouter --> Labor["Labor Matrix (BLS, Talent Pipeline)"]
        ToolRouter --> Policy["Policy & Volatility (FEC, Regulatory, Political)"]
        ToolRouter --> Infra["Infrastructure & Climate (EIA, HUD, Resilience)"]
        ToolRouter --> Specialized["Specialized Synthesis (Metro Matrix, Relocation, Trade)"]
    end
    
    Macro --> LiveAPIs([Structured Public APIs])
    Labor --> LiveAPIs
    Policy --> LiveAPIs
    Infra --> LiveAPIs
    Specialized --> LiveAPIs
    
    LiveAPIs -->|"Grounded Data"| Planner
    
    Planner --> Judge["Auditor Judge Agent (Critic)"]
    Judge --> Search["Serper.dev Live Search"]
    Search -->|"Context Tracking"| Judge
    
    Judge --> Narrative["Narrative Synthesis & Scribe"]
    Narrative -->|"[A2UI] Response"| User
    
    style Planner fill:#f9f,stroke:#333,stroke-width:2px
    style Judge fill:#ffcc99,stroke:#333,stroke-width:2px
    style Search fill:#bbf,stroke:#333,stroke-width:2px
```

---

## C. Setup & Execution

### 🔑 API Configuration (.env)

The ERA uses a modular grounding strategy. Set these in your `.env` file (see `.env.example`).

| Service | Category | Status | Signup Link |
| :--- | :--- | :--- | :--- |
| **FRED** | Macro & Labor | **Required** | [Sign up for FRED API](https://fredaccount.stlouisfed.org/login/secure/apikeys) |
| **BEA** | GDP & Income | **Required** | [Sign up for BEA API](https://apps.bea.gov/api/signup/index.cfm) |
| **BLS** | Labor Stats | **Required** | [Sign up for BLS API](https://data.bls.gov/registrationEngine/) |
| **Census** | Demographics | **Required** | [Sign up for Census API](https://api.census.gov/data/key_signup.html) |
| **HUD** | Affordability | **Required** | [Sign up for HUD API](https://www.huduser.gov/portal/dataset/fmr-api.html) |
| **FEC** | Political Risk | **Required** | [Sign up for FEC API](https://api.open.fec.gov/) |
| **EIA** | Energy & Power | **Optional** | [Sign up for EIA API](https://www.eia.gov/opendata/register.php) |
| **NewsAPI** | Sentiment | **Optional** | [Sign up for NewsAPI](https://newsapi.org/register) |
| **Serper** | Live Judge Search | **Optional** | [Sign up for Serper.dev](https://serper.dev/) |
| **CDC** | Healthcare Stats | **Optional** | [Sign up for CDC Data](https://data.cdc.gov/) |

### 🛠️ Installation

ERA uses `uv` for lightning-fast dependency management.

```bash
# Create and synchronize the virtual environment
uv sync --dev
```

### ☁️ Google Cloud Setup (Prerequisites)

Before deploying to the Vertex AI Reasoning Engine, ensure your local environment is authenticated with Google Cloud:

1. **Install the Google Cloud CLI**: Follow the [installation guide](https://cloud.google.com/sdk/docs/install).
2. **Set your active project**:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```
3. **Authenticate your credentials**:
   ```bash
   gcloud auth application-default login
   ```

### 📦 Using Google Agents CLI (Recommended)

We highly recommend setting up and deploying this agent using the [Google Agents CLI](https://github.com/google/agents-cli). Google Agents CLI provides a production-ready framework that includes:
- **Automated CI/CD Pipelines**: Pre-configured GitHub Actions for streamlined Vertex AI deployments.
- **Standardized Structure**: Adheres to Google Cloud best practices for modular agent repositories.
- **Interactive CLI Setup**: Guides you through provisioning staging buckets and Vertex AI resources automatically.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-economic-research-agent` with your project name):

```bash
agents-cli create my-economic-research-agent -a adk@economic-research-agent
```

### 🚀 Running the Agent

ERA offers multiple interaction protocols:

```bash
# 🧠 Option 1: Interactive CLI Session (Standard)
make run

# 🛰️ Option 2: Multi-Protocol MCP Server (For Claude/Cursor)
make mcp
```

---

## D. Customization & Extension

The ERA is designed for modular growth:
- **Modifying the Persona**: Edit `economic_research/prompt.py` to change the consultative tone.
- **Adding New Skills**: Add your skill in `economic_research/tools/`, then register it in `economic_research/agent.py`.
- **Altering Data Flows**: Use the `shared_libraries/helper.py` to add new HTTP/JSON normalization patterns for regional data.

---

## E. Evaluation

How do we know ERA is accurate?
- **Golden Suite**: We use a 21-question integration suite (`tests/integration/`) targeting specific NAICS scenarios.
- **Grounding Fidelity Metric**: The `eval/run_eval.py` script uses **LLM-as-a-Judge** (Gemini 3.1 Pro) to verify if the output contains actual numerical data from the APIs.
- **Regression Testing**: `pytest` handles unit-level verification of API response parsing.

```bash
# Run the full 21-question validation suite
uv run pytest tests/integration/test_full_golden_suite.py
```

---

## F. Deploy

### 🚀 Production Rollout

The ERA is built for the **Vertex AI Reasoning Engine** (ADK 2.0).

```bash
# 🌍 Step 1: Deploy to Google Cloud (Reasoning Engine)
make deploy
```

### 🔒 Cloud-Native Security & Privacy

The ERA is engineered for **Enterprise Privacy** within the Google Cloud perimeter:
- **Zero Data Retention**: No local databases or static tables are used. Data is processed in-memory.
- **Key-Safe Architecture**: Secrets are managed via `.env` or Google Secret Manager.

---

*Built for the Atomic Agents Initiative.*
