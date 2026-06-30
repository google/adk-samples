# Technical Designer

## A. Overview & Functionalities

This agent is intended to be used as part of a broader Software Development Life Cycle (SDLC) workflow. For more details on how it integrates with other agents, please see the [SDLC Agents Workflow](sdlc_agents_workflow.md):

![Full SDLC Workflow](sdlc_agents_workflow.png)

### Agent Details Table
| Interaction Type | Conversational / Workflow |
| :--- | :--- |
| **Complexity** | Easy |
| **Agent Type** | Single Agent |
| **Vertical** | Software Architecture / Engineering |
| **Key Features** | Architectural analysis, Spanner Code Knowledge Graph exploration, interactive technical clarification, Architecture Decision Record (ADR) creation, Mermaid architecture diagram generation, outputs standardized RFC Technical Design Markdown.

## B. Architecture

![Architecture Diagram](agent_pattern.png)

### Example Interaction
**User:** Here is the refined user story for the Slack notification integration:
# [STORY TITLE: Slack Integration for Critical System Alerts]
[...]
Please propose a technical design for implementing this.

**Agent:** I've queried the Code Knowledge Graph and analyzed the current notification architecture. Here is the technical design for the Slack integration:

# RFC: Slack Integration for Critical System Alerts

## 1. Context and Scope
* **Background:** The DevOps team needs immediate visibility into critical system alerts via Slack. This requires a workspace-level OAuth flow and a mechanism to post formatted messages.
* **Goals:** Implement Slack OAuth, store workspace tokens securely, and extend the existing `NotificationService` to support Slack as a destination.
[...]

## 2. Proposed Architecture
```mermaid
---
title: Structural Overview
---
flowchart LR
    AlertSystem([Alert System]) --> NotifService[Notification Service]
    NotifService --> SlackAPI[Slack API]
    OAuthFlow([OAuth Flow]) --> CredStore[(Credential Store)]
    NotifService -.-> CredStore
```
[...]

## C. Setup & Execution

### Prerequisites & Installation

*   Python 3.10+
*   uv
    *   For dependency management and packaging. Please follow the instructions on the official [uv website](https://docs.astral.sh/uv/) for installation.

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

Set up your environment variables by copying the example file:
```bash
cp .env.example .env
```
Populate `.env` with your GCP project and Spanner details. If left null, the agent will operate without external database queries.

Install dependencies:
```bash
uv sync --dev
```

### Running the Agent Locally
Run the agent locally via CLI:
```bash
uv run adk web sdlc_technical_designer
```

### Deploying to Google Cloud
You can deploy the agent to Google Cloud (Vertex AI Reasoning Engine) using the provided deployment script:
```bash
uv run deployment/deploy.py --create
```

### Alternative: Using Google Agents CLI

You can also use the [Google Agents CLI](https://github.com/google/agents-cli) to create a production-ready version of this agent with additional deployment options.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-technical-designer` with your project name):

```bash
agents-cli create my-technical-designer -a adk@sdlc-technical-designer
```

The Google Agents CLI will prompt you to select deployment options and provides additional production-ready features including automated CI/CD deployment scripts.

## D. Customization & Extension

- **Modifying the Flow:** Adjust the RFC document structure, diagram types, or required architecture checks in `sdlc_technical_designer/prompt.py`.
- **Adding Tools:** Introduce external integrations like cloud resource estimators or architecture linters in `sdlc_technical_designer/tools/`.
- **Changing Data Sources:** Configure Spanner queries to point to different codebase graph instances or knowledge bases.
