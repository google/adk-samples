# Marketing Agency

## Overview

This AI-powered assistant is engineered to enhance the capabilities of creative agencies when launching new websites or products. The process commences with an intelligent agent that guides users in selecting an optimal DNS domain, ensuring it aligns perfectly with the website's subject matter or the product's identity. Following this foundational step, another specialized agent takes over to facilitate the comprehensive creation of the entire website. Subsequently, a dedicated agent is deployed to generate a full suite of marketing materials. The workflow culminates with an agent focused on designing distinctive logos that are thematically consistent with the project's core topic. This multi-agent system aims to streamline and augment the creative output of the agency across the entire launch lifecycle.

## Agent Details

The key features of the Marketing Agency include:

| Feature | Description |
| --- | --- |
| **Interaction Type** | Conversational |
| **Complexity**  | Medium |
| **Agent Type**  | Multi Agent |
| **Components**  | Tools: built-in Google Search |
| **Vertical**  | Marketing |


### Agent architecture:

This diagram shows the detailed architecture of the agents and tools used
to implement this workflow.
<img src="marketing-agency.png" alt="marketing agency" width="800"/>

## Quick Start with Google Agents CLI (Recommended)

The fastest way to get a production-ready version of this agent is using the
[Google Agents CLI](https://github.com/google/agents-cli). It scaffolds a full
project with CI/CD, deployment scripts, and best practices built in.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-marketing-agency` with your project name):

```bash
agents-cli create my-marketing-agency -a adk@marketing-agency
```

This will:
- Copy the marketing-agency sample into a new project
- Prompt you to select deployment options (Cloud Run, Agent Runtime, etc.)
- Generate CI/CD pipelines and infrastructure-as-code
- Set up a ready-to-deploy project structure

Once created, follow the generated project's README for deployment instructions.

## Setup and Installation (Local Development)

If you prefer to run the agent directly from this repository without the
starter pack scaffolding, follow the steps below.

### Prerequisites

*   Python 3.10+
*   [uv](https://docs.astral.sh/uv/) for dependency management and packaging.

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

* A project on Google Cloud Platform
* [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)

### Installation

```bash
# Clone this repository.
git clone https://github.com/google/adk-samples.git
cd adk-samples/python/agents/marketing-agency
# Install the package and dependencies.
uv sync
```

### Configuration

1.  Copy `.env.example` to `.env` and fill in your project details:

    ```bash
    cp .env.example .env
    ```

2.  Required environment variables (set in `.env` or your shell):

    ```bash
    GOOGLE_GENAI_USE_VERTEXAI=1
    GOOGLE_CLOUD_PROJECT=<your-project-id>
    GOOGLE_CLOUD_LOCATION=<your-project-location>
    GOOGLE_CLOUD_STORAGE_BUCKET=<your-storage-bucket>  # Only required for deployment on Agent Engine
    ```

3.  Authenticate with Google Cloud:

    ```bash
    gcloud auth application-default login
    gcloud auth application-default set-quota-project $GOOGLE_CLOUD_PROJECT
    ```

The agent's `__init__.py` automatically loads `.env` via `python-dotenv`
and attempts to discover your GCP project via Application Default
Credentials (ADC). If ADC is configured, you only need to set
`GOOGLE_CLOUD_PROJECT` when your default project differs from what
`gcloud` returns.

## Running the Agent

ADK provides convenient ways to bring up agents locally and interact with them.
You may talk to the agent using the CLI:

```bash
uv run adk run marketing_agency
```

Or on a web interface:

```bash
uv run adk web
```

The command `adk web` will start a web server on your machine and print the URL.
You may open the URL, select "marketing_agency" in the top-left drop-down menu, and
a chatbot interface will appear on the right. The conversation is initially
blank. Here are some example requests you may ask the Marketing Agency:

```
who are you
```

Sampled responses of these requests are shown below in the [Example
Interaction](#example-interaction) section.

```
I am a marketing expert, and my goal is to help you establish a powerful online
presence and connect effectively with your audience. I will guide you through
defining your digital identity, from choosing the perfect domain name and
crafting a professional website, to strategizing online marketing campaigns,
designing a memorable logo, and creating engaging short videos.

What keywords are relevant to your brand? I'll use them to suggest some domain names.
```

### Example Interaction

Below is an example interaction with the Marketing Agency. Note that the exact output
of the agent may be different every time.

```
user: who are you

I am a marketing expert, and my goal is to help you establish a powerful online
presence and connect effectively with your audience. I will guide you through
defining your digital identity, from choosing the perfect domain name and
crafting a professional website, to strategizing online marketing campaigns,
designing a memorable logo, and creating engaging short videos.

help me to find the perfect domain

Okay, I can help with that! To start, please tell me some keywords relevant to
your brand. For example, if you're selling handmade jewelry, keywords might be
"handmade," "jewelry," "artisan," "unique," "gifts."

user: i want to sell cakes organic cakes

Great! I'll use "organic" and "cakes" as keywords to find some domain name
options for you.

Here are some available domain names based on your keywords:

PureOrganicBakes.com
EarthlyCakes.com
Organocake.com
FarmFreshCakes.com
ArtisanOrganicCakes.com
SimplyOrganicCakes.com
TrueOrganicBaking.com
GoldenOrganicCakes.com
BloomOrganicCakes.com
PetalOrganicCakes.com
Please review this list and let me know which domain name you'd like to choose.
```

## Running Tests

Install the dev dependencies:

```bash
uv sync --group dev
```

Then run the tests from the `marketing-agency` directory:

```bash
uv run pytest tests
uv run pytest eval
```

`tests` runs the agent on a sample request and makes sure that every component
is functional. `eval` is a demonstration of how to evaluate the agent, using the
`AgentEvaluator` in ADK. It sends a couple of requests to the agent and expects
that the agent's responses match a pre-defined response reasonably well.

## Deployment

### Using Google Agents CLI (Recommended)

The recommended way to deploy this agent is via the
[Google Agents CLI](https://github.com/google/agents-cli), which provides
production-ready deployment pipelines out of the box.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-marketing-agency` with your project name):

```bash
agents-cli create my-marketing-agency -a adk@marketing-agency
cd my-marketing-agency
```

The Google Agents CLI supports multiple deployment targets including Cloud Run and
Agent Runtime, and generates Terraform/CI/CD configuration
automatically. Follow the generated project's README for full deployment
instructions.

### Manual Deployment to Agent Runtime

You can also deploy directly to Agent Runtime without the starter pack:

```bash
uv sync --group deployment
uv run deployment/deploy.py --create
```

When the deployment finishes, it will print a line like this:

```
Created remote agent: projects/<PROJECT_NUMBER>/locations/<PROJECT_LOCATION>/reasoningEngines/<AGENT_ENGINE_ID>
```

If you forgot the AGENT_ENGINE_ID, you can list existing agents using:

```bash
uv run deployment/deploy.py --list
```

You may interact with the deployed agent using the `test_deployment.py` script:

```bash
export USER_ID=<any string>
uv run deployment/test_deployment.py --resource_id=${AGENT_ENGINE_ID} --user_id=${USER_ID}
```

To delete the deployed agent:

```bash
uv run deployment/deploy.py --delete --resource_id=${AGENT_ENGINE_ID}
```

## Customization

The Marketing Agency platform is designed with inherent flexibility, allowing for significant customization to align precisely with your unique operational requirements and marketing objectives. Its core capabilities can be substantially enhanced through targeted upgrades, for instance:

1. Upgrade DNS Suggestion with Real-time Availability Checks: The current DNS suggestion feature, while helpful, can be significantly improved by integrating real-time availability checks. This enhancement would ensure that any proposed domain names are instantly verified against existing registrations, eliminating the frustration of suggesting unavailable options. By providing immediate feedback on domain availability, the platform streamlines the website creation process, accelerates decision-making for clients, and enhances the overall efficiency of launching new online presences.

2. Implement Persistent Website Artifact Storage for Version Control and Archiving: To provide greater control, accountability, and flexibility, functionality can be added to save any generated website as a persistent digital artifact. This means the complete website structure, content, and assets would be securely stored and retrievable. This capability is crucial for enabling robust version control, allowing users to track changes over time, revert to previous iterations if needed, and maintain a clear history of development. Furthermore, it provides essential backup and archiving for future reference, auditing, or potential reuse, ensuring that valuable creative work is never lost.

3. Expand Media Creation Tools with Integrated Video Production (e.g., Veo Integration): To broaden the scope of marketing materials the platform can generate, the existing media creation tools can be significantly expanded to include video production capabilities. By integrating services like Veo, the agency would gain the ability to efficiently produce high-quality, short promotional videos, social media clips, or explainer content. This diversification into video marketing is critical in today's digital landscape, enabling the creation of more engaging, dynamic, and versatile marketing assets that can be utilized across various channels, from social media campaigns to website embeds, thereby greatly enhancing client reach and engagement.

## Disclaimer

This agent sample is provided for illustrative purposes only and is not intended for production use. It serves as a basic example of an agent and a foundational starting point for individuals or teams to develop their own agents.

This sample has not been rigorously tested, may contain bugs or limitations, and does not include features or optimizations typically required for a production environment (e.g., robust error handling, security measures, scalability, performance considerations, comprehensive logging, or advanced configuration options).

Users are solely responsible for any further development, testing, security hardening, and deployment of agents based on this sample. We recommend thorough review, testing, and the implementation of appropriate safeguards before using any derived agent in a live or critical system.
