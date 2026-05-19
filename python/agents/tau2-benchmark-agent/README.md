# Tau2-Bench Agent

## Overview

The Tau2-Bench Agent integrates with the [τ-bench](https://github.com/sierra-research/tau2-bench) framework using the Google Agent Development Kit (ADK). It is designed to evaluate agent performance across real-world customer service domains (airline, retail, telecom) by running structured benchmark simulations.

This sample is compatible with the [Google Agents CLI](https://github.com/google/agents-cli) and can be used as a base for creating production-ready agent deployments.

## Agent Details

| Feature | Description |
| --- | --- |
| **Interaction Type** | Autonomous |
| **Complexity** | Advanced |
| **Agent Type** | Single Agent |
| **Components** | Tools, Built-in Planner |
| **Vertical** | Benchmarking / Evaluation |

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- A project on Google Cloud Platform
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)

Authenticate with Google Cloud:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

## Using Google Agents CLI (Recommended)

Use the [Google Agents CLI](https://github.com/google/agents-cli) to scaffold a production-ready version of this agent with deployment and CI/CD options.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-tau2-benchmark` with your project name):

```bash
agents-cli create my-tau2-benchmark -a adk@tau2-benchmark-agent
cd my-tau2-benchmark
```

Install dependencies:

```bash
uv sync
```

Configure your environment by creating a `.env` file in the project root:

```bash
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=global
VERTEXAI_LOCATION=global
```

### Registering the ADK Agent with τ-bench

The `AdkAgent` must be registered in the τ-bench framework's registry. Add the following to `src/tau2/registry.py` inside the `tau2-bench` installation (found in the `.venv`):

1. Add the import at the top:

```python
from tau2.agent.adk_agent import AdkAgent
```

2. Register the agent in the `try` block:

```python
registry.register_agent(AdkAgent, "adk_agent")
```

### Running the Agent

#### Limited run (single task)

```bash
uv run tau2 run \
  --domain airline \
  --agent adk_agent \
  --agent-llm vertex_ai/gemini-2.5-pro \
  --user-llm vertex_ai/gemini-2.5-pro \
  --num-trials 1 \
  --num-tasks 1
```

#### Full evaluation run

```bash
# Retail domain
uv run tau2 run \
  --domain retail \
  --agent adk_agent \
  --agent-llm vertex_ai/gemini-3-pro-preview \
  --user-llm vertex_ai/gemini-3-pro-preview \
  --num-trials 4 \
  --save-to gemini_3_pro_retail \
  --user-llm-args '{"temperature": 1, "reasoning_effort": "high"}' \
  --agent-llm-args '{"temperature": 1, "reasoning_effort": "high"}'

# Airline domain
uv run tau2 run \
  --domain airline \
  --agent adk_agent \
  --agent-llm vertex_ai/gemini-3-pro-preview \
  --user-llm vertex_ai/gemini-3-pro-preview \
  --num-trials 4 \
  --save-to gemini_3_pro_airline \
  --user-llm-args '{"temperature": 1, "reasoning_effort": "high"}' \
  --agent-llm-args '{"temperature": 1, "reasoning_effort": "high"}'

# Telecom domain
uv run tau2 run \
  --domain telecom \
  --agent adk_agent \
  --agent-llm vertex_ai/gemini-3-pro-preview \
  --user-llm vertex_ai/gemini-3-pro-preview \
  --num-trials 4 \
  --save-to gemini_3_pro_telecom \
  --user-llm-args '{"temperature": 1, "reasoning_effort": "high"}' \
  --agent-llm-args '{"temperature": 1, "reasoning_effort": "high"}'
```

#### View trajectories

```bash
uv run tau2 view
```

#### Prepare submission package

```bash
uv run tau2 submit prepare data/tau2/simulations/gemini_3_pro_*.json \
  --output ./gemini_3_pro_submission
```

### Running Tests

```bash
uv run pytest tests -v
```

The starter pack will prompt you to select deployment options and provides additional production-ready features including automated CI/CD deployment scripts.

<details>
<summary>Running without Google Agents CLI</summary>

### Installation

```bash
git clone https://github.com/google/adk-samples.git
cd adk-samples/python/agents/tau2-benchmark-agent

# Sync dependencies (tau2-bench is pulled from GitHub automatically)
uv sync
```

### Configuration

Create a `.env` file in the project root:

```bash
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=global
VERTEXAI_LOCATION=global
```

### Registering the ADK Agent with τ-bench

The `AdkAgent` must be registered in the τ-bench framework's registry. Add the following to `src/tau2/registry.py` inside the `tau2-bench` installation (found in the `.venv`):

1. Add the import at the top:

```python
from tau2.agent.adk_agent import AdkAgent
```

2. Register the agent in the `try` block:

```python
registry.register_agent(AdkAgent, "adk_agent")
```

### Running the Agent

#### Limited run (single task)

```bash
uv run tau2 run \
  --domain airline \
  --agent adk_agent \
  --agent-llm vertex_ai/gemini-2.5-pro \
  --user-llm vertex_ai/gemini-2.5-pro \
  --num-trials 1 \
  --num-tasks 1
```

#### Full evaluation run

```bash
# Retail domain
uv run tau2 run \
  --domain retail \
  --agent adk_agent \
  --agent-llm vertex_ai/gemini-3-pro-preview \
  --user-llm vertex_ai/gemini-3-pro-preview \
  --num-trials 4 \
  --save-to gemini_3_pro_retail \
  --user-llm-args '{"temperature": 1, "reasoning_effort": "high"}' \
  --agent-llm-args '{"temperature": 1, "reasoning_effort": "high"}'

# Airline domain
uv run tau2 run \
  --domain airline \
  --agent adk_agent \
  --agent-llm vertex_ai/gemini-3-pro-preview \
  --user-llm vertex_ai/gemini-3-pro-preview \
  --num-trials 4 \
  --save-to gemini_3_pro_airline \
  --user-llm-args '{"temperature": 1, "reasoning_effort": "high"}' \
  --agent-llm-args '{"temperature": 1, "reasoning_effort": "high"}'

# Telecom domain
uv run tau2 run \
  --domain telecom \
  --agent adk_agent \
  --agent-llm vertex_ai/gemini-3-pro-preview \
  --user-llm vertex_ai/gemini-3-pro-preview \
  --num-trials 4 \
  --save-to gemini_3_pro_telecom \
  --user-llm-args '{"temperature": 1, "reasoning_effort": "high"}' \
  --agent-llm-args '{"temperature": 1, "reasoning_effort": "high"}'
```

#### View trajectories

```bash
uv run tau2 view
```

#### Prepare submission package

```bash
uv run tau2 submit prepare data/tau2/simulations/gemini_3_pro_*.json \
  --output ./gemini_3_pro_submission
```

### Running Tests

```bash
uv run pytest tests -v
```

</details>

## Customizing the Agent

You can swap out the underlying ADK agent implementation by modifying `_create_agent` in `tau2_agent/adk_agent.py`:

```python
def _create_agent(
    name: str, model: Union[str, BaseLlm], instruction: str, tools: List[Tool]
) -> BaseAgent:
    adk_tools = [
        AdkTool(
            types.FunctionDeclaration(
                name=tool.openai_schema['function']['name'],
                description=tool.openai_schema['function'].get('description', ''),
                parameters_json_schema=tool.openai_schema['function']['parameters'],
            )
        )
        for tool in tools
    ]
    return AdkLlmAgent(
        model=model,
        name=name,
        instruction=instruction,
        tools=adk_tools,
        planner=built_in_planner.BuiltInPlanner(
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        ),
    )
```

## Notes

- **Temperature**: When `adk_agent` is used, temperature defaults to `1`.
- **Reasoning level**: Only applies to Gemini 3 Pro. Defaults to `high` for `adk_agent`.
- **`This model isn't mapped yet` warnings**: These come from litellm's cost calculation and can be suppressed by using `--user-llm vertex_ai/gemini-2.5-pro` instead of the Gemini 3 Pro preview.

## Disclaimer

This agent sample is provided for illustrative purposes only and is not intended for production use. It serves as a basic example of an agent and a foundational starting point for individuals or teams to develop their own agents.

Users are solely responsible for any further development, testing, security hardening, and deployment of agents based on this sample.
