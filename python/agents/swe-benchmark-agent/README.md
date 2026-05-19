# SWE Benchmark Agent

## Overview

This agent is designed to show the basic principles for tackling software engineering problems from two prominent benchmarks: SWE-bench and TerminalBench. It is not meant to be a production ready implementation.

The [Google Agents CLI](https://github.com/google/agents-cli) is the **recommended** way to create a new project from this sample: you get a production-oriented layout, deployment choices, and CI/CD scaffolding. The copy in [adk-samples](https://github.com/google/adk-samples) remains the upstream source for browsing and contributions.

## Agent Details

| Feature | Description |
| --- | --- |
| **Interaction Type** | Autonomous |
| **Complexity**  | Advanced |
| **Agent Type**  | Single Agent |
| **Components**  | Tools: Shell |
| **Vertical**  | Software Engineering |

### Agent architecture:

The SWE Benchmark Agent uses a sophisticated orchestrator pattern:
- **Orchestrator**: Manages the agent lifecycle and coordinates tool execution
- **Environment**: Docker-based isolated execution environment (SWEBenchEnvironment or TerminalBenchEnvironment)
- **Tools**: File operations (read, edit, create), shell commands, and submission
- **Agent**: LLM-powered agent (Gemini) with built-in planner and thinking capabilities

The agent operates autonomously within the Docker environment, using shell commands and file operations to solve software engineering tasks.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- **Google Cloud SDK (`gcloud`)** installed and authenticated (for Vertex / Gemini)
- **Git**
- **Docker** (for SWE-bench and TerminalBench evaluation via `swe_benchmark_agent.main`)

### Recommended: Using Google Agents CLI

The Google Agents CLI is the recommended way to create and deploy a production-ready version of this agent.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-swe-agent` with your project name):

```bash
agents-cli create my-swe-agent -a adk@swe-benchmark-agent
cd my-swe-agent
```

Install dependencies (including dev tools for tests):

```bash
uv sync --group dev
```

Configure Google Cloud (environment variables or a `.env` file):

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=<your-project-id>
export GOOGLE_CLOUD_LOCATION=global
```

Authenticate:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project $GOOGLE_CLOUD_PROJECT
```

During setup, the Google Agents CLI will prompt you for deployment options and adds production-oriented tooling (for example automated CI/CD deployment scripts).

<details>
<summary>Clone this repository directly (contributors and advanced use)</summary>

Use this workflow when working against the upstream repository (for example to open a pull request). **New projects should still use the Google Agents CLI** as described above.

```bash
git clone https://github.com/google/adk-samples.git
cd adk-samples/python/agents/swe-benchmark-agent
uv sync --group dev
```

Set the same `GOOGLE_*` environment variables and run `gcloud auth application-default login` as in the recommended path. Running the agent, tests, and evaluations uses the same commands as below.

</details>

## Running the Agent

Talk to the sample agent with the ADK CLI:

```bash
uv run adk run swe_benchmark_agent
```

Or use the web UI:

```bash
uv run adk web
```

Select **swe_benchmark_agent** in the UI if prompted. The interactive agent explains how to run full benchmark evaluations; those use Docker via `swe_benchmark_agent.main` (see [Running evaluations](#running-evaluations)).

## Running tests

```bash
uv run pytest tests -v
```

## Running evaluations

The SWE Agent can be evaluated on both SWE-bench and TerminalBench benchmarks to measure its performance on real-world software engineering tasks.

### SWE-bench Evaluation

To run evaluation on the full SWE-bench Verified dataset:

```bash
uv run python -m swe_benchmark_agent.main --full-dataset --evaluate --max-workers 4
```

To evaluate on a specific number of instances (e.g., the first 10):

```bash
uv run python -m swe_benchmark_agent.main --instance-id-or-count 10 --evaluate
```

To evaluate on a single instance:

```bash
uv run python -m swe_benchmark_agent.main --instance-id-or-count django__django-12345 --evaluate
```

### TerminalBench Evaluation

To run evaluation on the full TerminalBench core dataset:

```bash
uv run python -m swe_benchmark_agent.main --dataset terminalbench --full-dataset --evaluate --max-workers 4
```

To evaluate on a specific number of tasks (e.g., the first 5):

```bash
uv run python -m swe_benchmark_agent.main --dataset terminalbench --instance-id-or-count 5 --evaluate
```

To evaluate on a single task:

```bash
uv run python -m swe_benchmark_agent.main --dataset terminalbench --instance-id-or-count blind-maze-explorer-5x5 --evaluate
```

## Customization

The SWE Agent can be customized to better suit your requirements. For example:

 1. **Use a different model:** Adjust the model in `swe_benchmark_agent/main.py` (benchmark orchestration) or `swe_benchmark_agent/agent.py` (interactive `adk run` entry point).
 2. **Add more tools:** Add tools to the agent to give it more capabilities.
 3. **Support more benchmarks:** Add support for more benchmarks by creating a new environment and updating `swe_benchmark_agent/main.py`.
