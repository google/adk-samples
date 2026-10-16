# alphaevolve-optimizer

An **AlphaEvolve-style evolutionary code optimizer** built with the ADK Python
SDK. A Gemini agent improves a program by repeating a **propose → verify →
select** loop:

1. **Propose** – the agent writes a new candidate program, building on the best
   programs found so far and the feedback they received.
2. **Verify** – the `evaluate_candidate` tool runs the candidate in a separate,
   resource-limited Python process and scores its outputs deterministically.
   The model never grades its own work.
3. **Select** – the best verified candidates are kept in session state and fed
   back as parents for the next round, until the evaluation budget is spent or
   the score reaches the known lower bound.

The recipe ships with a small **heuristic discovery** problem: one-dimensional
bin packing over five fixed benchmark instances. The baseline (first-fit) uses
314 bins, the lower bound is 280, and simple ideas such as first-fit decreasing
already reach 306, so there is visible room for the agent to improve.

```mermaid
flowchart LR
    U[User: "start"] --> A[Gemini agent]
    A -- get_problem / get_parents --> S[(Session state:<br/>population)]
    A -- evaluate_candidate(code) --> V[Sandboxed verifier]
    V -- score / error --> S
    S -- best parents + scores --> A
```

## What you will learn

- Grounding an LLM in a **deterministic verifier** so every claimed
  improvement is measured, not asserted.
- Keeping an evolving **population in ADK session state** with
  `ToolContext`, and enforcing an evaluation budget in tools.
- Running untrusted, model-written code with **process isolation**, timeouts
  and resource limits, and scoring it outside the child process.

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager - [Install](https://docs.astral.sh/uv/getting-started/installation/)
- Access to Gemini through Vertex AI or a Google AI Studio API key.

## Setup

> **Note**: All commands below must be run from the recipe root directory (`contrib/python/alphaevolve-optimizer/`).

1. Install required packages:
   ```bash
   uv sync
   ```

2. Set up environment variables. Copy `.env.example` to `.env` and configure
   either Vertex AI (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_GENAI_USE_VERTEXAI=True`)
   or `GEMINI_API_KEY`:
   ```bash
   cp .env.example .env
   ```

   | Variable | Default | Purpose |
   | --- | --- | --- |
   | `MODEL_NAME` | `gemini-3.7-flash` | Model used to propose candidates |
   | `MAX_EVALUATIONS` | `10` | Verifier calls allowed per session |
   | `POPULATION_SIZE` | `4` | Number of best candidates kept as parents |
   | `CANDIDATE_TIMEOUT_SECONDS` | `10` | Wall-clock and CPU limit per evaluation |

## Run

Start the agent in the terminal and type `start`:

```bash
uv run adk run app
```

Or start the local web server and open the ADK dev UI:

```bash
uv run uvicorn app.fast_api_app:app --reload
```

The agent reports the baseline score, the best score it found, the lower
bound, a table of the ideas it tried, and the best program.

### Example run

One run with `gemini-3.7-flash` and the default budget of 10 evaluations:

| Step | Idea proposed by the agent | Verified score |
| --- | --- | --- |
| baseline | First-fit | 314 |
| 1 | Best-fit decreasing | 306 |
| 3 | Minimum bin slack with subset-sum search | 304 |
| 4 | Bin elimination with tabu search and a quadratic overload penalty | **293** |
| 5-10 | Simulated annealing, late acceptance, 2-for-1 swaps, 3-into-2 consolidation | 293-294 |

The best program closed about 60% of the gap between the baseline and the
lower bound of 280. Results vary from run to run because the proposals come
from a model.

## Use your own problem

Everything problem-specific lives in [`app/problem.py`](app/problem.py):
`PROBLEM_STATEMENT`, `ENTRY_POINT`, `INSTANCES`, `BASELINE_PROGRAM`,
`lower_bound()` and `score_outputs()`. Replace them to evolve a different
program — for example a scheduling heuristic, a numerical kernel, or a prompt
template scored by an automatic metric. The loop, sandbox and selection logic
do not change.

## Safety

The verifier isolates candidates in a separate interpreter (`python -I -B`)
with an empty environment, a timeout, and CPU, memory and file-write limits.
This is **process isolation, not a security sandbox**. Run the recipe in its
container and add a stronger sandbox (for example gVisor) before exposing it
to untrusted users.

## Running Tests

The unit tests cover the verifier, the selection logic and the tools, and do
not call a model:

```bash
uv run pytest
```

Integration tests call Gemini and need credentials:

```bash
uv run pytest tests/integration
```

## Commands

| Command | Description |
| ------- | ----------- |
| `uv run adk run app` | Run the agent in interactive CLI mode |
| `uv run uvicorn app.fast_api_app:app --reload` | Start the local FastAPI development server |
| `uv run pytest` | Run the unit and runnability tests |
