# Crypto Payroll Agent

A Google ADK sample agent that batch-pays multiple recipients in ETH or
ERC-20 tokens on [Base](https://base.org) in a single transaction.

Built on the [Spraay batch payment tools](https://github.com/google/adk-python-community/pull/95)
in `google-adk-community`. Demonstrates a complete payroll, airdrop, and
revenue-split workflow with built-in safety gates.

## What this agent does

A treasury operator says something like:

> *"Pay 250 USDC each to alice.eth, bob.eth, and 0x742d…f44e."*

…or:

> *"Split this 10,000 USDC airdrop across these 50 addresses by their
> contribution weights."*

The agent:

1. Identifies whether the payment is **equal** (use `spraay_batch_eth` /
   `spraay_batch_token`) or **variable** (use the `_variable` siblings).
2. Looks up token metadata (decimals, canonical address) via the local
   token registry so the user never has to know that USDC has 6 decimals.
3. For variable distributions, splits a pool proportionally with the
   `split_pool_proportionally` helper.
4. Presents a single-screen plan: recipient count, amounts, total,
   estimated gas savings vs. individual transfers.
5. Waits for explicit user confirmation, then executes the batch on Base
   through the Spraay protocol.

## Why this is interesting

Most sample agents demo one or two narrow capabilities. This one shows
ADK's tool-selection in action: the model must pick the correct one of
four batch tools (eth vs. token × equal vs. variable) from free-form user
input, plus chain a helper tool when the user provides weights instead of
explicit per-recipient amounts. It's a small but realistic showcase of
multi-tool composition.

## Prerequisites

- Python 3.11+
- `uv` package manager: `pip install uv`
- A Base wallet with a small amount of ETH for gas and the token(s) you
  want to send. Base testnet (Sepolia) works for development; set
  `SPRAAY_RPC_URL` to a Sepolia endpoint.

## Setup

```bash
git clone https://github.com/google/adk-samples.git
cd adk-samples/python/agents/crypto-payroll-agent
uv sync
cp .env.example .env
# Edit .env: at minimum, set SPRAAY_PRIVATE_KEY and your Google Cloud project
```

> **Note:** This sample installs `google-adk-community` directly from
> the GitHub `main` branch because the Spraay batch tools merged in
> [adk-python-community#95](https://github.com/google/adk-python-community/pull/95)
> have not yet been included in a PyPI release. Once a release later
> than `0.4.1` ships, the `pyproject.toml` pin can be relaxed to a
> standard version constraint.

After `uv sync`, verify the four Spraay tools import correctly:

```bash
uv run python -c "from google.adk_community.tools.spraay import \
  spraay_batch_eth, spraay_batch_token, \
  spraay_batch_eth_variable, spraay_batch_token_variable; \
  print('Spraay tools ready')"
```

## Run the agent

```bash
uv run adk web
```

Then open `http://localhost:4200`, choose **crypto_payroll_agent**, and
try the example prompts below.

## Example interactions

### 1. Equal-amount payroll (`spraay_batch_token`)

> *Pay 250 USDC each to 0x9f2c…81a3, 0x4ab7…d109, and 0x8eee…2017 on Base.*

The agent looks up USDC's address and decimals, summarizes the plan, and
on `confirm` calls `spraay_batch_token` with `amount_per_recipient="250"`
and `token_decimals=6`.

### 2. Equal-amount ETH bounties (`spraay_batch_eth`)

> *Send 0.01 ETH each to these 12 bounty hunters: [list of 12 addresses].*

The agent rejects amounts over the configured safety ceiling, otherwise
calls `spraay_batch_eth` with `amount_per_recipient_eth="0.01"`.

### 3. Variable token amounts (`spraay_batch_token_variable`)

> *Pay this month's contractors. Alice gets 1500 USDC, Bob 2200, Carol
> 800.*

The agent extracts the per-recipient amounts and calls
`spraay_batch_token_variable`.

### 4. Proportional split (`split_pool_proportionally` + `spraay_batch_token_variable`)

> *Distribute 10,000 USDC to these 5 contributors by their commit counts:
> 120, 80, 45, 30, 25.*

The agent calls `split_pool_proportionally` first to compute per-recipient
amounts (rounding-safe so the total is exact), then executes the variable
batch.

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `SPRAAY_PRIVATE_KEY` | yes | Private key of the sending wallet on Base |
| `SPRAAY_RPC_URL` | no | Base RPC endpoint (default: `https://mainnet.base.org`) |
| `SPRAAY_CONTRACT_ADDRESS` | no | Override the Spraay batch contract |
| `GOOGLE_CLOUD_PROJECT` | yes¹ | GCP project for Vertex AI |
| `GOOGLE_CLOUD_LOCATION` | yes¹ | GCP region (e.g. `us-central1`) |
| `GOOGLE_GENAI_USE_VERTEXAI` | yes¹ | `1` for Vertex AI, `0` for AI Studio |
| `GOOGLE_API_KEY` | yes² | AI Studio key (alternative to Vertex AI) |
| `PAYROLL_AGENT_MODEL` | no | Gemini model (default: `gemini-2.5-flash`) |
| `PAYROLL_MAX_BATCH_USD` | no | Refuse batches above this total (default: `10000`) |

¹ if using Vertex AI · ² if using AI Studio

## Agent structure

```
crypto_payroll_agent/
├── agent.py          # LlmAgent wired to 4 Spraay tools + 2 helpers
├── prompt.py         # System instruction governing tool selection
├── config.py         # Model + safety + token registry
└── tools/
    └── helpers.py    # lookup_token_info, split_pool_proportionally
```

The agent has six tools total:

| Tool | Source | Purpose |
| --- | --- | --- |
| `spraay_batch_eth` | `google.adk_community.tools.spraay` | Equal ETH to N recipients |
| `spraay_batch_token` | same | Equal ERC-20 to N recipients |
| `spraay_batch_eth_variable` | same | Variable ETH amounts |
| `spraay_batch_token_variable` | same | Variable token amounts |
| `lookup_token_info` | local | Resolve symbol → {address, decimals} |
| `split_pool_proportionally` | local | Divide a pool by weights (rounding-safe) |

## Evaluation

```bash
uv run adk eval crypto_payroll_agent eval/crypto_payroll_eval_set.evalset.json
```

The evalset covers all four batch-tool selection paths plus a safety-gate
refusal case.

## Testing

```bash
uv run pytest tests/ -v
```

All tests are offline. Spraay tools are mocked; helpers are tested
directly.

## Deployment

See `deployment/deploy.py` for a reference Vertex AI Agent Engine
deployment script.

## Protocol details

- Spraay batch contract: `0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC`
  on Base mainnet
- Up to 200 recipients per transaction
- ~80% gas savings vs. individual transfers
- 0.3% protocol fee
- More info: [spraay.app](https://spraay.app)

This is a community sample. Spraay is not an official Google product.

## License

Apache 2.0 — see the repository root LICENSE.
