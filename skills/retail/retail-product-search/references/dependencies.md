# Dependencies

`pip install -e .` from the install dir resolves everything from
`pyproject.toml`. You don't install these manually.

## Required

- `google-cloud-bigquery>=3.0`
- `google-cloud-storage>=2.0`
- `google-cloud-vectorsearch>=0.5,<1.0` (preview, pinned)
- `google-cloud-aiplatform>=1.30`
- `google-genai>=1.0`
- `pyyaml>=6.0`, `requests>=2.28`, `pyOpenSSL` (mTLS for BigQuery)

## Optional `[adk]` extra

- `google-adk>=2.2.0` — for the ADK web UI

## Install quirks (handled by bootstrap.sh)

1. **zsh expands `[adk]` as a glob** and silently drops extras. Workaround:
   `bash -c "pip install -e '${SKILL_DIR}[adk]'"`.
2. **Agent shell tools reset state between calls.** Run the workspace setup
   block as a single shell invocation.
3. **Stripped PATH** in sandboxed terminals hides brew/pyenv Python. The
   bootstrap script falls back to absolute paths
   (`/opt/homebrew/bin/python3.13`, `~/.pyenv/shims/python3.12`, etc.).
