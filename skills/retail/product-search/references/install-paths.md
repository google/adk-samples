# Install Paths and Environment Quirks

## Known install dirs per host

| Host | Install dir |
|---|---|
| Claude Code | `~/.claude/skills/<skill-name>/` |
| `npx skills add` (Codex, Gemini CLI, GitHub Copilot, others) | `~/.agents/skills/<skill-name>/` |
| Gemini CLI standalone | `~/.gemini/skills/<skill-name>/` |
| Cursor | `~/.cursor/skills/<skill-name>/` |

`bootstrap.sh` tries these paths in order. If your host installs elsewhere,
set `SKILL_DIR` manually:

```bash
export SKILL_DIR=/path/to/your/install/dir
```

## Python interpreter fallback

`bootstrap.sh` tries `command -v python3.{13,12,11,10}` first, then falls
back to absolute paths (`/opt/homebrew/bin/`, `/usr/local/bin/`,
`~/.pyenv/shims/`). The fallback handles sandboxed shells with stripped PATH.

For conda/asdf/other layouts, set `PYTHON_BIN` and skip the loop.

## Why `bash -c` wraps the pip command

`pip install -e "$SKILL_DIR[adk]"` looks correct but breaks in zsh: zsh treats
`[adk]` as a glob character class, silently expands to empty, and the
`[adk]` extras are dropped. `bash -c "pip install -e '${SKILL_DIR}[adk]'"`
sidesteps this — pip runs under bash, which doesn't glob-expand `[adk]`.

## Why the workspace setup must run as one shell command

Agent shell tools reset cwd and clear variables between calls. If you set
`SKILL_DIR` in call 1 and run pip in call 2, `$SKILL_DIR` is empty in call 2
and the install becomes `pip install -e [adk]` (invalid).

## Description tuning for triggering

If your skill is registered but the agent doesn't pick it for relevant
prompts, the `description` field is what hosts' routers match against.
Include concrete trigger phrases alongside the abstract description.
