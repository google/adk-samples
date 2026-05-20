# Retail Agent Skills

Production-ready AI agent skills for retail on Google Cloud. Install a skill, open your AI agent, and it walks you through the rest.

**One foundation, three revenue levers** -- semantic product search at the base, with virtual try-on, recommendations, and content generation layered on top.

This repo serves two audiences:

- **[Use a skill](#use-a-skill)** -- you're a developer building a retail agent and you want a skill to guide your AI agent (Claude Code, Gemini CLI, Codex, ...).
- **[Hack on the skills](#hack-on-the-skills)** -- you've cloned this repo and want to edit or contribute a skill.

---

## Skills

| Skill | Description | Layered on |
|-------|-------------|------------|
| [retail-product-search](skills/retail-product-search/SKILL.md) | Semantic product search with Vector Search, RAG, optional voice | base |
| [retail-virtual-tryon](skills/retail-virtual-tryon/SKILL.md) | Virtual try-on with dedicated VTO model or Gemini image tiers | product-search |
| [retail-product-recommendation](skills/retail-product-recommendation/SKILL.md) | "You might also like" via collaborative / content-based / LLM | product-search |
| [retail-content-generation](skills/retail-content-generation/SKILL.md) | Product descriptions, SEO, marketing copy via Gemini | product-search |

See [skills/REGISTRY.md](skills/REGISTRY.md) for the full index.

---

## Use a skill

You're building a retail agent. Pick a skill, install it into your AI coding agent, and let the agent drive the rest.

### Prerequisites

- Node 18+ (for the installer)
- Python 3.10+ (for the sample code the agent will run)
- Google Cloud project with Vertex AI + BigQuery APIs enabled
- `gcloud auth application-default login`
- [agents-cli](https://github.com/google/agents-cli) -- `pip install google-agents-cli && agents-cli setup`
- An AI coding agent: [Gemini CLI](https://github.com/google/gemini-cli), [Claude Code](https://docs.anthropic.com/en/docs/claude-code), Codex, etc.

### 1. Install a skill into your agent

Pick `--target` based on your AI coding agent:

```bash
# Default: --target gemini (Gemini CLI, etc.)
npx --package github:google/vertical-skills install-vertical-skill retail-product-search

# Claude Code
npx --package github:google/vertical-skills install-vertical-skill retail-product-search --target claude

# Both at once (cross-agent demo)
npx --package github:google/vertical-skills install-vertical-skill retail-product-search --target both
```

| Your agent | Use this | Why |
|---|---|---|
| Gemini CLI | default (or `--target gemini`) | Gemini reads `.agents/skills/` |
| Claude Code | `--target claude` | Claude Code reads `.claude/skills/` |
| Codex / Cursor / Aider / Jules / Devin / ... | default -- they read AGENTS.md only | These agents have no skills mechanism; AGENTS.md is sufficient |

The installer drops:
- **`SKILL.md`** -> the chosen target dir (`.agents/skills/` or `.claude/skills/`, or both)
- **`AGENTS.md`** at project root -- routes the question flow to the retail skill regardless of which other skills (e.g. `google-agents-cli-*`) are also active.
- Slim sample tree -> `./<skill-name>/`
- Shared helpers -> `./_shared/`

> Once published, the shorter form `npx install-vertical-skill <skill>` will work directly. Until then, point at the source repo with `--package github:google/vertical-skills` (or use `--source <org>/<repo>[@branch]` to install from your own fork).

### 2. Open your agent and talk to it

#### Gemini CLI

```bash
gemini
```

Then describe what you want:
> *"I want to build a product search agent for my e-commerce site."*

Gemini reads the SKILL.md automatically from `.agents/skills/` and walks you through setup conversationally.

#### Claude Code

```bash
claude
```

If Claude starts with brainstorming or planning instead of the retail skill's Q-MODE flow, tell it:

> *"Read the file `.claude/skills/retail-product-search.md` and follow its instructions exactly."*

Claude will read the SKILL.md, see the `priority: high` frontmatter and the Q-MODE instructions, and switch to the retail skill's conversation flow.

#### Other agents (Codex, Cursor, Aider, Jules, Devin)

These agents read `AGENTS.md` from the project root. The installer inlines the Q-MODE first-response block there, so they follow the retail skill's flow without needing a separate skills directory.

**Skills are recipes, not scripts.** The SKILL.md is the entry point -- it tells your AI agent what to build and how.

---

## Hack on the skills

You've cloned this repo and want to edit a SKILL.md or contribute a new skill.

### Prerequisites

- Everything from [Use a skill](#use-a-skill), plus:
- This repo cloned locally

### Testing your changes end-to-end

Install your branch into an empty dir and drive the conversation manually to make sure the SKILL.md actually behaves as intended:

```bash
mkdir /tmp/skill-test && cd /tmp/skill-test
node /path/to/repo/packages/install-vertical-skill/bin/install.js \
    retail-product-search --local /path/to/repo --target gemini
gemini
# -> drive through the developer-style conversation; observe the agent's first
#   message matches the expected Q-MODE format
```

The first agent message is the load-bearing test -- if it doesn't match the expected Q-MODE pattern, your SKILL.md edits aren't being honored (or a conflicting global skill is winning).

### Contributing

- **Improve a skill** -> edit `SKILL.md`, open PR
- **Create a new skill** -> see the [SKILL.md](skills/SKILL.md) skill-creator guide

### Repository layout

```
skills/        Unified skill folders -- SKILL.md, app/, assets/, reference/, scripts/, tests/
packages/      npx installer source
vs             Contributor CLI (list)
```

---

## License

Apache 2.0 -- see [LICENSE](LICENSE).
