# How to contribute

We accept patches and contributions to this collection of ADK
recipes — small, runnable agents built with the
[Agent Development Kit](https://adk.dev).

## Contribute a recipe

Recipes live in two roots: [`core/`](./core/), curated by the
`agents-cli` team, and [`contrib/`](./contrib/), the community
root. Send new work to `contrib/`.

Start with the [contributor guide](./docs/README.md). It routes
you to the [recipe checklist](./docs/recipe-checklist.md) — the
one-page path from empty folder to PR — and to the
[recipe handbook](./docs/recipe-handbook/README.md) for context
and reference.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Contributing a new sample

If you are planning to contribute a new sample, please make sure it follows our
recommended structure:

```bash

├── agent-name
│   ├── agent_name/
│   │   ├── shared_libraries/               # Folder contains helper functions for tools
│   │   ├── sub_agents/                     # Folder for each sub-agent
│   │   │   │   ├── tools/                  # Tools folder for the sub-agent
│   │   │   │   ├── agent.py                # Core logic for the sub-agent
│   │   │   │   └── prompt.py               # Prompt of the sub-agent
│   │   │   └── ...                         # More sub-agents    
│   │   ├── __init__.py                     # Initializes the agent
│   │   ├── tools/                          # Contains the code for tools used by the router (root) agent
│   │   ├── agent.py                        # Contains the core logic of the agent
│   │   ├── prompt.py                       # Contains the prompt for the agent
│   ├── deployment/                         # Deployment to Agent Engine
│   ├── eval/                               # Folder containing the evaluation method
│   ├── tests/                              # Folder containing unit tests for tools
│   ├── agent_pattern.png                   # Diagram of the agent pattern
│   ├── .env.example                        # Store agent specific env variables
│   ├── pyproject.toml                      # Project configuration
│   └── README.md                           # Provides an overview of the agent
```

To make life easier we have a [Starter Template](starter-template/README.md) to
get you started.

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.
