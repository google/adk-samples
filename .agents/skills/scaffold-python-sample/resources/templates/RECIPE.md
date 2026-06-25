# Recipe Title - Please update it
This is a description of this recipe. Please update it

- **Recipe Type**: pattern
- **Owner**: Team Name | Team Email <!-- Update with your team name and email. Used for tracking ownership and notifications. -->
- **Point of Contact**: POC Name | POC Email <!-- Update with the primary maintainer's name and email for questions or support. -->
- **Status**: active
- **Tags**: A list of comma-separated tags
- **Languages**: python
- **Paired Skill**: `scaffold-python-sample`
- **Evaluation**: `tests/eval/evalsets/basic.evalset.json` (min score: 0.8)

## Intent

It highlights a simple ADK agent with 2 python tools.

## When To Use

- To scaffold a simple ADK agent.

## Requires
A GCP Project if the user wants to deploy it to Google Cloud

## Constraints

- **Must**: user interaction, Gemini Enterprise
- **Must Not**: service-account-only, no UI

## Composition

- **Composes with**: `agent-runtime-deployment`
- **Conflicts with**: None