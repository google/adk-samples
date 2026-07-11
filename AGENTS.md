# ADK Samples — Notes for AI Contributors

This file contains quick-reference information for agents (and humans) contributing to the [Agent Development Kit (ADK) Samples](https://github.com/google/adk-samples) repository.

## Project Overview

This repo provides ready-to-use sample agents built on top of the [Agent Development Kit (ADK)](https://adk.dev). It is a **multi-language** repository:

| Folder | Language / Platform |
|--------|---------------------|
| `python/` | Python agents |
| `typescript/` | TypeScript agents |
| `go/` | Go agents |
| `java/` | Java agents |
| `kotlin/` | Kotlin agents |
| `android/` | Android agents |

Each folder has its own `README.md` with language-specific setup and run instructions.

## Before You Start

- Install ADK for your target language. See the [ADK Installation Guide](https://adk.dev/get-started) and the language-specific READMEs.
- Sign the [Google Contributor License Agreement (CLA)](https://cla.developers.google.com/) if you have not already.
- Review [Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Development Commands

There is no single top-level build command. Use the instructions in the relevant language folder:

- **Python**: typically `pip install` / `uv add` dependencies and run the sample agent entry point.
- **TypeScript**: typically `npm install` and `npm run` the sample script.
- **Go**: typically `go run` the sample.
- **Java / Kotlin / Android**: follow the Gradle instructions in the corresponding README.

## Contribution Tips for Agents

- Keep changes scoped to one language / sample at a time.
- Do not add unrelated samples in the same PR.
- Update the relevant `README.md` if your change affects how a sample is run.
- This project is for demonstration purposes; avoid changes that imply production support.
- All submissions require review via GitHub pull requests.

## Getting Help

Report problems or ask questions through [GitHub issues](https://github.com/google/adk-samples/issues).
