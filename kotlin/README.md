# Agent Development Kit (ADK) Kotlin Samples

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

<img src="https://github.com/google/adk-docs/blob/main/docs/assets/agent-development-kit.png" alt="Agent Development Kit Logo" width="150">

This collection provides ready-to-use sample agents built on top of [ADK
Kotlin](https://github.com/google/adk-kotlin). These agents cover a range of
common use cases and complexities, from simple conversational bots to complex
multi-agent workflows.

## Getting Started with Kotlin Samples

Follow these steps to set up and run the sample agents:

1.  **Prerequisites:**
    *   **Install ADK Kotlin:** Ensure you have ADK
        Kotlin installed and configured. Follow the
        [Kotlin Quickstart](https://adk.dev/get-started/kotlin/)
        for setup instructions.
    *   **Java 17 or later** and **Gradle 8.0 or later** installed.
    *   **Set Up Environment Variables:** Each agent example requires a
        `GOOGLE_API_KEY` environment variable for the Gemini API. You can create
        a key in Google AI Studio on the
        [API Keys](https://aistudio.google.com/app/apikey) page.

2.  **Clone this repository:**

    To start working with the ADK Kotlin samples, first clone the public
    `adk-samples` repository:
    ```bash
    git clone https://github.com/google/adk-samples.git
    cd adk-samples/kotlin
    ```

3.  **Explore the Agents:**

    *   Navigate to the `agents/` directory.
    *   Browse the subdirectories. Each contains a specific sample agent with
        its own `README.md`.

4.  **Run an Agent:**
    *   Choose an agent from the `agents/` directory.
    *   Navigate into that agent's specific directory (e.g.,
        `cd agents/fun-facts`).
    *   Follow the instructions in *that agent's* `README.md` file for specific
        setup and running the agent.

**Notes:**

These agents have been built and tested using
[Google models](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models).
You can test these samples with other models as well. Please refer to
[ADK Tutorials](https://adk.dev/tutorials/) to use other
models for these samples.

## Repository Structure
```bash
.
├── kotlin                      # Contains all the Kotlin sample code
│   ├── agents                  # Contains individual agent samples
│   │   ├── agent1              # Specific agent directory
│   │   │   └── README.md       # Agent-specific instructions
│   │   ├── agent2
│   │   │   └── README.md
│   │   ├── ...
│   └── README.md               # This file (Repository overview)
```
