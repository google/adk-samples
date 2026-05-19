# Agent Development Kit (ADK) Android Samples

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

<img src="https://github.com/google/adk-docs/blob/main/docs/assets/agent-development-kit.png" alt="Agent Development Kit Logo" width="150">

This collection provides ready-to-use sample agents built as native Android
applications on top of [ADK
Kotlin](https://github.com/google/adk-kotlin). Each sample is a complete
Android app that demonstrates how to integrate ADK agents into a mobile
experience using Jetpack Compose.

## Getting Started with Android Samples

Follow these steps to set up and run the sample agents:

1.  **Prerequisites:**
    *   **Install ADK Kotlin:** Follow the
        [Android setup guide](https://adk.dev/get-started/installation/#kotlin)
        for project configuration and dependency setup.
    *   **Android Studio** (latest stable release) with Android SDK
        (compileSdk 34+, minSdk 26+).
    *   **Java 17 or later.**
    *   **Set Up API Key:** Each sample requires a Gemini API key. You can
        create a key in Google AI Studio on the
        [API Keys](https://aistudio.google.com/app/apikey) page. Provide it
        via the `GOOGLE_API_KEY` environment variable or in
        `local.properties`.

2.  **Clone this repository:**

    To start working with the ADK Android samples, first clone the public
    `adk-samples` repository:
    ```bash
    git clone https://github.com/google/adk-samples.git
    cd adk-samples/android
    ```

3.  **Explore the Agents:**

    *   Navigate to the `agents/` directory.
    *   Browse the subdirectories. Each contains a complete Android
        application with its own `README.md`.

4.  **Run an Agent:**
    *   Choose an agent from the `agents/` directory.
    *   Open the agent's directory as a project in Android Studio (e.g.,
        `agents/fun-facts`).
    *   Follow the instructions in *that agent's* `README.md` file for
        specific setup and running the app.

**Notes:**

These agents have been built and tested using
[Google models](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models).
You can test these samples with other models as well. Please refer to
[ADK Tutorials](https://adk.dev/tutorials/) to use other
models for these samples.

## Repository Structure
```bash
.
├── android                     # Contains all the Android sample code
│   ├── agents                  # Contains individual agent samples
│   │   ├── agent1              # Specific agent directory (complete Android app)
│   │   │   └── README.md       # Agent-specific instructions
│   │   ├── agent2
│   │   │   └── README.md
│   │   ├── ...
│   └── README.md               # This file (Repository overview)
```
