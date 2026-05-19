# Fun Facts Agent (Android)

A simple agent built with [Agent Development Kit (ADK)](https://adk.dev) for
Android. It uses the Gemini model to provide interesting and surprising fun
facts about any topic, with a Jetpack Compose chat interface.

## Overview

The Fun Facts Agent is a single-agent sample designed to demonstrate how to
integrate ADK Kotlin into a native Android application. It wraps the same
fun-facts agent from the [Kotlin samples](../../kotlin/agents/fun-facts/) in a
Compose-based chat UI, using `InMemoryRunner` inside a ViewModel to manage
agent interactions.

## Agent Details

| Feature | Description |
| --- | --- |
| **Interaction Type** | Conversational |
| **Complexity** | Beginner |
| **Agent Type** | Single Agent |
| **Components** | Model: Gemini |
| **Vertical** | Horizontal |

## Project Structure

```
fun-facts/
    app/src/main/java/com/example/adkdemoapp/
        MainActivity.kt             # Compose Activity entry point
        ChatScreen.kt               # Chat UI composable
        ChatViewModel.kt            # ViewModel using InMemoryRunner
        agents/
            FunFactsAgent.kt        # Agent definition (LlmAgent)
        ui/theme/
            Color.kt, Theme.kt, Type.kt
    app/build.gradle.kts            # App module configuration
    build.gradle.kts                # Root project configuration
    settings.gradle.kts
    README.md
```

## Setup and Installation

### Prerequisites

- [Android Studio](https://developer.android.com/studio) (latest stable)
- Android SDK (compileSdk 34+, minSdk 26+)
- Java 17 or later
- A [Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

1.  Clone the repository and navigate to the agent directory:

    ```bash
    cd android/agents/fun-facts
    ```

2.  Configure your Gemini API key using one of the following methods:

    **Option A: Environment variable (recommended)**

    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```

    **Option B: local.properties**

    Add the following line to `local.properties` in the project root:

    ```properties
    GEMINI_API_KEY=your_api_key_here
    ```

3.  Open the project in Android Studio, or build from the command line:

    ```bash
    ./gradlew :app:assembleDebug
    ```

## Running the Agent

### Android Studio

Open the `fun-facts` directory as a project in Android Studio, then run the
app on an emulator or connected device using the Run button.

### Command line

Build and install the debug APK:

```bash
./gradlew :app:installDebug
```

## How it Works

- **`FunFactsAgent.kt`**: Defines the `LlmAgent` with a Gemini model and a
  prompt instructing it to provide surprising fun facts.
- **`ChatViewModel.kt`**: Creates an `InMemoryRunner` and calls `runAsync()`
  to send user messages to the agent. Streaming responses are collected and
  displayed incrementally.
- **`ChatScreen.kt`**: A Jetpack Compose UI with a scrollable message list
  and a text input field.

## Example Interaction

Below is an example interaction with the Fun Facts Agent. Note that the exact
output of the agent may be different every time.

```
User: Tell me fun facts about octopuses

Agent: Here are some mind-blowing fun facts about octopuses:

* Octopuses have three hearts: two pump blood to the gills, while the third
  pumps it to the rest of the body. The heart that delivers blood to the body
  actually stops beating when the octopus swims, which is why they prefer
  crawling.

* They have blue blood. Their blood uses copper-based hemocyanin instead of
  iron-based hemoglobin, making it more efficient at transporting oxygen in
  cold, low-oxygen environments.
```

## Disclaimer

This agent sample is provided for illustrative purposes only and is not intended
for production use. It serves as a basic example of an agent and a foundational
starting point for individuals or teams to develop their own agents.

This sample has not been rigorously tested, may contain bugs or limitations, and
does not include features or optimizations typically required for a production
environment (e.g., robust error handling, security measures, scalability,
performance considerations, comprehensive logging, or advanced configuration
options).

Users are solely responsible for any further development, testing, security
hardening, and deployment of agents based on this sample. We recommend thorough
review, testing, and the implementation of appropriate safeguards before using
any derived agent in a live or critical system.
