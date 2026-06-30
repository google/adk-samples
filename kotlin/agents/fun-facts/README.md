# Fun Facts Agent (Kotlin)

A simple agent built with [Agent Development Kit (ADK)](https://adk.dev) for
Kotlin. It uses the Gemini model and Google Search grounding to provide
interesting and surprising fun facts about any topic you ask about.

## Overview

The Fun Facts Agent is a single-agent sample designed to be as simple as
possible, serving as a starting point for building agents with ADK for Kotlin.
When you provide a topic, the agent uses Google Search to find obscure and
surprising information, then presents it as fun facts with maximum "wow" factor.

## Agent Details

| Feature | Description |
| --- | --- |
| **Interaction Type** | Conversational |
| **Complexity** | Beginner |
| **Agent Type** | Single Agent |
| **Components** | Tools: Google Search (built-in) |
| **Vertical** | Horizontal |

## Project Structure

```
fun-facts/
    src/main/kotlin/com/google/adk/samples/agents/funfacts/
        FunFactsAgent.kt       # Agent definition
        Main.kt                # CLI entry point (ReplRunner)
        WebMain.kt             # Web UI entry point (AdkWebServer)
    build.gradle.kts           # Project configuration
    settings.gradle.kts
    README.md
```

## Setup and Installation

### Prerequisites

- Java 17 or later
- Gradle 8.0 or later
- A [Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

1.  Clone the repository and navigate to the agent directory:

    ```bash
    cd kotlin/agents/fun-facts
    ```

2.  Set your Gemini API key as an environment variable:

    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```

3.  Install dependencies and build:

    ```bash
    gradle build
    ```

## Running the Agent

### Command-line interface

Run the agent interactively in your terminal:

```bash
gradle run
```

### Web interface

Launch the ADK web UI to chat with the agent and inspect execution traces:

```bash
gradle run -PmainClass=com.google.adk.samples.agents.funfacts.WebMainKt
```

Then open http://localhost:8080 in your browser.

## Example Interaction

Below is an example interaction with the Fun Facts Agent. Note that the exact
output of the agent may be different every time.

```
user: Tell me fun facts about octopuses

fun_facts: Here are some mind-blowing fun facts about octopuses:

* Octopuses have three hearts: two pump blood to the gills, while the third
  pumps it to the rest of the body. The heart that delivers blood to the body
  actually stops beating when the octopus swims, which is why they prefer
  crawling.

* They have blue blood. Their blood uses copper-based hemocyanin instead of
  iron-based hemoglobin, making it more efficient at transporting oxygen in
  cold, low-oxygen environments.

* An octopus has about 500 million neurons, and roughly two-thirds of them are
  located in its arms. Each arm can essentially "think" and act independently,
  tasting and touching as it goes.
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
