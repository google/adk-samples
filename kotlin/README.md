# Agent Development Kit (ADK) Kotlin Samples

> [!IMPORTANT]
> **This folder is retired.** `kotlin/agents/` is now empty. Recipes live in
> **[`core/kotlin/`](../core/kotlin/)**, curated by the ADK team, and
> **[`contrib/kotlin/`](../contrib/kotlin/)**, contributed by the community.
>
> - **Contributing a new recipe?** It belongs in `contrib/kotlin/`. Start with
>   the [recipe checklist](../docs/recipe-checklist.md) and the
>   [Kotlin guidance](../docs/recipe-handbook/languages/kotlin.md). See also the
>   [contributor guide](../docs/README.md).
> - **Looking for the LLM Auditor?** It now lives at
>   [`core/kotlin/llm-auditor/`](../core/kotlin/llm-auditor/).
>
> Pull requests that add or modify files under `kotlin/agents/` fail CI.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)

<img src="https://github.com/google/adk-docs/blob/main/docs/assets/agent-development-kit.png" alt="Agent Development Kit Logo" width="150">

[`core/kotlin/`](../core/kotlin/) and [`contrib/kotlin/`](../contrib/kotlin/)
provide ready-to-use sample agents built on top of
[ADK Kotlin](https://github.com/google/adk-kotlin). These agents cover a range
of common use cases and complexities, from simple conversational bots to
complex multi-agent workflows.

> **Building for Android?** See the
> [Build ADK agents for Android](https://developer.android.com/ai/adk) guide.

## Getting Started

Install and configure ADK Kotlin by following the
[Kotlin Quickstart](https://adk.dev/get-started/kotlin/). You will need **Java 17
or later** and **Gradle 8.0 or later**, plus a `GOOGLE_API_KEY` environment
variable for the Gemini API — create a key in Google AI Studio on the
[API Keys](https://aistudio.google.com/app/apikey) page.

Then pick an agent under [`core/kotlin/`](../core/kotlin/) or
[`contrib/kotlin/`](../contrib/kotlin/) and follow the instructions in *that
agent's* `README.md`.

**Notes:**

These agents have been built and tested using
[Google models](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models).
You can test these samples with other models as well. Please refer to
[ADK Tutorials](https://adk.dev/tutorials/) to use other
models for these samples.
