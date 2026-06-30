# Image Scoring Agent

This agent functions as an automated image generation and evaluation system specifically designed to create and validate images based on text descriptions while adhering to predefined policies. Its primary role is to ensure that generated images meet specific compliance and quality standards by iteratively generating and evaluating images until they satisfy policy requirements.

## Overview

This agent generates and evaluates images based on text descriptions while ensuring compliance with predefined policies. Its primary purpose is to serve as an automated image generation and validation system that maintains high standards of quality and policy compliance.

*   Generates images from text descriptions using Imagen
*   Evaluates generated images against a set of predefined policies
*   Iteratively improves images that don't meet policy requirements
*   Provides detailed scoring and feedback for each generated image

This sample agent enables users to generate images from text descriptions while ensuring the output meets specific policy requirements through an automated evaluation and iteration process.

## Agent Architecture

This diagram shows the detailed architecture of the agents and tools used to implement this workflow.

<img src="image_scoring_architecture.png" alt="Image Scoring Architecture" width="800"/>

## Agent Details

| Feature | Description |
| --- | --- |
| **Interaction Type** | Workflow |
| **Complexity**  | Medium |
| **Agent Type**  | Multi Agent |
| **Components**  | Tools: Imagen, Image Evaluation Tools |
| **Vertical**  | Horizontal |

## Setup and Installation

### Prerequisites

- Python 3.10+
- uv for dependency management and packaging
  - See the official [uv website](https://docs.astral.sh/uv/) for installation.

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Google Agents CLI (recommended)

Use the [Google Agents CLI](https://github.com/google/agents-cli) to scaffold a production-ready project and choose your deployment target ([Agent Runtime](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime) or [Cloud Run](https://cloud.google.com/run)), with CI/CD and other production features.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-image-scorer` with your project name):

```bash
agents-cli create my-image-scorer -a adk@image-scoring
```

The Google Agents CLI will prompt you to select deployment options and set up your Google Cloud project.

From your newly created project directory (e.g. `my-image-scorer`), run:

```bash
cd my-image-scorer
uv sync --dev
uv run adk run image_scoring
```

For the web UI:

```bash
uv run adk web
```

Then select `image_scoring` from the dropdown menu.

---

<details>
<summary>Alternative: Local development (run from this sample repo)</summary>

### Agent Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/google/adk-samples.git
   cd adk-samples/python/agents/image-scoring
   ```

   For the rest of this tutorial **ensure you remain in the `python/agents/image-scoring` directory**.

2. Install the dependencies:

   ```bash
   uv sync --dev
   ```

3. Configure settings:

   There is a `.env-example` file included in the repository. Update this file with the values appropriate to your project, and save it as `.env`. The values in this file will be read into the environment of your application.

   Authenticate your GCloud account:

   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project $GOOGLE_CLOUD_PROJECT
   ```

### Running the Agent Locally

You can run the agent locally using the `adk` command in your terminal. Here are some example requests:

*   `a peaceful mountain landscape at sunset`
*   `a cat riding a bicycle`

1. To run the agent from the CLI:

   ```bash
   adk run image_scoring
   ```

2. To run the agent from the ADK web UI:

   ```bash
   adk web
   ```
   Then select `image_scoring` from the dropdown.

### Development

```bash
uv sync --dev
uv run pytest eval
```

`eval` is a demonstration of how to evaluate the agent, using the `AgentEvaluator` in ADK. It sends a sample request to the image_scoring agent and checks if the tool usage is as expected.

</details>

## Customization

The Image Scoring Agent can be customized to better suit your requirements. For example:

1.  **Policy Customization:** Modify the policy evaluation criteria to match your specific requirements and standards.
2.  **Image Generation Parameters:** Adjust the Imagen parameters to control image generation quality and characteristics.
3.  **Evaluation Metrics:** Add or modify evaluation metrics to assess different aspects of the generated images.
4.  **Iteration Strategy:** Customize the iteration process to optimize for specific aspects of image quality or policy compliance.

## Sub-Agents and Workflow

The Image Scoring Agent implements a sequential workflow using the following sub-agents:

1. **Prompt Generation Agent** (`prompt_agent.py`)
   * Primary responsibility: Creates optimized prompts for Imagen based on input text descriptions
   * Uses Gemini model to generate prompts that comply with policies
   * Outputs prompts that are stored in session state for image generation

2. **Image Generation Agent** (`imagen_agent.py`)
   * Primary responsibility: Generates images using Imagen 3.0 based on the prompts
   * Configures image generation parameters (aspect ratio, safety filters, etc.)
   * Saves generated images to Google Cloud Storage (GCS)
   * Stores image artifacts and GCS URIs in session state

3. **Scoring Agent** (`scoring_agent.py`)
   * Primary responsibility: Evaluates generated images against policy rules
   * Loads policy rules from `policy.json`
   * Analyzes images and assigns scores (0-5) for each policy criterion
   * Computes total score and stores it in session state
   * Provides detailed scoring feedback for each policy rule

4. **Checker Agent** (`checker_agent.py`)
   * Primary responsibility: Evaluates if the generated image meets quality thresholds
   * Manages iteration count and maximum iteration limits
   * Compares total score against configured threshold (default: 10)
   * Controls workflow termination based on score or iteration limits

### Workflow Sequence
1. The workflow starts with the Prompt Generation Agent creating an optimized prompt
2. The Image Generation Agent uses this prompt to create an image with Imagen 3.0
3. The Scoring Agent evaluates the generated image against policy rules
4. The Checker Agent determines if the score meets the threshold
5. If the score is below threshold and max iterations not reached, the process repeats
6. The workflow terminates when either:
   * The image score meets or exceeds the threshold
   * The maximum number of iterations is reached
