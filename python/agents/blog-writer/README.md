# Blogger Agent

This project contains the core logic for the Blogger Agent, a multi-agent system designed to assist users in creating technical blog posts. The agent is built using the Google Agent Development Kit (ADK) and follows a modular architecture.

## Getting Started with uv

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Installation

First, you need to install uv. You can do this using `pip`:

```bash
pip install uv
```

### Project Setup

Once uv is installed, you can install the project's dependencies:

```bash
uv pip install -r requirements.txt
```

This will create a virtual environment for the project and install all the required libraries.

### Running the Agent and Tests

To run any scripts or tests within the project's virtual environment, use `uv run`:

**Run the main agent (interactive playground):**

```bash
uv run adk web
```

**Run the main agent (command-line interface):**

```bash
uv run adk run blogger_agent
```

**Run the integration test:**

```bash
uv run python -m tests.test_agent
```

**Run the evaluation (smoke test):**

```bash
uv run pytest eval/test_eval.py
```

## Alternative: Using Google Agents CLI

You can also use the [Google Agents CLI](https://github.com/google/agents-cli) to create a production-ready version of this agent with additional deployment options.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-blogger-agent` with your project name):

```bash
agents-cli create my-blogger-agent -a adk@blogger-agent
```

The Google Agents CLI will prompt you to select deployment options and provides additional production-ready features including automated CI/CD deployment scripts.

## Project Structure

The project is organized as follows:

*   `blogger_agent/`: The main Python package for the agent.
    *   `agent.py`: Defines the main `interactive_blogger_agent` and orchestrates the sub-agents.
    *   `sub_agents/`: Contains the individual sub-agents, each responsible for a specific task.
        *   `blog_planner.py`: Generates the blog post outline.
        *   `blog_writer.py`: Writes the blog post.
        *   `blog_editor.py`: Edits the blog post based on user feedback.
        *   `social_media_writer.py`: Generates social media posts.
    *   `tools.py`: Defines the custom tools used by the agents.
    *   `config.py`: Contains the configuration for the agents, such as the models to use.
*   `eval/`: Contains the evaluation framework for the agent.
*   `tests/`: Contains integration tests for the agent.

## Agent Architecture

The Blogger Agent is a multi-agent system composed of a main orchestrator agent and several specialized sub-agents.

![adk_web.png](adk_web.png)

### Main Agent

*   **`interactive_blogger_agent`**: This is the main agent that interacts with the user. It manages the workflow of creating a blog post and delegates tasks to the appropriate sub-agents.

### Sub-Agents

The sub-agents are defined in the `blogger_agent/sub_agents/` directory. Each sub-agent is responsible for a specific task in the blog post creation process:

*   **`robust_blog_planner`**: Generates a blog post outline. It uses a loop to ensure a valid outline is created and can use Google Search to gather information.
*   **`robust_blog_writer`**: Writes the blog post based on the approved outline. It can also use Google Search to find relevant information and examples.
*   **`blog_editor`**: Edits the blog post based on user feedback.
*   **`social_media_writer`**: Generates social media posts to promote the blog post.

## Tools

The agents use the following custom tools, defined in `blogger_agent/tools.py`:

*   **`save_blog_post_to_file`**: Saves the blog post to a file.
*   **`analyze_codebase`**: Analyzes the codebase in a given directory to provide context for the blog post.

The agents also use the built-in `google_search` tool.

## Workflow

The `interactive_blogger_agent` follows this workflow:

1.  **Analyze Codebase (Optional):** If the user provides a directory, the agent analyzes the codebase to understand its structure and content.
2.  **Plan:** The agent delegates the task of generating a blog post outline to the `robust_blog_planner`.
3.  **Refine:** The user can provide feedback to refine the outline. The agent continues to refine the outline until it is approved by the user.
4.  **Visuals:** The agent asks the user to choose their preferred method for including visual content.
5.  **Write:** Once the user approves the outline, the agent delegates the task of writing the blog post to the `robust_blog_writer`.
6.  **Edit:** After the first draft is written, the agent presents it to the user and asks for feedback. The `blog_editor` revises the blog post based on the feedback. This process is repeated until the user is satisfied with the result.
7.  **Social Media:** After the user approves the blog post, the agent asks if they want to generate social media posts. If the user agrees, the `social_media_writer` is used.
8.  **Export:** When the user approves the final version, the agent asks for a filename and saves the blog post as a markdown file using the `save_blog_post_to_file` tool.

## Example Conversation

**User:** I want to write a blog post about the new Google Gemini 2.5 Flash Preview model, also known as Nanobanana.

**Agent:** I can do that. Here is a blog post outline for an article about the Google Gemini 2.5 Flash Preview model.

---

## Blog Post Outline: Introducing Gemini 2.5 Flash: Google's Newest, Fastest AI Model

### I. Title
Gemini 2.5 Flash: The Speed and Efficiency of Nanobanana

### II. Introduction
*   **Hook:** Announcing Google's latest AI model, Gemini 2.5 Flash, nicknamed Nanobanana.
*   **Context:** Explain that this new model is designed for speed and efficiency, making it ideal for on-device applications.
*   **Thesis Statement:** This post will explore the key features of Gemini 2.5 Flash, its capabilities, and how it compares to other models in the Gemini family.

### III. Main Body

#### A. What is Gemini 2.5 Flash?
*   **Lightweight and Cost-Effective:** Describe how Gemini 2.5 Flash is a smaller, more efficient model compared to its predecessors.
*   **Multimodal Capabilities:** Explain that it can process text, audio, images, and video.
*   **1-Million Token Context Window:** Discuss the large context window and its implications for processing large amounts of information.
*   **'Thinking' Capability:** Detail the new feature that provides transparency into the model's reasoning process.

#### B. Key Features and Enhancements
*   **Speed and Efficiency:** Emphasize the model's performance on low-power devices.
*   **Controlled Generation, Function Calling, and Search:** Discuss the model's advanced functionalities.
*   **Specialized Versions:** Mention the specialized versions for live API, image generation, and text-to-speech.

#### C. How to Get Access
*   **Availability:** Explain that the model is available through the Gemini API, Google AI Studio, and Vertex AI.

### IV. Conclusion
*   **Summary:** Recap the key features and benefits of Gemini 2.5 Flash.
*   **Call to Action:** Encourage readers to explore the new model and its capabilities.

---

**User:** looks good, write it

**Agent:** You have two options for including visual content in your blog post...
