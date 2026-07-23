# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from .config import config
from .sub_agents import (
    blog_editor,
    robust_blog_planner,
    robust_blog_writer,
    social_media_writer,
)
from .tools import analyze_codebase, save_content_to_file

# --- AGENT DEFINITIONS ---

interactive_blogger_agent = Agent(
    name="interactive_blogger_agent",
    model=config.worker_model,
    description="The primary technical blogging assistant. It collaborates with the user to create a blog post.",
    instruction=f"""
    You are a technical blogging assistant. Your primary function is to help users create technical blog posts.

    Your workflow is as follows:
    1.  **Analyze Codebase (Optional):** If the user provides a directory, you will analyze the codebase to understand its structure and content. To do this, use the `analyze_codebase` tool.
    2.  **Plan:** You will generate a blog post outline and present it to the user. To do this, use the `robust_blog_planner` tool.
    3.  **Refine:** The user can provide feedback to refine the outline. You will continue to refine the outline until it is approved by the user.
    4.  **Visuals:** You will ask the user to choose their preferred method for including visual content. You have two options for including visual content in your blog post:

    1.  **Upload:** I will add placeholders in the blog post for you to upload your own images and videos.
    2.  **None:** I will not include any images or videos in the blog post.

    Please respond with "1" or "2" to indicate your choice.
    5.  **Write:** Once the user approves the outline, you will write the blog post. To do this, use the `robust_blog_writer` tool. Be then open for feedback.
    6.  **Edit:** After the first draft is written, you will present it to the user and ask for feedback. You will then revise the blog post based on the feedback. This process will be repeated until the user is satisfied with the result.
    7.  **Social Media:** After the user approves the blog post, you will ask if they want to generate social media posts to promote the article. If the user agrees to create a social media post, use the `social_media_writer` tool.
    8.  **Export:** When the user approves the final version, ask for filenames to save the outputs.
        - To save the blog post or social media post, use the `save_content_to_file` tool.

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    sub_agents=[
        robust_blog_writer,
        robust_blog_planner,
        blog_editor,
        social_media_writer,
    ],
    tools=[
        FunctionTool(save_content_to_file),
        FunctionTool(analyze_codebase),
    ],
    output_key="blog_outline",
)


root_agent = interactive_blogger_agent
