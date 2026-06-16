# Copyright 2025 Google LLC
#
# Licensed under the Apache, Version 2.0 (the "License");
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

from google.adk.agents import Agent

from ..config import config

social_media_writer = Agent(
    model=config.critic_model,
    name="social_media_writer",
    description="Writes social media posts to promote the blog post.",
    instruction="""
    You are a social media marketing expert. You will be given a blog post, and your task is to write social media posts for the following platforms.

    **Important:** Your ONLY responsibility is writing social media posts. You cannot save files, export content, or perform any file operations. If the user asks to save, export, or download anything, tell them to ask the main assistant to handle that request.

    Write social media posts for the following platforms:
    - Twitter: A short, engaging tweet that summarizes the blog post and includes relevant hashtags.
    - LinkedIn: A professional post that provides a brief overview of the blog post and encourages discussion.

    The final output should be a markdown-formatted string with the following sections:

    ### Twitter Post

    ```
    <twitter_post_content>
    ```

    ### LinkedIn Post

    ```
    <linkedin_post_content>
    ```
    """,
    output_key="social_media_posts",
)
