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


from google.adk.agents import (
    LlmAgent,
    LoopAgent,
    ParallelAgent,
    SequentialAgent,
)
from google.genai import types

from . import instructions

# --- Configuration Constants ---
APP_NAME = "collaborative_story_writer"
SESSION_ID = "story_session_v1"
MODEL_NAME = "gemini-2.5-flash"
USER_ID = "author_user_01"

# User-Defined Constraints
N_CHAPTERS = 3  # Number of chapters to write
MAX_WORDS = 100  # Max words per chapter

# --- State Keys ---
KEY_USER_PROMPT = "user_prompt"
KEY_ENHANCED_PROMPT = "enhanced_prompt"
KEY_CURRENT_STORY = "current_story"
KEY_CREATIVE_CANDIDATE = "creative_chapter_candidate"
KEY_FOCUSED_CANDIDATE = "focused_chapter_candidate"
KEY_FINAL_STORY = "final_story"


def set_initial_story(callback_context, llm_request):
    callback_context.state[KEY_CURRENT_STORY] = "Chapter 1"


# --- 1. Agent Definitions ---

# Expands the user's simple idea into a full premise.
prompt_enhancer = LlmAgent(
    name="PromptEnhancerAgent",
    model=MODEL_NAME,
    instruction=instructions.PROMPT_ENHANCER_INSTRUCTION,
    description="Expands user prompt into a full story premise.",
    output_key=KEY_ENHANCED_PROMPT,
    before_model_callback=set_initial_story,
)

# Focuses on novelty and twists (High Temperature).
creative_writer = LlmAgent(
    name="CreativeStoryTellerAgent",
    model=MODEL_NAME,
    # High temperature for creativity/randomness
    generate_content_config=types.GenerateContentConfig(temperature=0.9),
    instruction=instructions.CREATIVE_WRITER_INSTRUCTION.format(
        max_words=MAX_WORDS
    ),
    description="Writes a creative, high-temperature chapter draft.",
    output_key=KEY_CREATIVE_CANDIDATE,
)

# Focuses on logic and consistency (Low Temperature).
focused_writer = LlmAgent(
    name="FocusedStoryTellerAgent",
    model=MODEL_NAME,
    # Low temperature for consistency/logic
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    instruction=instructions.FOCUSED_WRITER_INSTRUCTION.format(
        max_words=MAX_WORDS
    ),
    description="Writes a consistent, low-temperature chapter draft.",
    output_key=KEY_FOCUSED_CANDIDATE,
)

# Selects the best chapter and appends it to the story.
# Note: This agent reads the current story AND the candidates, then outputs the UPDATED full story.
critique_agent = LlmAgent(
    name="CritiqueAgent",
    model=MODEL_NAME,
    instruction=instructions.CRITIQUE_AGENT_INSTRUCTION,
    description="Selects the best chapter and updates the story state.",
    output_key=KEY_CURRENT_STORY,  # Overwrites current_story with the extended version
)

# Final Polish.
editor_agent = LlmAgent(
    name="EditorAgent",
    model=MODEL_NAME,
    instruction=instructions.EDITOR_AGENT_INSTRUCTION,
    description="Polishes the final draft.",
    output_key=KEY_FINAL_STORY,
)

# --- 2. Workflow Structure ---

# Runs the Creative and Focused writers at the same time.
parallel_writers = ParallelAgent(
    name="ParallelChapterGenerators",
    sub_agents=[creative_writer, focused_writer],
    description="Generates two chapter options in parallel.",
)

# This sequence runs inside the loop (Generate -> Critique/Append).
chapter_cycle = SequentialAgent(
    name="ChapterGenerationCycle",
    sub_agents=[parallel_writers, critique_agent],
    description="Runs parallel writers then selects the best chapter.",
)

# Repeats the Chapter Cycle N times.
story_loop = LoopAgent(
    name="StoryBuildingLoop",
    sub_agents=[chapter_cycle],
    max_iterations=N_CHAPTERS,
    description=f"Iteratively writes {N_CHAPTERS} chapters.",
)

# Enhance Prompt -> Loop Chapters -> Final Edit
root_agent = SequentialAgent(
    name="CollaborativeStoryWorkflow",
    sub_agents=[prompt_enhancer, story_loop, editor_agent],
    description="End-to-end story generation pipeline.",
)
