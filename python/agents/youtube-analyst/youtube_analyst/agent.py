import os
import pathlib

from google import genai
from google.adk.agents import Agent
from google.adk.skills import load_skill_from_dir
from google.adk.tools import load_artifacts  # type: ignore
from google.adk.tools.skill_toolset import SkillToolset

from .common.llm import GeminiWithLocation
from .common.utils import load_prompt
from .config import config
from .tools import (
    aggregate_comment_sentiment,
    calculate_engagement_metrics,
    get_channel_details,
    get_comment_replies,
    get_current_date_time,
    get_date_range,
    get_trending_videos,
    get_video_comments,
    get_video_details,
    get_video_transcript,
    publish_file,
    render_html,
    search_channel_videos,
    search_youtube,
    store_youtube_api_key,
    submit_feedback,
)
from .visualization_agent import visualization_agent

# ---------------------------------------------------------------------------
# YouTube Analyst Agent (Root)
# ---------------------------------------------------------------------------

# Dynamic paths for skills and prompts
skills_root = pathlib.Path(__file__).parent / "skills"

skills = [
    load_skill_from_dir(skills_root / name)
    for name in [
        "abcd-framework-audit",
        "creative-insight-analyzer",
        "daily-briefing",
        "debate-synthesizer",
        "deep-exploration",
        "industry-landscape-briefing",
        "kol-discovery",
        "multi-video-synthesis",
        "poi-discovery-briefing",
        "product-launch-audit",
        "sentiment-analysis",
        "visualization-reporting",
    ]
]

youtube_agent = Agent(
    model=GeminiWithLocation(
        model=config.agent_settings.model, location=config.GOOGLE_GENAI_LOCATION
    ),
    name="youtube_analyst",
    description="Agent for YouTube analysis and data retrieval",
    instruction=load_prompt(os.path.dirname(__file__), "youtube_agent.txt"),
    sub_agents=[visualization_agent],
    tools=[
        search_youtube,
        store_youtube_api_key,
        get_trending_videos,
        get_video_details,
        get_channel_details,
        search_channel_videos,
        get_video_comments,
        get_comment_replies,
        get_video_transcript,
        calculate_engagement_metrics,
        aggregate_comment_sentiment,
        get_current_date_time,
        get_date_range,
        publish_file,
        render_html,
        submit_feedback,
        # Re-exporting skills as tools if needed, or use the SkillToolset
        SkillToolset(skills=skills),
        # Skill loading tools for dynamic execution
        # load_skills, load_skill, load_skill_resource
        load_artifacts,
    ],
    generate_content_config=genai.types.GenerateContentConfig(
        max_output_tokens=config.YOUTUBE_AGENT_MAX_OUTPUT_TOKENS,
    ),
)

root_agent = youtube_agent
