import datetime
import json
import logging
import math
import os
import random
import re
import string

from google.adk.agents.context import Context as ToolContext
from google.cloud import storage
from google.genai import types
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi

from .config import config

logger = logging.getLogger(__name__)


def init_or_get_youtube_client(tool_context: ToolContext):
    """
    Initializes or retrieves the YouTube Data API client.
    
    Priority:
    1. Session state (onboarded by user)
    2. Environment variable (pre-configured in config)

    Returns:
        tuple: (youtube_client, error_message)
        If client is None, error_message contains the onboarding instructions.
    """
    # 1. Check session state first (allows user to override environment key)
    api_key = tool_context.state.get("youtube_api_key")

    # 2. Fallback to pre-configured environment key
    if not api_key:
        api_key = config.YOUTUBE_API_KEY

    if not api_key:
        error_msg = (
            "SYSTEM REJECTION: Missing YouTube API Key. "
            "I cannot access YouTube data without a valid API key. "
            "SUGGESTED ACTION: "
            "1. Visit https://developers.google.com/youtube/v3/getting-started to apply for a YouTube Data API v3 key. "
            "2. Once you have the key, please provide it to me. "
            "3. I will then use the 'store_youtube_api_key' tool to save it for this session."
        )
        return None, error_msg

    try:
        client = build("youtube", "v3", developerKey=api_key)
        return client, None
    except Exception as e:
        return None, f"ERROR: Failed to initialize YouTube client: {e!s}"


def store_youtube_api_key(api_key: str, tool_context: ToolContext) -> str:
    """
    Stores the YouTube API key in the persistent tool context state.
    Call this tool when the user provides a new API key.

    Args:
        api_key: The YouTube Data API v3 key provided by the user.
        tool_context: The ADK tool context.
    """
    tool_context.state["youtube_api_key"] = api_key
    return "YouTube API key has been stored successfully. You can now proceed with YouTube data requests."


def search_youtube(
    query: str,
    tool_context: ToolContext,
    max_results: int = 10,
    published_after: str = "",
    region_code: str = "",
    relevance_language: str = "",
    video_duration: str = "any",
) -> list[dict] | str:
    """
    Searches YouTube for videos matching the query with advanced filtering.

    Args:
        query: The search term.
        tool_context: The ADK tool context.
        max_results: Maximum number of results to return (default 10).
        published_after: Filter for videos published after this date (RFC 3339 format, e.g., '2023-01-01T00:00:00Z').
        region_code: ISO 3166-1 alpha-2 country code (e.g., 'US', 'GB', 'JP'). Returns videos viewable in this region.
        relevance_language: ISO 639-1 two-letter language code (e.g., 'en', 'es', 'zh-Hans'). Instructs the API to return results most relevant to this language.
        video_duration: Filter by duration. Acceptable values are 'any', 'long' (> 20 mins), 'medium' (4-20 mins), 'short' (< 4 mins).

    Returns:
        A list of dictionaries containing video title, videoId, and channelTitle, or error message.
    """
    youtube, error = init_or_get_youtube_client(tool_context)
    if error:
        return error
    assert youtube is not None

    try:
        kwargs = {
            "q": query,
            "type": "video",
            "part": "id,snippet",
            "maxResults": max_results,
            "videoDuration": video_duration,
        }
        if published_after:
            kwargs["publishedAfter"] = published_after
        if region_code:
            kwargs["regionCode"] = region_code
        if relevance_language:
            kwargs["relevanceLanguage"] = relevance_language

        search_response = youtube.search().list(**kwargs).execute()

        results = []
        for search_result in search_response.get("items", []):
            if search_result["id"]["kind"] == "youtube#video":
                results.append(
                    {
                        "title": search_result["snippet"]["title"],
                        "videoId": search_result["id"]["videoId"],
                        "channelTitle": search_result["snippet"][
                            "channelTitle"
                        ],
                        "description": search_result["snippet"]["description"],
                    }
                )
        return results
    except HttpError as e:
        logger.error(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return f"ERROR: YouTube API returned {e.resp.status}"


def get_video_details(video_ids: list[str], tool_context: ToolContext) -> list[dict] | str:
    """
    Retrieves statistics and snippet details for a list of video IDs.

    Args:
        video_ids: A list of video ID strings.
        tool_context: The ADK tool context.

    Returns:
        A list of dictionaries containing video statistics and details, or error message.
    """
    if not video_ids:
        return []

    youtube, error = init_or_get_youtube_client(tool_context)
    if error:
        return error
    assert youtube is not None

    try:
        # Join video IDs with comma
        ids_string = ",".join(video_ids)

        video_response = (
            youtube.videos()
            .list(
                id=ids_string,
                part="snippet,statistics,contentDetails,status,topicDetails",
            )
            .execute()
        )

        results = []
        for video_result in video_response.get("items", []):
            stats = video_result.get("statistics", {})
            snippet = video_result.get("snippet", {})
            content_details = video_result.get("contentDetails", {})
            status = video_result.get("status", {})
            topic_details = video_result.get("topicDetails", {})

            # Extract the best available thumbnail
            thumbnails = snippet.get("thumbnails", {})
            thumbnail_url = ""
            for quality in ["maxres", "standard", "high", "medium", "default"]:
                if quality in thumbnails:
                    thumbnail_url = thumbnails[quality]["url"]
                    break

            results.append(
                {
                    "videoId": video_result["id"],
                    "title": snippet.get("title", ""),
                    "channelTitle": snippet.get("channelTitle", ""),
                    "viewCount": stats.get("viewCount", 0),
                    "likeCount": stats.get("likeCount", 0),
                    "commentCount": stats.get("commentCount", 0),
                    "publishedAt": snippet.get("publishedAt", ""),
                    "duration": content_details.get("duration", ""),
                    "licensedContent": content_details.get(
                        "licensedContent", False
                    ),
                    "madeForKids": status.get("madeForKids", False),
                    "topicCategories": topic_details.get("topicCategories", []),
                    "thumbnail_url": thumbnail_url,
                    "tags": snippet.get("tags", []),
                }
            )
        return results
    except HttpError as e:
        logger.error(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return f"ERROR: YouTube API returned {e.resp.status}"


def get_trending_videos(
    tool_context: ToolContext, region_code: str = "US", video_category_id: str = ""
) -> list[dict] | str:
    """
    Retrieves the currently trending/most popular videos natively from YouTube, bypassing keyword search.
    Use this to proactively discover "What matters today" without needing a specific search query.

    Args:
        tool_context: The ADK tool context.
        region_code: ISO 3166-1 alpha-2 country code (e.g., 'US', 'GB', 'HK'). Default is 'US'.
        video_category_id: Optional category ID (e.g., '28' for Science & Technology, '20' for Gaming).

    Returns:
        A list of dictionaries containing trending video details, or error message.
    """
    youtube, error = init_or_get_youtube_client(tool_context)
    if error:
        return error
    assert youtube is not None

    try:
        kwargs = {
            "part": "id,snippet,statistics",
            "chart": "mostPopular",
            "regionCode": region_code,
            "maxResults": 10,
        }
        if video_category_id:
            kwargs["videoCategoryId"] = video_category_id

        video_response = youtube.videos().list(**kwargs).execute()

        results = []
        for video_result in video_response.get("items", []):
            stats = video_result.get("statistics", {})
            snippet = video_result.get("snippet", {})
            results.append(
                {
                    "videoId": video_result["id"],
                    "title": snippet.get("title", ""),
                    "channelTitle": snippet.get("channelTitle", ""),
                    "viewCount": stats.get("viewCount", 0),
                    "likeCount": stats.get("likeCount", 0),
                    "publishedAt": snippet.get("publishedAt", ""),
                }
            )
        return results
    except HttpError as e:
        logger.error(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return f"ERROR: YouTube API returned {e.resp.status}"


def get_channel_details(channel_ids: list[str], tool_context: ToolContext) -> list[dict] | str:
    """
    Retrieves statistics and snippet details for a list of channel IDs.

    Args:
        channel_ids: A list of channel ID strings.
        tool_context: The ADK tool context.

    Returns:
        A list of dictionaries containing channel statistics and details, or error message.
    """
    if not channel_ids:
        return []

    youtube, error = init_or_get_youtube_client(tool_context)
    if error:
        return error
    assert youtube is not None

    try:
        ids_string = ",".join(channel_ids)

        channel_response = (
            youtube.channels()
            .list(
                id=ids_string,
                part="snippet,statistics,brandingSettings,topicDetails",
            )
            .execute()
        )

        results = []
        for channel_result in channel_response.get("items", []):
            stats = channel_result.get("statistics", {})
            snippet = channel_result.get("snippet", {})
            branding = channel_result.get("brandingSettings", {}).get(
                "channel", {}
            )
            topics = channel_result.get("topicDetails", {})

            results.append(
                {
                    "channelId": channel_result["id"],
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "subscriberCount": stats.get("subscriberCount", 0),
                    "videoCount": stats.get("videoCount", 0),
                    "viewCount": stats.get("viewCount", 0),
                    "keywords": branding.get("keywords", ""),
                    "topicCategories": topics.get("topicCategories", []),
                }
            )
        return results
    except HttpError as e:
        logger.error(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return f"ERROR: YouTube API returned {e.resp.status}"


def get_video_comments(
    video_id: str, tool_context: ToolContext, max_results: int = 20, order: str = "time"
) -> list[str] | str:
    """
    Retrieves top-level comments for a specific video.

    Args:
        video_id: The ID of the video.
        tool_context: The ADK tool context.
        max_results: Maximum number of comments to return.
        order: The order of comments ('time' or 'relevance'). Default is 'time'.

    Returns:
        A list of comment strings, or error message.
    """
    youtube, error = init_or_get_youtube_client(tool_context)
    if error:
        return error
    assert youtube is not None

    try:
        comment_response = (
            youtube.commentThreads()
            .list(
                videoId=video_id,
                part="snippet",
                maxResults=max_results,
                textFormat="plainText",
                order=order,
            )
            .execute()
        )

        comments = []
        for item in comment_response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"][
                "textDisplay"
            ]
            comments.append(comment)
        return comments
    except HttpError as e:
        logger.error(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return f"ERROR: YouTube API returned {e.resp.status}"


def generate_timestamp_url(video_id: str, timestamp_str: str) -> str:
    """
    Generates a direct, clickable YouTube URL that jumps to a specific timestamp.

    Args:
        video_id: The ID of the YouTube video.
        timestamp_str: The timestamp string (e.g., '05:22', '1:05:22', or raw seconds '322').

    Returns:
        The exact URL (e.g., 'https://youtu.be/VIDEO_ID?t=322').
    """
    total_seconds = parse_timestamp_to_seconds(timestamp_str)
    return f"https://youtu.be/{video_id}?t={total_seconds}"


def calculate_engagement_metrics(
    view_count: int, like_count: int, comment_count: int, subscriber_count: int
) -> dict:
    """
    Calculates engagement rate and active rate based on video statistics.

    Args:
        view_count: Number of views.
        like_count: Number of likes.
        comment_count: Number of comments.
        subscriber_count: Number of subscribers.

    Returns:
        Dictionary containing 'engagement_rate' and 'active_rate' as percentages.
    """
    view_count = int(view_count) if view_count else 0
    like_count = int(like_count) if like_count else 0
    comment_count = int(comment_count) if comment_count else 0
    subscriber_count = int(subscriber_count) if subscriber_count else 0

    engagement_rate = (
        ((like_count + comment_count) / view_count * 100)
        if view_count > 0
        else 0
    )
    active_rate = (
        (view_count / subscriber_count * 100) if subscriber_count > 0 else 0
    )

    return {
        "engagement_rate": round(engagement_rate, 2),
        "active_rate": round(active_rate, 2),
    }


def calculate_match_score(
    subscribers: int,
    engagement_rate: float,
    active_rate: float,
    sentiment_score: float = 0.0,
) -> float:
    """
    Calculates a composite match score for a channel/video.

    Args:
        subscribers: Number of subscribers.
        engagement_rate: Engagement rate percentage (0-100).
        active_rate: Active rate percentage (0-100).
        sentiment_score: Sentiment score (-1 to 1).

    Returns:
        A float representing the match score (0-100).
    """
    # Log-normalize subscribers (assuming base 10, offset to avoid log(0))
    sub_log = math.log10(max(subscribers, 1))
    # Normalize sub score: loosely based on 10k-1M+ range mapping to 0-1
    sub_score = min(max((sub_log - 3) / 4, 0), 1) * 100

    # Cap engagement and active scores at decent thresholds
    eng_score = min((engagement_rate / 10), 1) * 100
    active_score = min((active_rate / 100), 1) * 100

    # Normalize sentiment (-1 to 1 -> 0 to 1)
    sent_score = ((sentiment_score + 1) / 2) * 100

    # Weights: Subs(40%) + Eng(30%) + Active(20%) + Sentiment(10%)
    total_score = (
        (sub_score * 0.4)
        + (eng_score * 0.3)
        + (active_score * 0.2)
        + (sent_score * 0.1)
    )

    return round(total_score, 1)


def analyze_sentiment_heuristic(text: str) -> float:
    """
    Simple heuristic sentiment analysis based on keyword matching.

    Args:
        text: The text to analyze.

    Returns:
        A score between -1 (negative) and 1 (positive).
    """
    if not text:
        return 0

    positives = [
        "great",
        "good",
        "love",
        "amazing",
        "best",
        "nice",
        "helpful",
        "excellent",
        "wow",
    ]
    negatives = [
        "bad",
        "hate",
        "worst",
        "terrible",
        "awful",
        "boring",
        "useless",
        "poor",
    ]

    lower_text = text.lower()
    score = 0

    for word in positives:
        if word in lower_text:
            score += 0.2

    for word in negatives:
        if word in lower_text:
            score -= 0.2

    return max(-1, min(1, score))


def get_current_date_time() -> str:
    """
    Returns the current date and time in UTC (RFC 3339 format).
    """
    return datetime.datetime.now(datetime.UTC).isoformat()


def get_date_range(time_span: str) -> str:
    """
    Calculates the date for the start of a time span relative to now.

    Args:
        time_span: The time span to calculate. Options: 'week', 'month', '3month', 'year'.

    Returns:
        A string representing the date in RFC 3339 format (e.g., '2023-01-01T00:00:00Z').
        Returns empty string if time_span is not recognized.
    """
    now = datetime.datetime.now(datetime.UTC)

    if time_span == "week":
        delta = datetime.timedelta(weeks=1)
    elif time_span == "month":
        delta = datetime.timedelta(days=30)
    elif time_span == "3month":
        delta = datetime.timedelta(days=90)
    elif time_span == "year":
        delta = datetime.timedelta(days=365)
    else:
        return ""

    past_date = now - delta
    return past_date.isoformat()


async def render_html(
    html_content: str, filename: str, tool_context: ToolContext
) -> str:
    """
    Saves HTML content to a file and optionally registers it as an artifact.

    Args:
        html_content: The raw HTML string.
        filename: The output filename.
        tool_context: The ADK tool context.

    Returns:
        The path to the saved file.
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, filename)

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Try to save artifact
        try:
            await tool_context.save_artifact(
                filename.replace(".", "_"),  # Simple artifact name
                types.Part.from_bytes(
                    data=html_content.encode("utf-8"), mime_type="text/html"
                ),
            )
            logger.info("Successfully saved HTML to tool context")
        except Exception as e:
            logger.warning(f"Warning: Failed to save artifact: {e}")

        return f"HTML saved to {filename}"
    except Exception as e:
        return f"Error saving HTML: {e!s}"


def parse_timestamp_to_seconds(timestamp_str: str) -> int:
    """
    Parses a timestamp string into an integer representing total seconds.

    Supports formats: raw seconds ('322'), MM:SS ('05:22'), or HH:MM:SS ('1:05:22').
    """
    try:
        # Match optional HH:, required MM:, required SS
        # Groups: 1=HH (optional), 2=MM (or raw seconds if no colon), 3=SS (optional)
        match = re.fullmatch(
            r"(?:(?:(\d+):)?(\d+):)?(\d+)", str(timestamp_str).strip()
        )
        if not match:
            return 0

        groups = match.groups()
        hh = groups[0]
        mm = groups[1]
        ss = groups[2]

        # If only the last group matched, it's just raw seconds
        if not hh and not mm:
            return int(ss)

        total = 0
        if hh:
            total += int(hh) * 3600
        if mm:
            total += int(mm) * 60
        if ss:
            total += int(ss)

        return total
    except Exception:
        pass
    return 0


# --- NEW METADATA/TEXT TOOLS FOR PR3 ---


def get_comment_replies(comment_id: str, tool_context: ToolContext, max_results: int = 50) -> list[str] | str:
    """
    Retrieves the replies to a specific top-level comment.
    Use this to dig deep into a specific community debate or controversy.

    Args:
        comment_id: The ID of the top-level comment (obtained from get_video_comments).
        tool_context: The ADK tool context.
        max_results: Maximum number of replies to return.

    Returns:
        A list of reply strings, or error message.
    """
    youtube, error = init_or_get_youtube_client(tool_context)
    if error:
        return error
    assert youtube is not None

    try:
        reply_response = (
            youtube.comments()
            .list(
                parentId=comment_id,
                part="snippet",
                maxResults=max_results,
                textFormat="plainText",
            )
            .execute()
        )

        replies = []
        for item in reply_response.get("items", []):
            replies.append(item["snippet"]["textDisplay"])
        return replies
    except HttpError as e:
        logger.error(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return f"ERROR: YouTube API returned {e.resp.status}"


def aggregate_comment_sentiment(
    video_ids: list[str], tool_context: ToolContext, comments_per_video: int = 10
) -> dict | str:
    """
    Fetches comments across multiple videos simultaneously to generate a macro-sentiment view.
    Use this for Product Launch Audits to see what the entire audience is saying across different reviews.

    Args:
        video_ids: A list of video IDs to fetch comments for.
        tool_context: The ADK tool context.
        comments_per_video: How many top comments to pull per video.

    Returns:
        A dictionary mapping videoId to a list of comment strings, or error message.
    """
    # We use the synchronous get_video_comments in a loop here.
    aggregated_data = {}
    for vid in video_ids:
        # Get top 'relevance' comments
        comments_data = get_video_comments(
            vid, tool_context=tool_context, max_results=comments_per_video, order="relevance"
        )
        # Handle cases where get_video_comments might return the error string
        if isinstance(comments_data, str) and "SYSTEM REJECTION" in comments_data:
            return comments_data
        aggregated_data[vid] = comments_data
    return aggregated_data


def search_channel_videos(
    channel_id: str, tool_context: ToolContext, max_results: int = 5, published_after: str = ""
) -> list[dict] | str:
    """
    Searches for the most recent videos uploaded by a specific channel.
    Use this for Industry Landscape Briefings to track specific 'analyst' or 'competitor' channels.

    Args:
        channel_id: The ID of the YouTube channel.
        tool_context: The ADK tool context.
        max_results: Maximum number of videos to return.
        published_after: Filter for videos published after this date (RFC 3339).

    Returns:
        A list of dictionaries containing video title, videoId, and publishedAt, or error message.
    """
    youtube, error = init_or_get_youtube_client(tool_context)
    if error:
        return error
    assert youtube is not None

    try:
        kwargs = {
            "channelId": channel_id,
            "type": "video",
            "part": "id,snippet",
            "order": "date",  # Always get the newest first
            "maxResults": max_results,
        }
        if published_after:
            kwargs["publishedAfter"] = published_after

        search_response = youtube.search().list(**kwargs).execute()

        results = []
        for search_result in search_response.get("items", []):
            if search_result["id"]["kind"] == "youtube#video":
                results.append(
                    {
                        "title": search_result["snippet"]["title"],
                        "videoId": search_result["id"]["videoId"],
                        "publishedAt": search_result["snippet"]["publishedAt"],
                        "description": search_result["snippet"]["description"],
                    }
                )
        return results
    except HttpError as e:
        logger.error(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return f"ERROR: YouTube API returned {e.resp.status}"


def get_video_transcript(video_id: str) -> str:
    """
    Retrieves the full transcript/captions for a specific YouTube video.
    Use this to "read" the video for the user to save them time, extract key arguments, or find exact timestamps.

    Args:
        video_id: The ID of the YouTube video.

    Returns:
        A string containing the transcript with timestamps, or an error message if captions are disabled.
    """
    try:
        # Fetch the transcript (prioritizes English, but falls back to others)
        transcript = YouTubeTranscriptApi.get_transcript(video_id) # type: ignore

        # Format it into a readable string with basic timestamps
        formatted_transcript = ""
        for entry in transcript:
            # Convert seconds to MM:SS
            start_val = entry["start"]
            text_val = entry["text"]

            start_time = int(start_val)
            minutes, seconds = divmod(start_time, 60)
            formatted_time = f"[{minutes:02d}:{seconds:02d}]"

            # Clean up newlines within single subtitle blocks
            clean_text = text_val.replace("\n", " ")
            formatted_transcript += f"{formatted_time} {clean_text}\n"

        return formatted_transcript
    except Exception as e:
        logger.warning(f"Failed to fetch transcript for {video_id}: {e}")
        return f"TRANSCRIPT_UNAVAILABLE: No captions found for video '{video_id}'. You MUST fall back to using 'get_video_details' to read the video description instead."


def submit_feedback(
    feedback_text: str,
    category: str = "general",
    tool_context: ToolContext | None = None,
) -> str:
    """
    Submits user feedback regarding the agent's performance, accuracy, or content relevance.
    Use this tool when the user explicitly provides feedback or corrects your behavior.

    Args:
        feedback_text: The user's feedback message.
        category: The type of feedback (e.g., 'accuracy', 'feature_request', 'general').
        tool_context: The ADK tool context.

    Returns:
        A confirmation message.
    """
    session_id = "unknown"
    user_id = "unknown"
    app_name = "unknown"
    if tool_context and tool_context.session:
        session_id = tool_context.session.id or "unknown"
        user_id = tool_context.session.user_id or "unknown"
        app_name = tool_context.session.app_name or "unknown"

    # Create a structured JSON payload for Google Cloud Logging
    feedback_payload = {
        "event_type": "USER_FEEDBACK",
        "category": category,
        "session_id": session_id,
        "user_id": user_id,
        "app_name": app_name,
        "feedback_text": feedback_text,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    # Log it as a WARNING so it stands out from standard INFO traffic
    # When deployed to Vertex, this JSON string will be parsed by Cloud Logging
    logger.warning(json.dumps(feedback_payload))

    return "Feedback successfully recorded. Thank you!"


def publish_file(
    content: str | bytes,
    filename: str,
    mime_type: str = "text/html",
    tool_context: ToolContext | None = None,
) -> str:
    """
    Publishes content to a temporary public URL using Google Cloud Storage.
    Use this tool when you have generated a comprehensive HTML report or image and need to
    provide the user with a direct link to view it in their browser.

    Args:
        content: The raw string or bytes content to publish.
        filename: The desired filename (e.g., 'report.html' or 'chart.png').
        mime_type: The MIME type of the file (default: 'text/html').
        tool_context: The ADK tool context.

    Returns:
        A string containing the public URL where the user can view the file, or an error message.
    """
    try:
        bucket_name = config.PUBLIC_ARTIFACT_BUCKET
        if not bucket_name:
            return "ERROR: PUBLIC_ARTIFACT_BUCKET is not configured. Please set this environment variable to enable public file publishing."

        public_url_prefix = f"https://storage.googleapis.com/{bucket_name}"

        # Determine path parameters
        now = datetime.datetime.now(datetime.UTC)
        date_str = now.strftime("%Y%m%d")

        # Extract session_id if available to group files
        session_folder = "no_session"
        if tool_context and tool_context.session:
            session_folder = tool_context.session.id or "no_session"

        random_str = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=6)
        )

        # New structure: exports/youtube-analyst/YYYYMMDD/session_id/random/filename
        path_suffix = f"exports/youtube-analyst/{date_str}/{session_folder}/{random_str}/{filename}"

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(path_suffix)

        # Upload the content
        if isinstance(content, str):
            blob.upload_from_string(content, content_type=mime_type)
        else:
            blob.upload_from_string(
                content, content_type=mime_type
            )  # works for bytes too

        logger.info(f"Successfully published file to {path_suffix}")

        return f"File successfully published. View it here: {public_url_prefix}/{path_suffix}"

    except Exception as e:
        logger.error(f"Failed to publish file: {e}")
        return f"Failed to publish file to public URL: {e!s}"
