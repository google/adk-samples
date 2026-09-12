# Copyright 2026 Google LLC
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

import asyncio
import base64
import datetime
import functools
import json
import logging
import os
import re
import uuid

from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.tools import ToolContext
from google.cloud import storage

logger = logging.getLogger(__name__)

load_dotenv()

# Use Vertex AI backend (ADC auth) instead of Gemini API (API key auth).
# Must be set before any ADK/GenAI client is created.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# When True, Jira is replaced by a lightweight GCS-based task system.
SKIP_JIRA = os.getenv("SKIP_JIRA", "false").lower() == "true"

# When True, Google Chat notifications are skipped.
SKIP_CHAT = os.getenv("SKIP_CHAT", "false").lower() == "true"


# GCS bucket used for task JSON files and video assets.
_TASK_BUCKET = os.getenv("ASSET_BUCKET_NAME", "")
_VIDEO_BUCKET = os.getenv("VEO_GCS_BUCKET", _TASK_BUCKET)
_TASK_PREFIX = "tasks/"


def _load_prompt(filename: str) -> str:
    """Load a prompt from the prompts directory."""
    return (_PROMPTS_DIR / filename).read_text()


# ---------------------------------------------------------------------------
# Shared async HTTP client (connection pooling + HTTP/2 support)
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Return a shared async HTTP client, creating it lazily."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
    return _http_client


async def close_http_client() -> None:
    """Close the shared HTTP client, releasing pooled connections.

    Should be called during application shutdown (e.g. in FastAPI lifespan).
    """
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


# ---------------------------------------------------------------------------
# GCS task helpers (used when SKIP_JIRA=true)
# ---------------------------------------------------------------------------


def _generate_task_id() -> str:
    """Generate a short unique task ID like TASK-A3F7B2C1."""
    return f"TASK-{uuid.uuid4().hex[:8].upper()}"


def _extract_video_urls(text: str) -> list[str]:
    """Extract URLs ending in common video extensions from text."""
    pattern = r'https?://[^\s<>"\']+\.(?:mp4|webm|mov|avi)'
    return re.findall(pattern, text, re.IGNORECASE)


def _get_gcs_client() -> storage.Client:
    """Return a GCS client."""
    return storage.Client()


def _upload_task_json(task_id: str, task_data: dict) -> None:
    """Upload a task JSON file to GCS."""
    client = _get_gcs_client()
    bucket = client.bucket(_TASK_BUCKET)
    blob = bucket.blob(f"{_TASK_PREFIX}{task_id}.json")
    blob.upload_from_string(
        json.dumps(task_data, indent=2),
        content_type="application/json",
    )
    logger.info("Uploaded task %s to gs://%s/%s%s.json", task_id, _TASK_BUCKET, _TASK_PREFIX, task_id)


def _download_task_json(task_id: str) -> dict | None:
    """Download a task JSON file from GCS. Returns None if not found."""
    client = _get_gcs_client()
    bucket = client.bucket(_TASK_BUCKET)
    blob = bucket.blob(f"{_TASK_PREFIX}{task_id}.json")
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


# ---------------------------------------------------------------------------
# Jira helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _jira_config() -> tuple[str | None, dict[str, str] | None]:
    """Return (base_url, headers) for Jira Cloud API, or (None, None).

    Cached after first call to avoid repeated env lookups and base64
    encoding on every tool invocation.
    """
    base_url = os.getenv("JIRA_BASE_URL")
    email = os.getenv("JIRA_USER_EMAIL")
    api_token = os.getenv("JIRA_API_TOKEN")
    if not all([base_url, email, api_token]):
        return None, None
    credentials = base64.b64encode(f"{email}:{api_token}".encode()).decode()
    headers = {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    return base_url.rstrip("/"), headers


def _jira_missing_config_error() -> str:
    return (
        "❌ Error: Jira credentials not configured. "
        "Set JIRA_BASE_URL, JIRA_USER_EMAIL, and JIRA_API_TOKEN."
    )


def _extract_description_text(description) -> str:
    """Extract plain text from Jira's Atlassian Document Format."""
    if not description:
        return "(no description)"
    if isinstance(description, str):
        return description
    # ADF format
    text_parts = []
    for block in description.get("content", []):
        for inline in block.get("content", []):
            if inline.get("type") == "text":
                text_parts.append(inline.get("text", ""))
    return "\n".join(text_parts) if text_parts else "(no description)"


# ---------------------------------------------------------------------------
# Tool: Send Google Chat Message
# ---------------------------------------------------------------------------


# Google Chat webhook text message limit (bytes). We use a conservative
# character limit to stay safely under the wire.
_CHAT_MESSAGE_MAX_CHARS = 4000


async def send_google_chat_message(message_text: str) -> str:
    """Sends a short notification message to the dev team via Google Chat.

    IMPORTANT: The message MUST be a concise summary (a few short paragraphs
    at most). Include only the key action items, task/ticket ID, and a
    one-line summary. Do NOT include full reports, video URLs, research data,
    or any large content — link to the task for details instead.

    Args:
        message_text: A short notification message. Supports Google Chat
            text formatting: *bold*, _italic_, ~strikethrough~,
            `inline code`, and links <URL|link text>.

    Returns:
        A confirmation message indicating whether the send was successful.
    """
    if SKIP_CHAT:
        return "ℹ️ Chat notification skipped because SKIP_CHAT is enabled."

    webhook_url = os.getenv("GOOGLE_CHAT_WEBHOOK_URL")
    if not webhook_url:
        return (
            "❌ Error: GOOGLE_CHAT_WEBHOOK_URL environment variable is not set. "
            "Please configure the webhook URL to send messages."
        )

    # Truncate to stay within Google Chat's message size limit.
    if len(message_text) > _CHAT_MESSAGE_MAX_CHARS:
        message_text = (
            message_text[: _CHAT_MESSAGE_MAX_CHARS - 100]
            + "\n\n⚠️ _Message truncated. See the task for full details._"
        )

    payload = {"text": message_text}
    client = _get_http_client()

    try:
        response = await client.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json; charset=UTF-8"},
        )
        response.raise_for_status()
        result = "✅ Message sent successfully to the Google Chat space."

        return result
    except httpx.HTTPStatusError as e:
        logger.exception("Google Chat webhook HTTP error")
        return f"❌ Failed to send message (HTTP {e.response.status_code}): {e}"
    except httpx.RequestError as e:
        logger.exception("Google Chat webhook request error")
        return f"❌ Failed to send message: {e}"


# ---------------------------------------------------------------------------
# Tool: Get Campaign Videos
# ---------------------------------------------------------------------------


def get_campaign_videos(
    task_id: str = "",
    tool_context: ToolContext | None = None,
) -> str:
    """Retrieves the campaign video URLs associated with a task.

    Looks up the task JSON file in GCS by task_id (or Jira ticket key)
    and returns the video URLs stored in it. If no task_id is provided,
    checks the session context for video URLs (e.g. from the product
    strategy agent). Falls back to listing video files directly from
    the GCS bucket.

    Args:
        task_id: Optional. The task ID (e.g. "TASK-a3f7b2c1") or Jira
            ticket key (e.g. "APPDEV-5") to look up videos for. If not
            provided, checks session context then lists from GCS bucket.

    Returns:
        A formatted list of video URLs, or a message if none are found.
    """
    task_id = task_id.strip().upper() if task_id else ""

    # Try to load from GCS task file first (only if a task_id was given)
    if task_id:
        task_data = _download_task_json(task_id)
    else:
        task_data = None

    # If no task_id provided, check session context for video URLs
    # (e.g. the product strategy agent may have stored them earlier)
    if not task_id and tool_context is not None:
        context_videos = tool_context.state.get("_campaign_video_urls")
        if context_videos and isinstance(context_videos, list):
            lines = ["🎬 **Campaign Videos (from session context):**\n"]
            for i, url in enumerate(context_videos, start=1):
                name = url.rsplit("/", 1)[-1].replace("%20", " ")
                name = name.rsplit(".", 1)[0] if "." in name else name
                lines.append(f"{i}. **{name}**\n   {url}")
            return "\n".join(lines)
    if task_data:
        video_urls = task_data.get("video_urls", [])
        if video_urls:
            lines = [f"🎬 **Campaign Videos for {task_id}:**\n"]
            for i, url in enumerate(video_urls, start=1):
                # Extract a readable name from the URL
                name = url.rsplit("/", 1)[-1].replace("%20", " ")
                # Remove file extension for display
                name = name.rsplit(".", 1)[0] if "." in name else name
                lines.append(f"{i}. **{name}**\n   {url}")
            return "\n".join(lines)
        else:
            return f"📋 Task **{task_id}** exists but has no video URLs associated with it."

    # Fallback: list the latest 3 video files from the GCS bucket directly.
    # This supports the Jira flow where no task JSON exists yet.
    if _VIDEO_BUCKET:
        try:
            client = _get_gcs_client()
            bucket = client.bucket(_VIDEO_BUCKET)
            blobs = list(bucket.list_blobs(prefix="videos/"))
            video_extensions = (".mp4", ".webm", ".mov", ".avi")
            video_blobs = [
                b for b in blobs
                if b.name.lower().endswith(video_extensions)
            ]
            # Sort by upload time (newest first) and take the latest 3
            video_blobs.sort(
                key=lambda b: b.time_created or datetime.datetime.min,
                reverse=True,
            )
            video_blobs = video_blobs[:3]
            if video_blobs:
                lines = ["🎬 **Campaign Videos (latest 3):**\n"]
                for i, blob in enumerate(video_blobs, start=1):
                    url = f"https://storage.googleapis.com/{_TASK_BUCKET}/{blob.name}"
                    name = blob.name.rsplit("/", 1)[-1].replace("%20", " ")
                    name = name.rsplit(".", 1)[0] if "." in name else name
                    lines.append(f"{i}. **{name}**\n   {url}")
                return "\n".join(lines)
        except Exception as e:
            logger.exception("Failed to list videos from GCS bucket")
            return f"❌ Failed to list videos from bucket: {e}"

    return (
        f"❌ Task **{task_id}** was not found and no video bucket is configured. "
        f"Please check the task ID and try again."
    )


# ---------------------------------------------------------------------------
# Tool: Create Jira Ticket (or GCS Task when SKIP_JIRA=true)
# ---------------------------------------------------------------------------


async def create_jira_ticket(
    summary: str,
    description: str,
    project_key: str = "APPDEV",
    issue_type: str = "Task",
    assignee_email: str = "",
    tool_context: ToolContext | None = None,
) -> str:
    """Creates a Jira ticket (or a local task when Jira is disabled).

    Use this tool to file engineering tasks, bug reports, or feature requests.
    Include all relevant context in the description so developers can get
    started immediately.

    IMPORTANT: Only ONE ticket/task should be created per user request. If you
    need to capture multiple requirements, combine them into a single ticket.

    Args:
        summary: A concise title for the ticket
            (e.g. "Build and deploy landing page for 'The Arden Collection'").
        description: A detailed description including requirements, video URLs,
            pricing updates, and any other context the developer needs.
        project_key: The Jira project key to file the ticket under.
            Defaults to "APPDEV".
        issue_type: The type of issue to create (e.g. "Task", "Story", "Bug").
            Defaults to "Task".
        assignee_email: Optional email address of the developer to assign
            the ticket to. Leave empty for unassigned.

    Returns:
        A confirmation message with the task/ticket ID.
    """
    # --- SKIP_JIRA: create a GCS-based task instead ---
    if SKIP_JIRA:
        logger.info("SKIP_JIRA is enabled — creating GCS-based task.")

        if not _TASK_BUCKET:
            return "❌ Error: ASSET_BUCKET_NAME is not configured. Cannot create task."

        task_id = _generate_task_id()
        video_urls = _extract_video_urls(description)
        today = datetime.date.today().isoformat()

        task_data = {
            "task_id": task_id,
            "summary": summary,
            "description": description,
            "video_urls": video_urls,
            "status": "Open",
            "issue_type": issue_type,
            "project_key": project_key,
            "assignee": assignee_email,
            "created": today,
            "start_date": None,
        }

        try:
            _upload_task_json(task_id, task_data)
        except Exception as e:
            logger.exception("Failed to upload task JSON to GCS")
            return f"❌ Failed to create task: {e}"

        if tool_context is not None:
            tool_context.state["_jira_ticket_created"] = task_id

        return (
            f"✅ **Task Created**\n\n"
            f"📋 **{task_id}**: {summary}\n\n"
            f"Status: Open | Type: {issue_type} | Project: {project_key}"
        )

    # --- Normal Jira flow ---

    # Guard: prevent multiple ticket creation in the same conversation turn.
    # (Primary guard is before_tool_callback; this is a secondary safety net.)
    if tool_context is not None:
        created_key = tool_context.state.get("_jira_ticket_created")
        if created_key and created_key != "__pending__":
            return (
                f"⚠️ A ticket was already created in this conversation: "
                f"{created_key}. Consolidate all requirements into a "
                f"single ticket instead of creating multiple tickets."
            )

    base_url, headers = _jira_config()
    if not base_url:
        return _jira_missing_config_error()

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type},
        }
    }

    if assignee_email:
        payload["fields"]["assignee"] = {"id": assignee_email}

    client = _get_http_client()

    try:
        response = await client.post(
            f"{base_url}/rest/api/2/issue",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

        result = response.json()
        ticket_key = result.get("key", "UNKNOWN")
        browse_url = f"{base_url}/browse/{ticket_key}"

        if tool_context is not None:
            tool_context.state["_jira_ticket_created"] = ticket_key

        result = (
            f"✅ **Jira Ticket Created**\n\n"
            f"📋 **{ticket_key}**: {summary}\n\n"
            f"🔗 Link: {browse_url}\n\n"
            f"Status: Open | Type: {issue_type} | Project: {project_key}"
        )

        return result

    except httpx.HTTPStatusError as e:
        error_body = ""
        try:
            error_body = e.response.json()
        except Exception:
            error_body = e.response.text
        logger.exception("Jira API HTTP error: %s", error_body)
        return f"❌ Failed to create Jira ticket (HTTP {e.response.status_code}): {error_body}"
    except httpx.RequestError as e:
        logger.exception("Jira API request error")
        return f"❌ Failed to create Jira ticket: {e}"


# ---------------------------------------------------------------------------
# Tool: Get Jira Ticket (or GCS Task when SKIP_JIRA=true)
# ---------------------------------------------------------------------------


async def get_jira_ticket(ticket_key: str) -> str:
    """Retrieves details for a task or Jira ticket and formats them nicely.

    Use this tool when the user wants to know about a specific task/ticket,
    e.g. "Tell me about TASK-a3f7b2c1" or "What is APPDEV-5".

    Args:
        ticket_key: The task ID (e.g. "TASK-a3f7b2c1") or Jira issue key
            (e.g. "APPDEV-5").

    Returns:
        A formatted summary of the task/ticket details.
    """
    ticket_key = ticket_key.upper()

    # --- SKIP_JIRA: read from GCS ---
    if SKIP_JIRA:
        logger.info("SKIP_JIRA is enabled — fetching task %s from GCS.", ticket_key)

        if not _TASK_BUCKET:
            return "❌ Error: ASSET_BUCKET_NAME is not configured. Cannot fetch task."

        task_data = _download_task_json(ticket_key)
        if not task_data:
            return f"❌ Task **{ticket_key}** was not found."

        video_section = ""
        video_urls = task_data.get("video_urls", [])
        if video_urls:
            video_lines = []
            for i, url in enumerate(video_urls, start=1):
                name = url.rsplit("/", 1)[-1].replace("%20", " ").rsplit(".", 1)[0]
                video_lines.append(f"  {i}. {name}: {url}")
            video_section = "\n**Videos:**\n" + "\n".join(video_lines)

        return (
            f"📋 **{task_data['task_id']}**: {task_data['summary']}\n\n"
            f"**Status:** {task_data.get('status', 'N/A')}\n"
            f"**Type:** {task_data.get('issue_type', 'N/A')}\n"
            f"**Project:** {task_data.get('project_key', 'N/A')}\n"
            f"**Assignee:** {task_data.get('assignee') or 'Unassigned'}\n"
            f"**Created:** {task_data.get('created', 'N/A')}\n"
            f"**Start Date:** {task_data.get('start_date') or 'Not started'}\n\n"
            f"**Description:**\n{task_data.get('description', '(no description)')}"
            f"{video_section}"
        )

    # --- Normal Jira flow ---
    base_url, headers = _jira_config()
    if not base_url:
        return _jira_missing_config_error()

    client = _get_http_client()

    try:
        response = await client.get(
            f"{base_url}/rest/api/2/issue/{ticket_key}",
            headers=headers,
        )
        response.raise_for_status()
        issue = response.json()

        fields = issue.get("fields", {})
        summary = fields.get("summary", "N/A")
        status = fields.get("status", {}).get("name", "N/A")
        issue_type = fields.get("issuetype", {}).get("name", "N/A")
        priority = fields.get("priority", {}).get("name", "N/A")
        assignee = fields.get("assignee")
        assignee_name = assignee.get("displayName", "N/A") if assignee else "Unassigned"
        project = fields.get("project", {}).get("key", "N/A")
        created = fields.get("created", "N/A")[:10]
        description = _extract_description_text(fields.get("description"))
        labels = ", ".join(fields.get("labels", [])) or "None"
        browse_url = f"{base_url}/browse/{ticket_key}"

        return (
            f"📋 **{ticket_key}**: {summary}\n\n"
            f"🔗 Link: {browse_url}\n\n"
            f"**Status:** {status}\n"
            f"**Type:** {issue_type}\n"
            f"**Priority:** {priority}\n"
            f"**Assignee:** {assignee_name}\n"
            f"**Project:** {project}\n"
            f"**Labels:** {labels}\n"
            f"**Created:** {created}\n\n"
            f"**Description:**\n{description}"
        )

    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        if status_code == 404:
            return f"❌ Ticket **{ticket_key}** was not found."
        error_body = ""
        try:
            error_body = e.response.json()
        except Exception:
            error_body = e.response.text
        logger.exception("Jira API error fetching %s: %s", ticket_key, error_body)
        return f"❌ Failed to fetch ticket (HTTP {status_code}): {error_body}"
    except httpx.RequestError as e:
        logger.exception("Jira API request error")
        return f"❌ Failed to fetch ticket: {e}"


# ---------------------------------------------------------------------------
# Tool: Start Working on Jira Ticket (or GCS Task when SKIP_JIRA=true)
# ---------------------------------------------------------------------------


async def start_jira_ticket(ticket_key: str) -> str:
    """Starts work on a task/ticket: transitions it to 'In Progress',
    assigns it to the current user, and sets the start date to today.

    Use this tool ONLY when the user explicitly says THEY PERSONALLY want
    to work on a specific task, e.g. "let me work on TASK-a3f7b2c1" or
    "I'll take APPDEV-2026" or "assign APPDEV-5 to me".

    Do NOT use this tool when the user is delegating or notifying others
    (e.g. "tell the dev team to get started", "send a chat about this").
    Delegation uses create_jira_ticket + send_google_chat_message instead.

    Args:
        ticket_key: The task ID (e.g. "TASK-a3f7b2c1") or Jira issue key
            (e.g. "APPDEV-5").

    Returns:
        A confirmation message with the updated task/ticket details.
    """
    ticket_key = ticket_key.upper()
    today = datetime.date.today().isoformat()
    user_email = os.getenv("JIRA_USER_EMAIL", "developer@example.com")

    # --- SKIP_JIRA: update GCS task ---
    if SKIP_JIRA:
        logger.info("SKIP_JIRA is enabled — starting task %s via GCS.", ticket_key)

        if not _TASK_BUCKET:
            return "❌ Error: ASSET_BUCKET_NAME is not configured. Cannot update task."

        task_data = _download_task_json(ticket_key)
        if not task_data:
            return f"❌ Task **{ticket_key}** was not found."

        task_data["status"] = "In Progress"
        task_data["assignee"] = user_email
        task_data["start_date"] = today

        try:
            _upload_task_json(ticket_key, task_data)
        except Exception as e:
            logger.exception("Failed to update task JSON in GCS")
            return f"❌ Failed to update task: {e}"

        return (
            f"✅ **Task Updated**\n\n"
            f"📋 **{ticket_key}** is now **In Progress**\n\n"
            f"**Assignee:** {user_email}\n"
            f"**Start Date:** {today}\n"
            f"**Status:** In Progress"
        )

    # --- Normal Jira flow ---
    base_url, headers = _jira_config()
    if not base_url:
        return _jira_missing_config_error()

    client = _get_http_client()
    errors = []

    # --- Phase 1: Fetch transitions and user account ID in parallel ---
    async def _fetch_transitions() -> str | None:
        """Get the 'In Progress' transition ID, or None."""
        resp = await client.get(
            f"{base_url}/rest/api/2/issue/{ticket_key}/transitions",
            headers=headers,
        )
        resp.raise_for_status()
        transitions = resp.json().get("transitions", [])
        for t in transitions:
            if t["name"].lower() in ("in progress", "start progress"):
                return t["id"]
        available = [t["name"] for t in transitions]
        errors.append(
            f"⚠️ Could not find 'In Progress' transition. "
            f"Available: {available}"
        )
        return None

    async def _lookup_user() -> str | None:
        """Look up the user's Jira account ID by email."""
        resp = await client.get(
            f"{base_url}/rest/api/2/user/search",
            params={"query": user_email},
            headers=headers,
        )
        resp.raise_for_status()
        users = resp.json()
        if users:
            return users[0].get("accountId")
        return None

    # Run both lookups concurrently — saves one full round-trip.
    results = await asyncio.gather(
        _fetch_transitions(),
        _lookup_user(),
        return_exceptions=True,
    )
    in_progress_id = results[0] if not isinstance(results[0], BaseException) else None
    account_id = results[1] if not isinstance(results[1], BaseException) else None

    for res in results:
        if isinstance(res, BaseException):
            if isinstance(res, httpx.HTTPStatusError):
                logger.exception("Parallel lookup error for %s", ticket_key)
                errors.append(f"⚠️ Lookup failed (HTTP {res.response.status_code}): {res}")
            else:
                logger.exception("Parallel lookup request error for %s", ticket_key)
                errors.append(f"⚠️ Lookup failed: {res}")

    # --- Phase 2: Apply transition and update fields ---
    async def _apply_transition():
        if not in_progress_id:
            return
        resp = await client.post(
            f"{base_url}/rest/api/2/issue/{ticket_key}/transitions",
            json={"transition": {"id": in_progress_id}},
            headers=headers,
        )
        resp.raise_for_status()

    async def _update_fields():
        update_fields = {}
        if account_id:
            update_fields["assignee"] = {"accountId": account_id}
        # customfield_10015 is the common start date field in Jira Cloud
        update_fields["customfield_10015"] = today
        resp = await client.put(
            f"{base_url}/rest/api/2/issue/{ticket_key}",
            json={"fields": update_fields},
            headers=headers,
        )
        resp.raise_for_status()

    phase2_results = await asyncio.gather(
        _apply_transition(),
        _update_fields(),
        return_exceptions=True,
    )

    for res, label in zip(phase2_results, ["Transition", "Field update"]):
        if isinstance(res, BaseException):
            if isinstance(res, httpx.HTTPStatusError):
                error_body = ""
                try:
                    error_body = res.response.json()
                except Exception:
                    error_body = res.response.text
                logger.exception("%s error for %s: %s", label, ticket_key, error_body)
                errors.append(f"⚠️ {label} failed: {error_body}")
            else:
                logger.exception("%s request error for %s", label, ticket_key)
                errors.append(f"⚠️ {label} failed: {res}")

    result = (
        f"✅ **Ticket Updated**\n\n"
        f"📋 **{ticket_key}** is now **In Progress**\n\n"
        f"🔗 Link: {base_url}/browse/{ticket_key}\n\n"
        f"**Assignee:** {user_email}\n"
        f"**Start Date:** {today}\n"
        f"**Status:** In Progress"
    )

    if errors:
        result += "\n\n**Warnings:**\n" + "\n".join(errors)

    return result


# ---------------------------------------------------------------------------
# Agent & App
# ---------------------------------------------------------------------------


def _before_tool_callback(
    tool: Any, args: dict[str, Any], tool_context: ToolContext
) -> Optional[dict[str, Any]]:
    """ADK-native guard that intercepts tool calls before dispatch.

    Session-level dedup: catches intra-session parallel tool scheduling.
    """
    if tool.name == "create_jira_ticket":
        existing = tool_context.state.get("_jira_ticket_created")
        if existing:  # catches both "__pending__" and a real ticket key
            return {
                "result": (
                    f"⚠️ A task/ticket was already created this session "
                    f"({existing}). Skipping duplicate — use the existing one."
                )
            }
        # Mark as pending immediately so any rapid second call is also blocked
        # before the first has finished.
        tool_context.state["_jira_ticket_created"] = "__pending__"

    if tool.name == "send_google_chat_message":
        if tool_context.state.get("_chat_message_sent"):
            return {
                "result": (
                    "⚠️ A Google Chat notification was already sent this session. "
                    "Skipping duplicate."
                )
            }
        tool_context.state["_chat_message_sent"] = True

    return None


root_agent = Agent(
    model=os.getenv("ADK_MODEL", "gemini-3-flash-preview"),
    name="dev_agent",
    description=(
        "A developer assistant agent for FS Direct that bridges "
        "business strategy and IT. Communicates with dev teams via Google Chat, "
        "and creates/manages engineering tasks for frictionless digital campaigns."
    ),
    instruction=_load_prompt("system_prompt.md"),
    before_tool_callback=_before_tool_callback,
    tools=[
        send_google_chat_message,
        get_campaign_videos,
        create_jira_ticket,
        get_jira_ticket,
        start_jira_ticket,
    ],
)
# Agent Engine / Gemini Enterprise expects a `version` attribute on the agent.
# BaseAgent uses extra='forbid', so bypass Pydantic validation.
object.__setattr__(root_agent, "version", "0.1.0")

app = App(
    name="app",
    root_agent=root_agent,
)
