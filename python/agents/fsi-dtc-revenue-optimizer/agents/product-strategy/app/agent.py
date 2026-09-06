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
import datetime
import logging
import os
import pathlib
import re
import uuid
from urllib.parse import urlparse, quote

import google.auth
import google.auth.transport.requests
from google.cloud import storage as gcs_storage
from googleapiclient.discovery import build as google_api_build

from app.a2ui_schema import A2UI_SCHEMA, VIDEO_A2UI_EXAMPLE
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools import LongRunningFunctionTool, ToolContext
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

load_dotenv(override=True)


def load_prompt(prompt_name: str) -> str:
    prompt_path = pathlib.Path(__file__).parent / "prompts" / prompt_name
    try:
        return prompt_path.read_text()
    except FileNotFoundError:
        logger.warning("Prompt file %s not found.", prompt_name)
        return ""


system_prompt = load_prompt("system_prompt.md")
system_instructions = load_prompt("system_instructions.md")

model_id = os.getenv("ADK_MODEL", "gemini-3-flash-preview")


def _descriptive_filename(source: str) -> str:
    """Derive a human-friendly filename from a GCS URI or prompt string."""
    # Strip path / extension, keep just the descriptive part
    name = os.path.splitext(os.path.basename(source))[0]
    # Replace non-alphanumeric chars with underscores, collapse multiples
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    # Truncate to keep it reasonable
    return name[:60] + ".mp4" if name else f"campaign_video_{uuid.uuid4().hex[:6]}.mp4"


def _gcs_uri_to_public_url(gcs_uri: str) -> str:
    """Converts a gs:// URI to a public HTTPS URL."""
    parsed = urlparse(gcs_uri)
    bucket_name = parsed.netloc
    blob_path = parsed.path.lstrip("/")
    return f"https://storage.googleapis.com/{bucket_name}/{quote(blob_path, safe='/')}"


# Default signed-URL lifetime; override with SIGNED_URL_EXPIRATION_MINUTES env var.
_SIGNED_URL_EXPIRATION_MINUTES = int(
    os.environ.get("SIGNED_URL_EXPIRATION_MINUTES", "60")
)


def _generate_signed_url(
    bucket_name: str,
    blob_name: str,
    expiration_minutes: int = _SIGNED_URL_EXPIRATION_MINUTES,
) -> str:
    """Generate a V4 signed URL for a private GCS object.

    This allows Gemini Enterprise (or any browser) to fetch the video
    without the bucket or object needing to be publicly accessible.

    Args:
        bucket_name: The GCS bucket name.
        blob_name: The blob path within the bucket.
        expiration_minutes: How long the URL is valid (default: 60 minutes).

    Returns:
        A signed HTTPS URL that provides temporary authenticated access.
    """
    from google.auth import compute_engine
    from google.auth.transport import requests as auth_requests

    credentials, project = google.auth.default()

    # On Cloud Run the default credentials are Compute Engine metadata-server
    # tokens which cannot sign blobs directly.  We pass the service account
    # email and a fresh access token so the google-cloud-storage library
    # calls the IAM signBlob API instead of requiring a local private key.
    # The service account needs iam.serviceAccounts.signBlob on itself
    # (granted by roles/iam.serviceAccountTokenCreator).
    if isinstance(credentials, compute_engine.Credentials):
        credentials.refresh(auth_requests.Request())
        storage_client = gcs_storage.Client(credentials=credentials, project=project)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=expiration_minutes),
            method="GET",
            service_account_email=credentials.service_account_email,
            access_token=credentials.token,
        )
    else:
        # Local dev with a service-account JSON key or user credentials
        # that can sign natively.
        storage_client = gcs_storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=expiration_minutes),
            method="GET",
        )

    logger.info(
        "Generated signed URL for gs://%s/%s (expires in %d min)",
        bucket_name,
        blob_name,
        expiration_minutes,
    )
    return url


def _copy_to_asset_bucket(source_uri: str, dest_bucket_name: str, dest_blob_name: str) -> str:
    """Copy a video from Veo's GCS location to the asset bucket.

    Args:
        source_uri: The gs:// URI returned by Veo (e.g., gs://veo-bucket/path/video.mp4).
        dest_bucket_name: The destination asset bucket name (VEO_GCS_BUCKET).
        dest_blob_name: The destination blob name (e.g., videos/organic_modern_living_room.mp4).

    Returns:
        A V4 signed URL providing temporary authenticated access to the
        copied video in the asset bucket.
    """
    parsed = urlparse(source_uri)
    src_bucket_name = parsed.netloc
    src_blob_path = parsed.path.lstrip("/")

    storage_client = gcs_storage.Client()
    src_bucket = storage_client.bucket(src_bucket_name)
    dst_bucket = storage_client.bucket(dest_bucket_name)
    src_blob = src_bucket.blob(src_blob_path)

    src_bucket.copy_blob(src_blob, dst_bucket, dest_blob_name)
    logger.info("Copied video: gs://%s/%s -> gs://%s/%s",
                src_bucket_name, src_blob_path, dest_bucket_name, dest_blob_name)

    return _generate_signed_url(dest_bucket_name, dest_blob_name)


def _upload_bytes_to_bucket(
    video_bytes: bytes,
    dest_bucket_name: str,
    dest_blob_name: str,
    content_type: str = "video/mp4",
) -> str:
    """Upload raw video bytes to the asset bucket.

    Args:
        video_bytes: The raw video content returned by Veo.
        dest_bucket_name: The destination asset bucket name.
        dest_blob_name: The destination blob name (e.g., videos/campaign.mp4).
        content_type: MIME type of the video (default: video/mp4).

    Returns:
        A V4 signed URL providing temporary authenticated access to the
        uploaded video.
    """
    storage_client = gcs_storage.Client()
    bucket = storage_client.bucket(dest_bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_string(video_bytes, content_type=content_type)
    logger.info("Uploaded video bytes to gs://%s/%s (%d bytes)",
                dest_bucket_name, dest_blob_name, len(video_bytes))
    return _generate_signed_url(dest_bucket_name, dest_blob_name)


async def _generate_single_video(
    client: genai.Client,
    prompt: str,
    video_index: int,
    asset_bucket: str,
) -> dict:
    """Generate a single video via Veo 3, copy it to the asset bucket, and return metadata.

    Args:
        client: The genai Client configured for Vertex AI.
        prompt: The creative brief for this specific video.
        video_index: 1-based index for logging and naming.
        asset_bucket: The GCS bucket name to store the final video.

    Returns:
        A dict with filename, url, and status for this video.
    """
    try:
        operation = client.models.generate_videos(
            model="veo-3.0-generate-001",
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                number_of_videos=1,
            ),
        )
        logger.info("Video %d — Veo 3 generation started: %s", video_index, operation.name)

        # Poll until completed (up to ~5 minutes)
        for _ in range(30):
            await asyncio.sleep(10)
            operation = client.operations.get(operation)
            if operation.done:
                break

        if not operation.done:
            logger.warning("Video %d — still running after 5 minutes: %s", video_index, operation.name)
            return {"status": "error", "detail": f"Video {video_index} timed out after 5 minutes."}

        if not operation.response or not operation.response.generated_videos:
            logger.error("Video %d — no generated_videos in response. Full response: %s",
                         video_index, operation.response)
            return {"status": "error", "detail": f"Video {video_index} completed but no video was returned."}

        generated_video = operation.response.generated_videos[0]
        logger.info("Video %d — generated_video type: %s, attrs: %s",
                     video_index, type(generated_video).__name__,
                     [a for a in dir(generated_video) if not a.startswith('_')])

        # The Veo API now returns video_bytes directly instead of a GCS URI.
        # We need to upload the bytes to our asset bucket ourselves.
        video_obj = generated_video.video
        if not video_obj:
            logger.error("Video %d — no video object in generated_video: %s",
                         video_index, generated_video)
            return {"status": "error", "detail": f"Video {video_index} completed but video data is missing."}

        # Copy from Veo's internal GCS to the asset bucket
        filename = _descriptive_filename(prompt)
        dest_blob = f"videos/{filename}"

        if video_obj.uri:
            # Legacy path: Veo returned a GCS URI — copy blob to asset bucket
            logger.info("Video %d — using GCS URI path: %s", video_index, video_obj.uri)
            loop = asyncio.get_event_loop()
            public_url = await loop.run_in_executor(
                None, _copy_to_asset_bucket, video_obj.uri, asset_bucket, dest_blob
            )
        elif video_obj.video_bytes:
            # New path: Veo returned raw bytes — upload directly to asset bucket
            logger.info("Video %d — using video_bytes path (%d bytes)",
                         video_index, len(video_obj.video_bytes))
            loop = asyncio.get_event_loop()
            public_url = await loop.run_in_executor(
                None, _upload_bytes_to_bucket, video_obj.video_bytes,
                asset_bucket, dest_blob, video_obj.mime_type or "video/mp4",
            )
        else:
            logger.error("Video %d — neither uri nor video_bytes available: %s",
                         video_index, generated_video)
            return {"status": "error", "detail": f"Video {video_index} completed but video data is missing."}

        logger.info("Video %d — found video URI: %s", video_index, public_url)

        logger.info("Video %d — saved to asset bucket: %s", video_index, public_url)
        return {"status": "success", "filename": filename, "url": public_url}

    except Exception as e:
        logger.exception("Video %d — generation error", video_index)
        return {"status": "error", "detail": f"Video {video_index} error: {e}"}

# Use setdefault so .env values are respected; only fill in if not already set.
_, _default_project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", _default_project_id or "")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


def request_user_input(message: str) -> dict:
    """Request additional input from the user.

    Use this tool when you need more information from the user to complete a task.
    Calling this tool will pause execution until the user responds.

    Args:
        message: The question or clarification request to show the user.
    """
    return {"status": "pending", "message": message}


async def generate_product_video(prompts: list[str], tool_context: ToolContext) -> dict:
    """Generates up to 3 product marketing videos in parallel using Google Veo 3.

    Use this tool to create campaign-ready video content for the website.
    Provide a list of 1–3 creative briefs, each describing a DIFFERENT video
    scene — vary the furniture piece, room setting, and lifestyle situation
    across prompts. All videos are generated simultaneously for speed.

    Each generated video is stored in the project's asset bucket and the
    public URLs are returned for rendering via A2UI Video components.

    This is a long-running operation that may take 1-3 minutes total
    (all videos generate in parallel).

    Args:
        prompts: A list of 1–3 creative briefs. Each prompt should describe
            a distinct cinematic product scene — different furniture, room,
            and situation. Be specific about materials, lighting, camera
            movement, and mood.
        tool_context: ADK tool context (unused — videos are returned as URLs).

    Returns:
        A dict with status, detail, and a list of video URLs.
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    asset_bucket = os.environ.get("VEO_GCS_BUCKET")
    if not asset_bucket:
        return {"status": "error", "detail": "VEO_GCS_BUCKET environment variable is not set."}

    # Limit to 3 prompts maximum
    video_prompts = prompts[:3]
    logger.info("Generating %d campaign videos in parallel...", len(video_prompts))

    try:
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location="us-central1",  # Veo availability region
        )

        # Launch all video generations in parallel
        tasks = [
            _generate_single_video(client, prompt, i + 1, asset_bucket)
            for i, prompt in enumerate(video_prompts)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful videos and errors
        saved_videos = []
        errors = []
        for i, result in enumerate(results, start=1):
            if isinstance(result, Exception):
                errors.append(f"Video {i}: {result}")
                logger.error("Video %d raised exception: %s", i, result)
            elif result.get("status") == "success":
                saved_videos.append({
                    "filename": result["filename"],
                    "url": result["url"],
                })
            else:
                errors.append(result.get("detail", f"Video {i} failed."))

        if not saved_videos:
            return {
                "status": "error",
                "detail": f"All video generations failed: {'; '.join(errors)}",
            }

        detail = (
            f"{len(saved_videos)} campaign video(s) generated and saved to asset bucket. "
            "Use the video URLs in your A2UI JSON response "
            "with the Video component to render them inline."
        )
        if errors:
            detail += f" ({len(errors)} video(s) failed: {'; '.join(errors)})"

        return {
            "status": "success",
            "detail": detail,
            "videos": saved_videos,
        }

    except Exception as e:
        logger.exception("Veo 3 video generation error")
        return {"status": "error", "detail": f"Error generating videos: {e}"}


def _parse_markdown_to_doc_requests(text: str) -> list[dict]:
    """Converts markdown text to Google Docs API batchUpdate requests.

    Produces a list of requests that, when applied in order, insert the full
    report text into a blank Google Doc and apply heading / bold styles.
    The Docs API requires inserts to happen in reverse order (bottom-up) when
    building content sequentially, but it's simpler to insert all text first
    and then apply styles based on discovered offsets.
    """
    # --- Pass 1: build plain text and track style ranges ----------------
    lines = text.split("\n")
    plain_lines: list[str] = []
    heading_ranges: list[tuple[int, int, str]] = []  # (start, end, style)
    bold_ranges: list[tuple[int, int]] = []

    offset = 1  # Google Docs body starts at index 1
    for line in lines:
        # Detect markdown headings
        heading_style = None
        clean_line = line
        if line.startswith("#### "):
            heading_style = "HEADING_4"
            clean_line = line[5:]
        elif line.startswith("### "):
            heading_style = "HEADING_3"
            clean_line = line[4:]
        elif line.startswith("## "):
            heading_style = "HEADING_2"
            clean_line = line[3:]
        elif line.startswith("# "):
            heading_style = "HEADING_1"
            clean_line = line[2:]

        # Strip inline bold markers and remember their positions
        processed = ""
        i = 0
        while i < len(clean_line):
            if clean_line[i : i + 2] == "**":
                # Find closing **
                end = clean_line.find("**", i + 2)
                if end != -1:
                    bold_start = offset + len(processed)
                    inner = clean_line[i + 2 : end]
                    processed += inner
                    bold_end = offset + len(processed)
                    bold_ranges.append((bold_start, bold_end))
                    i = end + 2
                    continue
            processed += clean_line[i]
            i += 1

        line_text = processed + "\n"
        if heading_style:
            heading_ranges.append(
                (offset, offset + len(line_text) - 1, heading_style)
            )

        plain_lines.append(line_text)
        offset += len(line_text)

    full_text = "".join(plain_lines)

    # --- Pass 2: build Docs API requests --------------------------------
    requests: list[dict] = []

    # 1. Insert all text at once
    requests.append(
        {"insertText": {"location": {"index": 1}, "text": full_text}}
    )

    # 2. Apply heading styles
    for start, end, style in heading_ranges:
        requests.append(
            {
                "updateParagraphStyle": {
                    "range": {"startIndex": start, "endIndex": end},
                    "paragraphStyle": {"namedStyleType": style},
                    "fields": "namedStyleType",
                }
            }
        )

    # 3. Apply bold styles
    for start, end in bold_ranges:
        requests.append(
            {
                "updateTextStyle": {
                    "range": {"startIndex": start, "endIndex": end},
                    "textStyle": {"bold": True},
                    "fields": "bold",
                }
            }
        )

    return requests


def export_report_to_google_doc(
    report_content: str,
    tool_context: ToolContext,
) -> dict:
    """Creates a Google Doc with the strategic report and saves it to a shared
    Google Drive folder.

    This tool is called automatically after every strategic report to export
    the report as a richly formatted Google Doc. The doc is named with the
    current week number (e.g., "Strategic Report - 2026-W15"). If a doc for
    the same week already exists in the folder, it is replaced.

    Call this tool EXACTLY ONCE per request, immediately after completing the
    strategic report. Do not call it again even if an earlier attempt appears
    to have failed — the tool handles retries internally.

    Args:
        report_content: The full markdown content of the strategic report
            (all sections: Executive Summary, Data Analysis, Strategic
            Recommendations, Next-Best-Actions).
        tool_context: ADK tool context used to prevent duplicate exports.

    Returns:
        A dict with status, the Google Doc URL, and the document title.
    """
    # --- Fix 3: Duplicate guard — only one export per agent session -------
    if tool_context.state.get("doc_exported"):
        logger.info("export_report_to_google_doc: already exported this session, skipping.")
        return {
            "status": "skipped",
            "detail": "Report was already exported to Google Drive this session.",
            "url": tool_context.state.get("doc_url", ""),
        }

    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        return {
            "status": "error",
            "detail": "GOOGLE_DRIVE_FOLDER_ID environment variable is not set.",
        }

    try:
        # Authenticate via ADC (service account on Cloud Run, gcloud auth locally).
        credentials, _ = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/drive",
                "https://www.googleapis.com/auth/documents",
            ]
        )
        # Ensure the credentials are fresh.
        credentials.refresh(google.auth.transport.requests.Request())

        drive_service = google_api_build(
            "drive", "v3", credentials=credentials
        )
        docs_service = google_api_build(
            "docs", "v1", credentials=credentials
        )

        # Fix 1: Always use a stable, code-generated title; never rely on
        # the LLM to provide a consistent name across retries.
        today = datetime.date.today()
        iso_year, iso_week, _ = today.isocalendar()
        week_label = f"{iso_year}-W{iso_week:02d}"
        doc_title = f"Strategic Report - {week_label}"

        # Search for an existing doc from the same week, scoped strictly to
        # the target Shared Drive (corpora="drive" + driveId) to avoid
        # picking up stale IDs from other drives the service account can see.
        escaped_week = week_label.replace("'", "\\'")
        query = (
            f"name contains '{escaped_week}' "
            f"and '{folder_id}' in parents "
            f"and mimeType = 'application/vnd.google-apps.document' "
            f"and trashed = false"
        )
        try:
            existing_files = (
                drive_service.files()
                .list(
                    q=query,
                    corpora="drive",
                    driveId=folder_id,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    fields="files(id, name, webViewLink)",
                )
                .execute()
                .get("files", [])
            )
        except Exception as list_err:
            logger.warning("Could not search for existing docs: %s", list_err)
            existing_files = []

        doc_requests = _parse_markdown_to_doc_requests(report_content)

        if existing_files:
            # --- Upsert path: reuse the existing doc (stable URL) ----------
            existing = existing_files[0]
            doc_id = existing["id"]
            doc_url = existing.get(
                "webViewLink",
                f"https://docs.google.com/document/d/{doc_id}",
            )
            logger.info(
                "Found existing doc '%s' (%s) — overwriting content.",
                existing["name"], doc_id,
            )

            # Rename the doc to the canonical title if it differs.
            try:
                drive_service.files().update(
                    fileId=doc_id,
                    body={"name": doc_title},
                    supportsAllDrives=True,
                    fields="id",
                ).execute()
            except Exception as rename_err:
                logger.warning("Could not rename existing doc: %s", rename_err)

            # Clear existing body content, then insert fresh content.
            try:
                existing_doc = docs_service.documents().get(
                    documentId=doc_id
                ).execute()
                body_end = existing_doc.get("body", {}).get("content", [{}])[-1].get("endIndex", 1)
                clear_and_insert = []
                if body_end > 2:  # doc has content beyond the final newline
                    clear_and_insert.append({
                        "deleteContentRange": {
                            "range": {"startIndex": 1, "endIndex": body_end - 1}
                        }
                    })
                clear_and_insert.extend(doc_requests)
                if clear_and_insert:
                    docs_service.documents().batchUpdate(
                        documentId=doc_id,
                        body={"requests": clear_and_insert},
                    ).execute()
            except Exception as update_err:
                logger.exception("Could not overwrite doc content: %s", update_err)
                return {
                    "status": "error",
                    "detail": f"Found existing doc but could not update content: {update_err}",
                }

        else:
            # --- Create path: no existing doc this week --------------------
            try:
                file_metadata = {
                    "name": doc_title,
                    "mimeType": "application/vnd.google-apps.document",
                    "parents": [folder_id],
                }
                doc = (
                    drive_service.files()
                    .create(
                        body=file_metadata,
                        supportsAllDrives=True,
                        fields="id, webViewLink",
                    )
                    .execute()
                )
                doc_id = doc["id"]
                doc_url = doc.get(
                    "webViewLink",
                    f"https://docs.google.com/document/d/{doc_id}",
                )
            except Exception as create_err:
                logger.exception("Could not create new Google Doc")
                return {
                    "status": "error",
                    "detail": f"Could not create Google Doc: {create_err}",
                }

            try:
                if doc_requests:
                    docs_service.documents().batchUpdate(
                        documentId=doc_id,
                        body={"requests": doc_requests},
                    ).execute()
            except Exception as content_err:
                logger.exception("Could not write content to new doc")
                # Doc was created but content failed — still return the URL.
                return {
                    "status": "partial",
                    "url": doc_url,
                    "title": doc_title,
                    "detail": (
                        f"Google Doc created but content could not be written: {content_err}. "
                        f"View empty doc here: {doc_url}"
                    ),
                }

        logger.info("Exported Google Doc: %s -> %s", doc_title, doc_url)

        # Mark session as exported so duplicate calls are no-ops.
        tool_context.state["doc_exported"] = True
        tool_context.state["doc_url"] = doc_url

        return {
            "status": "success",
            "url": doc_url,
            "title": doc_title,
            "detail": (
                f"Strategic report saved as Google Doc: '{doc_title}'. "
                f"View it here: {doc_url}"
            ),
        }

    except Exception as e:
        logger.exception("Error creating Google Doc")
        return {"status": "error", "detail": f"Error creating Google Doc: {e}"}


# ---------------------------------------------------------------------------
# Inject A2UI output instructions into the agent's system prompt.
# This tells the LLM to emit A2UI JSON with Video components after its
# conversational text response, separated by the ---a2ui_JSON--- delimiter.
# See https://a2ui.org/guides/agent-development/ for the pattern.
# ---------------------------------------------------------------------------
_A2UI_INSTRUCTIONS = f"""

---

## A2UI Rich UI Output

**IMPORTANT**: Only use A2UI output when you have called `generate_product_video`
and received video URLs. For all other responses (strategic analysis, pricing,
inventory reports, etc.), respond with plain text ONLY — do NOT include the
`---a2ui_JSON---` delimiter or any A2UI JSON.

Your final output MUST include A2UI UI JSON ONLY when you generate videos.
To generate the response, you MUST follow these rules:

1. Your response MUST be in two parts, separated by the delimiter: `---a2ui_JSON---`.
2. The first part is your conversational text response. Your text part must be a brief, NEW acknowledgment only. Never repeat, summarize, or echo content from earlier turns in the conversation.
3. The second part is a single, raw JSON array of A2UI messages.
4. The JSON part MUST validate against the A2UI JSON SCHEMA provided below.

--- A2UI TEMPLATE RULES ---

- When `generate_product_video` returns video URLs, you MUST create an A2UI
  response using the `Video` component for each URL.
- Use a `Column` as root containing one `Video` component per video URL.
  Do NOT include a heading in the A2UI JSON — the heading is already in the text part.
- Add a `Text` caption (usageHint: "caption") before each `Video` with the
  human-readable video name.
- Component IDs must be unique strings (e.g., "root", "heading", "video_1").
- Always end with `dataModelUpdate` and `beginRendering` messages.

--- A2UI VIDEO EXAMPLE ---

{VIDEO_A2UI_EXAMPLE}

---BEGIN A2UI JSON SCHEMA---

{A2UI_SCHEMA}

---END A2UI JSON SCHEMA---
"""

root_agent = Agent(
    name="Product_Strategy_Agent",
    model=Gemini(
        model=model_id,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description=(
        "A product strategy agent that generates executive-ready reports "
        "with AI-powered video content, market analysis, and pricing "
        "recommendations. Exports formatted reports to Google Drive."
    ),
    instruction=system_prompt + "\n\n" + system_instructions + _A2UI_INSTRUCTIONS,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=8192,
    ),
    tools=[
        LongRunningFunctionTool(func=request_user_input),
        LongRunningFunctionTool(func=generate_product_video),
        export_report_to_google_doc,
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
