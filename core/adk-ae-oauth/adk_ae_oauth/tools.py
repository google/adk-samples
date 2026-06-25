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

"""Tools for the Google Drive reader agent.

Implements the OAuth credential negotiation pattern from:
https://fmind.medium.com/powering-up-your-agent-in-production-with-adk-oauth-and-gemini-enterprise-a52b0716fcba
"""

import json
import logging

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.adk.tools import ToolContext
from googleapiclient.discovery import build

from adk_ae_oauth import auths

logger = logging.getLogger(__name__)


def negotiate_creds(tool_context: ToolContext) -> Credentials | dict:
    """Handle the OAuth 2.0 flow to get valid credentials.

    This function implements a three-stage credential resolution:
    1. Check for cached credentials in tool_context.state (including tokens
       injected by Gemini Enterprise via "temp:<AUTH_ID>").
    2. Check for an auth response from the ADK OAuth flow.
    3. If nothing is available, initiate the OAuth flow.
    """
    logger.info("Negotiating credentials using OAuth 2.0")

    # --- Stage 1: Check for cached / injected token ---
    # On Agent Engine + Gemini Enterprise, the token is injected as:
    #   tool_context.state["temp:<AUTH_ID>"]  (a raw access token string)
    # During local ADK dev, we cache the full credential dict under the
    # same key after the first successful auth.
    cached_token = tool_context.state.get(auths.TOKEN_CACHE_KEY)

    # Also check the "temp:" prefixed key that Gemini Enterprise uses
    if cached_token is None:
        cached_token = tool_context.state.get(f"temp:{auths.TOKEN_CACHE_KEY}")

    if cached_token:
        logger.debug("Found cached token in tool context state")

        if isinstance(cached_token, dict):
            logger.debug("Cached token is a dict — loading as Credentials object")
            try:
                creds = Credentials.from_authorized_user_info(
                    cached_token, list(auths.SCOPES.keys())
                )
                if creds.valid:
                    logger.debug("Cached credentials are valid")
                    return creds
                if creds.expired and creds.refresh_token:
                    logger.debug("Cached credentials expired — refreshing")
                    creds.refresh(Request())
                    tool_context.state[auths.TOKEN_CACHE_KEY] = json.loads(
                        creds.to_json()
                    )
                    logger.debug("Credentials refreshed and cached")
                    return creds
            except Exception as error:
                logger.error(f"Error loading/refreshing cached credentials: {error}")
                tool_context.state[auths.TOKEN_CACHE_KEY] = None

        elif isinstance(cached_token, str):
            logger.debug("Found raw access token string")
            return Credentials(token=cached_token)
        else:
            raise ValueError(
                f"Invalid cached token type. Expected dict or str, got {type(cached_token)}"
            )

    # --- Stage 2: Check for an auth response from the ADK OAuth flow ---
    logger.debug("No valid cached token — checking for auth response")
    if exchanged_creds := tool_context.get_auth_response(auths.AUTH_CONFIG):
        logger.debug("Received auth response — creating credentials")
        auth_scheme = auths.AUTH_CONFIG.auth_scheme
        auth_credential = auths.AUTH_CONFIG.raw_auth_credential
        creds = Credentials(
            token=exchanged_creds.oauth2.access_token,
            refresh_token=exchanged_creds.oauth2.refresh_token,
            token_uri=auth_scheme.flows.authorizationCode.tokenUrl,
            client_id=auth_credential.oauth2.client_id,
            client_secret=auth_credential.oauth2.client_secret,
            scopes=list(auth_scheme.flows.authorizationCode.scopes.keys()),
        )
        tool_context.state[auths.TOKEN_CACHE_KEY] = json.loads(creds.to_json())
        logger.debug("New credentials created and cached")
        return creds

    # --- Stage 3: Initiate OAuth flow ---
    logger.debug("No credentials available — requesting user authentication")
    tool_context.request_credential(auths.AUTH_CONFIG)
    logger.info("Awaiting user authentication")
    return {"pending": True, "message": "Awaiting user authentication"}


def read_drive_file(file_id: str, tool_context: ToolContext) -> dict:
    """Read the text content of a Google Drive file.

    Use this tool to read the content of a file stored in Google Drive.
    The user must provide the file ID (the alphanumeric string in the
    Google Drive URL). Supports Google Docs (exported as plain text)
    and plain text / CSV files.

    Args:
        file_id: The Google Drive file ID to read.

    Returns:
        A dict with 'status' and 'content' keys.
    """
    # 1. Negotiate credentials
    creds = negotiate_creds(tool_context)

    # If credentials are pending (OAuth flow in progress), return early
    if isinstance(creds, dict):
        return creds

    try:
        # 2. Build the Drive API client
        service = build("drive", "v3", credentials=creds)

        # 3. Get file metadata to determine mime type
        file_meta = (
            service.files()
            .get(fileId=file_id, fields="id,name,mimeType")
            .execute()
        )
        file_name = file_meta.get("name", "unknown")
        mime_type = file_meta.get("mimeType", "")

        logger.info(f"Reading file: {file_name} (type: {mime_type})")

        # 4. Read the file content
        if mime_type == "application/vnd.google-apps.document":
            # Google Docs → export as plain text
            content = (
                service.files()
                .export(fileId=file_id, mimeType="text/plain")
                .execute()
            )
            text_content = content.decode("utf-8") if isinstance(content, bytes) else content

        elif mime_type == "application/vnd.google-apps.spreadsheet":
            # Google Sheets → export as CSV
            content = (
                service.files()
                .export(fileId=file_id, mimeType="text/csv")
                .execute()
            )
            text_content = content.decode("utf-8") if isinstance(content, bytes) else content

        elif mime_type == "application/vnd.google-apps.presentation":
            # Google Slides → export as plain text
            content = (
                service.files()
                .export(fileId=file_id, mimeType="text/plain")
                .execute()
            )
            text_content = content.decode("utf-8") if isinstance(content, bytes) else content

        else:
            # Regular files (text, csv, json, etc.) → download directly
            content = (
                service.files()
                .get_media(fileId=file_id)
                .execute()
            )
            text_content = content.decode("utf-8") if isinstance(content, bytes) else content

        return {
            "status": "success",
            "file_name": file_name,
            "mime_type": mime_type,
            "content": text_content,
        }

    except Exception as e:
        logger.error(f"Error reading Drive file: {e}")
        return {
            "status": "error",
            "message": f"Failed to read file: {str(e)}",
        }
