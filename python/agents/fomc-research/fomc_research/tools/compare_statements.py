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

"""'compare_statements' tool for FOMC Research sample agent."""

import logging

from google.adk.tools import ToolContext
from google.genai.types import Part

from ..shared_libraries import file_utils

logger = logging.getLogger(__name__)


async def compare_statements_tool(tool_context: ToolContext) -> dict[str, str]:
    """Compares requested and previous statements and generates HTML redline.

    Args:
      tool_context: ToolContext object.

    Returns:
      A dict with "status" and (optional) "error_message" keys.
    """
    fed_hostname = "https://www.federalreserve.gov"

    reqd_statement_url = tool_context.state[
        "requested_meeting_statement_pdf_url"
    ]
    if not reqd_statement_url.startswith("https"):
        reqd_statement_url = fed_hostname + reqd_statement_url
    prev_statement_url = tool_context.state[
        "previous_meeting_statement_pdf_url"
    ]
    if not prev_statement_url.startswith("https"):
        prev_statement_url = fed_hostname + prev_statement_url

    # Download PDFs from URLs to artifacts
    reqd_pdf_path = await file_utils.download_file_from_url(
        reqd_statement_url, "curr.pdf", tool_context
    )
    prev_pdf_path = await file_utils.download_file_from_url(
        prev_statement_url, "prev.pdf", tool_context
    )

    if reqd_pdf_path is None or reqd_pdf_path is None:
        logger.error("Failed to download files, aborting")
        return {
            "status": "error",
            "error_message": "Failed to download statement files",
        }

    reqd_pdf_text = await file_utils.extract_text_from_pdf_artifact(
        reqd_pdf_path, tool_context
    )
    prev_pdf_text = await file_utils.extract_text_from_pdf_artifact(
        prev_pdf_path, tool_context
    )

    if reqd_pdf_text is None or prev_pdf_text is None:
        logger.error("Failed to extract text from PDFs, aborting")
        return {
            "status": "error",
            "error_message": "Failed to extract text from PDFs",
        }

    await tool_context.save_artifact(
        filename="requested_statement_fulltext",
        artifact=Part(text=reqd_pdf_text),
    )
    await tool_context.save_artifact(
        filename="previous_statement_fulltext",
        artifact=Part(text=prev_pdf_text),
    )

    redline_html = file_utils.create_html_redline(reqd_pdf_text, prev_pdf_text)
    await file_utils.save_html_to_artifact(
        redline_html, "statement_redline", tool_context
    )

    return {"status": "ok"}
