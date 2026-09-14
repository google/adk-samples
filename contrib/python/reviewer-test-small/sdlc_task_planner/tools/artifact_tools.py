import logging
from typing import Any

from google.adk.tools import ToolContext
from google.genai import types

logger = logging.getLogger(__name__)


async def save_artifact(
    tool_context: ToolContext,
    content: str,
    filename: str,
    format: str = "markdown",
) -> dict[str, Any]:
    """
    Saves text content as an ADK artifact.

    This tool takes text input and saves it as an artifact using
    the configured ArtifactService. The artifact will be versioned automatically.

    Args:
        tool_context (ToolContext): The ADK tool context providing access to
                                   artifact service methods.
        content (str): The text content to save as an artifact.
        filename (str): The name for the artifact file. The agent should choose a descriptive name.
        format (str): The format of the content. Currently supported: 'markdown'. Defaults to 'markdown'.

    Returns:
        dict[str, Any]: A dictionary containing:
            - status (str): 'success' or 'error'
            - filename (str): The name of the created artifact
            - version (int): The version number assigned to the artifact (on success)
            - message (str): A descriptive message about the operation result
            - error (str, optional): Error details if the operation failed
    """
    try:
        if not content or not filename:
            return {
                "status": "error",
                "filename": filename,
                "message": "Content and filename needs to be provided",
                "error": "Missing required parameters",
            }

        if format.lower() == "markdown":
            mime_type = "text/markdown"
            if not filename.lower().endswith(".md"):
                filename = f"{filename}.md"
        else:
            mime_type = "text/plain"

        content_bytes = content.encode("utf-8")

        artifact = types.Part.from_bytes(
            data=content_bytes, mime_type=mime_type
        )

        version = await tool_context.save_artifact(
            filename=filename, artifact=artifact
        )

        logger.info(
            f"Successfully saved artifact '{filename}' as version {version}"
        )

        return {
            "status": "success",
            "filename": filename,
            "version": version,
            "message": (
                f"Successfully saved content to artifact '{filename}' (version {version})"
            ),
        }

    except ValueError as e:
        logger.error(f"ValueError: {e!s}")
        return {
            "status": "error",
            "filename": filename,
            "message": (
                "ArtifactService not configured. Ensure artifact_service is provided to the Runner."
            ),
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e!s}")
        return {
            "status": "error",
            "filename": filename,
            "message": (
                "An unexpected error occurred while saving the artifact"
            ),
            "error": str(e),
        }
