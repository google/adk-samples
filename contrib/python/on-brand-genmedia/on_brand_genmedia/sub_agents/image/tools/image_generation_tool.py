# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import traceback
from pathlib import Path

from google import genai
from google.adk.tools import ToolContext
from google.cloud import storage
from google.genai import types

from .... import config

logger = logging.getLogger(__name__)

CLIENT_RETRY_ATTEMPTS = 5
CLIENT_TIMEOUT_MS = 120 * 1000


async def generate_images(
    image_gen_prompt: str,
    reference_images: list[str],
    tool_context: ToolContext,
):
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("RE_PROJECT_ID"),
        location=os.environ.get("RE_LOCATION"),
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                initial_delay=1.0,
                attempts=CLIENT_RETRY_ATTEMPTS,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
            timeout=CLIENT_TIMEOUT_MS,
        ),
    )
    logger.info("Entered generate_images tool.")
    logger.debug(f"image_gen_prompt: {image_gen_prompt}")
    logger.debug(f"reference_images: {reference_images}")
    try:
        parts: list[types.Part] = []
        if reference_images:
            storage_client = storage.Client()
            image_path_or_uri = reference_images[0]
            if image_path_or_uri.startswith("gs://"):
                path_parts = image_path_or_uri[5:].split("/", 1)
                if len(path_parts) != 2:
                    raise ValueError(
                        f"Invalid GCS URI format: {image_path_or_uri}"
                    )

                bucket_name, blob_name = path_parts
                if (
                    config.GCS_BUCKET_NAME
                    and bucket_name != config.GCS_BUCKET_NAME
                ):
                    raise ValueError(
                        f"Bucket name '{bucket_name}' does not match configured GCS_BUCKET_NAME '{config.GCS_BUCKET_NAME}'"
                    )
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                image_bytes = blob.download_as_bytes()
                mime_type = (
                    "image/jpeg"
                    if blob_name.lower().endswith((".jpg", ".jpeg"))
                    else "image/png"
                )
            else:
                base_dir = Path(__file__).resolve().parents[3]
                candidate_path = (
                    base_dir / "data" / image_path_or_uri
                ).resolve()
                data_dir = (base_dir / "data").resolve()
                if not candidate_path.is_relative_to(data_dir):
                    raise ValueError(
                        f"Path traversal detected: {image_path_or_uri}"
                    )
                if not candidate_path.exists():
                    raise FileNotFoundError(
                        f"Local image file not found: {candidate_path}"
                    )
                with open(candidate_path, "rb") as f:
                    image_bytes = f.read()
                mime_type = (
                    "image/jpeg"
                    if candidate_path.suffix.lower() in [".jpg", ".jpeg"]
                    else "image/png"
                )

            parts.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                )
            )

        parts.append(types.Part.from_text(text=image_gen_prompt))

        model = config.IMAGE_GEN_MODEL
        contents = [
            types.Content(role="user", parts=parts),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=32768,
            response_modalities=["TEXT", "IMAGE"],
            system_instruction=[
                types.Part.from_text(text=config.IMAGE_GEN_SYSTEM_INSTRUCTION)
            ],
            image_config=types.ImageConfig(
                aspect_ratio="1:1",
                image_size="1K",
                output_mime_type="image/png",
            ),
        )
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        generated_image_part = None
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    generated_image_part = part
                    break

        if generated_image_part:
            image_bytes = generated_image_part.inline_data.data
            counter = str(tool_context.state.get("loop_iteration", 0))
            artifact_name = f"generated_image_{counter}.png"

            report_artifact = types.Part.from_bytes(
                data=image_bytes, mime_type="image/png"
            )
            await tool_context.save_artifact(artifact_name, report_artifact)

            return {
                "status": "success",
                "message": f"Image generated. ADK artifact: {artifact_name}.",
                "artifact_name": artifact_name,
            }
        else:
            # Capture text response if present (refusals or explanations)
            response_text = ""
            if (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                for part in response.candidates[0].content.parts:
                    if part.text:
                        response_text += part.text

            if response_text:
                logger.warning(
                    f"Model refused image generation: {response_text}"
                )
                return {
                    "status": "refusal",
                    "message": f"Agent declined to generate image: {response_text}",
                }

            error_details = str(response)
            logger.error(f"No images generated. Response: {error_details}")
            tool_context.actions.escalate = True
            return {
                "status": "error",
                "message": f"No images generated. Response: {error_details}",
            }

    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        logger.error(f"Error generating images: {e}", exc_info=True)
        tool_context.actions.escalate = True
        return {"status": "error", "message": f"No images generated. {e}"}
