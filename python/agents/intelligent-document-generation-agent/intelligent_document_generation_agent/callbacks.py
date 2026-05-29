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

import logging
from datetime import datetime

from google.adk.agents.callback_context import CallbackContext

from .tools import (
    convert_markdown_to_pdf_and_get_signed_url,
    upload_generated_doc_to_gcs,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def _initialize_empty_state_variables(callback_context: CallbackContext):
    logging.info("Initializing empty state variables.")
    state_keys_to_initialize = {
        "summary_md": "",
        "feature_list_md": "",
        "security_overview_md": "",
        "project_name": "",
    }

    for key, default_value in state_keys_to_initialize.items():
        if key not in callback_context.state:
            callback_context.state[key] = default_value
            logging.info(f"[Callback] Initialized state key '{key}'.")


def before_extraction_callback(callback_context: CallbackContext):
    logging.info("Starting new agent run. Assigning ID and initializing state.")
    _initialize_empty_state_variables(callback_context)
    state = callback_context.state
    state["agent_run_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info(
        f"[Callback] Initialized state key 'agent_run_id': {state['agent_run_id']}."
    )


def after_extraction_callback(callback_context: CallbackContext):
    state = callback_context.state
    structured_json = state.get("populated_data_model_json")
    if structured_json and isinstance(structured_json, dict):
        state["project_name"] = structured_json.get("Project Name", "Unknown_Project")
        logging.info(f"Extraction complete for project: {state['project_name']}")


def get_doc_generation_callback(output_key: str):
    async def callback(callback_context: CallbackContext):
        state = callback_context.state
        logging.info(
            f"Document generation complete for {output_key}. Uploading to GCS..."
        )
        gcs_uri = await upload_generated_doc_to_gcs(state, output_key)
        if gcs_uri:
            logging.info(
                f"Uploading complete. Requesting PDF conversion for {gcs_uri}..."
            )
            signed_url = convert_markdown_to_pdf_and_get_signed_url(gcs_uri)
            if signed_url:
                state[f"{output_key}_pdf_url"] = signed_url
                logging.info(f"Saved PDF signed URL for {output_key}: {signed_url}")

    return callback


after_summary_generation_callback = get_doc_generation_callback("summary_md")
after_feature_list_generation_callback = get_doc_generation_callback("feature_list_md")
after_security_overview_generation_callback = get_doc_generation_callback(
    "security_overview_md"
)
