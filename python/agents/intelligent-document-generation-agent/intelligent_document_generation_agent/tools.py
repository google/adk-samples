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
import os
from datetime import datetime
from typing import Optional

import google.auth
import google.auth.transport.requests
import requests as http_requests
from google.auth import impersonated_credentials
from google.oauth2 import id_token

from .utils.config import settings
from .utils.gcs_utils import initialize_gcs_client


def json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


async def upload_generated_doc_to_gcs(state: dict, output_key: str) -> Optional[str]:
    try:
        gcs_client = initialize_gcs_client()
        folder_location_from_state = state.get("project_name")
        agent_run_id = state.get("agent_run_id")

        upload_path_prefix = ""
        if folder_location_from_state:
            project_name = os.path.basename(
                os.path.normpath(folder_location_from_state)
            )
            upload_path_prefix = f"{project_name}/{agent_run_id}"
        else:
            upload_path_prefix = agent_run_id

        docs_to_upload = {
            "summary_md": "summary.md",
            "feature_list_md": "feature_list.md",
            "security_overview_md": "security_overview.md",
        }

        filename = docs_to_upload.get(output_key)
        if not filename:
            logging.error(f"Tool error: Unknown output_key '{output_key}'")
            return None

        content = state.get(output_key)
        if not isinstance(content, str) or not content.strip():
            logging.warning(f"No content found in state for key '{output_key}'.")
            return None

        blob_name = f"{upload_path_prefix}/{filename}"

        bucket = gcs_client.get_bucket(settings.ADK_OUTPUT_BUCKET)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content, content_type="text/markdown")

        gcs_uri = f"gs://{settings.ADK_OUTPUT_BUCKET}/{blob_name}"
        logging.info(f"Successfully uploaded {blob_name} to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logging.error(f"Error during upload: {e}")
        return None


def _fetch_identity_token(audience: str) -> Optional[str]:
    """Mint an OIDC ID token for `audience`.

    In Agent Engine the runtime SA is attached via the metadata server, so
    id_token.fetch_id_token works directly. Locally, user ADC can't mint OIDC
    tokens, so we fall back to impersonating PROJECT_SERVICE_ACCOUNT (the user
    needs roles/iam.serviceAccountTokenCreator on it).
    """
    auth_req = google.auth.transport.requests.Request()
    try:
        return id_token.fetch_id_token(auth_req, audience)
    except Exception:
        pass

    source_creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    target_creds = impersonated_credentials.IDTokenCredentials(
        impersonated_credentials.Credentials(
            source_credentials=source_creds,
            target_principal=settings.PROJECT_SERVICE_ACCOUNT,
            target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
            lifetime=300,
        ),
        target_audience=audience,
        include_email=True,
    )
    target_creds.refresh(auth_req)
    return target_creds.token


def convert_markdown_to_pdf_and_get_signed_url(gcs_uri: str) -> Optional[str]:
    logging.info(f"Initiating remote conversion for GCS file: {gcs_uri}")
    service_url = settings.CONVERSION_SERVICE_URL
    if not service_url:
        logging.error("CONVERSION_SERVICE_URL is not configured in settings.")
        return None

    try:
        identity_token = _fetch_identity_token(service_url)
        if not identity_token:
            logging.error("Failed to obtain OIDC identity token.")
            return None

        response = http_requests.post(
            service_url,
            headers={"Authorization": f"Bearer {identity_token}"},
            params={"gcs_uri": gcs_uri},
        )
        response.raise_for_status()

        signed_url = response.text.strip()
        logging.info(f"Successfully received signed URL: {signed_url}")
        return signed_url
    except Exception as e:
        logging.error(f"Failed to call conversion service: {e}", exc_info=True)
        return None
