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

"""RAG Template configuration.

Handles environment setup, authentication, and a single Gemini client.
Loaded at import time — use `from src.utils.config import config, genai_client`.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / "config.env"

# ---------------------------------------------------------------------------
# Load environment variables from config.env
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=DEFAULT_ENV_PATH, override=False)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
if os.getenv("GOOGLE_API_KEY"):
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "False")
else:
    project_id = os.getenv("PROJECT_ID")
    if not project_id:
        try:
            import google.auth

            _, project_id = google.auth.default()
        except Exception:
            project_id = ""
            logger.warning("No PROJECT_ID set and ADC not available")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id or "")
    os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("DEFAULT_REGION", "us-central1")
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
def _get(key: str, default: str | None = None) -> str:
    """Get a config value from environment, raising if missing and no default."""
    val = os.environ.get(key, default)
    if val is None:
        raise ValueError(f"Missing required config: {key}")
    return val


@dataclass(frozen=True)
class Config:
    project_id: str
    region: str
    gcs_bucket: str
    gcs_prefix: str
    bq_dataset: str
    bq_object_table: str
    bq_preprocessed_table: str
    bq_chunks_table: str
    bq_gcs_connection_id: str
    vs_collection_id: str
    vs_documents_collection_id: str
    vs_embedding_model: str
    vs_embedding_dims: int
    vs_batch_size: int
    agent_gemini_model: str
    markdown_converter_gemini_model: str
    contextual_chunking_gemini_model: str
    relevance_gemini_model: str
    vs_semantic_weight: float
    chunk_size: int
    chunk_overlap: int

    @property
    def collection_path(self) -> str:
        return f"projects/{self.project_id}/locations/{self.region}/collections/{self.vs_collection_id}"

    @property
    def documents_collection_path(self) -> str:
        return f"projects/{self.project_id}/locations/{self.region}/collections/{self.vs_documents_collection_id}"

    @property
    def fq_object_table(self) -> str:
        return f"{self.project_id}.{self.bq_dataset}.{self.bq_object_table}"

    @property
    def fq_preprocessed_table(self) -> str:
        return f"{self.project_id}.{self.bq_dataset}.{self.bq_preprocessed_table}"

    @property
    def fq_chunks_table(self) -> str:
        return f"{self.project_id}.{self.bq_dataset}.{self.bq_chunks_table}"

    @property
    def gcs_uri_prefix(self) -> str:
        return f"gs://{self.gcs_bucket}/{self.gcs_prefix}"

    @property
    def bq_connection_path(self) -> str:
        return f"projects/{self.project_id}/locations/{self.region}/connections/{self.bq_gcs_connection_id}"


# Singleton config — loaded once at import time
config = Config(
    project_id=_get("PROJECT_ID"),
    region=_get("DEFAULT_REGION"),
    gcs_bucket=_get("GCS_BUCKET"),
    gcs_prefix=_get("GCS_PREFIX", "documents/"),
    bq_dataset=_get("BQ_DATASET"),
    bq_object_table=_get("BQ_OBJECT_TABLE"),
    bq_preprocessed_table=_get("BQ_PREPROCESSED_TABLE"),
    bq_chunks_table=_get("BQ_CHUNKS_TABLE"),
    bq_gcs_connection_id=_get("BQ_GCS_CONNECTION_ID"),
    vs_collection_id=_get("VS_COLLECTION_ID"),
    vs_documents_collection_id=_get("VS_DOCUMENTS_COLLECTION_ID"),
    vs_embedding_model=_get("VS_EMBEDDING_MODEL", "gemini-embedding-001"),
    vs_embedding_dims=int(_get("VS_EMBEDDING_DIMS", "3072")),
    vs_batch_size=int(_get("VS_BATCH_SIZE", "250")),
    agent_gemini_model=_get("AGENT_GEMINI_MODEL", "gemini-2.5-flash"),
    markdown_converter_gemini_model=_get(
        "MARKDOWN_CONVERTER_GEMINI_MODEL", "gemini-3-flash-preview"
    ),
    contextual_chunking_gemini_model=_get(
        "CONTEXTUAL_CHUNKING_GEMINI_MODEL", "gemini-2.5-flash-lite"
    ),
    relevance_gemini_model=_get("RELEVANCE_GEMINI_MODEL", "gemini-2.5-flash-lite"),
    vs_semantic_weight=float(_get("VS_SEMANTIC_WEIGHT", "0.7")),
    chunk_size=int(_get("CHUNK_SIZE", "800")),
    chunk_overlap=int(_get("CHUNK_OVERLAP", "50")),
)

# ---------------------------------------------------------------------------
# Shared Gemini client (single global endpoint — works for all models)
# ---------------------------------------------------------------------------
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

try:
    genai_client = genai.Client(
        vertexai=True,
        project=config.project_id,
        location="global",
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                initial_delay=1.0,
                attempts=5,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )
except Exception as e:
    genai_client = None
    logger.warning("Gemini client not available: %s", e)
