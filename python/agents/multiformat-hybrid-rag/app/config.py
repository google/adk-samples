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

"""Shared configuration for the serving layer (agent, MCP, REST API).

Single source of truth for env-var-driven defaults so agent.py,
mcp_server.py, and fast_api_app.py don't redefine the same constants.
"""

import os
from pathlib import Path

import google.auth
from dotenv import load_dotenv

# Load config.env from project root (one level up from app/) on import.
# Lets `make playground` / `make local-backend` work without sourcing
# the file in the shell — the app picks it up regardless of how it's
# launched. override=False so real env vars still win.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / "config.env", override=False)

# Prefer PROJECT_ID from config.env / shell over the gcloud ADC project.
# Otherwise the active gcloud config silently overrides what's in config.env.
PROJECT_ID = os.getenv("PROJECT_ID")
if not PROJECT_ID:
    _, PROJECT_ID = google.auth.default()

LOCATION = os.getenv("DEFAULT_REGION", "us-central1")

# --- Vector Search collections -------------------------------------------
SEMANTIC_WEIGHT = float(os.getenv("VS_SEMANTIC_WEIGHT", "0.7"))
VS_COLLECTION_ID = os.getenv("VS_COLLECTION_ID", "multiformat-hybrid-rag-collection")
VS_DOCUMENTS_COLLECTION_ID = os.getenv(
    "VS_DOCUMENTS_COLLECTION_ID", "multiformat-hybrid-rag-documents"
)
COLLECTION_PATH = os.getenv(
    "VECTOR_SEARCH_COLLECTION",
    f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{VS_COLLECTION_ID}",
)
DOCUMENTS_COLLECTION_PATH = os.getenv(
    "VECTOR_SEARCH_DOCUMENTS_COLLECTION",
    f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{VS_DOCUMENTS_COLLECTION_ID}",
)

# --- Gemini models -------------------------------------------------------
AGENT_MODEL = os.getenv("AGENT_GEMINI_MODEL", "gemini-2.5-flash")
MCP_TOOL_MODEL = os.getenv("MCP_TOOL_GEMINI_MODEL", "gemini-2.5-flash-lite")

# --- Answer generation ---------------------------------------------------
CONTEXT_WINDOW = int(os.getenv("MCP_CONTEXT_WINDOW", "10000"))
