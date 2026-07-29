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
import os

from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from pydantic import BaseModel

from app.config import (
    COLLECTION_PATH,
    CONTEXT_WINDOW,
    DOCUMENTS_COLLECTION_PATH,
    SEMANTIC_WEIGHT,
)
from app.mcp_server import _generate_answer
from app.mcp_server import server as mcp_server
from app.vector_search import search_collection

allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

# Artifact bucket for ADK (created by Terraform, passed via env var)
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# In-memory session configuration - no persistent storage
session_service_uri = None

artifact_service_uri = f"gs://{logs_bucket_name}" if logs_bucket_name else None

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=artifact_service_uri,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
    otel_to_cloud=True,
)
app.title = "multiformat-hybrid-rag"
app.description = "API for interacting with the Agent multiformat-hybrid-rag"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    conversation_summary: str = ""
    generative_answer: bool = True


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    """REST endpoint for knowledge base search.

    With generative_answer=True (default), returns an LLM-generated answer.
    With generative_answer=False, returns raw retrieved documents.
    """
    try:
        context = search_collection(
            query=req.query,
            collection_path=COLLECTION_PATH,
            documents_collection_path=DOCUMENTS_COLLECTION_PATH,
            top_k=req.top_k,
            semantic_weight=SEMANTIC_WEIGHT,
            context_window=CONTEXT_WINDOW if req.generative_answer else None,
        )
        if not req.generative_answer:
            return {"result": context}

        if context.startswith("No relevant documents"):
            return {"result": context}

        answer = _generate_answer(context, req.conversation_summary, req.query)
        return {"result": answer}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# --- Mount MCP server into FastAPI (same port, externally reachable) ---
# Don't pass mount_path here — FastAPI's mount() already prepends /mcp.
# Passing it would double-prefix the SSE messages URL to /mcp/mcp/messages/...
app.mount("/mcp", mcp_server.sse_app())

# Set MCP_SERVER_URL so the ADK agent connects to the mounted path.
# Use the same port as the running server (8080 on Cloud Run, 8000 local).
_server_port = os.getenv("PORT", "8080")
os.environ.setdefault("MCP_SERVER_URL", f"http://localhost:{_server_port}/mcp/sse")

# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
