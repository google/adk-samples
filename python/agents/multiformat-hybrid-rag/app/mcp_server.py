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
import argparse
import logging
import os
import re

import vertexai
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app.config import (
    COLLECTION_PATH,
    CONTEXT_WINDOW,
    DOCUMENTS_COLLECTION_PATH,
    LOCATION,
    MCP_TOOL_MODEL,
    PROJECT_ID,
    SEMANTIC_WEIGHT,
)
from app.vector_search import search_collection

logger = logging.getLogger(__name__)

vertexai.init(project=PROJECT_ID, location=LOCATION)

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            initial_delay=1.0,
            attempts=3,
            http_status_codes=[408, 429, 500, 502, 503, 504],
        ),
    ),
)

_SYSTEM_PROMPT = """\
You are a knowledge base assistant. Answer questions based exclusively on the \
provided documents.

Rules:
- Base your answer ONLY on the documents provided below. Do not use external knowledge.
- If the documents do not contain enough information to answer, state this clearly.
- When multiple documents cover the same topic from different time periods, \
prefer the most recent information. Do not mention outdated offers, prices, or \
procedures if a newer version is available in the documents.
- Cite the source document name when available.
- Reply in the same language as the question.
- Be concise and direct."""

# DNS rebinding protection only supports exact hosts or :* port wildcards
# (no subdomain wildcards). Cloud Run URLs change between deploys, so we
# disable the host check here — Cloud Run IAM (--no-allow-unauthenticated)
# is the real security boundary, not the Host header.
server = FastMCP(
    "multiformat-hybrid-rag",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[^\S\n]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _generate_answer(context: str, conversation_summary: str, question: str) -> str:
    context = _normalize_whitespace(context)
    resp = genai_client.models.generate_content(
        model=MCP_TOOL_MODEL,
        contents=(
            f"{context}\n\n"
            f"## Conversation so far:\n{conversation_summary}\n\n"
            f"## Question:\n{question}"
        ),
        config={"temperature": 0.2, "system_instruction": _SYSTEM_PROMPT},
    )
    return resp.text or ""


@server.tool()
async def ask_knowledge_base(
    conversation_summary: str,
    question: str,
    top_k: int = 10,
    generative_answer: bool = True,
) -> str:
    """Ask the knowledge base a question.

    Searches the knowledge base for relevant documents. When generative_answer
    is True (default), uses Gemini to generate a direct answer grounded in the
    retrieved content. When False, returns the raw retrieved documents.

    Args:
        conversation_summary: Summary of the conversation so far, for context.
        question: The user's question to answer.
        top_k: Number of documents to retrieve for context.
        generative_answer: If True, generate an answer with Gemini. If False,
            return the raw retrieved documents.

    Returns:
        A direct answer or raw document content from the knowledge base.
    """
    try:
        context = search_collection(
            query=question,
            collection_path=COLLECTION_PATH,
            documents_collection_path=DOCUMENTS_COLLECTION_PATH,
            top_k=top_k,
            semantic_weight=SEMANTIC_WEIGHT,
            context_window=CONTEXT_WINDOW if generative_answer else None,
        )
    except Exception as e:
        logger.error("Search failed: %s", e)
        return f"Search error: {type(e).__name__}: {e}"

    if context.startswith("No relevant documents"):
        return "I couldn't find any relevant documents in the knowledge base to answer this question."

    if not generative_answer:
        return context

    try:
        return _generate_answer(context, conversation_summary, question)
    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        return f"I found relevant documents but failed to generate an answer: {type(e).__name__}: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("MCP_SERVER_PORT", "8081"))
    )
    args = parser.parse_args()

    if args.transport == "sse":
        server.settings.port = args.port
    server.run(transport=args.transport)
