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

from google.cloud import vectorsearch_v1beta

from src.utils import vs_utils

# Cached at module load: the FastAPI process serves many /api/search
# requests, so reusing the gRPC channel across them avoids per-request
# client construction overhead.
_search_client: vectorsearch_v1beta.DataObjectSearchServiceClient | None = None


def _get_search_client() -> vectorsearch_v1beta.DataObjectSearchServiceClient:
    global _search_client
    if _search_client is None:
        _search_client = vectorsearch_v1beta.DataObjectSearchServiceClient()
    return _search_client


def _extract_window(
    full_text: str,
    chunk_index: int,
    chunk_size: int,
    chunk_overlap: int,
    window_chars: int,
) -> str:
    """Extract a window of text centered on the estimated chunk position.

    chunk_index comes from the chunk_id ({file_id}__{index}). The position
    is approximate (splitter may break at different points) but accurate
    enough for a multi-thousand-char window.
    """
    estimated_start = chunk_index * (chunk_size - chunk_overlap)
    start = max(0, estimated_start - window_chars)
    end = min(len(full_text), estimated_start + chunk_size + window_chars)
    return full_text[start:end]


def search_collection(
    query: str,
    collection_path: str,
    documents_collection_path: str,
    top_k: int = 10,
    semantic_weight: float = 0.7,
    context_window: int | None = None,
) -> str:
    """Search the chunks collection (hybrid semantic + text), then resolve
    each unique file_id hit to its full document text via the documents
    KV collection.

    Two-step retrieval keeps chunks small in VS2 (no document_text duplicated
    on every chunk) and lets a query that hits N chunks across M files return
    M full documents instead of N chunk snippets.

    When context_window is set, returns a window of text centered on the
    matched chunk instead of the full document. The position is derived from
    chunk_index (in the chunk_id) and the chunk_size/overlap defaults. This
    keeps context relevant and bounded for large documents.

    Args:
        query: The search query text.
        collection_path: Full resource path of the chunks collection.
        documents_collection_path: Full resource path of the documents-by-file_id
            KV collection. Looked up after dedup via point GetDataObject calls.
        top_k: Number of unique documents to return.
        semantic_weight: Weight for semantic search in RRF fusion (0-1).
            Text weight is 1 - semantic_weight.
        context_window: If set, return this many chars on each side of the
            matched chunk instead of the full document. None = full documents.

    Returns:
        Formatted string containing relevant document content.
    """
    if os.getenv("INTEGRATION_TEST") == "TRUE":
        return "## Context provided:\n<Document 0>\nMock vector search result for testing purposes.\n</Document 0>"

    client = _get_search_client()

    # chunk_text is still requested as a fallback in case a hit's file_id
    # is missing from the documents collection (e.g. mid-pipeline-run drift).
    output_fields = vectorsearch_v1beta.OutputFields(
        data_fields=["file_id", "chunk_id", "chunk_text", "gcs_uri"]
    )

    text_weight = 1.0 - semantic_weight

    request = vectorsearch_v1beta.BatchSearchDataObjectsRequest(
        parent=collection_path,
        searches=[
            vectorsearch_v1beta.Search(
                semantic_search=vectorsearch_v1beta.SemanticSearch(
                    search_text=query,
                    search_field="text_embedding",
                    task_type="RETRIEVAL_QUERY",
                    top_k=top_k * 2,
                    output_fields=output_fields,
                ),
            ),
            vectorsearch_v1beta.Search(
                text_search=vectorsearch_v1beta.TextSearch(
                    search_text=query,
                    data_field_names=["chunk_text"],
                    top_k=top_k * 2,
                    output_fields=output_fields,
                ),
            ),
        ],
        combine=vectorsearch_v1beta.BatchSearchDataObjectsRequest.CombineResultsOptions(
            ranker=vectorsearch_v1beta.Ranker(
                rrf=vectorsearch_v1beta.ReciprocalRankFusion(
                    weights=[semantic_weight, text_weight],
                ),
            ),
            output_fields=output_fields,
            top_k=top_k,
        ),
    )

    response = client.batch_search_data_objects(request)

    # Walk hits in rank order, dedup by file_id, remember per-file fallback chunk_text
    # and chunk_id (needed to derive chunk_index for windowed extraction).
    ordered_file_ids: list[str] = []
    seen_file_ids: set[str] = set()
    fallback_chunk_text: dict[str, str] = {}
    chunk_id_by_file: dict[str, str] = {}
    gcs_uri_by_file: dict[str, str] = {}
    for search_response in response.results:
        for result in search_response.results:
            file_id = result.data_object.data.get("file_id", "")
            if not file_id or file_id in seen_file_ids:
                continue
            seen_file_ids.add(file_id)
            ordered_file_ids.append(file_id)
            fallback_chunk_text[file_id] = result.data_object.data.get("chunk_text", "")
            chunk_id_by_file[file_id] = result.data_object.data.get("chunk_id", "")
            gcs_uri_by_file[file_id] = result.data_object.data.get("gcs_uri", "")

    if not ordered_file_ids:
        return "No relevant documents found."

    # One parallel batch-get to resolve full documents.
    docs = vs_utils.batch_get_documents(documents_collection_path, ordered_file_ids)

    # Default chunk params — only used for window position estimation.
    default_chunk_size = int(os.getenv("CHUNK_SIZE", "1500"))
    default_chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "300"))

    formatted_parts = []
    for idx, file_id in enumerate(ordered_file_ids):
        doc = docs.get(file_id)
        if doc and doc.get("content"):
            document_text = doc["content"]
            gcs_uri = doc.get("gcs_uri") or gcs_uri_by_file.get(file_id, "")
        else:
            document_text = fallback_chunk_text.get(file_id, "")
            gcs_uri = gcs_uri_by_file.get(file_id, "")

        if context_window and doc and doc.get("content"):
            chunk_id = chunk_id_by_file.get(file_id, "")
            try:
                chunk_index = int(chunk_id.rsplit("__", 1)[1])
            except (ValueError, IndexError):
                chunk_index = 0
            document_text = _extract_window(
                document_text,
                chunk_index,
                default_chunk_size,
                default_chunk_overlap,
                context_window,
            )

        source = gcs_uri.rsplit("/", 1)[-1] if gcs_uri else ""
        formatted_parts.append(
            f'<Document {idx} source="{source}">\n{document_text}\n</Document {idx}>'
        )

    return "## Context provided:\n" + "\n".join(formatted_parts)
