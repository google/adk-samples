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


def search_collection(
    query: str,
    collection_path: str,
    top_k: int = 10,
) -> str:
    """Search a Vector Search 2.0 Collection using semantic search.

    Args:
        query: The search query text.
        collection_path: Full resource path of the collection.
        top_k: Number of results to return.

    Returns:
        Formatted string containing relevant document content.
    """
    # For integration tests, return mock data instead of calling the real API
    if os.getenv("INTEGRATION_TEST") == "TRUE":
        return (
            "## Context provided:\n<Document 0>\n"
            "Mock vector search result for testing purposes.\n"
            "</Document 0>"
        )

    client = vectorsearch_v1beta.DataObjectSearchServiceClient()

    request = vectorsearch_v1beta.SearchDataObjectsRequest(
        parent=collection_path,
        semantic_search=vectorsearch_v1beta.SemanticSearch(
            search_text=query,
            search_field="text_embedding",
            task_type="RETRIEVAL_QUERY",
            top_k=top_k,
            output_fields=vectorsearch_v1beta.OutputFields(
                data_fields=["question_id", "text_chunk", "full_text_md"]
            ),
        ),
    )

    results = client.search_data_objects(request)

    formatted_parts = []
    for i, result in enumerate(results):
        text_chunk = result.data_object.data.get("text_chunk", "")
        formatted_parts.append(f"<Document {i}>\n{text_chunk}\n</Document {i}>")

    if not formatted_parts:
        return "No relevant documents found."

    return "## Context provided:\n" + "\n".join(formatted_parts)
