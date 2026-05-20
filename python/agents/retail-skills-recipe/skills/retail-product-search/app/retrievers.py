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

from google.cloud import vectorsearch


def _create_search_client():
    """Create a Vector Search client. Override in tests via dependency injection."""
    return vectorsearch.DataObjectSearchServiceClient()


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
    client = _create_search_client()

    request = vectorsearch.SearchDataObjectsRequest(
        parent=collection_path,
        semantic_search=vectorsearch.SemanticSearch(
            search_text=query,
            search_field="text_embedding",
            task_type="QUESTION_ANSWERING",
            top_k=top_k,
            output_fields=vectorsearch.OutputFields(
                data_fields=[
                    "product_id", "name", "price", "description",
                    "category", "brand", "rating", "stock",
                ]
            ),
        ),
    )

    results = client.search_data_objects(request)

    formatted_parts = []
    for i, result in enumerate(results):
        data = result.data_object.data
        name = data.get("name", "Unknown")
        price = data.get("price", "N/A")
        brand = data.get("brand", "")
        rating = data.get("rating", "")
        description = data.get("description", "")

        # Keep concise for voice responses
        parts = [f"{name}, ${price}"]
        if brand:
            parts.append(f"by {brand}")
        if rating:
            parts.append(f"rated {rating} out of 5")
        if description:
            parts.append(description[:80])

        formatted_parts.append(f"Product {i + 1}: " + ", ".join(parts))

    if not formatted_parts:
        return "No matching products found. Suggest broadening the search."

    return "Found " + str(len(formatted_parts)) + " products. " + ". ".join(formatted_parts)
