#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click",
#     "google-cloud-vectorsearch",
# ]
# ///
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

"""Creates a Vector Search 2.0 Collection with auto-embeddings.

Idempotent: checks if the collection already exists before creating.

The collection is configured with:
  - Data schema: question_id, text_chunk, full_text_md (all strings)
  - Vector schema: text_embedding with auto-embedding config
    (gemini-embedding-001, 3072 dims, task_type=RETRIEVAL_DOCUMENT)
"""

import os
import sys

import click
from google.api_core import exceptions
from google.cloud import vectorsearch_v1beta


@click.command()
@click.argument("project_id")
@click.argument("location")
@click.argument("collection_id")
def main(project_id: str, location: str, collection_id: str) -> None:
    """Create a Vector Search 2.0 Collection for RAG data."""
    client = vectorsearch_v1beta.VectorSearchServiceClient()
    parent = f"projects/{project_id}/locations/{location}"
    collection_name = f"{parent}/collections/{collection_id}"

    # Check if collection already exists
    try:
        client.get_collection(
            request=vectorsearch_v1beta.GetCollectionRequest(
                name=collection_name
            )
        )
        click.echo(
            f"Collection '{collection_id}' already exists. Skipping creation."
        )
        return
    except exceptions.NotFound:
        pass

    click.echo(f"Creating collection '{collection_id}'...")

    request = vectorsearch_v1beta.CreateCollectionRequest(
        parent=parent,
        collection_id=collection_id,
        collection={
            "data_schema": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "text_chunk": {"type": "string"},
                    "full_text_md": {"type": "string"},
                },
            },
            "vector_schema": {
                "text_embedding": {
                    "dense_vector": {
                        "dimensions": 3072,
                        "vertex_embedding_config": {
                            "model_id": os.getenv("MODEL_NAME_EMBEDDING"),
                            "text_template": "{text_chunk}",
                            "task_type": "RETRIEVAL_DOCUMENT",
                        },
                    },
                },
            },
        },
    )

    try:
        operation = client.create_collection(request=request)
        click.echo("Waiting for collection creation to complete...")
        operation.result()
        click.echo("Collection created successfully.")
    except Exception as e:
        click.echo(f"Error creating collection: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
