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

"""Deletes a Vector Search 2.0 Collection and all its data objects."""

import sys

import click
from google.api_core import exceptions
from google.cloud import vectorsearch_v1beta


def _purge_data_objects(collection_name: str) -> int:
    """Delete every data object in the collection.

    A Collection cannot be deleted while it still contains data objects
    (the API returns COLLECTION_HAS_DATA_OBJECTS), so we drain it first.
    There is no list API, so we repeatedly search (which returns up to
    `top_k` objects) and batch-delete until nothing remains.
    """
    search_client = vectorsearch_v1beta.DataObjectSearchServiceClient()
    data_client = vectorsearch_v1beta.DataObjectServiceClient()
    deleted = 0
    while True:
        request = vectorsearch_v1beta.SearchDataObjectsRequest(
            parent=collection_name,
            semantic_search=vectorsearch_v1beta.SemanticSearch(
                search_text="*",
                search_field="text_embedding",
                task_type="RETRIEVAL_QUERY",
                top_k=200,
            ),
        )
        names = []
        for result in search_client.search_data_objects(request):
            obj = result.data_object
            name = obj.name or (
                f"{collection_name}/dataObjects/{obj.data_object_id}"
                if obj.data_object_id
                else ""
            )
            if name:
                names.append(name)
        names = list(dict.fromkeys(names))  # de-dup, preserve order
        if not names:
            return deleted
        data_client.batch_delete_data_objects(
            vectorsearch_v1beta.BatchDeleteDataObjectsRequest(
                parent=collection_name,
                requests=[
                    vectorsearch_v1beta.DeleteDataObjectRequest(name=n)
                    for n in names
                ],
            )
        )
        deleted += len(names)
        click.echo(f"  purged {deleted} data objects...")


@click.command()
@click.argument("project_id")
@click.argument("location")
@click.argument("collection_id")
def main(project_id: str, location: str, collection_id: str) -> None:
    """Delete a Vector Search 2.0 Collection and all its data objects."""
    client = vectorsearch_v1beta.VectorSearchServiceClient()
    collection_name = (
        f"projects/{project_id}/locations/{location}"
        f"/collections/{collection_id}"
    )

    click.echo(f"Deleting collection: {collection_name}")

    try:
        # Drain data objects first; the collection delete fails otherwise.
        purged = _purge_data_objects(collection_name)
        click.echo(f"Purged {purged} data objects.")
        client.delete_collection(
            request=vectorsearch_v1beta.DeleteCollectionRequest(
                name=collection_name
            )
        )
        click.echo("Collection deleted successfully.")
    except exceptions.NotFound:
        click.echo("Collection not found (already deleted).")
    except Exception as e:
        click.echo(f"Error deleting collection: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
