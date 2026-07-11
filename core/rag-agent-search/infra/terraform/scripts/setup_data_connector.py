#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click",
#     "google-api-python-client",
#     "google-auth",
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

"""Sets up a GCS Data Connector for Agent Platform Search.

Idempotent: checks if the connector already exists before creating.

The connector uses `data_schema: "content"` by default to ingest unstructured
files (PDF, HTML, TXT, etc.) rather than the default JSON/NDJSON format.
Supported data_schema values:
  - "content"  : Unstructured files (PDF, HTML, TXT). One doc per file.
  - "document" : (default API) One JSON document per line (NDJSON/JSONL).
  - "csv"      : CSV with header row conforming to the data store schema.
  - "custom"   : Custom JSON format conforming to the data store schema.

See the Discovery Engine API reference for full details:
https://discoveryengine.googleapis.com/$discovery/rest?version=v1alpha
"""

import sys
import time

import click
import google.auth
from googleapiclient import discovery
from googleapiclient.errors import HttpError


def _build_service(location: str, project_id: str):
    """Build a Discovery Engine v1alpha service client."""
    credentials, _ = google.auth.default()
    if location == "global":
        endpoint = "https://discoveryengine.googleapis.com"
    else:
        endpoint = f"https://{location}-discoveryengine.googleapis.com"
    return discovery.build(
        "discoveryengine",
        "v1alpha",
        credentials=credentials,
        discoveryServiceUrl=f"{endpoint}/$discovery/rest?version=v1alpha",
    )


@click.command()
@click.argument("project_id")
@click.argument("location")
@click.argument("collection_id")
@click.argument("display_name")
@click.argument("gcs_uri")
@click.option(
    "--refresh-interval",
    default="86400s",
    help="Refresh interval for periodic sync (e.g. '86400s' for daily).",
)
@click.option(
    "--data-schema",
    default="content",
    type=click.Choice(["content", "document", "csv", "custom"]),
    help="Data schema for ingested files.",
)
def main(
    project_id: str,
    location: str,
    collection_id: str,
    display_name: str,
    gcs_uri: str,
    refresh_interval: str,
    data_schema: str,
) -> None:
    """Set up a GCS Data Connector for Agent Platform Search.

    Creates a data connector that syncs files from a GCS bucket into
    a Agent Platform Search data store.
    """
    gcs_uri = gcs_uri.rstrip("/") + "/*"
    service = _build_service(location, project_id)

    parent = f"projects/{project_id}/locations/{location}"
    connector_name = f"{parent}/collections/{collection_id}/dataConnector"

    # Check if connector already exists (singleton per collection)
    try:
        existing = (
            service.projects()
            .locations()
            .collections()
            .getDataConnector(name=connector_name)
            .execute()
        )
        state = existing.get("state", "UNKNOWN")
        click.echo(
            f"Data connector already exists in collection"
            f" '{collection_id}' (state: {state})"
        )
        click.echo("Skipping creation.")
        return
    except HttpError as e:
        if e.resp.status != 404:
            raise

    click.echo(f"Creating data connector '{display_name}'...")

    # Entity params vary by data schema (see API reference):
    # - "content": unstructured files (PDF, HTML, TXT). IDs are auto-generated
    #   from SHA256(URI). Needs CONTENT_REQUIRED for raw file storage.
    # - "document": structured NDJSON with Document.id in each line.
    # - "csv"/"custom": structured data; auto_generate_ids creates IDs from
    #   payload hash (only valid for these schemas per the API spec).
    entity_params = {"data_schema": data_schema}
    if data_schema == "content":
        entity_params["content_config"] = "CONTENT_REQUIRED"
    elif data_schema in ("csv", "custom"):
        entity_params["auto_generate_ids"] = True

    body = {
        "dataSource": "gcs",
        "refreshInterval": refresh_interval,
        "params": {"instance_uris": [gcs_uri]},
        "entities": [
            {"entityName": "documents", "params": entity_params},
        ],
        "staticIpEnabled": False,
    }

    operation = (
        service.projects()
        .locations()
        .setUpDataConnectorV2(
            parent=parent,
            collectionId=collection_id,
            collectionDisplayName=display_name,
            body=body,
        )
        .execute()
    )

    lro_name = operation.get("name", "")
    if operation.get("done", False):
        click.echo("Data connector created successfully.")
        return

    click.echo(f"Waiting for creation to complete (LRO: {lro_name})...")

    # Poll LRO until complete
    for attempt in range(1, 61):
        time.sleep(10)
        status = (
            service.projects()
            .locations()
            .collections()
            .dataConnector()
            .operations()
            .get(name=lro_name)
            .execute()
        )
        if status.get("done", False):
            if "error" in status:
                click.echo(
                    f"Error creating data connector: {status['error']}",
                    err=True,
                )
                sys.exit(1)
            click.echo("Data connector created successfully.")
            return
        click.echo(f"  Attempt {attempt}/60...")

    click.echo("Error: LRO did not complete within the timeout.", err=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
