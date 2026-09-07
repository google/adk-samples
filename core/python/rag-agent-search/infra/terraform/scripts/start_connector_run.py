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

"""Imports documents from GCS into a Agent Platform Search data store.

Uses the ImportDocuments API to sync files from the GCS bucket configured
in the data connector into the associated data store.
"""

import sys
import time

import click
import google.auth
from googleapiclient import discovery


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
@click.option("--project", required=True, help="GCP project ID.")
@click.option("--region", required=True, help="GCP region/location.")
@click.option(
    "--collection-id",
    required=True,
    help="Collection ID for the data connector.",
)
@click.option("--wait", is_flag=True, help="Poll until the import completes.")
def main(project: str, region: str, collection_id: str, wait: bool) -> None:
    """Import documents from GCS into the data connector's data store."""
    project_id = project
    location = region
    service = _build_service(location, project_id)

    connector_name = (
        f"projects/{project_id}/locations/{location}"
        f"/collections/{collection_id}/dataConnector"
    )

    # Get the connector to find the data store and GCS URI
    click.echo(f"Reading connector: {connector_name}")
    try:
        connector = (
            service.projects()
            .locations()
            .collections()
            .getDataConnector(name=connector_name)
            .execute()
        )
    except Exception as e:
        click.echo(f"Error reading connector: {e}", err=True)
        sys.exit(1)

    entities = connector.get("entities", [])
    if not entities:
        click.echo("Error: No entities found on the data connector.", err=True)
        sys.exit(1)

    entity = entities[0]
    data_store = entity.get("dataStore", "")
    data_schema = entity.get("params", {}).get("data_schema", "content")
    gcs_uris = connector.get("params", {}).get("instance_uris", [])

    if not data_store or not gcs_uris:
        click.echo(
            "Error: Could not determine data store or GCS URI from connector.",
            err=True,
        )
        sys.exit(1)

    # ImportDocuments uses branches/default_branch
    branch = f"{data_store}/branches/default_branch"

    click.echo(f"Importing from {gcs_uris} into {data_store}")
    click.echo(f"Data schema: {data_schema}")

    # Trigger the import
    try:
        operation = (
            service.projects()
            .locations()
            .collections()
            .dataStores()
            .branches()
            .documents()
            .import_(
                parent=branch,
                body={
                    "gcsSource": {
                        "inputUris": gcs_uris,
                        "dataSchema": data_schema,
                    },
                    "reconciliationMode": "FULL",
                },
            )
            .execute()
        )
        lro_name = operation.get("name", "")
        click.echo(f"Import started (LRO: {lro_name})")
    except Exception as e:
        click.echo(f"Error starting import: {e}", err=True)
        sys.exit(1)

    # Extract data store ID for the console URL
    parts = data_store.split("/")
    if "dataStores" in parts:
        idx = parts.index("dataStores")
        data_store_id = parts[idx + 1]
    else:
        data_store_id = f"{collection_id}_documents"

    console_url = (
        f"https://console.cloud.google.com/gen-app-builder"
        f"/locations/{location}/collections/{collection_id}"
        f"/data-stores/{data_store_id}/data/documents"
        f"?project={project_id}"
    )
    click.echo(f"\nView your documents: {console_url}")

    if not wait:
        return

    click.echo("\nWaiting for import to complete...")

    for _attempt in range(1, 61):
        time.sleep(10)
        try:
            status = (
                service.projects()
                .locations()
                .collections()
                .dataStores()
                .branches()
                .operations()
                .get(name=lro_name)
                .execute()
            )
        except Exception as e:
            click.echo(f"\nError polling import status: {e}", err=True)
            sys.exit(1)

        if not status.get("done", False):
            click.echo(".", nl=False)
            continue

        # Import completed
        if "error" in status:
            error = status["error"]
            click.echo(f"\nImport failed: {error.get('message', error)}")
            sys.exit(1)

        metadata = status.get("metadata", {})
        success_count = metadata.get("successCount", "0")
        total_count = metadata.get("totalCount", "0")
        failure_count = metadata.get("failureCount", "0")

        click.echo(
            f"\nImport completed: {success_count}/{total_count} documents"
            f" imported ({failure_count} failures)."
        )
        return

    click.echo("\nError: Import did not complete within the timeout.", err=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
