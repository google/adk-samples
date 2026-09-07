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

"""Deletes a GCS Data Connector collection and its associated data store."""

import sys

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
def main(project_id: str, location: str, collection_id: str) -> None:
    """Delete a data connector collection and its data store."""
    service = _build_service(location, project_id)

    collection_name = (
        f"projects/{project_id}/locations/{location}"
        f"/collections/{collection_id}"
    )

    click.echo(f"Deleting collection: {collection_name}")

    try:
        service.projects().locations().collections().delete(
            name=collection_name
        ).execute()
        click.echo("Collection deleted successfully.")
    except HttpError as e:
        if e.resp.status == 404:
            click.echo("Collection not found (already deleted).")
        else:
            click.echo(f"Error deleting collection: {e}", err=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
