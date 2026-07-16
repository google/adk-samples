# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Import Google Cloud Discovery Engine libraries
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine

# Define the path to the .env file
env_file_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_file_path)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def refresh_vertex_datastore(project_id: str) -> bool:
    """
    Triggers a FULL sync from BigQuery to the Vertex AI Search Datastore
    and waits for the operation to complete.

    Args:
        project_id (str): The Google Cloud project ID.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Fetch configurations from environment variables
        location = os.getenv("VERTEX_SEARCH_LOCATION", "global")
        data_store_id = os.getenv("VERTEX_AI_BQ_DATASTORE_ID")
        bigquery_dataset = os.getenv("BIGQUERY_DATASET")
        bigquery_table = os.getenv("BIGQUERY_TABLE")

        if not all([data_store_id, bigquery_dataset, bigquery_table]):
            raise ValueError(
                "Missing required environment variables for Datastore refresh. "
                "Ensure VERTEX_AI_BQ_DATASTORE_ID, BIGQUERY_DATASET, and BIGQUERY_TABLE are set in .env"
            )

        # Configure client options if not using the 'global' location
        client_options = (
            ClientOptions(
                api_endpoint=f"{location}-discoveryengine.googleapis.com"
            )
            if location != "global"
            else None
        )

        # Create a client
        client = discoveryengine.DocumentServiceClient(
            client_options=client_options
        )

        # The full resource name of the search engine branch
        parent = client.branch_path(
            project=project_id,
            location=location,
            data_store=data_store_id,
            branch="default_branch",
        )

        # Create the import request
        request = discoveryengine.ImportDocumentsRequest(
            parent=parent,
            bigquery_source=discoveryengine.BigQuerySource(
                project_id=project_id,
                dataset_id=bigquery_dataset,
                table_id=bigquery_table,
                data_schema="custom",
            ),
            # Using FULL and auto_generate_ids because the DDL lacks a primary key ID column
            reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.FULL,
            auto_generate_ids=True,
        )

        logging.info(
            f"Triggering document import for Datastore: {data_store_id}..."
        )

        # Make the request
        operation = client.import_documents(request=request)

        logging.info(
            f"Waiting for Vertex AI indexing operation to complete: {operation.operation.name}"
        )

        # This will block until the operation is fully complete (or fails)
        operation.result()

        # After the operation is complete, get information from operation metadata
        metadata = discoveryengine.ImportDocumentsMetadata(operation.metadata)

        logging.info(
            f"Datastore refresh complete! "
            f"Successfully indexed: {metadata.success_count} documents. "
            f"Failed: {metadata.failure_count} documents."
        )
        return True

    except Exception as e:
        logging.error(f"Failed to refresh Vertex Datastore: {e}")
        return False


# For standalone testing
# if __name__ == '__main__':
#     project_id = os.getenv("PROJECT_ID", "your-project-id")
#     refresh_vertex_datastore(project_id)
