"""Database utilities for fetching secrets and connection strings."""

import os

from google.cloud import secretmanager


def get_secret(secret_id: str, version_id: str = "latest") -> str:
    """Fetches a secret from Google Cloud Secret Manager."""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT environment variable is not set. Cannot fetch secret."
        )

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def get_db_connection_string() -> str:
    """Returns the database connection string from Secret Manager."""
    # The user specified the secret name is "hazmat-db-connection-string"
    db_string = get_secret("hazmat-db-connection-string")

    # If running on Cloud Run with Cloud SQL integration, use Unix socket
    cloudsql_instance = os.environ.get("CLOUDSQL_INSTANCE")
    if cloudsql_instance:
        # Replace the host:port with empty string and add host parameter
        import re

        # Match postgresql://user:pass@host:port/db
        match = re.match(r"(postgresql://[^@]+@)[^/]+(/.*)", db_string)
        if match:
            base, path = match.groups()
            db_string = f"{base}{path}?host=/cloudsql/{cloudsql_instance}"

    return db_string
