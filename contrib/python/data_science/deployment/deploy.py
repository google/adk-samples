# Copyright 2025 Google LLC
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

"""Deployment script for Data Science agent."""

import logging
import os

import vertexai
from absl import app, flags
from data_science.agent import root_agent
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions
from google.cloud import storage
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

FLAGS = flags.FLAGS
flags.DEFINE_string("project_id", None, "GCP project ID.")
flags.DEFINE_string("location", None, "GCP location.")
flags.DEFINE_string(
    "bucket", None, "GCP bucket name (without gs:// prefix)."
)  # Changed flag description
flags.DEFINE_string("resource_id", None, "ReasoningEngine resource ID.")

flags.DEFINE_bool("create", False, "Create a new agent.")
flags.DEFINE_bool("delete", False, "Delete an existing agent.")
flags.mark_bool_flags_as_mutual_exclusive(["create", "delete"])

AGENT_WHL_FILE = "data_science-0.1-py3-none-any.whl"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_staging_bucket(
    project_id: str, location: str, bucket_name: str
) -> str:
    """
    Checks if the staging bucket exists, creates it if not.

    Args:
        project_id: The GCP project ID.
        location: The GCP location for the bucket.
        bucket_name: The desired name for the bucket (without gs:// prefix).

    Returns:
        The full bucket path (gs://<bucket_name>).

    Raises:
        google_exceptions.GoogleCloudError: If bucket creation fails.
    """
    storage_client = storage.Client(project=project_id)
    try:
        # Check if the bucket exists
        bucket = storage_client.lookup_bucket(bucket_name)
        if bucket:
            logger.info("Staging bucket gs://%s already exists.", bucket_name)
        else:
            logger.info(
                "Staging bucket gs://%s not found. Creating...", bucket_name
            )
            # Create the bucket if it doesn't exist
            new_bucket = storage_client.create_bucket(
                bucket_name, project=project_id, location=location
            )
            logger.info(
                "Successfully created staging bucket gs://%s in %s.",
                new_bucket.name,
                location,
            )
            # Enable uniform bucket-level access for simplicity
            new_bucket.iam_configuration.uniform_bucket_level_access_enabled = (
                True
            )
            new_bucket.patch()
            logger.info(
                "Enabled uniform bucket-level access for gs://%s.",
                new_bucket.name,
            )

    except google_exceptions.Forbidden as e:
        logger.error(
            (
                "Permission denied error for bucket gs://%s. "
                "Ensure the service account has 'Storage Admin' role. Error: %s"
            ),
            bucket_name,
            e,
        )
        raise
    except google_exceptions.Conflict as e:
        logger.warning(
            (
                "Bucket gs://%s likely already exists but owned by another "
                "project or recently deleted. Error: %s"
            ),
            bucket_name,
            e,
        )
        # Assuming we can proceed if it exists, even with a conflict warning
    except google_exceptions.ClientError as e:
        logger.error(
            "Failed to create or access bucket gs://%s. Error: %s",
            bucket_name,
            e,
        )
        raise

    return f"gs://{bucket_name}"


def create(env_vars: dict[str, str]) -> None:
    """Creates and deploys the agent."""
    adk_app = AdkApp(
        agent=root_agent,
        enable_tracing=False,
    )

    if not os.path.exists(AGENT_WHL_FILE):
        logger.error("Agent wheel file not found at: %s", AGENT_WHL_FILE)
        # Consider adding instructions here on how to build the wheel file
        raise FileNotFoundError(f"Agent wheel file not found: {AGENT_WHL_FILE}")

    logger.info("Using agent wheel file: %s", AGENT_WHL_FILE)

    remote_agent = agent_engines.create(
        adk_app,
        requirements=[AGENT_WHL_FILE],
        extra_packages=[AGENT_WHL_FILE],
        env_vars=env_vars,
    )
    logger.info("Created remote agent: %s", remote_agent.resource_name)
    print(f"\nSuccessfully created agent: {remote_agent.resource_name}")


def delete(resource_id: str) -> None:
    """Deletes the specified agent."""
    logger.info("Attempting to delete agent: %s", resource_id)
    try:
        remote_agent = agent_engines.get(resource_id)
        remote_agent.delete(force=True)
        logger.info("Successfully deleted remote agent: %s", resource_id)
        print(f"\nSuccessfully deleted agent: {resource_id}")
    except google_exceptions.NotFound:
        logger.error("Agent with resource ID %s not found.", resource_id)
        print(f"\nAgent not found: {resource_id}")
    except Exception as e:
        logger.error(
            "An error occurred while deleting agent %s: %s", resource_id, e
        )
        print(f"\nError deleting agent {resource_id}: {e}")


def main(argv: list[str]) -> None:  # pylint: disable=unused-argument
    """Main execution function."""
    load_dotenv()
    env_vars = {}

    project_id = (
        FLAGS.project_id
        if FLAGS.project_id
        else os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    location = (
        FLAGS.location if FLAGS.location else os.getenv("GOOGLE_CLOUD_LOCATION")
    )
    # Default bucket name convention if not provided
    default_bucket_name = f"{project_id}-adk-staging" if project_id else None
    bucket_name = (
        FLAGS.bucket
        if FLAGS.bucket
        else os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET", default_bucket_name)
    )
    # Don't set "GOOGLE_CLOUD_PROJECT" or "GOOGLE_CLOUD_LOCATION"
    # when deploying to Agent Engine. Those are set by the backend.
    # Collect environment variables, filtering out None values and empty strings
    env_var_keys = [
        "ROOT_AGENT_MODEL",
        "ANALYTICS_AGENT_MODEL",
        "BASELINE_NL2SQL_MODEL",
        "BIGQUERY_AGENT_MODEL",
        "BQML_AGENT_MODEL",
        "CHASE_NL2SQL_MODEL",
        "BQ_DATASET_ID",
        "BQ_DATA_PROJECT_ID",
        "BQ_COMPUTE_PROJECT_ID",
        "BQML_RAG_CORPUS_NAME",
        "CODE_INTERPRETER_EXTENSION_NAME",
        "NL2SQL_METHOD",
    ]

    skipped_vars: list[str] = []
    for key in env_var_keys:
        value = os.getenv(key)
        # Only add to env_vars if the value is not None/empty and not just
        # whitespace
        if value and value.strip():
            env_vars[key] = value
        else:
            skipped_vars.append(key)

    if skipped_vars:
        logger.info(
            "Skipped empty/None/whitespace environment variables: %s",
            skipped_vars,
        )

    logger.info(
        "Environment variables to be passed to agent: %s", list(env_vars.keys())
    )

    logger.info("Using PROJECT: %s", project_id)
    logger.info("Using LOCATION: %s", location)
    logger.info("Using BUCKET NAME: %s", bucket_name)

    # --- Input Validation ---
    if not project_id:
        raise app.UsageError(
            "Missing required GCP Project ID. Set GOOGLE_CLOUD_PROJECT or "
            "use --project_id flag."
        )
    if not location:
        raise app.UsageError(
            "Missing required GCP Location. Set GOOGLE_CLOUD_LOCATION or use "
            "--location flag."
        )
    if not bucket_name:
        raise app.UsageError(
            "Missing required GCS Bucket Name. Set GOOGLE_CLOUD_STORAGE_BUCKET "
            "or use --bucket flag."
        )
    if not FLAGS.create and not FLAGS.delete:
        raise app.UsageError(
            "You must specify either --create or --delete flag."
        )
    if FLAGS.delete and not FLAGS.resource_id:
        raise app.UsageError(
            "--resource_id is required when using the --delete flag."
        )
    # --- End Input Validation ---

    try:
        # Setup staging bucket
        staging_bucket_uri = None
        if FLAGS.create:
            staging_bucket_uri = setup_staging_bucket(
                project_id, location, bucket_name
            )

        # Initialize Vertex AI *after* bucket setup and validation
        vertexai.init(
            project=project_id,
            location=location,
            staging_bucket=staging_bucket_uri,
        )

        if FLAGS.create:
            create(env_vars)
        elif FLAGS.delete:
            delete(FLAGS.resource_id)

    except google_exceptions.Forbidden as e:
        print(
            "Permission Error: Ensure the service account/user has necessary "
            "permissions (e.g., Storage Admin, Vertex AI User)."
            f"\nDetails: {e}"
        )
    except FileNotFoundError as e:
        print(f"\nFile Error: {e}")
        print(
            "Please ensure the agent wheel file exists in the 'deployment' "
            "directory and you have run the build script "
            "(e.g., poetry build --format=wheel --output=deployment')."
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # Log the full traceback for debugging purposes
        logger.exception("Unhandled exception in main:")


if __name__ == "__main__":
    app.run(main)
