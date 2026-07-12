"""
This module provides tools for interacting with Google Cloud Dataproc.
"""

import logging
import subprocess
from datetime import datetime, timedelta

from google.api_core import exceptions as google_exceptions
from google.cloud import logging_v2

from .utils import _process_log_iterator

logger = logging.getLogger("plumber-agent")


def _handle_log_report(
    collected_logs: list[str], message_prefix: str = "Fetched log entries"
) -> dict[str, str]:
    """
    Helper function to generate a consistent log report.
    """
    log_count = len(collected_logs)
    if log_count == 0:
        logger.info("No log entries found matching the criteria.")
        return {
            "status": "success",
            "report": "No log entries found matching the criteria.",
        }

    logger.info("\nSuccessfully fetched %s log entries.", log_count)
    return {
        "status": "success",
        "report": f"{message_prefix}:\n" + "\n".join(collected_logs),
    }


def get_dataproc_cluster_logs_with_name(
    project_id: str,
    dataproc_cluster_name: str = "",
    _limit: int = 10,
) -> dict[str, str]:
    """
    Fetches Google Cloud Logging log entries specifically for Dataproc clusters.

    This function retrieves log entries from Google Cloud Logging for a specified
    Dataproc cluster with it's name, with an option to limit the number of results.
    It is designed to provide an overview of recent cluster activities and potential issues.

    Args:
        project_id (str, required): The Google Cloud project ID. from which logs are to be fetched.
        dataproc_cluster_name (str, required): The name of the Dataproc cluster
            to filter logs for. If not provided don't call this tool.
        _limit (int, optional): The maximum number of log entries to retrieve.
            The function will fetch up to this many entries, Defaults to 10.

    Returns:
        dict[str, str]: A dictionary containing the status of the operation and the retrieved logs.
            The dictionary will have the following structure:
                - "status" (str): "success" if logs were fetched successfully, or "error"
                    if an error occurred.
                - "report" (str): A human-readable message detailing the outcome.
                    If successful, it includes a summary and a list of the fetched log entries.
                    If no logs are found, it indicates that.
                - "message" (str, optional): Only present if "status" is "error", providing
                    details about the error that occurred.

    Note: [IMPORTANT]
        - Call this tool only when user want's logs of a cluster with the name
    """
    try:
        collected_logs = get_dataproc_logs(
            project_id=project_id,
            resource_name="cloud_dataproc_cluster",
            query_str=dataproc_cluster_name,
            filter_key="cluster_name",
            _limit=_limit,
        )
        return _handle_log_report(collected_logs)

    except google_exceptions.GoogleAPIError as e:
        logger.error("A Google Cloud API error occurred: %s", e, exc_info=True)
        return {
            "status": "error",
            "report": f"A Google Cloud API error occurred: {e}",
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("An unexpected error occurred: %s", e, exc_info=True)
        return {
            "status": "error",
            "report": f"An unexpected error occurred: {e}",
        }


def get_dataproc_cluster_logs_with_id(
    project_id: str,
    dataproc_cluster_id: str,
    _limit: int = 10,
) -> dict[str, str]:
    """
    Fetches Google Cloud Logging log entries for a specific Dataproc cluster using its UUID or ID.

    This function retrieves log entries from Google Cloud Logging that are associated
    with a particular Dataproc cluster, identified by its unique ID (UUID). It allows
    for limiting the number of log entries returned, ordered by the most recent first.

    Args:
        project_id (str, required): The Google Cloud project ID. from which logs are to be fetched.
        cluster_id (str, required): The **UUID** of the Dataproc cluster to filter logs for.
            This uniquely identifies a Dataproc cluster across its lifecycle. If not
            provided don't call this tool.
            - format of UUID : 0278aa3c-085a-4ccc-b79d-78b82fbb2ba3
        _limit (int, optional): The maximum number of log entries to fetch. The function
            will retrieve up to this many entries, Defaults to 10.

    Returns:
        dict: A dictionary containing the status of the operation and the retrieved logs.
            The dictionary will have the following keys:
            - "status" (str): "success" if the logs were fetched successfully, or "error"
                if an issue occurred.
            - "report" (str): A descriptive message about the outcome. If successful,
                it includes a summary and a list of the fetched log entries. If no
                matching logs are found, it indicates that.
            - "message" (str, optional): Present only if "status" is "error", providing
                details about the specific error.

    Note: [IMPORTANT]
            - Call this tool only when user want's logs of a cluster with it's UUID or ID
    """
    try:
        collected_logs = get_dataproc_logs(
            project_id=project_id,
            resource_name="cloud_dataproc_cluster",
            query_str=dataproc_cluster_id,
            filter_key="cluster_uuid",
            _limit=_limit,
        )
        return _handle_log_report(collected_logs)

    except google_exceptions.GoogleAPIError as e:
        logger.error("A Google Cloud API error occurred: %s", e, exc_info=True)
        return {
            "status": "error",
            "report": f"A Google Cloud API error occurred: {e}",
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("An unexpected error occurred: %s", e, exc_info=True)
        return {
            "status": "error",
            "report": f"An unexpected error occurred: {e}",
        }


def get_dataproc_logs(
    project_id: str,
    resource_name: str = "",
    query_str: str = "",
    filter_key: str = "",
    _limit: int = 10,
) -> list[str]:
    """
    Fetches raw log entries from Google Cloud Logging for Dataproc resources.

    This is a common helper function that constructs a filter string to query
    Google Cloud Logging, primarily targeting Dataproc resources. It is used
    to retrieve logs based on a specific resource label (e.g., cluster UUID or name).

    Logs are filtered to include entries from the last 90 days, ordered by the
    most recent first, and limited by a maximum result count.

    Args:
        project_id (str, required): The Google Cloud project ID from which logs are to be fetched.
        resource_name (str, optional): The `resource.type` to filter logs by (e.g.,
                                       "cloud_dataproc_cluster"). Defaults to an empty string,
                                       but typically set to "cloud_dataproc_cluster"
                                       by the calling function.
        query_str (str, optional): The specific value to match in the log filter (e.g.,
                                    a cluster UUID). Defaults to an empty string.
        filter_key (str, optional): The `resource.labels` key to apply the filter to (e.g.,
                                    "cluster_uuid" or "cluster_name"). Defaults to an empty string.
        _limit (int, optional): The maximum number of log entries to fetch. Defaults to 10.

    Returns:
        list[str]: A list of formatted strings, where each string represents a retrieved log entry.
                   Returns an empty list if an exception occurs during the fetch.

    Raises:
        Exception: Catches and logs any errors encountered during the Google Cloud Logging API call.
                   The error is printed and an empty list is returned.
    """
    full_filter = (
        f'resource.type="{resource_name}" AND '
        f'resource.labels.{filter_key}="{query_str.strip()}" AND '
        f'timestamp >= "{(datetime.now() - timedelta(days=90)).isoformat()}Z"'
    )

    collected_logs = []

    try:
        client = logging_v2.Client(project=project_id)
        project_path = f"projects/{project_id}"
        iterator = client.list_entries(
            resource_names=[project_path],
            order_by="timestamp desc",
            filter_=full_filter,
            max_results=_limit,
        )

        collected_logs = _process_log_iterator(iterator, _limit)
        return collected_logs

    except google_exceptions.GoogleAPIError as e:
        logger.error("A Google Cloud API error occurred: %s", e, exc_info=True)
        return []
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("An unexpected error occurred: %s", e, exc_info=True)
        return []


def get_dataproc_normal_job_logs_with_id(
    project_id: str, region: str, job_id: str, filter_key: str
) -> subprocess.CompletedProcess | list[str]:
    """
    Executes a gcloud command to fetch and wait for (or stream) logs for a traditional
    Dataproc Job or Batch, returning the raw command result.

    This function constructs and executes a `gcloud dataproc {filter_key} wait {job_id}`
    command.No log entries found matching the criteria

    Args:
        project_id (str, required): The Google Cloud project ID.
        region (str, required): The GCP region where the Dataproc job resides (e.g., 'us-central1').
        job_id (str, required): The unique ID of the Dataproc job (e.g., 'pyspark_job-xyz').
        filter_key (str, required): The `gcloud dataproc` subcommand part ('jobs' or 'batches').

    Returns:
        subprocess.CompletedProcess | list[str]:
            - If successful or command ran, returns the raw `subprocess.CompletedProcess` object.
            - If `gcloud` is not found or another execution error occurs, returns an empty list.

    Note:
        This function uses `shell=True` for simplicity, which is less secure.
    """

    command = [
        "gcloud",
        "dataproc",
        filter_key,
        "wait",
        job_id,
        "--project",
        project_id,
        "--region",
        region,
    ]

    try:
        # Execute the command and capture the output.
        result = subprocess.run(
            command_str, capture_output=True, text=True, check=False, shell=False
        )
        logger.info(
            "Result for '%s' command: returncode=%s",
            filter_key,
            result.returncode,
        )
        return result
    except FileNotFoundError as e:
        logger.warning(
            "gcloud command not found. Please ensure it is installed and in your system's PATH. %s",
            e,
            exc_info=True,
        )
        return []
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "An error occurred while calling command: %s", e, exc_info=True
        )
        return []


# --- Main Tool Function (Orchestrates the log retrieval) ---


def get_dataproc_job_logs_with_id(
    project_id: str, region: str, job_id: str
) -> dict[str, str]:
    """
    Retrieves Dataproc job logs by executing the 'gcloud dataproc wait' command,
    attempting 'batches' first, then falling back to 'jobs'.

    Args:
        project_id (str, required): The Google Cloud project ID.
        region (str): The GCP region where the Dataproc job was executed.
        job_id (str, required): The unique identifier of the Dataproc job.

    Returns:
        dict[str, str]: A dictionary containing the status, report, and raw command output details.
    """

    if not job_id or not project_id or not region:
        return {
            "status": "error",
            "message": "Missing required arguments (project_id, region, or job_id).",
        }

    result = None
    command_type = ""

    try:
        # 1. Try as Dataproc Batch
        result = get_dataproc_normal_job_logs_with_id(
            project_id, region, job_id, "batches"
        )
        command_type = "Batch"

        # Check if batches failed and contained "No such batch"
        # gcloud returns non-zero code if not found or failed, but we only fall back
        # if we suspect "not found". We assume a failure to find the batch results
        # in a non-zero code and a message in stderr.
        if (
            isinstance(result, subprocess.CompletedProcess)
            and result.returncode != 0
            and "Not found" in result.stderr
        ):
            # 2. Fallback: Try as traditional Dataproc Job
            result = get_dataproc_normal_job_logs_with_id(
                project_id, region, job_id, "jobs"
            )
            command_type = "Job"

        # Handle core command execution failures (e.g., gcloud not found)
        if isinstance(result, list):
            return {
                "status": "error",
                "message": "Failed to execute 'gcloud' command. Check system PATH or installation.",
            }

        # 3. Final Status Reporting
        if result.returncode == 0:
            # Success: Command ran and job completed successfully.
            report = (
                f"Successfully waited for Dataproc {command_type} **{job_id}** "
                "to complete (gcloud exit code 0).\n\n"
                f"--- Log Output (stdout) ---\n{result.stdout}\n\n"
                f"--- Error/Warning Output (stderr) ---\n{result.stderr}"
            )
            return {"status": "success", "report": report}
        # Failure: Command or job execution failed (gcloud exit code is non-zero).
        report = (
            f"Dataproc {command_type} **{job_id}** failed, was not found, or the "
            f"'gcloud wait' command timed out (gcloud exit code {result.returncode}).\n\n"
            f"--- Console Output (stdout) ---\n{result.stdout}\n\n"
            f"--- Error/Diagnostic Output (stderr) ---\n{result.stderr}"
        )
        # Report as error because the command failed to exit 0
        return {
            "status": "error",
            "message": f"gcloud command failed with exit code {result.returncode}.",
            "report": report,
        }

    except google_exceptions.GoogleAPIError as e:
        logger.error("A Google Cloud API error occurred: %s", e, exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to get logs due to a Google Cloud API error: {e}",
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("An unexpected error occurred: %s", e, exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to get logs due to an unexpected error: {e}",
        }
