#!/usr/bin/env python3
"""Clean up all GCP resources created by the product search agent.

Deletes BigQuery datasets, GCS buckets, Vector Search indexes,
Pub/Sub topics, Cloud Functions, and Cloud Run services.

Usage:
    # Dry run (show what would be deleted, don't delete)
    python scripts/cleanup.py --config design-spec.md --dry-run

    # Delete everything
    python scripts/cleanup.py --config design-spec.md --confirm

    # Delete only specific resources
    python scripts/cleanup.py --config design-spec.md --confirm --only bigquery,gcs
"""

import argparse
import logging
import sys
from pathlib import Path

from google.cloud import bigquery, storage

# Add _shared to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "_shared"))
from setup_utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_RESOURCE_TYPES = ["bigquery", "gcs", "vectorsearch", "pubsub", "cloudrun", "cloudfunctions"]


def delete_bigquery(project_id: str, dataset_id: str, dry_run: bool) -> None:
    """Delete BigQuery dataset and all tables."""
    client = bigquery.Client(project=project_id)
    dataset_ref = f"{project_id}.{dataset_id}"

    try:
        client.get_dataset(dataset_ref)
    except Exception:
        logger.info(f"BigQuery dataset {dataset_ref} does not exist, skipping")
        return

    if dry_run:
        logger.info(f"[DRY RUN] Would delete BigQuery dataset: {dataset_ref} (and all tables)")
        return

    client.delete_dataset(dataset_ref, delete_contents=True, not_found_ok=True)
    logger.info(f"Deleted BigQuery dataset: {dataset_ref}")


def delete_gcs(project_id: str, bucket_prefixes: list[str], dry_run: bool) -> None:
    """Delete GCS buckets matching project prefixes."""
    client = storage.Client(project=project_id)

    for prefix in bucket_prefixes:
        bucket_name = f"{project_id}-{prefix}"
        try:
            bucket = client.get_bucket(bucket_name)
        except Exception:
            logger.info(f"GCS bucket {bucket_name} does not exist, skipping")
            continue

        if dry_run:
            logger.info(f"[DRY RUN] Would delete GCS bucket: {bucket_name}")
            continue

        # Delete all objects first
        blobs = list(bucket.list_blobs())
        if blobs:
            bucket.delete_blobs(blobs)
            logger.info(f"Deleted {len(blobs)} objects from {bucket_name}")

        bucket.delete()
        logger.info(f"Deleted GCS bucket: {bucket_name}")


def delete_vectorsearch_collection(project_id: str, location: str, collection_id: str, dry_run: bool) -> None:
    """Delete Vector Search 2.0 Collection."""
    from google.cloud import vectorsearch
    from google.api_core import exceptions

    client = vectorsearch.VectorSearchServiceClient()
    collection_name = f"projects/{project_id}/locations/{location}/collections/{collection_id}"

    try:
        client.get_collection(
            request=vectorsearch.GetCollectionRequest(name=collection_name)
        )
    except exceptions.NotFound:
        logger.info(f"Vector Search collection {collection_id} does not exist, skipping")
        return

    if dry_run:
        logger.info(f"[DRY RUN] Would delete Vector Search collection: {collection_name}")
        return

    try:
        operation = client.delete_collection(
            request=vectorsearch.DeleteCollectionRequest(name=collection_name)
        )
        operation.result()
        logger.info(f"Deleted Vector Search collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to delete collection {collection_id}: {e}")


def delete_vectorsearch_v1_index(project_id: str, location: str, index_name: str, dry_run: bool) -> None:
    """Delete legacy Vector Search v1 MatchingEngineIndex (if any remain)."""
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)

    indexes = aiplatform.MatchingEngineIndex.list(
        filter=f'display_name="{index_name}"'
    )

    if not indexes:
        return

    for index in indexes:
        if dry_run:
            logger.info(f"[DRY RUN] Would delete legacy Vector Search index: {index.resource_name}")
            continue

        index.delete()
        logger.info(f"Deleted legacy Vector Search index: {index.resource_name}")


def delete_pubsub(project_id: str, topic_name: str, dry_run: bool) -> None:
    """Delete Pub/Sub topic and subscriptions."""
    from google.cloud import pubsub_v1

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_name)

    try:
        publisher.get_topic(request={"topic": topic_path})
    except Exception:
        logger.info(f"Pub/Sub topic {topic_name} does not exist, skipping")
        return

    if dry_run:
        logger.info(f"[DRY RUN] Would delete Pub/Sub topic: {topic_path}")
        return

    # Delete subscriptions first
    subscriber = pubsub_v1.SubscriberClient()
    for sub in publisher.list_topic_subscriptions(request={"topic": topic_path}):
        subscriber.delete_subscription(request={"subscription": sub})
        logger.info(f"Deleted subscription: {sub}")

    publisher.delete_topic(request={"topic": topic_path})
    logger.info(f"Deleted Pub/Sub topic: {topic_path}")


def delete_cloudrun(project_id: str, location: str, service_name: str, dry_run: bool) -> None:
    """Delete Cloud Run service."""
    import subprocess

    result = subprocess.run(
        ["gcloud", "run", "services", "describe", service_name,
         "--region", location, "--project", project_id, "--format", "value(name)"],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        logger.info(f"Cloud Run service {service_name} does not exist, skipping")
        return

    if dry_run:
        logger.info(f"[DRY RUN] Would delete Cloud Run service: {service_name}")
        return

    subprocess.run(
        ["gcloud", "run", "services", "delete", service_name,
         "--region", location, "--project", project_id, "--quiet"],
        check=True,
    )
    logger.info(f"Deleted Cloud Run service: {service_name}")


def delete_cloudfunctions(project_id: str, location: str, function_name: str, dry_run: bool) -> None:
    """Delete Cloud Function."""
    import subprocess

    result = subprocess.run(
        ["gcloud", "functions", "describe", function_name,
         "--region", location, "--project", project_id, "--gen2", "--format", "value(name)"],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        logger.info(f"Cloud Function {function_name} does not exist, skipping")
        return

    if dry_run:
        logger.info(f"[DRY RUN] Would delete Cloud Function: {function_name}")
        return

    subprocess.run(
        ["gcloud", "functions", "delete", function_name,
         "--region", location, "--project", project_id, "--gen2", "--quiet"],
        check=True,
    )
    logger.info(f"Deleted Cloud Function: {function_name}")


def cleanup(config: dict, dry_run: bool, only: list[str]) -> None:
    """Run cleanup for all or selected resource types."""
    project_id = config.get("gcp_project_id", "")
    if not project_id:
        logger.error("gcp_project_id not set in config")
        sys.exit(1)

    location = config.get("gcp_region", "us-central1")
    project_name = config.get("project_name", "product-search")

    mode = "[DRY RUN] " if dry_run else ""
    logger.info(f"{mode}Cleaning up resources for project: {project_id}")
    logger.info(f"{mode}Resource types: {', '.join(only)}")
    logger.info("")

    if "bigquery" in only:
        delete_bigquery(project_id, "retail_skill_products", dry_run)

    if "gcs" in only:
        delete_gcs(project_id, ["retail-skill-products", "retail-skill-embeddings", "retail-skill-images"], dry_run)

    if "vectorsearch" in only:
        delete_vectorsearch_collection(project_id, location, "retail-skill-products-collection", dry_run)
        # Also clean up legacy v1 indexes if any remain from previous runs
        delete_vectorsearch_v1_index(project_id, location, "retail_skill_products_index", dry_run)

    if "pubsub" in only:
        delete_pubsub(project_id, "retail-skill-product-changes", dry_run)

    if "cloudrun" in only:
        delete_cloudrun(project_id, location, project_name, dry_run)

    if "cloudfunctions" in only:
        delete_cloudfunctions(project_id, location, "retail-skill-pubsub-sync", dry_run)

    logger.info("")
    if dry_run:
        logger.info("Dry run complete. No resources were deleted.")
        logger.info("Run with --confirm to actually delete.")
    else:
        logger.info("Cleanup complete. All selected resources deleted.")


def main():
    parser = argparse.ArgumentParser(description="Clean up all GCP resources created by the product search agent")
    parser.add_argument("--config", required=True, help="Path to design-spec.md")
    parser.add_argument("--confirm", action="store_true", help="Actually delete resources (without this flag, runs in dry-run mode)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--only", default="", help=f"Comma-separated resource types to delete: {','.join(ALL_RESOURCE_TYPES)}")

    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        logger.error(f"Config file not found or empty: {args.config}")
        sys.exit(1)

    dry_run = not args.confirm or args.dry_run

    if args.only:
        only = [r.strip() for r in args.only.split(",")]
        invalid = [r for r in only if r not in ALL_RESOURCE_TYPES]
        if invalid:
            parser.error(f"Invalid resource types: {invalid}. Valid: {ALL_RESOURCE_TYPES}")
    else:
        only = ALL_RESOURCE_TYPES

    if not dry_run:
        project_id = config.get("gcp_project_id", "unknown")
        print(f"\nYou are about to permanently delete GCP resources in project: {project_id}")
        print(f"Resource types: {', '.join(only)}")
        try:
            answer = input("\nAre you sure? Type 'yes' to confirm: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer != "yes":
            print("Aborted. No resources were deleted.")
            sys.exit(0)

    cleanup(config, dry_run, only)


if __name__ == "__main__":
    main()
