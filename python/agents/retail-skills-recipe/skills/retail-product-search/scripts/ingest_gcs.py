#!/usr/bin/env python3
"""Upload product images to Google Cloud Storage.

Supports uploading from local directories or from a file of image URLs.
Produces an image_manifest.json mapping product IDs to GCS URIs.

Usage:
    # From local directory
    python ingest_gcs.py \
        --project-id my-project \
        --bucket-name my-project-images \
        --source-type local \
        --source-path ./product_images/

    # From URL file
    python ingest_gcs.py \
        --project-id my-project \
        --bucket-name my-project-images \
        --source-type urls \
        --url-file image_urls.txt

    # Or use design-spec.md for defaults:
    python ingest_gcs.py --config design-spec.md --source-type local --source-path ./images/
"""

import argparse
import json
import logging
import mimetypes
import re
import sys
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import requests
from google.cloud import storage

# Add _shared to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "_shared"))
from setup_utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def extract_product_id(filename: str) -> str:
    """Extract product_id from image filename.

    Expects: prod-001.jpg, prod-001_1.jpg, prod-001_front.png
    """
    stem = Path(filename).stem
    match = re.match(r"^(.+?)(?:_\d+|_[a-zA-Z]+)?$", stem)
    return match.group(1) if match else stem


def upload_from_local(
    bucket: storage.Bucket,
    local_path: Path,
    prefix: str = "products",
) -> Dict[str, List[str]]:
    """Upload images from local directory to GCS."""
    manifest: Dict[str, List[str]] = {}

    for image_file in sorted(local_path.rglob("*")):
        if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        product_id = extract_product_id(image_file.name)
        relative = image_file.relative_to(local_path)
        dest = f"{prefix}/{relative}"

        bucket.blob(dest).upload_from_filename(str(image_file))
        gcs_uri = f"gs://{bucket.name}/{dest}"
        manifest.setdefault(product_id, []).append(gcs_uri)
        logger.info(f"Uploaded {image_file.name} -> {gcs_uri}")

    return manifest


def upload_from_urls(
    bucket: storage.Bucket,
    url_entries: List[Dict[str, str]],
    prefix: str = "products",
) -> Dict[str, List[str]]:
    """Download images from URLs and upload to GCS."""
    manifest: Dict[str, List[str]] = {}

    for entry in url_entries:
        url = entry["url"]
        product_id = entry.get("product_id", "")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            filename = Path(urlparse(url).path).name or f"image_{len(manifest)}.jpg"
            if not product_id:
                product_id = extract_product_id(filename)

            content_type = response.headers.get("Content-Type") or mimetypes.guess_type(filename)[0]
            dest = f"{prefix}/{product_id}/{filename}"
            bucket.blob(dest).upload_from_string(response.content, content_type=content_type)

            gcs_uri = f"gs://{bucket.name}/{dest}"
            manifest.setdefault(product_id, []).append(gcs_uri)
            logger.info(f"Uploaded {url} -> {gcs_uri}")

        except Exception as e:
            logger.error(f"Failed to upload {url}: {e}")

    return manifest


def parse_url_file(path: Path) -> List[Dict[str, str]]:
    """Parse URL file (plain URLs or JSONL with product_id + url)."""
    entries = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            entry = json.loads(line)
            entries.append({"product_id": entry.get("product_id", ""), "url": entry["url"]})
        else:
            entries.append({"product_id": "", "url": line})
    return entries


def write_manifest(manifest: Dict[str, List[str]], output_path: Path) -> None:
    """Write image_manifest.json."""
    data = [
        {
            "product_id": pid,
            "image_uris": uris,
            "primary_image": uris[0],
            "image_count": len(uris),
        }
        for pid, uris in sorted(manifest.items())
    ]
    output_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Wrote manifest: {output_path} ({len(data)} products)")


def main():
    parser = argparse.ArgumentParser(description="Upload product images to GCS")
    parser.add_argument("--config", default="", help="Path to design-spec.md (provides defaults for other args)")
    parser.add_argument("--project-id", help="GCP project ID")
    parser.add_argument("--bucket-name", help="GCS bucket name")
    parser.add_argument("--source-type", choices=["local", "urls"], required=True)
    parser.add_argument("--source-path", help="Local directory (for source-type=local)")
    parser.add_argument("--url-file", help="File with image URLs (for source-type=urls)")
    parser.add_argument("--prefix", default="products", help="GCS path prefix")
    parser.add_argument("--manifest-output", default="image_manifest.json", help="Manifest output path")

    args = parser.parse_args()

    # Load design-spec.md defaults -- CLI args override config values
    if args.config:
        cfg = load_config(args.config)
        if not args.project_id:
            args.project_id = cfg.get("gcp_project_id", "")
        if not args.bucket_name:
            args.bucket_name = cfg.get("gcp_project_id", "") + "-images"

    if not args.project_id:
        parser.error("--project-id is required (or set gcp_project_id in design-spec.md)")
    if not args.bucket_name:
        parser.error("--bucket-name is required")

    client = storage.Client(project=args.project_id)
    try:
        bucket = client.get_bucket(args.bucket_name)
    except Exception:
        logger.info(f"Creating bucket: {args.bucket_name}")
        bucket = client.create_bucket(args.bucket_name, location="US")

    if args.source_type == "local":
        manifest = upload_from_local(bucket, Path(args.source_path), args.prefix)
    else:
        entries = parse_url_file(Path(args.url_file))
        manifest = upload_from_urls(bucket, entries, args.prefix)

    total = sum(len(uris) for uris in manifest.values())
    logger.info(f"Uploaded {total} images for {len(manifest)} products")

    write_manifest(manifest, Path(args.manifest_output))

    # Also upload manifest to GCS
    bucket.blob(f"{args.prefix}/image_manifest.json").upload_from_filename(args.manifest_output)


if __name__ == "__main__":
    main()
