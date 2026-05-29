# Copyright 2025 Google LLC
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

import argparse
import io
import logging
import sys
from typing import Dict, List, Tuple

from google.cloud import storage

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from intelligent_document_generation_agent.utils.config import settings
else:
    from .config import settings

from intelligent_document_generation_agent.utils.logging_setup import (  # noqa: E402
    setup_logging,
)

setup_logging()


def initialize_gcs_client() -> storage.Client:
    """
    Initializes and returns a Google Cloud Storage client.

    Relies on Application Default Credentials (ADC) for authentication.
    https://cloud.google.com/docs/authentication/provide-credentials-adc

    Returns:
        storage.Client: An authenticated GCS client instance.
    """
    logging.info("Initializing Google Cloud Storage client.")
    return storage.Client(project=settings.GOOGLE_CLOUD_PROJECT)


def get_file_as_csv_string(client: storage.Client, file_uri: str) -> str:
    """
    Downloads a file from a GCS URI and returns its content as a CSV formatted string.
    Handles both .csv and .xlsx file types.

    Args:
        client: An authenticated GCS storage client.
        file_uri: The GCS URI of the file to download (e.g., 'gs://bucket/file.csv').

    Returns:
        A string containing the file's content in CSV format.

    Raises:
        ValueError: If the file type is not .csv or .xlsx.
    """
    bucket_name, blob_name = file_uri.replace("gs://", "").split("/", 1)

    if file_uri.lower().endswith(".csv"):
        try:
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                raise FileNotFoundError(
                    f"Object not found at gs://{bucket_name}/{blob_name}"
                )

            csv_string = blob.download_as_text()
            logging.info(
                f"Successfully downloaded CSV from gs://{bucket_name}/{blob_name}"
            )
            return csv_string
        except Exception as e:
            logging.error(f"Failed to download or parse CSV from GCS: {e}")
            raise
    elif file_uri.lower().endswith(".xlsx"):
        try:
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                raise FileNotFoundError(
                    f"Object not found at gs://{bucket_name}/{blob_name}"
                )

            import pandas as pd

            xlsx_bytes = blob.download_as_bytes()
            df = pd.read_excel(io.BytesIO(xlsx_bytes))
            csv_string = df.to_csv(index=False)

            logging.info(
                f"Successfully downloaded and converted XLSX from gs://{bucket_name}/{blob_name}"
            )
            return csv_string
        except Exception as e:
            logging.error(f"Failed to download or parse XLSX from GCS: {e}")
            raise
    else:
        raise ValueError(
            f"Unsupported file type for URI: {file_uri}. Only .csv and .xlsx are supported."
        )


def list_gcs_file_uris(
    client: storage.Client, bucket_and_folder_name: str
) -> List[str]:
    """
    Lists the GCS URIs of all files in a given GCS bucket and folder.

    Args:
        client: An authenticated GCS client instance.
        bucket_and_folder_name: The name of the GCS bucket and folder, e.g., 'my-bucket/my-folder'.

    Returns:
        A list of GCS URIs (e.g., 'gs://bucket-name/folder-name/file-name') for all blobs.
    """
    bucket_name = bucket_and_folder_name.split("/")[0]
    folder_name = "/".join(bucket_and_folder_name.split("/")[1:])
    # Ensure the folder_name for the prefix doesn't end in a slash for listing,
    # but add it back for comparison to filter out the folder object itself.
    prefix = folder_name.rstrip("/")
    logging.info(f"Listing files in bucket {bucket_name} and folder {prefix}.")
    # Convert the iterator to a list once to allow multiple passes (e.g., for logging and then processing)
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    for blob in blobs:
        logging.info(f"Found file: {blob.name}")
    return [
        f"gs://{bucket_name}/{blob.name}"
        for blob in blobs
        if not blob.name.endswith("/")
    ]


def download_pdf_bytes_from_uri(
    client: storage.Client, gcs_uri: str
) -> Tuple[bytes, str]:
    """
    Downloads the bytes of a file from a GCS URI.

    Args:
        client: An authenticated GCS client instance.
        gcs_uri: The GCS URI of the file to download (e.g., 'gs://bucket/file').

    Returns:
        The content of the file as bytes.

    Raises:
        ValueError: If the GCS URI is malformed.
        google.cloud.exceptions.NotFound: If the blob does not exist.
    """
    logging.info(f"[Tool] Attempting to download PDF from URI: {gcs_uri}")
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Malformed GCS URI: {gcs_uri}. Must start with 'gs://'.")

    # Remove the 'gs://' prefix and split into bucket and blob name
    path_parts = gcs_uri[5:].split("/", 1)
    if len(path_parts) != 2:
        raise ValueError(
            f"Malformed GCS URI: {gcs_uri}. Must contain bucket and object name."
        )

    bucket_name, blob_name = path_parts
    folder_name = blob_name.split("/", 1)[0]
    doc_name = blob_name.split("/", 1)[1]
    logging.info(
        f"[Tool] Downloading {doc_name} from bucket {bucket_name} and folder_name {folder_name}."
    )
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes(), doc_name


def upload_json_to_gcs(
    client: storage.Client,
    bucket_name: str,
    destination_blob_name: str,
    json_string: str,
):
    """
    Uploads a JSON string to a GCS bucket.

    Args:
        client: An authenticated GCS client instance.
        bucket_name: The name of the GCS bucket.
        destination_blob_name: The name of the blob to create.
        json_string: The JSON content as a string.
    """
    try:
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_string(json_string, content_type="application/json")

        logging.info(
            f"Successfully uploaded JSON to gs://{bucket_name}/{destination_blob_name}"
        )
        return f"gs://{bucket_name}/{destination_blob_name}"
    except Exception as e:
        logging.error(f"Failed to upload JSON to GCS: {e}")
        raise


def upload_csv_to_gcs(
    client: storage.Client,
    bucket_name: str,
    destination_blob_name: str,
    csv_string: str,
):
    """
    Uploads a CSV string to a GCS bucket.

    Args:
        client: An authenticated GCS client instance.
        bucket_name: The name of the GCS bucket.
        destination_blob_name: The name of the blob to create.
        csv_string: The CSV content as a string.
    """
    try:
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_string(csv_string, content_type="text/csv")

        logging.info(
            f"Successfully uploaded CSV to gs://{bucket_name}/{destination_blob_name}"
        )
        return f"gs://{bucket_name}/{destination_blob_name}"
    except Exception as e:
        logging.error(f"Failed to upload CSV to GCS: {e}")
        raise


def download_json_from_gcs(
    client: storage.Client, bucket_name: str, blob_name: str
) -> Dict:
    """
    Downloads and parses a JSON file from a GCS bucket.

    Args:
        client: An authenticated GCS client instance.
        bucket_name: The name of the GCS bucket.
        blob_name: The name of the blob to download.

    Returns:
        The content of the file as a dictionary.
    """
    import json

    try:
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            raise FileNotFoundError(
                f"Object not found at gs://{bucket_name}/{blob_name}"
            )

        json_string = blob.download_as_text()
        logging.info(
            f"Successfully downloaded JSON from gs://{bucket_name}/{blob_name}"
        )
        return json.loads(json_string)
    except Exception as e:
        logging.error(f"Failed to download or parse JSON from GCS: {e}")
        raise


def download_xlsx_from_gcs(
    client: storage.Client, bucket_name: str, blob_name: str
) -> bytes:
    """
    Downloads an XLSX file from a GCS bucket.

    Args:
        client: An authenticated GCS client instance.
        bucket_name: The name of the GCS bucket.
        blob_name: The name of the blob to download.

    Returns:
        The content of the file as bytes.
    """
    try:
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            raise FileNotFoundError(
                f"Object not found at gs://{bucket_name}/{blob_name}"
            )

        xlsx_bytes = blob.download_as_bytes()
        logging.info(
            f"Successfully downloaded XLSX from gs://{bucket_name}/{blob_name}"
        )
        return xlsx_bytes
    except Exception as e:
        logging.error(f"Failed to download XLSX from GCS: {e}")
        raise


def download_text_from_gcs(gcs_uri: str, storage_client: storage.Client) -> str:
    """Downloads a text file from a GCS URI as a string."""
    logging.info(f"Downloading from {gcs_uri}...")
    try:
        if not gcs_uri.startswith("gs://"):
            raise ValueError("Invalid GCS URI. Must start with 'gs://'.")

        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        blob = storage_client.bucket(bucket_name).blob(blob_name)

        if not blob.exists():
            raise FileNotFoundError(f"Object not found at {gcs_uri}")

        return blob.download_as_text()
    except Exception as e:
        logging.error(f"Error downloading or parsing {gcs_uri}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Google Cloud Storage Utility CLI for listing files, downloading PDFs, or uploading JSON."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for listing files in a folder
    list_parser = subparsers.add_parser(
        "list-folder", help="List files in a GCS bucket directory."
    )
    list_parser.add_argument(
        "bucket_and_folder_name",
        default="project-source-documents/01_Insider_Protection",
        help="The bucket and folder name to be processed",
    )

    # Subparser for downloading a PDF file
    download_parser = subparsers.add_parser(
        "download-file", help="Download a PDF file from GCS."
    )
    download_parser.add_argument(
        "file_uri", help="The URI of the PDF file to download."
    )
    download_parser.add_argument(
        "--output",
        "-o",
        default="downloaded_file.pdf",
        help="Output path for the downloaded PDF file. Defaults to 'downloaded_file.pdf'.",
    )

    # Subparser for uploading a JSON string
    upload_parser = subparsers.add_parser(
        "upload-json", help="Upload a JSON string to a GCS bucket."
    )
    upload_parser.add_argument(
        "bucket_name", help="The name of the destination GCS bucket."
    )
    upload_parser.add_argument(
        "destination_blob_name",
        help="The full path for the destination blob (e.g., 'folder/file.json').",
    )
    upload_parser.add_argument(
        "json_string", help="The JSON content to upload, as a string."
    )

    # Subparser for downloading a CSV file
    download_csv_parser = subparsers.add_parser(
        "download-csv", help="Download a CSV file from GCS."
    )
    download_csv_parser.add_argument(
        "bucket_name", help="The name of the source GCS bucket."
    )
    download_csv_parser.add_argument(
        "blob_name", help="The full path for the source blob (e.g., 'folder/file.csv')."
    )
    download_csv_parser.add_argument(
        "--output",
        "-o",
        default="downloaded_file.csv",
        help="Output path for the downloaded CSV file. Defaults to 'downloaded_file.csv'.",
    )

    # Subparser for downloading a XLSX file
    download_xlsx_parser = subparsers.add_parser(
        "download-xlsx", help="Download a XLSX file from GCS."
    )
    download_xlsx_parser.add_argument(
        "bucket_name", help="The name of the source GCS bucket."
    )
    download_xlsx_parser.add_argument(
        "blob_name",
        help="The full path for the source blob (e.g., 'folder/file.xlsx').",
    )
    download_xlsx_parser.add_argument(
        "--output",
        "-o",
        default="downloaded_file.xlsx",
        help="Output path for the downloaded XLSX file. Defaults to 'downloaded_file.xlsx'.",
    )

    args = parser.parse_args()

    client = initialize_gcs_client()

    if args.command == "list-folder":
        file_uris = list_gcs_file_uris(client, args.bucket_and_folder_name)
        if file_uris:
            logging.info("Found file URIs:")
            for uri in file_uris:
                logging.info(uri)
    elif args.command == "download-file":
        logging.info(f"Attempting to download PDF from URI: {args.file_uri}")
        pdf_data_and_name = download_pdf_bytes_from_uri(client, args.file_uri)
        if pdf_data_and_name:
            pdf_bytes, file_name = pdf_data_and_name
            logging.info(f"Successfully downloaded '{file_name}'")
            logging.info(f"First 20 bytes: {pdf_bytes[:20]}")
        else:
            logging.info(f"Failed to download PDF from {args.file_uri}")
    elif args.command == "upload-json":
        logging.info(
            f"Attempting to upload JSON to gs://{args.bucket_name}/{args.destination_blob_name}"
        )
        try:
            gcs_uri = upload_json_to_gcs(
                client, args.bucket_name, args.destination_blob_name, args.json_string
            )
            logging.info(f"Successfully uploaded JSON. URI: {gcs_uri}")
        except Exception as e:
            logging.info(f"Failed to upload JSON: {e}")
    elif args.command == "download-xlsx":
        logging.info(
            f"Attempting to download XLSX from gs://{args.bucket_name}/{args.blob_name}"
        )
        try:
            xlsx_bytes = download_xlsx_from_gcs(
                client, args.bucket_name, args.blob_name
            )
            with open(args.output, "wb") as f:
                f.write(xlsx_bytes)
            logging.info(f"Successfully downloaded XLSX to {args.output}")
        except Exception as e:
            logging.info(f"Failed to download XLSX: {e}")
    else:
        parser.print_help()
