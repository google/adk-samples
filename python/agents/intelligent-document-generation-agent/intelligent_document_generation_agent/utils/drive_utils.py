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

import argparse
import io
import logging
import re
from typing import List, Tuple

import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Get a logger specific to this module
logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_drive_service():
    """
    Authenticates with the Google Drive API using Application Default Credentials (ADC)
    and returns a service object.
    """
    try:
        # google.auth.default() will automatically find the credentials
        # from the environment.
        creds, _ = google.auth.default(scopes=SCOPES)
        return build("drive", "v3", credentials=creds)
    except google.auth.exceptions.DefaultCredentialsError as e:
        logger.info(
            "Authentication failed. Please configure Application Default Credentials. "
            "For local development, run 'gcloud auth application-default login'. "
            "On a Google Cloud environment, ensure the service account has Drive API access."
        )
        raise e


def download_pdf_from_drive(service, url: str) -> Tuple[bytes, str] | None:
    """
    Extracts the file ID from a Google Drive URL and downloads the file's content.

    Args:
        service: An authenticated Google Drive API service object.
        url: The full URL of the PDF file on Google Drive.

    Returns:
        A bytestring of the PDF's contents, or None if an error occurs.
    """
    # Regex to extract the file ID from various Google Drive URL formats
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        logger.info(f"Error: Could not extract file ID from URL: {url}")
        return None

    file_id = match.group(1)
    logger.info(f"Found file ID: {file_id}")

    try:
        # Get file metadata to retrieve the name
        file_metadata = service.files().get(fileId=file_id, fields="name").execute()
        file_name = file_metadata.get("name", "downloaded_file.pdf")

        # Request the file's media content
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            logger.info(f"Download {int(status.progress() * 100)}%.")

        return fh.getvalue(), file_name

    except HttpError as error:
        logger.info(f"An error occurred: {error}")
        return None


def list_files_in_drive_folder(service, folder_url: str) -> List[str] | None:
    """
    Extracts the folder ID from a Google Drive folder URL and lists the URLs of files within it.

    Args:
        service: An authenticated Google Drive API service object.
        folder_url: The full URL of the Google Drive folder.

    Returns:
        A list of webViewLinks (URLs) for files in the folder, or None if an error occurs.
    """
    # Regex to extract the folder ID from various Google Drive folder URL formats
    # Example: https://drive.google.com/drive/folders/1abcDEF_ghiJKL-mnoPQRstUVwXyZ
    match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_url)
    if not match:
        logger.info(f"Error: Could not extract folder ID from URL: {folder_url}")
        return None

    folder_id = match.group(1)
    logger.info(f"Found folder ID: {folder_id}")

    try:
        file_urls = []
        page_token = None
        while True:
            # List files in the specified folder
            results = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents",
                    fields="nextPageToken, files(webViewLink)",
                    pageToken=page_token,
                )
                .execute()
            )
            items = results.get("files", [])
            for item in items:
                if "webViewLink" in item:
                    file_urls.append(item["webViewLink"])
            page_token = results.get("nextPageToken")
            if not page_token:
                break
        if not file_urls:
            logger.info("No files found in the folder.")
            return None

        logger.info("Found file URLs:")
        for url in file_urls:
            logger.info(url)
        return file_urls

    except HttpError as error:
        logger.info(f"An error occurred while listing files: {error}")
        return None
    except google.auth.exceptions.DefaultCredentialsError as e:
        logger.info(
            "Authentication failed. Please configure Application Default Credentials. "
            "For local development, run 'gcloud auth application-default login'. "
            "On a Google Cloud environment, ensure the service account has Drive API access."
        )
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Google Drive Utility CLI for listing files or downloading PDFs."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for listing files in a folder
    list_parser = subparsers.add_parser(
        "list-folder", help="List files in a Google Drive folder."
    )
    list_parser.add_argument(
        "folder_url", help="The URL of the Google Drive folder to list files from."
    )

    # Subparser for downloading a PDF file
    download_parser = subparsers.add_parser(
        "download-file", help="Download a PDF file from Google Drive."
    )
    download_parser.add_argument(
        "file_url", help="The URL of the Google Drive PDF file to download."
    )
    download_parser.add_argument(
        "--output",
        "-o",
        default="downloaded_file.pdf",
        help="Output path for the downloaded PDF file. Defaults to 'downloaded_file.pdf'.",
    )

    args = parser.parse_args()

    service = get_drive_service()

    if args.command == "list-folder":
        list_files_in_drive_folder(service, args.folder_url)
    elif args.command == "download-file":
        logger.info(f"Attempting to download PDF from URL: {args.file_url}")
        pdf_data_and_name = download_pdf_from_drive(service, args.file_url)
        if pdf_data_and_name:
            pdf_bytes, file_name = pdf_data_and_name
            output_path = (
                args.output if args.output != "downloaded_file.pdf" else file_name
            )
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
            logger.info(f"Successfully downloaded '{file_name}' to '{output_path}'")
        else:
            logger.info(f"Failed to download PDF from {args.file_url}")
    else:
        parser.logger.info_help()
