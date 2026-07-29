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

"""Tests for shared/gcs_utils.py - Google Cloud Storage operations with mocked client."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from workflows.shared.gcs_utils import (
    download_file_from_gcs,
    get_storage_client,
    upload_bytes_to_gcs,
    upload_file_to_gcs,
    upload_folder_to_gcs,
)


@pytest.fixture
def mock_storage_client():
    """Create a mock GCS storage client."""
    client = Mock()
    bucket = Mock()
    blob = Mock()

    client.bucket.return_value = bucket
    bucket.blob.return_value = blob

    return client, bucket, blob


@pytest.fixture
def temp_folder_with_files():
    """Create a temporary folder with test files."""
    temp_dir = tempfile.mkdtemp()
    files = {
        "file1.txt": b"content1",
        "file2.png": b"png_content",
        "subdir/file3.jpg": b"jpg_content",
        "subdir/nested/file4.txt": b"nested_content",
    }

    for path, content in files.items():
        full_path = Path(temp_dir) / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)

    yield temp_dir, files

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


class TestGetStorageClient:
    """Tests for get_storage_client function."""

    @patch("workflows.shared.gcs_utils.storage.Client")
    def test_creates_client_without_project(self, mock_client_class):
        """Should create client without project ID."""
        get_storage_client()
        mock_client_class.assert_called_once_with(project=None)

    @patch("workflows.shared.gcs_utils.storage.Client")
    def test_creates_client_with_project(self, mock_client_class):
        """Should create client with specified project ID."""
        get_storage_client(project_id="my-project")
        mock_client_class.assert_called_once_with(project="my-project")


class TestUploadBytesToGcs:
    """Tests for upload_bytes_to_gcs function."""

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_uploads_bytes(self, mock_get_client, mock_storage_client):
        """Should upload bytes to GCS."""
        client, bucket, blob = mock_storage_client
        mock_get_client.return_value = client

        result = upload_bytes_to_gcs(
            bucket_name="test-bucket",
            file_bytes=b"test content",
            destination_blob_name="path/to/file.txt",
        )

        assert result == "gs://test-bucket/path/to/file.txt"
        client.bucket.assert_called_once_with("test-bucket")
        bucket.blob.assert_called_once_with("path/to/file.txt")
        blob.upload_from_string.assert_called_once_with(b"test content")

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_sets_content_type(self, mock_get_client, mock_storage_client):
        """Should set content type when provided."""
        client, _bucket, blob = mock_storage_client
        mock_get_client.return_value = client

        upload_bytes_to_gcs(
            bucket_name="test-bucket",
            file_bytes=b"test content",
            destination_blob_name="file.json",
            content_type="application/json",
        )

        assert blob.content_type == "application/json"

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_passes_project_id(self, mock_get_client, mock_storage_client):
        """Should pass project ID to client."""
        client, _bucket, _blob = mock_storage_client
        mock_get_client.return_value = client

        upload_bytes_to_gcs(
            bucket_name="test-bucket",
            file_bytes=b"test",
            destination_blob_name="file.txt",
            project_id="my-project",
        )

        mock_get_client.assert_called_once_with("my-project")

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_returns_gcs_uri(self, mock_get_client, mock_storage_client):
        """Should return correct GCS URI."""
        client, _bucket, _blob = mock_storage_client
        mock_get_client.return_value = client

        result = upload_bytes_to_gcs(
            bucket_name="my-bucket",
            file_bytes=b"data",
            destination_blob_name="folder/subfolder/file.bin",
        )

        assert result == "gs://my-bucket/folder/subfolder/file.bin"


class TestUploadFileToGcs:
    """Tests for upload_file_to_gcs function."""

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_uploads_file(self, mock_get_client, mock_storage_client):
        """Should upload file from path."""
        client, _bucket, blob = mock_storage_client
        mock_get_client.return_value = client

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".txt") as f:
            f.write(b"file content")
            temp_path = f.name

        try:
            result = upload_file_to_gcs(
                bucket_name="test-bucket",
                source_file_path=temp_path,
                destination_blob_name="uploaded.txt",
            )

            assert result == "gs://test-bucket/uploaded.txt"
            blob.upload_from_filename.assert_called_once_with(temp_path)
        finally:
            os.unlink(temp_path)

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_raises_on_missing_file(self, mock_get_client, mock_storage_client):
        """Should raise ValueError for non-existent file."""
        client, _bucket, _blob = mock_storage_client
        mock_get_client.return_value = client

        with pytest.raises(ValueError, match="Invalid source file"):
            upload_file_to_gcs(
                bucket_name="test-bucket",
                source_file_path="/nonexistent/file.txt",
                destination_blob_name="file.txt",
            )

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_raises_on_directory(self, mock_get_client, mock_storage_client):
        """Should raise ValueError when source is a directory."""
        client, _bucket, _blob = mock_storage_client
        mock_get_client.return_value = client

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Invalid source file"):
                upload_file_to_gcs(
                    bucket_name="test-bucket",
                    source_file_path=temp_dir,
                    destination_blob_name="file.txt",
                )

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_sets_content_type(self, mock_get_client, mock_storage_client):
        """Should set content type when provided."""
        client, _bucket, blob = mock_storage_client
        mock_get_client.return_value = client

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"content")
            temp_path = f.name

        try:
            upload_file_to_gcs(
                bucket_name="test-bucket",
                source_file_path=temp_path,
                destination_blob_name="file.png",
                content_type="image/png",
            )

            assert blob.content_type == "image/png"
        finally:
            os.unlink(temp_path)


class TestDownloadFileFromGcs:
    """Tests for download_file_from_gcs function."""

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_downloads_file(self, mock_get_client, mock_storage_client):
        """Should download file to local path."""
        client, bucket, blob = mock_storage_client
        mock_get_client.return_value = client

        with tempfile.TemporaryDirectory() as temp_dir:
            dest_path = os.path.join(temp_dir, "downloaded.txt")

            result = download_file_from_gcs(
                bucket_name="test-bucket",
                source_blob_name="path/file.txt",
                destination_file_path=dest_path,
            )

            assert result == dest_path
            client.bucket.assert_called_once_with("test-bucket")
            bucket.blob.assert_called_once_with("path/file.txt")
            blob.download_to_filename.assert_called_once_with(dest_path)

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_creates_parent_directories(self, mock_get_client, mock_storage_client):
        """Should create parent directories if they don't exist."""
        client, _bucket, _blob = mock_storage_client
        mock_get_client.return_value = client

        with tempfile.TemporaryDirectory() as temp_dir:
            dest_path = os.path.join(temp_dir, "nested", "dirs", "file.txt")

            result = download_file_from_gcs(
                bucket_name="test-bucket",
                source_blob_name="file.txt",
                destination_file_path=dest_path,
            )

            assert result == dest_path
            assert os.path.exists(os.path.dirname(dest_path))

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_passes_project_id(self, mock_get_client, mock_storage_client):
        """Should pass project ID to client."""
        client, _bucket, _blob = mock_storage_client
        mock_get_client.return_value = client

        with tempfile.TemporaryDirectory() as temp_dir:
            dest_path = os.path.join(temp_dir, "file.txt")

            download_file_from_gcs(
                bucket_name="test-bucket",
                source_blob_name="file.txt",
                destination_file_path=dest_path,
                project_id="my-project",
            )

            mock_get_client.assert_called_once_with("my-project")


class TestUploadFolderToGcs:
    """Tests for upload_folder_to_gcs function."""

    @patch("workflows.shared.gcs_utils.transfer_manager.upload_many")
    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_uploads_all_files(
        self,
        mock_get_client,
        mock_upload_many,
        mock_storage_client,
        temp_folder_with_files,
    ):
        """Should upload all files in folder."""
        client, _bucket, _blob = mock_storage_client
        mock_get_client.return_value = client
        mock_upload_many.return_value = [
            None,
            None,
            None,
            None,
        ]  # Success for each file

        temp_dir, _files = temp_folder_with_files

        result = upload_folder_to_gcs(
            bucket_name="test-bucket",
            source_folder_path=temp_dir,
            destination_prefix="output/",
        )

        assert len(result) == 4
        mock_upload_many.assert_called_once()

    @patch("workflows.shared.gcs_utils.transfer_manager.upload_many")
    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_filters_by_extension(
        self,
        mock_get_client,
        mock_upload_many,
        mock_storage_client,
        temp_folder_with_files,
    ):
        """Should filter files by extension."""
        client, _bucket, _blob = mock_storage_client
        mock_get_client.return_value = client
        mock_upload_many.return_value = [None, None]  # Success for txt files only

        temp_dir, _files = temp_folder_with_files

        upload_folder_to_gcs(
            bucket_name="test-bucket",
            source_folder_path=temp_dir,
            include_extensions=[".txt"],
        )

        # Should only upload .txt files (file1.txt and subdir/nested/file4.txt)
        call_args = mock_upload_many.call_args[0][0]
        assert len(call_args) == 2

    @patch("workflows.shared.gcs_utils.transfer_manager.upload_many")
    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_excludes_extensions(
        self,
        mock_get_client,
        mock_upload_many,
        mock_storage_client,
        temp_folder_with_files,
    ):
        """Should exclude files by extension."""
        client, _bucket, _blob = mock_storage_client
        mock_get_client.return_value = client
        mock_upload_many.return_value = [None, None, None]

        temp_dir, _files = temp_folder_with_files

        upload_folder_to_gcs(
            bucket_name="test-bucket",
            source_folder_path=temp_dir,
            exclude_extensions=[".txt"],
        )

        # Should exclude .txt files
        call_args = mock_upload_many.call_args[0][0]
        assert len(call_args) == 2  # Only png and jpg

    @patch("workflows.shared.gcs_utils.transfer_manager.upload_many")
    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_handles_destination_prefix(
        self, mock_get_client, mock_upload_many, temp_folder_with_files
    ):
        """Should prepend destination prefix to blob names."""
        client = Mock()
        bucket = Mock()

        # Create blob factory that returns blobs with proper names
        created_blobs = []

        def create_blob(name):
            blob = Mock()
            blob.name = name
            created_blobs.append(blob)
            return blob

        bucket.blob.side_effect = create_blob
        client.bucket.return_value = bucket
        mock_get_client.return_value = client
        mock_upload_many.return_value = [None, None, None, None]

        temp_dir, _files = temp_folder_with_files

        result = upload_folder_to_gcs(
            bucket_name="test-bucket",
            source_folder_path=temp_dir,
            destination_prefix="products/123",
        )

        # All URIs should start with the prefix
        for uri in result:
            assert uri.startswith("gs://test-bucket/products/123/")

    def test_raises_on_nonexistent_folder(self):
        """Should raise ValueError for non-existent folder."""
        with pytest.raises(ValueError, match="does not exist"):
            upload_folder_to_gcs(
                bucket_name="test-bucket",
                source_folder_path="/nonexistent/folder",
            )

    def test_raises_on_file_instead_of_folder(self):
        """Should raise ValueError when source is a file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="not a folder"):
                upload_folder_to_gcs(
                    bucket_name="test-bucket",
                    source_folder_path=temp_file,
                )
        finally:
            os.unlink(temp_file)

    @patch("workflows.shared.gcs_utils.transfer_manager.upload_many")
    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_returns_empty_for_empty_folder(
        self, mock_get_client, mock_upload_many, mock_storage_client
    ):
        """Should return empty list for empty folder."""
        client, _bucket, _blob = mock_storage_client
        mock_get_client.return_value = client

        with tempfile.TemporaryDirectory() as temp_dir:
            result = upload_folder_to_gcs(
                bucket_name="test-bucket",
                source_folder_path=temp_dir,
            )

            assert result == []
            mock_upload_many.assert_not_called()

    @patch("workflows.shared.gcs_utils.transfer_manager.upload_many")
    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_raises_on_upload_error(
        self,
        mock_get_client,
        mock_upload_many,
        mock_storage_client,
        temp_folder_with_files,
    ):
        """Should raise exception when upload fails."""
        client, _bucket, _blob = mock_storage_client
        mock_get_client.return_value = client
        mock_upload_many.return_value = [None, Exception("Upload failed"), None, None]

        temp_dir, _files = temp_folder_with_files

        with pytest.raises(Exception, match="Failed to upload"):
            upload_folder_to_gcs(
                bucket_name="test-bucket",
                source_folder_path=temp_dir,
            )

    @patch("workflows.shared.gcs_utils.transfer_manager.upload_many")
    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_respects_max_workers(
        self,
        mock_get_client,
        mock_upload_many,
        mock_storage_client,
        temp_folder_with_files,
    ):
        """Should pass max_workers to transfer manager."""
        client, _bucket, _blob = mock_storage_client
        mock_get_client.return_value = client
        mock_upload_many.return_value = [None, None, None, None]

        temp_dir, _files = temp_folder_with_files

        upload_folder_to_gcs(
            bucket_name="test-bucket",
            source_folder_path=temp_dir,
            max_workers=10,
        )

        call_kwargs = mock_upload_many.call_args.kwargs
        assert call_kwargs["max_workers"] == 10


class TestUploadBytesToGcsIntegration:
    """Integration-style tests for upload functions."""

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_upload_image_bytes(self, mock_get_client, mock_storage_client):
        """Should upload PNG image bytes correctly."""
        client, _bucket, blob = mock_storage_client
        mock_get_client.return_value = client

        # Create actual PNG bytes
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        result = upload_bytes_to_gcs(
            bucket_name="images-bucket",
            file_bytes=png_bytes,
            destination_blob_name="products/shoe.png",
            content_type="image/png",
        )

        assert result == "gs://images-bucket/products/shoe.png"
        blob.upload_from_string.assert_called_once_with(png_bytes)
        assert blob.content_type == "image/png"

    @patch("workflows.shared.gcs_utils.get_storage_client")
    def test_upload_video_bytes(self, mock_get_client, mock_storage_client):
        """Should upload video bytes correctly."""
        client, _bucket, blob = mock_storage_client
        mock_get_client.return_value = client

        video_bytes = b"fake_video_mp4_content"

        result = upload_bytes_to_gcs(
            bucket_name="videos-bucket",
            file_bytes=video_bytes,
            destination_blob_name="outputs/product_360.mp4",
            content_type="video/mp4",
        )

        assert result == "gs://videos-bucket/outputs/product_360.mp4"
        blob.upload_from_string.assert_called_once_with(video_bytes)
