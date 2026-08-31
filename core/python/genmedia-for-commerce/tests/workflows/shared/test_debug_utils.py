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

"""Tests for shared/debug_utils.py - Debug image saving utilities."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from workflows.shared import debug_utils
from workflows.shared.debug_utils import (
    get_debug_session_folder,
    reset_session,
    save_debug_image,
    set_debug_enabled,
)


class TestSetDebugEnabled:
    """Tests for set_debug_enabled function."""

    def test_enable_debug(self):
        """Should enable debug mode."""
        original = debug_utils.DEBUG_ENABLED
        try:
            set_debug_enabled(True)
            assert debug_utils.DEBUG_ENABLED is True
        finally:
            debug_utils.DEBUG_ENABLED = original

    def test_disable_debug(self):
        """Should disable debug mode."""
        original = debug_utils.DEBUG_ENABLED
        try:
            set_debug_enabled(False)
            assert debug_utils.DEBUG_ENABLED is False
        finally:
            debug_utils.DEBUG_ENABLED = original


class TestResetSession:
    """Tests for reset_session function."""

    def test_resets_session_folder(self):
        """Should reset the session folder to None."""
        # Set a dummy session folder
        debug_utils._session_folder = Path("/tmp/dummy")

        reset_session()

        assert debug_utils._session_folder is None


class TestGetDebugSessionFolder:
    """Tests for get_debug_session_folder function."""

    def test_creates_timestamped_folder(self):
        """Should create a timestamped folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch DEBUG_FOLDER to use temp directory
            with patch.object(debug_utils, "DEBUG_FOLDER", Path(tmpdir)):
                folder = get_debug_session_folder()

                assert folder.exists()
                assert folder.is_dir()
                # Folder name should be timestamp format YYYYMMDD_HHMMSS
                assert len(folder.name) == 15  # e.g., "20250128_143052"
                assert "_" in folder.name


class TestSaveDebugImage:
    """Tests for save_debug_image function."""

    def test_returns_none_when_disabled(self, sample_image_bytes):
        """Should return None when debug is disabled."""
        original = debug_utils.DEBUG_ENABLED
        try:
            debug_utils.DEBUG_ENABLED = False
            result = save_debug_image(sample_image_bytes, "test_step")
            assert result is None
        finally:
            debug_utils.DEBUG_ENABLED = original

    def test_saves_image_when_enabled(self, sample_image_bytes):
        """Should save image and return path when enabled."""
        original_enabled = debug_utils.DEBUG_ENABLED
        original_session = debug_utils._session_folder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Enable debug and set temp folder
                debug_utils.DEBUG_ENABLED = True
                debug_utils._session_folder = Path(tmpdir)

                result = save_debug_image(sample_image_bytes, "test_step")

                assert result is not None
                assert result.exists()
                assert result.name == "test_step.png"
        finally:
            debug_utils.DEBUG_ENABLED = original_enabled
            debug_utils._session_folder = original_session

    def test_saves_with_prefix(self, sample_image_bytes):
        """Should include prefix in filename."""
        original_enabled = debug_utils.DEBUG_ENABLED
        original_session = debug_utils._session_folder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                debug_utils.DEBUG_ENABLED = True
                debug_utils._session_folder = Path(tmpdir)

                result = save_debug_image(
                    sample_image_bytes, "test_step", prefix="myprefix"
                )

                assert result is not None
                assert result.name == "myprefix_test_step.png"
        finally:
            debug_utils.DEBUG_ENABLED = original_enabled
            debug_utils._session_folder = original_session

    def test_handles_invalid_image_bytes(self):
        """Should return None for invalid image bytes."""
        original_enabled = debug_utils.DEBUG_ENABLED
        original_session = debug_utils._session_folder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                debug_utils.DEBUG_ENABLED = True
                debug_utils._session_folder = Path(tmpdir)

                result = save_debug_image(b"not an image", "test_step")

                assert result is None
        finally:
            debug_utils.DEBUG_ENABLED = original_enabled
            debug_utils._session_folder = original_session
