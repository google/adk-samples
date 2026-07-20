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

"""Shared pytest fixtures for GenMedia tests."""

import io

import pytest
from PIL import Image

# PNG magic bytes (minimal valid PNG header)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# JPEG magic bytes
JPEG_SIGNATURE = b"\xff\xd8\xff\xe0\x00\x10JFIF"

# WebP magic bytes (RIFF + size + WEBP)
WEBP_SIGNATURE = b"RIFF\x00\x00\x00\x00WEBP"

# GIF magic bytes
GIF_SIGNATURE = b"GIF89a"

# AVIF magic bytes (ftyp box with avif brand)
AVIF_SIGNATURE = b"\x00\x00\x00\x1cftypavifall"


@pytest.fixture
def png_bytes():
    """Minimal valid PNG bytes with magic header."""
    return PNG_SIGNATURE + b"\x00" * 20  # Pad to meet minimum length


@pytest.fixture
def jpeg_bytes():
    """Minimal JPEG bytes with magic header."""
    return JPEG_SIGNATURE + b"\x00" * 20


@pytest.fixture
def webp_bytes():
    """Minimal WebP bytes with magic header."""
    return WEBP_SIGNATURE + b"\x00" * 20


@pytest.fixture
def gif_bytes():
    """Minimal GIF bytes with magic header."""
    return GIF_SIGNATURE + b"\x00" * 20


@pytest.fixture
def avif_bytes():
    """Minimal AVIF bytes with magic header."""
    return AVIF_SIGNATURE + b"\x00" * 20


@pytest.fixture
def unknown_bytes():
    """Unknown format bytes (no recognizable magic header)."""
    return b"\x00\x01\x02\x03" * 10


@pytest.fixture
def sample_image_bytes():
    """Create a simple 100x100 red image as PNG bytes."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_image_bytes_200x100():
    """Create a 200x100 blue image as PNG bytes (landscape)."""
    img = Image.new("RGB", (200, 100), color=(0, 0, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_image_bytes_100x200():
    """Create a 100x200 green image as PNG bytes (portrait)."""
    img = Image.new("RGB", (100, 200), color=(0, 255, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_rgba_image_bytes():
    """Create a 100x100 RGBA image with transparency as PNG bytes."""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
