"""Tests for VLM utilities.

Tests cover:
- Image loading from URLs and local paths
- MIME type detection
- Base64 encoding
- Gemini format preparation
- Error handling
"""

import base64
import tempfile
from pathlib import Path

import pytest

from multiview.inference.vlm_utils import (
    detect_mime_type,
    encode_image_base64,
    load_image_from_source,
    prepare_image_for_gemini,
    validate_image_list,
)


class TestImageLoading:
    """Test image loading from different sources."""

    def test_load_from_url(self):
        """Test loading image from URL."""
        # Use a small public domain image
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg"

        image_bytes = load_image_from_source(url)

        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0
        # JPEG magic bytes
        assert image_bytes[:2] == b"\xff\xd8"

    def test_load_from_local_path(self):
        """Test loading image from local file path."""
        # Create a temporary JPEG file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Minimal valid JPEG
            jpeg_data = (
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
                b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
                b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
                b"\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01"
                b"\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00"
                b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xda\x00\x08\x01\x01\x00\x00"
                b"?\x00\xd2\xcf \xff\xd9"
            )
            f.write(jpeg_data)
            temp_path = f.name

        try:
            image_bytes = load_image_from_source(temp_path)

            assert isinstance(image_bytes, bytes)
            assert len(image_bytes) > 0
            assert image_bytes[:2] == b"\xff\xd8"  # JPEG magic bytes
        finally:
            Path(temp_path).unlink()

    def test_load_from_invalid_url(self):
        """Test that invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="Error loading image from URL"):
            load_image_from_source("https://invalid-domain-that-does-not-exist-12345.com/image.jpg")

    def test_load_from_nonexistent_path(self):
        """Test that nonexistent file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_from_source("/path/that/does/not/exist/image.jpg")

    def test_load_empty_source(self):
        """Test that empty source raises ValueError."""
        with pytest.raises(ValueError, match="Image source cannot be empty"):
            load_image_from_source("")


class TestMimeTypeDetection:
    """Test MIME type detection."""

    def test_detect_jpeg_from_magic_bytes(self):
        """Test JPEG detection from magic bytes."""
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        mime_type = detect_mime_type(jpeg_bytes)
        assert mime_type == "image/jpeg"

    def test_detect_png_from_magic_bytes(self):
        """Test PNG detection from magic bytes."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mime_type = detect_mime_type(png_bytes)
        assert mime_type == "image/png"

    def test_detect_gif_from_magic_bytes(self):
        """Test GIF detection from magic bytes."""
        gif_bytes = b"GIF89a" + b"\x00" * 100
        mime_type = detect_mime_type(gif_bytes)
        assert mime_type == "image/gif"

    def test_detect_from_path_extension(self):
        """Test MIME type detection from file extension."""
        unknown_bytes = b"\x00" * 100
        mime_type = detect_mime_type(unknown_bytes, source_path="test.png")
        assert mime_type == "image/png"

    def test_default_to_jpeg(self):
        """Test that unknown formats default to JPEG."""
        unknown_bytes = b"\x00" * 100
        mime_type = detect_mime_type(unknown_bytes)
        assert mime_type == "image/jpeg"


class TestBase64Encoding:
    """Test base64 encoding."""

    def test_encode_simple_bytes(self):
        """Test encoding simple bytes."""
        test_bytes = b"Hello, World!"
        encoded = encode_image_base64(test_bytes)

        assert isinstance(encoded, str)
        # Verify it's valid base64
        decoded = base64.b64decode(encoded)
        assert decoded == test_bytes

    def test_encode_empty_bytes(self):
        """Test encoding empty bytes."""
        encoded = encode_image_base64(b"")
        assert isinstance(encoded, str)
        assert len(encoded) == 0


class TestGeminiPreparation:
    """Test preparing images for Gemini."""

    def test_prepare_image_from_url(self):
        """Test preparing image from URL for Gemini."""
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg"

        prepared = prepare_image_for_gemini(url)

        assert isinstance(prepared, dict)
        assert "inline_data" in prepared
        assert "mime_type" in prepared["inline_data"]
        assert "data" in prepared["inline_data"]
        assert prepared["inline_data"]["mime_type"].startswith("image/")
        assert len(prepared["inline_data"]["data"]) > 0

    def test_prepare_image_structure(self):
        """Test that prepared image has correct structure."""
        # Create minimal JPEG
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            jpeg_data = (
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
                b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
                b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
                b"\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01"
                b"\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00"
                b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xda\x00\x08\x01\x01\x00\x00"
                b"?\x00\xd2\xcf \xff\xd9"
            )
            f.write(jpeg_data)
            temp_path = f.name

        try:
            prepared = prepare_image_for_gemini(temp_path)

            # Check structure matches Gemini's expected format
            assert "inline_data" in prepared
            inline_data = prepared["inline_data"]
            assert "mime_type" in inline_data
            assert "data" in inline_data
            assert inline_data["mime_type"] == "image/jpeg"

            # Verify base64 can be decoded
            decoded = base64.b64decode(inline_data["data"])
            assert len(decoded) > 0
        finally:
            Path(temp_path).unlink()


class TestImageListValidation:
    """Test image list validation."""

    def test_validate_valid_list(self):
        """Test validation of valid image list."""
        images = ["image1.jpg", "image2.jpg", None, "image3.jpg"]
        validated = validate_image_list(images)
        assert validated == images

    def test_validate_empty_list(self):
        """Test validation of empty list."""
        images = []
        validated = validate_image_list(images)
        assert validated == []

    def test_validate_all_none(self):
        """Test validation of list with all None."""
        images = [None, None, None]
        validated = validate_image_list(images)
        assert validated == images

    def test_validate_non_list_raises_error(self):
        """Test that non-list input raises ValueError."""
        with pytest.raises(ValueError, match="Images must be a list"):
            validate_image_list("not a list")

    def test_validate_tuple_raises_error(self):
        """Test that tuple input raises ValueError."""
        with pytest.raises(ValueError, match="Images must be a list"):
            validate_image_list(("image1.jpg", "image2.jpg"))
