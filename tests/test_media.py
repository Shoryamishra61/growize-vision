"""Media decoding, normalisation and path-safety tests."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from growize_vision.media import (
    MediaError,
    decode_image_bytes,
    image_sha256,
    load_image_file,
    resize_for_processing,
    safe_export_stem,
    validate_bgr_frame,
)


def encoded(image: np.ndarray, extension: str) -> bytes:
    ok, buffer = cv2.imencode(extension, image)
    assert ok
    return buffer.tobytes()


@pytest.mark.parametrize(
    ("extension", "expected_format"),
    [(".png", "PNG"), (".jpg", "JPEG"), (".webp", "WEBP")],
)
def test_supported_formats_decode_to_bgr(extension: str, expected_format: str) -> None:
    bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    bgr[:, :, 2] = 180
    decoded = decode_image_bytes(encoded(bgr, extension))
    assert decoded.format == expected_format
    assert decoded.width == 64
    assert decoded.height == 48
    assert decoded.bgr.shape == (48, 64, 3)
    assert len(decoded.sha256) == 64
    assert decoded.bgr[0, 0, 2] > 100


def test_exif_orientation_is_applied() -> None:
    image = Image.new("RGB", (80, 40), (25, 80, 140))
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees clockwise on display
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    decoded = decode_image_bytes(buffer.getvalue())
    assert decoded.exif_transposed is True
    assert (decoded.width, decoded.height) == (40, 80)


def test_transparent_png_is_composited_on_white() -> None:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[:, :, 3] = 0
    ok, buffer = cv2.imencode(".png", rgba)
    assert ok
    decoded = decode_image_bytes(buffer.tobytes())
    assert decoded.alpha_composited is True
    assert decoded.bgr.mean() > 245
    assert any("white background" in item for item in decoded.warnings)


def test_grayscale_frame_is_normalised_to_three_channels() -> None:
    frame, warnings_list = validate_bgr_frame(np.full((20, 30), 90, dtype=np.uint8))
    assert frame.shape == (20, 30, 3)
    assert warnings_list == []


def test_non_uint8_frame_is_clipped_with_a_warning() -> None:
    values = np.full((20, 20, 3), 128.0, dtype=np.float32)
    values[0, 0, 0] = np.nan
    with pytest.raises(MediaError, match="non-finite"):
        validate_bgr_frame(values)
    clipped, warnings_list = validate_bgr_frame(
        np.array(
            [[[-5.0, 128.0, 260.0]] + [[1.0, 1.0, 1.0]] * 19] * 20,
            dtype=np.float32,
        )
    )
    assert clipped.dtype == np.uint8
    assert clipped.min() == 0
    assert clipped.max() == 255
    assert any("clipped" in item for item in warnings_list)


def test_alpha_frame_is_composited_by_validate_frame() -> None:
    frame = np.zeros((32, 32, 4), dtype=np.uint8)
    frame[:, :, 3] = 0
    normalised, warnings_list = validate_bgr_frame(frame)
    assert normalised.mean() > 245
    assert any("Alpha" in item for item in warnings_list)


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.array([], dtype=np.uint8),
        np.array(["not", "pixels"]),
        np.zeros((32, 32, 2), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ],
)
def test_invalid_frames_are_refused(frame: object) -> None:
    with pytest.raises(MediaError):
        validate_bgr_frame(frame)


def test_pixel_limit_is_enforced_for_frames_and_uploads() -> None:
    with pytest.raises(MediaError, match="limit"):
        validate_bgr_frame(np.zeros((80, 80, 3), dtype=np.uint8), max_pixels=1000)
    with pytest.raises(MediaError, match="pixels"):
        decode_image_bytes(encoded(np.zeros((80, 80, 3), dtype=np.uint8), ".png"), max_pixels=1000)


def test_empty_and_corrupt_bytes_are_refused() -> None:
    with pytest.raises(MediaError, match="empty"):
        decode_image_bytes(b"")
    with pytest.raises(MediaError, match="not a decodable"):
        decode_image_bytes(b"this is not an image")
    with pytest.raises(MediaError, match="not a decodable"):
        decode_image_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)


def test_unsupported_format_is_refused() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buffer, format="GIF")
    with pytest.raises(MediaError, match="Unsupported image format"):
        decode_image_bytes(buffer.getvalue())


def test_byte_limit_is_enforced() -> None:
    raw = encoded(np.zeros((32, 32, 3), dtype=np.uint8), ".png")
    with pytest.raises(MediaError, match="bytes"):
        decode_image_bytes(raw, max_bytes=10)


def test_load_image_file_rejects_missing_and_oversized_files(tmp_path) -> None:
    with pytest.raises(MediaError, match="does not exist"):
        load_image_file(tmp_path / "missing.png")
    path = tmp_path / "large.png"
    path.write_bytes(encoded(np.zeros((400, 400, 3), dtype=np.uint8), ".png"))
    with pytest.raises(MediaError, match="bytes"):
        load_image_file(path, max_bytes=10)


def test_resize_preserves_aspect_ratio_and_copies_small_images() -> None:
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    resized, scale = resize_for_processing(image, 50)
    assert resized.shape == (50, 25, 3)
    assert scale == pytest.approx(0.25)
    same, identity_scale = resize_for_processing(image, 500)
    assert identity_scale == 1.0
    assert same.shape == image.shape
    assert same is not image
    with pytest.raises(MediaError):
        resize_for_processing(image, 0)


def test_safe_export_stem_neutralises_traversal_and_weird_names() -> None:
    for raw in ("../../evil name.png", "..\\..\\secret.json", "C:\\Windows\\system32"):
        stem = safe_export_stem(raw)
        assert "/" not in stem and "\\" not in stem and ":" not in stem
        assert ".." not in stem
    assert safe_export_stem("") == "inspection"
    assert safe_export_stem("...") == "inspection"
    assert len(safe_export_stem("a" * 300)) <= 80


def test_image_hash_is_deterministic_and_content_sensitive() -> None:
    first = np.zeros((20, 20, 3), dtype=np.uint8)
    second = first.copy()
    assert image_sha256(first) == image_sha256(second)
    second[0, 0, 0] = 1
    assert image_sha256(first) != image_sha256(second)
