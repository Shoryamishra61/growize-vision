"""Media decoding and bounded image normalisation.

Uploaded content is validated by decoding, not by trusting a file extension.
Remote URLs are deliberately not supported, avoiding server-side request
forgery and accidental upload of private farm media.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


class MediaError(ValueError):
    pass


@dataclass
class DecodedImage:
    bgr: np.ndarray
    format: str
    width: int
    height: int
    exif_transposed: bool = False
    alpha_composited: bool = False
    warnings: list[str] = field(default_factory=list)
    sha256: str = ""


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def validate_bgr_frame(
    frame: Any,
    *,
    max_pixels: int = 40_000_000,
    name: str = "frame",
) -> tuple[np.ndarray, list[str]]:
    """Validate an in-memory BGR frame and normalise channel count."""
    warnings_list: list[str] = []
    if frame is None:
        raise MediaError(f"{name} is empty.")
    array = np.asarray(frame)
    if array.size == 0:
        raise MediaError(f"{name} is empty.")
    if not np.issubdtype(array.dtype, np.number):
        raise MediaError(f"{name} is not a numeric image.")
    # Check finiteness before casting: NaN/Inf silently become 0 on an
    # integer cast, which could otherwise turn corrupt input into a plausible
    # image.
    if not np.isfinite(array).all():
        raise MediaError(f"{name} contains non-finite pixel values.")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
        warnings_list.append("Input was not uint8; values were clipped to 0..255.")
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    elif array.ndim == 3 and array.shape[2] == 4:
        alpha = array[:, :, 3:4].astype(np.float32) / 255.0
        bgr = array[:, :, :3].astype(np.float32)
        array = np.clip(bgr * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
        warnings_list.append("Alpha channel was composited on a white background.")
    elif array.ndim == 3 and array.shape[2] == 3:
        pass
    else:
        raise MediaError(f"{name} has unsupported shape {array.shape}.")
    h, w = array.shape[:2]
    if h < 16 or w < 16:
        raise MediaError(f"{name} is too small ({w}x{h}); minimum is 16x16.")
    if h * w > max_pixels:
        raise MediaError(
            f"{name} has {h * w:,} pixels; limit is {max_pixels:,}. "
            "Downscale before upload."
        )
    return np.ascontiguousarray(array), warnings_list


def _decode_pil_image(raw: bytes, max_pixels: int, max_bytes: int) -> DecodedImage:
    if not raw:
        raise MediaError("Image file is empty.")
    if len(raw) > max_bytes:
        raise MediaError(
            f"Image is {len(raw):,} bytes; limit is {max_bytes:,} bytes."
        )
    if not (raw.startswith(b"\xff\xd8\xff") or raw.startswith(b"\x89PNG\r\n\x1a\n") or raw[:4] == b"RIFF"):
        # RIFF is also used by WebP and other formats; final authority is PIL.
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as probe:
                fmt = (probe.format or "").upper()
                if fmt not in SUPPORTED_IMAGE_FORMATS:
                    raise MediaError(
                        f"Unsupported image format {fmt or 'unknown'}; "
                        f"supported: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}."
                    )
                if getattr(probe, "n_frames", 1) > 1:
                    raise MediaError("Animated/multi-frame images are not supported.")
                w, h = probe.size
                if w * h > max_pixels:
                    raise MediaError(
                        f"Image has {w*h:,} pixels; limit is {max_pixels:,}."
                    )
                probe.verify()
            with Image.open(io.BytesIO(raw)) as image:
                fmt = (image.format or "").upper()
                alpha = image.mode in ("RGBA", "LA", "PA") or (
                    image.mode == "P" and "transparency" in image.info
                )
                exif_transposed = False
                transposed = ImageOps.exif_transpose(image)
                if transposed is not image:
                    exif_transposed = True
                if alpha:
                    # Composite transparency on white *before* dropping alpha;
                    # otherwise fully transparent pixels become black and can
                    # contaminate segmentation and quality metrics.
                    rgba = np.asarray(transposed.convert("RGBA"), dtype=np.float32)
                    alpha_channel = rgba[:, :, 3:4] / 255.0
                    array = np.clip(
                        rgba[:, :, :3] * alpha_channel
                        + 255.0 * (1.0 - alpha_channel),
                        0,
                        255,
                    ).astype(np.uint8)
                    if array.shape[1] * array.shape[0] > max_pixels:
                        raise MediaError("Decoded image exceeds the pixel limit.")
                else:
                    rgb = transposed.convert("RGB")
                    if rgb.size[0] * rgb.size[1] > max_pixels:
                        raise MediaError("Decoded image exceeds the pixel limit.")
                    array = np.asarray(rgb)
    except Image.DecompressionBombError as exc:
        raise MediaError("Image exceeds the safe decompression pixel limit.") from exc
    except Image.DecompressionBombWarning as exc:
        raise MediaError("Image triggered the decompression-bomb warning limit.") from exc
    except UnidentifiedImageError as exc:
        raise MediaError("File is not a decodable JPEG, PNG or WebP image.") from exc
    except OSError as exc:
        raise MediaError(f"Image decoder failed: {exc}") from exc

    import hashlib

    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return DecodedImage(
        bgr=bgr,
        format=fmt,
        width=int(bgr.shape[1]),
        height=int(bgr.shape[0]),
        exif_transposed=exif_transposed,
        alpha_composited=alpha,
        warnings=(
            ["A white background is assumed for transparency."] if alpha else []
        ),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def decode_image_bytes(
    raw: bytes,
    *,
    max_pixels: int = 40_000_000,
    max_bytes: int = 25_000_000,
) -> DecodedImage:
    return _decode_pil_image(raw, max_pixels=max_pixels, max_bytes=max_bytes)


def load_image_file(
    path: Path | str,
    *,
    max_pixels: int = 40_000_000,
    max_bytes: int = 25_000_000,
) -> DecodedImage:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise MediaError(f"Image file does not exist: {path.name}")
    if path.stat().st_size > max_bytes:
        raise MediaError(
            f"Image is {path.stat().st_size:,} bytes; limit is {max_bytes:,} bytes."
        )
    return decode_image_bytes(
        path.read_bytes(), max_pixels=max_pixels, max_bytes=max_bytes
    )


def resize_for_processing(
    image_bgr: np.ndarray, max_dimension: int
) -> tuple[np.ndarray, float]:
    if max_dimension <= 0:
        raise MediaError("max_dimension must be > 0.")
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return image_bgr.copy(), 1.0
    scale = max_dimension / float(longest)
    new_w = max(16, int(round(w * scale)))
    new_h = max(16, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def image_sha256(image_bgr: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(image_bgr).tobytes()).hexdigest()


def safe_export_stem(name: str, fallback: str = "inspection") -> str:
    """Return a filesystem-safe stem without allowing traversal."""
    stem = Path(str(name or "")).name
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
    stem = stem.strip("._")[:80]
    return stem or fallback
