"""Image-quality screening for the controlled-station baseline."""

from __future__ import annotations

import cv2
import numpy as np

from .config import QualitySettings
from .domain import ImageQuality


def _fraction(mask: np.ndarray, total: int | None = None) -> float:
    denominator = int(total if total is not None else mask.size)
    if denominator <= 0:
        return 0.0
    return float(np.count_nonzero(mask) / denominator)


def _touches_edge(mask: np.ndarray, border_px: int) -> bool:
    if mask is None or mask.size == 0:
        return False
    border_px = max(1, int(border_px))
    edges = [
        mask[:border_px, :],
        mask[-border_px:, :],
        mask[:, :border_px],
        mask[:, -border_px:],
    ]
    return any(np.any(edge > 0) for edge in edges)


def _object_roi(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None or not np.any(mask):
        return np.ones(shape, dtype=bool)
    # A small dilation includes the object boundary, where clipping/glare is
    # often most visible.
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1).astype(bool)


def assess_quality(
    image_bgr: np.ndarray,
    settings: QualitySettings,
    *,
    marker_mask: np.ndarray | None = None,
    object_mask: np.ndarray | None = None,
    background_busy: bool = False,
) -> ImageQuality:
    if image_bgr is None or image_bgr.size == 0:
        return ImageQuality(
            blur_score=0.0,
            contrast=0.0,
            clipped_low_fraction=0.0,
            clipped_high_fraction=0.0,
            glare_fraction=0.0,
            passed=False,
            flags=["EMPTY_FRAME"],
        )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    p05, p95 = np.percentile(gray, [5, 95])
    contrast = float(p95 - p05)
    roi = _object_roi(object_mask, gray.shape)
    if marker_mask is not None:
        roi = np.logical_and(roi, marker_mask == 0)
    roi_pixels = max(1, int(np.count_nonzero(roi)))
    clipped_low = float(np.count_nonzero(gray[roi] <= 2) / roi_pixels)
    clipped_high = float(np.count_nonzero(gray[roi] >= 253) / roi_pixels)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Glare is bright and desaturated within the produce ROI; the white mat is
    # excluded by the object mask.  If no object exists, this number is only
    # diagnostic.
    glare = (hsv[:, :, 2] >= 245) & (hsv[:, :, 1] <= 28)
    glare_fraction = float(
        np.count_nonzero(np.logical_and(glare, roi)) / roi_pixels
    )

    flags: list[str] = []
    if blur < settings.blur_min:
        flags.append(
            f"BLUR_LOW (Laplacian variance {blur:.1f} < {settings.blur_min:.1f})"
        )
    if contrast < settings.min_contrast:
        flags.append(
            f"CONTRAST_LOW ({contrast:.1f} < {settings.min_contrast:.1f})"
        )
    if clipped_low > settings.max_clipped_low_fraction:
        flags.append(
            f"SHADOW_CLIPPING ({clipped_low:.1%} object ROI <= 2/255)"
        )
    if clipped_high > settings.max_clipped_high_fraction:
        flags.append(
            f"HIGHLIGHT_CLIPPING ({clipped_high:.1%} object ROI >= 253/255)"
        )
    if glare_fraction > settings.max_glare_fraction:
        flags.append(f"SPECULAR_GLARE ({glare_fraction:.1%} of object ROI)")
    if background_busy:
        flags.append(
            "BUSY_BACKGROUND: border colour is inconsistent; use a plain matte "
            "mat and remove clutter."
        )

    object_fraction = (
        float(np.count_nonzero(object_mask) / object_mask.size)
        if object_mask is not None and object_mask.size
        else 0.0
    )
    touches = (
        _touches_edge(object_mask, settings.boundary_touch_px)
        if object_mask is not None
        else False
    )

    hard_flags = [
        f
        for f in flags
        if f.startswith(
            ("BLUR_LOW", "CONTRAST_LOW", "SHADOW_CLIPPING", "HIGHLIGHT_CLIPPING", "SPECULAR_GLARE")
        )
    ]
    passed = len(hard_flags) == 0 and not background_busy
    return ImageQuality(
        blur_score=blur,
        contrast=contrast,
        clipped_low_fraction=clipped_low,
        clipped_high_fraction=clipped_high,
        glare_fraction=glare_fraction,
        object_touches_edge=touches,
        object_area_fraction=object_fraction,
        passed=passed,
        flags=flags,
        metrics={
            "laplacian_variance": blur,
            "p95_minus_p05": contrast,
            "object_roi_pixels": float(roi_pixels),
        },
    )
