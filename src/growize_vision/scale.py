"""Reference-length calibration for photos without a printed marker.

Blueprint §3.2, method 1: any object of known physical length placed on the
same plane as the produce establishes scale.  The operator marks the two end
points of that reference in the image and states its true length; the
calibration is a uniform metric scale (isotropic, no perspective model).

This is deliberately weaker than the ArUco homography: it assumes a
top-down camera, a flat scene, and the reference lying in the produce plane.
The assumptions are recorded on the Calibration object and surface in every
measurement's warnings and uncertainty envelope.
"""

from __future__ import annotations

import math

from .config import CalibrationSettings
from .domain import Calibration

MIN_REFERENCE_SPAN_PX = 20.0
MIN_REFERENCE_MM = 5.0
MAX_REFERENCE_MM = 1000.0

REFERENCE_ASSUMPTIONS = (
    "Two-point reference-length calibration: uniform scale only. Assumes a "
    "top-down camera, a flat scene, and the reference object lying in the "
    "same plane as the produce; perspective and lens distortion are not "
    "modelled."
)


def calibrate_from_two_point_reference(
    image_shape: tuple[int, ...],
    point_a_px: tuple[float, float],
    point_b_px: tuple[float, float],
    reference_length_mm: float,
    settings: CalibrationSettings,
    *,
    reference_uncertainty_mm: float = 2.0,
    points_uncertainty_px: float = 3.0,
) -> Calibration:
    """Build a uniform-scale Calibration from a clicked known length."""
    try:
        ax, ay = float(point_a_px[0]), float(point_a_px[1])
        bx, by = float(point_b_px[0]), float(point_b_px[1])
        length_mm = float(reference_length_mm)
    except (TypeError, ValueError, IndexError):
        return _invalid(
            "Reference points and length must be numeric image coordinates and a number."
        )
    if not all(math.isfinite(v) for v in (ax, ay, bx, by, length_mm)):
        return _invalid("Reference points or length are not finite numbers.")
    if not (MIN_REFERENCE_MM <= length_mm <= MAX_REFERENCE_MM):
        return _invalid(
            f"Reference length must be between {MIN_REFERENCE_MM:g} and "
            f"{MAX_REFERENCE_MM:g} mm."
        )
    span_px = math.hypot(bx - ax, by - ay)
    if span_px < MIN_REFERENCE_SPAN_PX:
        return _invalid(
            f"The marked reference spans only {span_px:.1f} px in the image. "
            "Use a longer reference or move the camera closer."
        )
    height, width = image_shape[:2]
    if not (0 <= ax < width and 0 <= bx < width and 0 <= ay < height and 0 <= by < height):
        return _invalid("Reference points must lie inside the image.")
    scale_mm_per_px = length_mm / span_px
    if not math.isfinite(scale_mm_per_px) or scale_mm_per_px <= 0:
        return _invalid("Reference calibration produced an invalid scale.")

    uncertainty_mm = math.sqrt(
        max(reference_uncertainty_mm, 0.0) ** 2
        + (points_uncertainty_px * scale_mm_per_px) ** 2
    )
    uncertainty_mm = max(uncertainty_mm, settings.default_marker_measurement_uncertainty_mm)
    homography = [
        [scale_mm_per_px, 0.0, 0.0],
        [0.0, scale_mm_per_px, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return Calibration(
        valid=True,
        calibration_id="two_point_reference",
        marker_id=None,
        marker_side_mm=length_mm,
        homography=homography,
        mm_per_px=scale_mm_per_px,
        corner_reprojection_rms_px=None,
        marker_measurement_uncertainty_mm=uncertainty_mm,
        source="two_point_reference_length",
        warnings=[REFERENCE_ASSUMPTIONS],
    )


def _invalid(reason: str) -> Calibration:
    return Calibration(
        valid=False,
        calibration_id="unavailable",
        reasons=[f"TWO_POINT_REFERENCE_INVALID: {reason}"],
    )
