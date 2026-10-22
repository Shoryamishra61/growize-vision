"""Projected-plane metrology and uncertainty bookkeeping."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import CalibrationSettings
from .domain import Calibration, PhysicalMeasurement


class MetrologyError(ValueError):
    pass


def transform_points(
    points_px: np.ndarray, homography: np.ndarray
) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
    if not np.all(np.isfinite(transformed)):
        raise MetrologyError("Homography produced non-finite physical coordinates.")
    return transformed


def _max_feret_mm(points_mm: np.ndarray) -> float:
    if len(points_mm) < 2:
        return 0.0
    hull = cv2.convexHull(points_mm.astype(np.float32).reshape(-1, 1, 2)).reshape(-1, 2)
    if len(hull) < 2:
        return 0.0
    best = 0.0
    # Contours after resizing are usually < 1000 hull vertices.  This exact
    # pairwise search is deterministic and easier to audit than an approximate
    # rotating-calipers implementation.
    for i in range(len(hull)):
        deltas = hull[i + 1 :] - hull[i]
        if len(deltas) == 0:
            continue
        best = max(best, float(np.sqrt(np.max(np.sum(deltas * deltas, axis=1)))))
    return best


def _local_mm_per_px(
    homography: np.ndarray, point_px: np.ndarray
) -> float:
    base = transform_points(point_px.reshape(1, 2), homography)[0]
    offsets = transform_points(
        np.array(
            [
                [point_px[0] + 1.0, point_px[1]],
                [point_px[0], point_px[1] + 1.0],
            ],
            dtype=np.float32,
        ),
        homography,
    )
    return float(np.mean(np.linalg.norm(offsets - base, axis=1)))


def _distance_from_marker_ratio(
    contour_px: np.ndarray, calibration: Calibration
) -> float:
    if not calibration.marker_side_mm or not calibration.mm_per_px:
        return 0.0
    marker_side_px = calibration.marker_side_mm / calibration.mm_per_px
    if marker_side_px <= 0:
        return 0.0
    moments = cv2.moments(contour_px)
    if moments["m00"] <= 0:
        return 0.0
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    # Marker center is not stored separately; the homography's origin is the
    # marker's top-left corner.  The distance from the origin is a conservative
    # proxy for leverage away from the calibration square.
    origin = transform_points(np.array([[0.0, 0.0]], dtype=np.float32), np.linalg.inv(np.asarray(calibration.homography)))[0]
    return float(np.linalg.norm(np.array([cx, cy]) - origin) / marker_side_px)


def measure_contour(
    contour_px: np.ndarray,
    calibration: Calibration,
    metric: str,
    calibration_settings: CalibrationSettings,
) -> tuple[PhysicalMeasurement, np.ndarray]:
    if not calibration.valid or not calibration.homography:
        raise MetrologyError("A valid planar calibration is required for millimetres.")
    contour = np.asarray(contour_px, dtype=np.float32).reshape(-1, 2)
    if len(contour) < 3:
        raise MetrologyError("Object contour has fewer than three points.")
    area_px = abs(float(cv2.contourArea(contour.reshape(-1, 1, 2))))
    if not math.isfinite(area_px) or area_px <= 0:
        raise MetrologyError("Object mask has zero or invalid area.")
    homography = np.asarray(calibration.homography, dtype=np.float64)
    points_mm = transform_points(contour, homography)
    area_mm2 = abs(float(cv2.contourArea(points_mm.astype(np.float32).reshape(-1, 1, 2))))
    perimeter_mm = float(cv2.arcLength(points_mm.astype(np.float32).reshape(-1, 1, 2), True))
    if not math.isfinite(area_mm2) or area_mm2 <= 0:
        raise MetrologyError("Projected physical area is zero or non-finite.")
    rect = cv2.minAreaRect(points_mm.astype(np.float32).reshape(-1, 1, 2))
    (rx, ry), (rw, rh), _angle = rect
    rect_w, rect_h = float(rw), float(rh)
    if rect_w <= 0 or rect_h <= 0:
        raise MetrologyError("Minimum-area rectangle is degenerate.")
    short_mm = min(rect_w, rect_h)
    long_mm = max(rect_w, rect_h)
    feret_mm = _max_feret_mm(points_mm)
    equiv_diameter_mm = 2.0 * math.sqrt(area_mm2 / math.pi)
    elongation = long_mm / max(short_mm, 1e-9)
    curvature_ratio = (perimeter_mm / 2.0) / max(long_mm, 1e-9)

    if metric == "min_area_rect_short_mm":
        value_mm = short_mm
        metric_unit = "mm"
    elif metric == "min_area_rect_long_mm":
        value_mm = long_mm
        metric_unit = "mm"
    elif metric == "max_feret_mm":
        value_mm = feret_mm
        metric_unit = "mm"
    elif metric == "projected_area_cm2":
        value_mm = area_mm2 / 100.0
        metric_unit = "cm2"
    else:
        raise MetrologyError(f"Unsupported measurement metric: {metric!r}.")
    if not math.isfinite(value_mm) or value_mm <= 0:
        raise MetrologyError("Computed physical dimension is zero or non-finite.")

    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        centroid_px = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float32,
        )
    else:
        centroid_px = contour.mean(axis=0)
    local_scale = _local_mm_per_px(homography, centroid_px)
    marker_side = float(calibration.marker_side_mm or 0.0)
    if marker_side <= 0:
        raise MetrologyError("Calibration is missing the actual marker side length.")
    offset_ratio = _distance_from_marker_ratio(contour, calibration)
    print_scale_error = (
        float(calibration.marker_measurement_uncertainty_mm) / marker_side
    ) * value_mm
    corner_error = (
        calibration_settings.corner_uncertainty_px
        * local_scale
        * (1.0 + 2.0 * min(offset_ratio, 8.0))
    )
    boundary_error = 2.0 * calibration_settings.boundary_uncertainty_px * local_scale
    parallax_error = calibration_settings.parallax_uncertainty_mm
    # Root-sum-square is used only as an engineering propagation envelope.  It
    # is not a statistical 95% interval; the coverage note says so explicitly.
    uncertainty = math.sqrt(
        print_scale_error**2 + corner_error**2 + boundary_error**2 + parallax_error**2
    )
    uncertainty = max(float(uncertainty), 0.75)
    if metric == "projected_area_cm2":
        # The linear envelope above is in mm; convert it into a conservative
        # area envelope in cm2.  The field name remains uncertainty_mm for
        # contract compatibility, while `unit` is cm2 and the warning below
        # makes the unit explicit.
        relative_scale_error = (
            float(calibration.marker_measurement_uncertainty_mm) / marker_side
        )
        characteristic_mm = math.sqrt(area_mm2)
        linear_relative = math.sqrt(
            relative_scale_error**2
            + (corner_error / max(characteristic_mm, 1e-9)) ** 2
            + (parallax_error / max(characteristic_mm, 1e-9)) ** 2
        )
        boundary_area_cm2 = (
            perimeter_mm * (2.0 * calibration_settings.boundary_uncertainty_px * local_scale)
        ) / 100.0
        uncertainty = math.sqrt(
            (linear_relative * value_mm) ** 2 + boundary_area_cm2**2
        )
        uncertainty = max(uncertainty, 0.04 * value_mm)

    measurement = PhysicalMeasurement(
        metric=metric,
        value_mm=float(value_mm),
        uncertainty_mm=float(uncertainty),
        lower_mm=float(value_mm - uncertainty),
        upper_mm=float(value_mm + uncertainty),
        unit=metric_unit,
        method="aruco_planar_homography_contour_geometry",
        calibration_id=calibration.calibration_id,
        secondary={
            "area_mm2": area_mm2,
            "perimeter_mm": perimeter_mm,
            "min_area_rect_short_mm": short_mm,
            "min_area_rect_long_mm": long_mm,
            "max_feret_mm": feret_mm,
            "equivalent_diameter_mm": equiv_diameter_mm,
            "elongation": elongation,
            "curvature_ratio": curvature_ratio,
            "local_mm_per_px": local_scale,
            "offset_from_marker_side_ratio": offset_ratio,
            "print_scale_error_mm": print_scale_error,
            "corner_error_mm": corner_error,
            "boundary_error_mm": boundary_error,
            "parallax_error_mm": parallax_error,
        },
        warnings=[
            "Projected-plane measurement, not weight/volume and not a 3-D surface measurement.",
            "Parallax from object height above the marker plane is included only as a configured engineering bound, not a measured correction.",
            *(
                [
                    "Projected area is reported in cm2; the numeric uncertainty "
                    "field is an area envelope (cm2), not a length in mm."
                ]
                if metric == "projected_area_cm2"
                else []
            ),
        ],
    )
    return measurement, points_mm


def overlay_axes_mm(points_mm: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    rect = cv2.minAreaRect(points_mm.astype(np.float32).reshape(-1, 1, 2))
    box = cv2.boxPoints(rect)
    (cx, cy), (w, h), angle = rect
    if w >= h:
        long_axis = ((box[0][0], box[0][1]), (box[1][0], box[1][1]))
        short_axis = ((box[1][0], box[1][1]), (box[2][0], box[2][1]))
    else:
        long_axis = ((box[1][0], box[1][1]), (box[2][0], box[2][1]))
        short_axis = ((box[0][0], box[0][1]), (box[1][0], box[1][1]))
    return {
        "center": (float(cx), float(cy), float(w), float(h)),
        "long_axis": (
            float(long_axis[0][0]),
            float(long_axis[0][1]),
            float(long_axis[1][0]),
            float(long_axis[1][1]),
        ),
        "short_axis": (
            float(short_axis[0][0]),
            float(short_axis[0][1]),
            float(short_axis[1][0]),
            float(short_axis[1][1]),
        ),
    }
