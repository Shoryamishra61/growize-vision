"""Projected-plane metrology and uncertainty bookkeeping."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .centerline import CenterlineResult, centerline_analysis
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


def _pca_orientation(points_mm: np.ndarray) -> tuple[float, float]:
    """Major-axis orientation in degrees [0, 180) and the anisotropy ratio."""
    centered = points_mm - points_mm.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    large, small = float(eigenvalues[-1]), float(eigenvalues[0])
    major = eigenvectors[:, -1]
    orientation = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180.0
    anisotropy = math.sqrt(large / small) if small > 1e-12 else float("inf")
    return orientation, anisotropy


def _centerline_summary(result: CenterlineResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "status": result.status,
        "length_mm": result.length_mm,
        "endpoint_count": result.endpoint_count,
        "pruned_branch_count": result.pruned_branch_count,
        "terminal_radius_start_mm": result.terminal_radius_start_mm,
        "terminal_radius_end_mm": result.terminal_radius_end_mm,
        "skeleton_nodes": result.skeleton_nodes,
        "reasons": list(result.reasons),
    }


def _volume_proxy(
    result: CenterlineResult | None, elongation: float
) -> dict[str, Any] | None:
    """Explicitly-labelled slender-axisymmetric volume proxy, never a grade input."""
    if result is None or result.status != "OK" or not result.volume_proxy_cm3:
        return None
    if elongation < 2.0:
        return None
    return {
        "method": "frustum_integration_along_projected_centreline",
        "value_cm3": float(result.volume_proxy_cm3),
        "model": (
            "Slender axisymmetric model (Huynh et al. 2020): circular "
            "cross-sections perpendicular to the projected centreline, radii "
            "taken from the in-plane distance transform."
        ),
        "assumptions": [
            "The produce is approximately axisymmetric around its centreline.",
            "The silhouette is a scaled orthographic projection of that body.",
        ],
        "limits": [
            "Flattened produce (leaves, side-laid gourds) violates the circular "
            "cross-section assumption and inflates the proxy.",
            "Bent produce hides true 3-D arc length; the proxy follows the "
            "projected path only.",
            "Not a validated 3-D volume and never a grading metric.",
        ],
    }


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
    pca_orientation_deg, pca_anisotropy = _pca_orientation(points_mm)

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

    # The skeleton pass costs tens of milliseconds; only pay it when the
    # policy asked for the curved centreline, or when the silhouette is
    # elongated enough for a centreline (and the slender volume proxy) to be
    # meaningful at all.
    centerline: CenterlineResult | None = None
    if metric == "centerline_length_mm" or elongation >= 2.0:
        centerline = centerline_analysis(
            contour,
            homography,
            prune_mm=calibration_settings.centerline_prune_mm,
            mm_per_px_approx=local_scale,
        )

    if metric == "min_area_rect_short_mm":
        value_mm = short_mm
        metric_unit = "mm"
    elif metric == "min_area_rect_long_mm":
        value_mm = long_mm
        metric_unit = "mm"
    elif metric == "max_feret_mm":
        value_mm = feret_mm
        metric_unit = "mm"
    elif metric == "centerline_length_mm":
        if centerline is None or centerline.status != "OK" or centerline.length_mm is None:
            status = centerline.status if centerline is not None else "NOT_COMPUTED"
            raise MetrologyError(
                f"Curved centreline refused ({status}): the skeleton is "
                "ambiguous. Separate touching produce or recapture the item "
                "flat; no straight-axis fallback is applied."
            )
        value_mm = float(centerline.length_mm)
        metric_unit = "mm"
    elif metric == "projected_area_cm2":
        value_mm = area_mm2 / 100.0
        metric_unit = "cm2"
    else:
        raise MetrologyError(f"Unsupported measurement metric: {metric!r}.")
    if not math.isfinite(value_mm) or value_mm <= 0:
        raise MetrologyError("Computed physical dimension is zero or non-finite.")

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
    terms = [print_scale_error, corner_error, boundary_error, parallax_error]
    if metric == "centerline_length_mm":
        # Both terminal radii compensate tip shortening; each carries a
        # boundary-scale error of its own.
        terms.append(2.0 * calibration_settings.boundary_uncertainty_px * local_scale)
    uncertainty = math.sqrt(sum(term**2 for term in terms))
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

    volume_proxy = _volume_proxy(centerline, elongation)
    warnings = [
        "Projected-plane measurement, not weight/volume and not a 3-D surface measurement.",
        "Parallax from object height above the marker plane is included only as a configured engineering bound, not a measured correction.",
    ]
    if metric == "centerline_length_mm":
        warnings.append(
            "Curved-centreline length: endpoint compensation adds the terminal "
            "silhouette radii; the true 3-D arc of a bent pod can exceed the "
            "projected path."
        )
    if centerline is not None and centerline.status != "OK":
        warnings.append(
            f"Curved-centreline measurement refused: {centerline.status}."
        )
    if volume_proxy is not None:
        warnings.append(
            "volume_proxy_cm3 is a model proxy assuming circular cross-sections "
            "along the projected centreline; it is not a validated 3-D volume, "
            "not a mass, and never a grading input."
        )
    if metric == "projected_area_cm2":
        warnings.append(
            "Projected area is reported in cm2; the numeric uncertainty "
            "field is an area envelope (cm2), not a length in mm."
        )

    measurement = PhysicalMeasurement(
        metric=metric,
        value_mm=float(value_mm),
        uncertainty_mm=float(uncertainty),
        lower_mm=float(value_mm - uncertainty),
        upper_mm=float(value_mm + uncertainty),
        unit=metric_unit,
        method=(
            "aruco_planar_homography_skeleton_centreline"
            if metric == "centerline_length_mm"
            else "aruco_planar_homography_contour_geometry"
        ),
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
            "pca_orientation_deg": pca_orientation_deg,
            "pca_anisotropy": pca_anisotropy,
            "local_mm_per_px": local_scale,
            "offset_from_marker_side_ratio": offset_ratio,
            "print_scale_error_mm": print_scale_error,
            "corner_error_mm": corner_error,
            "boundary_error_mm": boundary_error,
            "parallax_error_mm": parallax_error,
        },
        centerline=_centerline_summary(centerline),
        volume_proxy=volume_proxy,
        warnings=warnings,
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
