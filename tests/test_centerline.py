"""Curved-centreline skeleton metrology tests with analytic ground truth."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from growize_vision.centerline import (
    _adjacency,
    _prune_spurs,
    centerline_analysis,
)
from growize_vision.config import CalibrationSettings
from growize_vision.domain import Calibration
from growize_vision.geometry import MetrologyError, measure_contour

SETTINGS = CalibrationSettings()


def calibration_with(homography: np.ndarray, *, side_mm: float = 50.0) -> Calibration:
    return Calibration(
        valid=True,
        calibration_id="calib_test",
        marker_id=0,
        marker_side_mm=side_mm,
        homography=np.asarray(homography, dtype=float).tolist(),
        mm_per_px=float(side_mm / 100.0),
        corner_reprojection_rms_px=0.01,
        marker_measurement_uncertainty_mm=0.5,
    )


def stroke_mask(
    path_points: list[tuple[float, float]], radius_px: float, canvas: tuple[int, int]
) -> np.ndarray:
    """Capsule mask: Minkowski sum of the polyline with a disk."""
    mask = np.zeros(canvas, dtype=np.uint8)
    dense: list[tuple[float, float]] = []
    for (x1, y1), (x2, y2) in zip(path_points, path_points[1:]):
        steps = max(2, int(math.hypot(x2 - x1, y2 - y1)))
        for t in np.linspace(0.0, 1.0, steps, endpoint=False):
            dense.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    dense.append(path_points[-1])
    for x, y in dense:
        cv2.circle(mask, (int(round(x)), int(round(y))), int(round(radius_px)), 1, -1)
    return mask


def path_length(path_points: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(path_points, path_points[1:])
    )


def mask_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea).reshape(-1, 2)


def quarter_arc_path(
    cx: float, cy: float, radius: float, samples: int = 60
) -> list[tuple[float, float]]:
    angles = np.linspace(0.0, math.pi / 2.0, samples)
    return [
        (cx + radius * math.cos(a), cy - radius * math.sin(a)) for a in angles
    ]


def test_straight_capsule_length_matches_geometry() -> None:
    path = [(60.0, 100.0), (240.0, 100.0)]
    radius = 12.0
    mask = stroke_mask(path, radius, (220, 320))
    result = centerline_analysis(
        mask_contour(mask), np.eye(3), prune_mm=3.0, mm_per_px_approx=1.0
    )
    assert result.status == "OK"
    expected = path_length(path) + 2.0 * radius  # 180 + 24 = 204
    assert result.length_mm == pytest.approx(expected, rel=0.05)


def test_quarter_arc_capsule_recovers_curved_length() -> None:
    radius_arc = 90.0
    radius_body = 10.0
    path = quarter_arc_path(120.0, 200.0, radius_arc)
    mask = stroke_mask(path, radius_body, (360, 360))
    result = centerline_analysis(
        mask_contour(mask), np.eye(3), prune_mm=3.0, mm_per_px_approx=1.0
    )
    assert result.status == "OK"
    expected = (radius_arc * math.pi / 2.0) + 2.0 * radius_body
    assert result.length_mm == pytest.approx(expected, abs=6.0)
    # The straight-axis proxy must underestimate the curved pod.
    _, points_mm = measure_contour(
        mask_contour(mask), calibration_with(np.eye(3)), "min_area_rect_long_mm", SETTINGS
    )
    straight = max(
        cv2.minAreaRect(points_mm.astype(np.float32).reshape(-1, 1, 2))[1]
    )
    assert straight < expected - 10.0


def test_centerline_is_scale_invariant_in_millimetres() -> None:
    path = quarter_arc_path(120.0, 200.0, 90.0)
    mask = stroke_mask(path, 10.0, (360, 360))
    contour = mask_contour(mask)
    full = centerline_analysis(contour, np.eye(3), prune_mm=3.0, mm_per_px_approx=1.0)
    # Half-resolution camera: the same physical object spans twice the pixels
    # and the px->mm homography halves, so millimetres must match.
    half = centerline_analysis(
        contour * 2.0, np.diag([0.5, 0.5, 1.0]), prune_mm=3.0, mm_per_px_approx=0.5
    )
    assert full.status == "OK" and half.status == "OK"
    assert half.length_mm == pytest.approx(full.length_mm, rel=0.05)


def test_long_side_branch_is_refused_not_measured() -> None:
    main = [(60.0, 100.0), (240.0, 100.0)]
    fork = [(150.0, 100.0), (150.0, 40.0)]  # 60 px spur, far above the prune length
    mask = stroke_mask(main, 12.0, (220, 320)) | stroke_mask(fork, 6.0, (220, 320))
    result = centerline_analysis(
        mask_contour(mask), np.eye(3), prune_mm=3.0, mm_per_px_approx=1.0
    )
    assert result.status == "AMBIGUOUS_SKELETON_BRANCHED"
    assert result.length_mm is None
    assert result.endpoint_count > 2


def test_measure_contour_refuses_ambiguous_centreline_metric() -> None:
    main = [(60.0, 100.0), (240.0, 100.0)]
    fork = [(150.0, 100.0), (150.0, 40.0)]
    mask = stroke_mask(main, 12.0, (220, 320)) | stroke_mask(fork, 6.0, (220, 320))
    with pytest.raises(MetrologyError, match="refused"):
        measure_contour(
            mask_contour(mask),
            calibration_with(np.eye(3)),
            "centerline_length_mm",
            SETTINGS,
        )


def test_measure_contour_centreline_metric_matches_arc() -> None:
    path = quarter_arc_path(120.0, 200.0, 90.0)
    mask = stroke_mask(path, 10.0, (360, 360))
    measurement, _ = measure_contour(
        mask_contour(mask),
        calibration_with(np.eye(3)),
        "centerline_length_mm",
        SETTINGS,
    )
    expected = (90.0 * math.pi / 2.0) + 20.0
    assert measurement.metric == "centerline_length_mm"
    assert measurement.method == "aruco_planar_homography_skeleton_centreline"
    assert measurement.value_mm == pytest.approx(expected, abs=6.0)
    assert measurement.centerline is not None
    assert measurement.centerline["status"] == "OK"
    assert any("endpoint compensation" in w for w in measurement.warnings)


def test_volume_proxy_tracks_capsule_volume() -> None:
    path = [(60.0, 100.0), (220.0, 100.0)]
    radius = 10.0
    mask = stroke_mask(path, radius, (220, 320))
    result = centerline_analysis(
        mask_contour(mask), np.eye(3), prune_mm=3.0, mm_per_px_approx=1.0
    )
    assert result.status == "OK"
    true_volume_px3 = (
        math.pi * radius**2 * path_length(path) + (4.0 / 3.0) * math.pi * radius**3
    )
    assert result.volume_proxy_cm3 is not None
    assert result.volume_proxy_cm3 * 1000.0 == pytest.approx(
        true_volume_px3, rel=0.10
    )


def test_geometry_attaches_labelled_volume_proxy_and_pca() -> None:
    path = [(60.0, 100.0), (220.0, 100.0)]
    mask = stroke_mask(path, 10.0, (220, 320))
    measurement, _ = measure_contour(
        mask_contour(mask),
        calibration_with(np.eye(3)),
        "min_area_rect_long_mm",
        SETTINGS,
    )
    assert measurement.volume_proxy is not None
    assert measurement.volume_proxy["value_cm3"] > 0
    assert "Not a validated 3-D volume" in measurement.volume_proxy["limits"][-1]
    assert 0.0 <= measurement.secondary["pca_orientation_deg"] < 180.0
    assert measurement.secondary["pca_anisotropy"] > 1.0
    assert any("model proxy" in w for w in measurement.warnings)


def test_boxy_silhouette_skips_skeleton_and_proxy() -> None:
    # 3:2 rectangle: curvature signal is high but elongation 1.5 < 2.0, so
    # the skeleton pass is skipped entirely for compact produce.
    rectangle = np.array(
        [
            [200.0, 200.0],
            [320.0, 200.0],
            [320.0, 266.0],
            [200.0, 266.0],
        ],
        dtype=np.float32,
    )
    measurement, _ = measure_contour(
        rectangle, calibration_with(np.eye(3)), "min_area_rect_long_mm", SETTINGS
    )
    assert measurement.centerline is None
    assert measurement.volume_proxy is None


def _skeleton_from_pixels(pixels: list[tuple[int, int]]) -> np.ndarray:
    skeleton = np.zeros((60, 60), dtype=np.uint8)
    for y, x in pixels:
        skeleton[y, x] = 1
    return skeleton


def test_prune_spurs_removes_short_branch_keeps_main_path() -> None:
    # Vertical main path with a 3 px spur; prune threshold 5 px.
    pixels = [(y, 10) for y in range(5, 45)]
    pixels += [(11, 11), (11, 12), (11, 13)]
    graph = _adjacency(_skeleton_from_pixels(pixels))
    pruned = _prune_spurs(graph, 5.0)
    endpoints = [node for node, links in graph.items() if len(links) == 1]
    assert pruned == 1
    assert len(endpoints) == 2
    assert (11, 10) in graph  # the junction on the main path survives


def test_prune_spurs_keeps_long_branch() -> None:
    pixels = [(y, 10) for y in range(5, 45)]
    pixels += [(20, x) for x in range(11, 21)]  # 10 px perpendicular spur
    graph = _adjacency(_skeleton_from_pixels(pixels))
    pruned = _prune_spurs(graph, 5.0)
    endpoints = [node for node, links in graph.items() if len(links) == 1]
    assert pruned == 0
    assert len(endpoints) == 3
