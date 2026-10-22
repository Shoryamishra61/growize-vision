"""Projected-plane metrology tests with vector-defined ground truth."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from growize_vision.config import CalibrationSettings
from growize_vision.domain import Calibration
from growize_vision.geometry import (
    MetrologyError,
    measure_contour,
    overlay_axes_mm,
    transform_points,
)

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


def rectangle(cx: float, cy: float, width: float, height: float) -> np.ndarray:
    return np.array(
        [
            [cx - width / 2, cy - height / 2],
            [cx + width / 2, cy - height / 2],
            [cx + width / 2, cy + height / 2],
            [cx - width / 2, cy + height / 2],
        ],
        dtype=np.float32,
    )


def test_identity_homography_reports_pixel_units_as_millimetres() -> None:
    measurement, points_mm = measure_contour(
        rectangle(300, 220, 100, 60),
        calibration_with(np.eye(3)),
        "min_area_rect_short_mm",
        SETTINGS,
    )
    assert measurement.value_mm == pytest.approx(60.0, abs=1e-6)
    assert measurement.secondary["min_area_rect_long_mm"] == pytest.approx(100.0, abs=1e-6)
    assert points_mm.shape == (4, 2)


def test_uniform_scale_converts_consistently_and_is_scale_invariant() -> None:
    low_res = calibration_with(np.diag([0.5, 0.5, 1.0]))
    high_res = calibration_with(np.diag([0.125, 0.125, 1.0]))
    low_measurement, _ = measure_contour(
        rectangle(200, 150, 100, 60), low_res, "min_area_rect_short_mm", SETTINGS
    )
    high_measurement, _ = measure_contour(
        rectangle(800, 600, 400, 240), high_res, "min_area_rect_short_mm", SETTINGS
    )
    assert low_measurement.value_mm == pytest.approx(30.0, abs=1e-6)
    assert high_measurement.value_mm == pytest.approx(30.0, abs=1e-6)
    assert low_measurement.value_mm == pytest.approx(high_measurement.value_mm, abs=1e-6)


def test_perspective_homography_recovers_known_rectangle() -> None:
    src_mm = np.array([[10, 10], [90, 10], [90, 50], [10, 50]], dtype=np.float32)
    dst_px = np.array(
        [[120, 90], [640, 140], [690, 470], [80, 520]], dtype=np.float32
    )
    mm_to_px = cv2.getPerspectiveTransform(src_mm, dst_px)
    px_to_mm = np.linalg.inv(mm_to_px)
    measurement, _ = measure_contour(
        dst_px, calibration_with(px_to_mm), "min_area_rect_short_mm", SETTINGS
    )
    assert measurement.value_mm == pytest.approx(40.0, abs=1e-3)
    assert measurement.secondary["min_area_rect_long_mm"] == pytest.approx(80.0, abs=1e-3)


def test_rotated_rectangle_is_measured_by_min_area_rect() -> None:
    base = rectangle(0, 0, 120, 50)
    angle = math.radians(37)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    points = (base @ rotation.T) + np.array([500, 400])
    measurement, _ = measure_contour(
        points.astype(np.float32),
        calibration_with(np.eye(3)),
        "min_area_rect_short_mm",
        SETTINGS,
    )
    assert measurement.value_mm == pytest.approx(50.0, abs=1e-4)
    assert measurement.secondary["min_area_rect_long_mm"] == pytest.approx(120.0, abs=1e-4)


def test_long_axis_and_feret_metrics() -> None:
    calibration = calibration_with(np.eye(3))
    long_measurement, _ = measure_contour(
        rectangle(250, 250, 120, 50), calibration, "min_area_rect_long_mm", SETTINGS
    )
    feret_measurement, _ = measure_contour(
        rectangle(250, 250, 120, 50), calibration, "max_feret_mm", SETTINGS
    )
    assert long_measurement.value_mm == pytest.approx(120.0, abs=1e-4)
    assert feret_measurement.value_mm == pytest.approx(math.hypot(120, 50), abs=1e-3)


def test_projected_area_is_in_square_centimetres() -> None:
    measurement, _ = measure_contour(
        rectangle(250, 250, 80, 40),
        calibration_with(np.eye(3)),
        "projected_area_cm2",
        SETTINGS,
    )
    assert measurement.unit == "cm2"
    assert measurement.value_mm == pytest.approx(32.0, abs=1e-4)
    assert measurement.secondary["area_mm2"] == pytest.approx(3200.0, abs=1e-2)
    assert measurement.uncertainty_mm >= 0.04 * measurement.value_mm
    assert any("area envelope" in warning for warning in measurement.warnings)


def test_uncertainty_envelope_is_positive_and_brackets_value() -> None:
    measurement, _ = measure_contour(
        rectangle(250, 250, 80, 40),
        calibration_with(np.eye(3), side_mm=50.0),
        "min_area_rect_short_mm",
        SETTINGS,
    )
    assert measurement.uncertainty_mm > 0
    assert measurement.lower_mm < measurement.value_mm < measurement.upper_mm
    assert measurement.lower_mm == pytest.approx(
        measurement.value_mm - measurement.uncertainty_mm
    )
    assert measurement.coverage_note == "engineering_error_bound_not_statistical_95pct"
    assert any("not weight/volume" in warning for warning in measurement.warnings)


def test_invalid_calibration_refuses_measurement() -> None:
    calibration = Calibration(valid=False, calibration_id="none")
    with pytest.raises(MetrologyError, match="valid planar calibration"):
        measure_contour(
            rectangle(100, 100, 50, 50), calibration, "min_area_rect_short_mm", SETTINGS
        )


def test_calibration_without_homography_refuses_measurement() -> None:
    calibration = Calibration(valid=True, calibration_id="missing", marker_side_mm=50.0)
    with pytest.raises(MetrologyError, match="valid planar calibration"):
        measure_contour(
            rectangle(100, 100, 50, 50), calibration, "min_area_rect_short_mm", SETTINGS
        )


def test_degenerate_contour_refuses_measurement() -> None:
    calibration = calibration_with(np.eye(3))
    with pytest.raises(MetrologyError, match="fewer than three"):
        measure_contour(
            np.array([[10, 10], [20, 20]], dtype=np.float32),
            calibration,
            "min_area_rect_short_mm",
            SETTINGS,
        )
    with pytest.raises(MetrologyError, match="zero or invalid area"):
        measure_contour(
            np.array([[10, 10], [10, 10], [10, 10]], dtype=np.float32),
            calibration,
            "min_area_rect_short_mm",
            SETTINGS,
        )


def test_singular_projection_refuses_measurement() -> None:
    calibration = calibration_with(np.zeros((3, 3)))
    with pytest.raises(MetrologyError):
        measure_contour(
            rectangle(100, 100, 50, 50), calibration, "min_area_rect_short_mm", SETTINGS
        )


def test_non_finite_projection_refuses_measurement() -> None:
    homography = np.eye(3)
    homography[0, 0] = np.nan
    calibration = calibration_with(homography)
    with pytest.raises(MetrologyError, match="non-finite"):
        measure_contour(
            rectangle(100, 100, 50, 50), calibration, "min_area_rect_short_mm", SETTINGS
        )


def test_unsupported_metric_refuses() -> None:
    with pytest.raises(MetrologyError, match="Unsupported measurement metric"):
        measure_contour(
            rectangle(100, 100, 50, 50),
            calibration_with(np.eye(3)),
            "estimated_weight_g",
            SETTINGS,
        )


def test_missing_marker_side_cannot_claim_millimetres() -> None:
    calibration = calibration_with(np.eye(3), side_mm=0.0)
    calibration.marker_side_mm = None
    with pytest.raises(MetrologyError, match="actual marker side"):
        measure_contour(
            rectangle(100, 100, 50, 50), calibration, "min_area_rect_short_mm", SETTINGS
        )


def test_transform_points_rejects_non_finite_output() -> None:
    homography = np.eye(3)
    homography[1, 1] = np.inf
    with pytest.raises(MetrologyError, match="non-finite"):
        transform_points(np.array([[1.0, 1.0]], dtype=np.float32), homography)


def test_overlay_axes_returns_two_orthogonal_axes() -> None:
    points_mm = np.array(
        [[0, 0], [80, 0], [80, 40], [0, 40]], dtype=np.float32
    )
    axes = overlay_axes_mm(points_mm)
    for key in ("long_axis", "short_axis"):
        x1, y1, x2, y2 = axes[key]
        assert math.isfinite(math.hypot(x2 - x1, y2 - y1))
    long_length = math.hypot(
        axes["long_axis"][2] - axes["long_axis"][0],
        axes["long_axis"][3] - axes["long_axis"][1],
    )
    short_length = math.hypot(
        axes["short_axis"][2] - axes["short_axis"][0],
        axes["short_axis"][3] - axes["short_axis"][1],
    )
    assert {round(long_length, 6), round(short_length, 6)} == {40.0, 80.0}
