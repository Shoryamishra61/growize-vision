"""ArUco detection and calibration safety tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from growize_vision.config import CalibrationSettings
from growize_vision.domain import MarkerDetection
from growize_vision.geometry import transform_points
from growize_vision.markers import (
    MarkerError,
    calibrate_from_marker,
    detect_best_marker,
    detect_markers,
    generate_marker,
    marker_mask,
    marker_polygon,
    marker_side_summary,
)

SETTINGS = CalibrationSettings()


def matte(height: int = 900, width: int = 900) -> np.ndarray:
    rng = np.random.default_rng(7)
    image = np.full((height, width, 3), 176, dtype=np.float32)
    image += rng.normal(0, 1.4, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def marker_scene() -> np.ndarray:
    image = matte()
    canvas = generate_marker(
        marker_id=0,
        dictionary_name="DICT_4X4_50",
        side_mm=50.0,
        dpi=300,
        quiet_zone_mm=10.0,
    )
    image[30 : 30 + canvas.shape[0], 30 : 30 + canvas.shape[1]] = canvas
    return image


def test_generated_marker_is_detected_and_calibrated() -> None:
    scene = marker_scene()
    marker, reasons = detect_best_marker(scene, SETTINGS, expected_id=0)
    assert reasons == []
    assert marker is not None
    assert marker.marker_id == 0
    assert marker.side_px > SETTINGS.min_marker_side_px
    assert marker.area_px > 0

    calibration = calibrate_from_marker(scene.shape, marker, 50.0, SETTINGS, 0.5)
    assert calibration.valid
    assert calibration.marker_id == 0
    assert calibration.marker_side_mm == 50.0
    assert calibration.mm_per_px == pytest.approx(50.0 / marker.side_px)
    assert calibration.corner_reprojection_rms_px < 1e-3
    assert calibration.calibration_id.startswith("calib_aruco_0_")
    assert "coplanar" in calibration.warnings[-1]

    projected = transform_points(
        np.asarray(marker.corners_px, dtype=np.float32),
        np.asarray(calibration.homography, dtype=float),
    )
    assert projected[:, 0].min() == pytest.approx(0.0, abs=1e-3)
    assert projected[:, 0].max() == pytest.approx(50.0, abs=1e-3)
    assert projected[:, 1].min() == pytest.approx(0.0, abs=1e-3)
    assert projected[:, 1].max() == pytest.approx(50.0, abs=1e-3)


def test_perspective_marker_is_detected_and_calibrated() -> None:
    scene = marker_scene()
    source = np.array(
        [[30, 30], [857, 30], [857, 857], [30, 857]], dtype=np.float32
    )
    destination = np.array(
        [[140, 110], [780, 180], [740, 820], [90, 700]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        scene,
        transform,
        (1000, 900),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    marker, reasons = detect_best_marker(warped, SETTINGS, expected_id=0)
    assert marker is not None, reasons
    calibration = calibrate_from_marker(warped.shape, marker, 50.0, SETTINGS)
    assert calibration.valid
    assert calibration.mm_per_px > 0
    assert len(calibration.homography or []) == 3


def test_no_marker_returns_specific_reason() -> None:
    marker, reasons = detect_best_marker(matte(500, 500), SETTINGS, expected_id=0)
    assert marker is None
    assert "No ArUco marker detected" in reasons[0]


def test_wrong_marker_id_is_not_silently_accepted() -> None:
    scene = marker_scene()
    marker, reasons = detect_best_marker(scene, SETTINGS, expected_id=7)
    assert marker is None
    assert "Expected marker ID 7" in reasons[0]
    assert "0" in reasons[0]


def test_all_marker_candidates_are_reported_for_diagnostics() -> None:
    scene = marker_scene()
    markers = detect_markers(scene, SETTINGS, expected_id=None)
    assert len(markers) == 1
    assert markers[0].marker_id == 0
    assert not markers[0].reasons


def test_calibration_requires_a_measured_side_length() -> None:
    scene = marker_scene()
    marker, _ = detect_best_marker(scene, SETTINGS, expected_id=0)
    assert marker is not None
    calibration = calibrate_from_marker(scene.shape, marker, None, SETTINGS)
    assert not calibration.valid
    assert calibration.calibration_id == "unavailable"
    assert "actual printed marker side length" in calibration.reasons[0]


@pytest.mark.parametrize("side_mm", [0.0, 4.9, 1000.1, float("nan"), float("inf")])
def test_implausible_side_lengths_are_rejected(side_mm: float) -> None:
    scene = marker_scene()
    marker, _ = detect_best_marker(scene, SETTINGS, expected_id=0)
    assert marker is not None
    calibration = calibrate_from_marker(scene.shape, marker, side_mm, SETTINGS)
    assert not calibration.valid
    assert calibration.reasons


def test_tiny_marker_pixels_are_rejected_even_if_detected() -> None:
    marker = MarkerDetection(
        marker_id=0,
        corners_px=[(10, 10), (30, 10), (30, 30), (10, 30)],
        side_px=20.0,
        area_px=400.0,
        dictionary="DICT_4X4_50",
    )
    calibration = calibrate_from_marker((400, 400, 3), marker, 50.0, SETTINGS)
    assert not calibration.valid
    assert "only 20.0 px" in calibration.reasons[0]


def test_marker_with_geometry_reasons_is_rejected_directly() -> None:
    marker = MarkerDetection(
        marker_id=0,
        corners_px=[(10, 10), (300, 10), (300, 300), (10, 300)],
        side_px=290.0,
        area_px=84100.0,
        dictionary="DICT_4X4_50",
        reasons=["Marker quadrilateral is not a clean planar square."],
    )
    calibration = calibrate_from_marker((500, 500, 3), marker, 50.0, SETTINGS)
    assert not calibration.valid
    assert "not a clean planar square" in calibration.reasons[0]


def test_marker_mask_covers_the_marker_with_margin() -> None:
    scene = marker_scene()
    marker, _ = detect_best_marker(scene, SETTINGS, expected_id=0)
    assert marker is not None
    mask = marker_mask(scene.shape, marker, margin_px=12)
    points = marker_polygon(marker).reshape(-1, 2)
    for x, y in points:
        assert mask[int(y), int(x)] == 255
    assert mask.sum() > marker.area_px
    assert marker_mask(scene.shape, None).sum() == 0


def test_marker_side_summary_is_json_friendly() -> None:
    scene = marker_scene()
    marker, _ = detect_best_marker(scene, SETTINGS, expected_id=0)
    assert marker is not None
    summary = marker_side_summary(marker)
    assert summary["marker_id"] == 0
    assert isinstance(summary["side_px"], float)
    assert len(summary["corners_px"]) == 4


@pytest.mark.parametrize(
    ("side_mm", "dpi"),
    [(0.0, 300), (-1.0, 300), (1001.0, 300), (50.0, 71)],
)
def test_marker_generator_rejects_invalid_parameters(side_mm: float, dpi: int) -> None:
    with pytest.raises(MarkerError):
        generate_marker(side_mm=side_mm, dpi=dpi)


def test_empty_frame_has_no_markers() -> None:
    assert detect_markers(np.zeros((0, 0, 3), dtype=np.uint8), SETTINGS) == []
