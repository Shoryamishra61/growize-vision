"""Reference-length calibration and marker-less end-to-end grading."""

from __future__ import annotations

import math

import numpy as np
import pytest

from growize_vision.config import CalibrationSettings
from growize_vision.domain import Status
from growize_vision.inference import InferenceRequest, InferenceService
from growize_vision.scale import calibrate_from_two_point_reference

SETTINGS = CalibrationSettings()


def test_two_point_reference_sets_uniform_metric_scale() -> None:
    calibration = calibrate_from_two_point_reference(
        (480, 640), (20.0, 20.0), (120.0, 20.0), 100.0, SETTINGS
    )
    assert calibration.valid
    assert calibration.source == "two_point_reference_length"
    assert calibration.mm_per_px == pytest.approx(1.0)
    assert calibration.marker_side_mm == 100.0
    assert calibration.marker_measurement_uncertainty_mm >= 0.5
    assert any("uniform scale" in warning for warning in calibration.warnings)


def test_two_point_reference_diagonal_span() -> None:
    calibration = calibrate_from_two_point_reference(
        (480, 640), (0.0, 0.0), (300.0, 400.0), 500.0, SETTINGS
    )
    assert calibration.valid
    assert calibration.mm_per_px == pytest.approx(1.0)  # 500 mm over 500 px


def test_two_point_reference_refuses_short_image_span() -> None:
    calibration = calibrate_from_two_point_reference(
        (480, 640), (10.0, 10.0), (20.0, 10.0), 100.0, SETTINGS
    )
    assert not calibration.valid
    assert any("px" in reason for reason in calibration.reasons)


def test_two_point_reference_refuses_bad_lengths_and_points() -> None:
    cases = [
        (((20.0, 20.0), (120.0, 20.0)), 0.0),
        (((20.0, 20.0), (120.0, 20.0)), 5000.0),
        (((20.0, 20.0), (120.0, 20.0)), float("nan")),
        (((-5.0, 20.0), (120.0, 20.0)), 100.0),
        (((20.0, 20.0), (9999.0, 20.0)), 100.0),
    ]
    for (a, b), length in cases:
        calibration = calibrate_from_two_point_reference((480, 640), a, b, length, SETTINGS)
        assert not calibration.valid


def _markerless_scene() -> np.ndarray:
    """Large saturated green blob on a light neutral mat; ~240x140 px.

    The object covers >5% of the frame so the whole-image contrast gate
    (p95-p05) sees it, and sensor-like noise keeps the blur gate happy.
    """
    import cv2

    rng = np.random.default_rng(11)
    scene = np.full((480, 640, 3), 200, np.uint8)
    scene = np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)
    cv2.ellipse(scene, (320, 240), (120, 70), 15, 0, 360, (60, 160, 60), -1)
    return np.clip(
        (scene.astype(int) + rng.integers(-7, 8, scene.shape)), 0, 255
    ).astype(np.uint8)


def test_markerless_image_grades_via_two_point_reference() -> None:
    service = InferenceService()
    result = service.inspect_frame(
        _markerless_scene(),
        request=InferenceRequest(
            crop_key="potato",
            reference_points_px=(20.0, 20.0, 120.0, 20.0),
            reference_length_mm=100.0,
        ),
    )
    assert result.status is Status.ACCEPTED
    assert result.size_band == "large"  # ~140 mm wide lands above the 60 mm band edge
    measurement = result.instances[0].measurement
    assert measurement.metric == "min_area_rect_short_mm"
    assert measurement.value_mm == pytest.approx(140.0, abs=4.0)
    assert any("uniform scale" in w for w in result.warnings)


def test_markerless_image_without_reference_still_refuses() -> None:
    service = InferenceService()
    result = service.inspect_frame(
        _markerless_scene(),
        request=InferenceRequest(crop_key="potato"),
    )
    assert result.status is Status.CALIBRATION_REQUIRED
    assert any("known length" in reason for reason in result.reasons)


def test_camera_prior_scale_is_uniform_with_wide_uncertainty() -> None:
    from growize_vision.scale import calibrate_from_camera_prior

    calibration = calibrate_from_camera_prior((480, 640), SETTINGS)
    assert calibration.valid
    assert calibration.source == "camera_prior_estimate"
    expected_scale = 300.0 / ((640 / 2) / math.tan(math.radians(66) / 2))
    assert calibration.mm_per_px == pytest.approx(expected_scale, rel=1e-9)
    # ±30% scale error carried through the envelope.
    assert calibration.marker_measurement_uncertainty_mm == pytest.approx(30.0)
    assert any("DISTANCE_ASSUMED" in w for w in calibration.warnings)


def test_single_okra_grades_with_no_scale_of_any_kind() -> None:
    service = InferenceService()
    result = service.inspect_frame(
        _markerless_scene(),
        request=InferenceRequest(crop_key="okra", allow_scale_estimate=True),
    )
    # Okra has no appearance priors, so the distance-assumed tier engages.
    assert result.calibration is not None
    assert result.calibration.source == "camera_prior_estimate"
    assert any(w.startswith("DISTANCE_ASSUMED") for w in result.warnings)
    assert result.status in (Status.ACCEPTED, Status.REVIEW_BORDERLINE)
    assert result.instances[0].measurement is not None
    assert result.instances[0].measurement.metric == "centerline_length_mm"


def test_scale_estimate_never_overrides_real_calibration_paths() -> None:
    service = InferenceService()
    result = service.inspect_frame(
        _markerless_scene(),
        request=InferenceRequest(crop_key="okra", allow_scale_estimate=False),
    )
    assert result.status is Status.CALIBRATION_REQUIRED
    assert result.calibration is None or not result.calibration.valid
