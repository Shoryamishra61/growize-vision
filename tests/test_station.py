"""Fixed-camera station automation: calibrate once, then grade hands-free."""

from __future__ import annotations

import numpy as np
import pytest

from growize_vision.config import load_inference_settings
from growize_vision.domain import Status
from growize_vision.inference import InferenceRequest, InferenceService
from growize_vision.markers import detect_best_marker
from growize_vision.media import load_image_file

SETTINGS = load_inference_settings()


def demo_frame(name: str) -> np.ndarray:
    return load_image_file(f"data/demo/{name}").bgr


def marker_painted_over(frame: np.ndarray) -> np.ndarray:
    """Erase the marker by filling its bbox with the surrounding mat colour."""
    marker, _ = detect_best_marker(frame, SETTINGS.calibration)
    assert marker is not None
    corners = np.asarray(marker.corners_px, dtype=float)
    x1, y1 = corners.reshape(-1, 2).min(axis=0).astype(int)
    x2, y2 = corners.reshape(-1, 2).max(axis=0).astype(int)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 - 8), max(0, y1 - 8)
    x2, y2 = min(w, x2 + 8), min(h, y2 + 8)
    patch = frame[max(0, y1 - 40) : y1].reshape(-1, 3)
    mat_colour = np.median(patch, axis=0).astype(frame.dtype)
    covered = frame.copy()
    covered[y1:y2, x1:x2] = mat_colour
    return covered


def test_station_commissioning_stores_calibration() -> None:
    service = InferenceService()
    result = service.inspect_frame(
        demo_frame("potato_big_synthetic.png"),
        request=InferenceRequest(
            crop_key="potato", marker_side_mm=50.0, station_mode=True, station_id="pack-line-1"
        ),
    )
    assert result.status is Status.ACCEPTED
    assert "pack-line-1" in service.station_calibrations


def _station_frames() -> tuple[np.ndarray, np.ndarray]:
    """(commissioning frame with marker, production frame with it covered)."""
    import cv2

    rng = np.random.default_rng(11)
    base = np.full((480, 640, 3), 200, np.uint8)
    base = np.clip(base.astype(int) + rng.integers(-7, 8, base.shape), 0, 255).astype(np.uint8)
    cv2.ellipse(base, (320, 240), (120, 70), 15, 0, 360, (60, 160, 60), -1)
    base = np.clip(base.astype(int) + rng.integers(-7, 8, base.shape), 0, 255).astype(np.uint8)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 120)
    commissioning = base.copy()
    # A 120 px marker claimed as 60 mm => 0.5 mm/px station scale.
    x0, y0 = 20, 340
    region = commissioning[y0 : y0 + 120, x0 : x0 + 120]
    region[marker > 127] = 255
    region[marker <= 127] = 0
    # Cover it with mat-coloured noise for the production frame (same shape).
    production = commissioning.copy()
    patch = np.clip(
        np.full((120, 120, 3), 200, np.int32)
        + rng.integers(-7, 8, (120, 120, 3)),
        0,
        255,
    ).astype(np.uint8)
    production[y0 : y0 + 120, x0 : x0 + 120] = patch
    return commissioning, production


def test_station_reuse_grades_without_visible_marker() -> None:
    service = InferenceService()
    commissioning, production = _station_frames()
    commission = service.inspect_frame(
        commissioning,
        request=InferenceRequest(
            crop_key="potato", marker_side_mm=60.0, station_mode=True, station_id="line-a"
        ),
    )
    assert commission.status is Status.ACCEPTED, commission.warnings
    result = service.inspect_frame(
        production,
        request=InferenceRequest(crop_key="potato", station_mode=True, station_id="line-a"),
    )
    assert result.status is Status.ACCEPTED, result.warnings
    assert result.calibration is not None and result.calibration.valid
    assert result.calibration.source == "printed_aruco_marker"
    assert any("STATION_CALIBRATION_REUSED" in w for w in result.warnings)
    measurement = result.instances[0].measurement
    assert measurement is not None
    # 140 px short axis at the stored 0.5 mm/px scale => ~70 mm => large band.
    assert measurement.value_mm == pytest.approx(70.0, abs=4.0)
    assert result.size_band == "large"


def test_without_station_mode_covered_marker_refuses() -> None:
    service = InferenceService()
    commissioning, production = _station_frames()
    service.inspect_frame(
        commissioning,
        request=InferenceRequest(
            crop_key="potato", marker_side_mm=60.0, station_mode=True, station_id="line-b"
        ),
    )
    result = service.inspect_frame(
        production,
        request=InferenceRequest(crop_key="potato", station_mode=False, station_id="line-b"),
    )
    assert result.status is Status.CALIBRATION_REQUIRED


def test_station_drift_recalibrates_and_warns() -> None:
    service = InferenceService()
    service.inspect_frame(
        demo_frame("potato_big_synthetic.png"),
        request=InferenceRequest(
            crop_key="potato", marker_side_mm=50.0, station_mode=True, station_id="line-c"
        ),
    )
    original_scale = service.station_calibrations["line-c"]["calibration"].mm_per_px
    result = service.inspect_frame(
        demo_frame("potato_big_synthetic.png"),
        request=InferenceRequest(
            crop_key="potato", marker_side_mm=25.0, station_mode=True, station_id="line-c"
        ),
    )
    # The marker now claims half the size: 100% scale disagreement.
    assert any("STATION_CALIBRATION_DRIFT" in w for w in result.warnings)
    updated_scale = service.station_calibrations["line-c"]["calibration"].mm_per_px
    assert updated_scale == pytest.approx(0.5 * original_scale, rel=1e-6)
    assert result.calibration is not None and result.calibration.valid


def test_strict_auto_crop_grades_with_published_reference(demo_dir) -> None:
    from pathlib import Path

    from growize_vision.crop_identity import SpeciesPrediction

    class FakeIdentity:
        is_classifier_available = True
        is_deep = False
        classifier_path = Path(__file__)
        deep_classifier_path = Path(__file__)

        @staticmethod
        def detect_produce_presence(frame):
            return True, "FAKE"

        @staticmethod
        def predict_species(frame):
            return SpeciesPrediction("potato", "Potato", 0.9, {"Potato": 0.9})

    service = InferenceService(identity_service=FakeIdentity())
    frame = load_image_file(demo_dir / "potato_big_synthetic.png").bgr
    result = service.inspect_frame(
        frame,
        request=InferenceRequest(
            crop_key="auto", marker_side_mm=50.0, station_mode=True, station_id="auto-line"
        ),
    )
    assert result.status is Status.ACCEPTED
    assert result.policy is not None and result.policy.crop_key == "potato"
    assert result.size_band == "large"
