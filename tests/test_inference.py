"""End-to-end contract tests for the shared inference engine."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from growize_vision.domain import SizeClass, SourceType, Status
from growize_vision.inference import InferenceRequest
from growize_vision.media import load_image_file

DEMO_CASES = [
    ("potato_small_synthetic.png", "potato", Status.ACCEPTED, SizeClass.SMALL, 35.079),
    ("potato_big_synthetic.png", "potato", Status.ACCEPTED, SizeClass.BIG, 72.131),
    ("okra_small_synthetic.png", "okra", Status.ACCEPTED, SizeClass.SMALL, 81.997),
    ("okra_big_synthetic.png", "okra", Status.ACCEPTED, SizeClass.BIG, 142.257),
    ("no_marker_synthetic.png", "potato", Status.CALIBRATION_REQUIRED, None, None),
    ("empty_mat_synthetic.png", "potato", Status.NO_OBJECT, None, None),
    ("blurred_synthetic.png", "potato", Status.LOW_IMAGE_QUALITY, None, None),
    ("edge_clipped_synthetic.png", "potato", Status.OCCLUDED, None, None),
]


def inspect_demo(service, demo_dir: Path, filename: str, crop: str, *, demo: bool = True):
    decoded = load_image_file(demo_dir / filename)
    request = InferenceRequest(
        crop_key=crop,
        marker_side_mm=50.0,
        marker_measurement_uncertainty_mm=0.5,
        expected_marker_id=0,
        demo_policy=demo,
    )
    return service.inspect_frame(decoded.bgr, request=request)


@pytest.mark.parametrize(
    ("filename", "crop", "status", "size_class", "actual_mm"), DEMO_CASES
)
def test_synthetic_fixtures_follow_the_contract(
    service, demo_dir, filename, crop, status, size_class, actual_mm
) -> None:
    result = inspect_demo(service, demo_dir, filename, crop)
    assert result.status is status
    assert result.size_class is size_class
    assert result.context.source_type is SourceType.IMAGE
    assert result.original_shape is not None
    assert result.processed_shape is not None
    if status is Status.ACCEPTED:
        assert result.instances
        decision = result.instances[0].decision
        assert decision is not None
        assert decision.calibration_id == result.calibration.calibration_id
        assert decision.policy_version
        assert result.policy is not None and result.policy.demo
        assert result.calibration is not None and result.calibration.valid
        measurement = result.instances[0].measurement
        assert measurement is not None
        assert measurement.value_mm == pytest.approx(actual_mm, abs=1.0)
        assert measurement.lower_mm < measurement.value_mm < measurement.upper_mm
    else:
        assert result.size_class is None
        assert all(instance.measurement is None for instance in result.instances)


def test_strict_mode_withholds_measurement_and_grade(service, demo_dir) -> None:
    result = inspect_demo(
        service, demo_dir, "potato_big_synthetic.png", "potato", demo=False
    )
    assert result.status is Status.POLICY_REQUIRED
    assert result.size_class is None
    assert result.policy is not None
    assert result.policy.demo is False
    assert result.policy.threshold_mm is None
    assert result.instances
    assert result.instances[0].measurement is None
    assert "NO_APPROVED_THRESHOLD" in result.reasons[0]


def test_missing_marker_never_produces_millimetres(service, demo_dir) -> None:
    result = inspect_demo(service, demo_dir, "no_marker_synthetic.png", "potato")
    assert result.status is Status.CALIBRATION_REQUIRED
    assert result.calibration is not None
    assert result.calibration.valid is False
    assert result.size_class is None
    assert all(instance.measurement is None for instance in result.instances)
    assert any("marker" in reason.lower() for reason in result.reasons)


def test_unknown_crop_is_refused_without_guessing(service, demo_dir) -> None:
    result = inspect_demo(service, demo_dir, "potato_big_synthetic.png", "dragonfruit")
    assert result.status is Status.POLICY_REQUIRED
    assert result.size_class is None
    assert "Unknown crop policy" in " ".join(result.reasons)


def test_no_crop_selection_is_refused(service, demo_dir) -> None:
    result = inspect_demo(service, demo_dir, "potato_big_synthetic.png", None)
    assert result.status is Status.POLICY_REQUIRED
    assert result.size_class is None


def test_neural_mode_is_explicitly_unavailable(service, demo_dir) -> None:
    decoded = load_image_file(demo_dir / "potato_big_synthetic.png")
    result = service.inspect_frame(
        decoded.bgr,
        request=InferenceRequest(
            crop_key="potato",
            marker_side_mm=50.0,
            segmentation_mode="neural",
        ),
    )
    assert result.status is Status.MODEL_UNAVAILABLE
    assert result.size_class is None
    assert "not installed" in result.reasons[0]
    assert result.annotated_image is not None


def test_invalid_frame_returns_invalid_input(service) -> None:
    result = service.inspect_frame(np.zeros((5, 5, 3), dtype=np.uint8))
    assert result.status is Status.INVALID_INPUT
    assert result.size_class is None
    assert result.instances == []
    assert result.reasons


def test_empty_mat_has_no_object_even_with_valid_calibration(service, demo_dir) -> None:
    result = inspect_demo(service, demo_dir, "empty_mat_synthetic.png", "potato")
    assert result.status is Status.NO_OBJECT
    assert result.size_class is None
    assert result.instances == []
    assert result.calibration is not None and result.calibration.valid


def test_two_objects_never_receive_a_single_aggregate_grade(service, demo_dir) -> None:
    decoded = load_image_file(demo_dir / "empty_mat_synthetic.png")
    scene = decoded.bgr.copy()
    cv2.ellipse(scene, (760, 620), (160, 95), 12, 0, 360, (58, 112, 62), -1)
    cv2.ellipse(scene, (1110, 620), (150, 88), -18, 0, 360, (72, 118, 55), -1)
    result = service.inspect_frame(
        scene,
        request=InferenceRequest(
            crop_key="potato",
            marker_side_mm=50.0,
            demo_policy=True,
            expected_marker_id=0,
        ),
    )
    assert result.status is Status.MULTIPLE_OBJECTS
    assert result.size_class is None
    assert len(result.instances) >= 2
    assert all(instance.decision is not None for instance in result.instances)


def test_only_accepted_results_carry_small_or_big(service, demo_dir) -> None:
    for filename, crop, *_ in DEMO_CASES:
        result = inspect_demo(service, demo_dir, filename, crop)
        if result.status is Status.ACCEPTED:
            assert result.size_class in (SizeClass.SMALL, SizeClass.BIG)
        else:
            assert result.size_class is None


def test_result_contract_is_json_serialisable_and_auditable(service, demo_dir) -> None:
    result = inspect_demo(service, demo_dir, "potato_big_synthetic.png", "potato")
    payload = result.to_dict()
    serialised = json.dumps(payload)
    assert len(serialised) > 100
    assert payload["accepted"] is True
    assert payload["status"] == "ACCEPTED"
    assert payload["size_class"] == "BIG"
    assert payload["calibration"]["valid"] is True
    assert payload["policy"]["demo"] is True
    assert payload["model_sha"] is None
    assert payload["pipeline_sha256"]
    assert payload["pipeline_version"]
    assert payload["software_version"]
    assert payload["original_shape"] == list(result.original_shape)
    assert len(payload["pipeline_sha256"]) == 64
    for instance in payload["instances"]:
        assert "mask" not in instance
    assert set(result.timings_ms) >= {
        "preprocess_ms",
        "calibration_ms",
        "segmentation_ms",
        "quality_ms",
        "render_ms",
    }
    assert result.processing_ms > 0


def test_repeated_runs_are_deterministic(service, demo_dir) -> None:
    first = inspect_demo(service, demo_dir, "okra_big_synthetic.png", "okra")
    second = inspect_demo(service, demo_dir, "okra_big_synthetic.png", "okra")
    assert first.status is second.status
    assert first.size_class is second.size_class
    assert first.instances[0].measurement is not None
    assert second.instances[0].measurement is not None
    assert first.instances[0].measurement.value_mm == pytest.approx(
        second.instances[0].measurement.value_mm, abs=1e-9
    )
    assert first.calibration.calibration_id == second.calibration.calibration_id


def test_missing_marker_side_degrades_to_calibration_required(service, demo_dir) -> None:
    decoded = load_image_file(demo_dir / "potato_big_synthetic.png")
    result = service.inspect_frame(
        decoded.bgr,
        request=InferenceRequest(
            crop_key="potato",
            marker_side_mm=None,
            expected_marker_id=0,
            demo_policy=True,
        ),
    )
    assert result.status is Status.CALIBRATION_REQUIRED
    assert result.size_class is None
    assert all(instance.measurement is None for instance in result.instances)


def test_measurement_is_finite_and_policy_metric_matches(service, demo_dir) -> None:
    result = inspect_demo(service, demo_dir, "okra_big_synthetic.png", "okra")
    measurement = result.instances[0].measurement
    assert measurement is not None
    assert measurement.metric == "min_area_rect_long_mm"
    assert math.isfinite(measurement.value_mm)
    assert math.isfinite(measurement.uncertainty_mm)
    assert measurement.unit == "mm"
    assert measurement.calibration_id == result.calibration.calibration_id
