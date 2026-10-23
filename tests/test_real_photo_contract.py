"""Provenance contract checks without creating or evaluating synthetic images."""

from pathlib import Path

import pytest

from growize_vision.real_photos import verify_original_photo, verify_specimen_splits


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"is_original_capture": True},
        {"is_original_capture": True, "synthetic": True, "augmented": False},
        {"is_original_capture": True, "synthetic": False, "augmented": True},
    ],
)
def test_unverified_or_generated_photos_are_ineligible(flags):
    with pytest.raises(ValueError, match="original photographs"):
        verify_original_photo(flags, Path("."))


@pytest.mark.parametrize("identity", ["specimen_id", "capture_id", "sha256"])
def test_shared_identity_cannot_cross_splits(identity):
    left = dict(specimen_id="a", capture_id="a", sha256="a", split="train")
    right = dict(specimen_id="b", capture_id="b", sha256="b", split="test")
    right[identity] = left[identity]
    with pytest.raises(ValueError, match="multiple splits"):
        verify_specimen_splits([left, right])


def test_repeated_capture_in_same_split_is_allowed():
    row = dict(specimen_id="a", capture_id="b", sha256="c", split="test")
    verify_specimen_splits([row, row.copy()])


def test_empty_physical_validation_is_blocked(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts.validate_real_size import evaluate

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    report = evaluate(manifest, tmp_path, tmp_path / "report.json")
    assert report["status"] == "BLOCKED"
    assert report["accuracy"] is None
    assert report["original_photos"] == 0


def test_empty_ui_input_clears_output(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from apps.gradio_app import inspect_image_ui, live_inspect_ui, process_video_ui
    from growize_vision.tracking import LiveSession

    image, message = inspect_image_ui(None, "Potato", None, [], None, "Ruler (±1 mm)")
    assert image is None and "Image needed" in message
    image, message, session = live_inspect_ui(None, LiveSession(), "Potato")
    assert image is None and session.last_annotated_rgb is None
    video, message = process_video_ui(None, "Potato")
    assert video is None and "Video needed" in message


def test_reference_average_counts_items_not_photos(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts.build_size_reference import reference_statistics

    first = dict(
        split="train",
        specimen_id="a",
        measured_value=80.0,
        measurement_unit="mm",
        measurement_reviewed_by="reviewer",
        crop_key="okra",
        physical_unit="single_pod",
        metric="min_area_rect_long_mm",
    )
    second = dict(first, specimen_id="b", measured_value=120.0)
    held_out = dict(first, specimen_id="c", split="test", measured_value=500.0)
    references = reference_statistics([first, first.copy(), second, held_out])
    assert references[0]["mean"] == 100.0
    assert references[0]["specimens"] == 2
    assert references[0]["status"] == "REFERENCE_NOT_VALIDATED"


def test_reference_rejects_inconsistent_repeat_measurements(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts.build_size_reference import reference_statistics

    first = dict(
        split="train",
        specimen_id="a",
        measured_value=80.0,
        measurement_unit="mm",
        measurement_reviewed_by="reviewer",
        crop_key="okra",
        physical_unit="single_pod",
        metric="min_area_rect_long_mm",
    )
    with pytest.raises(ValueError, match="inconsistent"):
        reference_statistics([first, dict(first, measured_value=120.0)])


def test_empirical_average_is_not_reported_as_approved_standard():
    from growize_vision.domain import GradePolicy, PhysicalMeasurement, Status
    from growize_vision.grading import evaluate_measurement

    policy = GradePolicy(
        crop_key="okra",
        crop_name="Okra",
        physical_unit="single_pod",
        metric="min_area_rect_long_mm",
        metric_description="Pod length",
        threshold_mm=100,
        approval_status="EMPIRICAL_REFERENCE",
    )
    measurement = PhysicalMeasurement(
        metric=policy.metric, value_mm=120, uncertainty_mm=1, lower_mm=119, upper_mm=121
    )
    result = evaluate_measurement(measurement, policy, "calibration-receipt")
    assert result.status == Status.ACCEPTED
    assert "APPROVED_POLICY" not in result.reasons
    assert result.warning and "Sample reference" in result.warning


def test_size_evaluation_uses_frozen_training_median_without_image_inference(tmp_path, monkeypatch):
    """Control-flow contract only: the image decoder and engine are mocked."""
    import json
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from growize_vision.config import load_policies
    from growize_vision.domain import InspectionResult, MediaContext, SizeClass, Status
    from scripts import validate_real_size as evaluation

    rows = []
    for index, (split, value) in enumerate(
        [("train", 80), ("train", 100), ("train", 120), ("train", 500), ("test", 70), ("test", 130)]
    ):
        rows.append(
            dict(
                split=split,
                specimen_id=str(index),
                capture_id=str(index),
                sha256=str(index),
                path=str(index),
                measured_value=value,
                measurement_unit="mm",
                measurement_reviewed_by="unit-test",
                physical_unit="single_pod",
                crop_key="okra",
                metric="centerline_length_mm",
                marker_side_mm=50,
            )
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(evaluation, "verify_original_photo", lambda row, root: root / row["path"])
    monkeypatch.setattr(evaluation, "load_image_file", lambda path: SimpleNamespace(bgr=None))
    outcomes = iter([Status.CALIBRATION_REQUIRED, Status.ACCEPTED])

    class PolicyOnlyEngine:
        def __init__(self, policies=None):
            self.policies = policies or load_policies()

        def inspect_frame(self, frame, request):
            policy = self.policies.policy(request.crop_key)
            assert policy.threshold_mm == 110
            assert policy.approval_status == "EMPIRICAL_REFERENCE"
            status = next(outcomes)
            return InspectionResult(
                result_id="unit-test",
                context=MediaContext(),
                status=status,
                size_class=SizeClass.BIG if status == Status.ACCEPTED else None,
            )

    monkeypatch.setattr(evaluation, "InferenceService", PolicyOnlyEngine)
    report = evaluation.evaluate(manifest, tmp_path, tmp_path / "report.json")
    assert report["status"] == "EVALUATED_NOT_RELEASED"
    assert report["reference_statistic"] == "median"
    assert report["reference_averages"][0]["mean"] == 200
    assert report["reference_averages"][0]["median"] == 110
    assert report["coverage"] == 0.5
    assert report["accuracy"] == 0.5
    assert report["accepted_accuracy"] == 1
    assert report["confusion"] == {"SMALL->REFUSED": 1, "BIG->BIG": 1}
