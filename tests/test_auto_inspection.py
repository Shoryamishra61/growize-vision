"""Shared adapter contracts; these synthetic fixtures do not measure ML accuracy."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from growize_vision.crop_identity import AppearanceSizeEstimate, SpeciesPrediction
from growize_vision.domain import MediaContext, SourceType, Status
from growize_vision.inference import InferenceRequest, InferenceService
from growize_vision.media import load_image_file


def fake_identity(prediction):
    return SimpleNamespace(
        predict_species=Mock(return_value=prediction),
        detect_produce_presence=Mock(return_value=(True, "OBJECT_CANDIDATE")),
        estimate_appearance_size=Mock(return_value=None),
        render_appearance_overlay=Mock(side_effect=lambda image, *args: image.copy()),
        is_deep=False, classifier_path=Path(__file__),
    )


@pytest.mark.parametrize("source", list(SourceType))
def test_auto_policy_is_resolved_in_shared_engine(demo_dir, source):
    identity = fake_identity(SpeciesPrediction("potato", "Potato", 0.9, {"Potato": 0.9, "Onion": 0.1}))
    service = InferenceService(identity_service=identity)
    frame = load_image_file(demo_dir / "potato_small_synthetic.png").bgr
    result = service.inspect_frame(frame, MediaContext(source_type=source), InferenceRequest(crop_key="auto", marker_side_mm=50, demo_policy=True))
    assert result.status == Status.ACCEPTED
    assert result.policy.crop_key == "potato"
    assert result.to_dict()["species_prediction"]["confidence"] == 0.9
    assert result.model_sha


def test_unresolved_auto_crop_never_inherits_a_default_policy(demo_dir):
    service = InferenceService(identity_service=fake_identity(None))
    frame = load_image_file(demo_dir / "potato_small_synthetic.png").bgr
    result = service.inspect_frame(frame, request=InferenceRequest(crop_key="auto", marker_side_mm=50, demo_policy=True))
    assert result.status == Status.POLICY_REQUIRED
    assert result.size_class is None
    assert result.policy is None


def test_appearance_is_opt_in_and_never_changes_physical_status(demo_dir):
    identity = fake_identity(SpeciesPrediction("potato", "Potato", 0.9, {"Potato": 1.0}))
    identity.estimate_appearance_size.return_value = AppearanceSizeEstimate("potato", 1000, 80, 900, 70, 1.1, "BIG")
    service = InferenceService(identity_service=identity)
    frame = load_image_file(demo_dir / "no_marker_synthetic.png").bgr
    strict = service.inspect_frame(frame, request=InferenceRequest(crop_key="potato"))
    assert strict.appearance_estimate is None
    identity.estimate_appearance_size.assert_not_called()
    appearance = service.inspect_frame(frame, request=InferenceRequest(crop_key="potato", allow_appearance_estimate=True))
    assert appearance.status == Status.CALIBRATION_REQUIRED
    assert appearance.size_class is None
    assert appearance.appearance_estimate["appearance_size_class"] == "BIG"
    assert appearance.to_dict()["accepted"] is False
    assert all(instance.measurement is None for instance in appearance.instances)


def test_non_produce_refusal_is_shared_and_exportable(demo_dir):
    identity = fake_identity(None)
    identity.detect_produce_presence.return_value = (False, "NON_PRODUCE_DOCUMENT_OR_TEXT")
    service = InferenceService(identity_service=identity)
    frame = load_image_file(demo_dir / "potato_small_synthetic.png").bgr
    result = service.inspect_frame(frame)
    assert result.status == Status.NOT_PRODUCE
    assert result.to_dict()["size_class"] is None
    assert result.annotated_image is not None
    assert result.to_dict()["reasons"] == ["NON_PRODUCE_DOCUMENT_OR_TEXT"]


@pytest.mark.parametrize(("reason", "status"), [
    ("FRAME_TOO_DARK", Status.LOW_IMAGE_QUALITY),
    ("EMPTY_UNIFORM_SCENE", Status.NO_OBJECT),
    ("FRAME_TOO_SMALL", Status.INVALID_INPUT),
    ("NO_PRODUCE_DETECTED", Status.OUT_OF_DISTRIBUTION),
])
def test_input_screen_preserves_failure_meaning(demo_dir, reason, status):
    identity = fake_identity(None)
    identity.detect_produce_presence.return_value = (False, reason)
    result = InferenceService(identity_service=identity).inspect_frame(
        load_image_file(demo_dir / "potato_small_synthetic.png").bgr)
    assert result.status == status
    assert result.size_class is None


def test_scene_suggestion_never_labels_separate_instances(demo_dir, monkeypatch):
    from growize_vision.segmentation_cv import segment_foreground

    def two_instances(*args, **kwargs):
        result = segment_foreground(*args, **kwargs)
        result.objects = result.objects * 2
        return result

    monkeypatch.setattr("growize_vision.inference.segment_foreground", two_instances)
    identity = fake_identity(SpeciesPrediction("potato", "Potato", .99, {"Potato": .99, "Onion": .01}))
    frame = load_image_file(demo_dir / "potato_small_synthetic.png").bgr
    result = InferenceService(identity_service=identity).inspect_frame(
        frame, request=InferenceRequest(crop_key="auto", marker_side_mm=50, demo_policy=True))
    assert len(result.instances) == 2
    assert result.to_dict()["species_prediction_scope"] == "scene"
    assert all(instance.crop_key is None for instance in result.instances)
    assert all(instance.decision.size_class is None for instance in result.instances)
