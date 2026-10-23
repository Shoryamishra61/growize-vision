"""Probability and failure contracts independent of optional model downloads."""

from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from growize_vision.crop_identity import CropIdentityService


@pytest.fixture
def service():
    service = CropIdentityService.__new__(CropIdentityService)
    service.classes = ["Bean", "Okra", "Bitter_Gourd"]
    service.is_deep = False
    service.model = Mock()
    service.extract_features = Mock(return_value=np.zeros(53))
    return service


def produce_with_text(text_color=(255, 0, 0)):
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (200, 230), (45, 120), 10, 0, 360, (40, 130, 60), -1)
    cv2.putText(image, "Fresh produce", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)
    return image


def test_no_geometry_can_inflate_or_relabel_model_confidence(service):
    service.model.predict_proba.return_value = [[0.80, 0.15, 0.05]]
    prediction = service.predict_species(produce_with_text())
    assert prediction.species_name == "Bean"
    assert prediction.confidence == 0.8
    assert sum(prediction.all_probabilities.values()) == pytest.approx(1)


def test_canonical_policy_key_retains_underscores(service):
    prediction = service._prediction_from_probabilities([0.01, 0.01, 0.98])
    assert prediction.species_key == "bitter_gourd"


@pytest.mark.parametrize("probabilities", [
    [np.nan, 0, 1], [np.inf, 0, 0], [-0.2, 0.2, 1], [2, 0, 0],
    [0.9, 0.9, 0], [0.7, 0.3], [0.4, 0.3, 0.3],
])
def test_invalid_or_ambiguous_probabilities_abstain(service, probabilities):
    assert service._prediction_from_probabilities(probabilities) is None


def test_failed_provider_abstains_instead_of_crashing(service):
    service.model.predict_proba.side_effect = RuntimeError("provider failed")
    assert service.predict_species(produce_with_text()) is None


@pytest.mark.parametrize("color", [(0, 0, 0), (255, 0, 0), (0, 0, 255)])
def test_text_outside_produce_does_not_veto_object(color):
    present, reason = CropIdentityService.detect_produce_presence(produce_with_text(color))
    assert present, reason


@pytest.mark.parametrize("color", [(0, 0, 0), (255, 255, 255), (34, 139, 34), (255, 0, 0)])
def test_uniform_surfaces_are_not_produce(color):
    image = np.full((224, 224, 3), color, dtype=np.uint8)
    assert not CropIdentityService.detect_produce_presence(image)[0]


def test_red_channel_arithmetic_does_not_wrap():
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (200, 200), (60, 100), 0, 0, 360, (180, 140, 250), -1)
    assert CropIdentityService.detect_produce_presence(image)[0]


def test_large_upload_presence_screen_has_bounded_working_resolution(monkeypatch):
    image = np.full((1600, 3200, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (1600, 800), (300, 600), 0, 0, 360, (40, 130, 60), -1)
    original = cv2.cvtColor
    dimensions = []

    def record_dimensions(frame, *args, **kwargs):
        dimensions.append(max(frame.shape[:2]))
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(cv2, "cvtColor", record_dimensions)
    assert CropIdentityService.detect_produce_presence(image)[0]
    assert dimensions and max(dimensions) <= 960


@pytest.mark.parametrize("angle", [0, 30, 45, 60, 90, 135])
def test_slender_produce_is_not_rejected_as_text_when_rotated(angle):
    image = np.full((320, 320, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (160, 160), (100, 10), angle, 0, 360, (40, 130, 60), -1)
    present, reason = CropIdentityService.detect_produce_presence(image)
    assert present, reason
