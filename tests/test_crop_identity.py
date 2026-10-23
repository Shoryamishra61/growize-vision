"""Unit tests for the CropIdentityService layer."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from growize_vision.crop_identity import CropIdentityService


@pytest.fixture(scope="module")
def identity_service():
    return CropIdentityService()


def test_identity_service_loads(identity_service):
    assert identity_service.is_classifier_available
    assert len([label for label in identity_service.classes if label != "Unknown"]) == 18
    assert identity_service.stats


def test_predict_species_on_real_produce(identity_service):
    from pathlib import Path
    sample = Path("data/raw/cropped_okra_pods/crop_100_0.jpg")
    if sample.exists():
        img = cv2.imread(str(sample))
        pred = identity_service.predict_species(img)
        assert pred is not None
        assert pred.species_name == "Okra"
        assert pred.confidence >= 0.65
    else:
        pytest.skip("Local real-produce sample unavailable; synthetic images are not crop recognition evidence")


def test_estimate_appearance_size(identity_service):
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (112, 112), (60, 25), 0, 0, 360, (34, 139, 34), -1)
    original_stats = identity_service.stats
    try:
        identity_service.stats = {"Okra": {"reference_size_px": 224, "method": "controlled_background_v2",
                                           "area_median": 3000.0, "axis_median": 90.0}}
        estimate = identity_service.estimate_appearance_size(image, "Okra")
        assert estimate is not None
        assert estimate.appearance_size_class in ["SMALL", "BIG"]
        assert "UNCALIBRATED_APPEARANCE_ESTIMATE" in estimate.disclaimer
        resized = identity_service.estimate_appearance_size(cv2.resize(image, (896, 896)), "Okra")
        assert resized is not None
        assert resized.area_ratio_to_median == pytest.approx(estimate.area_ratio_to_median, rel=0.02)
        assert resized.appearance_size_class == estimate.appearance_size_class
    finally:
        identity_service.stats = original_stats


def test_non_produce_signature_rejected(identity_service):
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Signature", (15, 200), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 0), 2)
    present, reason = identity_service.detect_produce_presence(image)
    assert not present
    assert "NON_PRODUCE" in reason
    assert identity_service.predict_species(image) is None
