"""Unit tests for the CropIdentityService layer."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from growize_vision.crop_identity import CropIdentityService


@pytest.fixture
def identity_service():
    return CropIdentityService()


def test_identity_service_loads(identity_service):
    assert identity_service.is_classifier_available
    assert len(identity_service.classes) == 18
    assert len(identity_service.stats) == 18


def test_predict_species_on_synthetic_patch(identity_service):
    # Pure red image (synthetic)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)  # Pure red in BGR
    pred = identity_service.predict_species(img)
    assert pred is not None
    assert pred.species_name in [c.replace("_", " ") for c in identity_service.classes]
    assert 0.0 <= pred.confidence <= 1.0


def test_estimate_appearance_size(identity_service):
    # White background with black circle
    img = np.full((224, 224, 3), 255, dtype=np.uint8)
    cv2.circle(img, (112, 112), 40, (0, 0, 0), -1)  # Area ~ pi * 40^2 ~ 5026 px

    est = identity_service.estimate_appearance_size(img, species_name="Potato")
    assert est is not None
    assert est.pixel_area > 4000
    assert est.appearance_size_class in ["SMALL", "BIG", "BORDERLINE"]
    assert "UNCALIBRATED_APPEARANCE_ESTIMATE" in est.disclaimer
