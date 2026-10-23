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
        # Green realistic synthetic patch
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :] = (34, 139, 34)  # Forest green in BGR
        present, _ = identity_service.detect_produce_presence(img)
        assert present is True


def test_estimate_appearance_size(identity_service):
    from pathlib import Path
    sample = Path("data/raw/cropped_okra_pods/crop_100_0.jpg")
    if sample.exists():
        img = cv2.imread(str(sample))
        est = identity_service.estimate_appearance_size(img, species_name="Okra")
        assert est is not None
        assert est.pixel_area > 10000
        assert est.appearance_size_class in ["SMALL", "BIG"]
        assert "UNCALIBRATED_APPEARANCE_ESTIMATE" in est.disclaimer
    else:
        # Green organic ellipse
        img = np.full((224, 224, 3), 255, dtype=np.uint8)
        cv2.ellipse(img, (112, 112), (60, 25), 0, 0, 360, (34, 139, 34), -1)
        est = identity_service.estimate_appearance_size(img, species_name="Okra")
        assert est is not None
        assert est.appearance_size_class in ["SMALL", "BIG"]


def test_non_produce_signature_rejected(identity_service):
    from pathlib import Path
    sample = Path("test_signature_crop.jpg")
    if sample.exists():
        img = cv2.imread(str(sample))
        present, reason = identity_service.detect_produce_presence(img)
        assert present is False
        assert "NON_PRODUCE" in reason
        pred = identity_service.predict_species(img)
        assert pred is None
