"""Batch-relative grading: numeric 1/2/3 grades with no scale of any kind."""

from __future__ import annotations

import numpy as np
import pytest

from growize_vision.domain import SizeClass, Status
from growize_vision.inference import InferenceRequest, InferenceService


def batch_scene() -> np.ndarray:
    """Three clearly separated potato blobs of different sizes, no marker."""
    import cv2

    rng = np.random.default_rng(5)
    scene = np.full((560, 800, 3), 200, np.uint8)
    scene = np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)
    # Areas approx 32k / 13k / 8k px: every blob passes the 1.5% area gate.
    cv2.ellipse(scene, (170, 280), (130, 78), 10, 0, 360, (70, 150, 70), -1)
    cv2.ellipse(scene, (430, 300), (82, 50), -8, 0, 360, (80, 140, 80), -1)
    cv2.ellipse(scene, (640, 260), (64, 40), 4, 0, 360, (60, 160, 60), -1)
    return np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)


def test_batch_tercile_grades_need_no_scale() -> None:
    service = InferenceService()
    result = service.inspect_frame(
        batch_scene(),
        request=InferenceRequest(crop_key="potato", batch_mode=True),
    )
    assert result.status is Status.MULTIPLE_OBJECTS
    assert any(r.startswith("BATCH_SPLIT:") for r in result.reasons)
    bands = sorted(
        (i.decision.band_name for i in result.instances if i.decision and i.decision.band_name),
        key=int,
    )
    assert bands == ["1", "2", "3"]  # every item gets a numeric grade
    big = [
        i for i in result.instances
        if i.decision and i.decision.size_class is SizeClass.BIG
    ]
    small = [
        i for i in result.instances
        if i.decision and i.decision.size_class is SizeClass.SMALL
    ]
    assert len(big) == 1 and len(small) == 1  # grade 3 = BIG, grade 1 = SMALL
    for instance in result.instances:
        assert instance.measurement is None  # no millimetres claimed


def test_batch_binary_split_still_available() -> None:
    service = InferenceService()
    result = service.inspect_frame(
        batch_scene(),
        request=InferenceRequest(crop_key="potato", batch_mode=True, batch_grades=2),
    )
    graded = [
        i for i in result.instances
        if i.decision and i.decision.size_class is not None
    ]
    review = [
        i for i in result.instances
        if i.decision and i.decision.status is Status.REVIEW_BORDERLINE
    ]
    assert len(graded) == 2  # clear BIG and SMALL
    assert len(review) == 1  # the median item sits inside the ±5% margin
    reasons = " ".join(r for i in result.instances if i.decision for r in i.decision.reasons)
    assert "BATCH_RELATIVE_UNCALIBRATED" in reasons


def test_batch_grades_without_any_crop_policy() -> None:
    """Auto-detect failure must not dead-end a batch: sizes still grade 1/2/3."""
    service = InferenceService()
    result = service.inspect_frame(
        batch_scene(),
        request=InferenceRequest(crop_key=None, batch_mode=True),
    )
    assert result.status is Status.MULTIPLE_OBJECTS
    assert any(r.startswith("BATCH_SPLIT:") for r in result.reasons)
    bands = sorted(
        (i.decision.band_name for i in result.instances if i.decision and i.decision.band_name),
        key=int,
    )
    assert bands == ["1", "2", "3"]
    reasons = " ".join(r for i in result.instances if i.decision for r in i.decision.reasons)
    assert "CROP_UNRESOLVED" in reasons


def test_coriander_bunches_grade_in_batch_without_a_threshold() -> None:
    """Leafy crops with no published size rule still grade as batches."""
    service = InferenceService()
    result = service.inspect_frame(
        batch_scene(),
        request=InferenceRequest(crop_key="coriander", batch_mode=True),
    )
    assert result.status is Status.MULTIPLE_OBJECTS
    assert any(r.startswith("BATCH_SPLIT:") for r in result.reasons)
    bands = sorted(
        (i.decision.band_name for i in result.instances if i.decision and i.decision.band_name),
        key=int,
    )
    assert bands == ["1", "2", "3"]


def test_batch_single_item_explains_it_cannot_split() -> None:
    import cv2

    rng = np.random.default_rng(3)
    scene = np.full((560, 800, 3), 200, np.uint8)
    scene = np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)
    cv2.ellipse(scene, (400, 280), (140, 85), 0, 0, 360, (70, 150, 70), -1)
    scene = np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)
    service = InferenceService()
    result = service.inspect_frame(
        scene,
        request=InferenceRequest(crop_key="potato", batch_mode=True),
    )
    assert result.status is Status.CALIBRATION_REQUIRED
    assert any("BATCH_SPLIT_NEEDS_MORE_ITEMS" in r for r in result.reasons)


def test_batch_mode_still_refuses_edge_clipped_items() -> None:
    import cv2

    rng = np.random.default_rng(7)
    scene = np.full((560, 800, 3), 200, np.uint8)
    scene = np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)
    cv2.ellipse(scene, (200, 280), (120, 70), 0, 0, 360, (70, 150, 70), -1)
    cv2.ellipse(scene, (795, 280), (90, 60), 0, 0, 360, (80, 140, 80), -1)  # clipped by edge
    scene = np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)
    service = InferenceService()
    result = service.inspect_frame(
        scene,
        request=InferenceRequest(crop_key="potato", batch_mode=True),
    )
    occluded = [
        i for i in result.instances
        if i.decision and i.decision.status is Status.OCCLUDED
    ]
    graded = [i for i in result.instances if i.decision and i.decision.size_class]
    assert occluded and not graded  # the clipped item never receives a class
