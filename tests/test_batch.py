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
    # A sorter must label every item: near-cut items are assigned by exact
    # position (with a note), never left in review.
    assert len(graded) == 3
    big = [i for i in graded if i.decision.size_class is SizeClass.BIG]
    small = [i for i in graded if i.decision.size_class is SizeClass.SMALL]
    assert len(big) == 2 and len(small) == 1
    reasons = " ".join(r for i in result.instances if i.decision for r in i.decision.reasons)
    assert "BATCH_RELATIVE_UNCALIBRATED" in reasons
    assert "NEAR_BATCH_CUT" in reasons  # the median item sits on the cut


def test_two_big_potatoes_both_grade_big_with_absolute_bands() -> None:
    """Absolute grading: items are compared against the crop's published size
    bands, so two genuinely big potatoes are both BIG — a relative split that
    forces one SMALL and one BIG on identical items is wrong."""
    import cv2

    rng = np.random.default_rng(21)
    scene = np.full((560, 800, 3), 190, np.uint8)
    scene = np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)
    # Two wide tubers (~150 px short axis) and no marker.
    cv2.ellipse(scene, (230, 280), (150, 96), 6, 0, 360, (95, 130, 70), -1)
    cv2.ellipse(scene, (580, 290), (146, 92), -5, 0, 360, (110, 115, 60), -1)
    scene = np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)
    service = InferenceService()
    result = service.inspect_frame(
        scene,
        request=InferenceRequest(crop_key="potato", allow_scale_estimate=True),
    )
    assert any(r.startswith("BATCH_SPLIT:") for r in result.reasons)
    graded = [
        i for i in result.instances
        if i.decision and i.decision.size_class is not None
    ]
    words = {i.decision.size_class.value for i in graded}
    assert words == {"BIG"}  # both big against the 40/60 mm bands, no forced small
    # Millimetre measurements back the verdicts.
    assert all(i.measurement is not None for i in graded)
    assert all(i.measurement.value_mm > 60.0 for i in graded)


def test_batch_grades_despite_busy_background_check() -> None:
    """Produce near the frame border trips the plain-mat heuristic; a batch
    sort must continue with a warning instead of refusing everything."""
    import cv2

    rng = np.random.default_rng(11)
    scene = np.full((560, 800, 3), 200, np.uint8)
    scene = np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)
    cv2.ellipse(scene, (250, 280), (105, 64), 8, 0, 360, (95, 130, 70), -1)
    cv2.ellipse(scene, (560, 290), (100, 62), -6, 0, 360, (110, 115, 60), -1)
    cv2.ellipse(scene, (390, 460), (98, 58), 3, 0, 360, (85, 135, 75), -1)
    cv2.ellipse(scene, (120, 480), (42, 28), 0, 0, 360, (90, 125, 72), -1)
    scene = np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)
    service = InferenceService()
    result = service.inspect_frame(
        scene, request=InferenceRequest(crop_key="potato", allow_scale_estimate=True))
    graded = [
        i for i in result.instances
        if i.decision and i.decision.size_class is not None
    ]
    assert graded, "batch must grade items even when the background check complains"
    assert all(i.measurement is not None for i in graded)


def test_uniform_batch_labels_every_item_the_same() -> None:
    """Three nearly identical potatoes: big/small is meaningless, all get 2."""
    import cv2

    rng = np.random.default_rng(9)
    scene = np.full((560, 800, 3), 200, np.uint8)
    scene = np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)
    for cx, rx in ((160, 100), (400, 99), (630, 101)):
        cv2.ellipse(scene, (cx, 280), (rx, 62), 0, 0, 360, (70, 150, 70), -1)
    scene = np.clip(scene.astype(int) + rng.integers(-7, 8, scene.shape), 0, 255).astype(np.uint8)
    service = InferenceService()
    result = service.inspect_frame(
        scene, request=InferenceRequest(crop_key="potato", batch_mode=True))
    bands = [i.decision.band_name for i in result.instances if i.decision]
    assert bands == ["2", "2", "2"]
    reasons = " ".join(r for i in result.instances if i.decision for r in i.decision.reasons)
    assert "UNIFORM_BATCH" in reasons


def test_multiple_items_grade_automatically_without_batch_flag() -> None:
    """The operator uploads a photo of several vegetables: no checkbox,
    no crop picking — grading happens by itself, relative to the batch."""
    service = InferenceService()
    result = service.inspect_frame(
        batch_scene(), request=InferenceRequest(crop_key="potato"))
    assert result.status is Status.MULTIPLE_OBJECTS
    assert any(r.startswith("BATCH_SPLIT:") for r in result.reasons)
    bands = sorted(
        (i.decision.band_name for i in result.instances if i.decision and i.decision.band_name),
        key=int,
    )
    assert bands == ["1", "2", "3"]


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


def test_batch_grades_edge_clipped_items_with_a_warning() -> None:
    """A sorter must label every item: an item clipped by the frame edge is
    graded from its visible part with an explicit EDGE_CLIPPED_GRADED flag
    instead of being refused."""
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
        request=InferenceRequest(crop_key="potato", batch_mode=True, allow_scale_estimate=True),
    )
    graded = [
        i for i in result.instances
        if i.decision and (i.decision.size_class or i.decision.band_name)
    ]
    clipped = [i for i in result.instances if "TOUCHES_FRAME_EDGE" in i.quality_flags]
    assert clipped, "fixture must contain an edge-clipped item"
    assert clipped[0] in graded
    assert any("EDGE_CLIPPED_GRADED" in f for f in clipped[0].quality_flags)
