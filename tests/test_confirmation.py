"""Two-capture confirmation: agreement grading, disagreement refusal."""

from __future__ import annotations

import pytest

from growize_vision.confirmation import confirm_captures
from growize_vision.domain import (
    GradeDecision,
    InspectionResult,
    InstanceObservation,
    MediaContext,
    PhysicalMeasurement,
    SizeClass,
    Status,
)


def capture_result(
    *,
    status: Status = Status.ACCEPTED,
    size_class: SizeClass | None = SizeClass.BIG,
    band: str | None = None,
    value_mm: float = 70.0,
    uncertainty_mm: float = 2.0,
) -> InspectionResult:
    decision = GradeDecision(
        size_class=size_class,
        status=status,
        reasons=["UNIT_TEST"],
        band_name=band,
    )
    instance = InstanceObservation(
        instance_id="obj_001",
        crop_key="potato",
        bbox_px=(0, 0, 10, 10),
        contour_px=[[0, 0]],
        mask_area_px=100.0,
        decision=decision,
        measurement=PhysicalMeasurement(
            metric="min_area_rect_short_mm",
            value_mm=value_mm,
            uncertainty_mm=uncertainty_mm,
            lower_mm=value_mm - uncertainty_mm,
            upper_mm=value_mm + uncertainty_mm,
        ),
    )
    return InspectionResult(
        result_id="unit",
        context=MediaContext(),
        status=status,
        instances=[instance],
    )


def test_agreeing_captures_confirm_the_grade() -> None:
    outcome = confirm_captures(
        [capture_result(band="large"), capture_result(band="large", value_mm=71.0)]
    )
    assert outcome.status is Status.ACCEPTED
    assert outcome.size_class is SizeClass.BIG
    assert outcome.size_band == "large"
    assert any(r.startswith("DUAL_CAPTURE_CONFIRMED") for r in outcome.reasons)


def test_disagreeing_classes_are_refused_not_averaged() -> None:
    outcome = confirm_captures(
        [
            capture_result(size_class=SizeClass.SMALL, band="small", value_mm=45.0),
            capture_result(size_class=SizeClass.BIG, band="large", value_mm=72.0),
        ]
    )
    assert outcome.status is Status.REVIEW_BORDERLINE
    assert outcome.size_class is None
    assert any(r.startswith("DUAL_CAPTURE_CLASS_DISAGREEMENT") for r in outcome.reasons)


def test_band_disagreement_is_refused_even_with_same_binary_class() -> None:
    outcome = confirm_captures(
        [capture_result(band="medium"), capture_result(band="large")]
    )
    assert outcome.status is Status.REVIEW_BORDERLINE
    assert outcome.size_band is None


def test_failed_capture_blocks_confirmation() -> None:
    outcome = confirm_captures(
        [
            capture_result(band="large"),
            capture_result(status=Status.REVIEW_BORDERLINE, size_class=None, band=None),
        ]
    )
    assert outcome.status is Status.REVIEW_BORDERLINE
    assert any(r.startswith("CONFIRMATION_BLOCKED") for r in outcome.reasons)


def test_large_value_spread_is_flagged_but_grade_holds() -> None:
    outcome = confirm_captures(
        [capture_result(band="large", value_mm=66.0, uncertainty_mm=1.0),
         capture_result(band="large", value_mm=90.0, uncertainty_mm=1.0)]
    )
    assert outcome.status is Status.ACCEPTED
    assert any(r.startswith("DUAL_CAPTURE_VALUE_SPREAD_HIGH") for r in outcome.reasons)


def test_confirmation_requires_exactly_two_captures() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        confirm_captures([capture_result()])
    with pytest.raises(ValueError, match="exactly two"):
        confirm_captures([capture_result(), capture_result(), capture_result()])


def test_confirmation_requires_one_graded_instance_per_capture() -> None:
    empty = InspectionResult(
        result_id="unit", context=MediaContext(), status=Status.NO_OBJECT
    )
    with pytest.raises(ValueError, match="exactly one graded instance"):
        confirm_captures([capture_result(), empty])
