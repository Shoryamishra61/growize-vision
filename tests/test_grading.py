"""Policy evaluation contract: only SMALL/BIG may be accepted, boundaries are safe."""

from __future__ import annotations

import math

import pytest

from growize_vision.domain import (
    GradePolicy,
    PhysicalMeasurement,
    SizeBand,
    SizeClass,
    Status,
)
from growize_vision.grading import evaluate_measurement


def policy(
    *,
    threshold: float | None = 60.0,
    comparator: str = "greater_than_or_equal",
    guard: float = 0.5,
    demo: bool = False,
) -> GradePolicy:
    return GradePolicy(
        crop_key="potato",
        crop_name="Potato",
        physical_unit="single_tuber",
        metric="min_area_rect_short_mm",
        metric_description="test",
        threshold_mm=threshold,
        comparator=comparator,
        guard_band_mm=guard,
        policy_version="test/demo" if demo else "test",
        approval_status="ASSESSMENT_DEMO_ONLY" if demo else "APPROVED_SOURCE",
        demo=demo,
    )


def measurement(
    value: float,
    uncertainty: float = 0.0,
    *,
    metric: str = "min_area_rect_short_mm",
    lower: float | None = None,
    upper: float | None = None,
) -> PhysicalMeasurement:
    low = value - uncertainty if lower is None else lower
    high = value + uncertainty if upper is None else upper
    return PhysicalMeasurement(
        metric=metric,
        value_mm=value,
        uncertainty_mm=uncertainty,
        lower_mm=low,
        upper_mm=high,
    )


def test_interval_clear_above_threshold_is_big() -> None:
    decision = evaluate_measurement(measurement(70.0, 1.0), policy(), "calib_x")
    assert decision.accepted
    assert decision.status is Status.ACCEPTED
    assert decision.size_class is SizeClass.BIG
    assert decision.calibration_id == "calib_x"
    assert "INTERVAL_ENTIRELY_AT_OR_ABOVE_THRESHOLD" in decision.reasons


def test_interval_clear_below_threshold_is_small() -> None:
    decision = evaluate_measurement(measurement(50.0, 1.0), policy(), "calib_x")
    assert decision.accepted
    assert decision.status is Status.ACCEPTED
    assert decision.size_class is SizeClass.SMALL
    assert "INTERVAL_ENTIRELY_BELOW_THRESHOLD" in decision.reasons


def test_guard_band_boundaries_are_inclusive_for_big() -> None:
    decision = evaluate_measurement(measurement(60.5, 0.0), policy(), "calib")
    assert decision.status is Status.ACCEPTED
    assert decision.size_class is SizeClass.BIG


def test_guard_band_boundary_is_exclusive_for_small() -> None:
    # upper == threshold - guard is not strictly below the guarded cutoff.
    decision = evaluate_measurement(measurement(59.5, 0.0), policy(), "calib")
    assert decision.status is Status.REVIEW_BORDERLINE
    assert decision.size_class is None


def test_just_below_guarded_cutoff_is_small() -> None:
    decision = evaluate_measurement(measurement(59.49, 0.0), policy(), "calib")
    assert decision.status is Status.ACCEPTED
    assert decision.size_class is SizeClass.SMALL


def test_uncertainty_interval_crossing_cutoff_is_review() -> None:
    decision = evaluate_measurement(measurement(60.0, 2.0), policy(), "calib")
    assert decision.status is Status.REVIEW_BORDERLINE
    assert decision.size_class is None
    assert "UNCERTAINTY_OR_GUARD_BAND_CROSSES_THRESHOLD" in decision.reasons
    assert decision.warning


@pytest.mark.parametrize(
    ("value", "uncertainty"),
    [(58.0, 0.0), (59.0, 0.0), (60.0, 0.0), (61.0, 0.0), (100.0, 1.0)],
)
def test_only_accepted_decisions_carry_a_size(value: float, uncertainty: float) -> None:
    decision = evaluate_measurement(measurement(value, uncertainty), policy(), "calib")
    if decision.status is Status.ACCEPTED:
        assert decision.size_class in (SizeClass.SMALL, SizeClass.BIG)
    else:
        assert decision.size_class is None
        assert not decision.accepted


def test_missing_policy_refuses() -> None:
    decision = evaluate_measurement(measurement(70.0), None, None)
    assert decision.status is Status.POLICY_REQUIRED
    assert decision.size_class is None
    assert "NO_CROP_POLICY" in decision.reasons[0]


def test_null_threshold_refuses_even_with_a_measurement() -> None:
    decision = evaluate_measurement(measurement(70.0), policy(threshold=None), "calib")
    assert decision.status is Status.POLICY_REQUIRED
    assert decision.size_class is None
    assert "NO_THRESHOLD_POLICY" in decision.reasons[0]


def test_metric_mismatch_refuses() -> None:
    decision = evaluate_measurement(
        measurement(70.0, metric="min_area_rect_long_mm"),
        policy(),
        "calib",
    )
    assert decision.status is Status.POLICY_REQUIRED
    assert "METRIC_MISMATCH" in decision.reasons[0]


def test_missing_measurement_is_review_not_a_guess() -> None:
    decision = evaluate_measurement(None, policy(), "calib")
    assert decision.status is Status.REVIEW_BORDERLINE
    assert decision.size_class is None
    assert "NO_PHYSICAL_MEASUREMENT" in decision.reasons[0]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_review(bad: float) -> None:
    decision = evaluate_measurement(measurement(bad), policy(), "calib")
    assert decision.status is Status.REVIEW_BORDERLINE
    assert decision.size_class is None


def test_malformed_interval_is_review() -> None:
    decision = evaluate_measurement(
        measurement(70.0, 1.0, lower=80.0, upper=60.0), policy(), "calib"
    )
    assert decision.status is Status.REVIEW_BORDERLINE
    assert "INVALID_UNCERTAINTY" in decision.reasons[0]


def test_demo_policy_is_labelled_on_acceptance() -> None:
    decision = evaluate_measurement(measurement(70.0), policy(demo=True), "calib")
    assert decision.status is Status.ACCEPTED
    assert "ASSESSMENT_DEMO_POLICY_NOT_ARL_APPROVED" in decision.reasons
    assert "DEMO CUT-OFF ONLY" in (decision.warning or "")


def test_demo_policy_is_labelled_on_review() -> None:
    decision = evaluate_measurement(measurement(60.0, 2.0), policy(demo=True), "calib")
    assert decision.status is Status.REVIEW_BORDERLINE
    assert "DEMO POLICY" in (decision.warning or "")


def test_greater_than_comparator() -> None:
    above = evaluate_measurement(
        measurement(61.0), policy(comparator="greater_than", guard=0.0), "calib"
    )
    equal = evaluate_measurement(
        measurement(60.0), policy(comparator="greater_than", guard=0.0), "calib"
    )
    assert above.size_class is SizeClass.BIG
    assert equal.size_class is SizeClass.SMALL


def test_less_than_comparator() -> None:
    below = evaluate_measurement(
        measurement(59.0), policy(comparator="less_than", guard=0.0), "calib"
    )
    above = evaluate_measurement(
        measurement(61.0), policy(comparator="less_than", guard=0.0), "calib"
    )
    assert below.size_class is SizeClass.BIG
    assert above.size_class is SizeClass.SMALL


def test_invalid_comparator_refuses() -> None:
    decision = evaluate_measurement(
        measurement(60.0), policy(comparator="approx_equal"), "calib"
    )
    assert decision.status is Status.POLICY_REQUIRED
    assert "UNSUPPORTED_COMPARATOR" in decision.reasons[0]


def band_policy(bands, *, guard: float = 0.5, demo: bool = False) -> GradePolicy:
    return GradePolicy(
        crop_key="potato",
        crop_name="Potato",
        physical_unit="single_tuber",
        metric="min_area_rect_short_mm",
        metric_description="test",
        threshold_mm=None,
        guard_band_mm=guard,
        policy_version="test/demo" if demo else "test",
        approval_status="ASSESSMENT_DEMO_ONLY" if demo else "APPROVED_SOURCE",
        demo=demo,
        bands=tuple(bands),
    )


THREE_BANDS = (
    SizeBand(name="small", min_mm=0.0),
    SizeBand(name="medium", min_mm=45.0),
    SizeBand(name="large", min_mm=65.0),
)


def test_band_middle_grade_accepts_without_binary_class() -> None:
    decision = evaluate_measurement(measurement(55.0, 1.0), band_policy(THREE_BANDS), "cal_x")
    assert decision.accepted
    assert decision.status is Status.ACCEPTED
    assert decision.band_name == "medium"
    assert decision.size_class is None
    assert "INTERVAL_ENTIRELY_WITHIN_BAND:medium" in decision.reasons


def test_band_edges_map_to_small_and_big() -> None:
    small = evaluate_measurement(measurement(35.0, 1.0), band_policy(THREE_BANDS), "cal_x")
    large = evaluate_measurement(measurement(80.0, 1.0), band_policy(THREE_BANDS), "cal_x")
    assert small.size_class is SizeClass.SMALL and small.band_name == "small"
    assert large.size_class is SizeClass.BIG and large.band_name == "large"


def test_band_interval_crossing_edge_is_reviewed() -> None:
    # 66 +/- 2 crosses the 65 mm large edge after the 0.5 mm guard band.
    decision = evaluate_measurement(measurement(66.0, 2.0), band_policy(THREE_BANDS), "cal_x")
    assert decision.status is Status.REVIEW_BORDERLINE
    assert decision.band_name is None
    assert "UNCERTAINTY_OR_GUARD_BAND_CROSSES_BAND_EDGE" in decision.reasons
    assert any("CROSSES_BAND_EDGE" in reason for reason in decision.reasons)


def test_value_inside_band_guard_is_reviewed() -> None:
    decision = evaluate_measurement(measurement(65.2, 0.1), band_policy(THREE_BANDS), "cal_x")
    assert decision.status is Status.REVIEW_BORDERLINE
    assert "VALUE_INSIDE_BAND_GUARD" in " ".join(decision.reasons)


def test_two_bands_reproduce_binary_rule() -> None:
    two = (SizeBand(name="small", min_mm=0.0), SizeBand(name="big", min_mm=60.0))
    big = evaluate_measurement(measurement(70.0, 1.0), band_policy(two), "cal_x")
    small = evaluate_measurement(measurement(50.0, 1.0), band_policy(two), "cal_x")
    assert big.size_class is SizeClass.BIG and big.band_name == "big"
    assert small.size_class is SizeClass.SMALL and small.band_name == "small"
    borderline = evaluate_measurement(measurement(60.0, 1.0), band_policy(two), "cal_x")
    assert borderline.status is Status.REVIEW_BORDERLINE


def test_bands_satisfy_policy_without_binary_threshold() -> None:
    decision = evaluate_measurement(measurement(55.0, 1.0), band_policy(THREE_BANDS), "cal_x")
    assert decision.status is Status.ACCEPTED
    decision = evaluate_measurement(
        measurement(55.0, 1.0), policy(threshold=None), "cal_x"
    )
    assert decision.status is Status.POLICY_REQUIRED


def test_demo_bands_carry_disclaimer() -> None:
    decision = evaluate_measurement(
        measurement(55.0, 1.0), band_policy(THREE_BANDS, demo=True), "cal_x"
    )
    assert "ASSESSMENT_DEMO_POLICY_NOT_ARL_APPROVED" in decision.reasons
    assert decision.warning and "DEMO BANDS" in decision.warning
