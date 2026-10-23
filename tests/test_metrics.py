"""Validation statistics: error summaries, confidence intervals, sample sizes."""

from __future__ import annotations

import math

import pytest

from growize_vision.metrics import (
    error_summary,
    repeatability_summary,
    sample_size_reference,
    wilson_interval,
    zero_failure_sample_size,
    zero_failure_upper_bound,
)


def test_error_summary_matches_hand_computed_values() -> None:
    summary = error_summary([1.0, -2.0, 3.0])
    assert summary["n"] == 3
    assert summary["mae"] == pytest.approx(2.0)
    assert summary["bias"] == pytest.approx((1.0 - 2.0 + 3.0) / 3.0)
    assert summary["rmse"] == pytest.approx(math.sqrt((1.0 + 4.0 + 9.0) / 3.0))
    assert summary["median_abs_error"] == pytest.approx(2.0)
    assert summary["p95_abs_error"] == pytest.approx(3.0)
    assert summary["max_abs_error"] == pytest.approx(3.0)


def test_error_summary_p95_is_nearest_rank() -> None:
    # 20 values; nearest-rank p95 is the 19th ordered absolute value.
    values = [float(i) for i in range(1, 21)]
    summary = error_summary(values)
    assert summary["p95_abs_error"] == pytest.approx(19.0)


def test_error_summary_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        error_summary([])
    with pytest.raises(ValueError, match="finite"):
        error_summary([1.0, float("nan")])
    with pytest.raises(ValueError, match="numeric"):
        error_summary([1.0, True])  # type: ignore[list-item]


def test_wilson_interval_brackets_point_estimate() -> None:
    low, high = wilson_interval(95, 100)
    assert 0.88 < low < 0.95 < high < 0.99


def test_wilson_interval_zero_failures_upper_bound_near_3_n() -> None:
    # With 0/100 the Wilson upper bound should sit close to the rule-of-three 3%.
    low, high = wilson_interval(0, 100)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert 0.030 < high < 0.050


def test_wilson_interval_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_zero_failure_upper_bound_matches_formula() -> None:
    # (1 - p)^n = alpha  =>  p = 1 - alpha^(1/n)
    assert zero_failure_upper_bound(59, 0.05) == pytest.approx(0.05, abs=0.002)
    assert zero_failure_upper_bound(299, 0.05) == pytest.approx(0.01, abs=0.001)
    assert zero_failure_upper_bound(100, 0.05) == pytest.approx(0.0295, abs=0.001)


def test_zero_failure_upper_bound_never_zero() -> None:
    assert zero_failure_upper_bound(10_000) > 0.0


def test_zero_failure_sample_size_matches_table() -> None:
    # Blueprint table: 5% -> 59, 1% -> 299, 0.1% -> 2995 at 95% confidence.
    assert zero_failure_sample_size(0.05) == 59
    assert zero_failure_sample_size(0.01) == 299
    assert zero_failure_sample_size(0.001) == 2995
    assert zero_failure_sample_size(0.05, alpha=0.01) == 90


def test_sample_size_and_bound_round_trip() -> None:
    for rate in (0.05, 0.02, 0.01, 0.005, 0.001):
        for alpha in (0.05, 0.01):
            n = zero_failure_sample_size(rate, alpha)
            # After n zero-failure trials the bound must not exceed the rate.
            assert zero_failure_upper_bound(n, alpha) <= rate + 1e-9
            # And n-1 trials would not be enough.
            assert zero_failure_upper_bound(n - 1, alpha) > rate


def test_sample_size_reference_table_has_all_rows() -> None:
    table = sample_size_reference()
    assert len(table) == 10
    by_key = {(row["max_error_rate"], row["confidence"]): row for row in table}
    assert by_key[(0.05, 0.95)]["required_zero_failure_samples"] == 59
    assert by_key[(0.01, 0.95)]["required_zero_failure_samples"] == 299
    assert by_key[(0.001, 0.99)]["required_zero_failure_samples"] == 4603


def test_repeatability_summary_splits_within_and_between() -> None:
    groups = {
        "specimen_a": [100.0, 101.0, 99.0],
        "specimen_b": [200.0, 201.0, 199.0],
    }
    summary = repeatability_summary(groups)
    assert summary["specimens"] == 2
    assert summary["captures_total"] == 6
    assert summary["per_specimen"]["specimen_a"]["spread"] == pytest.approx(2.0)
    assert summary["pooled_within_specimen_std"] == pytest.approx(1.0)
    # Specimen means (100, 200) are far apart relative to within-noise.
    assert summary["between_specimen_std"] == pytest.approx(70.71, rel=0.01)
    assert "not population accuracy" in summary["note"]


def test_repeatability_summary_single_capture_has_no_std() -> None:
    summary = repeatability_summary({"specimen_x": [55.5]})
    assert summary["per_specimen"]["specimen_x"]["sample_std"] is None
    assert summary["pooled_within_specimen_std"] is None
    assert summary["between_specimen_std"] is None


def test_repeatability_summary_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        repeatability_summary({})
    with pytest.raises(ValueError, match="finite"):
        repeatability_summary({"a": [1.0, float("inf")]})
