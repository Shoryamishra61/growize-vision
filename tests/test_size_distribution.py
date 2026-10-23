"""Numerical policy checks; no generated images or model accuracy claims."""

import pytest

from growize_vision.real_photos import reference_statistics, size_distribution


def test_median_is_boundary_and_half_medians_are_only_anchors():
    result = size_distribution([10, 20, 30, 200])
    assert result["median"] == 25
    assert result["mean"] == 65
    assert result["small_group_median"] == 15
    assert result["big_group_median"] == 115
    assert result["small_specimens"] == result["big_specimens"] == 2


def test_ties_belong_to_big_without_fabricating_a_lower_half():
    result = size_distribution([20, 20, 20])
    assert result["small_group_median"] is None
    assert result["small_specimens"] == 0
    assert result["big_group_median"] == 20
    assert result["big_specimens"] == 3


@pytest.mark.parametrize("values", [[], [True], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_dimensions_are_rejected(values):
    with pytest.raises(ValueError, match="finite positive"):
        size_distribution(values)


def test_reference_median_excludes_held_out_and_duplicate_photos():
    base = dict(
        split="train",
        crop_key="okra",
        physical_unit="single_pod",
        metric="min_area_rect_long_mm",
        measurement_unit="mm",
        measurement_reviewed_by="numerical-contract-check",
    )
    rows = [
        dict(base, specimen_id=str(i), measured_value=value)
        for i, value in enumerate([10, 20, 30, 200])
    ]
    rows += [rows[0].copy(), dict(base, split="test", specimen_id="test", measured_value=999)]
    reference = reference_statistics(rows)[0]
    assert reference["specimens"] == 4
    assert reference["median"] == 25
    assert reference["mean"] == 65
