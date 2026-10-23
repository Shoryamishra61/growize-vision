"""Validation statistics for grading evidence.

Implements the metrology and selective-prediction statistics the validation
plan requires: signed-error summaries (MAE/RMSE/bias/median/p95), Wilson
confidence intervals for proportions, the exact one-sided binomial bound
after zero observed failures, the matching sample-size formula, and
within-specimen repeatability across repeated captures.

These functions are pure arithmetic with no model or image dependency so
both scripts and tests can replay them from a report.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

_Z_95 = 1.959963984540054


def _require_finite(values: Sequence[float], name: str) -> list[float]:
    if not values:
        raise ValueError(f"{name} requires at least one value.")
    out = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} requires numeric values.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} requires finite values.")
        out.append(number)
    return out


def error_summary(signed_errors: Sequence[float]) -> dict[str, Any]:
    """Summarise signed errors ``predicted - ground_truth`` for one unit."""
    errors = _require_finite(signed_errors, "error_summary")
    absolutes = sorted(abs(error) for error in errors)
    count = len(errors)
    rank = max(0, math.ceil(0.95 * count) - 1)  # nearest-rank percentile
    return {
        "n": count,
        "mae": sum(absolutes) / count,
        "rmse": math.sqrt(sum(error * error for error in errors) / count),
        "bias": sum(errors) / count,
        "median_abs_error": (
            (absolutes[count // 2] if count % 2 else 0.5 * (absolutes[count // 2 - 1] + absolutes[count // 2]))
        ),
        "p95_abs_error": absolutes[rank],
        "max_abs_error": absolutes[-1],
    }


def wilson_interval(
    successes: int, n: int, z: float = _Z_95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("wilson_interval requires n >= 1.")
    successes = int(successes)
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n.")
    proportion = successes / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)
        )
        / denominator
    )
    return (
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    )


def zero_failure_upper_bound(n: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper error bound after zero failures in ``n`` trials.

    ``P(0 failures | p) = (1 - p)^n``; solving ``(1 - p)^n = alpha`` gives
    ``p_upper = 1 - alpha**(1/n)``.  Zero observed failures never proves a
    zero error rate.
    """
    if n <= 0:
        raise ValueError("zero_failure_upper_bound requires n >= 1.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    return 1.0 - alpha ** (1.0 / n)


def zero_failure_sample_size(p0: float, alpha: float = 0.05) -> int:
    """Independent zero-failure decisions needed to support bound ``p0``.

    Smallest ``n`` with ``(1 - p0)^n <= alpha``, i.e.
    ``n >= ln(alpha) / ln(1 - p0)``.
    """
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be in (0, 1).")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    return math.ceil(math.log(alpha) / math.log(1.0 - p0))


def sample_size_reference(
    error_rates: Sequence[float] = (0.05, 0.02, 0.01, 0.005, 0.001),
    confidences: Sequence[float] = (0.95, 0.99),
) -> list[dict[str, Any]]:
    """Zero-failure sample-size table for the requested bounds."""
    table = []
    for rate in error_rates:
        for confidence in confidences:
            table.append(
                {
                    "max_error_rate": rate,
                    "confidence": confidence,
                    "required_zero_failure_samples": zero_failure_sample_size(
                        rate, 1.0 - confidence
                    ),
                }
            )
    return table


def repeatability_summary(
    groups: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Within-specimen spread of repeated captures of the same physical unit.

    Repeated captures quantify repeatability only; a population accuracy
    claim needs independent specimens, which is why the note is embedded in
    the summary itself.
    """
    cleaned: dict[str, list[float]] = {}
    for specimen, values in groups.items():
        if len(values) < 1:
            continue
        cleaned[str(specimen)] = _require_finite(values, "repeatability_summary")
    if not cleaned:
        raise ValueError("repeatability_summary requires at least one group.")

    per_specimen: dict[str, dict[str, Any]] = {}
    variances: list[float] = []
    means: list[float] = []
    for specimen, values in sorted(cleaned.items()):
        count = len(values)
        mean = sum(values) / count
        per_specimen[specimen] = {
            "captures": count,
            "mean": mean,
            "spread": max(values) - min(values),
            "sample_std": (
                math.sqrt(sum((v - mean) ** 2 for v in values) / (count - 1))
                if count > 1
                else None
            ),
        }
        means.append(mean)
        if count > 1:
            variances.append(per_specimen[specimen]["sample_std"] ** 2)

    grand_mean = sum(means) / len(means)
    return {
        "specimens": len(per_specimen),
        "captures_total": sum(len(v) for v in cleaned.values()),
        "per_specimen": per_specimen,
        "pooled_within_specimen_std": (
            math.sqrt(sum(variances) / len(variances)) if variances else None
        ),
        "between_specimen_std": (
            math.sqrt(
                sum((m - grand_mean) ** 2 for m in means) / (len(means) - 1)
            )
            if len(means) > 1
            else None
        ),
        "note": (
            "Repeated captures estimate repeatability, not population "
            "accuracy; the physical specimen is the independent unit."
        ),
    }
