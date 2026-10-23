"""Deterministic, versioned size policy evaluation."""

from __future__ import annotations

import math

from .domain import (
    GradeDecision,
    GradePolicy,
    PhysicalMeasurement,
    SizeBand,
    SizeClass,
    Status,
)

COMPARATORS = {
    "greater_than_or_equal",
    "greater_than",
    "less_than",
    "less_than_or_equal",
}


def _provenance_disclaimer(policy: GradePolicy) -> tuple[list[str], str | None]:
    """Reason/warning suffix for the policy's evidence tier."""
    if policy.demo:
        return (
            ["ASSESSMENT_DEMO_POLICY_NOT_ARL_APPROVED"],
            "DEMO CUT-OFF ONLY: this is not an ARL/Growize-approved grade.",
        )
    if policy.approval_status == "EMPIRICAL_REFERENCE":
        return (
            ["EMPIRICAL_REFERENCE_AVERAGE_NOT_POPULATION_STANDARD"],
            "Sample reference only; not a validated population standard.",
        )
    if policy.approval_status == "PUBLISHED_REFERENCE":
        return (
            ["PUBLISHED_REFERENCE_NOT_POPULATION_VALIDATED"],
            "Published-reference cut-off (standard, trade practice, or research "
            "data); not validated on the local population. Confirm with the "
            "buyer or an approved standard before commercial use.",
        )
    return ["APPROVED_POLICY"], None


def _evaluate_bands(
    measurement: PhysicalMeasurement,
    policy: GradePolicy,
    calibration_id: str | None,
) -> GradeDecision:
    """Ordered size-band rule: interval must sit entirely inside one band.

    Band i covers ``[min_mm_i, min_mm_{i+1})``; the first band starts at 0
    and the last is unbounded above.  The measurement interval (with the
    guard band applied to each edge) must not cross any band boundary, so a
    MEDIUM/LARGE style grade carries the same refusal discipline as the
    binary rule.  Two bands reproduce the binary cut-off exactly, with
    SMALL = first band and BIG = last band.
    """
    bands = policy.bands or ()
    guard = max(0.0, float(policy.guard_band_mm))
    lower = float(measurement.lower_mm)
    upper = float(measurement.upper_mm)
    value = float(measurement.value_mm)
    for index, band in enumerate(bands):
        lower_ok = index == 0 or lower >= band.min_mm + guard
        upper_edge = bands[index + 1].min_mm if index + 1 < len(bands) else None
        upper_ok = upper_edge is None or upper < upper_edge - guard
        if not (lower_ok and upper_ok):
            continue
        size_class = (
            SizeClass.SMALL if index == 0
            else SizeClass.BIG if index == len(bands) - 1
            else None
        )
        disclaimer_reasons, disclaimer_warning = _provenance_disclaimer(policy)
        reasons = [f"INTERVAL_ENTIRELY_WITHIN_BAND:{band.name}", *disclaimer_reasons]
        if policy.bands_derived:
            reasons.append(
                "DERIVED_BANDS_AROUND_PUBLISHED_CUT: band edges at ±20% of the "
                "published cut-off are engineering-derived for numeric 1/2/3 "
                "grading, not published band edges."
            )
        warning = (
            "DEMO BANDS ONLY: these cut-offs are not ARL/Growize-approved grades."
            if policy.demo
            else disclaimer_warning
        )
        if policy.bands_derived and warning:
            warning += " Band edges are derived (±20% of the published cut)."
        return GradeDecision(
            size_class=size_class,
            status=Status.ACCEPTED,
            reasons=reasons,
            policy=policy,
            measurement=measurement,
            threshold_mm=policy.threshold_mm,
            comparator=policy.comparator,
            policy_version=policy.policy_version,
            calibration_id=calibration_id,
            warning=warning,
            band_name=band.name,
        )

    edges = [band.min_mm for band in bands[1:]] or [0.0]
    nearest = min(edges, key=lambda edge: abs(value - edge))
    reasons = ["UNCERTAINTY_OR_GUARD_BAND_CROSSES_BAND_EDGE"]
    if abs(value - nearest) <= guard:
        reasons.append(
            "VALUE_INSIDE_BAND_GUARD: the measured value sits in the deliberate "
            "no-decision band around a grade edge; route to manual review."
        )
    else:
        reasons.append(
            "MEASUREMENT_INTERVAL_CROSSES_BAND_EDGE: the uncertainty interval "
            "reaches a grade boundary; recapture flatter and closer with the "
            "marker fully visible to narrow the interval."
        )
    return GradeDecision(
        size_class=None,
        status=Status.REVIEW_BORDERLINE,
        reasons=reasons,
        policy=policy,
        measurement=measurement,
        threshold_mm=policy.threshold_mm,
        comparator=policy.comparator,
        policy_version=policy.policy_version,
        calibration_id=calibration_id,
        warning=(
            "DEMO BANDS ONLY: this review state is still a safe refusal."
            if policy.demo
            else "No automatic size grade; recapture or request manual review."
        ),
        band_name=None,
    )


def evaluate_measurement(
    measurement: PhysicalMeasurement | None,
    policy: GradePolicy | None,
    calibration_id: str | None,
) -> GradeDecision:
    """Apply the guard-band interval rule.

    For the normal ``>=`` policy:

    * BIG only when ``value - uncertainty >= threshold + guard``
    * SMALL only when ``value + uncertainty < threshold - guard``
    * otherwise REVIEW_BORDERLINE, with no size class

    The guard band is a deliberate no-decision band around the threshold.  It
    does not replace the uncertainty interval; both must be clear.
    """
    if policy is None:
        return GradeDecision(
            size_class=None,
            status=Status.POLICY_REQUIRED,
            reasons=["NO_CROP_POLICY: select an operator crop policy."],
            calibration_id=calibration_id,
        )
    if policy.threshold_mm is None and not policy.bands:
        return GradeDecision(
            size_class=None,
            status=Status.POLICY_REQUIRED,
            reasons=[
                "NO_THRESHOLD_POLICY: no ARL-approved, published-reference, or "
                "demo threshold exists for this crop. Physical grading is "
                "withheld rather than guessed."
            ],
            policy=policy,
            threshold_mm=None,
            comparator=policy.comparator,
            policy_version=policy.policy_version,
            calibration_id=calibration_id,
        )
    if policy.comparator not in COMPARATORS:
        return GradeDecision(
            size_class=None,
            status=Status.POLICY_REQUIRED,
            reasons=[f"UNSUPPORTED_COMPARATOR: {policy.comparator!r}"],
            policy=policy,
            threshold_mm=policy.threshold_mm,
            policy_version=policy.policy_version,
            calibration_id=calibration_id,
        )
    if measurement is None:
        return GradeDecision(
            size_class=None,
            status=Status.REVIEW_BORDERLINE,
            reasons=["NO_PHYSICAL_MEASUREMENT: cannot evaluate size policy."],
            policy=policy,
            threshold_mm=policy.threshold_mm,
            comparator=policy.comparator,
            policy_version=policy.policy_version,
            calibration_id=calibration_id,
        )
    if measurement.metric != policy.metric:
        return GradeDecision(
            size_class=None,
            status=Status.POLICY_REQUIRED,
            reasons=[
                "METRIC_MISMATCH: policy requires "
                f"{policy.metric!r}, measurement produced {measurement.metric!r}."
            ],
            policy=policy,
            measurement=measurement,
            threshold_mm=policy.threshold_mm,
            comparator=policy.comparator,
            policy_version=policy.policy_version,
            calibration_id=calibration_id,
        )
    values: tuple[float, ...] = (
        measurement.value_mm,
        measurement.uncertainty_mm,
        measurement.lower_mm,
        measurement.upper_mm,
        policy.guard_band_mm,
    )
    if policy.threshold_mm is not None:
        values += (policy.threshold_mm,)
    if not all(math.isfinite(float(v)) for v in values):
        return GradeDecision(
            size_class=None,
            status=Status.REVIEW_BORDERLINE,
            reasons=["NONFINITE_VALUE: measurement or policy contains NaN/inf."],
            policy=policy,
            measurement=measurement,
            threshold_mm=policy.threshold_mm,
            comparator=policy.comparator,
            policy_version=policy.policy_version,
            calibration_id=calibration_id,
        )
    if measurement.uncertainty_mm < 0 or measurement.lower_mm > measurement.upper_mm:
        return GradeDecision(
            size_class=None,
            status=Status.REVIEW_BORDERLINE,
            reasons=["INVALID_UNCERTAINTY: measurement interval is malformed."],
            policy=policy,
            measurement=measurement,
            threshold_mm=policy.threshold_mm,
            comparator=policy.comparator,
            policy_version=policy.policy_version,
            calibration_id=calibration_id,
        )

    if policy.bands and len(policy.bands) >= 2:
        return _evaluate_bands(measurement, policy, calibration_id)
    threshold = float(policy.threshold_mm)
    guard = max(0.0, float(policy.guard_band_mm))
    lower = float(measurement.lower_mm)
    upper = float(measurement.upper_mm)
    comparator = policy.comparator
    reasons: list[str] = []
    size_class: SizeClass | None = None
    accepted = False

    if comparator == "greater_than_or_equal":
        if lower >= threshold + guard:
            size_class, accepted = SizeClass.BIG, True
            reasons.append("INTERVAL_ENTIRELY_AT_OR_ABOVE_THRESHOLD")
        elif upper < threshold - guard:
            size_class, accepted = SizeClass.SMALL, True
            reasons.append("INTERVAL_ENTIRELY_BELOW_THRESHOLD")
    elif comparator == "greater_than":
        if lower > threshold + guard:
            size_class, accepted = SizeClass.BIG, True
            reasons.append("INTERVAL_ENTIRELY_ABOVE_THRESHOLD")
        elif upper <= threshold - guard:
            size_class, accepted = SizeClass.SMALL, True
            reasons.append("INTERVAL_ENTIRELY_AT_OR_BELOW_THRESHOLD")
    elif comparator == "less_than":
        if upper < threshold - guard:
            size_class, accepted = SizeClass.BIG, True
            reasons.append("INTERVAL_ENTIRELY_BELOW_THRESHOLD")
        elif lower > threshold + guard:
            size_class, accepted = SizeClass.SMALL, True
            reasons.append("INTERVAL_ENTIRELY_ABOVE_THRESHOLD")
    elif comparator == "less_than_or_equal":
        if upper <= threshold - guard:
            size_class, accepted = SizeClass.BIG, True
            reasons.append("INTERVAL_ENTIRELY_AT_OR_BELOW_THRESHOLD")
        elif lower > threshold + guard:
            size_class, accepted = SizeClass.SMALL, True
            reasons.append("INTERVAL_ENTIRELY_ABOVE_THRESHOLD")

    if not accepted:
        size_class = None
        reasons.append("UNCERTAINTY_OR_GUARD_BAND_CROSSES_THRESHOLD")
        value = float(measurement.value_mm)
        if threshold - guard <= value <= threshold + guard:
            reasons.append(
                "VALUE_INSIDE_GUARD_BAND: the measured value sits in the "
                "deliberate no-decision band around the threshold; route to "
                "manual review."
            )
        else:
            reasons.append(
                "MEASUREMENT_INTERVAL_CROSSES_THRESHOLD: the uncertainty "
                "interval reaches the cut-off; recapture flatter and closer "
                "with the marker fully visible to narrow the interval."
            )
        return GradeDecision(
            size_class=None,
            status=Status.REVIEW_BORDERLINE,
            reasons=reasons,
            policy=policy,
            measurement=measurement,
            threshold_mm=threshold,
            comparator=comparator,
            policy_version=policy.policy_version,
            calibration_id=calibration_id,
            warning="No automatic size class; recapture with the object flatter and "
            "the marker clearly visible, or request manual review."
            if not policy.demo
            else "DEMO POLICY: no approved threshold exists. This review state is "
            "still a safe refusal.",
        )

    disclaimer_reasons, disclaimer_warning = _provenance_disclaimer(policy)
    reasons.extend(disclaimer_reasons)
    return GradeDecision(
        size_class=size_class,
        status=Status.ACCEPTED,
        reasons=reasons,
        policy=policy,
        measurement=measurement,
        threshold_mm=threshold,
        comparator=comparator,
        policy_version=policy.policy_version,
        calibration_id=calibration_id,
        warning=disclaimer_warning,
    )
