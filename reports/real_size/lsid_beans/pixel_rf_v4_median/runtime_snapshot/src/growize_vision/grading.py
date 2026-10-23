"""Deterministic, versioned size policy evaluation."""

from __future__ import annotations

import math

from .domain import (
    GradeDecision,
    GradePolicy,
    PhysicalMeasurement,
    SizeClass,
    Status,
)

COMPARATORS = {
    "greater_than_or_equal",
    "greater_than",
    "less_than",
    "less_than_or_equal",
}


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
    if policy.threshold_mm is None:
        return GradeDecision(
            size_class=None,
            status=Status.POLICY_REQUIRED,
            reasons=[
                "NO_APPROVED_THRESHOLD: ARL/Growize has not supplied a threshold "
                "for this crop. Enable demo mode only for an explicitly labelled "
                "engineering demonstration."
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
    values = (
        measurement.value_mm,
        measurement.uncertainty_mm,
        measurement.lower_mm,
        measurement.upper_mm,
        policy.threshold_mm,
        policy.guard_band_mm,
    )
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

    if policy.demo:
        reasons.append("ASSESSMENT_DEMO_POLICY_NOT_ARL_APPROVED")
    elif policy.approval_status == "EMPIRICAL_REFERENCE":
        reasons.append("EMPIRICAL_REFERENCE_AVERAGE_NOT_POPULATION_STANDARD")
    else:
        reasons.append("APPROVED_POLICY")
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
        warning=(
            "DEMO CUT-OFF ONLY: this is not an ARL/Growize-approved grade."
            if policy.demo
            else "Sample reference only; not a validated population standard."
            if policy.approval_status == "EMPIRICAL_REFERENCE"
            else None
        ),
    )
