"""Two-capture confirmation for high-stakes grading.

A single capture can be fooled by a bad mask boundary, a specular highlight,
or an unlucky uncertainty interval.  Confirmation grading converts those
residual single-capture errors into refusals: an automatic grade is emitted
only when two independent captures of the same specimen (repositioned or
rotated between shots) agree on the class.  Disagreement is never resolved by
averaging into a decision; it goes to review.

This trades coverage for correctness, which is the only honest way to push
accepted-decision error toward zero without ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .domain import InspectionResult, SizeClass, Status


@dataclass
class ConfirmationOutcome:
    status: Status
    size_class: SizeClass | None
    size_band: str | None
    reasons: list[str] = field(default_factory=list)
    warning: str | None = None
    per_capture: list[dict[str, Any]] = field(default_factory=list)


def _capture_summary(result: InspectionResult, index: int) -> dict[str, Any] | None:
    if len(result.instances) != 1 or result.instances[0].decision is None:
        return None
    decision = result.instances[0].decision
    measurement = result.instances[0].measurement
    return {
        "capture": index,
        "status": decision.status.value,
        "size_class": decision.size_class.value if decision.size_class else None,
        "size_band": decision.band_name,
        "metric": measurement.metric if measurement else None,
        "value_mm": measurement.value_mm if measurement else None,
        "uncertainty_mm": measurement.uncertainty_mm if measurement else None,
    }


def confirm_captures(results: list[InspectionResult]) -> ConfirmationOutcome:
    """Require two independent accepted captures that agree on the grade."""
    if len(results) != 2:
        raise ValueError("Confirmation grading needs exactly two captures.")
    summaries: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        summary = _capture_summary(result, index)
        if summary is None:
            raise ValueError(
                f"Capture {index} does not contain exactly one graded instance."
            )
        summaries.append(summary)

    for summary in summaries:
        if summary["status"] != Status.ACCEPTED.value:
            return ConfirmationOutcome(
                status=Status.REVIEW_BORDERLINE,
                size_class=None,
                size_band=None,
                reasons=[
                    "CONFIRMATION_BLOCKED: a capture did not produce an accepted "
                    f"grade (capture {summary['capture']}: {summary['status']}). "
                    "Fix the capture condition and re-shoot both frames.",
                ],
                warning="No confirmed grade; single-capture failure blocks confirmation.",
                per_capture=summaries,
            )

    first, second = summaries
    classes_agree = first["size_class"] == second["size_class"]
    bands_agree = first["size_band"] == second["size_band"]
    if not (classes_agree and bands_agree):
        return ConfirmationOutcome(
            status=Status.REVIEW_BORDERLINE,
            size_class=None,
            size_band=None,
            reasons=[
                "DUAL_CAPTURE_CLASS_DISAGREEMENT: capture A graded "
                f"{first['size_band'] or first['size_class']} and capture B graded "
                f"{second['size_band'] or second['size_class']}. Recapture both or "
                "route to manual review; the disagreement is never averaged away."
            ],
            warning="Two captures disagree; no automatic grade.",
            per_capture=summaries,
        )

    reasons = [
        "DUAL_CAPTURE_CONFIRMED: two independent captures agree on "
        f"{first['size_band'] or first['size_class']}."
    ]
    warning = None
    if (
        first["value_mm"] is not None
        and second["value_mm"] is not None
        and first["uncertainty_mm"] is not None
        and second["uncertainty_mm"] is not None
    ):
        spread = abs(float(first["value_mm"]) - float(second["value_mm"]))
        combined = float(first["uncertainty_mm"]) + float(second["uncertainty_mm"])
        if spread > combined:
            reasons.append(
                "DUAL_CAPTURE_VALUE_SPREAD_HIGH: class agrees but measured values "
                f"differ by {spread:.2f} mm (combined uncertainty {combined:.2f} mm); "
                "check station setup."
            )
    return ConfirmationOutcome(
        status=Status.ACCEPTED,
        size_class=SizeClass(first["size_class"]) if first["size_class"] else None,
        size_band=first["size_band"],
        reasons=reasons,
        warning=warning,
        per_capture=summaries,
    )
