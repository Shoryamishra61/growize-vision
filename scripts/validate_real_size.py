"""Evaluate calibrated sizing on reviewed original photos with independent measurements.

This isolates metrology using the recorded crop policy; it does not validate
automatic species identification. No generated images or transforms are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from growize_vision.domain import Status  # noqa: E402
from growize_vision.inference import InferenceRequest, InferenceService  # noqa: E402
from growize_vision.media import load_image_file  # noqa: E402
from growize_vision.metrics import (  # noqa: E402
    error_summary,
    repeatability_summary,
    sample_size_reference,
    wilson_interval,
    zero_failure_upper_bound,
)
from growize_vision.real_photos import (  # noqa: E402
    reference_statistics,
    verify_original_photo,
    verify_specimen_splits,
)


def evaluate(manifest: Path, root: Path, output: Path) -> dict:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    report = {
        "scope": "physical_metrology_with_recorded_crop_policy",
        "reference_statistic": "median",
        "status": "BLOCKED",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "original_photos": len(rows),
        "unique_specimens": len({r.get("specimen_id") for r in rows}),
        "accuracy": None,
        "accepted_accuracy": None,
        "measurement_mae_by_unit": {},
        "limitations": [
            "Finite test results do not establish universal 100% accuracy.",
            "Original-photo authenticity requires independent provenance review.",
            "Recorded crop policy is supplied; automatic recognition is not evaluated.",
            "Reference averages are estimated from training specimens only; population representativeness is unverified.",
        ],
    }
    try:
        if not rows:
            raise ValueError(
                "No reviewed original photographs with physical measurements were supplied."
            )
        verify_specimen_splits(rows)
        service = InferenceService()
        for row in rows:
            verify_original_photo(row, root)
        references = reference_statistics(rows)
        if not references:
            raise ValueError(
                "No independently measured training specimens for the reference average"
            )
        report["reference_averages"] = references
        configured = set()
        for reference in references:
            policy = service.policies.policy(reference["crop_key"], demo=False)
            if (
                reference["metric"] != policy.metric
                or reference["physical_unit"] != policy.physical_unit
            ):
                raise ValueError(
                    "Reference item unit or measurement metric does not match the policy"
                )
            if reference["specimens"] < 2 or policy.crop_key in configured:
                raise ValueError(
                    "Reference needs multiple independent specimens and one definition per crop"
                )
            configured.add(policy.crop_key)
            service.policies.crops[policy.crop_key] = dict(
                service.policies.crops[policy.crop_key],
                threshold_mm=reference["median"],
                approval_status="EMPIRICAL_REFERENCE",
                source=f"Training-specimen median from manifest {report['manifest_sha256']}; not a population standard",
            )
        service.policies.config_sha256 = hashlib.sha256(
            json.dumps(service.policies.crops, sort_keys=True).encode()
        ).hexdigest()
        service = InferenceService(policies=service.policies)
        prepared = []
        for row in rows:
            path = verify_original_photo(row, root)
            policy = service.policies.policy(row["crop_key"], demo=False)
            if policy.threshold_mm is None:
                raise ValueError(f"No measured reference average for {row['crop_key']}")
            if row.get("metric") != policy.metric:
                raise ValueError("Ground-truth and grading metrics differ")
            measured = row.get("measured_value")
            unit = "cm2" if policy.metric == "projected_area_cm2" else "mm"
            if (
                row.get("measurement_unit") != unit
                or row.get("physical_unit") != policy.physical_unit
            ):
                raise ValueError(
                    "Ground-truth measurement unit or item unit does not match the policy"
                )
            marker = row.get("marker_side_mm")
            for value in (measured, marker):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        "Independent measurement and measured marker side must be finite positive numbers"
                    )
            if not row.get("measurement_reviewed_by"):
                raise ValueError("Missing independent measurement reviewer")
            expected = "BIG" if measured >= policy.threshold_mm else "SMALL"
            if row.get("expected_size", expected) != expected:
                raise ValueError("Recorded size label disagrees with the frozen training median")
            if row.get("split") != "test":
                continue
            prepared.append((row, path, expected))
        if not prepared:
            raise ValueError("No independent test specimens supplied")
        predictions = []
        errors_by_unit = defaultdict(list)
        values_by_specimen = defaultdict(list)
        for row, path, expected in prepared:
            decoded = load_image_file(path)
            result = service.inspect_frame(
                decoded.bgr,
                request=InferenceRequest(
                    crop_key=row["crop_key"],
                    marker_side_mm=row["marker_side_mm"],
                    expected_marker_id=row.get("marker_id", 0),
                    demo_policy=False,
                ),
            )
            label = (
                result.size_class.value
                if result.status == Status.ACCEPTED and result.size_class is not None
                else None
            )
            measurement = (
                result.instances[0].decision.measurement
                if len(result.instances) == 1 and result.instances[0].decision
                else None
            )
            if measurement:
                errors_by_unit[measurement.unit].append(
                    measurement.value_mm - row["measured_value"]
                )
                values_by_specimen[row["specimen_id"]].append(measurement.value_mm)
            predictions.append(
                {
                    "path": row["path"],
                    "specimen_id": row["specimen_id"],
                    "expected": expected,
                    "predicted": label,
                    "status": result.status.value,
                    "pipeline_sha256": result.pipeline_sha256,
                }
            )
        accepted = [r for r in predictions if r["predicted"] is not None]
        correct = sum(r["predicted"] == r["expected"] for r in predictions)
        correct_accepted = sum(r["predicted"] == r["expected"] for r in accepted)
        failures_accepted = len(accepted) - correct_accepted
        if failures_accepted == 0 and accepted:
            accepted_error_bound = {
                "observed_failures": 0,
                "accepted_decisions": len(accepted),
                "upper_error_rate_95": zero_failure_upper_bound(len(accepted)),
                "interpretation": (
                    "Zero failures do not prove zero error: with "
                    f"{len(accepted)} accepted decisions the one-sided 95% "
                    "upper error bound remains "
                    f"{zero_failure_upper_bound(len(accepted)):.4f}."
                ),
            }
        else:
            accepted_error_bound = {
                "observed_failures": failures_accepted,
                "accepted_decisions": len(accepted),
                "interpretation": (
                    "Failures observed; use accepted_accuracy_ci95 (Wilson), "
                    "not a zero-failure bound."
                ),
            }
        repeatability = None
        if any(len(values) > 1 for values in values_by_specimen.values()):
            repeatability = repeatability_summary(values_by_specimen)
        report.update(
            status="EVALUATED_NOT_RELEASED",
            evaluated_photos=len(predictions),
            evaluated_specimens=len({r["specimen_id"] for r in predictions}),
            coverage=len(accepted) / len(predictions),
            coverage_ci95=wilson_interval(len(accepted), len(predictions)),
            accuracy=correct / len(predictions),
            accuracy_ci95=wilson_interval(correct, len(predictions)),
            accepted_accuracy=correct_accepted / len(accepted) if accepted else None,
            accepted_accuracy_ci95=(
                wilson_interval(correct_accepted, len(accepted)) if accepted else None
            ),
            accepted_error_bound=accepted_error_bound,
            measurement_error_by_unit={
                unit: error_summary(values) for unit, values in errors_by_unit.items()
            },
            measurement_mae_by_unit={
                unit: sum(abs(v) for v in values) / len(values)
                for unit, values in errors_by_unit.items()
            },
            repeatability_by_specimen=repeatability,
            sample_size_reference=sample_size_reference(),
            confusion=dict(
                Counter(f"{r['expected']}->{r['predicted'] or 'REFUSED'}" for r in predictions)
            ),
            predictions=predictions,
        )
    except (ValueError, KeyError, OSError) as exc:
        report["reason"] = str(exc)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/real_size/validation.json")
    args = parser.parse_args()
    report = evaluate(args.manifest, args.root, args.output)
    print(
        json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2)
    )
    return 0 if report["status"] == "EVALUATED_NOT_RELEASED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
