"""Evaluate real measured specimens, or explicitly refuse to invent a result.

The script never substitutes synthetic images for real accuracy evidence and
never invents a threshold.  If the manifest is empty, unmeasured, unlicensed, or
the crop policy has no approved threshold, the report status is BLOCKED and no
confusion matrix is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from growize_vision.config import load_inference_settings, load_policies
from growize_vision.dataset import validate_manifest
from growize_vision.domain import MediaContext, SourceType
from growize_vision.evidence import (
    PROJECT_ROOT,
    environment_metadata,
    load_json,
    relative_to_project,
    write_json,
)
from growize_vision.inference import InferenceRequest, InferenceService
from growize_vision.media import load_image_file

DEFAULT_TEMPLATE = PROJECT_ROOT / "data" / "dataset_manifest_template.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "baseline_evaluation.json"
GRADES = ("SMALL", "BIG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--marker-side-mm", type=float, default=None)
    parser.add_argument("--expected-marker-id", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--demo-policy",
        action="store_true",
        help="Use clearly-labelled assessment demo cut-offs instead of real policy.",
    )
    return parser.parse_args()


def _primary_image(specimen: dict[str, Any]) -> dict[str, Any] | None:
    images = specimen.get("images")
    if not isinstance(images, list) or not images:
        return None
    for image in images:
        if isinstance(image, dict) and image.get("role") == "primary":
            return image
    first = images[0]
    return first if isinstance(first, dict) else None


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    policies = load_policies()
    settings = load_inference_settings()
    report: dict[str, Any] = {
        "report_type": "real_specimen_baseline_evaluation",
        "status": "BLOCKED",
        "evidence_class": (
            "ASSESSMENT_DEMO_ONLY_NOT_REAL_ACCURACY"
            if args.demo_policy
            else "REAL_MEASURED_SPECIMENS"
        ),
        "environment": environment_metadata(),
        "manifest": {
            "path": relative_to_project(manifest_path),
            "exists": manifest_path.exists(),
            "sha256": (
                hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if manifest_path.exists()
                else None
            ),
        },
        "config": {
            "policy_version": policies.version,
            "policy_sha256": policies.config_sha256,
            "settings_version": settings.version,
            "settings_sha256": settings.config_sha256,
        },
        "protocol": {
            "marker_side_mm": args.marker_side_mm,
            "expected_marker_id": args.expected_marker_id,
            "demo_policy": args.demo_policy,
            "one_primary_image_per_specimen": True,
        },
        "blockers": [],
        "warnings": [],
        "skipped_crops": [],
        "summary": None,
        "confusion": None,
        "status_counts": None,
        "observations": [],
        "limitations": [
            "Only an empty/pending manifest template exists unless a licensed, "
            "measured dataset has been supplied.",
            "No real threshold is approved; strict evaluation therefore cannot run.",
            "A demo-policy run is engineering evidence, never buyer or ARL evidence.",
        ],
    }

    if not manifest_path.exists():
        report["blockers"].append(f"Manifest does not exist: {manifest_path}")
        output = write_json(args.output, report)
        print(f"BLOCKED: manifest missing. Report: {output}")
        return 2

    manifest = load_json(manifest_path)
    validation = validate_manifest(
        manifest_path,
        image_root=args.image_root,
        require_images=True,
        allow_empty=False,
        allowed_crops=set(policies.crop_keys),
    )
    report["validation"] = validation
    if not validation["ok"]:
        report["blockers"].extend(validation["errors"])
        report["blockers"].append("Dataset manifest validation failed.")
        output = write_json(args.output, report)
        print(f"BLOCKED: manifest validation failed. Report: {output}")
        return 2

    if args.marker_side_mm is None or args.marker_side_mm <= 0:
        report["blockers"].append(
            "A measured marker side length (--marker-side-mm) is required; "
            "millimetres cannot be claimed without it."
        )

    specimens = manifest.get("specimens", [])
    real_specimens = [item for item in specimens if not bool(item.get("synthetic"))]
    report["counts"] = {
        "total_specimens": len(specimens),
        "real_specimens": len(real_specimens),
        "synthetic_specimens": len(specimens) - len(real_specimens),
    }
    if not real_specimens:
        report["blockers"].append(
            "NO_REAL_SPECIMENS: the manifest contains no real measured specimens; "
            "synthetic fixtures cannot produce an accuracy claim."
        )

    crop_policies: dict[str, Any] = {}
    for crop in sorted({str(item.get("crop")) for item in real_specimens}):
        try:
            policy = policies.policy(crop, demo=args.demo_policy)
        except Exception as exc:
            report["skipped_crops"].append({"crop": crop, "reason": str(exc)})
            continue
        crop_policies[crop] = policy
        if policy.threshold_mm is None:
            report["skipped_crops"].append(
                {
                    "crop": crop,
                    "reason": (
                        "NO_APPROVED_THRESHOLD: no ARL/Growize threshold supplied. "
                        "Use --demo-policy only for non-authoritative engineering checks."
                    ),
                }
            )
        elif policy.demo:
            report["warnings"].append(
                f"{crop}: ASSESSMENT_DEMO_ONLY threshold {policy.threshold_mm} mm; "
                "not ARL approved."
            )

    evaluable = [
        item
        for item in real_specimens
        if item.get("reference_grade") in GRADES
        and str(item.get("crop")) in crop_policies
        and crop_policies[str(item.get("crop"))].threshold_mm is not None
    ]
    report["counts"]["evaluable_specimens"] = len(evaluable)
    if not evaluable:
        if not report["blockers"]:
            report["blockers"].append(
                "NO_EVALUABLE_SPECIMENS: no real specimen has both a SMALL/BIG "
                "reference grade and an approved crop threshold."
            )
        output = write_json(args.output, report)
        print(f"BLOCKED: no evaluable real specimens. Report: {output}")
        return 2

    if report["blockers"]:
        output = write_json(args.output, report)
        print(f"BLOCKED: prerequisites unmet. Report: {output}")
        return 2

    image_root = Path(args.image_root) if args.image_root else manifest_path.parent
    service = InferenceService(policies=policies, settings=settings)
    status_counts: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = {
        truth: Counter({predicted: 0 for predicted in GRADES}) for truth in GRADES
    }
    measurement_errors: list[float] = []
    accepted = 0
    refused = 0
    ungraded = 0

    for specimen in evaluable:
        specimen_id = str(specimen["specimen_id"])
        crop = str(specimen["crop"])
        policy = crop_policies[crop]
        image = _primary_image(specimen)
        if image is None:
            ungraded += 1
            report["observations"].append(
                {"specimen_id": specimen_id, "outcome": "UNGRADED", "reason": "no image"}
            )
            continue
        image_path = image_root / str(image["path"])
        decoded = load_image_file(
            image_path,
            max_pixels=settings.max_image_pixels,
            max_bytes=settings.max_upload_bytes,
        )
        result = service.inspect_frame(
            decoded.bgr,
            MediaContext(
                source_type=SourceType.IMAGE,
                source_id=specimen_id,
                device="measured-dataset",
                batch_id=manifest.get("provenance", {}).get("dataset_name"),
            ),
            InferenceRequest(
                crop_key=crop,
                marker_side_mm=args.marker_side_mm,
                expected_marker_id=args.expected_marker_id,
                demo_policy=args.demo_policy,
            ),
        )
        status_counts[result.status.value] += 1
        predicted = result.size_class.value if result.size_class else None
        truth = str(specimen["reference_grade"])
        observation: dict[str, Any] = {
            "specimen_id": specimen_id,
            "crop": crop,
            "reference_grade": truth,
            "status": result.status.value,
            "predicted_grade": predicted,
            "processing_ms": round(result.processing_ms, 3),
        }
        if result.status.value == "ACCEPTED" and predicted in GRADES:
            accepted += 1
            confusion[truth][predicted] += 1
            measurement = (
                result.instances[0].measurement
                if result.instances and result.instances[0].measurement
                else None
            )
            if (
                measurement is not None
                and str(specimen.get("measured_metric")) == policy.metric
                and specimen.get("measurement_1_mm") is not None
            ):
                error = abs(
                    measurement.value_mm - float(specimen["measurement_1_mm"])
                )
                measurement_errors.append(error)
                observation["measurement_mm"] = round(measurement.value_mm, 3)
                observation["caliper_mm"] = round(
                    float(specimen["measurement_1_mm"]), 3
                )
                observation["absolute_error_mm"] = round(error, 3)
            observation["outcome"] = "GRADED"
        else:
            refused += 1
            observation["outcome"] = "REFUSED"
            observation["reasons"] = result.reasons[:2]
        report["observations"].append(observation)

    graded_total = sum(sum(counter.values()) for counter in confusion.values())
    correct = sum(confusion[grade][grade] for grade in GRADES)
    report["status"] = "EVALUATED" if graded_total else "NOT_EVALUATED"
    report["status_counts"] = dict(sorted(status_counts.items()))
    report["confusion"] = {
        truth: dict(confusion[truth]) for truth in GRADES
    }
    report["summary"] = {
        "observations_attempted": len(evaluable),
        "accepted_observations": accepted,
        "refused_observations": refused,
        "ungraded_observations": ungraded,
        "coverage_of_evaluable": (
            round(accepted / len(evaluable), 4) if evaluable else None
        ),
        "graded_observations": graded_total,
        "correct_on_graded_subset": correct,
        "accuracy_on_graded_subset": (
            round(correct / graded_total, 4) if graded_total else None
        ),
        "measurement_error_observations": len(measurement_errors),
        "measurement_mae_mm": (
            round(statistics.mean(measurement_errors), 3)
            if measurement_errors
            else None
        ),
        "measurement_median_absolute_error_mm": (
            round(statistics.median(measurement_errors), 3)
            if measurement_errors
            else None
        ),
        "measurement_max_absolute_error_mm": (
            round(max(measurement_errors), 3) if measurement_errors else None
        ),
        "metric_scope_note": (
            "Measurement error is reported only where the specimen caliper metric "
            "matches the crop policy metric. Refusals are excluded from the graded "
            "subset and reported separately as coverage."
        ),
    }
    if graded_total == 0:
        report["blockers"].append(
            "All evaluable attempts were refused, so no accuracy or measurement "
            "metric can be computed."
        )
    output = write_json(args.output, report)
    print(f"Status {report['status']}. Report written to {output}")
    if report["summary"]["accuracy_on_graded_subset"] is not None:
        print(
            "graded-subset accuracy={:.1%} coverage={:.1%} n_graded={}".format(
                report["summary"]["accuracy_on_graded_subset"],
                report["summary"]["coverage_of_evaluable"],
                graded_total,
            )
        )
    return 0 if report["status"] == "EVALUATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
