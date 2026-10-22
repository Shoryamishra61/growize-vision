"""Run the synthetic contract suite and archive exact observed outcomes.

Synthetic fixtures have vector-defined geometry.  Passing this suite proves the
pipeline honours its status/refusal contract and recovers known geometry; it is
NOT evidence of real-produce accuracy, robustness, or field readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
from pathlib import Path

from growize_vision.config import load_inference_settings, load_policies
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

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "demo" / "demo_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "synthetic_evaluation.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    fixtures = [
        fixture
        for fixture in manifest.get("fixtures", [])
        if Path(str(fixture.get("path", ""))).suffix.lower() in IMAGE_SUFFIXES
    ]
    if not fixtures:
        raise SystemExit("No image fixtures found in the synthetic manifest.")

    policies = load_policies()
    settings = load_inference_settings()
    service = InferenceService(policies=policies, settings=settings)

    checks: list[dict[str, object]] = []
    failures: list[str] = []
    geometry_deltas: list[float] = []

    for fixture in fixtures:
        fixture_id = str(fixture["fixture_id"])
        path = PROJECT_ROOT / str(fixture["path"])
        expected = str(fixture["expected_demonstration_class"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_ok = digest == str(fixture.get("sha256", ""))
        crop = str(fixture["crop"])
        policy = policies.policy(crop, demo=True)
        policy_ok = policy.metric == str(fixture["metric"])
        threshold_ok = (
            policy.threshold_mm is not None
            and abs(policy.threshold_mm - float(fixture["demo_threshold_mm"])) < 1e-9
        )

        decoded = load_image_file(
            path,
            max_pixels=settings.max_image_pixels,
            max_bytes=settings.max_upload_bytes,
        )
        result = service.inspect_frame(
            decoded.bgr,
            MediaContext(
                source_type=SourceType.IMAGE,
                source_id=fixture_id,
                device="synthetic-fixture",
            ),
            InferenceRequest(
                crop_key=crop,
                marker_side_mm=float(fixture["marker_side_mm"]),
                expected_marker_id=0,
                demo_policy=True,
            ),
        )
        actual_status = result.status.value
        actual_class = result.size_class.value if result.size_class else None
        expectation_met = actual_status == expected or actual_class == expected

        measurement = (
            result.instances[0].measurement
            if result.instances and result.instances[0].measurement
            else None
        )
        delta_mm = None
        truth = fixture.get("actual_metric_mm")
        if measurement is not None and truth is not None:
            delta_mm = abs(measurement.value_mm - float(truth))
            geometry_deltas.append(delta_mm)

        row = {
            "fixture_id": fixture_id,
            "synthetic": True,
            "path": relative_to_project(path),
            "sha256_match": hash_ok,
            "expected": expected,
            "actual_status": actual_status,
            "actual_size_class": actual_class,
            "expectation_met": expectation_met,
            "policy_metric_matches_fixture": policy_ok,
            "demo_threshold_matches_fixture": threshold_ok,
            "measurement_mm": round(measurement.value_mm, 3) if measurement else None,
            "vector_truth_mm": truth,
            "absolute_geometry_delta_mm": (
                round(delta_mm, 3) if delta_mm is not None else None
            ),
            "measurement_uncertainty_mm": (
                round(measurement.uncertainty_mm, 3) if measurement else None
            ),
            "reasons": result.reasons[:2],
        }
        checks.append(row)
        if not (hash_ok and expectation_met and policy_ok and threshold_ok):
            failures.append(fixture_id)

    passed = len(checks) - len(failures)
    report = {
        "report_type": "synthetic_contract_evaluation",
        "status": "PASS" if not failures else "FAIL",
        "scope": (
            "Synthetic fixtures only. Verifies status precedence, refusal logic, "
            "threshold wiring, and known planar geometry. Not real-produce accuracy."
        ),
        "evidence_class": "SYNTHETIC_CONTRACT_TEST_ONLY",
        "environment": environment_metadata(),
        "config": {
            "policy_version": policies.version,
            "policy_sha256": policies.config_sha256,
            "settings_version": settings.version,
            "settings_sha256": settings.config_sha256,
        },
        "summary": {
            "fixtures_checked": len(checks),
            "checks_passed": passed,
            "checks_failed": len(failures),
            "failed_fixtures": failures,
            "geometry_observations": len(geometry_deltas),
            "median_absolute_geometry_delta_mm": (
                round(float(statistics.median(geometry_deltas)), 3)
                if geometry_deltas
                else None
            ),
            "max_absolute_geometry_delta_mm": (
                round(max(geometry_deltas), 3) if geometry_deltas else None
            ),
            "note": (
                "Geometry deltas compare synthetic vector truth with the measured "
                "silhouette under demo thresholds; they do not estimate crop sizing "
                "error on real produce."
            ),
        },
        "fixtures": checks,
        "limitations": [
            "No real produce, real cameras, real backgrounds, or measured specimens.",
            "Demo thresholds are not ARL/Growize approved.",
            "Synthetic contract pass rate must never be quoted as accuracy.",
        ],
    }
    output = write_json(args.output, report)
    print(f"Synthetic contract report written to {output}")
    print(f"{passed}/{len(checks)} checks passed; status={report['status']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
