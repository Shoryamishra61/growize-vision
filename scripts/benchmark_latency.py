"""Measure CPU latency of the deterministic OpenCV baseline.

This report answers one narrow question: how long does one already-decoded
frame take on this machine?  It is not an accuracy, throughput-under-load,
or field-readiness benchmark.  Synthetic demo fixtures are used only because
no licensed measured dataset is available.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import statistics
import time
from pathlib import Path

from growize_vision.config import load_inference_settings, load_policies
from growize_vision.domain import MediaContext, SourceType
from growize_vision.evidence import (
    PROJECT_ROOT,
    environment_metadata,
    latency_summary,
    load_json,
    relative_to_project,
    write_json,
)
from growize_vision.inference import InferenceRequest, InferenceService
from growize_vision.media import load_image_file

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "demo" / "demo_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "latency_report.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs", type=int, default=10, help="timed repetitions per image")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=None,
        help="Override configured processing dimension for the benchmark only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1 or args.warmup < 0:
        raise SystemExit("--runs must be >= 1 and --warmup must be >= 0.")

    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    fixtures = [
        fixture
        for fixture in manifest.get("fixtures", [])
        if Path(str(fixture.get("path", ""))).suffix.lower() in IMAGE_SUFFIXES
    ]
    if not fixtures:
        raise SystemExit("No synthetic image fixtures found in the manifest.")

    policies = load_policies()
    settings = load_inference_settings()
    if args.max_dimension is not None:
        settings = dataclasses.replace(
            settings, max_processing_dimension=int(args.max_dimension)
        )
    service = InferenceService(policies=policies, settings=settings)

    rows = []
    all_wall_ms: list[float] = []
    all_internal_ms: list[float] = []
    for fixture in fixtures:
        path = PROJECT_ROOT / str(fixture["path"])
        decoded = load_image_file(
            path,
            max_pixels=settings.max_image_pixels,
            max_bytes=settings.max_upload_bytes,
        )
        request = InferenceRequest(
            crop_key=str(fixture["crop"]),
            marker_side_mm=float(fixture["marker_side_mm"]),
            demo_policy=True,
        )
        context = MediaContext(
            source_type=SourceType.IMAGE,
            source_id=str(fixture["fixture_id"]),
            device="benchmark-fixture",
        )
        for _ in range(args.warmup):
            service.inspect_frame(decoded.bgr, context, request)
        wall_samples: list[float] = []
        internal_samples: list[float] = []
        status = ""
        for _ in range(args.runs):
            started = time.perf_counter()
            result = service.inspect_frame(decoded.bgr, context, request)
            wall_samples.append((time.perf_counter() - started) * 1000.0)
            internal_samples.append(float(result.processing_ms))
            status = result.status.value
        all_wall_ms.extend(wall_samples)
        all_internal_ms.extend(internal_samples)
        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "shape_hwc": list(decoded.bgr.shape),
                "status": status,
                "wall": latency_summary(wall_samples),
                "internal": latency_summary(internal_samples),
            }
        )

    wall = latency_summary(all_wall_ms)
    internal = latency_summary(all_internal_ms)
    report = {
        "report_type": "cpu_latency_benchmark",
        "status": "MEASURED",
        "scope": (
            "Single-frame, already-decoded, synthetic-fixture CPU latency on one "
            "workstation. Not accuracy, camera throughput, concurrent-load, or "
            "field performance."
        ),
        "evidence_class": "ENGINEERING_MEASUREMENT_ONLY",
        "environment": {
            **environment_metadata(),
            "cpu_count": os.cpu_count(),
            "settings_version": settings.version,
            "settings_sha256": settings.config_sha256,
            "policy_version": policies.version,
            "policy_sha256": policies.config_sha256,
        },
        "protocol": {
            "manifest": relative_to_project(manifest_path),
            "fixtures": len(rows),
            "warmup_runs_per_fixture": args.warmup,
            "timed_runs_per_fixture": args.runs,
            "timed_observations": args.runs * len(rows),
            "demo_policy": True,
            "max_processing_dimension": settings.max_processing_dimension,
            "decode_included": False,
            "annotation_render_included": True,
        },
        "aggregate": {
            "wall": wall,
            "internal": internal,
            "median_of_fixture_medians_ms": round(
                statistics.median([row["wall"]["median_ms"] for row in rows]), 3
            ),
        },
        "fixtures": rows,
        "limitations": [
            "Synthetic fixtures are not real produce.",
            "One process, one machine, one image at a time; no queue or camera I/O.",
            "Image decode is excluded; webcam capture overhead is excluded.",
            "Results are not a service-level guarantee on other hardware.",
        ],
    }
    output = write_json(args.output, report)
    print(f"Measured latency written to {output}")
    print(
        "wall median={median_ms:.1f} ms p95={p95_ms:.1f} ms over {count} observations".format(
            **wall
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
