"""CSV/JSON/PNG export with an explicit audit trail."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2

from .domain import InspectionResult
from .media import safe_export_stem


def _row_for_instance(result: InspectionResult, instance: Any) -> dict[str, Any]:
    decision = instance.decision
    measurement = instance.measurement or (decision.measurement if decision else None)
    policy = decision.policy if decision and decision.policy else result.policy
    return {
        "result_id": result.result_id,
        "timestamp_utc": result.timestamp,
        "source_type": result.context.source_type.value,
        "source_id": result.context.source_id,
        "source_sha256": result.context.source_sha256,
        "processed_frame_sha256": result.processed_frame_sha256,
        "frame_id": result.context.frame_id,
        "timestamp_s": result.context.timestamp_s,
        "batch_id": result.context.batch_id,
        "instance_id": instance.instance_id,
        "tracking_id": instance.tracking_id,
        "crop_key": instance.crop_key,
        "status": decision.status.value if decision else "NO_DECISION",
        "size_class": decision.size_class.value
        if decision and decision.size_class
        else None,
        "metric": measurement.metric if measurement else None,
        "value_mm": round(measurement.value_mm, 3) if measurement else None,
        "uncertainty_mm": round(measurement.uncertainty_mm, 3)
        if measurement
        else None,
        "lower_mm": round(measurement.lower_mm, 3) if measurement else None,
        "upper_mm": round(measurement.upper_mm, 3) if measurement else None,
        "unit": measurement.unit if measurement else None,
        "threshold_mm": decision.threshold_mm if decision else None,
        "comparator": decision.comparator if decision else None,
        "policy_version": decision.policy_version if decision else None,
        "approval_status": policy.approval_status if policy else None,
        "policy_is_demo": bool(policy.demo) if policy else None,
        "calibration_id": result.calibration.calibration_id
        if result.calibration
        else None,
        "calibration_valid": bool(result.calibration and result.calibration.valid),
        "model_sha": result.model_sha or "",
        "pipeline_sha256": result.pipeline_sha256,
        "pipeline_version": result.pipeline_version,
        "software_version": result.software_version,
        "visible_fraction": round(instance.visible_fraction, 4),
        "bbox_px": "|".join(str(v) for v in instance.bbox_px),
        "reasons": " | ".join(decision.reasons if decision else []),
        "quality_flags": " | ".join(instance.quality_flags),
        "processing_ms": round(result.processing_ms, 3),
    }


def export_inspection(
    result: InspectionResult,
    annotated_bgr: Any | None,
    *,
    output_root: Path | str = "outputs",
    stem: str = "inspection",
) -> dict[str, str]:
    """Persist the annotated image, result JSON and one CSV row per object."""
    root = Path(output_root)
    session = safe_export_stem(result.context.session_id or "local", "local")
    folder = root / session / safe_export_stem(result.result_id, "inspection")
    folder.mkdir(parents=True, exist_ok=True)
    stem = safe_export_stem(stem, "inspection")

    paths: dict[str, str] = {}
    if annotated_bgr is not None:
        image_path = folder / f"{stem}_annotated.png"
        if not cv2.imwrite(str(image_path), annotated_bgr):
            raise OSError(f"Could not write annotated image: {image_path}")
        paths["annotated_image"] = str(image_path)

    json_path = folder / f"{stem}_result.json"
    csv_path = folder / f"{stem}_objects.csv"
    # Record all planned output paths before serialising so the JSON's own
    # provenance is complete even though the files are written immediately
    # afterwards.
    paths["json"] = str(json_path)
    paths["csv"] = str(csv_path)
    result.exported_paths.update(paths)
    payload = result.to_dict()
    payload["export"] = {
        "annotated_image": paths.get("annotated_image"),
        "json": str(json_path),
        "csv": str(csv_path),
        "notice": (
            "Demo policy labels are engineering demonstrations, not ARL-approved "
            "grades." if result.policy and result.policy.demo else
            "No ARL-approved threshold is attached unless policy.approval_status "
            "is APPROVED_SOURCE."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = [_row_for_instance(result, instance) for instance in result.instances]
    fieldnames = list(rows[0].keys()) if rows else [
        "result_id", "timestamp_utc", "source_type", "status", "size_class"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    paths["csv"] = str(csv_path)
    result.exported_paths.update(paths)
    return paths


def export_video_report(
    *,
    report_payload: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    output_root: Path | str,
    stem: str = "video",
) -> dict[str, str]:
    root = Path(output_root)
    folder = root / "video" / safe_export_stem(stem, "video")
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / "video_result.json"
    csv_path = folder / "video_frames.csv"
    # Frame rows mix object rows (full audit fields) with no-object rows (a
    # shorter subset).  The CSV header must be the union of every row's keys,
    # otherwise DictWriter raises when the first sampled frame has no object.
    fieldnames: list[str] = []
    for row in frame_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = [
            "result_id", "frame_id", "timestamp_s", "status", "instance_id",
            "size_class", "metric", "value_mm", "uncertainty_mm", "threshold_mm",
        ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_rows)
    exports = {"json": str(json_path), "csv": str(csv_path)}
    report_payload["exports"] = dict(exports)
    json_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return exports


def flatten_result_rows(results: list[InspectionResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for instance in result.instances:
            rows.append(_row_for_instance(result, instance))
        if not result.instances:
            rows.append(
                {
                    "result_id": result.result_id,
                    "timestamp_utc": result.timestamp,
                    "source_type": result.context.source_type.value,
                    "source_id": result.context.source_id,
                    "frame_id": result.context.frame_id,
                    "timestamp_s": result.context.timestamp_s,
                    "status": result.status.value,
                    "size_class": None,
                    "reasons": " | ".join(result.reasons),
                    "processing_ms": round(result.processing_ms, 3),
                }
            )
    return rows


def write_batch_csv(results: list[InspectionResult], path: Path | str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = flatten_result_rows(results)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["result_id", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)
