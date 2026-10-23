"""Real LSID-Beans size experiment; automatic masks and oracle diagnostics stay separate."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from growize_vision.config import load_inference_settings, load_policies  # noqa: E402
from growize_vision.domain import GradePolicy, SizeClass, Status  # noqa: E402
from growize_vision.geometry import MetrologyError, measure_contour  # noqa: E402
from growize_vision.grading import evaluate_measurement  # noqa: E402
from growize_vision.inference import InferenceRequest, InferenceService  # noqa: E402
from growize_vision.markers import (  # noqa: E402
    calibrate_from_marker,
    detect_best_marker,
    marker_mask,
)
from growize_vision.quality import assess_quality  # noqa: E402

SOURCE = ROOT / "data/external/lsid_beans_f42hwwrpgn_v3"
RUN = ROOT / "reports/real_size/lsid_beans"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_sources():
    paths = sorted((ROOT / "src/growize_vision").glob("*.py"))
    paths += [ROOT / "config/inference.yaml", ROOT / "config/grade_rules.yaml"]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def load_records():
    contract = json.loads((RUN / "split_contract.json").read_text())
    acquired = [
        json.loads(line)
        for line in (SOURCE / "acquisition_manifest.jsonl").read_text().splitlines()
    ]
    if len(acquired) != 612:
        raise ValueError("Full 612-leaf acquisition is required before fitting/evaluation")
    if {row["leaf_id"] for row in acquired} != set(range(1, 613)):
        raise ValueError("Acquisition must contain each physical leaf exactly once")
    records = []
    plant_splits = {}
    for row in acquired:
        files = {key: SOURCE / value for key, value in row["files"].items()}
        for key, path in files.items():
            if digest(path) != row["publisher_sha256"][key]:
                raise ValueError(f"Source changed: {path.name}")
        xml = ET.parse(files["annotation"])
        leaf_id = int(xml.findtext(".//leaf-number"))
        if leaf_id != row["leaf_id"]:
            raise ValueError("Annotation and capture specimen identity differ")
        area = float(xml.findtext(".//leaf-area"))
        marker_side_mm = float(xml.findtext(".//marker-real-area")) ** 0.5 * 10
        if (
            not np.isfinite(area)
            or area <= 0
            or not np.isfinite(marker_side_mm)
            or marker_side_mm <= 0
        ):
            raise ValueError("Invalid independently measured dimensions")
        split = contract["leaf_assignment"][str(leaf_id)]
        plant = row["plant_id"]
        if plant != (leaf_id - 1) // 2 + 1:
            raise ValueError("Plant identity differs from the published leaf-pair protocol")
        if plant in plant_splits and plant_splits[plant] != split:
            raise ValueError("Related leaves cross experiment splits")
        plant_splits[plant] = split
        records.append(
            dict(row, files=files, split=split, area_cm2=area, marker_side_mm=marker_side_mm)
        )
    return records


def decode(record):
    image = cv2.imread(str(record["files"]["image"]))
    mask = np.fromfile(record["files"]["segmentation"], dtype=np.uint8).reshape(512, 512)
    if (
        image is None
        or image.shape != (512, 512, 3)
        or not set(np.unique(mask)).issubset({0, 1, 2})
    ):
        raise ValueError("Source capture or human mask is malformed")
    return image, mask


def pixel_features(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32) / np.array(
        [180, 255, 255], dtype=np.float32
    )
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    smooth = [cv2.GaussianBlur(rgb, (0, 0), sigma) for sigma in (1, 3)]
    variances = [
        np.sqrt(
            np.maximum(cv2.blur(gray * gray, (size, size)) - cv2.blur(gray, (size, size)) ** 2, 0)
        )[:, :, None]
        for size in (3, 11)
    ]
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))[
        :, :, None
    ]
    return np.concatenate([rgb, hsv, lab, *smooth, *variances, gradient], axis=2).reshape(-1, 18)


def fit(records, output, plan_path):
    if output.exists():
        raise ValueError("Checkpoint exists; use it without --fit or choose a new run directory")
    plan = json.loads(plan_path.read_text())
    params = plan["candidate_parameters"]
    features, targets = [], []
    training = [row for row in records if row["split"] == "train"]
    rng = np.random.default_rng(params["random_state"])
    for index, row in enumerate(training):
        image, annotation = decode(row)
        labels = (annotation.ravel() == 1).astype(np.uint8)
        sampled = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(labels == value),
                    min(
                        params["training_pixels_per_capture"] // 2,
                        np.count_nonzero(labels == value),
                    ),
                    replace=False,
                )
                for value in (0, 1)
            ]
        )
        features.append(pixel_features(image)[sampled])
        targets.append(labels[sampled])
        if (index + 1) % 50 == 0:
            print(f"Prepared original training captures {index + 1}/{len(training)}", flush=True)
    model = RandomForestClassifier(
        n_estimators=params["trees"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=params["random_state"],
        n_jobs=params["workers"],
        class_weight="balanced",
    )
    started = time.perf_counter()
    model.fit(np.concatenate(features), np.concatenate(targets))
    artifact = {
        "model": model,
        "reference_mean_cm2": float(np.mean([row["area_cm2"] for row in training])),
        "train_leaf_ids": [row["leaf_id"] for row in training],
        "source_script_sha256": digest(__file__),
        "acquisition_sha256": digest(SOURCE / "acquisition_manifest.jsonl"),
        "split_sha256": digest(RUN / "split_contract.json"),
        "plan_sha256": digest(plan_path),
        "training_seconds": time.perf_counter() - started,
        "parameters": params,
        "image_policy": "Unchanged published original camera captures; no generated augmentation.",
        "runtime_sources": runtime_sources(),
        "measurement_method": plan.get("measurement_method", "planar_homography"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(artifact, handle)
    (output.parent / "train_source.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps({key: artifact[key] for key in ["reference_mean_cm2", "training_seconds"]}),
        flush=True,
    )
    return artifact


def predicted_leaf_mask(image, model, exclusion):
    probability = model.predict_proba(pixel_features(image))[:, 1].reshape(image.shape[:2])
    binary = (probability >= 0.5).astype(np.uint8)
    binary[exclusion > 0] = 0
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape
    mask = np.zeros(binary.shape, dtype=np.uint8)
    if not contours:
        return mask
    dominant = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(dominant)
    # Discarding a larger ambiguous leaf must not promote a small unrelated
    # weed or background fragment to the inspected object.
    if (
        cv2.contourArea(dominant) < 220
        or x <= 1
        or y <= 1
        or x + w >= width - 1
        or y + h >= height - 1
    ):
        return mask
    cv2.drawContours(mask, [dominant], -1, 1, thickness=-1)
    return mask


def physical_grade(image, mask, marker, calibration, policy, settings, method="planar_homography"):
    if marker is None or calibration is None or not calibration.valid:
        return {"size": None, "status": "CALIBRATION_REQUIRED", "measurement": None}
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return {"size": None, "status": "NO_COMPLETE_LEAF", "measurement": None}
    exclusion = marker_mask(image.shape, marker, settings.segmentation.marker_exclusion_margin_px)
    if np.any((mask > 0) & (exclusion > 0)):
        return {"size": None, "status": "OCCLUDED_MARKER", "measurement": None}
    quality = assess_quality(image, settings.quality, marker_mask=exclusion, object_mask=mask)
    if not quality.passed:
        return {"size": None, "status": "LOW_IMAGE_QUALITY", "measurement": None}
    if method == "marker_area_scalar":
        scale = calibration.marker_side_mm / np.sqrt(marker.area_px)
        origin_x, origin_y = marker.corners_px[0]
        calibration = replace(
            calibration,
            homography=[[scale, 0, -scale * origin_x], [0, scale, -scale * origin_y], [0, 0, 1]],
            mm_per_px=scale,
            corner_reprojection_rms_px=None,
            source="marker_area_weak_perspective_estimate",
            calibration_id=calibration.calibration_id + "_scalar",
        )
    elif method != "planar_homography":
        raise ValueError("Unknown physical measurement experiment")
    try:
        measurement, _ = measure_contour(
            max(contours, key=cv2.contourArea), calibration, policy.metric, settings.calibration
        )
    except MetrologyError as exc:
        return {"size": None, "status": "METROLOGY_FAILED", "reason": str(exc), "measurement": None}
    if method == "marker_area_scalar":
        measurement.method = "marker_area_weak_perspective_estimate"
        measurement.warnings.append(
            "Weak-perspective estimate; differing marker and leaf depths/tilts remain unobserved."
        )
    decision = evaluate_measurement(measurement, policy, calibration.calibration_id)
    return {
        "size": decision.size_class.value if decision.accepted else None,
        "status": decision.status.value,
        "measurement": measurement.to_dict(),
    }


def aggregate(results, field):
    accepted = [row for row in results if row[field]["size"] is not None]
    measurements = [row for row in results if row[field]["measurement"] is not None]
    correct = sum(row[field]["size"] == row["expected"] for row in accepted)
    return {
        "captures": len(results),
        "accepted": len(accepted),
        "correct": correct,
        "accepted_coverage": len(accepted) / len(results),
        "accepted_accuracy": correct / len(accepted) if accepted else None,
        "correct_fraction_all_captures": correct / len(results),
        "measured_captures": len(measurements),
        "measurement_mae_cm2": float(
            np.mean(
                [abs(row[field]["measurement"]["value"] - row["truth_cm2"]) for row in measurements]
            )
        )
        if measurements
        else None,
        "engineering_bound_truth_coverage": float(
            np.mean(
                [
                    row[field]["measurement"]["lower_mm"]
                    <= row["truth_cm2"]
                    <= row[field]["measurement"]["upper_mm"]
                    for row in measurements
                ]
            )
        )
        if measurements
        else None,
        "confusion": dict(
            Counter(f"{row['expected']}->{row[field]['size'] or 'REFUSED'}" for row in results)
        ),
        "statuses": dict(Counter(row[field]["status"] for row in results)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN / "pixel_rf_v1")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--plan-file", type=Path, default=RUN / "experiment_plan.json")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args()
    records = load_records()
    checkpoint = args.run_dir / "candidate.pkl"
    if args.fit:
        if args.split != "validation":
            raise ValueError("Fit and inspect validation before any test evaluation")
        artifact = fit(records, checkpoint, args.plan_file)
    else:
        with checkpoint.open("rb") as handle:
            artifact = pickle.load(handle)
        if artifact["runtime_sources"] != runtime_sources():
            raise ValueError("Runtime sources/configuration changed since fitting")
        for expected, path in [
            ("source_script_sha256", __file__),
            ("acquisition_sha256", SOURCE / "acquisition_manifest.jsonl"),
            ("split_sha256", RUN / "split_contract.json"),
            ("plan_sha256", args.plan_file),
        ]:
            if artifact[expected] != digest(path):
                raise ValueError(f"Checkpoint contract changed: {expected}")
    settings = load_inference_settings()
    settings = replace(
        settings, calibration=replace(settings.calibration, dictionary="LSID_BEANS_3X3")
    )
    policy = GradePolicy(
        crop_key="bean_leaf",
        crop_name="Bean leaf",
        physical_unit="single_leaf",
        metric="projected_area_cm2",
        metric_description="Projected area of one approximately coplanar bean leaf.",
        threshold_mm=artifact["reference_mean_cm2"],
        guard_band_mm=0.5,
        approval_status="EMPIRICAL_REFERENCE",
        source="Training leaf mean from LSID-Beans cv_1/split_1; held-out test leaves never fit the mean.",
    )
    policies = load_policies()
    policies.crops = {
        "bean_leaf": dict(
            display_name="Bean leaf",
            physical_unit="single_leaf",
            metric=policy.metric,
            threshold_mm=policy.threshold_mm,
            comparator="greater_than_or_equal",
            min_visible_fraction=0.98,
            approval_status="EMPIRICAL_REFERENCE",
            source=policy.source,
        )
    }
    policies.config_sha256 = hashlib.sha256(
        json.dumps(policies.crops, sort_keys=True).encode()
    ).hexdigest()
    baseline = InferenceService(settings=settings, policies=policies)
    results = []
    selected = [row for row in records if row["split"] == args.split]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / f"{args.split}_predictions.jsonl"
    if output.exists():
        raise ValueError("Evaluation output already exists; preserve it instead of overwriting")
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected):
            image, truth_mask = decode(row)
            started = time.perf_counter()
            marker, _ = detect_best_marker(image, settings.calibration, expected_id=0)
            calibration = (
                calibrate_from_marker(
                    image.shape, marker, row["marker_side_mm"], settings.calibration
                )
                if marker
                else None
            )
            exclusion = marker_mask(
                image.shape, marker, settings.segmentation.marker_exclusion_margin_px
            )
            predicted = predicted_leaf_mask(image, artifact["model"], exclusion)
            intersection = np.count_nonzero((predicted > 0) & (truth_mask == 1))
            union = np.count_nonzero((predicted > 0) | (truth_mask == 1))
            raw_baseline = baseline.inspect_frame(
                image,
                request=InferenceRequest(
                    crop_key="bean_leaf", marker_side_mm=row["marker_side_mm"]
                ),
            )
            record = {
                "leaf_id": row["leaf_id"],
                "plant_id": row["plant_id"],
                "capture_id": row["capture_id"],
                "image_sha256": row["publisher_sha256"]["image"],
                "truth_cm2": row["area_cm2"],
                "expected": SizeClass.BIG.value
                if row["area_cm2"] >= policy.threshold_mm
                else SizeClass.SMALL.value,
                "mask_iou": intersection / union if union else 0,
                "calibration_valid": bool(calibration and calibration.valid),
                "automatic": physical_grade(
                    image,
                    predicted,
                    marker,
                    calibration,
                    policy,
                    settings,
                    artifact.get("measurement_method", "planar_homography"),
                ),
                "oracle_mask_diagnostic": physical_grade(
                    image,
                    (truth_mask == 1).astype(np.uint8),
                    marker,
                    calibration,
                    policy,
                    settings,
                    artifact.get("measurement_method", "planar_homography"),
                ),
                "existing_controlled_mat_status": raw_baseline.status.value,
                "existing_controlled_mat_size": raw_baseline.size_class.value
                if raw_baseline.status == Status.ACCEPTED and raw_baseline.size_class
                else None,
                "processing_ms": (time.perf_counter() - started) * 1000,
            }
            results.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if (index + 1) % 10 == 0:
                print(
                    f"Evaluated original {args.split} captures {index + 1}/{len(selected)}",
                    flush=True,
                )
    summary = {
        "scope": "Single-source bean-leaf area sizing; recorded crop identity, one capture per leaf; not universal produce grading.",
        "split": args.split,
        "checkpoint_sha256": digest(checkpoint),
        "script_sha256": digest(__file__),
        "acquisition_sha256": artifact["acquisition_sha256"],
        "split_sha256": artifact["split_sha256"],
        "reference_mean_cm2": policy.threshold_mm,
        "measurement_method": artifact.get("measurement_method", "planar_homography"),
        "training_leaves": len(artifact["train_leaf_ids"]),
        "automatic": aggregate(results, "automatic"),
        "oracle_mask_diagnostic_not_automatic": aggregate(results, "oracle_mask_diagnostic"),
        "mask_iou_mean": float(np.mean([row["mask_iou"] for row in results])),
        "existing_controlled_mat_statuses": dict(
            Counter(row["existing_controlled_mat_status"] for row in results)
        ),
        "median_processing_ms": float(np.median([row["processing_ms"] for row in results])),
        "deployment_decision": "RESEARCH_ONLY_NOT_PROMOTED",
        "limitations": [
            "No automatic species identification or cross-crop validation.",
            "Source frames were cropped/resized by their authors, not digitally augmented here.",
            "Engineering uncertainty bounds are not statistical coverage guarantees.",
            "One development pilot (leaf 001) was inspected; exclude its plant for a clean test summary.",
        ],
    }
    if args.split == "test":
        clean = [row for row in results if row["plant_id"] != 1]
        summary["clean_test_excluding_development_pilot_plant"] = aggregate(clean, "automatic")
    (args.run_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
