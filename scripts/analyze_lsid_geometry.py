"""Validation-only comparison of planar homography and marker-area scale.

This is a diagnostic on human masks, not an automatic model accuracy result.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from growize_vision.config import CalibrationSettings
from growize_vision.markers import detect_best_marker
from scripts.benchmark_lsid_size import RUN, decode, load_records  # noqa: E402


def main():
    predictions = {
        row["leaf_id"]: row
        for row in [
            json.loads(line)
            for line in (RUN / "pixel_rf_v1/validation_predictions.jsonl").read_text().splitlines()
        ]
    }
    results = []
    for row in load_records():
        if row["split"] != "validation":
            continue
        image, mask = decode(row)
        marker, _ = detect_best_marker(image, CalibrationSettings(dictionary="LSID_BEANS_3X3"))
        measured = predictions[row["leaf_id"]]["oracle_mask_diagnostic"]["measurement"]
        if marker is None or measured is None:
            continue
        contours, _ = cv2.findContours(
            (mask == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        pixel_area = cv2.contourArea(max(contours, key=cv2.contourArea))
        scalar_area = pixel_area * row["marker_side_mm"] ** 2 / marker.area_px / 100
        projective_area = measured["value"]
        results.append(
            {
                "leaf_id": row["leaf_id"],
                "truth_cm2": row["area_cm2"],
                "scalar_cm2": scalar_area,
                "projective_cm2": projective_area,
                "projective_scalar_ratio": projective_area / scalar_area,
                "oracle_signed_error_cm2": projective_area - row["area_cm2"],
                "automatic_mask_iou": predictions[row["leaf_id"]]["mask_iou"],
            }
        )
    summary = {
        "scope": "Validation-only oracle-mask geometry diagnostic, not model accuracy",
        "captures": len(results),
        "homography_mae_cm2": float(
            np.mean([abs(row["projective_cm2"] - row["truth_cm2"]) for row in results])
        ),
        "scalar_mae_cm2": float(
            np.mean([abs(row["scalar_cm2"] - row["truth_cm2"]) for row in results])
        ),
        "projective_scaling_slices": {
            str(limit): {
                "captures": sum(
                    abs(row["projective_scalar_ratio"] - 1) <= limit for row in results
                ),
                "homography_mae_cm2": float(
                    np.mean(
                        [
                            abs(row["projective_cm2"] - row["truth_cm2"])
                            for row in results
                            if abs(row["projective_scalar_ratio"] - 1) <= limit
                        ]
                    )
                ),
            }
            for limit in [0.05, 0.10, 0.15, 0.20, 0.30]
        },
        "results": results,
    }
    (RUN / "validation_geometry_diagnostic.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
