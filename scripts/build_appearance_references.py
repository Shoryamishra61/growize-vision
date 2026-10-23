"""Compute explicitly uncalibrated reference silhouettes from frozen training rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from growize_vision.crop_identity import CropIdentityService  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports/species_training_v2")
    args = parser.parse_args()
    manifest = args.run_dir / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    by_class = defaultdict(list)
    seen_groups = set()
    for row in rows:
        if row["split"] != "train" or row["label"] == "Unknown" or row["group"] in seen_groups:
            continue
        seen_groups.add(row["group"])
        image = cv2.imread(str(ROOT / row["path"]))
        contour = CropIdentityService.appearance_contour(image)
        if contour is not None:
            by_class[row["label"]].append((cv2.contourArea(contour), max(cv2.minAreaRect(contour)[1])))
    references = {}
    for label, measurements in by_class.items():
        if len(measurements) < 10:
            continue
        array = np.array(measurements)
        references[label] = {"count": len(measurements), "area_median": float(np.median(array[:, 0])),
                             "axis_median": float(np.median(array[:, 1])), "reference_size_px": 224,
                             "method": "controlled_background_v2", "physical_truth": False,
                             "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    target = args.run_dir / "candidate/vegetable_population_statistics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(references, indent=2), encoding="utf-8")
    print({label: row["count"] for label, row in references.items()}, flush=True)


if __name__ == "__main__":
    main()
