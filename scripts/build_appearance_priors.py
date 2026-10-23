"""Build real-photo appearance size priors for uncalibrated grading.

Reads the species manifest, keeps only original (non-augmented) captures,
normalizes each image to the fixed 224-pixel reference frame, extracts the
item silhouette, and stores per-species median footprint/axis references in
config/vegetable_population_statistics.json.

These priors power the referenceless SINGLE-item SMALL/BIG estimate: an item
whose 224-frame footprint is far above the species median is BIG, far below
is SMALL, borderline stays review.  They are appearance comparisons, never
millimetres.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from growize_vision.crop_identity import CropIdentityService  # noqa: E402

MANIFEST = ROOT / "reports" / "species_training_v5" / "manifest.jsonl"
STATS_OUT = ROOT / "config" / "vegetable_population_statistics.json"
SAMPLE_PER_LABEL = 150
MIN_SUCCESS = 40


def is_augmented(path: str) -> bool:
    name = Path(path).name.lower()
    return "aug" in name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--sample", type=int, default=SAMPLE_PER_LABEL)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if is_augmented(row["path"]) or row["label"] == "Unknown":
            continue
        by_label[row["label"]].append(row)

    service = CropIdentityService()
    stats: dict[str, dict] = {}
    rng = random.Random(20261024)
    for label in sorted(by_label):
        rows_for_label = sorted(by_label[label], key=lambda r: r["path"])
        sample = (
            rows_for_label
            if len(rows_for_label) <= args.sample
            else rng.sample(rows_for_label, args.sample)
        )
        areas, axes = [], []
        for row in sample:
            path = ROOT / row["path"]
            image = cv2.imread(str(path))
            if image is None:
                continue
            small = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            contour = service.appearance_contour(small)
            if contour is None or len(contour) < 3:
                continue
            areas.append(float(cv2.contourArea(contour)))
            axes.append(float(max(cv2.minAreaRect(contour.reshape(-1, 1, 2))[1])))
        if len(areas) < MIN_SUCCESS:
            print(f"  {label}: only {len(areas)} usable captures; skipped")
            continue
        stats[label] = {
            "reference_size_px": 224,
            "method": "real_dataset_v2",
            "area_median": round(statistics.median(areas), 1),
            "axis_median": round(statistics.median(axes), 1),
            "samples": len(areas),
            "sources": sorted({row["source"] for row in sample}),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Median 224-px-frame footprint of original (non-augmented) real "
                "captures. Powers the uncalibrated appearance estimate only; "
                "never a millimetre claim."
            ),
        }
        print(
            f"  {label}: n={len(areas)} area_median={stats[label]['area_median']} "
            f"axis_median={stats[label]['axis_median']}"
        )

    STATS_OUT.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(stats)} species priors to {STATS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
