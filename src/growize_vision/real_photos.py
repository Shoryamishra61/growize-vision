"""Evidence checks for original real photographs; declarations require human review."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path


def verify_original_photo(row: dict, root: Path) -> Path:
    if (
        row.get("is_original_capture") is not True
        or row.get("synthetic") is not False
        or row.get("augmented") is not False
    ):
        raise ValueError(
            "Only explicitly reviewed original photographs are eligible; generated or augmented images are excluded."
        )
    for name in ("specimen_id", "capture_id", "reviewed_by", "source", "license"):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError(f"Missing real-photo provenance: {name}")
    path = (root / row["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Photo must be inside the dataset root")
    if not path.is_file():
        raise ValueError(f"Original photo missing: {path.name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row.get("sha256"):
        raise ValueError("Original photo SHA256 mismatch")
    return path


def verify_specimen_splits(rows: list[dict]) -> None:
    seen = {}
    for row in rows:
        for name in ("specimen_id", "capture_id", "sha256"):
            value = row.get(name)
            if not value:
                raise ValueError(f"Missing {name}")
            key = (name, value)
            if key in seen and seen[key] != row["split"]:
                raise ValueError(f"{name} appears in multiple splits")
            seen[key] = row["split"]


def size_distribution(values):
    """Describe measured specimens; class medians are anchors, not scale estimates."""
    values = list(values)
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in values
    ):
        raise ValueError("Size distribution requires finite positive measurements")
    median = statistics.median(values)
    small = [value for value in values if value < median]
    big = [value for value in values if value >= median]
    return {
        "mean": statistics.mean(values),
        "median": median,
        "small_group_median": statistics.median(small) if small else None,
        "big_group_median": statistics.median(big),
        "small_specimens": len(small),
        "big_specimens": len(big),
        "minimum": min(values),
        "maximum": max(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
        "boundary_rule": "SMALL < median; BIG >= median; ties belong to BIG",
    }


def reference_statistics(rows):
    specimens = {}
    for row in rows:
        if row["split"] != "train":
            continue
        value = row.get("measured_value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Reference measurements must be finite and positive")
        if not row.get("measurement_reviewed_by"):
            raise ValueError("Independent measurement reviewer is required")
        unit = "cm2" if row["metric"] == "projected_area_cm2" else "mm"
        if row.get("measurement_unit") != unit:
            raise ValueError("Reference measurement unit does not match its metric")
        identity = (row["crop_key"], row["physical_unit"], row["metric"], unit)
        observation = (identity, float(value))
        specimen = row["specimen_id"]
        if specimen in specimens and specimens[specimen] != observation:
            raise ValueError("Repeated specimen has inconsistent physical ground truth")
        specimens[specimen] = observation
    groups = defaultdict(list)
    for identity, value in specimens.values():
        groups[identity].append(value)
    return [
        {
            "crop_key": key[0],
            "physical_unit": key[1],
            "metric": key[2],
            "unit": key[3],
            "specimens": len(values),
            **size_distribution(values),
            "status": "REFERENCE_NOT_VALIDATED",
        }
        for key, values in sorted(groups.items())
    ]
