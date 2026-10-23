"""Measured-specimen manifest validation and leakage-safe grouped splitting.

This module deliberately treats an empty dataset as a valid *empty* state, not
as evidence.  Synthetic fixtures are allowed in a manifest for pipeline tests,
but they are counted separately and can never satisfy a real evaluation gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "growize-measured-manifest-1.0"
ALLOWED_METRICS = {
    "min_area_rect_short_mm",
    "min_area_rect_long_mm",
    "max_feret_mm",
    "projected_area_cm2",
}
ALLOWED_UNITS = {
    "single_tuber",
    "single_bulb",
    "single_fruit",
    "single_pod",
    "single_leaf",
    "standardized_bunch",
}
ALLOWED_SPLITS = ("train", "val", "test")
ALLOWED_GRADES = {"SMALL", "BIG"}
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class DatasetError(ValueError):
    """Raised when a manifest cannot be used without inventing evidence."""


def load_manifest(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise DatasetError(f"Dataset manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Dataset manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetError("Dataset manifest must be a JSON object.")
    return payload


def _text(value: Any, name: str, errors: list[str], *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        errors.append(f"{name} must be a non-empty string.")
        return ""
    return value.strip()


def _finite_positive(value: Any, name: str, errors: list[str]) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be numeric.")
        return None
    if not math.isfinite(number) or number <= 0:
        errors.append(f"{name} must be finite and > 0.")
        return None
    return number


def _group_key(specimen: dict[str, Any]) -> str:
    lot = str(specimen.get("lot") or "").strip()
    session = str(specimen.get("capture_session") or "").strip()
    specimen_id = str(specimen.get("specimen_id") or "").strip()
    if lot:
        return f"lot:{lot}"
    if session:
        return f"session:{session}"
    return f"specimen:{specimen_id}"


def _safe_image_path(
    raw_path: Any,
    image_root: Path,
    specimen_id: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        errors.append(f"{specimen_id}: image path must be a non-empty string.")
        return None
    candidate = Path(raw_path)
    if (
        candidate.is_absolute()
        or candidate.drive
        or raw_path.startswith(("/", "\\"))
    ):
        errors.append(f"{specimen_id}: image path must be relative, got {raw_path!r}.")
        return None
    if any(part == ".." for part in candidate.parts):
        errors.append(f"{specimen_id}: image path may not contain '..'.")
        return None
    resolved = (image_root / candidate).resolve()
    try:
        resolved.relative_to(image_root.resolve())
    except ValueError:
        errors.append(f"{specimen_id}: image path escapes the image root.")
        return None
    return resolved


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        value = item.get(key)
        counts[str(value) if value not in (None, "") else "UNSPECIFIED"] += 1
    return dict(sorted(counts.items()))


def _summarise(specimens: list[dict[str, Any]]) -> dict[str, Any]:
    real = [s for s in specimens if not bool(s.get("synthetic"))]
    synthetic = [s for s in specimens if bool(s.get("synthetic"))]
    return {
        "specimens_total": len(specimens),
        "real_specimens": len(real),
        "synthetic_specimens": len(synthetic),
        "by_crop": _count_by(specimens, "crop"),
        "real_by_crop": _count_by(real, "crop"),
        "by_grade": _count_by(specimens, "reference_grade"),
        "real_by_grade": _count_by(real, "reference_grade"),
        "by_source": _count_by(specimens, "source"),
        "by_split": _count_by(specimens, "split"),
        "real_by_split": _count_by(real, "split"),
    }


def validate_manifest(
    manifest: dict[str, Any] | Path | str,
    *,
    image_root: Path | str | None = None,
    require_images: bool = True,
    allow_empty: bool = False,
    allowed_crops: set[str] | None = None,
) -> dict[str, Any]:
    """Validate provenance, physical truth, hashes and split leakage.

    The returned report is JSON-serialisable and is intended to be archived as
    evidence.  Errors make ``ok`` false; warnings do not.
    """
    manifest_path: Path | None = None
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        payload = load_manifest(manifest_path)
    else:
        payload = manifest
    if not isinstance(payload, dict):
        raise DatasetError("Manifest must be a JSON object.")

    root = Path(image_root) if image_root is not None else (
        manifest_path.parent if manifest_path is not None else Path.cwd()
    )
    errors: list[str] = []
    warnings: list[str] = []
    checked_images = 0
    verified_hashes = 0

    version = payload.get("version")
    if version != MANIFEST_VERSION:
        errors.append(
            f"Manifest version must be {MANIFEST_VERSION!r}, got {version!r}."
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("Manifest must contain a provenance object.")
        provenance = {}
    for field in ("dataset_name", "owner", "collection_protocol", "license", "source"):
        value = provenance.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"provenance.{field} must be supplied.")
    if provenance.get("contains_personal_data") not in (True, False):
        warnings.append(
            "provenance.contains_personal_data is not explicitly true/false."
        )

    specimens = payload.get("specimens")
    if not isinstance(specimens, list):
        errors.append("Manifest must contain a specimens array.")
        specimens = []
    if not specimens and not allow_empty:
        errors.append("Dataset is empty; no real measured specimens are available.")
    elif not specimens and allow_empty:
        warnings.append(
            "Dataset is intentionally empty. It is a template, not evidence, and "
            "cannot support accuracy, coverage or field-readiness claims."
        )

    seen_specimens: set[str] = set()
    seen_hashes: dict[str, str] = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    normalised: list[dict[str, Any]] = []

    for index, raw in enumerate(specimens):
        if not isinstance(raw, dict):
            errors.append(f"specimens[{index}] must be an object.")
            continue
        specimen = copy.deepcopy(raw)
        prefix = f"specimens[{index}]"
        specimen_id = _text(specimen.get("specimen_id"), f"{prefix}.specimen_id", errors)
        if specimen_id:
            prefix = specimen_id
            if specimen_id in seen_specimens:
                errors.append(f"{prefix}: duplicate specimen_id.")
            seen_specimens.add(specimen_id)

        crop = _text(specimen.get("crop"), f"{prefix}.crop", errors)
        if crop and allowed_crops is not None and crop not in allowed_crops:
            errors.append(
                f"{prefix}: crop {crop!r} has no policy; supported: "
                f"{', '.join(sorted(allowed_crops))}."
            )
        unit = _text(specimen.get("physical_unit"), f"{prefix}.physical_unit", errors)
        if unit and unit not in ALLOWED_UNITS:
            errors.append(f"{prefix}: unsupported physical_unit {unit!r}.")
        metric = _text(specimen.get("measured_metric"), f"{prefix}.measured_metric", errors)
        if metric and metric not in ALLOWED_METRICS:
            errors.append(f"{prefix}: unsupported measured_metric {metric!r}.")
        synthetic = specimen.get("synthetic")
        if not isinstance(synthetic, bool):
            errors.append(f"{prefix}: synthetic must be explicitly true or false.")
            synthetic = False
        _text(specimen.get("lot"), f"{prefix}.lot", errors)
        _text(
            specimen.get("capture_session"),
            f"{prefix}.capture_session",
            errors,
        )
        source = _text(specimen.get("source"), f"{prefix}.source", errors)
        license_name = _text(specimen.get("license"), f"{prefix}.license", errors)
        consent = _text(specimen.get("consent"), f"{prefix}.consent", errors)
        if not synthetic:
            for label, value in (
                ("source", source),
                ("license", license_name),
                ("consent", consent),
            ):
                if value.strip().lower() in {"unknown", "tbd", "n/a", "none"}:
                    errors.append(
                        f"{prefix}.{label} cannot be {value!r} for real specimens."
                    )

        measurement_1 = _finite_positive(
            specimen.get("measurement_1_mm"), f"{prefix}.measurement_1_mm", errors
        )
        measurement_2_raw = specimen.get("measurement_2_mm")
        measurement_2: float | None = None
        if measurement_2_raw not in (None, ""):
            measurement_2 = _finite_positive(
                measurement_2_raw, f"{prefix}.measurement_2_mm", errors
            )
            if (
                measurement_1 is not None
                and measurement_2 is not None
                and abs(measurement_1 - measurement_2) > 1.0
            ):
                warnings.append(
                    f"{prefix}: repeat measurements disagree by "
                    f"{abs(measurement_1 - measurement_2):.2f} mm (> 1.0 mm)."
                )
        reference_grade = specimen.get("reference_grade")
        if reference_grade not in ALLOWED_GRADES:
            warnings.append(
                f"{prefix}: reference_grade is missing or not SMALL/BIG; "
                "the specimen cannot be counted in a real confusion matrix."
            )

        split = specimen.get("split")
        if split is not None:
            if split not in ALLOWED_SPLITS:
                errors.append(f"{prefix}: split must be one of {ALLOWED_SPLITS}.")
            else:
                group_splits[_group_key(specimen)].add(str(split))
        if synthetic and split in ALLOWED_SPLITS:
            warnings.append(
                f"{prefix}: synthetic specimen is assigned to {split}; it must be "
                "excluded from real accuracy claims."
            )

        images = specimen.get("images")
        if not isinstance(images, list):
            errors.append(f"{prefix}.images must be an array.")
            images = []
        if not synthetic and not images:
            errors.append(f"{prefix}: real specimens require at least one image.")
        normalised_images: list[dict[str, str]] = []
        for image_index, image in enumerate(images):
            image_prefix = f"{prefix}.images[{image_index}]"
            if not isinstance(image, dict):
                errors.append(f"{image_prefix} must be an object.")
                continue
            raw_path = image.get("path")
            digest = str(image.get("sha256") or "").strip().lower()
            if not SHA256_RE.fullmatch(digest):
                errors.append(f"{image_prefix}.sha256 must be 64 lowercase hex chars.")
            resolved = _safe_image_path(raw_path, root, image_prefix, errors)
            if digest:
                previous = seen_hashes.get(digest)
                if previous is not None:
                    errors.append(
                        f"{image_prefix}: duplicate image hash {digest[:12]}... "
                        f"already used by {previous}; duplicate samples create leakage."
                    )
                else:
                    seen_hashes[digest] = specimen_id or prefix
            if resolved is not None and require_images:
                checked_images += 1
                if not resolved.exists() or not resolved.is_file():
                    errors.append(f"{image_prefix}: image file not found: {raw_path!r}.")
                elif digest:
                    actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
                    if actual != digest:
                        errors.append(
                            f"{image_prefix}: SHA-256 mismatch (manifest {digest[:12]}..., "
                            f"actual {actual[:12]}...)."
                        )
                    else:
                        verified_hashes += 1
            normalised_images.append(
                {
                    "path": str(raw_path or ""),
                    "sha256": digest,
                    **(  # preserve optional annotation metadata
                        {"role": str(image["role"])}
                        if isinstance(image.get("role"), str)
                        else {}
                    ),
                }
            )
        specimen["images"] = normalised_images
        normalised.append(specimen)

    for group, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            errors.append(
                f"Leakage: group {group!r} appears in multiple splits: "
                f"{', '.join(sorted(splits))}."
            )

    summary = _summarise(normalised)
    if summary["real_specimens"] == 0 and specimens:
        warnings.append("No real measured specimens are present.")
    if summary["real_specimens"] and not any(
        specimen.get("reference_grade") in ALLOWED_GRADES
        for specimen in normalised
        if not specimen.get("synthetic")
    ):
        warnings.append(
            "No real specimen has a reference_grade; a confusion matrix cannot be built."
        )

    return {
        "status": "PASS" if not errors else "FAIL",
        "ok": not errors,
        "manifest": str(manifest_path) if manifest_path else "<mapping>",
        "manifest_version": version,
        "errors": errors,
        "warnings": warnings,
        "counts": summary,
        "checked_images": checked_images,
        "verified_hashes": verified_hashes,
        "leakage_groups": {
            group: sorted(splits) for group, splits in sorted(group_splits.items())
        },
        "real_evaluable": bool(
            summary["real_specimens"] > 0
            and not errors
            and all(
                specimen.get("reference_grade") in ALLOWED_GRADES
                for specimen in normalised
                if not specimen.get("synthetic")
            )
        ),
    }


def _split_targets(total: int, ratios: dict[str, float]) -> dict[str, float]:
    return {split: total * ratio for split, ratio in ratios.items()}


def assign_grouped_splits(
    specimens: list[dict[str, Any]],
    *,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Assign every specimen to one split without dividing a lot/session.

    Assignment is deterministic for a given manifest and seed.  Greedy
    assignment fills whichever split is furthest below its target, so large
    lots are balanced instead of accidentally all landing in train.
    """
    ratios = {"train": float(train), "val": float(val), "test": float(test)}
    if any(not math.isfinite(value) or value <= 0 for value in ratios.values()):
        raise DatasetError("Split ratios must be finite and > 0.")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise DatasetError("Split ratios must sum to 1.0.")
    if not specimens:
        raise DatasetError("Cannot split an empty specimen list.")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, specimen in enumerate(specimens):
        groups[_group_key(specimen)].append(index)
    if len(groups) < len(ratios):
        raise DatasetError(
            f"Need at least {len(ratios)} independent groups for a grouped split; "
            f"found {len(groups)}."
        )

    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}|{group}".encode()).hexdigest(),
    )
    assignments: dict[str, str] = {}
    current = {split: 0 for split in ratios}
    targets = _split_targets(len(specimens), ratios)

    # Seed each split with one group so no split is empty.
    for split in ALLOWED_SPLITS:
        group = ordered.pop(0)
        assignments[group] = split
        current[split] += len(groups[group])

    for group in ordered:
        # Prefer the split furthest below target; stable tie-break by split order.
        split = max(
            ALLOWED_SPLITS,
            key=lambda candidate: (
                targets[candidate] - current[candidate],
                -ALLOWED_SPLITS.index(candidate),
            ),
        )
        assignments[group] = split
        current[split] += len(groups[group])

    assigned: list[dict[str, Any]] = []
    for specimen in specimens:
        group = _group_key(specimen)
        item = copy.deepcopy(specimen)
        item["split"] = assignments[group]
        item["split_group"] = group
        assigned.append(item)

    split_counts = Counter(item["split"] for item in assigned)
    group_sizes = {group: len(indices) for group, indices in sorted(groups.items())}
    return {
        "seed": seed,
        "ratios": ratios,
        "specimens": assigned,
        "group_assignments": dict(sorted(assignments.items())),
        "group_sizes": group_sizes,
        "counts": {split: split_counts.get(split, 0) for split in ALLOWED_SPLITS},
        "total": len(assigned),
    }


def split_manifest(
    manifest: dict[str, Any],
    *,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copy of the manifest with splits plus the split report."""
    specimens = manifest.get("specimens")
    if not isinstance(specimens, list):
        raise DatasetError("Manifest has no specimens array.")
    for specimen in specimens:
        if "split" in specimen:
            raise DatasetError(
                "Manifest already contains split assignments; remove them before "
                "asking for a new leakage-safe split."
            )
    result = assign_grouped_splits(
        specimens, train=train, val=val, test=test, seed=seed
    )
    output = copy.deepcopy(manifest)
    output["specimens"] = result["specimens"]
    output["split_protocol"] = {
        "method": "deterministic grouped greedy assignment",
        "group_key": "lot, else capture_session, else specimen_id",
        "seed": seed,
        "ratios": result["ratios"],
    }
    return output, result


def write_manifest(manifest: dict[str, Any], path: Path | str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def manifest_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
