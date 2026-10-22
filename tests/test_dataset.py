"""Measured-manifest validation and leakage-safe grouped splitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from growize_vision.dataset import (
    MANIFEST_VERSION,
    DatasetError,
    assign_grouped_splits,
    manifest_sha256,
    split_manifest,
    validate_manifest,
    write_manifest,
)


def specimen(index: int, **overrides) -> dict:
    value = 40.0 + index * 7.0
    payload = {
        "specimen_id": f"S{index:03d}",
        "crop": "potato",
        "physical_unit": "single_tuber",
        "measured_metric": "min_area_rect_short_mm",
        "measurement_1_mm": value,
        "measurement_2_mm": value + 0.2,
        "reference_grade": "SMALL" if value < 60 else "BIG",
        "lot": f"LOT-{index}",
        "capture_session": f"CS-{index}",
        "source": "ARL internal measured collection",
        "license": "internal-research-only",
        "consent": "informed_consent",
        "synthetic": False,
        "images": [
            {
                "path": f"images/S{index:03d}.png",
                "sha256": hashlib.sha256(f"image-{index}".encode()).hexdigest(),
            }
        ],
    }
    payload.update(overrides)
    return payload


def manifest(count: int = 6) -> dict:
    return {
        "version": MANIFEST_VERSION,
        "provenance": {
            "dataset_name": "ARL measured pilot (test fixture)",
            "owner": "AgResearch Labs",
            "collection_protocol": "Calibrated overhead capture, two caliper repeats.",
            "license": "internal-research-only",
            "source": "ARL lab",
            "contains_personal_data": False,
        },
        "specimens": [specimen(index) for index in range(count)],
    }


def test_valid_manifest_passes_provenance_and_truth_checks(tmp_path) -> None:
    report = validate_manifest(manifest(), image_root=tmp_path, require_images=False)
    assert report["ok"] is True
    assert report["status"] == "PASS"
    assert report["counts"]["specimens_total"] == 6
    assert report["counts"]["real_specimens"] == 6
    assert report["counts"]["synthetic_specimens"] == 0
    assert report["counts"]["real_by_crop"] == {"potato": 6}
    assert report["counts"]["real_by_grade"] == {"BIG": 3, "SMALL": 3}
    assert report["real_evaluable"] is True
    assert report["verified_hashes"] == 0


def test_empty_dataset_is_not_silently_evidence(tmp_path) -> None:
    empty = manifest()
    empty["specimens"] = []
    blocked = validate_manifest(empty, image_root=tmp_path, require_images=False)
    assert blocked["ok"] is False
    assert "empty" in blocked["errors"][0].lower()
    allowed = validate_manifest(
        empty, image_root=tmp_path, require_images=False, allow_empty=True
    )
    assert allowed["ok"] is True
    assert allowed["real_evaluable"] is False
    assert any("empty" in warning.lower() for warning in allowed["warnings"])


def test_duplicate_specimen_and_image_hashes_are_rejected(tmp_path) -> None:
    payload = manifest(3)
    payload["specimens"][1]["specimen_id"] = payload["specimens"][0]["specimen_id"]
    report = validate_manifest(payload, image_root=tmp_path, require_images=False)
    assert report["ok"] is False
    assert any("duplicate specimen_id" in error for error in report["errors"])

    payload = manifest(3)
    payload["specimens"][1]["images"][0]["sha256"] = payload["specimens"][0]["images"][0][
        "sha256"
    ]
    report = validate_manifest(payload, image_root=tmp_path, require_images=False)
    assert report["ok"] is False
    assert any("duplicate image hash" in error for error in report["errors"])


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload["provenance"].pop("license"),
        lambda payload: payload["specimens"][0].update({"source": "unknown"}),
        lambda payload: payload["specimens"][0].update({"license": "TBD"}),
        lambda payload: payload["specimens"][0].update({"measurement_1_mm": None}),
        lambda payload: payload["specimens"][0].update({"measurement_1_mm": -3}),
        lambda payload: payload["specimens"][0].update({"images": []}),
        lambda payload: payload["specimens"][0]["images"][0].update({"sha256": "nope"}),
        lambda payload: payload["specimens"][0].update({"physical_unit": "bucket"}),
        lambda payload: payload["specimens"][0].update({"measured_metric": "weight_g"}),
        lambda payload: payload["specimens"][0].update({"synthetic": "false"}),
    ],
)
def test_missing_or_invalid_provenance_is_rejected(tmp_path, mutator) -> None:
    payload = manifest(2)
    mutator(payload)
    report = validate_manifest(payload, image_root=tmp_path, require_images=False)
    assert report["ok"] is False
    assert report["errors"]


def test_image_hash_and_path_are_verified_when_requested(tmp_path) -> None:
    payload = manifest(1)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    raw = b"not-a-real-png-but-hash-verified"
    (image_dir / "S000.png").write_bytes(raw)
    payload["specimens"][0]["images"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    report = validate_manifest(payload, image_root=tmp_path, require_images=True)
    assert report["ok"] is True
    assert report["checked_images"] == 1
    assert report["verified_hashes"] == 1

    payload["specimens"][0]["images"][0]["sha256"] = hashlib.sha256(b"other").hexdigest()
    report = validate_manifest(payload, image_root=tmp_path, require_images=True)
    assert report["ok"] is False
    assert any("SHA-256 mismatch" in error for error in report["errors"])


@pytest.mark.parametrize(
    "bad_path",
    ["../outside.png", "/absolute/outside.png", "C:\\Windows\\outside.png"],
)
def test_image_paths_cannot_escape_the_manifest_root(tmp_path, bad_path) -> None:
    payload = manifest(1)
    payload["specimens"][0]["images"][0]["path"] = bad_path
    report = validate_manifest(payload, image_root=tmp_path, require_images=False)
    assert report["ok"] is False
    assert any("relative" in error or "may not contain" in error for error in report["errors"])


def test_split_leakage_within_a_lot_is_rejected(tmp_path) -> None:
    payload = manifest(3)
    payload["specimens"][0]["lot"] = "LOT-SHARED"
    payload["specimens"][1]["lot"] = "LOT-SHARED"
    payload["specimens"][0]["split"] = "train"
    payload["specimens"][1]["split"] = "test"
    report = validate_manifest(payload, image_root=tmp_path, require_images=False)
    assert report["ok"] is False
    assert any("Leakage" in error for error in report["errors"])


def test_synthetic_fixtures_are_counted_but_never_real_evaluable(tmp_path) -> None:
    payload = manifest(2)
    for item in payload["specimens"]:
        item["synthetic"] = True
        item["source"] = "synthetic_fixture_generator"
        item["license"] = "generated-in-repo"
        item["consent"] = "not_applicable_synthetic"
        item["split"] = "train"
    report = validate_manifest(payload, image_root=tmp_path, require_images=False)
    assert report["ok"] is True
    assert report["counts"]["real_specimens"] == 0
    assert report["counts"]["synthetic_specimens"] == 2
    assert report["real_evaluable"] is False
    assert any("synthetic" in warning.lower() for warning in report["warnings"])


def test_grouped_split_is_deterministic_and_leakage_free() -> None:
    specimens = [specimen(index) for index in range(9)]
    first = assign_grouped_splits(specimens, train=0.6, val=0.2, test=0.2, seed=7)
    second = assign_grouped_splits(specimens, train=0.6, val=0.2, test=0.2, seed=7)
    assert first["counts"] == second["counts"]
    assert first["group_assignments"] == second["group_assignments"]
    assert sum(first["counts"].values()) == 9
    assert set(first["counts"]) == {"train", "val", "test"}
    seen: dict[str, set[str]] = {}
    for item in first["specimens"]:
        group = item["split_group"]
        seen.setdefault(group, set()).add(item["split"])
    assert all(len(splits) == 1 for splits in seen.values())


def test_grouped_split_keeps_a_multi_specimen_lot_together() -> None:
    specimens = [specimen(index) for index in range(6)]
    specimens[1]["lot"] = "LOT-0"
    specimens[2]["lot"] = "LOT-0"
    result = assign_grouped_splits(specimens, seed=3)
    lot_splits = {
        item["split"] for item in result["specimens"] if item["lot"] == "LOT-0"
    }
    assert len(lot_splits) == 1
    assert result["group_sizes"]["lot:LOT-0"] == 3


def test_split_manifest_writes_protocol_and_rejects_prior_splits() -> None:
    payload = manifest(6)
    split, report = split_manifest(payload, seed=11)
    assert split["split_protocol"]["seed"] == 11
    assert all(item["split"] in {"train", "val", "test"} for item in split["specimens"])
    assert report["total"] == 6
    with pytest.raises(DatasetError, match="already contains split"):
        split_manifest(split, seed=12)


def test_split_requires_enough_groups_and_valid_ratios() -> None:
    with pytest.raises(DatasetError, match="at least 3"):
        assign_grouped_splits([specimen(0), specimen(1)], seed=1)
    with pytest.raises(DatasetError, match="sum to 1.0"):
        assign_grouped_splits([specimen(i) for i in range(6)], train=0.8, val=0.2, test=0.2)
    with pytest.raises(DatasetError, match="> 0"):
        assign_grouped_splits([specimen(i) for i in range(6)], train=1.0, val=0.0, test=0.0)
    with pytest.raises(DatasetError, match="empty"):
        assign_grouped_splits([], seed=1)


def test_manifest_roundtrip_and_digest(tmp_path) -> None:
    path = write_manifest(manifest(2), tmp_path / "nested" / "manifest.json")
    written = json.loads(Path(path).read_text(encoding="utf-8"))
    assert written["version"] == MANIFEST_VERSION
    assert len(manifest_sha256(path)) == 64
    report = validate_manifest(path, image_root=tmp_path, require_images=False)
    assert report["ok"] is True


def test_repeat_measurement_disagreement_is_a_warning_not_silent(tmp_path) -> None:
    payload = manifest(1)
    payload["specimens"][0]["measurement_2_mm"] = (
        payload["specimens"][0]["measurement_1_mm"] + 3.0
    )
    report = validate_manifest(payload, image_root=tmp_path, require_images=False)
    assert report["ok"] is True
    assert any("disagree" in warning for warning in report["warnings"])
