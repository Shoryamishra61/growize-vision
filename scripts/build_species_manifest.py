"""Freeze licensed species rows, keeping derived captures and perceptual duplicates together."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/species_training_v2"


def perceptual_hash(image: Image.Image) -> int:
    gray = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    low = cv2.dct(gray)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return sum(int(bit) << index for index, bit in enumerate(bits))


def grouped_split(group: str) -> str:
    bucket = int(hashlib.sha256(("growize-v2|" + group).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--include-field-okra", action="store_true")
    parser.add_argument("--negative-manifest", type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    previous_manifest = args.cache_manifest or output / "manifest.jsonl"
    cached = {row["path"]: row for row in
              (json.loads(line) for line in previous_manifest.read_text(encoding="utf-8").splitlines())} if previous_manifest.exists() else {}
    rows = []
    # Supplied Kaggle split is preserved unless duplicate linkage requires quarantine.
    inventory = ROOT / "reports/species_audit/inventory.jsonl"
    for line in inventory.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["source"] == "kaggle_veg":
            row.update(split=row["original_split"], group="kaggle:" + row["sha256"], license="CC BY-SA 4.0")
            rows.append(row)
    metadata_path = ROOT / "data/external/kaggle_veg_source/source.json"
    if not metadata_path.exists():
        response = requests.get("https://www.kaggle.com/api/v1/datasets/list", params={"search": "vegetable-image-dataset"}, timeout=30)
        response.raise_for_status()
        metadata = next(row for row in response.json() if row["ref"] == "misrakahmed/vegetable-image-dataset")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["licenseName"] != "CC BY-SA 4.0" or metadata["currentVersionNumber"] != 1:
        raise ValueError("Base dataset license/version changed; review required")
    sources = [{"id": "kaggle_veg", "metadata": metadata,
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/"}]
    for source in ["j33g3nsm9k_v1"]:
        folder = ROOT / "data/external" / source
        metadata = json.loads((folder / "source.json").read_text(encoding="utf-8"))
        if metadata["data_licence"]["short_name"] != "CC BY 4.0" or not (folder / "source.zip").exists():
            raise ValueError(f"Unverified/incomplete source: {source}")
        sources.append({"id": source, "metadata": metadata, "archive_sha256": (folder / "archive.sha256").read_text().strip()})
        images = sorted(p for p in (folder / "images").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        captures = defaultdict(list)
        for path in images:
            if source == "j33g3nsm9k_v1":
                label = path.parent.parent.name
                group = source + ":" + path.stem.split("_aug")[0]
            else:
                label = path.parent.name
                group = source + ":" + path.stem
            aliases = {"Ladies Finger": "Okra", "Ladies_Finger": "Okra", "Lady Finger": "Okra",
                       "Bitter Melon": "Bitter_Gourd", "Bitter_Melon": "Bitter_Gourd", "Beans": "Bean"}
            label = aliases.get(label, label)
            if label not in {"Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli", "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber", "Okra", "Onion", "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"}:
                label = "Unknown"
            captures[(group, label)].append(path)
        for (group, label), paths in sorted(captures.items()):
            split = grouped_split(group)
            # Evaluation gives one vote per source capture, not per augmentation.
            for path in paths[:3 if split == "train" else 1]:
                rows.append({"path": path.relative_to(ROOT).as_posix(), "label": label, "split": split,
                             "group": group, "source": source, "license": "CC BY 4.0"})
    for crop in ["spinach", "onion"]:
        folder = ROOT / "data/external" / f"kritikseth_{crop}"
        metadata = json.loads((folder / "source.json").read_text(encoding="utf-8"))
        if metadata["licenseName"] != "CC0: Public Domain":
            raise ValueError("Unapproved crop subset license")
        sources.append({"id": f"kritikseth_{crop}", "metadata": metadata})
        for line in (folder / "receipts.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["status"] == "OK":
                row.update(split=row["original_split"], group=crop + ":" + row["sha256"])
                rows.append(row)

    if args.include_field_okra:
        folder = ROOT / "data/external/rahulhavalad_okra"
        metadata = json.loads((folder / "source.json").read_text(encoding="utf-8"))
        if metadata["licenseName"] != "Apache 2.0":
            raise ValueError("Field source license not approved")
        sources.append({"id": "rahulhavalad_okra", "metadata": metadata,
                        "archive_sha256": (folder / "archive.sha256").read_text().strip()})
        for line in (folder / "derived_manifest.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["split"] = grouped_split(row["group"])
            rows.append(row)

    if args.negative_manifest:
        negative_rows = [json.loads(line) for line in args.negative_manifest.read_text(encoding="utf-8").splitlines()]
        for row in negative_rows:
            if (row["review_status"] != "APPROVED_SOURCE_NEGATIVE" or row["label"] != "Unknown"
                    or row["license"] != "CC BY 2.0" or row["license_url"].rstrip("/") != "https://creativecommons.org/licenses/by/2.0"):
                raise ValueError("Unapproved negative source row")
            rows.append(row)
        sources.append({"id": "OpenImages-validation", "source": "https://storage.googleapis.com/openimages/web/download_v7.html",
                        "license": "CC BY 2.0", "reviewer": "Codex visual review; not human QA",
                        "attributions": negative_rows,
                        "manifest_sha256": hashlib.sha256(args.negative_manifest.read_bytes()).hexdigest()})

    # Link near-identical pHashes and source captures with union-find and 8-band lookup.
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        parents[find(left)] = find(right)

    captures = {}
    bands = defaultdict(list)
    hashes = []
    for index, row in enumerate(rows):
        path = ROOT / row["path"]
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if row["path"] in cached and cached[row["path"]]["sha256"] == row["sha256"]:
            hash_value = int(cached[row["path"]]["phash"], 16)
        else:
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                hash_value = min(perceptual_hash(image.rotate(angle, expand=True)) for angle in (0, 90, 180, 270))
        row["phash"] = f"{hash_value:016x}"
        hashes.append(hash_value)
        if row["group"] in captures:
            union(index, captures[row["group"]])
        captures[row["group"]] = index
        candidates = set()
        for band in range(8):
            candidates.update(bands[(band, (hash_value >> (band * 8)) & 255)])
        for previous in candidates:
            if (hash_value ^ hashes[previous]).bit_count() <= 4:
                union(index, previous)
        for band in range(8):
            bands[(band, (hash_value >> (band * 8)) & 255)].append(index)
        if index % 2000 == 0:
            print(f"Hashed {index}/{len(rows)}", flush=True)
    clusters = defaultdict(list)
    for index in range(len(rows)):
        clusters[find(index)].append(index)
    quarantined = []
    for indices in clusters.values():
        labels = {rows[index]["label"] for index in indices}
        splits = {rows[index]["split"] for index in indices}
        cluster = hashlib.sha256("|".join(sorted(rows[index]["sha256"] for index in indices)).encode()).hexdigest()
        subset_only = all("data/external/kritikseth_" in rows[index]["path"] for index in indices)
        # The small subset's supplied test files repeat training images. Repartition
        # whole duplicate groups before fitting; do not preserve that leaked split.
        if subset_only and len(labels) == 1:
            split = grouped_split(cluster)
            for index in indices:
                rows[index]["split"] = split
            splits = {split}
        if len(labels) > 1 or len(splits) > 1:
            quarantined.extend({**rows[index], "quarantine_reason": "RELATED_CROSS_SPLIT_OR_LABEL_CONFLICT"} for index in indices)
            for index in indices:
                rows[index]["split"] = "quarantine"
        for index in indices:
            rows[index]["group"] = cluster
        if len(splits) == 1 and next(iter(splits)) in {"validation", "test"} and len(labels) == 1:
            for index in indices[1:]:
                quarantined.append({**rows[index], "quarantine_reason": "RELATED_EVALUATION_DUPLICATE"})
                rows[index]["split"] = "quarantine"
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = {"images": len(rows), "groups": len(clusters), "quarantined": len(quarantined),
              "split_counts": dict(Counter(row["split"] for row in rows)),
              "label_split_counts": dict(Counter(f"{row['label']}:{row['split']}" for row in rows)),
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "limitations": ["Capture groups and pHash do not establish physical specimen independence.",
                              "Public data have no measured SMALL/BIG ground truth.",
                              "Pre-existing unlicensed/incorrectly labeled leaf and derived images are excluded."]}
    (output / "source_ledger.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    (output / "split_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
