"""Verify the field-okra source and derive species crops grouped by source capture."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import cv2
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_species_source import validate_member  # noqa: E402


def capture_key(stem: str) -> str:
    return re.sub(r"(?:_jpg|_jpeg|_png)+$", "", stem.split(".rf.")[0], flags=re.IGNORECASE)


def main():
    ref = "rahulhavalad/okra-data-set"
    root = ROOT / "data/external/rahulhavalad_okra"
    root.mkdir(parents=True, exist_ok=True)
    metadata_response = requests.get("https://www.kaggle.com/api/v1/datasets/list", params={"search": "okra_dataset"}, timeout=30)
    metadata_response.raise_for_status()
    metadata = next(row for row in metadata_response.json() if row["ref"] == ref)
    if metadata["licenseName"] != "Apache 2.0":
        raise ValueError("Source license changed")
    (root / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    archive = root / "source.zip"
    if not archive.exists():
        partial = root / "source.zip.partial"
        started = time.monotonic()
        size = 0
        with requests.get("https://www.kaggle.com/api/v1/datasets/download/" + ref, stream=True, timeout=(30, 60)) as response:
            response.raise_for_status()
            with partial.open("wb") as target:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > 200_000_000 or time.monotonic() - started > 600:
                        raise ValueError("Source download exceeded resource bound")
                    target.write(chunk)
        partial.replace(archive)
    images = root / "verified"
    if not images.exists():
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                validate_member(info.filename)
            if sum(info.file_size for info in bundle.infolist()) > 500_000_000:
                raise ValueError("Oversized field source")
            bundle.extractall(images)
    derived = ROOT / "data/derived/field_okra"
    derived.mkdir(parents=True, exist_ok=True)
    rows = []
    quarantined = []
    for path in sorted(images.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = Path(str(path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        if not label_path.exists():
            quarantined.append({"path": path.relative_to(ROOT).as_posix(), "reason": "MISSING_ANNOTATION"})
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Decode failure: {path}")
        height, width = image.shape[:2]
        group = "field_okra:" + capture_key(path.stem)
        common = {"source": ref, "license": "Apache 2.0", "group": group, "label": "Okra",
                  "source_image": path.relative_to(ROOT).as_posix(),
                  "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        annotations = label_path.read_text().splitlines()
        if not annotations:
            continue
        rows.append({**common, "path": path.relative_to(ROOT).as_posix(), "sha256": common["source_sha256"]})
        for index, line in enumerate(annotations):
            values = [float(value) for value in line.split()]
            if len(values) != 5 or values[0] not in {0, 1, 2} or not all(0 <= v <= 1 for v in values[1:]):
                raise ValueError(f"Invalid YOLO annotation: {label_path}")
            _, center_x, center_y, box_width, box_height = values
            x0 = max(0, int((center_x - box_width * 0.60) * width))
            y0 = max(0, int((center_y - box_height * 0.60) * height))
            x1 = min(width, int((center_x + box_width * 0.60) * width))
            y1 = min(height, int((center_y + box_height * 0.60) * height))
            if min(x1 - x0, y1 - y0) < 8:
                continue
            crop = image[y0:y1, x0:x1]
            name = hashlib.sha256((str(path.relative_to(images)) + str(index)).encode()).hexdigest()[:24]
            target = derived / f"{name}.png"
            if not cv2.imwrite(str(target), crop):
                raise OSError("Could not write derived crop")
            rows.append({**common, "path": target.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                         "bbox_px": [x0, y0, x1, y1], "annotation": label_path.relative_to(ROOT).as_posix()})
    (root / "derived_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (root / "quarantined.json").write_text(json.dumps(quarantined, indent=2), encoding="utf-8")
    (root / "archive.sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest(), encoding="ascii")
    print(json.dumps({"images_and_crops": len(rows), "capture_groups": len({r['group'] for r in rows}), "quarantined_images": len(quarantined),
                      "notice": "Species-only source annotations; no verified physical size, maturity or commercial provenance guarantee inferred."}), flush=True)


if __name__ == "__main__":
    main()
