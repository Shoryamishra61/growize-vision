"""Collect a small attributed Open Images negative discovery set for visual review."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/openimages_negatives"
BASE = "https://storage.googleapis.com/openimages/"
FILES = {
    "images.csv": "2018_04/validation/validation-images-with-rotation.csv",
    "labels.csv": "v5/validation-annotations-human-imagelabels-boxable.csv",
    "classes.csv": "v5/class-descriptions-boxable.csv",
}
WANTED = {"Book", "Laptop", "Computer keyboard", "Mobile phone", "Chair", "Desk", "Pen", "Pencil", "Newspaper", "Envelope", "Clock", "Human hand"}
EXCLUDED = {"Food", "Fruit", "Vegetable", "Plant", "Flower", "Human face", "Tomato", "Potato", "Broccoli", "Carrot", "Cucumber", "Salad"}


def download(metadata):
    image_id = metadata["ImageID"]
    if len(image_id) != 16 or any(character not in "0123456789abcdef" for character in image_id):
        raise ValueError("Invalid Open Images identifier")
    path = OUT / "images" / f"{image_id}.jpg"
    url = f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
    if not path.exists():
        try:
            response = requests.get(url, timeout=(20, 40))
            response.raise_for_status()
            if len(response.content) > 15_000_000:
                raise ValueError("Image exceeds resource bound")
            path.write_bytes(response.content)
        except (requests.RequestException, ValueError) as exc:
            return {"image_id": image_id, "status": "DOWNLOAD_FAILED", "error": str(exc)}
    with Image.open(path) as image:
        image.verify()
    return {"path": path.relative_to(ROOT).as_posix(), "source": "OpenImages-validation",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "status": "REQUIRES_VISUAL_REVIEW",
            "license": "CC BY 2.0", "license_url": metadata["License"],
            "author": metadata["Author"], "author_profile": metadata["AuthorProfileURL"],
            "original_landing_url": metadata["OriginalLandingURL"], "original_url": metadata["OriginalURL"],
            "download_url": url, "title": metadata["Title"], "original_md5": metadata["OriginalMD5"],
            "positive_source_labels": metadata["positive_labels"], "image_id": image_id,
            "notice": "CVDF-resized pixels; publisher per-image license metadata retained; not measured produce or a universal negative label."}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    receipts = {}
    for name, suffix in FILES.items():
        path = OUT / name
        if not path.exists():
            response = requests.get(BASE + suffix, timeout=(20, 60))
            response.raise_for_status()
            if len(response.content) > 50_000_000:
                raise ValueError("Metadata exceeds resource bound")
            path.write_bytes(response.content)
        receipts[name] = {"url": BASE + suffix, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with (OUT / "classes.csv").open(encoding="utf-8", newline="") as stream:
        classes = dict(csv.reader(stream))
    labels = defaultdict(set)
    with (OUT / "labels.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Confidence"] == "1":
                labels[row["ImageID"]].add(classes[row["LabelName"]])
    with (OUT / "images.csv").open(encoding="utf-8", newline="") as stream:
        metadata = {row["ImageID"]: row for row in csv.DictReader(stream)}
    chosen = {}
    for label in sorted(WANTED):
        eligible = [image_id for image_id in sorted(labels) if label in labels[image_id]
                    and not (labels[image_id] & EXCLUDED) and image_id in metadata
                    and metadata[image_id]["License"].rstrip("/") == "https://creativecommons.org/licenses/by/2.0"]
        for image_id in eligible[:8]:
            chosen[image_id] = {**metadata[image_id], "positive_labels": sorted(labels[image_id])}
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(download, chosen.values()))
    (OUT / "discovery.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (OUT / "source.json").write_text(json.dumps({"source": "https://storage.googleapis.com/openimages/web/download_v7.html",
                                               "metadata": receipts, "selection": sorted(WANTED),
                                               "training_status": "WITHHELD_UNTIL_VISUAL_REVIEW"}, indent=2), encoding="utf-8")
    print(f"Downloaded {sum(row['status'] == 'REQUIRES_VISUAL_REVIEW' for row in rows)}/{len(rows)} discovery images", flush=True)


if __name__ == "__main__":
    main()
