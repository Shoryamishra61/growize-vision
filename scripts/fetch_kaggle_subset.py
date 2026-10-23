"""Fetch a crop subset from a Kaggle CC0 classification source."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REF = "kritikseth/fruit-and-vegetable-image-recognition"


def fetch(item: tuple[str, str, int]) -> dict:
    crop, split, number = item
    name = f"{split}/{crop}/Image_{number}.jpg"
    target = ROOT / "data/external" / f"kritikseth_{crop}" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.kaggle.com/api/v1/datasets/download/{REF}"
    if not target.exists():
        response = requests.get(url, params={"file_name": name}, timeout=(20, 60))
        if response.status_code == 404:
            return {"name": name, "status": "NOT_FOUND"}
        response.raise_for_status()
        if len(response.content) > 20_000_000:
            raise ValueError("Individual image exceeds 20 MB")
        target.write_bytes(response.content)
    # Kaggle wraps larger individual files in a ZIP response.
    if zipfile.is_zipfile(target):
        archive = target.with_suffix(".download.zip")
        target.replace(archive)
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) != 1 or members[0].file_size > 20_000_000:
                raise ValueError("Unexpected single-image archive")
            target.write_bytes(bundle.read(members[0]))
    with Image.open(target) as image:
        image.verify()
    return {"path": target.relative_to(ROOT).as_posix(), "status": "OK", "label": crop.title(),
            "original_split": split, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "source": REF, "license": "CC0: Public Domain"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crop", choices=["spinach", "onion"])
    args = parser.parse_args()
    metadata = requests.get("https://www.kaggle.com/api/v1/datasets/list", params={"search": "fruit-and-vegetable-image-recognition"}, timeout=30)
    metadata.raise_for_status()
    source = next(r for r in metadata.json() if r["ref"] == REF)
    if source["licenseName"] != "CC0: Public Domain":
        raise ValueError("Source license changed; review required")
    output = ROOT / "data/external" / f"kritikseth_{args.crop}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    items = [(args.crop, split, number) for split, count in [("train", 100), ("validation", 10), ("test", 10)] for number in range(1, count + 1)]
    with ThreadPoolExecutor(max_workers=6) as executor:
        rows = list(executor.map(fetch, items))
    (output / "receipts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"Verified {sum(r['status'] == 'OK' for r in rows)}/{len(rows)} images", flush=True)


if __name__ == "__main__":
    main()
