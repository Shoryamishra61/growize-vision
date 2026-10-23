"""Acquire unchanged published LSID-Beans photographs and annotations, with hashes.

One physical capture per leaf is selected lexicographically before measurement
analysis. Source cv_1/split_1 is preserved; additional views are not manufactured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1] / "data/external/lsid_beans_f42hwwrpgn_v3"
API = "https://data.mendeley.com/public-api/datasets/f42hwwrpgn"
VERSION = 3


def get_json(url, params=None):
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=(15, 30))
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def catalog(folder, category):
    path = ROOT / "catalog" / f"{folder}_{category}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = get_json(API + "/files", {"version": VERSION, "folder_id": folder})
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def download(row, directory):
    filename = row["filename"]
    if Path(filename).name != filename or "\\" in filename or ":" in filename:
        raise ValueError("Unsafe publisher filename")
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = row["content_details"]["sha256_hash"]
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return path
    for attempt in range(3):
        try:
            with requests.get(
                row["content_details"]["download_url"], stream=True, timeout=(15, 30)
            ) as response:
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_content(1024 * 1024):
                    content.extend(chunk)
                    if len(content) > 5_000_000:
                        raise ValueError("Unexpectedly large source file")
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("Publisher SHA256 mismatch")
            partial = path.with_suffix(path.suffix + ".partial")
            partial.write_bytes(content)
            partial.replace(path)
            return path
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def annotation_capture_id(filename):
    if Path(filename).suffix.lower() != ".xml":
        raise ValueError("Expected a source XML annotation")
    return re.sub(r"(?:_annotation)+$", "", Path(filename).stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-leaves", type=int, default=612)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.limit_leaves <= 612 or not 1 <= args.workers <= 8:
        raise ValueError("Invalid acquisition bounds")
    ROOT.mkdir(parents=True, exist_ok=True)
    source = get_json(API, {"version": VERSION})
    if source["data_licence"]["short_name"] != "CC BY 4.0":
        raise ValueError("Source license changed")
    (ROOT / "source.json").write_text(json.dumps(source, indent=2))
    folders = get_json(API + f"/folders/{VERSION}")
    (ROOT / "source_folders.json").write_text(json.dumps(folders, indent=2))
    by_id = {row["id"]: row for row in folders}

    def folder_path(row):
        prefix = folder_path(by_id[row["parent_id"]]) + "/" if row.get("parent_id") in by_id else ""
        return prefix + row["name"]

    folder_ids = {folder_path(row): row["id"] for row in folders}
    fold = "cross-validation/cv_1/split_1"
    for row in catalog(folder_ids[fold], "fold"):
        download(row, ROOT / fold)

    def fetch_leaf(number):
        leaf = f"{number:03d}"
        prefix = f"LSID-Beans/{leaf}"
        images = sorted(
            catalog(folder_ids[prefix + "/images"], "images"), key=lambda row: row["filename"]
        )
        if not images:
            raise ValueError(f"No captures for leaf {leaf}")
        image = images[0]
        stem = Path(image["filename"]).stem
        files = {"image": download(image, ROOT / prefix / "images")}
        hashes = {"image": image["content_details"]["sha256_hash"]}
        for category, suffix in [
            ("annotation", "_annotation.xml"),
            ("segmentation", "_segmentation.raw"),
        ]:
            matches = [
                row
                for row in catalog(folder_ids[prefix + "/" + category], category)
                if (
                    annotation_capture_id(row["filename"]) == stem
                    if category == "annotation"
                    else row["filename"] == stem + suffix
                )
            ]
            if len(matches) != 1:
                raise ValueError(f"Missing or duplicated {category} for {stem}")
            files[category] = download(matches[0], ROOT / prefix / category)
            hashes[category] = matches[0]["content_details"]["sha256_hash"]
        return {
            "leaf_id": number,
            "plant_id": (number - 1) // 2 + 1,
            "capture_id": stem,
            "files": {key: str(path.relative_to(ROOT)) for key, path in files.items()},
            "publisher_sha256": hashes,
        }

    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(fetch_leaf, number): number
            for number in range(1, args.limit_leaves + 1)
        }
        for completed in as_completed(pending):
            try:
                rows.append(completed.result())
            except Exception as exc:
                failures.append({"leaf_id": pending[completed], "error": str(exc)})
            if (len(rows) + len(failures)) % 25 == 0:
                print(
                    f"Acquired {len(rows)}, failed {len(failures)}, requested {args.limit_leaves}",
                    flush=True,
                )
    rows.sort(key=lambda row: row["leaf_id"])
    (ROOT / "acquisition_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (ROOT / "acquisition_failures.json").write_text(json.dumps(failures, indent=2))
    print(json.dumps({"acquired_leaves": len(rows), "failures": len(failures)}), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
