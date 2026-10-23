"""Fetch a versioned, CC BY 4.0 Mendeley dataset and retain its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path, PurePosixPath

import requests


def validate_member(name: str) -> None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or ":" in normalized:
        raise ValueError(f"Unsafe archive path: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["j33g3nsm9k", "b9rvg4f2st"])
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "data/external" / f"{args.dataset}_v{args.version}"
    root.mkdir(parents=True, exist_ok=True)
    base = "https://data.mendeley.com/public-api"
    metadata_response = requests.get(f"{base}/datasets/{args.dataset}", params={"version": args.version}, timeout=30)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    if metadata.get("data_licence", {}).get("short_name") != "CC BY 4.0":
        raise ValueError("Dataset license has changed; review required")
    (root / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    file_response = requests.get(f"{base}/datasets/{args.dataset}/files", params={"folder_id": "root", "version": args.version}, timeout=30)
    file_response.raise_for_status()
    source_files = file_response.json()
    (root / "source_files.json").write_text(json.dumps(source_files, indent=2), encoding="utf-8")
    download_url = f"{base}/zip/{args.dataset}/download/{args.version}"
    expected_sha256 = None
    if isinstance(source_files, list) and len(source_files) == 1 and source_files[0]["filename"].endswith(".zip"):
        details = source_files[0]["content_details"]
        download_url = details["download_url"]
        expected_sha256 = details["sha256_hash"]
    archive = root / "source.zip"
    if not archive.exists():
        partial = root / "source.zip.partial"
        digest = hashlib.sha256()
        size_bytes = 0
        started = time.monotonic()
        with requests.get(download_url, stream=True, timeout=(30, 60)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    size_bytes += len(chunk)
                    if time.monotonic() - started > 20 * 60:
                        raise TimeoutError("Source download exceeded 20 minutes")
                    if size_bytes > 2 * 1024**3:
                        raise ValueError("Source archive exceeds 2 GiB limit")
                    digest.update(chunk)
                    output.write(chunk)
        if expected_sha256 and digest.hexdigest() != expected_sha256:
            raise ValueError("Publisher archive checksum mismatch")
        partial.replace(archive)
        (root / "archive.sha256").write_text(digest.hexdigest() + "\n", encoding="ascii")
    target = root / "images"
    if not target.exists():
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                validate_member(member.filename)
            if sum(member.file_size for member in bundle.infolist()) > 5 * 1024**3:
                raise ValueError("Expanded archive exceeds 5 GiB limit")
            bundle.extractall(target)
    print(f"Downloaded {archive.stat().st_size} bytes; extracted to {target}", flush=True)


if __name__ == "__main__":
    main()
