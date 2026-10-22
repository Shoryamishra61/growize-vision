#!/usr/bin/env python
"""Write CHECKSUMS.sha256 for the delivered source, docs, config, and evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "README.md",
    "pyproject.toml",
    "apps",
    "config",
    "docs",
    "scripts",
    "src",
    "tests",
    "data/demo",
    "assets",
    "reports/DEMO_REPORT.md",
    "reports/EXPERIMENTS.md",
    "reports/metrics",
    "reports/failure_gallery",
    "test_evidence",
    "samples/README.md",
]
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    files: list[Path] = []
    for relative in INCLUDE:
        target = ROOT / relative
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(
                path
                for path in target.rglob("*")
                if path.is_file()
                and path.suffix.lower() not in EXCLUDE_SUFFIXES
                and "__pycache__" not in path.parts
            )
    unique = sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in unique
    ]
    output = ROOT / "CHECKSUMS.sha256"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output} with {len(lines)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
