"""Build per-kind median size references from original, measured training items.

Repeated photographs do not increase the specimen count. Validation and test
items never enter the reference distribution. Outputs are unvalidated references,
not a released grading policy or a claim of population-wide average size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from growize_vision.real_photos import (  # noqa: E402
    reference_statistics,
    verify_original_photo,
    verify_specimen_splits,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/real_size/reference_averages.json"
    )
    args = parser.parse_args()
    report = {
        "status": "BLOCKED",
        "references": [],
        "rule": "SMALL below the same-kind measured training median; BIG at or above it; refuse ambiguous measurements.",
        "reference_statistic": "median",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "limitations": [
            "Sample averages are not established population averages.",
            "Independent original real-photo size validation is still required.",
        ],
    }
    try:
        rows = [
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        verify_specimen_splits(rows)
        for row in rows:
            if row["split"] == "train":
                verify_original_photo(row, args.root)
        references = reference_statistics(rows)
        if not references:
            raise ValueError(
                "No reviewed, physically measured original training specimens supplied"
            )
        report.update(status="REFERENCE_BUILT_NOT_VALIDATED", references=references)
    except (ValueError, KeyError, OSError) as exc:
        report["reason"] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["references"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
