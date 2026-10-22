"""Assign leakage-safe grouped train/validation/test splits to a manifest.

Groups are whole lots, else whole capture sessions, else specimen IDs.  An
empty template cannot be split and fails with a clear message.  Synthetic-only
manifests are allowed for plumbing tests but are marked non-evidence.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from growize_vision.config import load_policies
from growize_vision.dataset import (
    DatasetError,
    split_manifest,
    validate_manifest,
    write_manifest,
)
from growize_vision.evidence import (
    PROJECT_ROOT,
    load_json,
    relative_to_project,
    write_json,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "dataset_manifest_split.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "group_split_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest does not exist: {manifest_path}")
    manifest = load_json(manifest_path)
    policies = load_policies()

    validation = validate_manifest(
        manifest_path,
        require_images=True,
        allow_empty=False,
        allowed_crops=set(policies.crop_keys),
    )
    if not validation["ok"]:
        print("FAIL: fix manifest errors before splitting:")
        for error in validation["errors"][:20]:
            print(f"  - {error}")
        return 1

    try:
        split_manifest_payload, result = split_manifest(
            manifest,
            train=args.train,
            val=args.val,
            test=args.test,
            seed=args.seed,
        )
    except DatasetError as exc:
        print(f"FAIL: {exc}")
        return 1

    output_manifest = write_manifest(split_manifest_payload, args.output)
    real_count = sum(
        1 for item in result["specimens"] if not bool(item.get("synthetic"))
    )
    report = {
        "report_type": "leakage_safe_group_split",
        "status": "PASS" if real_count else "TEMPLATE_ONLY",
        "evidence_class": "REAL_SPLIT" if real_count else "SYNTHETIC_PLUMBING_ONLY",
        "source_manifest": relative_to_project(manifest_path),
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "output_manifest": relative_to_project(output_manifest),
        "split_protocol": split_manifest_payload["split_protocol"],
        "counts": result["counts"],
        "total": result["total"],
        "real_specimens": real_count,
        "group_sizes": result["group_sizes"],
        "group_assignments": result["group_assignments"],
        "warnings": (
            ["No real specimens: this split is plumbing evidence, not model evidence."]
            if not real_count
            else []
        ),
        "notes": [
            "All images from a lot/session/specimen remain in the same split.",
            "Split assignment is deterministic for the recorded seed.",
        ],
    }
    output_report = write_json(args.report, report)
    print(f"Split manifest: {output_manifest}")
    print(f"Split report:   {output_report}")
    print(
        "Counts: train={train} val={val} test={test}; real={real}".format(
            **result["counts"], real=real_count
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
