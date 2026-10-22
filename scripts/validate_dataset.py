"""Validate a measured-specimen manifest and archive the full report.

Exit code 0 means the manifest passed provenance, measurement, hash, duplicate
and leakage checks.  It does not mean a model has been trained or evaluated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from growize_vision.config import load_policies
from growize_vision.dataset import DatasetError, validate_manifest
from growize_vision.evidence import PROJECT_ROOT, relative_to_project, write_json

DEFAULT_TEMPLATE = PROJECT_ROOT / "data" / "dataset_manifest_template.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "dataset_validation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Accept an intentionally empty template (still not evidence).",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Validate metadata without opening image files or verifying hashes.",
    )
    parser.add_argument(
        "--allowed-crops",
        default=None,
        help="Comma-separated crop keys; defaults to configured policy crops.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"FAIL: manifest does not exist: {manifest_path}", file=sys.stderr)
        return 1

    if args.allowed_crops:
        allowed = {
            crop.strip().lower().replace(" ", "_")
            for crop in args.allowed_crops.split(",")
            if crop.strip()
        }
    else:
        allowed = set(load_policies().crop_keys)

    try:
        report = validate_manifest(
            manifest_path,
            image_root=args.image_root,
            require_images=not args.skip_image_check,
            allow_empty=args.allow_empty,
            allowed_crops=allowed,
        )
    except DatasetError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    output = write_json(
        args.output,
        {
            "report_type": "measured_dataset_validation",
            "manifest": relative_to_project(manifest_path),
            **report,
            "limitations": [
                "A passing validation is not accuracy evidence.",
                "The template is empty until licensed, measured specimens are supplied.",
                "Synthetic fixtures are counted separately and excluded from real metrics.",
            ],
        },
    )
    print(f"Status: {report['status']}")
    print(
        "Specimens: {specimens_total} total / {real_specimens} real / "
        "{synthetic_specimens} synthetic".format(**report["counts"])
    )
    print(
        f"Images checked: {report['checked_images']}; hashes verified: "
        f"{report['verified_hashes']}; leakage groups: {len(report['leakage_groups'])}"
    )
    if report["warnings"]:
        print(f"Warnings: {len(report['warnings'])}")
    if report["errors"]:
        print(f"Errors: {len(report['errors'])}", file=sys.stderr)
        for error in report["errors"][:20]:
            print(f"  - {error}", file=sys.stderr)
    print(f"Full report: {output}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
