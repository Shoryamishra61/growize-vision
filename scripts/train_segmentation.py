"""Refuse to train a segmentation model without licensed, measured evidence.

This is intentionally a refusal, not a training implementation.  Uncommenting a
few imports and printing invented metrics would be worse than shipping no model.
"""

from __future__ import annotations

import argparse

from growize_vision.evidence import PROJECT_ROOT, environment_metadata, write_json

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "training_blocked.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blockers = [
        "No licensed, consented, physically measured real-produce dataset is present.",
        "No QA-approved segmentation masks with inter-annotator review are present.",
        "No ARL/Growize-approved crop thresholds exist, so a learned grader could "
        "not be validated against a real decision policy.",
        "No compatible GPU is available in this environment; CPU-only Torch cannot "
        "support a credible training/evaluation run within the handoff window.",
        "Training without a leakage-safe, grouped real split would produce "
        "numbers that must not be reported.",
    ]
    report = {
        "report_type": "segmentation_training_attempt",
        "status": "BLOCKED",
        "evidence_class": "NOT_RUN",
        "environment": environment_metadata(),
        "requested_manifest": args.manifest,
        "requested_epochs": args.epochs,
        "blockers": blockers,
        "what_exists_instead": [
            "Deterministic OpenCV baseline shipped as the release path.",
            "Synthetic fixtures verify geometry and refusal contracts only.",
            "Manifest validator and grouped splitter are ready for real data.",
            "A model card records the absent model rather than inventing one.",
        ],
        "exit_meaning": (
            "No weights, metrics, checkpoints, or training logs were produced."
        ),
    }
    output = write_json(args.output, report)
    print("TRAINING BLOCKED - no model, weights, or metrics were produced.")
    for blocker in blockers:
        print(f"  - {blocker}")
    print(f"Refusal report: {output}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
