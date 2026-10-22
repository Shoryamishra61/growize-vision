#!/usr/bin/env python
"""Regenerate final automated evidence and index raw outputs.

Run from the repository root with the editable install active:

    python scripts/finalize_evidence.py

It does NOT run the camera/webcam preflight or the Gradio UI. It runs the
automated suite, lint, synthetic contract, latency, and failure gallery, copies
raw outputs into ``test_evidence/``, and finally writes ``CHECKSUMS.sha256``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EVIDENCE = ROOT / "test_evidence"
COMMANDS = [
    ("pytest", [sys.executable, "-m", "pytest", "-q", "--junitxml=reports/pytest_junit.xml"]),
    ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
    ("synthetic", [sys.executable, "scripts/evaluate_synthetic.py"]),
    (
        "latency",
        [sys.executable, "scripts/benchmark_latency.py", "--runs", "10", "--warmup", "2"],
    ),
    ("failure_gallery", [sys.executable, "scripts/build_failure_gallery.py"]),
]


def run(name: str, command: list[str]) -> dict[str, object]:
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    finished = datetime.now(timezone.utc).isoformat()
    output = (completed.stdout or "") + (completed.stderr or "")
    for target in (REPORTS / f"{name}_command.txt", EVIDENCE / f"{name}_command.txt"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            f"$ {' '.join(command)}\n\n{output}\n\nexit_code={completed.returncode}\n",
            encoding="utf-8",
        )
    return {
        "name": name,
        "command": command,
        "exit_code": completed.returncode,
        "started_utc": started,
        "finished_utc": finished,
        "output_tail": output.strip().splitlines()[-8:],
    }


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    results = []
    for name, command in COMMANDS:
        result = run(name, command)
        results.append(result)
        print(f"{name}: exit {result['exit_code']}")

    # Save the complete pytest output separately as the primary raw evidence file.
    pytest_command = EVIDENCE / "pytest_command.txt"
    if pytest_command.exists():
        shutil.copyfile(pytest_command, REPORTS / "pytest_full.txt")

    # Copy already-archived environment and blocked-evaluation reports.
    copies = {
        "preflight.json": "preflight.json",
        "synthetic_evaluation.json": "synthetic_evaluation.json",
        "latency_report.json": "latency_report.json",
        "baseline_evaluation.json": "baseline_evaluation.json",
        "training_blocked.json": "training_blocked.json",
        "dataset_validation.json": "dataset_validation.json",
        "group_split_report.json": "group_split_report.json",
    }
    for source_name, target_name in copies.items():
        source = REPORTS / source_name
        if source.exists():
            shutil.copyfile(source, EVIDENCE / target_name)

    gallery = REPORTS / "failure_gallery"
    if gallery.exists():
        target = EVIDENCE / "failure_gallery"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(gallery, target)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "commands": results,
        "note": "Synthetic and latency evidence are not real-produce accuracy.",
    }
    (EVIDENCE / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (REPORTS / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    # Checksums last, so they cover the evidence files just written.
    checksum_result = run("checksums", [sys.executable, "scripts/build_checksums.py"])
    results.append(checksum_result)
    failed = [item for item in results if item["exit_code"] != 0]
    print(f"Checksums: exit {checksum_result['exit_code']}")
    print("FAILED: " + ", ".join(str(item["name"]) for item in failed) if failed else "ALL PASS")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
