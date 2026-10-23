"""Run species inference over all local imagery; mixed-pool results are not a test score."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from growize_vision.crop_identity import CropIdentityService  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports/species_training_v2")
    parser.add_argument("--additional-manifest", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    service = CropIdentityService(deep_classifier_path=args.checkpoint)
    if not service.is_deep:
        raise RuntimeError("Requested provider is unavailable")
    rows = [json.loads(line) for line in (ROOT / "reports/species_audit/inventory.jsonl").read_text(encoding="utf-8").splitlines()]
    for path in sorted((ROOT / "data/external/j33g3nsm9k_v1/images").rglob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            label = path.parent.parent.name
            rows.append({"path": path.relative_to(ROOT).as_posix(), "source": "j33g3nsm9k_v1", "label": label})
    for crop in ["spinach", "onion"]:
        receipt = ROOT / "data/external" / f"kritikseth_{crop}" / "receipts.jsonl"
        rows.extend(json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines() if json.loads(line)["status"] == "OK")
    if args.additional_manifest:
        rows.extend(json.loads(line) for line in args.additional_manifest.read_text(encoding="utf-8").splitlines())
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate inventory paths")
    output = args.run_dir / "full_runtime_audit"
    output.mkdir(parents=True, exist_ok=True)
    identity = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "identity_source_sha256": hashlib.sha256((ROOT / "src/growize_vision/crop_identity.py").read_bytes()).hexdigest(),
                "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Audit identity changed; use a separate run directory")
    identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    counts = defaultdict(Counter)
    failures = []
    started = time.perf_counter()
    partial = output / "predictions.partial.jsonl.gz"
    with gzip.open(partial, "wt", encoding="utf-8") as ledger, torch.inference_mode():
        for offset in range(0, len(rows), 32):
            batch = rows[offset:offset + 32]
            chunk_path = chunks / f"{offset:06d}.jsonl"
            if chunk_path.exists():
                records = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines()]
                if {row["path"] for row in batch} != {record["path"] for record in records}:
                    raise ValueError("Cached audit chunk has different input rows")
                for record in records:
                    if hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
                        raise ValueError("An audited image has changed")
            else:
                presences, tensors, valid_rows, records = [], [], [], []
                for row in batch:
                    file_sha = hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
                    image = cv2.imread(str(ROOT / row["path"]))
                    if image is None:
                        records.append({"path": row["path"], "sha256": file_sha, "source": row["source"], "reason": "DECODE_FAILED"})
                        continue
                    # Retain only small tensors and screening metadata, not 32 full-resolution photos.
                    presences.append(service.detect_produce_presence(image))
                    tensors.append(service.deep_transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
                    valid_rows.append({**row, "sha256": file_sha})
                if tensors:
                    features = service.backbone(torch.stack(tensors)).cpu().numpy()
                    probabilities = service.deep_model.predict_proba(features)
                    for row, presence, probability in zip(valid_rows, presences, probabilities, strict=True):
                        present, presence_reason = presence
                        prediction = service._prediction_from_probabilities(probability) if present else None
                        index = int(np.argmax(probability))
                        records.append({"path": row["path"], "sha256": row["sha256"],
                                        "source": row["source"], "declared_label": row.get("label"),
                                        "raw_prediction": service.classes[index], "raw_confidence": float(probability[index]),
                                        "accepted_species": prediction.species_key if prediction else None,
                                        "reason": presence_reason if not present else ("SPECIES_SUPPORTED" if prediction else "UNKNOWN_OR_LOW_CONFIDENCE"),
                                        "physical_size_class": None})
                temporary = chunk_path.with_suffix(".partial")
                temporary.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
                temporary.replace(chunk_path)
            for record in records:
                ledger.write(json.dumps(record) + "\n")
                if record["reason"] == "DECODE_FAILED":
                    failures.append(record)
                else:
                    counts[record["source"]]["processed"] += 1
                    counts[record["source"]]["species_returned" if record["accepted_species"] else "abstained"] += 1
            if offset % 512 == 0:
                print(f"Full-pool species inference: {offset}/{len(rows)}", flush=True)
    partial.replace(output / "predictions.jsonl.gz")
    summary = {"evidence": "MIXED_TRAIN_VALIDATION_TEST_AND_UNLICENSED_DISCOVERY_INPUTS; runtime diagnostic only, not accuracy or physical-grade validation",
               "inventory_images": len(rows), "processed_images": sum(count["processed"] for count in counts.values()),
               "sources": {key: dict(value) for key, value in counts.items()}, "failures": failures,
               "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "duration_seconds": time.perf_counter() - started, "device": "cpu", "threads": 4,
               "physical_ground_truth_available": False, "physical_size_accuracy": None}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
