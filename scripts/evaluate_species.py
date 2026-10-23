"""Evaluate a species checkpoint with row-level evidence; never infer physical size."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from growize_vision.crop_identity import (  # noqa: E402
    DEFAULT_DEEP_CLASSIFIER_PATH,
    CropIdentityService,
)
from growize_vision.real_photos import verify_original_photo, verify_specimen_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT / "reports/species_audit/inventory.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DEEP_CLASSIFIER_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--variant", choices=["clean"], default="clean")
    args = parser.parse_args()
    torch.set_num_threads(4)
    service = CropIdentityService(deep_classifier_path=args.checkpoint)
    if not service.is_deep:
        raise RuntimeError("Deep checkpoint unavailable; refusing to benchmark a different provider")
    rows = [json.loads(line) for line in args.inventory.read_text(encoding="utf-8").splitlines()]
    verify_specimen_splits([row for row in rows if row.get("split") != "quarantine"])
    rows = [r for r in rows if r.get("split", r.get("original_split")) == args.split and r.get("label") in service.classes]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No eligible rows")
    for row in rows:
        verify_original_photo(row, ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    features = []
    latencies = []
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            images = []
            tensors = []
            tick = time.perf_counter()
            for row in batch:
                image = cv2.imread(str(ROOT / row["path"]))
                if image is None:
                    raise ValueError(f"Image decode failed: {row['path']}")
                if args.variant == "text":
                    image = image.copy()
                    cv2.putText(image, "Fresh produce", (3, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
                elif args.variant == "rotate90":
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif args.variant == "dim":
                    image = (image.astype(float) * 0.45).astype(np.uint8)
                elif args.variant == "rotate45":
                    height, width = image.shape[:2]
                    bound = int(np.ceil((height + width) / np.sqrt(2)))
                    transform = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)
                    transform[:, 2] += [(bound - width) / 2, (bound - height) / 2]
                    image = cv2.warpAffine(image, transform, (bound, bound), borderValue=(238, 238, 238))
                elif args.variant in {"warm", "cool"}:
                    factors = [0.65, 1.0, 1.15] if args.variant == "warm" else [1.15, 0.95, 0.70]
                    image = np.clip(image.astype(float) * factors, 0, 255).astype(np.uint8)
                elif args.variant == "blur":
                    image = cv2.GaussianBlur(image, (7, 7), 1.8)
                elif args.variant == "zoom_out":
                    height, width = image.shape[:2]
                    small = cv2.resize(image, (max(1, int(width * 0.6)), max(1, int(height * 0.6))))
                    canvas = np.full_like(image, 238)
                    top, left = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
                    canvas[top:top + small.shape[0], left:left + small.shape[1]] = small
                    image = canvas
                elif args.variant == "perspective2d":
                    height, width = image.shape[:2]
                    original = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
                    projected = np.float32([[width * .12, height * .10], [width * .88, height * .10], [width - 1, height - 1], [0, height - 1]])
                    image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(original, projected), (width, height), borderValue=(238, 238, 238))
                images.append(image)
                tensors.append(service.deep_transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
            embedding = service.backbone(torch.stack(tensors)).cpu().numpy()
            probabilities = service.deep_model.predict_proba(embedding)
            features.append(embedding)
            for row, image, probability in zip(batch, images, probabilities, strict=True):
                present, reason = service.detect_produce_presence(image)
                prediction = service._prediction_from_probabilities(probability) if present else None
                index = int(np.argmax(probability))
                results.append({"path": row["path"], "sha256": row["sha256"], "label": row["label"],
                                "raw_prediction": service.classes[index], "raw_confidence": float(probability[index]),
                                "accepted_prediction": prediction.species_name.replace(" ", "_") if prediction else None,
                                "reason": reason if not present else ("ACCEPTED" if prediction else "LOW_CONFIDENCE"),
                                "probabilities": probability.tolist()})
            latencies.append((time.perf_counter() - tick) * 1000 / len(batch))
            print(f"{args.variant}: {len(results)}/{len(rows)}", flush=True)
    truth = [r["label"] for r in results]
    raw = [r["raw_prediction"] for r in results]
    accepted = [r for r in results if r["accepted_prediction"]]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": "IMAGE_OR_CAPTURE_GROUP_DIAGNOSTIC_NOT_PHYSICAL_SPECIMEN_VALIDATION",
        "candidate_manifest_matches": getattr(service, "model_manifest_sha256", None) == hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "crop_identity_sha256": hashlib.sha256((ROOT / "src/growize_vision/crop_identity.py").read_bytes()).hexdigest(),
        "confidence_floor": service.confidence_floor,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "variant": args.variant, "images": len(rows), "classes": service.classes,
        "raw_accuracy": float(np.mean(np.array(truth) == np.array(raw))),
        "classification_report": classification_report(truth, raw, labels=service.classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, raw, labels=service.classes).tolist(),
        "accepted_coverage": len(accepted) / len(rows),
        "accepted_accuracy": float(np.mean([r["label"] == r["accepted_prediction"] for r in accepted])) if accepted else None,
        "reasons": dict(Counter(r["reason"] for r in results)),
        "duration_seconds": time.perf_counter() - started,
        "batch_amortized_ms_p50": float(np.median(latencies)),
        "torch_version": torch.__version__, "device": "cpu", "threads": torch.get_num_threads(),
        "physical_size_evaluated": False,
    }
    (args.output / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "embeddings.npz", features=np.concatenate(features), paths=[r["path"] for r in rows], labels=truth)
    print(json.dumps({k: summary[k] for k in ["images", "raw_accuracy", "accepted_coverage", "accepted_accuracy", "duration_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
