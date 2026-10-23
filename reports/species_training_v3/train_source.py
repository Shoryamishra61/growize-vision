"""Train a reproducible CPU species candidate; preserve the incumbent artifact."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from growize_vision.crop_identity import CropIdentityService  # noqa: E402

OUT = ROOT / "reports/species_training_v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def augment(image: np.ndarray, seed: int) -> np.ndarray:
    """One deterministic robustness view per training image, after split freeze."""
    rng = np.random.default_rng(seed)
    image = np.rot90(image, int(rng.integers(0, 4))).copy()
    image = np.clip(image.astype(float) * rng.uniform(0.5, 1.4), 0, 255).astype(np.uint8)
    if seed % 2 == 0:
        cv2.putText(image, "Fresh produce", (3, max(18, image.shape[0] // 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.4, image.shape[1] / 400),
                    [(255, 0, 0), (0, 0, 0), (255, 255, 255)][seed % 3], 2)
    return image


def reusable_features(run_dir, cache_name, backbone_sha256):
    if run_dir is None:
        return {}
    experiment = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
    if experiment["backbone_sha256"] != backbone_sha256:
        raise ValueError("Cannot reuse embeddings from different backbone weights")
    frozen = ast.parse((run_dir / "train_source.py").read_text(encoding="utf-8"))
    frozen_augment = next(node for node in frozen.body if isinstance(node, ast.FunctionDef) and node.name == "augment")
    current_augment = ast.parse(inspect.getsource(augment)).body[0]
    if ast.dump(frozen_augment) != ast.dump(current_augment):
        raise ValueError("Training view changed; embedding reuse is invalid")
    old_rows = [json.loads(line) for line in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    split = "train" if cache_name.startswith("train") else cache_name.split("_")[0]
    old_rows = [row for row in old_rows if row["split"] == split]
    manifest_key = hashlib.sha256(json.dumps(old_rows, sort_keys=True).encode()).hexdigest()
    key = hashlib.sha256((manifest_key + experiment["script_sha256"] + backbone_sha256 + cache_name).encode()).hexdigest()[:20]
    reused = {}
    for offset in range(0, len(old_rows), 32):
        features = np.load(run_dir / "cache" / f"{key}_{offset:06d}.npy", allow_pickle=False)
        for row, feature in zip(old_rows[offset:offset + 32], features, strict=True):
            reused[(row["path"], row["sha256"])] = feature
    return reused


def extract(rows, service, cache_name, run_dir, augmented=False, reuse_dir=None):
    cache = run_dir / "cache"
    cache.mkdir(exist_ok=True)
    manifest_key = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    cache_key = hashlib.sha256((manifest_key + digest(__file__) + service.embedding_sha256 + cache_name).encode()).hexdigest()[:20]
    reused = reusable_features(reuse_dir, cache_name, service.embedding_sha256)
    print(f"{cache_name}: {sum((row['path'], row['sha256']) in reused for row in rows)} reusable embeddings", flush=True)
    chunks = []
    with torch.inference_mode():
        for offset in range(0, len(rows), 32):
            chunk_path = cache / f"{cache_key}_{offset:06d}.npy"
            if chunk_path.exists():
                features = np.load(chunk_path, allow_pickle=False)
            else:
                tensors, missing_indices = [], []
                features = np.empty((min(32, len(rows) - offset), service.deep_model.n_features_in_), dtype=np.float32)
                for index, row in enumerate(rows[offset:offset + 32]):
                    path = ROOT / row["path"]
                    if digest(path) != row["sha256"]:
                        raise ValueError(f"Source mutated: {row['path']}")
                    if (row["path"], row["sha256"]) in reused:
                        features[index] = reused[(row["path"], row["sha256"])]
                        continue
                    image = cv2.imread(str(path))
                    if image is None:
                        raise ValueError(f"Decode failed: {row['path']}")
                    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                    if augmented:
                        image = augment(image, int(row["sha256"][:8], 16))
                    tensors.append(service.deep_transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
                    missing_indices.append(index)
                if tensors:
                    features[missing_indices] = service.backbone(torch.stack(tensors)).cpu().numpy()
                np.save(chunk_path, features, allow_pickle=False)
            if features.shape != (min(32, len(rows) - offset), service.deep_model.n_features_in_) or not np.all(np.isfinite(features)):
                raise ValueError("Invalid cached embeddings")
            chunks.append(features)
            if offset % 512 == 0:
                print(f"{cache_name}: {offset}/{len(rows)}", flush=True)
    return np.concatenate(chunks)


def metrics(truth, predictions, classes):
    return {"accuracy": float(np.mean(np.array(truth) == np.array(predictions))),
            "classification_report": classification_report(truth, predictions, labels=classes, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predictions, labels=classes).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=OUT)
    parser.add_argument("--reuse-run", type=Path)
    parser.add_argument("--zero-validation-errors", action="store_true")
    args = parser.parse_args()
    output = args.run_dir
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    manifest = output / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    groups = defaultdict(set)
    for row in rows:
        if row["split"] != "quarantine":
            groups[row["group"]].add(row["split"])
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("Group leakage")
    partitions = {split: [r for r in rows if r["split"] == split] for split in ["train", "validation", "test"]}
    classes = sorted({r["label"] for r in partitions["train"]})
    for split, partition in partitions.items():
        if set(classes) != {r["label"] for r in partition}:
            raise ValueError(f"Missing class support in {split}")
    service = CropIdentityService()
    if not service.is_deep:
        raise ValueError("Backbone unavailable")
    model_dir = output / "candidate"
    model_dir.mkdir(exist_ok=True)
    backbone_path = model_dir / "mobilenetv3_backbone.pt"
    if args.reuse_run and not backbone_path.exists():
        import shutil

        shutil.copyfile(args.reuse_run / "candidate/mobilenetv3_backbone.pt", backbone_path)
    if not backbone_path.exists():
        torch.save(service.backbone.state_dict(), backbone_path)
    service.embedding_sha256 = digest(backbone_path)
    experiment = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING",
                  "manifest_sha256": digest(manifest), "script_sha256": digest(__file__),
                  "backbone_sha256": digest(backbone_path), "seed": 42, "device": "cpu",
                  "torch": torch.__version__, "split_counts": {s: dict(Counter(r["label"] for r in p)) for s, p in partitions.items()},
                  "hypothesis": "Correct taxonomy, all licensed training images and grouped robustness views improve crop recognition without confidence rewriting.",
                  "physical_size_training": False}
    experiment["reuse_run"] = str(args.reuse_run) if args.reuse_run else None
    experiment["validation_error_target"] = 0.0 if args.zero_validation_errors else 0.01
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    train_features = np.concatenate([extract(partitions["train"], service, "train_clean", output, reuse_dir=args.reuse_run),
                                     extract(partitions["train"], service, "train_augmented", output, augmented=True, reuse_dir=args.reuse_run)])
    train_labels = np.array([r["label"] for r in partitions["train"]] * 2)
    validation_features = extract(partitions["validation"], service, "validation_clean", output, reuse_dir=args.reuse_run)
    validation_labels = np.array([r["label"] for r in partitions["validation"]])
    runs = []
    best_model = None
    best_score = -1
    with threadpool_limits(limits=4):
        for regularization in [0.3, 1.0]:
            classifier = LogisticRegression(C=regularization, max_iter=1000, random_state=42, class_weight="balanced")
            classifier.fit(train_features, train_labels)
            report = metrics(validation_labels, classifier.predict(validation_features), classes)
            score = report["classification_report"]["macro avg"]["f1-score"]
            runs.append({"C": regularization, "validation": report})
            print(f"C={regularization}, validation macro F1={score:.6f}", flush=True)
            if score > best_score:
                best_model, best_score = classifier, score
    probabilities = best_model.predict_proba(validation_features)
    predicted = best_model.classes_[probabilities.argmax(axis=1)]
    known = predicted != "Unknown"
    confidence_floor = None
    threshold_reports = []
    for threshold in [0.65, 0.75, 0.85, 0.90, 0.95, 0.975, 0.99]:
        accepted = known & (probabilities.max(axis=1) >= threshold)
        error = float(np.mean(predicted[accepted] != validation_labels[accepted])) if accepted.any() else None
        threshold_reports.append({"threshold": threshold, "coverage": float(np.mean(accepted)), "error": error})
        if error is not None and error <= experiment["validation_error_target"] and confidence_floor is None:
            confidence_floor = threshold
    experiment["validation_risk_gate_passed"] = confidence_floor is not None
    if confidence_floor is None:
        confidence_floor = 1.0
    artifact = {"backbone_name": "mobilenetv3_large_100", "classifier": best_model,
                "classes": best_model.classes_.tolist(), "confidence_floor": confidence_floor,
                "backbone_file": backbone_path.name, "backbone_sha256": digest(backbone_path),
                "manifest_sha256": digest(manifest), "training_script_sha256": digest(__file__),
                "validation_macro_f1": best_score, "format_version": 2}
    checkpoint = model_dir / "deep_vegetable_classifier.pkl"
    with checkpoint.open("wb") as checkpoint_stream:
        pickle.dump(artifact, checkpoint_stream)
    # Only open test predictions after model and threshold are frozen.
    test_features = extract(partitions["test"], service, "test_clean", output, reuse_dir=args.reuse_run)
    test_labels = [r["label"] for r in partitions["test"]]
    test_probabilities = best_model.predict_proba(test_features)
    test_predictions = best_model.classes_[test_probabilities.argmax(axis=1)]
    incumbent_probabilities = service.deep_model.predict_proba(test_features)
    incumbent_predictions = np.array(service.classes)[incumbent_probabilities.argmax(axis=1)]
    test_rows = [{"path": row["path"], "sha256": row["sha256"], "group": row["group"], "source": row["source"],
                  "label": row["label"], "prediction": str(prediction), "confidence": float(probability.max()),
                  "incumbent_prediction": str(incumbent), "probabilities": probability.tolist()}
                 for row, prediction, probability, incumbent in zip(partitions["test"], test_predictions, test_probabilities, incumbent_predictions, strict=True)]
    (output / "test_predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in test_rows), encoding="utf-8")
    experiment.update(status="EVALUATED_NOT_PROMOTED", duration_seconds=time.perf_counter() - started,
                      checkpoint_sha256=digest(checkpoint), validation_runs=runs, thresholds=threshold_reports,
                      confidence_floor=confidence_floor, test=metrics(test_labels, test_predictions, classes),
                      incumbent_test=metrics(test_labels, incumbent_predictions, classes),
                      limitations=["Image/capture-group evaluation; no physical specimen identity or field validation.",
                                   "Legacy incumbent training membership is unknown.",
                                   "No physical SMALL/BIG accuracy was measured."])
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    print(json.dumps({"test_accuracy": experiment["test"]["accuracy"], "incumbent_test_accuracy": experiment["incumbent_test"]["accuracy"],
                      "duration_seconds": experiment["duration_seconds"], "confidence_floor": confidence_floor}), flush=True)


if __name__ == "__main__":
    main()
