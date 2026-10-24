"""Lean species retrain (v6): streaming features + SGD logistic head.

The full train_species_v2.py pipeline loads every embedding into RAM at once
and OOMs on this machine when the manifest grows past ~20k images.  This
trainer keeps the same evidence rules (manifest provenance already verified
by the v6 manifest build) but streams: images are embedded in batches and
the head is fit with partial_fit, so memory stays flat.  The artifact format
matches the incumbent exactly so CropIdentityService can load it unchanged.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUN = ROOT / "reports" / "species_training_v6"
CANDIDATE = RUN / "candidate"
BATCH = 32
SEED = 42


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import torch
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score
    from growize_vision.crop_identity import CropIdentityService

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()

    rows = [
        json.loads(line)
        for line in (RUN / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["split"] != "quarantine"
    ]
    splits = defaultdict(list)
    for row in rows:
        splits[row["split"]].append(row)
    classes = sorted({r["label"] for r in splits["train"]})
    print("loading backbone...", flush=True)
    print(f"rows: train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])} classes={len(classes)}", flush=True)

    service = CropIdentityService()
    if not service.is_deep:
        raise SystemExit("backbone unavailable")
    backbone_path = ROOT / "config" / "models" / "mobilenetv3_backbone.pt"
    backbone_sha = digest(backbone_path)

    def embed(rows_batch):
        from PIL import Image
        import cv2

        tensors = []
        for r in rows_batch:
            img = cv2.imread(str(ROOT / r["path"]))
            if img is None:
                raise SystemExit(f"unreadable image: {r['path']}")
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensors.append(service.deep_transform(pil))
        stacked = torch.stack(tensors)
        with torch.inference_mode():
            return service.backbone(stacked).cpu().numpy().astype(np.float32)

    def stream(split):
        batch_rows = []
        for r in splits[split]:
            batch_rows.append(r)
            if len(batch_rows) == BATCH:
                yield batch_rows, embed(batch_rows)
                batch_rows = []
        if batch_rows:
            yield batch_rows, embed(batch_rows)

    # First pass: streaming feature statistics so the head sees standardized
    # inputs (raw mobilenet magnitudes overflow a float32 logistic head).
    n_seen = 0
    f_sum = None
    f_sqsum = None
    for batch_rows, feats in stream("train"):
        f64 = feats.astype(np.float64)
        if f_sum is None:
            f_sum = np.zeros(f64.shape[1]); f_sqsum = np.zeros(f64.shape[1])
        f_sum += f64.sum(axis=0)
        f_sqsum += (f64 ** 2).sum(axis=0)
        n_seen += len(f64)
    mean = f_sum / n_seen
    var = np.maximum(f_sqsum / n_seen - mean ** 2, 1e-12)
    std = np.sqrt(var)
    print(f"feature stats over {n_seen} rows (std median {np.median(std):.3f})", flush=True)

    def standardize(feats):
        return ((feats.astype(np.float64) - mean) / std).astype(np.float32)

    model = SGDClassifier(
        loss="log_loss", alpha=1e-5, max_iter=1, tol=None, random_state=SEED
    )
    n_iter = 5
    for epoch in range(n_iter):
        order = list(splits["train"])
        rng.shuffle(order)
        splits["train"] = order
        for batch_rows, feats in stream("train"):
            labels = [r["label"] for r in batch_rows]
            model.partial_fit(standardize(feats), labels, classes=classes)
        print(f"epoch {epoch + 1}/{n_iter} done at {time.perf_counter() - started:.0f}s", flush=True)

    def evaluate(split):
        y_true, y_pred = [], []
        for batch_rows, feats in stream(split):
            y_true.extend(r["label"] for r in batch_rows)
            y_pred.extend(model.predict(standardize(feats)))
        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        f1 = float(f1_score(y_true, y_pred, labels=classes, average="macro"))
        per = defaultdict(lambda: [0, 0])
        for t, p in zip(y_true, y_pred):
            per[t][1] += 1
            per[t][0] += t == p
        return acc, f1, {c: (ok, n) for c, (ok, n) in per.items()}

    val_acc, val_f1, val_per = evaluate("validation")
    test_acc, test_f1, test_per = evaluate("test")
    print(f"validation acc={val_acc:.4f} macroF1={val_f1:.4f}", flush=True)
    print(f"test       acc={test_acc:.4f} macroF1={test_f1:.4f}", flush=True)
    for c in classes:
        ok, n = test_per.get(c, (0, 0))
        print(f"  test {c:12s} {ok}/{n} = {ok / max(n, 1):.0%}", flush=True)

    CANDIDATE.mkdir(parents=True, exist_ok=True)
    from growize_vision.crop_identity import StandardizedHead

    artifact = {
        "backbone_name": "mobilenetv3_large_100",
        "classifier": StandardizedHead(model, mean, std),
        "classes": classes,
        "confidence_floor": 0.975,
        "backbone_file": backbone_path.name,
        "backbone_sha256": backbone_sha,
        "manifest_sha256": digest(RUN / "manifest.jsonl"),
        "training_script_sha256": digest(Path(__file__)),
        "validation_macro_f1": val_f1,
        "format_version": 2,
        "trainer": "train_species_v6_lean (streaming SGD, 12 epochs)",
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
    }
    out = CANDIDATE / "species_classifier_v6.pkl"
    import pickle

    with open(out, "wb") as f:
        pickle.dump(artifact, f)
    with open(CANDIDATE / "lean_experiment.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "trainer": "lean-streaming-sgd",
                "rows": {k: len(v) for k, v in splits.items()},
                "classes": classes,
                "epochs": n_iter,
                "validation_accuracy": val_acc,
                "validation_macro_f1": val_f1,
                "test_accuracy": test_acc,
                "test_macro_f1": test_f1,
                "test_per_class": {c: list(v) for c, v in test_per.items()},
                "elapsed_s": time.perf_counter() - started,
            },
            f,
            indent=1,
        )
    print(f"saved {out} in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
