"""Batch-mode validation on composites of real photos.

Cuts a real vegetable out of a clean single-item photo (using the engine's
own mask) and composites three scaled copies onto a plain mat, so the true
size ordering is known exactly.  Checks, per class:
  - the vegetable is named correctly,
  - every item receives a grade,
  - measured size is monotonic in the composite scale.
Run after a species model promotion.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

BASE = ROOT / "data/external/j33g3nsm9k_v1/images/Vegetables Image Dataset for Machine Applications/Vegetables Images/Vegetables"
CLASSES = ["Okra", "Potato", "Tomato", "Cucumber", "Bean", "Chili"]
EXPECT_KEY = {"Okra": "okra", "Potato": "potato", "Tomato": "tomato",
              "Cucumber": "cucumber", "Bean": "green_bean", "Chili": "green_chilli"}
SCALES = (1.0, 0.78, 0.58)


def cutout(service, path: Path):
    """Return an RGBA cutout of the largest cleanly-segmented instance."""
    from growize_vision.inference import InferenceRequest
    from growize_vision.media import load_image_file

    result = service.inspect_frame(
        load_image_file(str(path)).bgr,
        request=InferenceRequest(crop_key="auto", allow_scale_estimate=False),
    )
    if len(result.instances) != 1:
        return None
    inst = result.instances[0]
    if inst.decision is None or inst.decision.status.value in ("OCCLUDED", "OUT_OF_DISTRIBUTION"):
        return None
    bgr = load_image_file(str(path)).bgr
    mask = np.array(inst.mask, dtype=np.uint8)
    if mask.shape[:2] != bgr.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask > 0)
    if len(xs) < 500:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if x0 < 4 or y0 < 4 or x1 > bgr.shape[1] - 4 or y1 > bgr.shape[0] - 4:
        return None  # touches frame edge; unsafe cutout
    rgba = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask[y0:y1, x0:x1]
    return rgba


def compose(items: list[np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(3)
    canvas = np.full((640, 1100, 3), 170, np.uint8)
    canvas = np.clip(
        canvas.astype(int) + rng.integers(-6, 7, canvas.shape), 0, 255
    ).astype(np.uint8)
    xs = [40, 400, 760]
    for rgba, x in zip(items, xs):
        h, w = rgba.shape[:2]
        alpha = (rgba[:, :, 3:4].astype(float) / 255.0)
        y = (canvas.shape[0] - h) // 2
        region = canvas[y:y + h, x:x + w].astype(float)
        canvas[y:y + h, x:x + w] = (
            rgba[:, :, :3].astype(float) * alpha + region * (1 - alpha)
        ).astype(np.uint8)
    return canvas


def main() -> None:
    from apps.gradio_app import AUTO_CROP_CHOICE, inspect_image_ui
    from growize_vision.inference import InferenceService

    service = InferenceService()
    random.seed(11)
    report = {}
    for cls in CLASSES:
        files = sorted(p for p in BASE.rglob(f"{cls}/*.jpg")) or sorted(
            p for p in BASE.rglob(f"{cls}/**/*.jpg")
        )
        random.shuffle(files)
        rgba = None
        for p in files[:40]:
            rgba = cutout(service, p)
            if rgba is not None:
                break
        if rgba is None:
            report[cls] = "NO CLEAN CUTOUT"
            continue
        h, w = rgba.shape[:2]
        target_h = 340
        items = []
        for s in SCALES:
            th = int(target_h * s)
            tw = max(1, int(w * th / h))
            items.append(cv2.resize(rgba, (tw, th), interpolation=cv2.INTER_AREA))
        canvas = compose(items)
        tmp = ROOT / "state" / f"_composite_{cls}.png"
        cv2.imwrite(str(tmp), canvas)
        img, out = inspect_image_ui(str(tmp), AUTO_CROP_CHOICE, None, None, None, None, False)
        text = re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", out))
        # per-item measurements via the engine
        r = service.inspect_frame(canvas, request=None)
        sizes = [
            i.measurement.value_mm
            for i in r.instances
            if i.measurement is not None
        ]
        named = EXPECT_KEY[cls] in text[:120].lower()
        graded = sum(1 for i in r.instances if i.decision and i.decision.accepted)
        mono = all(
            sizes[i] >= sizes[i + 1] - 1e-6 for i in range(len(sizes) - 1)
        ) if len(sizes) == 3 else False
        report[cls] = {
            "title": text[:110],
            "named": named,
            "graded_items": graded,
            "sizes_mm": [round(s, 1) for s in sizes],
            "monotonic": mono,
        }
        print(f"{cls:9s} named={named} graded={graded}/3 mono={mono} sizes={[round(s,1) for s in sizes]}")
        print("   ", text[:110])
    out_path = ROOT / "reports" / "species_training_v6" / "batch_composite_validation.json"
    import json

    out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("saved", out_path)


if __name__ == "__main__":
    main()
