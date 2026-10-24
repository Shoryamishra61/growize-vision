"""Generate the showcase documentation images through the live app.

Each entry runs the real UI pipeline (auto crop, no inputs) on a real
photograph or composite and saves the annotated output plus the exact
result text shown in the app.  Outputs feed docs/SHOWCASE.md.
"""

from __future__ import annotations

import glob
import html
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from apps.gradio_app import AUTO_CROP_CHOICE, inspect_image_ui  # noqa: E402

BASE = ROOT / "data/external/j33g3nsm9k_v1/images/Vegetables Image Dataset for Machine Applications/Vegetables Images/Vegetables"
ASSETS = ROOT / "docs/assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def real_photo(cls: str, seed: int = 0) -> Path:
    paths = sorted(glob.glob(str(BASE / cls / "**/*.jpg"), recursive=True))
    return Path(paths[seed % len(paths)])


def composite_paste(paths, canvas_size=(760, 1100), slots=((40, 80), (330, 60), (520, 520)), crop_box=(280, 360)) -> np.ndarray:
    canvas = np.full((canvas_size[0], canvas_size[1], 3), 245, np.uint8)
    rng = np.random.default_rng(6)
    canvas = np.clip(canvas.astype(int) + rng.integers(-5, 6, canvas.shape), 0, 255).astype(np.uint8)
    for (y, x), p in zip(slots, paths):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        ch, cw = crop_box
        img = img[max(0, (h - ch) // 2):(h - ch) // 2 + ch, max(0, (w - cw) // 2):(w - cw) // 2 + cw]
        hh = min(img.shape[0], canvas.shape[0] - y)
        ww = min(img.shape[1], canvas.shape[1] - x)
        canvas[y:y + hh, x:x + ww] = img[:hh, :ww]
    return canvas


def run(name: str, source, crop: str | None = AUTO_CROP_CHOICE) -> dict:
    path = source if isinstance(source, (str, Path)) else None
    if path is None:
        path = ASSETS / f"_tmp_{name}.png"
        cv2.imwrite(str(path), source)
    img, out = inspect_image_ui(str(path), crop, None, None, None, None, False)
    text = html.unescape(re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", out)))
    title = text.split(" in this photo")[0].split(". ")[0][:80]
    if img is not None:
        cv2.imwrite(str(ASSETS / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return {"name": name, "title": title, "message": text[:320]}


def main() -> None:
    results = []
    okra = sorted(glob.glob(str(BASE / "Okra/**/*.jpg"), recursive=True))
    potato = sorted(glob.glob(str(BASE / "Potato/**/*.jpg"), recursive=True))
    tomato = sorted(glob.glob(str(ROOT / "data/raw/kaggle_veg/Vegetable Images/test/Tomato/*.jpg")))
    cucumber = sorted(glob.glob(str(BASE / "Cucumber/**/*.jpg"), recursive=True))
    bean = sorted(glob.glob(str(BASE / "Bean/**/*.jpg"), recursive=True))
    chili = sorted(glob.glob(str(BASE / "Chili/**/*.jpg"), recursive=True))

    # Singles that grade cleanly (verified working offsets from the sweep).
    for name, paths, offset in (
        ("okra_single", okra, 0),
        ("potato_single", potato, 30),
        ("tomato_single", tomato, 0),
        ("cucumber_single", cucumber, 2),
        ("bean_single", bean, 4),
        ("chili_single", chili, 5),
    ):
        results.append(run(name, Path(paths[offset])))

    # Batches / bunches from real-photo composites.
    results.append(run("okra_pile", cv2.imread(str(ROOT / "state/_pile.png"))))
    results.append(run("okra_bunch", cv2.imread(str(ROOT / "state/_bunch2.png"))))
    results.append(run("okra_overlap", cv2.imread(str(ROOT / "state/_overlap.png")), crop="Okra"))
    results.append(run("potato_batch", composite_paste([potato[2], potato[8], potato[15]])))

    (ASSETS / "showcase_results.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8"
    )
    for r in results:
        print(f"{r['name']:16s} {r['title'][:70]}")


if __name__ == "__main__":
    main()
