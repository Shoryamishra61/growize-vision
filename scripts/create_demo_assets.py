#!/usr/bin/env python
"""Generate explicitly synthetic fixtures for geometry and pipeline tests.

These are vector-defined shapes rendered into a controlled-mat scene.  They are
useful for catching unit/parallax/segmentation bugs, but they are NOT evidence of
real-produce accuracy and must never be counted as measured specimens.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "data" / "demo"
ASSET_DIR = ROOT / "assets"
PIXELS_PER_MM = 6.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matte(h: int, w: int, base: int = 172) -> np.ndarray:
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    image = np.full((h, w, 3), base, dtype=np.float32)
    image += ((x - 0.5) * 5.0 + (y - 0.5) * 3.0)[:, :, None]
    rng = np.random.default_rng(20261022)
    image += rng.normal(0, 1.7, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    # Very light, thin matte seams: not produce, but harmless texture.
    for x_mm in (25, 210):
        p = int(x_mm * PIXELS_PER_MM)
        if p < w:
            cv2.line(image, (p, 0), (p, h), (165, 165, 165), 1)
    return image


def _dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _paste_marker(image: np.ndarray, x: int, y: int, marker_id: int = 0, side_px: int = 300) -> None:
    marker = cv2.aruco.generateImageMarker(_dictionary(), marker_id, side_px)
    # Ensure the marker has a quiet zone that matches the matte background.
    image[y : y + side_px, x : x + side_px] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)


def _blob_contour(
    cx: float,
    cy: float,
    width_mm: float,
    height_mm: float,
    angle_deg: float,
    seed: int,
    irregularity: float = 0.035,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = 96
    t = np.linspace(0, 2 * math.pi, count, endpoint=False)
    rx = width_mm / 2.0 * PIXELS_PER_MM
    ry = height_mm / 2.0 * PIXELS_PER_MM
    radial = 1.0 + irregularity * (
        0.65 * np.sin(3 * t + rng.uniform(0, 3)) + 0.35 * np.sin(5 * t + rng.uniform(0, 3))
    )
    x = rx * radial * np.cos(t)
    y = ry * radial * np.sin(t)
    theta = math.radians(angle_deg)
    xr = x * math.cos(theta) - y * math.sin(theta)
    yr = x * math.sin(theta) + y * math.cos(theta)
    return np.stack([xr + cx, yr + cy], axis=1).astype(np.float32)


def _render_object(
    image: np.ndarray,
    points: np.ndarray,
    colour: tuple[int, int, int],
    seed: int,
) -> None:
    poly = points.astype(np.int32)
    cv2.fillPoly(image, [poly.reshape(-1, 1, 2)], colour)
    # Crop-like texture, kept inside the silhouette.
    rng = np.random.default_rng(seed)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)
    texture = np.zeros_like(image)
    for _ in range(55):
        x = int(np.clip(rng.uniform(poly[:, 0].min(), poly[:, 0].max()), 0, image.shape[1] - 1))
        y = int(np.clip(rng.uniform(poly[:, 1].min(), poly[:, 1].max()), 0, image.shape[0] - 1))
        if mask[y, x] == 0:
            continue
        r = int(rng.integers(1, max(2, int(1.4 * PIXELS_PER_MM))))
        shade = tuple(int(np.clip(c + rng.normal(0, 16), 0, 255)) for c in colour)
        cv2.circle(texture, (x, y), r, shade, -1, cv2.LINE_AA)
    mask3 = (mask > 0).astype(np.float32)[:, :, None]
    image[:] = np.clip(
        image.astype(np.float32) * (1 - 0.18 * mask3)
        + texture.astype(np.float32) * (0.18 * mask3),
        0,
        255,
    ).astype(np.uint8)
    cv2.polylines(image, [poly.reshape(-1, 1, 2)], True, (35, 55, 55), 1, cv2.LINE_AA)


def _actual_metric(points: np.ndarray, metric: str) -> float:
    mm = points.astype(np.float32) / PIXELS_PER_MM
    _center, (w, h), _angle = cv2.minAreaRect(mm.reshape(-1, 1, 2))
    short, long = min(w, h), max(w, h)
    if metric == "min_area_rect_short_mm":
        return float(short)
    if metric == "min_area_rect_long_mm":
        return float(long)
    raise ValueError(metric)


def _base_scene(with_marker: bool = True) -> np.ndarray:
    image = _matte(1000, 1400)
    if with_marker:
        _paste_marker(image, 90, 80, marker_id=0, side_px=300)
        # A faint 100 mm check line is printed on the mat, not part of grading.
        y = 470
        x0, x1 = 100, int(100 + 100 * PIXELS_PER_MM)
        cv2.line(image, (x0, y), (x1, y), (150, 150, 150), 1)
        for tick in range(0, 101, 25):
            x = int(x0 + tick * PIXELS_PER_MM)
            cv2.line(image, (x, y - 5), (x, y + 5), (150, 150, 150), 1)
    return image


def _save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _record(
    fixture_id: str,
    path: Path,
    *,
    crop: str,
    metric: str,
    actual_mm: float | None,
    expected: str,
    threshold_mm: float,
    synthetic: bool = True,
    notes: str = "",
) -> dict:
    return {
        "fixture_id": fixture_id,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": _sha256(path),
        "synthetic": synthetic,
        "crop": crop,
        "physical_unit": "single_tuber" if crop == "potato" else "single_pod",
        "metric": metric,
        "actual_metric_mm": round(actual_mm, 3) if actual_mm is not None else None,
        "expected_demonstration_class": expected,
        "demo_threshold_mm": threshold_mm,
        "marker_side_mm": 50.0,
        "pixels_per_mm": PIXELS_PER_MM,
        "notes": notes,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # Small and big potato-like blobs, both on the calibrated matte plane.
    for name, width, height, angle, seed in (
        ("small", 45.0, 35.0, 12.0, 11),
        ("big", 92.0, 72.0, -18.0, 22),
    ):
        image = _base_scene()
        points = _blob_contour(880, 590, width, height, angle, seed)
        _render_object(image, points, (63, 104, 148), seed)
        path = OUT_DIR / f"potato_{name}_synthetic.png"
        _save(path, image)
        actual = _actual_metric(points, "min_area_rect_short_mm")
        records.append(
            _record(
                f"potato_{name}_synthetic",
                path,
                crop="potato",
                metric="min_area_rect_short_mm",
                actual_mm=actual,
                expected="SMALL" if actual < 60.0 else "BIG",
                threshold_mm=60.0,
                notes="Synthetic blob; geometry ground truth is vector-defined. NOT real produce.",
            )
        )

    # Okra-like elongated pods.
    for name, width, height, angle, seed in (
        ("small", 82.0, 24.0, 24.0, 33),
        ("big", 142.0, 30.0, -22.0, 44),
    ):
        image = _base_scene()
        points = _blob_contour(870, 590, width, height, angle, seed, irregularity=0.02)
        _render_object(image, points, (51, 113, 70), seed)
        path = OUT_DIR / f"okra_{name}_synthetic.png"
        _save(path, image)
        actual = _actual_metric(points, "min_area_rect_long_mm")
        records.append(
            _record(
                f"okra_{name}_synthetic",
                path,
                crop="okra",
                metric="min_area_rect_long_mm",
                actual_mm=actual,
                expected="SMALL" if actual < 100.0 else "BIG",
                threshold_mm=100.0,
                notes="Synthetic pod; straight-axis length. NOT real produce.",
            )
        )

    # Failure fixtures.
    image = _base_scene(with_marker=False)
    points = _blob_contour(880, 590, 60.0, 48.0, 8.0, 55)
    _render_object(image, points, (63, 104, 148), 55)
    path = OUT_DIR / "no_marker_synthetic.png"
    _save(path, image)
    records.append(
        _record(
            "no_marker_synthetic",
            path,
            crop="potato",
            metric="min_area_rect_short_mm",
            actual_mm=None,
            expected="CALIBRATION_REQUIRED",
            threshold_mm=60.0,
            notes="Same synthetic object without a marker; must never produce mm/class.",
        )
    )

    image = _base_scene()
    path = OUT_DIR / "empty_mat_synthetic.png"
    _save(path, image)
    records.append(
        _record(
            "empty_mat_synthetic",
            path,
            crop="potato",
            metric="min_area_rect_short_mm",
            actual_mm=None,
            expected="NO_OBJECT",
            threshold_mm=60.0,
            notes="Marker-only empty matte scene.",
        )
    )

    image = _base_scene()
    points = _blob_contour(880, 590, 92.0, 72.0, -18.0, 22)
    _render_object(image, points, (63, 104, 148), 22)
    blurred = cv2.GaussianBlur(image, (0, 0), 7.5)
    path = OUT_DIR / "blurred_synthetic.png"
    _save(path, blurred)
    records.append(
        _record(
            "blurred_synthetic",
            path,
            crop="potato",
            metric="min_area_rect_short_mm",
            actual_mm=_actual_metric(points, "min_area_rect_short_mm"),
            expected="LOW_IMAGE_QUALITY",
            threshold_mm=60.0,
            notes="Synthetic blur stress case; no accepted grade expected.",
        )
    )

    image = _base_scene()
    points = _blob_contour(1300, 590, 92.0, 72.0, -10.0, 22)
    _render_object(image, points, (63, 104, 148), 22)
    path = OUT_DIR / "edge_clipped_synthetic.png"
    _save(path, image)
    records.append(
        _record(
            "edge_clipped_synthetic",
            path,
            crop="potato",
            metric="min_area_rect_short_mm",
            actual_mm=None,
            expected="OCCLUDED",
            threshold_mm=60.0,
            notes="Synthetic object deliberately clipped by frame edge.",
        )
    )

    # Short MP4 for the video path.  One synthetic object over time; the
    # video report must say per-frame observations, not unique vegetables.
    video_path = OUT_DIR / "synthetic_potato_video.mp4"
    base = _base_scene()
    points = _blob_contour(880, 590, 78.0, 64.0, -12.0, 66)
    _render_object(base, points, (63, 104, 148), 66)
    writer = None
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (base.shape[1], base.shape[0]))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        raise RuntimeError("Could not create the synthetic MP4; no tested codec available.")
    for frame_index in range(30):
        shift = int(8 * math.sin(frame_index / 5.0))
        frame = base.copy()
        # Translate a soft object layer by redrawing the same blob.
        layer = np.full_like(base, 172)
        shifted = points.copy()
        shifted[:, 0] += shift
        _render_object(layer, shifted, (63, 104, 148), 66)
        mask = np.zeros(base.shape[:2], np.uint8)
        cv2.fillPoly(mask, [shifted.astype(np.int32).reshape(-1, 1, 2)], 255)
        frame[mask > 0] = layer[mask > 0]
        writer.write(frame)
    writer.release()
    records.append(
        {
            "fixture_id": "synthetic_potato_video",
            "path": str(video_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(video_path),
            "synthetic": True,
            "crop": "potato",
            "physical_unit": "single_tuber",
            "metric": "min_area_rect_short_mm",
            "actual_metric_mm": round(_actual_metric(points, "min_area_rect_short_mm"), 3),
            "expected_demonstration_class": "BIG",
            "demo_threshold_mm": 60.0,
            "marker_side_mm": 50.0,
            "pixels_per_mm": PIXELS_PER_MM,
            "frames": 30,
            "fps": 10.0,
            "notes": "Synthetic video; repeated observations of one synthetic object. NOT unique produce.",
        }
    )

    manifest = {
        "version": "synthetic-demo-2026-10-22.1",
        "warning": (
            "All fixtures in this manifest are SYNTHETIC. They verify geometry, "
            "failure handling and media plumbing; they are not real-produce "
            "accuracy evidence and must not be used for production claims."
        ),
        "threshold_source": "assessment demo cut-offs only; not ARL approved",
        "fixtures": records,
    }
    manifest_path = OUT_DIR / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} synthetic fixtures to {OUT_DIR}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
