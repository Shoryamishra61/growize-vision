"""Interpretable controlled-background foreground segmentation.

This is intentionally a baseline, not a universal vegetable detector.  It works
on a plain matte tray with a visible fiducial.  It refuses or flags backgrounds
that are too busy, objects touching the frame edge, and ambiguous merged masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import SegmentationSettings
from .domain import Calibration, MarkerDetection
from .markers import marker_mask


@dataclass
class SegmentedObject:
    mask: np.ndarray = field(repr=False)
    contour_px: np.ndarray = field(repr=False)
    area_px: float
    bbox_px: tuple[int, int, int, int]
    touches_edge: bool = False
    touches_marker: bool = False
    solidity: float = 1.0
    holes_px: float = 0.0
    flags: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict:
        return {
            "area_px": round(self.area_px, 1),
            "bbox_px": list(self.bbox_px),
            "touches_edge": self.touches_edge,
            "touches_marker": self.touches_marker,
            "solidity": round(self.solidity, 4),
            "holes_px": round(self.holes_px, 1),
            "flags": list(self.flags),
        }


@dataclass
class SegmentationResult:
    mask: np.ndarray = field(repr=False)
    objects: list[SegmentedObject] = field(default_factory=list)
    background_busy: bool = False
    color_distance_threshold: float = 0.0
    background_lab: tuple[float, float, float] = (0.0, 0.0, 0.0)
    border_delta_p95: float = 0.0
    ensemble_iou: float | None = None
    warnings: list[str] = field(default_factory=list)


def _corner_mask(shape: tuple[int, int], fraction: float = 0.16) -> np.ndarray:
    """Sample matte background from corner patches.

    Looking at the four corners, rather than the whole frame border, reduces
    contamination when a clipped object touches one edge.  A plain inspection
    station still supplies clean corners; an arbitrary farm/office scene does
    not, and is flagged as out of distribution.
    """
    h, w = shape[:2]
    patch_h = max(8, int(round(h * fraction)))
    patch_w = max(8, int(round(w * fraction)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:patch_h, :patch_w] = True
    mask[:patch_h, -patch_w:] = True
    mask[-patch_h:, :patch_w] = True
    mask[-patch_h:, -patch_w:] = True
    return mask


def _background_model(
    lab: np.ndarray, marker: MarkerDetection | None, settings: SegmentationSettings
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    h, w = lab.shape[:2]
    corner = _corner_mask((h, w))
    if marker is not None:
        excluded = marker_mask((h, w, 3), marker, settings.marker_exclusion_margin_px)
        corner &= excluded == 0
    pixels = lab[corner]
    if pixels.shape[0] < 100:
        # Fallback: all pixels except the marker.  This is less stable, so the
        # caller receives a busy/out-of-distribution warning.
        fallback = np.ones((h, w), dtype=bool)
        if marker is not None:
            fallback &= (
                marker_mask((h, w, 3), marker, settings.marker_exclusion_margin_px) == 0
            )
        pixels = lab[fallback]
    background = np.median(pixels, axis=0).astype(np.float32)
    distances = np.sqrt(np.sum((pixels.astype(np.float32) - background) ** 2, axis=1))
    p95 = float(np.percentile(distances, 95))
    # A plain mat is colour-consistent.  The generous limits avoid rejecting a
    # little texture or shadow while catching arbitrary farm/office scenes.
    p99 = float(np.percentile(distances, 99))
    busy = p99 > 70.0
    return background, pixels.astype(np.float32), busy, p95


def _shadow_suppression(
    lab: np.ndarray,
    background_lab: np.ndarray,
    candidate: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """Remove soft neutral shadows while preserving chromatically distinct crop.

    This is deliberately conservative.  A black object on a grey mat may still
    need a different capture configuration; the pipeline reports that instead
    of silently accepting a shadow as produce.
    """
    delta_l = lab[:, :, 0].astype(np.float32) - float(background_lab[0])
    delta_a = lab[:, :, 1].astype(np.float32) - float(background_lab[1])
    delta_b = lab[:, :, 2].astype(np.float32) - float(background_lab[2])
    chroma = np.sqrt(delta_a * delta_a + delta_b * delta_b)
    darker = delta_l < -4.0
    neutral = chroma < max(4.0, 0.8 * penalty)
    shadow = candidate & darker & neutral
    return candidate & ~shadow


def _chroma_candidate(image_bgr: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Markerless chromatic-energy Otsu mask used as the independent opinion."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
    energy = np.clip(0.5 * sat + 0.5 * chroma, 0, 255).astype(np.uint8)
    _, candidate = cv2.threshold(energy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return candidate


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_set = a > 0
    b_set = b > 0
    union = int(np.logical_or(a_set, b_set).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a_set, b_set).sum()) / union


def segment_foreground(
    image_bgr: np.ndarray,
    calibration: Calibration | None,
    marker: MarkerDetection | None,
    settings: SegmentationSettings,
    *,
    min_area_mm2: float,
    min_area_px: float,
) -> SegmentationResult:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot segment an empty frame.")
    lab_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lab = lab_u8.astype(np.float32)
    background_lab, _border_pixels, busy, p95 = _background_model(
        lab, marker, settings
    )
    threshold = max(settings.color_distance_min, p95 + settings.adaptive_margin)
    ensemble_iou: float | None = None
    masks_disagree = False
    if marker is None or busy:
        # In-the-wild produce segmentation:
        # Use chromatic energy & saturation so real produce is detected cleanly
        # across variable table, wood, cutting board, and ambient backgrounds.
        candidate = _chroma_candidate(image_bgr, lab)
    else:
        delta = lab - background_lab.reshape(1, 1, 3)
        distance = np.sqrt(np.sum(delta * delta, axis=2))
        candidate = (distance > threshold).astype(np.uint8) * 255
        candidate = _shadow_suppression(
            lab, background_lab, candidate > 0, settings.shadow_value_only_penalty
        ).astype(np.uint8) * 255
        if settings.ensemble_min_iou > 0.0:
            # Blueprint mask-quality gate: the two independent methods must
            # agree; disagreement means lighting/background fooled one of
            # them, and the measurement must not trust either silently.
            k = max(3, int(settings.morph_kernel))
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            alternate = _chroma_candidate(image_bgr, lab)
            alternate = cv2.morphologyEx(alternate, cv2.MORPH_OPEN, kernel, iterations=1)
            alternate = cv2.morphologyEx(alternate, cv2.MORPH_CLOSE, kernel, iterations=2)
            comparison = candidate.copy()
            if marker is not None:
                excluded = marker_mask(
                    image_bgr.shape, marker, settings.marker_exclusion_margin_px
                )
                alternate[excluded > 0] = 0
                comparison[excluded > 0] = 0
            ensemble_iou = _mask_iou(comparison, alternate)
            masks_disagree = ensemble_iou < settings.ensemble_min_iou

    if marker is not None:
        excluded = marker_mask(
            image_bgr.shape, marker, settings.marker_exclusion_margin_px
        )
        candidate[excluded > 0] = 0

    k = max(3, int(settings.morph_kernel))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Fill only holes that are not connected to the image boundary.  This keeps
    # genuine background openings while repairing small specular holes.
    flood = candidate.copy()
    ff_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(candidate, holes)

    min_component_ratio = 0.015 if (marker is None or busy) else float(settings.min_component_fraction)
    min_area = max(
        min_component_ratio * image_bgr.shape[0] * image_bgr.shape[1],
        float(min_area_px),
    )
    if calibration and calibration.valid and calibration.mm_per_px:
        mm2_to_px = 1.0 / max(calibration.mm_per_px**2, 1e-9)
        min_area = max(
            min_component_ratio * image_bgr.shape[0] * image_bgr.shape[1],
            float(min_area_px),
            float(min_area_mm2) * mm2_to_px,
        )

    contours, hierarchy = cv2.findContours(
        filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    objects: list[SegmentedObject] = []
    warnings_list: list[str] = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
    for index, contour in enumerate(contours):
        area = float(abs(cv2.contourArea(contour)))
        if area < min_area:
            continue
        parent = hierarchy[index][3] if hierarchy is not None else -1
        if parent != -1:
            # A child contour is a hole; count holes against the parent below.
            continue
        x, y, w, h = cv2.boundingRect(contour)
        component_mask = np.zeros(image_bgr.shape[:2], np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, -1)
        holes_px = float(
            max(0.0, cv2.contourArea(cv2.convexHull(contour)) - area)
        )
        hull = cv2.convexHull(contour)
        hull_area = float(abs(cv2.contourArea(hull))) or 1.0
        solidity = area / hull_area
        touches_edge = bool(
            x <= 2
            or y <= 2
            or x + w >= image_bgr.shape[1] - 2
            or y + h >= image_bgr.shape[0] - 2
        )
        touches_marker = False
        if marker is not None:
            mx, my, mw, mh = cv2.boundingRect(
                np.asarray(marker.corners_px, dtype=np.int32).reshape(-1, 1, 2)
            )
            margin = settings.marker_exclusion_margin_px
            inter_x = max(0, min(x + w, mx + mw + margin) - max(x, mx - margin))
            inter_y = max(0, min(y + h, my + mh + margin) - max(y, my - margin))
            touches_marker = inter_x > 0 and inter_y > 0
        flags: list[str] = []
        if touches_edge:
            flags.append("TOUCHES_FRAME_EDGE")
        if touches_marker:
            flags.append("MASK_NEAR_MARKER")
        if solidity < 0.82:
            flags.append("LOW_SOLIDITY_POSSIBLY_MERGED")
        if holes_px > 0.08 * max(area, 1.0):
            flags.append("SIGNIFICANT_MASK_CONCAVITY")
        if masks_disagree:
            flags.append("SEGMENTATION_DISAGREEMENT")
        objects.append(
            SegmentedObject(
                mask=component_mask,
                contour_px=contour,
                area_px=area,
                bbox_px=(int(x), int(y), int(w), int(h)),
                touches_edge=touches_edge,
                touches_marker=touches_marker,
                solidity=solidity,
                holes_px=holes_px,
                flags=flags,
            )
        )
    objects.sort(key=lambda obj: obj.area_px, reverse=True)
    if len(objects) > settings.max_instances:
        warnings_list.append(
            f"{len(objects)} components found; only the largest "
            f"{settings.max_instances} are retained."
        )
        objects = objects[: settings.max_instances]
    if not objects:
        warnings_list.append(
            "No foreground component passed the area threshold. Use a clean "
            "matte mat with a visible marker and place the vegetable fully in frame."
        )
    if busy:
        warnings_list.append(
            "The border colour is inconsistent. A plain matte tray is required "
            "for physical measurement."
        )
    if masks_disagree and ensemble_iou is not None:
        warnings_list.append(
            f"SEGMENTATION_DISAGREEMENT: background-difference and chroma masks "
            f"agree on only {ensemble_iou:.2f} IoU (minimum "
            f"{settings.ensemble_min_iou:.2f}). Lighting or background may be "
            "non-standard; no measurement is trusted from either mask alone."
        )

    union_mask = np.zeros(image_bgr.shape[:2], np.uint8)
    for obj in objects:
        union_mask[obj.mask > 0] = 255
    return SegmentationResult(
        mask=union_mask,
        objects=objects,
        background_busy=busy,
        color_distance_threshold=float(threshold),
        background_lab=tuple(float(v) for v in background_lab),
        border_delta_p95=float(p95),
        ensemble_iou=ensemble_iou,
        warnings=warnings_list,
    )

def _rebuild_object(part_mask, image_shape, inherited_flags):
    """Build a SegmentedObject from a binary part mask."""
    contours, _ = cv2.findContours(
        part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    x, y, w, h = cv2.boundingRect(contour)
    hull_area = float(abs(cv2.contourArea(cv2.convexHull(contour)))) or 1.0
    touches_edge = bool(
        x <= 2 or y <= 2
        or x + w >= image_shape[1] - 2
        or y + h >= image_shape[0] - 2
    )
    flags = list(inherited_flags)
    if touches_edge and "TOUCHES_FRAME_EDGE" not in flags:
        flags.append("TOUCHES_FRAME_EDGE")
    if "SPLIT_FROM_MERGED_MASK" not in flags:
        flags.append("SPLIT_FROM_MERGED_MASK")
    return SegmentedObject(
        mask=part_mask,
        contour_px=contour,
        area_px=area,
        bbox_px=(int(x), int(y), int(w), int(h)),
        touches_edge=touches_edge,
        solidity=area / hull_area,
        holes_px=0.0,
        flags=flags,
    )


def split_touching_objects(objects, image_bgr, min_area_px):
    """Split merged masks of touching single-unit produce (watershed).

    Piles and rows of produce segment as one mass; a distance-transform
    watershed recovers the individual items.  Conservative by design: an
    object is only split when its mask is flagged as possibly merged or it
    covers a large share of the frame, and the split is kept only when
    every retained part is still a substantial share of the original.
    Returns the new object list, or None when nothing was split.
    """
    if not objects:
        return None
    frame_area = float(image_bgr.shape[0] * image_bgr.shape[1])
    changed = False
    result = []
    for obj in objects:
        suspicious = (
            "LOW_SOLIDITY_POSSIBLY_MERGED" in obj.flags
            or obj.area_px > 0.15 * frame_area
        )
        if not suspicious:
            result.append(obj)
            continue
        contour = np.asarray(obj.contour_px, dtype=np.float32).reshape(-1, 1, 2)
        _c, (rw, rh), _a = cv2.minAreaRect(contour)
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        if aspect > 3.2:
            # One long bent pod/fruit, not a merged pile: never split it.
            result.append(obj)
            continue
        mask = obj.mask
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        peak = float(dist.max())
        if peak < 14.0:
            result.append(obj)
            continue
        _, seeds = cv2.threshold(dist, 0.45 * peak, 255, 0)
        seeds = seeds.astype(np.uint8)
        n_seeds, seed_labels = cv2.connectedComponents(seeds)
        if n_seeds - 1 < 2 or n_seeds - 1 > 8:
            result.append(obj)
            continue
        markers = seed_labels.astype(np.int32)
        markers[mask == 0] = 0
        bgr = image_bgr if image_bgr.dtype == np.uint8 else image_bgr.astype(np.uint8)
        ws = cv2.watershed(bgr, markers)
        parts = []
        inherited = [f for f in obj.flags if f != "LOW_SOLIDITY_POSSIBLY_MERGED"]
        for label in range(1, n_seeds):
            part_mask = np.where(ws == label, 255, 0).astype(np.uint8)
            part_mask = cv2.bitwise_and(part_mask, mask)
            part = _rebuild_object(part_mask, image_bgr.shape, inherited)
            if part is None:
                continue
            if part.area_px >= 0.15 * obj.area_px and part.area_px >= min_area_px:
                parts.append(part)
        if len(parts) < 2:
            result.append(obj)
            continue
        changed = True
        result.extend(parts[:6])
    if not changed:
        return None
    result.sort(key=lambda o: o.area_px, reverse=True)
    return result
