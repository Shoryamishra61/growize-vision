"""High-contrast, audit-friendly visual overlays."""

from __future__ import annotations

import cv2
import numpy as np

from .domain import InspectionResult, InstanceObservation, SizeClass, Status

_COLORS_BGR = {
    Status.ACCEPTED: (52, 211, 153),       # emerald green
    SizeClass.SMALL: (52, 211, 153),
    SizeClass.BIG: (52, 211, 153),
    Status.REVIEW_BORDERLINE: (52, 211, 153),
    Status.CALIBRATION_REQUIRED: (52, 211, 153),
    Status.POLICY_REQUIRED: (52, 211, 153),
    Status.OCCLUDED: (0, 128, 255),
    Status.LOW_IMAGE_QUALITY: (0, 128, 255),
    Status.OUT_OF_DISTRIBUTION: (0, 128, 255),
    Status.NO_OBJECT: (200, 200, 200),
    Status.MULTIPLE_OBJECTS: (52, 211, 153),
    Status.INVALID_INPUT: (0, 0, 255),
    Status.MODEL_UNAVAILABLE: (0, 0, 255),
}
_WHITE = (245, 245, 245)
_DARK = (22, 27, 34)
_RED = (60, 60, 230)
_NEUTRAL = (140, 180, 230)
# Grade colours a sorter can read at a glance (BGR): orange = small,
# amber = medium, green = big.
_GRADE_COLOURS_BGR = {
    "small": (10, 170, 255),
    "medium": (40, 205, 255),
    "big": (70, 210, 90),
}
_GRADE_NUMBERS = {"small": 1, "medium": 2, "big": 3}
_NUMBER_WORDS = {"1": "small", "2": "medium", "3": "big"}


def _colour_for(status: Status) -> tuple[int, int, int]:
    return _COLORS_BGR.get(status, (0, 191, 255))


def _grade_word(decision) -> str | None:
    """Plain size word for an accepted decision, or None."""
    if decision is None or not decision.accepted:
        return None
    band = (decision.band_name or "").lower()
    if decision.size_class is SizeClass.SMALL:
        return "small"
    if decision.size_class is SizeClass.BIG:
        return "big"
    if band in _NUMBER_WORDS:
        return _NUMBER_WORDS[band]
    if band in ("medium",):
        return "medium"
    if band in ("large", "big"):
        return "big"
    if decision.size_class is not None:
        return decision.size_class.value.lower()
    return None


def _chip(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    colour: tuple[int, int, int],
    *,
    scale: float = 0.62,
) -> None:
    """Solid readable label chip with the grade colour as its border."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    pad = 5
    x0, y0 = max(2, x), max(2, y - th - pad * 2)
    cv2.rectangle(image, (x0, y0), (x0 + tw + pad * 2, y0 + th + pad * 2), _DARK, -1)
    cv2.rectangle(
        image,
        (x0, y0),
        (x0 + tw + pad * 2, y0 + th + pad * 2),
        colour,
        2,
    )
    _put_text(
        image,
        text,
        (x0 + pad, y0 + th + pad),
        scale=scale,
        colour=colour,
        thickness=2,
    )


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    colour: tuple[int, int, int] = _WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _panel(
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    colour: tuple[int, int, int] = _DARK,
    alpha: float = 0.82,
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, thickness=-1)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, dst=image)


def _draw_instance(image: np.ndarray, instance: InstanceObservation) -> None:
    """Tight silhouette outline plus one readable grade chip per item.

    No rotated bounding boxes, axis arrows, or raw status strings: operators
    on a packing line need the item's outline and its size word, nothing else.
    """
    contour = np.asarray(instance.contour_px, dtype=np.float32).reshape(-1, 1, 2)
    decision = instance.decision
    word = _grade_word(decision)
    colour = _GRADE_COLOURS_BGR.get(word, _NEUTRAL)
    points = contour.astype(np.int32)
    # Subtle tint inside the item so the grade colour groups visually.
    fill = image.copy()
    cv2.drawContours(fill, [points], -1, colour, thickness=-1)
    cv2.addWeighted(fill, 0.16, image, 0.84, 0, dst=image)
    # Tight outline of the actual detected silhouette.
    cv2.drawContours(image, [points], -1, colour, 3, cv2.LINE_AA)

    x, y, w, h = instance.bbox_px
    if word:
        number = _GRADE_NUMBERS.get(word)
        label = f"{word.upper()} ({number})" if number else word.upper()
        _chip(image, label, max(4, x), max(26, y), colour)
    elif instance.measurement is None:
        # Ungraded item: small neutral tick so it is still visible.
        _chip(image, "CHECK", max(4, x), max(26, y), _NEUTRAL, scale=0.5)
    if instance.measurement:
        m = instance.measurement
        second = f"{m.value_mm:.0f} +/- {m.uncertainty_mm:.0f} {m.unit}"
        _put_text(
            image,
            second,
            (max(4, x), min(image.shape[0] - 8, y + h + 18)),
            scale=0.5,
            colour=_WHITE,
            thickness=1,
        )


def render_inspection(image_bgr: np.ndarray, result: InspectionResult) -> np.ndarray:
    canvas = image_bgr.copy()
    # Detected outlines are always drawn: the operator must see what the
    # system found even when scale or policy blocked a numeric grade.
    for instance in result.instances:
        _draw_instance(canvas, instance)
    # In unscaled/uncalibrated mode, skip internal overlay box so produce is
    # clean and unobscured (an appearance overlay may replace this canvas).
    if result.calibration and not result.calibration.valid:
        return canvas

    h, w = canvas.shape[:2]
    margin = max(8, int(min(h, w) * 0.018))
    line_h = max(18, int(min(h, w) * 0.031))
    lines: list[str] = []
    if result.status == Status.ACCEPTED and (result.size_class or result.size_band):
        band = (result.size_band or "").lower()
        word = _NUMBER_WORDS.get(band) or ({"large": "big"}.get(band))
        if not word and result.size_class:
            word = result.size_class.value.lower()
        if word:
            number = _GRADE_NUMBERS.get(word)
            grade = f"{word.upper()} ({number})" if number else word.upper()
        else:
            grade = (result.size_band or result.size_class.value).upper()
        lines.append(f"RESULT  {grade}")
    elif result.calibration and not result.calibration.valid:
        lines.append("UNSCALED APPEARANCE MODE")
        if len(result.instances) > 1:
            lines.append(f"{len(result.instances)} produce items detected")
        lines.append("Add ArUco marker for calibrated physical mm")
    else:
        lines.append(f"STATUS  {result.status.value}")
    if len(result.instances) == 1 and result.instances[0].decision:
        d = result.instances[0].decision
        if d.measurement:
            lines.append(
                f"{d.measurement.metric}: {d.measurement.value_mm:.1f} +/- "
                f"{d.measurement.uncertainty_mm:.1f} {d.measurement.unit}"
            )
        if d.threshold_mm is not None:
            lines.append(
                f"cutoff {d.threshold_mm:.1f} {d.measurement.unit if d.measurement else 'mm'}"
            )
        elif d.status == Status.POLICY_REQUIRED:
            lines.append("No standard cutoff for this crop")
    elif result.instances and (result.calibration and result.calibration.valid):
        lines.append(f"{len(result.instances)} instance(s) - see per-item rows")
    if result.calibration and result.calibration.valid:
        if result.calibration.calibration_id.startswith("camera_prior"):
            lines.append("calib camera_prior_estimate (distance assumed)")
        else:
            lines.append(
                f"calib {result.calibration.calibration_id[:26]} | "
                f"marker {result.calibration.marker_id} @ "
                f"{result.calibration.marker_side_mm:.1f} mm"
            )
    if result.context.source_type.value:
        lines.append(
            f"source {result.context.source_type.value} | "
            f"{result.pipeline_version} | {result.processing_ms:.0f} ms"
        )
    panel_h = margin * 2 + line_h * len(lines)
    panel_w = min(w - margin * 2, max(260, int(w * 0.52)))
    _panel(canvas, margin, margin, panel_w, panel_h, _DARK, 0.86)
    for i, line in enumerate(lines):
        colour = _WHITE
        if i == 0:
            colour = _colour_for(result.status)
            if result.status == Status.ACCEPTED:
                word = None
                if len(result.instances) == 1:
                    word = _grade_word(result.instances[0].decision)
                colour = _GRADE_COLOURS_BGR.get(word, _colour_for(Status.ACCEPTED))
                if result.policy and result.policy.demo:
                    colour = (0, 191, 255)
        if "DEMO" in line or "NOT VALID" in line:
            colour = _RED
        _put_text(
            canvas,
            line,
            (margin + 10, margin + line_h * (i + 1) - 4),
            scale=0.48 if i else 0.62,
            colour=colour,
            thickness=2 if i == 0 else 1,
        )

    if result.status not in (Status.ACCEPTED, Status.MULTIPLE_OBJECTS):
        # Short, actionable footer.  The full reason is in JSON.
        reason = (result.reasons or result.warnings or ["Review required."])[0]
        reason = reason[:118]
        footer_h = 30
        _panel(canvas, 0, h - footer_h, w, footer_h, _DARK, 0.82)
        _put_text(
            canvas,
            reason,
            (12, h - 10),
            scale=0.45,
            colour=(0, 191, 255) if result.status != Status.NO_OBJECT else _WHITE,
            thickness=1,
        )
    return canvas
