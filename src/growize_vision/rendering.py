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


def _colour_for(status: Status) -> tuple[int, int, int]:
    return _COLORS_BGR.get(status, (0, 191, 255))


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
    contour = np.asarray(instance.contour_px, dtype=np.float32).reshape(-1, 1, 2)
    decision = instance.decision
    status = decision.status if decision else Status.REVIEW_BORDERLINE
    colour = _colour_for(status)
    fill = image.copy()
    cv2.drawContours(fill, [contour.astype(np.int32)], -1, colour, thickness=-1)
    cv2.addWeighted(fill, 0.20, image, 0.80, 0, dst=image)
    cv2.drawContours(image, [contour.astype(np.int32)], -1, colour, 2, cv2.LINE_AA)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(image, [box.reshape(-1, 1, 2)], -1, colour, 1, cv2.LINE_AA)
    (cx, cy), (rw, rh), angle = rect
    if rw >= rh:
        long_axis = ((box[0][0], box[0][1]), (box[1][0], box[1][1]))
        short_axis = ((box[1][0], box[1][1]), (box[2][0], box[2][1]))
    else:
        long_axis = ((box[1][0], box[1][1]), (box[2][0], box[2][1]))
        short_axis = ((box[0][0], box[0][1]), (box[1][0], box[1][1]))
    cv2.arrowedLine(
        image,
        tuple(int(v) for v in long_axis[0]),
        tuple(int(v) for v in long_axis[1]),
        (52, 211, 153),
        2,
        cv2.LINE_AA,
        tipLength=0.035,
    )
    cv2.arrowedLine(
        image,
        tuple(int(v) for v in short_axis[0]),
        tuple(int(v) for v in short_axis[1]),
        (180, 240, 200),
        1,
        cv2.LINE_AA,
        tipLength=0.04,
    )

    x, y, w, h = instance.bbox_px
    label_y = max(18, y - 8)
    band_grade_words = {"small": "1", "medium": "2", "large": "3"}
    if decision and decision.accepted and decision.band_name:
        band = decision.band_name.lower()
        number = band if band.isdigit() else band_grade_words.get(band)
        if number and number.isdigit():
            headline = f"GRADE {number}"
            if band in band_grade_words:
                headline += f" · {band.upper()}"
        else:
            headline = decision.band_name.upper()
    elif decision and decision.accepted and decision.size_class:
        headline = decision.size_class.value
    elif decision and decision.status != Status.CALIBRATION_REQUIRED:
        headline = decision.status.value
    else:
        headline = ""
    if headline:
        _put_text(image, headline, (max(4, x), label_y), scale=0.62, colour=colour, thickness=2)
    if instance.measurement:
        m = instance.measurement
        unit = m.unit
        second = f"{m.value_mm:.1f} +/- {m.uncertainty_mm:.1f} {unit}"
        _put_text(
            image,
            second,
            (max(4, x), min(image.shape[0] - 8, label_y + 20)),
            scale=0.46,
            colour=_WHITE,
            thickness=1,
        )
    if decision and decision.threshold_mm is not None and instance.measurement:
        third = (
            f"policy {decision.policy_version or '?'} | cutoff "
            f"{decision.threshold_mm:.1f} {instance.measurement.unit}"
        )
        _put_text(
            image,
            third,
            (max(4, x), min(image.shape[0] - 8, label_y + 38)),
            scale=0.39,
            colour=(205, 215, 225),
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
        band_words = {"small": "1", "medium": "2", "large": "3"}
        if result.size_band:
            band = result.size_band.lower()
            number = band if band.isdigit() else band_words.get(band)
            grade = f"GRADE {number} · {band.upper()}" if number else result.size_band.upper()
        else:
            grade = result.size_class.value
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
                colour = _colour_for(result.size_class or Status.ACCEPTED)
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
