"""ArUco fiducial detection and planar calibration.

The marker establishes a millimetre coordinate system on the matte plane.  It
does not measure object height: a tall vegetable's top surface projects onto a
larger footprint than its contact patch.  The resulting parallax component is
carried as a named, non-statistical uncertainty rather than hidden.
"""

from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np

from .config import CalibrationSettings
from .domain import Calibration, MarkerDetection


class MarkerError(ValueError):
    pass


def get_aruco_dictionary(name: str) -> cv2.aruco.Dictionary:
    enum = getattr(cv2.aruco, name, None)
    if enum is None:
        available = sorted(
            attr for attr in dir(cv2.aruco) if attr.startswith("DICT_") and "ARUCO" not in attr
        )
        raise MarkerError(f"Unsupported ArUco dictionary '{name}'. Available: {available}")
    return cv2.aruco.getPredefinedDictionary(enum)


def detector_parameters() -> cv2.aruco.DetectorParameters:
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 31
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.015
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.045
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 40
    params.cornerRefinementMinAccuracy = 0.02
    params.errorCorrectionRate = 0.6
    return params


def _side_lengths(corners: np.ndarray) -> list[float]:
    return [
        float(np.linalg.norm(corners[(i + 1) % 4] - corners[i]))
        for i in range(4)
    ]


def _validate_quad(
    corners: np.ndarray,
    image_shape: tuple[int, ...],
    settings: CalibrationSettings,
) -> tuple[bool, list[str], float]:
    reasons: list[str] = []
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return False, ["Marker corners are malformed or non-finite."], 0.0
    sides = _side_lengths(corners)
    shortest = min(sides)
    longest = max(sides)
    if shortest < settings.min_marker_side_px:
        reasons.append(
            f"Marker is too small for reliable scale ({shortest:.1f} px side; "
            f"minimum {settings.min_marker_side_px:.0f} px). Move it closer."
        )
    if shortest <= 0 or longest / shortest > settings.max_side_length_ratio:
        reasons.append("Marker quadrilateral is geometrically implausible.")
    contour = corners.astype(np.float32).reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(contour)))
    image_area = float(image_shape[0] * image_shape[1])
    if area / max(image_area, 1.0) < settings.min_marker_area_fraction:
        reasons.append("Marker occupies too little of the frame.")
    if not cv2.isContourConvex(contour):
        reasons.append("Marker quadrilateral is not convex.")
    hull_area = abs(float(cv2.contourArea(cv2.convexHull(contour))))
    if hull_area <= 0 or area / hull_area < 0.98:
        reasons.append("Marker quadrilateral is not a clean planar square.")
    h, w = image_shape[:2]
    if (
        corners[:, 0].min() < 1
        or corners[:, 1].min() < 1
        or corners[:, 0].max() > w - 2
        or corners[:, 1].max() > h - 2
    ):
        reasons.append("Marker touches or crosses the image boundary.")
    return len(reasons) == 0, reasons, area


def detect_markers(
    image_bgr: np.ndarray,
    settings: CalibrationSettings,
    expected_id: int | None = 0,
) -> list[MarkerDetection]:
    """Detect ArUco markers, returning only geometrically usable candidates."""
    if image_bgr is None or image_bgr.size == 0:
        return []
    dictionary = get_aruco_dictionary(settings.dictionary)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_parameters())
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _rejected = detector.detectMarkers(gray)
    detections: list[MarkerDetection] = []
    if ids is None:
        return detections
    for corners, marker_id in zip(corners_list, ids.ravel().tolist(), strict=True):
        points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        valid, reasons, area = _validate_quad(points, image_bgr.shape, settings)
        sides = _side_lengths(points)
        if not valid:
            # Retained as warning information by the caller but not usable for
            # physical measurement.
            detections.append(
                MarkerDetection(
                    marker_id=int(marker_id),
                    corners_px=[tuple(map(float, p)) for p in points],
                    side_px=float(min(sides)) if sides else 0.0,
                    area_px=float(area),
                    dictionary=settings.dictionary,
                    reasons=reasons,
                )
            )
            continue
        detections.append(
            MarkerDetection(
                marker_id=int(marker_id),
                corners_px=[tuple(map(float, p)) for p in points],
                side_px=float(min(sides)),
                area_px=float(area),
                dictionary=settings.dictionary,
            )
        )
    if expected_id is not None:
        expected = [d for d in detections if d.marker_id == expected_id]
        if expected:
            return sorted(expected, key=lambda d: d.area_px, reverse=True)
        # Keep wrong-ID detections so the caller can give a specific reason.
        return detections
    return sorted(detections, key=lambda d: d.area_px, reverse=True)


def detect_best_marker(
    image_bgr: np.ndarray,
    settings: CalibrationSettings,
    expected_id: int | None = 0,
) -> tuple[MarkerDetection | None, list[str]]:
    markers = detect_markers(image_bgr, settings, expected_id=expected_id)
    if not markers:
        return None, [
            "No ArUco marker detected. Place the printed marker flat beside the "
            "vegetable on the same matte plane.",
        ]
    usable = [m for m in markers if not m.reasons and m.area_px > 0]
    if expected_id is not None:
        matching = [m for m in usable if m.marker_id == expected_id]
        if not matching:
            found = ", ".join(str(m.marker_id) for m in markers)
            return None, [
                f"Expected marker ID {expected_id}, but detected: {found or 'none'}. "
                "Print/use the configured marker or update the expected ID."
            ]
        best = max(matching, key=lambda m: m.area_px)
    else:
        if not usable:
            reasons: list[str] = []
            for marker in markers:
                reasons.extend(marker.reasons)
            return None, reasons or ["Detected marker is not geometrically usable."]
        best = max(usable, key=lambda m: m.area_px)
    if best.reasons:
        return None, list(best.reasons)
    return best, []


def calibrate_from_marker(
    image_shape: tuple[int, ...],
    marker: MarkerDetection,
    marker_side_mm: float | None,
    settings: CalibrationSettings,
    marker_measurement_uncertainty_mm: float | None = None,
) -> Calibration:
    reasons: list[str] = []
    warnings: list[str] = []
    if marker_side_mm is None or not math.isfinite(float(marker_side_mm)):
        return Calibration(
            valid=False,
            calibration_id="unavailable",
            marker_id=marker.marker_id,
            reasons=[
                "The actual printed marker side length is required. Measure the "
                "outer black square in millimetres; a nominal PDF size is not proof."
            ],
        )
    side_mm = float(marker_side_mm)
    if side_mm <= 5.0 or side_mm > 1000.0:
        return Calibration(
            valid=False,
            calibration_id="invalid",
            marker_id=marker.marker_id,
            marker_side_mm=side_mm,
            reasons=["Marker side must be between 5 and 1000 mm."],
        )
    if marker.reasons:
        return Calibration(
            valid=False,
            calibration_id="invalid",
            marker_id=marker.marker_id,
            marker_side_mm=side_mm,
            reasons=list(marker.reasons),
        )
    if marker.side_px < settings.min_marker_side_px:
        return Calibration(
            valid=False,
            calibration_id="invalid",
            marker_id=marker.marker_id,
            marker_side_mm=side_mm,
            reasons=[
                f"Marker side is only {marker.side_px:.1f} px; minimum is "
                f"{settings.min_marker_side_px:.0f} px. Move the marker closer."
            ],
        )
    image_area = float(image_shape[0] * image_shape[1])
    if image_area <= 0 or marker.area_px / image_area < settings.min_marker_area_fraction:
        return Calibration(
            valid=False,
            calibration_id="invalid",
            marker_id=marker.marker_id,
            marker_side_mm=side_mm,
            reasons=["Marker occupies too little of the frame for reliable scale."],
        )
    corners = np.asarray(marker.corners_px, dtype=np.float32)
    if corners.shape != (4, 2):
        reasons.append("Marker corners are unavailable.")
    if not np.all(np.isfinite(corners)):
        reasons.append("Marker corners contain non-finite values.")
    if reasons:
        return Calibration(
            valid=False,
            calibration_id="invalid",
            marker_id=marker.marker_id,
            marker_side_mm=side_mm,
            reasons=reasons,
        )

    plane_corners = np.array(
        [[0.0, 0.0], [side_mm, 0.0], [side_mm, side_mm], [0.0, side_mm]],
        dtype=np.float32,
    )
    homography, _mask = cv2.findHomography(corners, plane_corners, 0)
    if homography is None or not np.all(np.isfinite(homography)):
        return Calibration(
            valid=False,
            calibration_id="invalid",
            marker_id=marker.marker_id,
            marker_side_mm=side_mm,
            reasons=["Could not derive a finite planar homography from the marker."],
        )
    # Homography with exactly four points is exact at the marker corners;
    # nevertheless report a reprojection residual as a guard against numerical
    # degeneracy.
    projected = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected - plane_corners) ** 2, axis=1))))
    if not math.isfinite(rms) or rms > 1e-3:
        warnings.append(
            "Marker homography reprojection residual is non-negligible; "
            "verify marker corners and try a cleaner capture."
        )
    side_px = float(marker.side_px)
    mm_per_px = side_mm / side_px if side_px > 0 else float("nan")
    if not math.isfinite(mm_per_px) or mm_per_px <= 0:
        reasons.append("Derived scale is invalid.")
    det = float(np.linalg.det(homography[:2, :2]))
    if abs(det) < 1e-12:
        reasons.append("Homography is singular.")
    if reasons:
        return Calibration(
            valid=False,
            calibration_id="invalid",
            marker_id=marker.marker_id,
            marker_side_mm=side_mm,
            reasons=reasons,
        )

    corner_sig = ",".join(f"{x:.2f}:{y:.2f}" for x, y in corners.ravel().reshape(-1, 2))
    calib_hash = hashlib.sha256(
        f"{marker.marker_id}|{side_mm:.4f}|{corner_sig}".encode()
    ).hexdigest()[:12]
    uncertainty = (
        float(marker_measurement_uncertainty_mm)
        if marker_measurement_uncertainty_mm is not None
        else settings.default_marker_measurement_uncertainty_mm
    )
    if not math.isfinite(uncertainty) or uncertainty < 0:
        uncertainty = settings.default_marker_measurement_uncertainty_mm
        warnings.append("Invalid marker measurement uncertainty; used configured default.")
    warning = (
        "Planar homography assumes the measured vegetable contact footprint is "
        "coplanar with the marker. Elevated top surfaces have parallax bias."
    )
    warnings.append(warning)

    return Calibration(
        valid=True,
        calibration_id=f"calib_aruco_{marker.marker_id}_{calib_hash}",
        marker_id=marker.marker_id,
        marker_side_mm=side_mm,
        homography=np.asarray(homography, dtype=float).tolist(),
        mm_per_px=float(mm_per_px),
        corner_reprojection_rms_px=rms,
        marker_measurement_uncertainty_mm=uncertainty,
        warnings=warnings,
    )


def marker_mask(
    shape: tuple[int, int, int] | tuple[int, int],
    marker: MarkerDetection | None,
    margin_px: int = 10,
) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if marker is None:
        return mask
    points = np.asarray(marker.corners_px, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(points)
    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(shape[1], x + w + margin_px)
    y1 = min(shape[0], y + h + margin_px)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
    return mask


def marker_polygon(marker: MarkerDetection) -> np.ndarray:
    return np.asarray(marker.corners_px, dtype=np.int32).reshape(-1, 1, 2)


def marker_side_summary(marker: MarkerDetection) -> dict[str, float | int | str]:
    return {
        "marker_id": marker.marker_id,
        "dictionary": marker.dictionary,
        "side_px": round(marker.side_px, 2),
        "area_px": round(marker.area_px, 1),
        "corners_px": [[round(x, 2), round(y, 2)] for x, y in marker.corners_px],
    }


def draw_detected_marker(
    image: np.ndarray,
    marker: MarkerDetection,
    side_mm: float | None = None,
    color: tuple[int, int, int] = (0, 220, 255),
) -> None:
    points = marker_polygon(marker)
    cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
    x, y = points.reshape(-1, 2).min(axis=0)
    label = f"MARKER {marker.marker_id}"
    if side_mm is not None:
        label += f" | {side_mm:.1f} mm actual"
    cv2.putText(
        image,
        label,
        (int(x), max(18, int(y) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )


def generate_marker(
    marker_id: int = 0,
    dictionary_name: str = "DICT_4X4_50",
    side_mm: float = 50.0,
    dpi: int = 300,
    quiet_zone_mm: float = 10.0,
) -> np.ndarray:
    """Generate a white-background marker at a known nominal print size.

    The caller must still measure the printed outer black square.  Printer
    scaling makes nominal dimensions untrustworthy.
    """
    if side_mm <= 0 or side_mm > 1000:
        raise MarkerError("side_mm must be between 0 and 1000.")
    if dpi < 72:
        raise MarkerError("dpi must be >= 72.")
    dictionary = get_aruco_dictionary(dictionary_name)
    side_px = max(64, int(round(side_mm / 25.4 * dpi)))
    # generateImageMarker includes the black outer border.  The quiet zone is
    # added explicitly and is not part of the measured side.
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    quiet_px = max(12, int(round(quiet_zone_mm / 25.4 * dpi)))
    canvas = np.full(
        (side_px + 2 * quiet_px, side_px + 2 * quiet_px, 3), 255, dtype=np.uint8
    )
    canvas[quiet_px : quiet_px + side_px, quiet_px : quiet_px + side_px, 0] = marker
    canvas[quiet_px : quiet_px + side_px, quiet_px : quiet_px + side_px, 1] = marker
    canvas[quiet_px : quiet_px + side_px, quiet_px : quiet_px + side_px, 2] = marker
    return canvas


def list_dictionaries() -> list[str]:
    return [
        attr
        for attr in dir(cv2.aruco)
        if attr.startswith("DICT_") and attr.isupper() and "ARUCO" not in attr
    ]
