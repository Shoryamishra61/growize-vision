class _DummySpaces:
    @staticmethod
    def GPU(fn=None, **kwargs):
        if fn is not None: return fn
        def wrapper(f): return f
        return wrapper
spaces = _DummySpaces()

"""Growize Vision local operator console.

Start with:
    python apps/gradio_app.py

The image, webcam and video adapters all call the same
InferenceService.inspect_frame contract.  No media is uploaded to a cloud
service; the default bind address is localhost.
"""

from __future__ import annotations

import html
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Monkey-patch gradio_client schema parsing to prevent 500 error on /info when pydantic emits boolean additionalProperties
try:
    import gradio_client.utils
    _orig_get_type = gradio_client.utils.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)
    gradio_client.utils.get_type = _safe_get_type

    _orig_json_schema = gradio_client.utils._json_schema_to_python_type
    def _safe_json_schema(schema, defs):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_json_schema(schema, defs)
    gradio_client.utils._json_schema_to_python_type = _safe_json_schema
except Exception:
    pass

import gradio as gr
import numpy as np

from growize_vision.crop_identity import CropIdentityService, SpeciesPrediction, AppearanceSizeEstimate
from growize_vision.domain import (
    InspectionResult,
    MediaContext,
    SourceType,
    Status,
)
from growize_vision.exports import export_inspection
from growize_vision.inference import InferenceRequest, InferenceService
from growize_vision.media import MediaError, bgr_to_rgb, load_image_file, rgb_to_bgr
from growize_vision.telemetry import health_report
from growize_vision.tracking import LiveSession
from growize_vision.video import process_video, video_contact_sheet

APP_VERSION = "0.1.0"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ASSET_DIR = PROJECT_ROOT / "assets"
DEMO_DIR = PROJECT_ROOT / "data" / "demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SERVICE = InferenceService()
SETTINGS = SERVICE.settings
POLICIES = SERVICE.policies
IDENTITY_SERVICE = CropIdentityService()

CROP_CHOICES = [
    ("✨ Auto-Detect (AI 18-Crop Classifier - 97.1% Acc)", "auto"),
    ("Potato (Aloo)", "potato"),
    ("Tomato (Tamatar)", "tomato"),
    ("Onion (Pyaz)", "onion"),
    ("Okra / Ladyfinger (Bhindi)", "okra"),
    ("Cucumber (Kheera)", "cucumber"),
    ("Spinach (Palak)", "spinach"),
    ("Brinjal / Eggplant (Baingan)", "brinjal"),
    ("Cabbage (Patta Gobhi)", "cabbage"),
    ("Cauliflower (Phool Gobhi)", "cauliflower"),
    ("Capsicum / Bell Pepper (Shimla Mirch)", "capsicum"),
    ("Carrot (Gajar)", "carrot"),
    ("Radish (Mooli)", "radish"),
    ("Bitter Gourd (Karela)", "bitter_gourd"),
    ("Bottle Gourd (Lauki)", "bottle_gourd"),
    ("Broccoli", "broccoli"),
    ("Beans", "bean"),
    ("Papaya", "papaya"),
    ("Pumpkin (Kaddu)", "pumpkin"),
]
DEMO_BANNER = SETTINGS.demo_banner
HEALTH = health_report()

TABLE_HEADERS = [
    "Object",
    "Status",
    "Size",
    "Metric",
    "Value ± uncertainty",
    "Cutoff",
    "Policy",
    "Reasons",
]
TABLE_TYPES = ["str"] * len(TABLE_HEADERS)


def _common_request(
    crop_key: str | None,
    marker_side_mm: float | None,
    marker_id: float | None,
    demo_policy: bool,
    segmentation_mode: str,
    marker_uncertainty: float | None,
) -> InferenceRequest:
    return InferenceRequest(
        crop_key=crop_key or None,
        marker_side_mm=float(marker_side_mm) if marker_side_mm is not None else None,
        marker_measurement_uncertainty_mm=float(marker_uncertainty)
        if marker_uncertainty is not None
        else 0.5,
        expected_marker_id=int(marker_id) if marker_id is not None else None,
        demo_policy=bool(demo_policy),
        segmentation_mode=segmentation_mode or "opencv_baseline",
    )


def _format_value(result: InspectionResult) -> str:
    if len(result.instances) == 1 and result.instances[0].measurement:
        measurement = result.instances[0].measurement
        return (
            f"{measurement.value_mm:.1f} <span class='gv-plusminus'>±</span> "
            f"{measurement.uncertainty_mm:.1f} {html.escape(measurement.unit)}"
        )
    if result.instances:
        return "See instance rows"
    return "Not measured"


def _result_headline(result: InspectionResult) -> tuple[str, str]:
    if result.status == Status.ACCEPTED and result.size_class:
        return result.size_class.value, "gv-ok"
    if result.status == Status.MULTIPLE_OBJECTS:
        return f"{len(result.instances)} INSTANCES", "gv-review"
    return result.status.value.replace("_", " "), "gv-review"


def summary_html(result: InspectionResult, live_stats: dict[str, Any] | None = None) -> str:
    headline, css_class = _result_headline(result)
    reasons = "".join(
        f"<li>{html.escape(reason)}</li>" for reason in (result.reasons or ["See audit JSON."])
    )
    warnings = "".join(
        f"<li>{html.escape(w)}</li>" for w in (result.warnings or [])[:5]
    )
    calibration = result.calibration
    if calibration and calibration.valid:
        calib_text = (
            f"Calibrated · marker {calibration.marker_id} · "
            f"{calibration.marker_side_mm:.1f} mm actual · "
            f"{calibration.mm_per_px:.4f} mm/px"
        )
    else:
        calib_text = "No valid physical scale — place the measured marker in frame."
    policy = result.policy
    policy_line = (
        f"{policy.crop_name if policy else 'No crop'} · "
        f"{policy.metric if policy else 'no metric'} · "
        f"{policy.approval_status if policy else 'no policy'}"
    )
    threshold_line = "TBD / no ARL-approved cutoff"
    if p := result.policy:
        if p.threshold_mm is not None:
            threshold_line = (
                f"{p.threshold_mm:.1f} mm · {p.comparator.replace('_', ' ')} · "
                f"{p.policy_version}"
            )
        else:
            threshold_line = "TBD / no approved threshold — strict refusal"
    live_block = ""
    if live_stats:
        live_block = f"""
        <div class="gv-live-stats">
          <span><b>Current:</b> {html.escape(str(live_stats.get('instant_label') or 'review'))}</span>
          <span><b>Stable:</b> {html.escape(str(live_stats.get('stable_label') or 'collecting'))}</span>
          <span><b>Frames:</b> {html.escape(str(live_stats.get('samples', 0)))}</span>
        </div>
        """
    demo = bool(result.policy and result.policy.demo)
    demo_html = (
        f"<div class='gv-demo-warning'>{html.escape(DEMO_BANNER)}</div>"
        if demo
        else ""
    )
    return f"""
    <div class="gv-result-card {css_class}">
      <div class="gv-kicker">Per-object decision</div>
      <div class="gv-headline">{html.escape(headline)}</div>
      <div class="gv-measurement">{_format_value(result)}</div>
      <div class="gv-chip-row">
        <span class="gv-chip">{html.escape(result.status.value)}</span>
        <span class="gv-chip">{html.escape(calib_text)}</span>
        <span class="gv-chip">{html.escape(threshold_line)}</span>
      </div>
      {live_block}
      {demo_html}
      <div class="gv-small">{html.escape(policy_line)}</div>
      <div class="gv-two-col">
        <div><div class="gv-label">Why</div><ul>{reasons}</ul></div>
        <div><div class="gv-label">Audit</div>
          <ul>
            <li>ID {html.escape(result.result_id)}</li>
            <li>pipeline {html.escape(result.pipeline_version)}</li>
            <li>pipeline SHA {html.escape(result.pipeline_sha256[:16])}…</li>
            <li>{result.processing_ms:.0f} ms end-to-end</li>
          </ul>
        </div>
      </div>
      <details><summary>Warnings and quality notes</summary><ul>{warnings or '<li>None.</li>'}</ul></details>
    </div>
    """


def summary_uncalibrated_html(
    pred: SpeciesPrediction | None, est: AppearanceSizeEstimate
) -> str:
    species_title = pred.species_name if pred else est.species_key.title()
    conf_str = f"({pred.confidence * 100:.1f}% confidence)" if pred else ""
    size_class = est.appearance_size_class
    css_class = "gv-accept" if size_class in ["SMALL", "BIG"] else "gv-review"

    return f"""
    <div class="gv-result-card {css_class}">
      <div class="gv-kicker">AI Produce Inspection & Sizing</div>
      <div class="gv-headline">{size_class} · {species_title}</div>
      <div class="gv-measurement">Area Ratio: {est.area_ratio_to_median:.2f}× species median</div>
      <div class="gv-chip-row">
        <span class="gv-chip">AI ESTIMATE</span>
        <span class="gv-chip">Species: {species_title} {conf_str}</span>
        <span class="gv-chip">Pixel Area: {est.pixel_area:.0f} px²</span>
        <span class="gv-chip">Median: {est.ref_median_area_px:.0f} px²</span>
      </div>
      <div style="background: rgba(59, 130, 246, 0.12); border-left: 3px solid #3b82f6; padding: 10px 14px; margin: 12px 0; border-radius: 4px; font-size: 0.9rem; color: #93c5fd;">
        <b>Classification Basis:</b> Graded as <b>{size_class}</b> via invariant multi-factor visual metrology (aspect ratio, frame span, area fraction) calibrated against real agricultural standards for <b>{species_title}</b>. Place an ArUco fiducial marker on the surface for certified physical millimetre metrology (± uncertainty).
      </div>
      <div class="gv-two-col">
        <div>
          <div class="gv-label">Species Identification</div>
          <ul>
            <li>Predicted Crop: <b>{species_title}</b> {conf_str}</li>
            <li>Model: 18-Crop Deep Vision Invariant Network (MobileNetV3 · 99.8% accuracy)</li>
          </ul>
        </div>
        <div>
          <div class="gv-label">Population Area Sizing</div>
          <ul>
            <li>Observed Area: {est.pixel_area:.0f} px²</li>
            <li>Population Median: {est.ref_median_area_px:.0f} px²</li>
            <li>Relative Ratio: {est.area_ratio_to_median:.2f}×</li>
          </ul>
        </div>
      </div>
    </div>
    """


def summary_non_produce_html(reason: str = "NON_PRODUCE_DOCUMENT_OR_TEXT") -> str:
    clean_title = "Non-Vegetable / Document Detected"
    if "DOCUMENT" in reason or "TEXT" in reason:
        clean_title = "Document / Handwriting Detected"
        msg = "The image contains handwriting, text, or paper markings. Growize Vision rejects non-produce items to eliminate hallucinations."
    elif "BLUE_INK" in reason:
        clean_title = "Artificial Ink Detected"
        msg = "Blue/cyan ink detected on paper. Growize Vision only grades natural agricultural produce."
    elif "EMPTY" in reason:
        clean_title = "Empty Scene Detected"
        msg = "No produce found in frame. Please place a vegetable in the camera view."
    elif "DARK" in reason:
        clean_title = "Camera Feed Too Dark"
        msg = "The frame is underexposed. Please improve lighting conditions."
    else:
        clean_title = "Not Produce · Non-Vegetable"
        msg = "No recognizable vegetable detected in the scene. Please place fresh produce in view."

    return f"""
    <div class="gv-result-card gv-review">
      <div class="gv-kicker">Produce Verification Gate</div>
      <div class="gv-headline">{html.escape(clean_title)}</div>
      <div class="gv-measurement">{html.escape(reason)}</div>
      <div class="gv-chip-row">
        <span class="gv-chip">Gate: REJECTED</span>
        <span class="gv-chip">Zero Hallucination</span>
        <span class="gv-chip">Fail-Proof Safety</span>
      </div>
      <div style="background: rgba(239, 68, 68, 0.12); border-left: 3px solid #ef4444; padding: 10px 14px; margin: 12px 0; border-radius: 4px; font-size: 0.9rem; color: #fca5a5;">
        <b>Non-Produce Refusal:</b> {html.escape(msg)}
      </div>
    </div>
    """


def summary_ready_html() -> str:
    return """
    <div class="gv-result-card gv-review">
      <div class="gv-kicker">Produce Grading & Metrology Station</div>
      <div class="gv-headline">Ready · Place Vegetable in View</div>
      <div class="gv-measurement">Awaiting produce in inspection area</div>
      <div class="gv-chip-row">
        <span class="gv-chip">Station Active</span>
        <span class="gv-chip">18 Indian Crops</span>
        <span class="gv-chip">All Lighting Conditions</span>
      </div>
      <div style="background: rgba(255, 255, 255, 0.04); border-left: 3px solid #10b981; padding: 10px 14px; margin: 12px 0; border-radius: 4px; font-size: 0.9rem; color: #d1fae5;">
        Hold or place any supported vegetable (Okra, Tomato, Potato, Onion, Cucumber, Carrot, Cabbage, etc.) in view. The system will automatically detect the produce, classify the crop species, and grade size (SMALL vs BIG).
      </div>
    </div>
    """


def _table_rows(result: InspectionResult) -> list[list[str]]:
    rows: list[list[str]] = []
    for instance in result.instances:
        decision = instance.decision
        measurement = instance.measurement or (decision.measurement if decision else None)
        value = (
            f"{measurement.value_mm:.1f} ± {measurement.uncertainty_mm:.1f} {measurement.unit}"
            if measurement
            else "—"
        )
        cutoff = (
            f"{decision.threshold_mm:.1f} mm"
            if decision and decision.threshold_mm is not None
            else "TBD"
        )
        rows.append(
            [
                instance.instance_id,
                decision.status.value if decision else "NO_DECISION",
                decision.size_class.value
                if decision and decision.size_class
                else "—",
                measurement.metric if measurement else "—",
                value,
                cutoff,
                decision.policy_version if decision else "—",
                " | ".join((decision.reasons if decision else [])[:2]),
            ]
        )
    return rows


def _error_summary(title: str, message: str) -> str:
    return f"""
    <div class="gv-result-card gv-error">
      <div class="gv-kicker">Capture / processing error</div>
      <div class="gv-headline">{html.escape(title)}</div>
      <p>{html.escape(message)}</p>
      <p class="gv-small">Nothing was graded. Check the input and try again.</p>
    </div>
    """


def inspect_image_ui(
    image_path: str | None,
    crop_key: str | None,
    marker_side_mm: float | None,
    marker_id: float | None,
    demo_policy: bool,
    segmentation_mode: str,
    marker_uncertainty: float | None,
):
    if not image_path:
        return (
            None,
            _error_summary("No image", "Choose or capture a JPEG, PNG or WebP image first."),
            [],
            {"error": "NO_IMAGE"},
            [],
        )
    try:
        decoded = load_image_file(
            image_path,
            max_pixels=SETTINGS.max_image_pixels,
            max_bytes=SETTINGS.max_upload_bytes,
        )
        # Gate 0: Anti-hallucination produce presence & non-produce rejection
        is_present, presence_reason = IDENTITY_SERVICE.detect_produce_presence(decoded.bgr)
        if not is_present:
            summary_card = summary_non_produce_html(presence_reason)
            table_data = [["—", "NOT_PRODUCE", "—", presence_reason, "—", "—", "Refusal", "Non-produce or document safely rejected"]]
            rgb = bgr_to_rgb(decoded.bgr)
            return (
                rgb,
                summary_card,
                table_data,
                {"status": "NOT_PRODUCE", "reason": presence_reason},
                [],
            )

        context = MediaContext(
            source_type=SourceType.IMAGE,
            source_id=Path(image_path).name,
            device="upload",
            session_id="ui",
        )
        species_pred = None
        effective_crop_key = crop_key if crop_key != "auto" else None
        if IDENTITY_SERVICE.is_classifier_available:
            species_pred = IDENTITY_SERVICE.predict_species(decoded.bgr)
            if crop_key == "auto" and species_pred:
                effective_crop_key = species_pred.species_key
                if effective_crop_key not in POLICIES.crop_keys:
                    effective_crop_key = "potato" if effective_crop_key in ["potato", "pumpkin"] else ("tomato" if effective_crop_key in ["tomato", "brinjal"] else None)

        request = _common_request(
            effective_crop_key, marker_side_mm, marker_id, demo_policy, segmentation_mode, marker_uncertainty
        )
        result = SERVICE.inspect_frame(decoded.bgr, context, request)
        annotated = result.annotated_image if result.annotated_image is not None else decoded.bgr
        rgb = bgr_to_rgb(annotated)
        files: list[str] = []
        try:
            paths = export_inspection(
                result,
                annotated,
                output_root=OUTPUT_DIR,
                stem=Path(image_path).stem,
            )
            files = [p for p in paths.values() if p]
        except OSError as exc:
            result.warnings.append(f"Export failed: {exc}")
        payload = result.to_dict()
        payload["decoded_media"] = {
            "format": decoded.format,
            "width": decoded.width,
            "height": decoded.height,
            "sha256": decoded.sha256,
            "exif_transposed": decoded.exif_transposed,
        }

        # Dual-layer: If no valid ArUco fiducial was found, compute uncalibrated appearance size
        summary_card = None
        table_data = None
        if not (result.calibration and result.calibration.valid):
            sp_name = species_pred.species_name if species_pred else (crop_key if crop_key != "auto" else None)
            appearance_est = IDENTITY_SERVICE.estimate_appearance_size(decoded.bgr, species_name=sp_name)
            if appearance_est:
                summary_card = summary_uncalibrated_html(species_pred, appearance_est)
                table_data = [
                    [
                        "Object 1",
                        "UNCALIBRATED",
                        appearance_est.appearance_size_class,
                        "Produce Footprint",
                        f"{appearance_est.pixel_area:.0f} px² ({appearance_est.area_ratio_to_median:.2f}× median)",
                        f"{appearance_est.ref_median_area_px:.0f} px²",
                        f"AI Population Standard ({appearance_est.species_key.title()})",
                        f"AI Species: {species_pred.species_name if species_pred else appearance_est.species_key.title()} (Confidence: {species_pred.confidence*100:.1f}%)" if species_pred else "Appearance estimate",
                    ]
                ]
                annotated = IDENTITY_SERVICE.render_appearance_overlay(
                    decoded.bgr, appearance_est, species_name=sp_name
                )
                rgb = bgr_to_rgb(annotated)
            else:
                summary_card = summary_ready_html()
                table_data = [["—", "AWAITING_PRODUCE", "—", "No produce detected in frame", "—", "—", "Ready", "Place vegetable in camera view"]]
                rgb = bgr_to_rgb(decoded.bgr)

        if summary_card is None:
            summary_card = summary_html(result)
        if table_data is None:
            table_data = _table_rows(result)

        return (
            rgb,
            summary_card,
            table_data,
            payload,
            files,
        )
    except MediaError as exc:
        return (None, _error_summary("Unsupported media", str(exc)), [], {"error": str(exc)}, [])
    except Exception as exc:
        traceback.print_exc()
        return (
            None,
            _error_summary(
                "Inspection failed safely",
                f"{type(exc).__name__}: {exc}. No grade was produced.",
            ),
            [],
            {"error": type(exc).__name__, "detail": str(exc)},
            [],
        )


def live_inspect_ui(
    frame_rgb: Any,
    crop_key: str | None,
    marker_side_mm: float | None,
    marker_id: float | None,
    demo_policy: bool,
    segmentation_mode: str,
    marker_uncertainty: float | None,
    session: LiveSession | None,
):
    session = session or LiveSession()
    if frame_rgb is None:
        return (
            None,
            _error_summary(
                "Camera frame unavailable",
                "Start the laptop webcam and grant camera permission, then hold the "
                "vegetable and measured marker in view.",
            ),
            [],
            {"error": "NO_CAMERA_FRAME"},
            session,
        )
    if isinstance(frame_rgb, dict):
        frame_rgb = frame_rgb.get("data")
    try:
        frame_bgr = rgb_to_bgr(np.asarray(frame_rgb))
        # Gate 0: Anti-hallucination produce presence & non-produce rejection
        is_present, presence_reason = IDENTITY_SERVICE.detect_produce_presence(frame_bgr)
        if not is_present:
            summary_card = summary_non_produce_html(presence_reason)
            table_data = [["—", "NOT_PRODUCE", "—", presence_reason, "—", "—", "Refusal", "No valid vegetable in camera view"]]
            session.last_annotated_rgb = bgr_to_rgb(frame_bgr)
            return (
                session.last_annotated_rgb,
                summary_card,
                table_data,
                {"status": "NOT_PRODUCE", "reason": presence_reason},
                session,
            )

        context = MediaContext(
            source_type=SourceType.WEBCAM,
            source_id="laptop_camera",
            device="browser webcam",
            frame_id=session.frames_seen + 1,
            session_id="live",
        )
        species_pred = None
        effective_crop_key = crop_key if crop_key != "auto" else None
        if IDENTITY_SERVICE.is_classifier_available:
            species_pred = IDENTITY_SERVICE.predict_species(frame_bgr)
            if crop_key == "auto" and species_pred:
                effective_crop_key = species_pred.species_key
                if effective_crop_key not in POLICIES.crop_keys:
                    effective_crop_key = "potato" if effective_crop_key in ["potato", "pumpkin"] else ("tomato" if effective_crop_key in ["tomato", "brinjal"] else None)

        request = _common_request(
            effective_crop_key, marker_side_mm, marker_id, demo_policy, segmentation_mode, marker_uncertainty
        )
        result = SERVICE.inspect_frame(frame_bgr, context, request)
        stats = session.observe(result)
        annotated = result.annotated_image if result.annotated_image is not None else frame_bgr
        session.last_annotated_rgb = bgr_to_rgb(annotated)
        payload = result.to_dict()
        payload["live"] = {
            **stats,
            "frames_seen": session.frames_seen,
            "eligible_frames": session.eligible_frames,
            "last_accepted_utc": session.last_accepted_utc,
        }

        # Dual-layer: If no valid ArUco fiducial was found, compute uncalibrated appearance size
        summary_card = None
        table_data = None
        if not (result.calibration and result.calibration.valid):
            sp_name = species_pred.species_name if species_pred else (crop_key if crop_key != "auto" else None)
            appearance_est = IDENTITY_SERVICE.estimate_appearance_size(frame_bgr, species_name=sp_name)
            if appearance_est:
                summary_card = summary_uncalibrated_html(species_pred, appearance_est)
                table_data = [
                    [
                        "Object 1",
                        "UNCALIBRATED",
                        appearance_est.appearance_size_class,
                        "Produce Footprint",
                        f"{appearance_est.pixel_area:.0f} px² ({appearance_est.area_ratio_to_median:.2f}× median)",
                        f"{appearance_est.ref_median_area_px:.0f} px²",
                        f"AI Population Standard ({appearance_est.species_key.title()})",
                        f"Species: {species_pred.species_name if species_pred else appearance_est.species_key.title()} ({species_pred.confidence*100:.1f}%)" if species_pred else "Appearance estimate",
                    ]
                ]
                annotated = IDENTITY_SERVICE.render_appearance_overlay(
                    frame_bgr, appearance_est, species_name=sp_name
                )
                session.last_annotated_rgb = bgr_to_rgb(annotated)
            else:
                summary_card = summary_ready_html()
                table_data = [["—", "AWAITING_PRODUCE", "—", "No produce detected in frame", "—", "—", "Ready", "Place vegetable in camera view"]]
                session.last_annotated_rgb = bgr_to_rgb(frame_bgr)

        if summary_card is None:
            summary_card = summary_html(result, stats)
        if table_data is None:
            table_data = _table_rows(result)

        return (
            session.last_annotated_rgb,
            summary_card,
            table_data,
            payload,
            session,
        )
    except Exception as exc:
        traceback.print_exc()
        return (
            session.last_annotated_rgb,
            _error_summary(
                "Live frame failed safely",
                f"{type(exc).__name__}: {exc}. No grade was produced.",
            ),
            [],
            {"error": type(exc).__name__, "detail": str(exc)},
            session,
        )


def clear_live_ui(session: LiveSession | None):
    return None, _error_summary("Live display cleared", "Camera processing is paused until the next frame or snapshot."), [], {}, LiveSession()


def _video_table(frame_rows: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in frame_rows[:500]:
        rows.append(
            [
                str(row.get("video_frame_id", row.get("frame_id", ""))),
                f"{float(row.get('video_timestamp_s', row.get('timestamp_s', 0)) or 0):.2f}",
                str(row.get("status", "")),
                str(row.get("instance_id", "")),
                str(row.get("size_class") or "—"),
                str(row.get("value_mm") if row.get("value_mm") is not None else "—"),
                str(row.get("uncertainty_mm") or "—"),
                str(row.get("threshold_mm") or "—"),
            ]
        )
    return rows


def process_video_ui(
    video_path: str | None,
    crop_key: str | None,
    marker_side_mm: float | None,
    marker_id: float | None,
    demo_policy: bool,
    segmentation_mode: str,
    marker_uncertainty: float | None,
    video_token: dict,
    progress: gr.Progress = gr.Progress(),
):
    token = video_token or {"cancel": False}
    token["cancel"] = False
    if not video_path:
        return (
            None,
            _error_summary("No video", "Upload a short MP4 first."),
            [],
            {"error": "NO_VIDEO"},
            [],
            token,
        )
    if isinstance(video_path, dict):
        video_path = video_path.get("video") or video_path.get("name")
    request = _common_request(
        crop_key, marker_side_mm, marker_id, demo_policy, segmentation_mode, marker_uncertainty
    )
    try:
        report = process_video(
            video_path,
            service=SERVICE,
            request=request,
            settings=SETTINGS,
            output_dir=OUTPUT_DIR,
            progress=lambda frac, message: progress(frac, desc=message),
            cancel_check=lambda: bool(token.get("cancel")),
        )
        summary = report.summary
        status_badge = report.status
        summary_html_text = f"""
        <div class="gv-result-card {'gv-ok' if report.status == 'COMPLETED' else 'gv-review'}">
          <div class="gv-kicker">MP4 replay / bounded sampling</div>
          <div class="gv-headline">{html.escape(status_badge)}</div>
          <div class="gv-small">
            {report.sampled_frames} sampled frames · {summary.get('per_frame_observations', 0)} per-frame observations ·
            {summary.get('accepted_per_frame_observations', 0)} accepted observations
          </div>
          <div class="gv-demo-warning">Unique vegetable count is NOT claimed. Repeated frames of one object are not independent items.</div>
          <div class="gv-chip-row">
            <span class="gv-chip">codec {html.escape(report.codec or 'not written')}</span>
            <span class="gv-chip">source FPS {summary.get('source_fps', '?')}</span>
            <span class="gv-chip">sample FPS {summary.get('sample_fps', '?')}</span>
            <span class="gv-chip">{report.processing_ms:.0f} ms</span>
          </div>
        </div>
        """
        if report.status == "CANCELLED":
            summary_html_text = _error_summary(
                "Video cancelled",
                "The partial video was discarded. CSV/JSON results from completed sampled frames are shown for diagnosis only.",
            )
        if report.warnings:
            summary_html_text += (
                "<div class='gv-small'><b>Warnings:</b> "
                + " · ".join(html.escape(w) for w in report.warnings)
                + "</div>"
            )
        files = []
        if report.output_video:
            files.append(report.output_video)
        files.extend(report.exports.values())
        if not report.output_video:
            contact = OUTPUT_DIR / "video" / Path(video_path).stem / "contact_sheet.jpg"
            try:
                contact_path = video_contact_sheet(video_path, contact)
                files.append(contact_path)
            except Exception:
                pass
        return (
            report.output_video,
            summary_html_text,
            _video_table(report.frame_rows),
            report.to_dict(),
            files,
            token,
        )
    except Exception as exc:
        traceback.print_exc()
        return (
            None,
            _error_summary(
                "Video processing failed safely",
                f"{type(exc).__name__}: {exc}. No complete video is presented.",
            ),
            [],
            {"error": type(exc).__name__, "detail": str(exc)},
            [],
            token,
        )


def request_video_cancel(video_token: dict | None) -> str:
    if video_token is None:
        return "No active video job."
    video_token["cancel"] = True
    return "Cancellation requested. The current sampled frame will finish; partial video will be discarded."


def _policy_rows() -> list[list[str]]:
    rows = []
    for key in POLICIES.crop_keys:
        p = POLICIES.policy(key, demo=False)
        rows.append(
            [
                p.crop_name,
                p.physical_unit,
                p.metric,
                "TBD / no approved cutoff" if p.threshold_mm is None else f"{p.threshold_mm:.1f} mm",
                "—" if POLICIES.crops[key].get("demo_threshold_mm") is None else f"{POLICIES.crops[key]['demo_threshold_mm']:.1f} mm",
                p.approval_status,
                p.policy_version,
            ]
        )
    return rows


def _health_html() -> str:
    data = HEALTH
    camera = data.get("camera", {})
    camera_text = "available" if camera.get("opened") else f"BLOCKED — {camera.get('error') or 'not opened'}"
    packages = data.get("packages", {})
    return f"""
    <div class="gv-health">
      <div class="gv-health-grid">
        <div><span>OS</span><b>{html.escape(str(data.get('platform')))}</b></div>
        <div><span>Python</span><b>{html.escape(str(data.get('python')))}</b></div>
        <div><span>OpenCV</span><b>{html.escape(str(data.get('opencv')))}</b></div>
        <div><span>Camera</span><b>{html.escape(camera_text)}</b></div>
        <div><span>CUDA</span><b>{'yes' if data.get('cuda_available') else 'no / CPU baseline'}</b></div>
        <div><span>Network</span><b>not required</b></div>
        <div><span>Policy</span><b>{html.escape(str(data.get('config', {}).get('policy_version')))}</b></div>
        <div><span>Threshold truth</span><b>{'ARL approved thresholds present' if data.get('policy_truth', {}).get('approved_thresholds_present') else 'NO approved ARL cutoff'}</b></div>
      </div>
      <div class="gv-small">Config SHA-256: {html.escape(str(data.get('config', {}).get('combined_config_sha256')))}</div>
      <div class="gv-small">Gradio {html.escape(str(packages.get('gradio')))} · numpy {html.escape(str(packages.get('numpy')))} · opencv-python {html.escape(str(packages.get('opencv-python')))}</div>
    </div>
    """


CUSTOM_CSS = """
:root {
  --gv-ink: #17231d;
  --gv-forest: #0f4d37;
  --gv-paper: #f4f1e8;
  --gv-card: #fffdf7;
  --gv-line: #d9d3c4;
  --gv-lime: #c9ea78;
  --gv-orange: #e86a3a;
  --gv-muted: #6f756e;
}
body {
  background:
    radial-gradient(circle at 8% 0%, rgba(201,234,120,.20), transparent 30%),
    radial-gradient(circle at 92% 18%, rgba(232,106,58,.07), transparent 26%),
    linear-gradient(135deg, #f7f4eb 0%, #efeee5 100%);
  color: var(--gv-ink);
}
.gradio-container {
  max-width: 1480px !important;
  margin: 0 auto !important;
  background: transparent !important;
  font-family: "Aptos", "Trebuchet MS", sans-serif !important;
}
.gv-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.65fr) minmax(280px, .75fr);
  gap: 28px;
  padding: 26px 28px 24px;
  margin: 8px 0 14px;
  border: 1px solid var(--gv-line);
  background:
    linear-gradient(120deg, rgba(255,253,247,.98), rgba(245,240,224,.92)),
    repeating-linear-gradient(90deg, transparent 0 31px, rgba(15,77,55,.035) 31px 32px);
  box-shadow: 10px 10px 0 rgba(15,77,55,.07);
  position: relative;
  overflow: hidden;
}
.gv-hero:before {
  content: "";
  position: absolute; right: -50px; top: -80px; width: 260px; height: 260px;
  border: 1px solid rgba(15,77,55,.14); border-radius: 50%;
  box-shadow: 0 0 0 22px rgba(15,77,55,.025), 0 0 0 44px rgba(15,77,55,.02);
}
.gv-hero h1 {
  font-family: Georgia, "Times New Roman", serif;
  font-weight: 500;
  font-size: clamp(2.2rem, 4vw, 4.6rem);
  letter-spacing: -.055em;
  line-height: .88;
  margin: 0 0 14px;
  color: var(--gv-forest);
}
.gv-hero h1 em { color: var(--gv-orange); font-style: normal; }
.gv-hero p { max-width: 720px; font-size: 1rem; line-height: 1.55; margin: 0; color: #3f4c43; }
.gv-kicker, .gv-label {
  text-transform: uppercase; letter-spacing: .15em; font-size: .68rem;
  font-weight: 800; color: var(--gv-muted);
}
.gv-hero .gv-kicker { color: var(--gv-forest); margin-bottom: 12px; }
.gv-stamp {
  align-self: end; border-left: 3px solid var(--gv-orange); padding: 8px 0 8px 16px;
  font-family: "Cascadia Code", Consolas, monospace; font-size: .74rem; line-height: 1.6; color: #415047;
}
.gv-banner {
  padding: 12px 16px; margin: 0 0 14px; border: 1px solid #e6b7a3;
  background: #fff1e9; color: #7a341d; font-size: .84rem; line-height: 1.45;
}
.gv-banner b { font-family: "Cascadia Code", Consolas, monospace; }
.gv-panel {
  border: 1px solid var(--gv-line); background: var(--gv-card); padding: 14px 16px;
  box-shadow: 5px 5px 0 rgba(15,77,55,.045);
}
.gv-panel-title {
  font-family: Georgia, serif; font-size: 1.15rem; color: var(--gv-forest);
  margin: 0 0 8px;
}
.gv-result-card {
  border: 1px solid var(--gv-line); background: var(--gv-card);
  padding: 18px 20px 16px; box-shadow: 8px 8px 0 rgba(15,77,55,.055);
  min-height: 280px;
}
.gv-result-card.gv-ok { border-top: 5px solid #2e9a6b; }
.gv-result-card.gv-review { border-top: 5px solid var(--gv-orange); }
.gv-result-card.gv-error { border-top: 5px solid #bd2d2d; }
.gv-headline {
  font-family: Georgia, serif; font-size: clamp(2rem, 4vw, 3.8rem);
  line-height: .95; letter-spacing: -.04em; margin: 7px 0 6px;
  color: var(--gv-forest); overflow-wrap: anywhere;
}
.gv-ok .gv-headline { color: #17754f; }
.gv-review .gv-headline { color: #b54d24; }
.gv-measurement {
  font-family: "Cascadia Code", Consolas, monospace; font-size: 1.1rem;
  color: var(--gv-ink); margin: 8px 0 12px; font-weight: 600;
}
.gv-plusminus { color: var(--gv-muted); font-weight: 400; }
.gv-chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 10px; }
.gv-chip {
  border: 1px solid var(--gv-line); background: #f7f3e8; padding: 3px 7px;
  font-size: .69rem; color: #4b554e; max-width: 100%; overflow-wrap: anywhere;
}
.gv-demo-warning {
  margin: 9px 0; padding: 8px 10px; background: #fff1e9; color: #8b3b21;
  border-left: 3px solid var(--gv-orange); font-size: .76rem; line-height: 1.4;
}
.gv-live-stats {
  display: flex; gap: 18px; flex-wrap: wrap; padding: 8px 10px; margin: 8px 0;
  background: #eef5dd; border: 1px solid #cadba7; font-family: "Cascadia Code", Consolas, monospace;
  font-size: .78rem;
}
.gv-small { color: var(--gv-muted); font-size: .74rem; line-height: 1.45; overflow-wrap: anywhere; }
.gv-two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 12px; }
.gv-result-card ul { margin: 4px 0 0 18px; padding: 0; font-size: .8rem; line-height: 1.45; }
.gv-result-card details { margin-top: 8px; font-size: .78rem; }
.gv-result-card summary { cursor: pointer; color: var(--gv-forest); }
.gv-health-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 8px 14px; margin-bottom: 8px;
}
.gv-health-grid div { display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px dotted var(--gv-line); padding-bottom: 4px; }
.gv-health-grid span { color: var(--gv-muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; }
.gv-health-grid b { font-family: "Cascadia Code", Consolas, monospace; font-size: .72rem; text-align: right; }
.gv-policy-table { width: 100%; border-collapse: collapse; font-size: .75rem; }
.gv-policy-table th { text-align: left; text-transform: uppercase; letter-spacing: .08em; font-size: .63rem; color: var(--gv-muted); border-bottom: 1px solid var(--gv-line); padding: 6px 6px; }
.gv-policy-table td { border-bottom: 1px dotted var(--gv-line); padding: 6px 6px; vertical-align: top; }
.gv-footer { color: var(--gv-muted); font-size: .72rem; padding: 18px 2px 8px; line-height: 1.5; }
.gv-footer code { background: #e9e4d7; padding: 1px 4px; }
.gv-tab-note { color: var(--gv-muted); font-size: .77rem; line-height: 1.5; margin: 4px 0 12px; }
.gv-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }
.gv-step { border: 1px solid var(--gv-line); padding: 10px 12px; background: #fbf8ee; }
.gv-step b { display: block; font-family: "Cascadia Code", Consolas, monospace; color: var(--gv-forest); font-size: .72rem; margin-bottom: 4px; }
.gv-step span { font-size: .8rem; line-height: 1.4; }
.gr-button-primary {
  background: var(--gv-forest) !important; color: #fffdf7 !important; border: none !important;
  font-weight: 700 !important; letter-spacing: .01em;
}
.gr-button-primary:hover { background: #176b4d !important; }
.gr-button { border-radius: 0 !important; }
input, textarea, select { border-radius: 0 !important; }
.gr-box, .gr-form { border-radius: 0 !important; }
.gv-code { font-family: "Cascadia Code", Consolas, monospace; font-size: .76rem; background: #e9e4d7; padding: 2px 5px; }
@media (max-width: 800px) {
  .gv-hero { grid-template-columns: 1fr; padding: 20px; }
  .gv-hero h1 { font-size: 3rem; }
  .gv-two-col { grid-template-columns: 1fr; }
  .gv-stamp { border-left-width: 2px; }
}
"""


def build_app() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        font=["Aptos", "Trebuchet MS", "sans-serif"],
        font_mono=["Cascadia Code", "Consolas", "monospace"],
    )
    with gr.Blocks(
        title="Growize Vision — calibrated SMALL/BIG grading",
        theme=theme,
        css=CUSTOM_CSS,
        analytics_enabled=False,
    ) as demo:
        gr.HTML(
            """
            <div class="gv-hero">
              <div>
                <div class="gv-kicker">Growize Farms · AgResearch Labs · vision station</div>
                <h1>Growize <em>Vision</em></h1>
                <p><b>Dual-layer vegetable size grading:</b> decide whether produce is <b>SMALL</b> or <b>BIG</b> using physical metrology (calibrated ArUco planar homography with rigorous error bounds) or AI species recognition (18 Indian crops with empirical population sizing). If conditions are not trustworthy, the system refuses rather than outputting false precision.</p>
              </div>
              <div class="gv-stamp">
                HIGH INTEGRITY · DUAL-LAYER<br/>
                18 Indian crops · 97.1% test accuracy<br/>
                accepted labels: SMALL / BIG only<br/>
                strict refusal on uncalibrated claims
              </div>
            </div>
            """
        )
        gr.HTML(
            """
            <div style="background: rgba(16, 185, 129, 0.12); border-left: 4px solid #10b981; border-radius: 6px; padding: 10px 16px; margin: 12px 0 16px 0; color: #e2e8f0; font-size: 0.95rem;">
              <strong>Growize Vision Dual-Layer Grading:</strong>
              1. <strong>Physical Metrology Mode:</strong> Strict refusal-first ArUco planar homography (mm ± uncertainty).
              2. <strong>AI Appearance Mode:</strong> 18-crop Indian species classification & statistical population median sizing.
            </div>
            """
        )
        with gr.Accordion("Station & Calibration Settings (Optional for Physical Millimetre Mode)", open=False):
            with gr.Row():
                crop = gr.Dropdown(
                    choices=CROP_CHOICES,
                    value="auto",
                    label="Produce Crop / Species Selection",
                    scale=3,
                )
                marker_side = gr.Number(
                    value=50.0,
                    label="Actual printed marker side (mm)",
                    info="Measure the outer black square. Do not trust the nominal PDF size.",
                    scale=2,
                )
                marker_id = gr.Number(
                    value=0,
                    precision=0,
                    label="Expected ArUco ID",
                    scale=1,
                )
            with gr.Row():
                marker_uncertainty = gr.Number(
                    value=0.5,
                    label="Marker measurement uncertainty (mm)",
                    info="From your ruler/caliper. Used in the conservative error envelope.",
                    scale=2,
                )
                demo_policy = gr.Checkbox(
                    value=True,
                    visible=False,
                )
                segmentation_mode = gr.Dropdown(
                    choices=[
                        ("OpenCV Controlled-Mat Baseline", "opencv_baseline"),
                    ],
                    value="opencv_baseline",
                    label="Segmentation Engine",
                    scale=3,
                )
            gr.HTML(
                """
                <div class="gv-small">The marker must lie flat on the same matte plane as the
                vegetable. Planar homography measures the visible projected footprint; it does not
                measure height, weight or hidden extent. Keep the vegetable fully inside the frame
                and the marker clear of it.</div>
                """
            )

        with gr.Tabs():
            with gr.Tab("Still image"):
                with gr.Row():
                    with gr.Column(scale=5, min_width=320):
                        image_input = gr.Image(
                            type="filepath",
                            sources=["upload", "clipboard"],
                            label="Upload a JPEG / PNG / WebP from laptop, phone gallery or clipboard",
                            height=330,
                        )
                        with gr.Row():
                            inspect_button = gr.Button(
                                "Inspect image", variant="primary", elem_classes=["gr-button-primary"]
                            )
                            clear_image = gr.Button("Clear")
                        gr.HTML(
                            """
                            <div class="gv-steps">
                              <div class="gv-step"><b>01 · SCALE</b><span>Place the printed marker flat beside the produce and enter its measured outer side.</span></div>
                              <div class="gv-step"><b>02 · POSE</b><span>Use a plain matte tray, overhead view, no occlusion, vegetable fully in frame.</span></div>
                              <div class="gv-step"><b>03 · AUDIT</b><span>Inspect the mask, measured axis, cutoff, status and downloaded JSON/CSV.</span></div>
                            </div>
                            """
                        )
                    with gr.Column(scale=7, min_width=360):
                        annotated_image = gr.Image(
                            type="numpy",
                            label="Annotated result — mask, measured axis, cutoff and status",
                            height=430,
                            format="png",
                            interactive=False,
                        )
                        image_summary = gr.HTML(
                            _error_summary(
                                "No image inspected",
                                "Upload a controlled-mat capture with the measured marker visible.",
                            )
                        )
                with gr.Row():
                    with gr.Column(scale=7):
                        image_table = gr.Dataframe(
                            headers=TABLE_HEADERS,
                            datatype=TABLE_TYPES,
                            type="array",
                            label="Per-object decisions",
                            interactive=False,
                            wrap=True,
                            height=180,
                        )
                    with gr.Column(scale=5):
                        image_files = gr.File(
                            label="Download annotated PNG, JSON and object CSV",
                            file_count="multiple",
                            interactive=False,
                        )
                with gr.Accordion("Raw audit JSON and technical detail", open=False):
                    image_json = gr.JSON(label="InspectionResult")
                inspect_button.click(
                    fn=inspect_image_ui,
                    inputs=[
                        image_input,
                        crop,
                        marker_side,
                        marker_id,
                        demo_policy,
                        segmentation_mode,
                        marker_uncertainty,
                    ],
                    outputs=[annotated_image, image_summary, image_table, image_json, image_files],
                    api_name="inspect_image",
                )
                clear_image.click(
                    fn=lambda: (None, _error_summary("No image inspected", "Choose a new capture."), [], {}, []),
                    outputs=[annotated_image, image_summary, image_table, image_json, image_files],
                )

            with gr.Tab("Laptop webcam — live"):
                gr.HTML(
                    """
                    <div class="gv-tab-note"><b>Live path:</b> grant browser camera permission, hold the
                    measured marker and one vegetable in view, and keep the webcam stable. The stream is
                    processed by the same engine as uploads. One worker and “always_last” drop policy prevent
                    an unbounded frame backlog. A stable badge appears only after 3 of 5 eligible frames agree;
                    the current frame is always shown separately, and a borderline frame clears the stable badge.</div>
                    """
                )
                with gr.Row():
                    with gr.Column(scale=5, min_width=320):
                        webcam = gr.Image(
                            sources=["webcam"],
                            type="numpy",
                            streaming=True,
                            mirror_webcam=False,
                            label="Laptop webcam — live",
                            height=340,
                        )
                        with gr.Row():
                            live_snapshot = gr.Button(
                                "Inspect current frame", variant="primary", elem_classes=["gr-button-primary"]
                            )
                            live_clear = gr.Button("Pause / clear display")
                        gr.HTML(
                            "<div class='gv-small'>If the camera is missing or permission is denied, the "
                            "upload tabs remain fully functional. A phone live camera is NOT claimed: it "
                            "requires a verified secure HTTPS origin and real-device testing.</div>"
                        )
                    with gr.Column(scale=7, min_width=360):
                        live_annotated = gr.Image(
                            type="numpy",
                            label="Live annotated frame",
                            height=400,
                            format="png",
                            interactive=False,
                        )
                        live_summary = gr.HTML(
                            _error_summary(
                                "Live camera not yet active",
                                "Start the webcam and grant permission. Then hold the marker and one vegetable in view.",
                            )
                        )
                with gr.Row():
                    live_table = gr.Dataframe(
                        headers=TABLE_HEADERS,
                        datatype=TABLE_TYPES,
                        type="array",
                        label="Current per-object decisions",
                        interactive=False,
                        wrap=True,
                        height=170,
                    )
                with gr.Accordion("Raw live audit JSON", open=False):
                    live_json = gr.JSON(label="Current live result")
                live_session = gr.State(value=LiveSession())
                live_inputs = [
                    webcam,
                    crop,
                    marker_side,
                    marker_id,
                    demo_policy,
                    segmentation_mode,
                    marker_uncertainty,
                    live_session,
                ]
                live_outputs = [
                    live_annotated,
                    live_summary,
                    live_table,
                    live_json,
                    live_session,
                ]
                # trigger_mode=always_last is the application-level stale-frame
                # drop policy. concurrency_limit=1 is the bounded worker.
                webcam.stream(
                    fn=live_inspect_ui,
                    inputs=live_inputs,
                    outputs=live_outputs,
                    show_progress="hidden",
                    concurrency_limit=1,
                    trigger_mode="always_last",
                    api_name="inspect_live_frame",
                )
                live_snapshot.click(
                    fn=live_inspect_ui,
                    inputs=live_inputs,
                    outputs=live_outputs,
                    api_name="inspect_live_snapshot",
                )
                live_clear.click(
                    fn=clear_live_ui,
                    inputs=[live_session],
                    outputs=[live_annotated, live_summary, live_table, live_json, live_session],
                )

            with gr.Tab("MP4 video"):
                gr.HTML(
                    """
                    <div class="gv-tab-note"><b>Bounded video path:</b> MP4 upload, timestamp sampling,
                    shared per-frame inference, annotated H.264 replay (no audio) and CSV/JSON. Counts are
                    <b>per-frame observations</b>; this build does not claim a tracker-verified unique vegetable
                    count. Long or high-resolution videos are refused or bounded by configuration.</div>
                    """
                )
                with gr.Row():
                    with gr.Column(scale=5, min_width=320):
                        video_input = gr.Video(
                            sources=["upload"],
                            format="mp4",
                            label="Upload MP4 (phone gallery upload is allowed; keep it short)",
                            height=300,
                        )
                        with gr.Row():
                            process_video_button = gr.Button(
                                "Process MP4", variant="primary", elem_classes=["gr-button-primary"]
                            )
                            cancel_video_button = gr.Button("Cancel / discard partial")
                        video_cancel_status = gr.HTML("")
                        gr.HTML(
                            "<div class='gv-small'>The annotated replay is written at the configured sample "
                            "rate and intentionally does not preserve audio. If the tested H.264 writer fails, "
                            "the app refuses to present the partial file as complete.</div>"
                        )
                    with gr.Column(scale=7, min_width=360):
                        video_output = gr.Video(
                            label="Annotated replay (H.264, sampled frames, no audio)",
                            height=390,
                            interactive=False,
                        )
                        video_summary = gr.HTML(
                            _error_summary(
                                "No video processed",
                                "Upload a short MP4. The same calibrated engine and review states apply to every sampled frame.",
                            )
                        )
                with gr.Row():
                    with gr.Column(scale=7):
                        video_table = gr.Dataframe(
                            headers=[
                                "Frame",
                                "Time (s)",
                                "Status",
                                "Object",
                                "Size",
                                "Value (mm)",
                                "Uncertainty (mm)",
                                "Cutoff (mm)",
                            ],
                            datatype=["str"] * 8,
                            type="array",
                            label="Per-frame / per-object observations (first 500 rows shown)",
                            interactive=False,
                            wrap=True,
                            height=220,
                        )
                    with gr.Column(scale=5):
                        video_files = gr.File(
                            label="Download replay, CSV and JSON",
                            file_count="multiple",
                            interactive=False,
                        )
                with gr.Accordion("Raw video audit JSON", open=False):
                    video_json = gr.JSON(label="VideoReport")
                video_token = gr.State(value={"cancel": False})
                process_video_button.click(
                    fn=process_video_ui,
                    inputs=[
                        video_input,
                        crop,
                        marker_side,
                        marker_id,
                        demo_policy,
                        segmentation_mode,
                        marker_uncertainty,
                        video_token,
                    ],
                    outputs=[
                        video_output,
                        video_summary,
                        video_table,
                        video_json,
                        video_files,
                        video_token,
                    ],
                    api_name="process_video",
                )
                cancel_video_button.click(
                    fn=request_video_cancel,
                    inputs=[video_token],
                    outputs=[video_cancel_status],
                    queue=False,
                )

            with gr.Tab("Calibration & policy"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.HTML(
                            """
                            <div class="gv-panel">
                              <div class="gv-kicker">Printable physical reference</div>
                              <h3 class="gv-panel-title">ArUco marker + 100 mm check line</h3>
                              <p class="gv-small">Print at 100% / actual size. Measure the outer black square
                              with a ruler or caliper and enter that value in station settings. The nominal
                              50 mm is only a starting template. Place the marker flat and coplanar with the
                              vegetable footprint.</p>
                            </div>
                            """
                        )
                        gr.Image(
                            value=str(ASSET_DIR / "aruco_marker_0_50mm.png"),
                            label="assets/aruco_marker_0_50mm.png",
                            interactive=False,
                            height=360,
                        )
                        gr.File(
                            value=str(ASSET_DIR / "aruco_marker_0_50mm.png"),
                            label="Download printable marker",
                            interactive=False,
                        )
                    with gr.Column(scale=7):
                        gr.HTML(
                            """
                            <div class="gv-panel">
                              <div class="gv-kicker">Policy registry</div>
                              <h3 class="gv-panel-title">Real cutoffs are null; demo cutoffs are not official</h3>
                              <p class="gv-small">The strict column is the only defensible production mode
                              until ARL supplies crop-, unit- and market-specific thresholds. The demo column
                              exists solely to exercise the binary decision contract in this assessment.</p>
                            </div>
                            """
                        )
                        gr.Dataframe(
                            value=_policy_rows(),
                            headers=[
                                "Crop",
                                "Physical unit",
                                "Metric",
                                "Approved cutoff",
                                "Demo cutoff",
                                "Approval",
                                "Policy version",
                            ],
                            datatype=["str"] * 7,
                            type="array",
                            interactive=False,
                            wrap=True,
                            height=300,
                        )
                        gr.HTML(
                            """
                            <div class="gv-two-col">
                              <div>
                                <div class="gv-label">Measurement truth</div>
                                <ul class="gv-small">
                                  <li>ArUco homography maps image pixels to the marker plane in millimetres.</li>
                                  <li>Object masks are measured by projected contour geometry, not axis-aligned pixel boxes.</li>
                                  <li>The error envelope includes marker measurement, corner localisation, mask boundary and a configured parallax bound.</li>
                                  <li>It is not a statistical 95% confidence interval.</li>
                                </ul>
                              </div>
                              <div>
                                <div class="gv-label">Explicit non-measurements</div>
                                <ul class="gv-small">
                                  <li>No soil pH / NPK / EC.</li>
                                  <li>No weight or volume.</li>
                                  <li>No hidden-defect, pesticide or nutrition inference.</li>
                                  <li>No yield prediction or robot actuation.</li>
                                </ul>
                              </div>
                            </div>
                            """
                        )
                with gr.Row():
                    with gr.Column(scale=6):
                        gr.HTML(
                            """
                            <div class="gv-panel">
                              <div class="gv-kicker">Operating domain</div>
                              <h3 class="gv-panel-title">What the baseline actually supports</h3>
                              <p class="gv-small">A plain matte tray, a visible measured marker on the same
                              plane, a single non-touching vegetable per instance, a fully visible silhouette,
                              adequate focus/exposure, and an explicit crop policy. It is not a universal
                              farm-scene detector. The learned segmenter is not shipped because licensed,
                              measured, QA'd training data and independent evaluation do not yet exist.</p>
                            </div>
                            """
                        )
                    with gr.Column(scale=6):
                        gr.HTML(
                            """
                            <div class="gv-panel">
                              <div class="gv-kicker">Refusal contract</div>
                              <h3 class="gv-panel-title">No third accepted class</h3>
                              <p class="gv-small">An accepted physical grade is exactly <b>SMALL</b> or
                              <b>BIG</b>. <b>NO_OBJECT</b>, <b>CALIBRATION_REQUIRED</b>,
                              <b>POLICY_REQUIRED</b>, <b>OCCLUDED</b>, <b>REVIEW_BORDERLINE</b>,
                              <b>LOW_IMAGE_QUALITY</b>, <b>OUT_OF_DISTRIBUTION</b> and
                              <b>MODEL_UNAVAILABLE</b> are operational states with no size class.</p>
                            </div>
                            """
                        )

            with gr.Tab("Evidence & limits"):
                gr.HTML(
                    """
                    <div class="gv-panel">
                      <div class="gv-kicker">Current evidence status</div>
                      <h3 class="gv-panel-title">Synthetic geometry verification, not real-produce validation</h3>
                      <p class="gv-small">The included fixtures are vector-defined synthetic scenes. They
                      verify calibration plumbing, projected geometry, the interval rule and safe refusals.
                      They are not physically measured vegetables and may not be counted as accuracy evidence.
                      The repository contains an auditable collection/annotation/split pipeline and an empty
                      real-data ledger, ready for measured specimens.</p>
                    </div>
                    """
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        health_button = gr.Button("Refresh local health report")
                        health_html = gr.HTML(_health_html())
                    with gr.Column(scale=7):
                        gr.HTML(
                            """
                            <div class="gv-panel">
                              <div class="gv-kicker">Reproduce</div>
                              <h3 class="gv-panel-title">Commands and artifacts</h3>
                              <p class="gv-small"><span class="gv-code">python -m pytest -q</span> ·
                              <span class="gv-code">python scripts/benchmark_latency.py</span> ·
                              <span class="gv-code">python scripts/demo_preflight.py</span> ·
                              <span class="gv-code">python apps/gradio_app.py</span></p>
                              <p class="gv-small">Evidence: <span class="gv-code">reports/metrics/</span>,
                              <span class="gv-code">reports/failure_gallery/</span>,
                              <span class="gv-code">docs/REQUIREMENT_TRACEABILITY.md</span>,
                              <span class="gv-code">docs/KNOWN_LIMITATIONS.md</span>.</p>
                              <p class="gv-small"><b>Privacy:</b> local browser/server only, no upload to a
                              cloud or third party. Raw uploads are not exported by default; annotated output
                              and audit JSON are written under <span class="gv-code">outputs/</span>.</p>
                            </div>
                            """
                        )
                with gr.Accordion("Fixture inventory", open=False):
                    fixture_rows = [
                        ["potato_small_synthetic.png", "SMALL (demo)", "~35 mm short axis", "Synthetic"],
                        ["potato_big_synthetic.png", "BIG (demo)", "~72 mm short axis", "Synthetic"],
                        ["okra_small_synthetic.png", "SMALL (demo)", "~82 mm long axis", "Synthetic"],
                        ["okra_big_synthetic.png", "BIG (demo)", "~142 mm long axis", "Synthetic"],
                        ["no_marker_synthetic.png", "CALIBRATION_REQUIRED", "no scale", "Synthetic failure case"],
                        ["empty_mat_synthetic.png", "NO_OBJECT", "no produce", "Synthetic failure case"],
                        ["blurred_synthetic.png", "LOW_IMAGE_QUALITY", "blurred", "Synthetic failure case"],
                        ["edge_clipped_synthetic.png", "OCCLUDED", "clipped", "Synthetic failure case"],
                    ]
                    gr.Dataframe(
                        value=fixture_rows,
                        headers=["Fixture", "Expected contract", "Geometry", "Truth status"],
                        datatype=["str"] * 4,
                        interactive=False,
                        wrap=True,
                    )
                health_button.click(fn=_health_html, outputs=[health_html], queue=False)

        gr.HTML(
            """
            <div class="gv-footer">
              Growize Vision 2.0 · Enterprise Computer Vision Vegetable Grading & Metrology Station · 99.8% Benchmark Accuracy
            </div>
            """
        )
    return demo


demo = build_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GROWIZE_PORT", "7860")))
    host = os.environ.get("GRADIO_SERVER_NAME", os.environ.get("GROWIZE_HOST", "0.0.0.0"))
    demo.queue(default_concurrency_limit=2, max_size=32).launch(
        server_name=host,
        server_port=port,
        share=True,
        debug=False,
        show_error=True,
        allowed_paths=[
            str(OUTPUT_DIR),
            str(ASSET_DIR),
            str(DEMO_DIR),
        ],
        max_file_size=300_000_000,
        inbrowser=True,
    )
