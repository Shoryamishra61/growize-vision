"""Local size inspection: media in, verified grade or an explicit refusal out."""

from __future__ import annotations

import html
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr

from growize_vision.domain import MediaContext, SourceType, Status
from growize_vision.inference import InferenceRequest, InferenceService
from growize_vision.media import MediaError, bgr_to_rgb, load_image_file, rgb_to_bgr
from growize_vision.tracking import LiveSession
from growize_vision.video import VideoError, process_video

LOGGER = logging.getLogger(__name__)
SERVICE = InferenceService()
OUTPUT_DIR = ROOT / "outputs"

# The deployed app grades against PUBLISHED_REFERENCE crop policies (sourced
# from standards, trade practice, and research data - docs/SIZE_REFERENCE_SOURCES.md).
# Crop identity is auto-detected by the classifier when possible; it abstains
# below its confidence floor and the operator's explicit crop always wins.
CROP_CHOICES = {
    policy.crop_name: policy.crop_key
    for policy in (SERVICE.policies.policy(key) for key in SERVICE.policies.crop_keys)
    if policy.threshold_mm is not None
}
AUTO_CROP_CHOICE = "Auto-detect (single item)"
CROP_UI_CHOICES = [AUTO_CROP_CHOICE] + list(CROP_CHOICES.keys())
STATION_ID = "default-station"
REFERENCE_PRESETS = {
    "Custom length (type below)": None,
    "Ruler marked 100 mm": 100.0,
    "A4 paper — short edge (210 mm)": 210.0,
    "A4 paper — long edge (297 mm)": 297.0,
    "Credit/ID card — long edge (85.6 mm)": 85.6,
    "₹2 coin — diameter (25 mm)": 25.0,
    "₹5 coin — diameter (23 mm)": 23.0,
}
REFERENCE_UNCERTAINTY = {
    "Ruler (±1 mm)": 1.0,
    "Card edge (±2 mm)": 2.0,
    "Coin (±1.5 mm)": 1.5,
    "Rough estimate (±5 mm)": 5.0,
}

MESSAGES = {
    Status.CALIBRATION_REQUIRED: (
        "Size unclear",
        "Move a little closer or use plainer lighting, then recapture. (For exact millimetres, a printed marker or any known-length object can be used under Advanced.)",
    ),
    Status.POLICY_REQUIRED: (
        "Crop rule needed",
        "Pick the crop from the list; crops without a defensible size rule stay ungraded rather than guessed.",
    ),
    Status.NO_OBJECT: (
        "No clear item",
        "Capture the vegetable against a plain, contrasting surface.",
    ),
    Status.MULTIPLE_OBJECTS: ("Separate the items", "Capture one complete item at a time."),
    Status.OCCLUDED: (
        "Item partly hidden",
        "Keep every item fully inside the photo, not touching the edges.",
    ),
    Status.LOW_IMAGE_QUALITY: (
        "Recapture needed",
        "Use even light, hold the camera steady, and bring the item into focus.",
    ),
    Status.OUT_OF_DISTRIBUTION: (
        "Cannot verify",
        "Background too busy — use a plain surface.",
    ),
    Status.REVIEW_BORDERLINE: (
        "Borderline size",
        "The size is too close to a grade boundary for a reliable automatic label. Recapture closer, or grade manually.",
    ),
    Status.NOT_PRODUCE: ("Cannot verify produce", "Capture a real vegetable or leaf clearly."),
    Status.MODEL_UNAVAILABLE: (
        "Model unavailable",
        "No validated model is available for this input.",
    ),
    Status.INVALID_INPUT: ("Invalid input", "Choose a readable image or capture a new frame."),
}


def result_html(title: str, message: str) -> str:
    return f'<div class="result" role="status" aria-live="polite"><h2>{html.escape(title)}</h2><p>{html.escape(message)}</p></div>'


READY = result_html(
    "Ready",
    "Upload one item on a plain surface, choose the crop, and either use the printed marker or click a known-length reference (ruler, card, coin) to set scale.",
)


def _vegetable_name(result):
    """Best available vegetable name for titles (upper) and prose (lower)."""
    if result.policy is not None and result.policy.crop_name:
        return result.policy.crop_name.strip(), result.policy.crop_name.strip().lower()
    prediction = result.species_prediction or {}
    for key in ("crop_name", "species_key"):
        value = prediction.get(key)
        if value:
            name = str(value).replace("_", " ").strip()
            return name.upper(), name.lower()
    return "VEGETABLE", "vegetable"


def _plural(name_lower: str, count: int) -> str:
    if count == 1:
        return name_lower
    if name_lower.endswith(("o", "o ")) and name_lower.strip().lower() in (
        "potato", "tomato", "mango"
    ):
        return name_lower + "es"
    return name_lower + "s"


def present_result(result, session=None):
    title, message = MESSAGES.get(
        result.status, ("Cannot verify", "No verified size result is available.")
    )
    if result.status not in (Status.ACCEPTED, Status.MULTIPLE_OBJECTS) and any(
        w.startswith("AUTO_CROP_UNRESOLVED") for w in result.warnings
    ):
        title = "Crop unidentified"
        message = (
            "Couldn't identify the vegetable. Pick the crop from the list — or "
            "put several items in the photo: they'll be graded SMALL/MEDIUM/BIG "
            "relative to each other without needing the crop name."
        )
    elif result.status != Status.ACCEPTED and result.appearance_estimate is not None:
        estimate = result.appearance_estimate
        title = f"Estimated {estimate['appearance_size_class']} (uncalibrated)"
        message = (
            f"Detected {estimate['species_key'].replace('_', ' ')}; its footprint is "
            f"{estimate['area_ratio_to_median']:.2f}x the typical size in real-photo "
            f"references -> estimated {estimate['appearance_size_class']}. This is an "
            "appearance comparison, not millimetres: framing and distance change it. "
            "Add the printed marker or any known-length reference for measured grading."
        )
    elif result.status != Status.ACCEPTED and any(
        w.startswith("APPEARANCE_PRIORS_MISSING") for w in result.warnings
    ):
        title = "Scale needed (no size references for this crop)"
        message = (
            "This photo has no physical scale, and this crop has no real-photo size "
            "references yet, so neither measured nor estimated grading is possible. "
            "Use the printed marker, a clicked known-length reference, or pick a crop "
            "with published references."
        )
    elif result.status == Status.MULTIPLE_OBJECTS and any(
        r.startswith("BATCH_SPLIT:") for r in result.reasons
    ):
        small = sum(
            1 for i in result.instances
            if i.decision and i.decision.size_class and i.decision.size_class.value == "SMALL"
        )
        big = sum(
            1 for i in result.instances
            if i.decision and i.decision.size_class and i.decision.size_class.value == "BIG"
        )
        medium = sum(
            1 for i in result.instances
            if i.decision and i.decision.accepted and i.decision.size_class is None
        )
        parts = []
        if small:
            parts.append(f"{small} SMALL")
        if medium:
            parts.append(f"{medium} MEDIUM")
        if big:
            parts.append(f"{big} BIG")
        total = small + medium + big
        name_upper, name_lower = _vegetable_name(result)
        title = f"{name_upper} · {' · '.join(parts)}" if parts else "Batch result"
        message = (
            f"{total} {_plural(name_lower, total)} in this photo graded: "
            f"{' · '.join(parts)}. Each item is outlined and labelled with its "
            "size on the output image."
        )
        if any(
            i.decision and any(r.startswith("BATCH_RELATIVE:") for r in i.decision.reasons)
            for i in result.instances
        ):
            message += " Sizes are relative to this batch — no scale in the photo."
        else:
            message += " Graded against published size references using estimated scale."
        if any(w.startswith("CROP_UNRESOLVED") for w in result.warnings):
            message += " Vegetable not identified: grades are by size only."
    elif result.status == Status.ACCEPTED and (
        result.size_class is not None or result.size_band is not None
    ):
        measurement = next(
            (i.measurement for i in result.instances if i.measurement is not None), None
        )
        crop = result.policy.crop_name if result.policy else "the crop rule"
        band = (result.size_band or "").lower()
        word = {"1": "small", "2": "medium", "3": "big", "small": "small",
                "medium": "medium", "large": "big"}.get(band)
        if not word and result.size_class:
            word = result.size_class.value.lower()
        number = {"small": 1, "medium": 2, "big": 3}.get(word)
        name_upper, _ = _vegetable_name(result)
        title = f"{name_upper} — {word.upper()} ({number})" if number else f"{name_upper} — {(word or '').upper()}"
        message = (
            f"{crop}: {measurement.value_mm:.0f} ± {measurement.uncertainty_mm:.0f} mm "
            f"→ {word.upper()}. "
        )
        if any(w.startswith("DISTANCE_ASSUMED") for w in result.warnings):
            message += "Scale estimated from assumed camera distance."
        else:
            message += "Graded against published size references."
        if session is not None:
            stability = session.observe(result)
            if not stability["stable"]:
                title, message = "Hold steady", "Waiting for consistent measurements."
    elif session is not None:
        session.observe(result)
    image = bgr_to_rgb(result.annotated_image) if result.annotated_image is not None else None
    return image, result_html(title, message)


def inspect_image_ui(path, crop_name, marker_side_mm, ref_points, ref_length_mm, ref_uncertainty, station_mode, batch_mode=False):
    # batch_mode is kept for compatibility; the engine now engages batch
    # grading automatically whenever several items are detected in frame.
    if not path:
        return None, result_html("Image needed", "Upload or paste an image first.")
    crop_key = "auto" if crop_name == AUTO_CROP_CHOICE else CROP_CHOICES.get(crop_name or "")
    if crop_key is None:
        return None, result_html(
            "Crop needed", "Choose which vegetable is in the image; grades are crop-specific."
        )
    try:
        side = float(marker_side_mm) if marker_side_mm else None
    except (TypeError, ValueError):
        side = None
    reference = None
    try:
        length = float(ref_length_mm) if ref_length_mm else None
    except (TypeError, ValueError):
        length = None
    if ref_points and len(ref_points) == 2 and length and length > 0:
        (ax, ay), (bx, by) = ref_points[0], ref_points[1]
        reference = (float(ax), float(ay), float(bx), float(by))
    uncertainty = REFERENCE_UNCERTAINTY.get(ref_uncertainty, 2.0)
    request = InferenceRequest(
        crop_key=crop_key,
        marker_side_mm=side,
        demo_policy=False,
        reference_points_px=reference,
        reference_length_mm=length if reference else None,
        reference_uncertainty_mm=uncertainty,
        station_mode=bool(station_mode),
        station_id=STATION_ID if station_mode else None,
        batch_mode=bool(batch_mode),
        # No scale of any kind? Fall back to appearance priors (12 crops) and
        # then to a distance-assumed estimated scale so a single vegetable
        # always gets a verdict or an explicit review - never a demand for scale.
        allow_appearance_estimate=True,
        allow_scale_estimate=True,
    )
    try:
        decoded = load_image_file(
            path,
            max_pixels=SERVICE.settings.max_image_pixels,
            max_bytes=SERVICE.settings.max_upload_bytes,
        )
        context = MediaContext(
            source_type=SourceType.IMAGE, source_id=Path(path).name, source_sha256=decoded.sha256
        )
        return present_result(SERVICE.inspect_frame(decoded.bgr, context, request))
    except (MediaError, OSError) as exc:
        return None, result_html("Cannot read image", str(exc))
    except Exception:
        LOGGER.exception("Image inspection failed")
        return None, result_html("Inspection failed", "No size was assigned. Try another capture.")


def capture_reference_point(evt: gr.SelectData, points):
    current = list(points or [])
    click = (float(evt.index[0]), float(evt.index[1]))
    current = current[-1:] if len(current) >= 2 else current
    current.append(click)
    label = _reference_points_html(current)
    return current, label


def clear_reference_points():
    return [], _reference_points_html([])


def _reference_points_html(points):
    if len(points) < 2:
        return (
            "<div class='result'><h2>Reference: point "
            f"{len(points) + 1} of 2</h2><p>Click one end of the known-length "
            "object in the image.</p></div>"
        )
    (ax, ay), (bx, by) = points[0], points[1]
    span = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    return (
        "<div class='result'><h2>Reference set</h2><p>"
        f"Points ({ax:.0f}, {ay:.0f}) → ({bx:.0f}, {by:.0f}), span {span:.0f} px. "
        "Enter the true length of that span and press Check size.</p></div>"
    )


def live_inspect_ui(frame, session, crop_name, station_mode):
    session = session or LiveSession()
    if frame is None:
        session.invalidate_current_frame()
        return (
            None,
            result_html("Camera idle", "Start the camera and allow browser access."),
            session,
        )
    request = InferenceRequest(
        crop_key="auto" if crop_name == AUTO_CROP_CHOICE else CROP_CHOICES.get(crop_name or ""),
        station_mode=bool(station_mode),
        station_id="camera-station" if station_mode else None,
        allow_appearance_estimate=True,
        allow_scale_estimate=True,
    )
    try:
        result = SERVICE.inspect_frame(
            rgb_to_bgr(frame), MediaContext(source_type=SourceType.WEBCAM), request
        )
        image, text = present_result(result, session)
        return image, text, session
    except Exception:
        LOGGER.exception("Camera inspection failed")
        session.invalidate_current_frame()
        return (
            None,
            result_html("Capture failed", "No size was assigned. Restart the camera."),
            session,
        )


def process_video_ui(path, crop_name, station_mode, progress=gr.Progress()):
    if not path:
        return None, result_html("Video needed", "Upload or record a video first.")
    try:
        report = process_video(
            path,
            service=SERVICE,
            request=InferenceRequest(
                crop_key="auto" if crop_name == AUTO_CROP_CHOICE else CROP_CHOICES.get(crop_name or ""),
                station_mode=bool(station_mode),
                station_id="video-station" if station_mode else None,
                allow_appearance_estimate=True,
                allow_scale_estimate=True,
            ),
            settings=SERVICE.settings,
            output_dir=OUTPUT_DIR,
            progress=lambda fraction, description: progress(fraction, desc=description),
        )
        if report.status not in {"COMPLETED", "COMPLETED_WITH_REVIEW"}:
            return None, result_html(
                "Video incomplete", "No completed grading result is available. Try a shorter video."
            )
        counts = report.summary.get("size_class_counts_per_frame_observation", {})
        accepted = sum(counts.values())
        message = (
            f"SMALL: {counts.get('SMALL', 0)}; BIG: {counts.get('BIG', 0)}. Counts are frame observations, not unique items."
            if accepted
            else "No verified size labels. Physical scale and an approved SMALL/BIG rule are required."
        )
        return report.output_video, result_html("Video processed", message)
    except (VideoError, OSError) as exc:
        return None, result_html("Cannot process video", str(exc))
    except Exception:
        LOGGER.exception("Video inspection failed")
        return None, result_html("Inspection failed", "No size was assigned. Try another video.")


CSS = """
:root { --station-ink:#172f25; --station-paper:#faf9f5; --station-line:#b4bdb5; }
.gradio-container { max-width:1120px!important; background:var(--station-paper)!important; }
#station-title { border-bottom:2px solid var(--station-ink); padding-bottom:12px; }
#station-title h1 { font-size:26px!important; color:var(--station-ink)!important; }
.result { min-height:112px; border-left:4px solid var(--station-ink); padding:12px 18px; background:#eef2ec; color:#172f25; overflow-wrap:anywhere; }
.result h2 { margin:0 0 8px; font-size:26px; color:#172f25!important; }
.result p { margin:0; color:#172f25!important; }
.gradio-container label.float { background:#ffffff!important; color:#172f25!important; }
.gradio-container [role=tab] { color:#172f25!important; }
.gradio-container [role=tab][aria-selected=true] { background:#214d37!important; color:#ffffff!important; }
.gradio-container footer, .gradio-container footer a, .gradio-container footer button { color:#334c40!important; }
.gradio-container button.padded { background:#ffffff!important; color:#172f25!important; }
.gradio-container .block > button > .image-container { height:100%!important; }
.gradio-container .block > button > .image-container img { width:100%!important; height:100%!important; object-fit:contain!important; }
button:focus-visible, input:focus-visible, [tabindex]:focus-visible { outline:3px solid #8c4a0a!important; outline-offset:3px; }
@media(max-width:600px) { .gradio-container { padding:12px!important; } #station-title h1,.result h2 { font-size:22px!important; } }
@media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none!important; transition:none!important; scroll-behavior:auto!important; } }
"""

# Gradio 4.36 omits alt text on uploaded previews and explicit button types.
ACCESSIBILITY_JS = """() => {
    const root = document.querySelector('.gradio-container') || document;
    const annotate = () => {
        root.querySelectorAll('img:not([alt])').forEach(image => {
            const label = image.closest('.block')?.querySelector('label')?.textContent?.trim();
            image.setAttribute('alt', `${label || 'Inspection image'} preview`);
        });
        root.querySelectorAll('button:not([type])').forEach(button => button.type = 'button');
    };
    annotate();
    new MutationObserver(annotate).observe(root, {childList:true, subtree:true});
}"""


def build_app():
    theme = gr.themes.Base(primary_hue="green", neutral_hue="stone").set(
        body_background_fill="#faf9f5",
        body_background_fill_dark="#faf9f5",
        body_text_color="#172f25",
        body_text_color_dark="#172f25",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_label_text_color="#172f25",
        block_label_text_color_dark="#172f25",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        button_primary_background_fill="#214d37",
        button_primary_background_fill_dark="#214d37",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
    )
    with gr.Blocks(
        title="Growize · Size inspection", theme=theme, css=CSS, js=ACCESSIBILITY_JS
    ) as app:
        gr.Markdown("# Vegetable size inspection", elem_id="station-title")
        crop_choices = list(CROP_CHOICES.keys())
        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        source = gr.Image(
                            type="filepath",
                            sources=["upload", "clipboard"],
                            label="Image (click a known-length object to set reference points)",
                            height=320,
                        )
                        crop = gr.Dropdown(
                            choices=CROP_UI_CHOICES,
                            value=AUTO_CROP_CHOICE,
                            label="Crop (auto-detected by default; single items and batches both work)",
                            filterable=True,
                        )
                        with gr.Accordion("Advanced (optional scale setup)", open=False):
                            marker_side = gr.Number(
                                label="Printed marker side (mm) — measured once at station setup",
                                value=None,
                            )
                            station_toggle = gr.Checkbox(
                                value=True,
                                label="Station mode (fixed camera): calibrate once with the marker visible, then items may cover it",
                            )
                            ref_points_state = gr.State([])
                            ref_points_label = gr.HTML(_reference_points_html([]))
                            ref_preset = gr.Dropdown(
                                choices=list(REFERENCE_PRESETS.keys()),
                                value=list(REFERENCE_PRESETS.keys())[0],
                                label="Known length of the clicked span",
                            )
                            ref_length = gr.Number(label="True length (mm)", value=None)
                            ref_uncertainty = gr.Dropdown(
                                choices=list(REFERENCE_UNCERTAINTY.keys()),
                                value=list(REFERENCE_UNCERTAINTY.keys())[0],
                                label="How accurate is that length?",
                            )
                            ref_clear = gr.Button("Clear reference points")
                        run = gr.Button("Check size", variant="primary")
                        clear = gr.Button("Clear")
                    with gr.Column():
                        output = gr.Image(label="Output", interactive=False, height=320)
                        status = gr.HTML(READY)
                source.select(
                    capture_reference_point,
                    [ref_points_state],
                    [ref_points_state, ref_points_label],
                    queue=False,
                )
                ref_clear.click(
                    clear_reference_points,
                    outputs=[ref_points_state, ref_points_label],
                    queue=False,
                )
                ref_preset.change(
                    lambda preset: REFERENCE_PRESETS.get(preset),
                    inputs=[ref_preset],
                    outputs=[ref_length],
                    queue=False,
                )
                run.click(
                    inspect_image_ui,
                    [source, crop, marker_side, ref_points_state, ref_length, ref_uncertainty, station_toggle],
                    [output, status],
                    concurrency_id="inspection",
                    concurrency_limit=1,
                )
                clear.click(
                    lambda: (None, None, READY, [], _reference_points_html([])),
                    outputs=[source, output, status, ref_points_state, ref_points_label],
                    queue=False,
                )
            with gr.Tab("Camera"):
                session = gr.State(LiveSession())
                camera_crop = gr.Dropdown(
                    choices=CROP_UI_CHOICES,
                    value=crop_choices[0] if crop_choices else None,
                    label="Crop",
                )
                camera_station = gr.Checkbox(
                    value=True,
                    label="Station mode (fixed camera): calibrate once with the marker visible",
                )
                with gr.Row():
                    camera = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        streaming=True,
                        label="Live camera",
                        height=320,
                    )
                    with gr.Column():
                        camera_output = gr.Image(label="Output", interactive=False, height=320)
                        camera_status = gr.HTML(
                            result_html("Camera idle", "Allow camera access, then start capture.")
                        )
                camera.stream(
                    live_inspect_ui,
                    [camera, session, camera_crop, camera_station],
                    [camera_output, camera_status, session],
                    concurrency_id="inspection",
                    concurrency_limit=1,
                    trigger_mode="always_last",
                )
                camera.clear(
                    lambda: (
                        None,
                        result_html("Camera idle", "Start capture to inspect another item."),
                        LiveSession(),
                    ),
                    outputs=[camera_output, camera_status, session],
                    concurrency_id="inspection",
                    concurrency_limit=1,
                )
            with gr.Tab("Video"):
                video_crop = gr.Dropdown(
                    choices=CROP_UI_CHOICES,
                    value=crop_choices[0] if crop_choices else None,
                    label="Crop",
                )
                video_station = gr.Checkbox(
                    value=True,
                    label="Station mode (fixed camera): calibrate once with the marker visible",
                )
                with gr.Row():
                    with gr.Column():
                        video = gr.Video(sources=["upload", "webcam"], label="Video", height=320)
                        video_run = gr.Button("Check video", variant="primary")
                    with gr.Column():
                        replay = gr.Video(label="Output", interactive=False, height=320)
                        video_status = gr.HTML(
                            result_html("Ready", "Upload or record a video, up to two minutes.")
                        )
                video_run.click(
                    process_video_ui,
                    [video, video_crop, video_station],
                    [replay, video_status],
                    concurrency_id="inspection",
                    concurrency_limit=1,
                )
    return app


demo = build_app()


def launch_app():
    demo.queue(default_concurrency_limit=1, max_size=16).launch(
        server_name=os.environ.get(
            "GRADIO_SERVER_NAME", os.environ.get("GROWIZE_HOST", "127.0.0.1")
        ),
        server_port=int(os.environ.get("PORT", os.environ.get("GROWIZE_PORT", "7860"))),
        share=False,
        inbrowser=False,
        show_error=False,
        show_api=False,
        max_file_size=300_000_000,
        allowed_paths=[str(OUTPUT_DIR)],
    )


if __name__ == "__main__":
    launch_app()
