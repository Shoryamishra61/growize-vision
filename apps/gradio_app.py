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
# A species model trained with generated views is not eligible under the new
# real-original-only requirement. Do not load it or infer physical size from it.
REQUEST = InferenceRequest(crop_key=None, demo_policy=False, allow_appearance_estimate=False)

MESSAGES = {
    Status.CALIBRATION_REQUIRED: (
        "Scale needed",
        "Place a measured reference marker beside the item on the same flat surface. Its physical size must be configured before measurement.",
    ),
    Status.POLICY_REQUIRED: (
        "Reference average needed",
        "No measured reference average is available for this kind of item. A size label would be a guess.",
    ),
    Status.NO_OBJECT: (
        "No clear item",
        "Capture one complete vegetable or leaf against a plain, contrasting surface.",
    ),
    Status.MULTIPLE_OBJECTS: ("Separate the items", "Capture one complete item at a time."),
    Status.OCCLUDED: (
        "Item partly hidden",
        "Keep the entire item visible, away from the image edges and marker.",
    ),
    Status.LOW_IMAGE_QUALITY: (
        "Recapture needed",
        "Use even light, hold the camera steady, and bring the item into focus.",
    ),
    Status.OUT_OF_DISTRIBUTION: (
        "Cannot verify",
        "This scene is outside the validated capture conditions.",
    ),
    Status.REVIEW_BORDERLINE: (
        "Manual review needed",
        "The measurement is too close to the size boundary for a reliable automatic label.",
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
    "Validation incomplete",
    "SMALL/BIG grading is not released. Add an image to inspect capture conditions; no verified size model is available yet.",
)


def present_result(result, session=None):
    title, message = MESSAGES.get(
        result.status, ("Cannot verify", "No verified size result is available.")
    )
    if result.status == Status.ACCEPTED and result.size_class is not None:
        title = result.size_class.value
        message = "Measured against the configured size rule."
        if session is not None:
            stability = session.observe(result)
            if not stability["stable"]:
                title, message = "Hold steady", "Waiting for consistent measurements."
    elif session is not None:
        session.observe(result)
    image = bgr_to_rgb(result.annotated_image) if result.annotated_image is not None else None
    return image, result_html(title, message)


def inspect_image_ui(path):
    if not path:
        return None, result_html("Image needed", "Upload or paste an image first.")
    try:
        decoded = load_image_file(
            path,
            max_pixels=SERVICE.settings.max_image_pixels,
            max_bytes=SERVICE.settings.max_upload_bytes,
        )
        context = MediaContext(
            source_type=SourceType.IMAGE, source_id=Path(path).name, source_sha256=decoded.sha256
        )
        return present_result(SERVICE.inspect_frame(decoded.bgr, context, REQUEST))
    except (MediaError, OSError) as exc:
        return None, result_html("Cannot read image", str(exc))
    except Exception:
        LOGGER.exception("Image inspection failed")
        return None, result_html("Inspection failed", "No size was assigned. Try another capture.")


def live_inspect_ui(frame, session):
    session = session or LiveSession()
    if frame is None:
        session.invalidate_current_frame()
        return (
            None,
            result_html("Camera idle", "Start the camera and allow browser access."),
            session,
        )
    try:
        result = SERVICE.inspect_frame(
            rgb_to_bgr(frame), MediaContext(source_type=SourceType.WEBCAM), REQUEST
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


def process_video_ui(path, progress=gr.Progress()):
    if not path:
        return None, result_html("Video needed", "Upload or record a video first.")
    try:
        report = process_video(
            path,
            service=SERVICE,
            request=REQUEST,
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
        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        source = gr.Image(
                            type="filepath",
                            sources=["upload", "clipboard"],
                            label="Image",
                            height=320,
                        )
                        run = gr.Button("Check size", variant="primary")
                        clear = gr.Button("Clear")
                    with gr.Column():
                        output = gr.Image(label="Output", interactive=False, height=320)
                        status = gr.HTML(READY)
                run.click(
                    inspect_image_ui,
                    source,
                    [output, status],
                    concurrency_id="inspection",
                    concurrency_limit=1,
                )
                clear.click(
                    lambda: (None, None, READY), outputs=[source, output, status], queue=False
                )
            with gr.Tab("Camera"):
                session = gr.State(LiveSession())
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
                    [camera, session],
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
                    video,
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
