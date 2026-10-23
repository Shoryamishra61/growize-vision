"""Bounded MP4 ingestion, timestamp sampling and annotated replay."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .config import InferenceSettings
from .domain import MediaContext, SourceType, Status, new_id
from .exports import export_video_report, flatten_result_rows
from .inference import InferenceRequest, InferenceService
from .media import safe_export_stem


class VideoError(ValueError):
    pass


@dataclass
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float | None
    fps_estimated: bool = False
    warnings: list[str] = field(default_factory=list)
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_s": round(self.duration_s, 3) if self.duration_s else None,
            "fps_estimated": self.fps_estimated,
            "warnings": list(self.warnings),
            "sha256": self.sha256,
        }


@dataclass
class VideoReport:
    status: str
    source_path: str
    output_video: str | None
    metadata: VideoMetadata | None
    sampled_frames: int
    frame_results: list[dict[str, Any]]
    frame_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    warnings: list[str]
    exports: dict[str, str] = field(default_factory=dict)
    processing_ms: float = 0.0
    codec: str = ""
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source_path": self.source_path,
            "output_video": self.output_video,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "sampled_frames": self.sampled_frames,
            "summary": self.summary,
            "warnings": self.warnings,
            "source_sha256": self.metadata.sha256 if self.metadata else None,
            "exports": self.exports,
            "processing_ms": round(self.processing_ms, 3),
            "codec": self.codec,
            "cancelled": self.cancelled,
            "count_semantics": "per-frame observations",
            "unique_vegetable_count": None,
            "frames": self.frame_results,
        }


def inspect_video_metadata(
    path: Path | str,
    *,
    max_bytes: int = 300_000_000,
    max_width: int = 1920,
    max_height: int = 1080,
    max_duration_s: float = 120.0,
) -> VideoMetadata:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise VideoError("Video file does not exist.")
    if path.stat().st_size > max_bytes:
        raise VideoError(
            f"Video is {path.stat().st_size:,} bytes; limit is {max_bytes:,} bytes."
        )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoError("Video decoder could not open the file.")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        warnings_list: list[str] = []
        fps_estimated = False
        if not math.isfinite(fps) or fps <= 0 or fps > 240:
            fps = 25.0
            fps_estimated = True
            warnings_list.append(
                "Container FPS is missing/invalid; sampling timestamp is estimated "
                "at 25 fps."
            )
        if frame_count <= 0:
            frame_count = 0
            warnings_list.append("Container frame count is unavailable.")
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
            else:
                raise VideoError("Video contains no decodable frames.")
        if width > max_width or height > max_height:
            raise VideoError(
                f"Video is {width}x{height}; limit is {max_width}x{max_height}. "
                "Transcode/downscale before upload."
            )
        duration = frame_count / fps if frame_count > 0 else None
        if duration is not None and duration > max_duration_s:
            raise VideoError(
                f"Video duration {duration:.1f} s exceeds the configured "
                f"{max_duration_s:.1f} s limit."
            )
        source_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                source_digest.update(chunk)
        return VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_s=duration,
            fps_estimated=fps_estimated,
            warnings=warnings_list,
            sha256=source_digest.hexdigest(),
        )
    finally:
        cap.release()


def _even_size(width: int, height: int) -> tuple[int, int]:
    w = max(2, width - (width % 2))
    h = max(2, height - (height % 2))
    return w, h


def _open_browser_writer(path: Path, fps: float, width: int, height: int) -> tuple[Any, str]:
    """Prefer H.264/libx264 via bundled imageio-ffmpeg for browser playback."""
    width, height = _even_size(width, height)
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            str(path),
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        return writer, "h264/libx264"
    except Exception as exc:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise VideoError(
                "No tested MP4 writer is available (libx264 and mp4v both failed)."
            ) from exc
        return writer, "mpeg4/mp4v"


def _append_frame(writer: Any, codec: str, frame_bgr: np.ndarray) -> None:
    h, w = frame_bgr.shape[:2]
    ew, eh = _even_size(w, h)
    if (w, h) != (ew, eh):
        frame_bgr = cv2.resize(frame_bgr, (ew, eh), interpolation=cv2.INTER_AREA)
    if codec.startswith("h264"):
        writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    else:
        writer.write(frame_bgr)


def _close_writer(writer: Any, codec: str) -> None:
    try:
        if codec.startswith("h264"):
            writer.close()
        else:
            writer.release()
    except Exception:
        pass


def _verify_video(path: Path) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size < 1024:
        return False, "output file is missing or empty"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False, "output file cannot be reopened by OpenCV"
    ok, frame = cap.read()
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok or frame is None or frame.size == 0:
        return False, "output file has no decodable frames"
    if count <= 0:
        return False, "output file reports zero frames"
    return True, f"{count} frames"


def process_video(
    source_path: Path | str,
    *,
    service: InferenceService,
    request: InferenceRequest,
    settings: InferenceSettings,
    output_dir: Path | str = "outputs",
    sample_fps: float | None = None,
    max_frames: int | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> VideoReport:
    started = time.perf_counter()
    source = Path(source_path)
    meta = inspect_video_metadata(
        source,
        max_bytes=settings.video.max_file_bytes,
        max_width=settings.video.max_width,
        max_height=settings.video.max_height,
        max_duration_s=settings.video.max_duration_s,
    )
    requested_fps = float(sample_fps or settings.video.default_sample_fps)
    if not math.isfinite(requested_fps) or requested_fps <= 0:
        requested_fps = settings.video.default_sample_fps
    effective_fps = min(requested_fps, meta.fps)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise VideoError("Video decoder could not open the file for processing.")
    output_folder = Path(output_dir) / "video" / f"{safe_export_stem(source.stem)}_{new_id('run')}"
    output_folder.mkdir(parents=True, exist_ok=False)
    partial_path = output_folder / "annotated.partial.mp4"
    final_path = output_folder / "annotated.mp4"
    writer = None
    codec = ""
    frame_results: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    warnings_list = list(meta.warnings)
    status_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {"SMALL": 0, "BIG": 0}
    sampled = 0
    instance_observations = 0
    accepted_observations = 0
    cancelled = False
    next_sample_t = 0.0
    frame_index = 0
    duration_limit = settings.video.max_duration_s
    max_samples = max_frames or settings.video.max_frames_to_sample
    try:
        while True:
            if cancel_check and cancel_check():
                cancelled = True
                warnings_list.append("Processing cancelled by the operator.")
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            size = frame.shape[:2]
            if size[0] > settings.video.max_height or size[1] > settings.video.max_width:
                raise VideoError("Decoded frame exceeds configured dimensions.")
            # Timestamp is based on container FPS.  Missing FPS is reported as
            # estimated; we do not pretend it is exact.
            timestamp_s = frame_index / meta.fps
            if timestamp_s > duration_limit:
                warnings_list.append(
                    f"Stopped at configured duration limit {duration_limit:.1f} s."
                )
                break
            if timestamp_s + 1e-9 >= next_sample_t:
                context = MediaContext(
                    source_type=SourceType.VIDEO,
                    source_id=source.name,
                    device="video-upload",
                    frame_id=frame_index,
                    timestamp_s=float(timestamp_s),
                    batch_id=request.notes or source.stem,
                    session_id="video",
                    source_sha256=meta.sha256,
                )
                result = service.inspect_frame(frame, context, request)
                annotated = result.annotated_image if result.annotated_image is not None else frame
                if writer is None:
                    writer, codec = _open_browser_writer(
                        partial_path,
                        effective_fps,
                        annotated.shape[1],
                        annotated.shape[0],
                    )
                _append_frame(writer, codec, annotated)
                sampled += 1
                frame_results.append(
                    {
                        "frame_id": frame_index,
                        "timestamp_s": round(timestamp_s, 4),
                        "status": result.status.value,
                        "size_class": result.size_class.value if result.size_class else None,
                        "instance_count": len(result.instances),
                        "processing_ms": round(result.processing_ms, 3),
                        "species_prediction": result.species_prediction,
                        "species_prediction_scope": "scene" if result.species_prediction else None,
                        "appearance_estimate": result.appearance_estimate,
                        "model_sha": result.model_sha,
                        "pipeline_sha256": result.pipeline_sha256,
                        "processed_frame_sha256": result.processed_frame_sha256,
                        "instances": [
                            {
                                "instance_id": inst.instance_id,
                                "crop_key": inst.crop_key,
                                "status": inst.decision.status.value
                                if inst.decision
                                else "NO_DECISION",
                                "size_class": inst.decision.size_class.value
                                if inst.decision and inst.decision.size_class
                                else None,
                                "measurement_mm": round(inst.measurement.value_mm, 3)
                                if inst.measurement
                                else None,
                                "uncertainty_mm": round(inst.measurement.uncertainty_mm, 3)
                                if inst.measurement
                                else None,
                                "threshold_mm": inst.decision.threshold_mm
                                if inst.decision
                                else None,
                            }
                            for inst in result.instances
                        ],
                    }
                )
                rows = flatten_result_rows([result])
                for row in rows:
                    row["video_frame_id"] = frame_index
                    row["video_timestamp_s"] = round(timestamp_s, 4)
                    frame_rows.append(row)
                status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1
                for inst in result.instances:
                    instance_observations += 1
                    if inst.decision and inst.decision.accepted and inst.decision.size_class:
                        accepted_observations += 1
                        class_counts[inst.decision.size_class.value] += 1
                if progress:
                    total_hint = (
                        min(int(math.ceil(meta.duration_s * effective_fps)), max_samples)
                        if meta.duration_s
                        else max_samples
                    )
                    progress(
                        min(0.99, sampled / max(1, total_hint)),
                        f"Processed {sampled} sampled frame(s) at {timestamp_s:.2f}s",
                    )
                next_sample_t += 1.0 / effective_fps
                if sampled >= max_samples:
                    warnings_list.append(
                        f"Stopped at configured sample limit {max_samples} frames."
                    )
                    break
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            _close_writer(writer, codec)

    if cancelled:
        if partial_path.exists():
            partial_path.unlink()
        report = VideoReport(
            status="CANCELLED",
            source_path=str(source),
            output_video=None,
            metadata=meta,
            sampled_frames=sampled,
            frame_results=frame_results,
            frame_rows=frame_rows,
            summary={
                "per_frame_observations": instance_observations,
                "accepted_per_frame_observations": accepted_observations,
                "unique_vegetable_count": None,
                "count_semantics": "per-frame observations",
            },
            warnings=warnings_list,
            processing_ms=(time.perf_counter() - started) * 1000.0,
            codec=codec,
            cancelled=True,
        )
        report.exports = export_video_report(
            report_payload=report.to_dict(), frame_rows=frame_rows,
            output_root=output_dir, stem=output_folder.name,
        )
        return report

    if sampled == 0 or writer is None:
        raise VideoError("No frames could be sampled/decoded from the video.")
    playable, check_reason = _verify_video(partial_path)
    if playable:
        partial_path.replace(final_path)
        output_video = str(final_path)
    else:
        warnings_list.append(
            f"Annotated MP4 failed playback verification ({check_reason}); "
            "the CSV/JSON results are still available. The partial video is retained "
            "for diagnosis and must not be presented as complete."
        )
        output_video = None
    summary = {
        "sampled_frames": sampled,
        "source_fps": round(meta.fps, 3),
        "sample_fps": round(effective_fps, 3),
        "per_frame_observations": instance_observations,
        "accepted_per_frame_observations": accepted_observations,
        "unique_vegetable_count": None,
        "count_semantics": (
            "per-frame observations; repeated frames of one object are NOT unique "
            "vegetables and no tracker-verified unique count is claimed"
        ),
        "status_counts": status_counts,
        "size_class_counts_per_frame_observation": class_counts,
    }
    report_status = "COMPLETED_WITH_REVIEW" if playable else "OUTPUT_FAILED"
    if playable and status_counts and all(k == Status.ACCEPTED.value for k in status_counts):
        report_status = "COMPLETED"
    report = VideoReport(
        status=report_status,
        source_path=str(source),
        output_video=output_video,
        metadata=meta,
        sampled_frames=sampled,
        frame_results=frame_results,
        frame_rows=frame_rows,
        summary=summary,
        warnings=warnings_list,
        processing_ms=(time.perf_counter() - started) * 1000.0,
        codec=codec,
    )
    exports = export_video_report(
        report_payload=report.to_dict(),
        frame_rows=frame_rows,
        output_root=output_dir,
        stem=output_folder.name,
    )
    report.exports = exports
    if progress:
        progress(1.0, "Video complete")
    return report


def video_contact_sheet(
    source_path: Path | str,
    output_path: Path | str,
    *,
    max_tiles: int = 6,
) -> str:
    """Make a six-frame contact sheet for diagnosis when MP4 playback is suspect."""
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise VideoError("Cannot open video for contact sheet.")
    frames: list[np.ndarray] = []
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count > 0:
        indices = np.linspace(0, max(0, count - 1), max_tiles, dtype=int)
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    else:
        while len(frames) < max_tiles:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    cap.release()
    if not frames:
        raise VideoError("No frames available for contact sheet.")
    tiles = []
    for frame in frames:
        thumb = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        tiles.append(thumb)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    sheet = np.vstack(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)
    return str(output)
