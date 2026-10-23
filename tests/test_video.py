"""Bounded MP4 processing, output verification and count semantics."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import pytest

from growize_vision.inference import InferenceRequest
from growize_vision.video import (
    VideoError,
    inspect_video_metadata,
    process_video,
    video_contact_sheet,
)


def request() -> InferenceRequest:
    return InferenceRequest(
        crop_key="potato",
        marker_side_mm=50.0,
        marker_measurement_uncertainty_mm=0.5,
        expected_marker_id=0,
        demo_policy=True,
    )


def test_cancelling_repeated_filename_preserves_completed_run(service, demo_dir, tmp_path):
    source = demo_dir / "synthetic_potato_video.mp4"
    first = process_video(source, service=service, request=request(), settings=service.settings,
                          output_dir=tmp_path, max_frames=1)
    assert first.output_video is not None
    original = Path(first.output_video).read_bytes()
    second = process_video(source, service=service, request=request(), settings=service.settings,
                           output_dir=tmp_path, cancel_check=lambda: True)
    assert second.cancelled
    assert second.output_video is None
    assert Path(first.output_video).read_bytes() == original
    assert first.exports["json"] != second.exports["json"]


def test_failed_replay_is_not_reported_as_completed(service, demo_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("growize_vision.video._verify_video", lambda path: (False, "simulated codec failure"))
    report = process_video(demo_dir / "synthetic_potato_video.mp4", service=service,
                           request=request(), settings=service.settings, output_dir=tmp_path, max_frames=1)
    assert report.status == "OUTPUT_FAILED"
    assert report.output_video is None
    assert Path(report.exports["json"]).exists()


def test_synthetic_video_metadata_is_read(demo_dir) -> None:
    metadata = inspect_video_metadata(demo_dir / "synthetic_potato_video.mp4")
    assert metadata.width == 1400
    assert metadata.height == 1000
    assert metadata.fps == pytest.approx(10.0)
    assert metadata.frame_count == 30
    assert metadata.duration_s == pytest.approx(3.0)
    assert not metadata.fps_estimated
    import hashlib

    assert metadata.to_dict()["sha256"] == hashlib.sha256((demo_dir / "synthetic_potato_video.mp4").read_bytes()).hexdigest()


def test_video_limits_are_enforced(demo_dir) -> None:
    path = demo_dir / "synthetic_potato_video.mp4"
    with pytest.raises(VideoError, match="duration"):
        inspect_video_metadata(path, max_duration_s=1.0)
    with pytest.raises(VideoError, match="bytes"):
        inspect_video_metadata(path, max_bytes=1000)
    with pytest.raises(VideoError, match="limit"):
        inspect_video_metadata(path, max_width=800)
    with pytest.raises(VideoError, match="does not exist"):
        inspect_video_metadata(demo_dir / "missing.mp4")


def test_process_video_writes_playable_h264_and_honest_counts(
    service, demo_dir, tmp_path
) -> None:
    report = process_video(
        demo_dir / "synthetic_potato_video.mp4",
        service=service,
        request=request(),
        settings=service.settings,
        output_dir=tmp_path,
        sample_fps=2.0,
        max_frames=2,
    )
    assert report.status in {"COMPLETED", "COMPLETED_WITH_REVIEW"}
    assert report.sampled_frames == 2
    assert report.output_video is not None
    assert Path(report.output_video).is_file()
    assert report.codec.startswith("h264")
    assert report.summary["unique_vegetable_count"] is None
    assert "per-frame observations" in report.summary["count_semantics"]
    assert report.summary["per_frame_observations"] >= 1
    assert len(report.frame_results) == 2
    assert report.exports and Path(report.exports["json"]).is_file()
    assert Path(report.exports["csv"]).is_file()

    payload = json.loads(Path(report.exports["json"]).read_text(encoding="utf-8"))
    assert payload["count_semantics"] == "per-frame observations"
    assert payload["unique_vegetable_count"] is None
    assert payload["exports"]["csv"] == report.exports["csv"]
    assert payload["frames"]

    cap = cv2.VideoCapture(str(report.output_video))
    assert cap.isOpened()
    ok, frame = cap.read()
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert ok and frame is not None and frame.size > 0
    assert count >= 1

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(report.output_video)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = probe.stderr
    assert "Video:" in output
    assert "Audio:" not in output


def test_cancelled_video_discards_partial_output(service, demo_dir, tmp_path) -> None:
    report = process_video(
        demo_dir / "synthetic_potato_video.mp4",
        service=service,
        request=request(),
        settings=service.settings,
        output_dir=tmp_path,
        max_frames=5,
        cancel_check=lambda: True,
    )
    assert report.status == "CANCELLED"
    assert report.cancelled is True
    assert report.output_video is None
    assert report.sampled_frames == 0
    assert not (tmp_path / "video" / "synthetic_potato_video" / "annotated.partial.mp4").exists()


def test_video_reprocessing_replaces_stale_output(service, demo_dir, tmp_path) -> None:
    first = process_video(
        demo_dir / "synthetic_potato_video.mp4",
        service=service,
        request=request(),
        settings=service.settings,
        output_dir=tmp_path,
        sample_fps=2.0,
        max_frames=1,
    )
    assert first.output_video is not None
    first_size = Path(first.output_video).stat().st_size
    second = process_video(
        demo_dir / "synthetic_potato_video.mp4",
        service=service,
        request=request(),
        settings=service.settings,
        output_dir=tmp_path,
        sample_fps=2.0,
        max_frames=1,
    )
    assert second.output_video is not None
    assert Path(second.output_video).is_file()
    assert Path(second.output_video).stat().st_size > 0
    assert Path(second.output_video).stat().st_size != first_size or first_size > 0


def test_contact_sheet_is_written(service, demo_dir, tmp_path) -> None:
    report = process_video(
        demo_dir / "synthetic_potato_video.mp4",
        service=service,
        request=request(),
        settings=service.settings,
        output_dir=tmp_path,
        sample_fps=2.0,
        max_frames=1,
    )
    assert report.output_video is not None
    output = tmp_path / "contact.jpg"
    path = video_contact_sheet(report.output_video, output, max_tiles=4)
    image = cv2.imread(path)
    assert image is not None
    assert image.shape[0] > 0 and image.shape[1] > 0
