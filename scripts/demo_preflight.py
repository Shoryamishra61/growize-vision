"""Verify the local demo stack before a handoff or browser smoke test.

This is a readiness check for the *demo environment*, not a production or field
readiness certificate.  It verifies assets, configuration, refusal contracts,
video tooling, and optionally a real webcam frame.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
from typing import Any

from growize_vision.config import load_inference_settings, load_policies
from growize_vision.domain import MediaContext, SourceType
from growize_vision.evidence import (
    PROJECT_ROOT,
    environment_metadata,
    load_json,
    relative_to_project,
    write_json,
)
from growize_vision.inference import InferenceRequest, InferenceService
from growize_vision.media import load_image_file
from growize_vision.video import inspect_video_metadata, process_video

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "demo" / "demo_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "preflight.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
UI_PACKAGES = (
    "gradio",
    "gradio-client",
    "huggingface-hub",
    "fastapi",
    "starlette",
    "pydantic",
    "imageio-ffmpeg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Open the default webcam for one frame (may prompt for permission).",
    )
    parser.add_argument(
        "--full-video",
        action="store_true",
        help="Process the demo MP4 and verify H.264 output.",
    )
    return parser.parse_args()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    settings = load_inference_settings()
    policies = load_policies()
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    warnings: list[str] = []

    def add(
        name: str,
        ok: bool | None,
        detail: Any,
        *,
        required: bool = True,
        warning: str | None = None,
    ) -> None:
        status = "NOT_RUN" if ok is None else ("PASS" if ok else "FAIL")
        checks.append(
            {"name": name, "status": status, "required": required, "detail": detail}
        )
        if ok is False and required:
            failures.append(name)
        if warning:
            warnings.append(warning)

    # 1. Package/app presence and pinned UI stack.
    add(
        "inference_package_import",
        True,
        "growize_vision imports and configuration loads.",
    )
    app_path = PROJECT_ROOT / "apps" / "gradio_app.py"
    add("gradio_app_file", app_path.is_file(), relative_to_project(app_path))
    versions = {package: _version(package) for package in UI_PACKAGES}
    add(
        "ui_dependencies",
        all(version is not None for version in versions.values()),
        ", ".join(f"{name}=={version or 'MISSING'}" for name, version in versions.items()),
    )

    # 2. Configuration truthfulness.
    null_thresholds = [
        crop.key
        for crop in (policies.ontology(key) for key in policies.crop_keys)
        if policies.policy(crop.key).threshold_mm is None
    ]
    add(
        "real_thresholds_are_null",
        len(null_thresholds) == len(policies.crop_keys),
        "Real thresholds absent for: " + ", ".join(null_thresholds),
        required=False,
        warning=(
            "Real thresholds are still absent. Demo mode is the only grading path."
            if null_thresholds
            else None
        ),
    )
    add(
        "policy_digest",
        bool(policies.config_sha256),
        f"{policies.version} sha256={policies.config_sha256[:16]}...",
    )
    add(
        "settings_digest",
        bool(settings.config_sha256),
        f"{settings.version} sha256={settings.config_sha256[:16]}...",
    )

    # 3. Demo assets and hashes.
    manifest = load_json(manifest_path)
    fixtures = manifest.get("fixtures", [])
    missing: list[str] = []
    mismatched: list[str] = []
    for fixture in fixtures:
        path = PROJECT_ROOT / str(fixture.get("path", ""))
        if not path.is_file():
            missing.append(str(fixture.get("fixture_id")))
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != fixture.get("sha256"):
            mismatched.append(str(fixture.get("fixture_id")))
    add(
        "demo_assets_present",
        not missing,
        f"{len(fixtures)} fixtures; missing={missing or 'none'}",
    )
    add(
        "demo_asset_hashes",
        not mismatched,
        f"mismatched={mismatched or 'none'}",
    )

    # 4. Contract execution on every synthetic image.
    service = InferenceService(policies=policies, settings=settings)
    contract_failures: list[str] = []
    for fixture in fixtures:
        path = PROJECT_ROOT / str(fixture.get("path", ""))
        if Path(str(fixture.get("path", ""))).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        decoded = load_image_file(
            path,
            max_pixels=settings.max_image_pixels,
            max_bytes=settings.max_upload_bytes,
        )
        result = service.inspect_frame(
            decoded.bgr,
            MediaContext(
                source_type=SourceType.IMAGE,
                source_id=str(fixture["fixture_id"]),
                device="preflight",
            ),
            InferenceRequest(
                crop_key=str(fixture["crop"]),
                marker_side_mm=float(fixture["marker_side_mm"]),
                expected_marker_id=0,
                demo_policy=True,
            ),
        )
        expected = str(fixture["expected_demonstration_class"])
        actual = result.size_class.value if result.size_class else result.status.value
        if actual != expected:
            contract_failures.append(
                f"{fixture['fixture_id']}: expected {expected}, got {actual}"
            )
    add(
        "synthetic_contract",
        not contract_failures,
        "All image fixtures matched expected status/class."
        if not contract_failures
        else "; ".join(contract_failures),
    )

    # 5. Video path.
    video_fixture = next(
        (
            fixture
            for fixture in fixtures
            if Path(str(fixture.get("path", ""))).suffix.lower() == ".mp4"
        ),
        None,
    )
    video_path = (
        PROJECT_ROOT / str(video_fixture["path"]) if video_fixture else None
    )
    if video_path and video_path.is_file():
        metadata = inspect_video_metadata(
            video_path,
            max_bytes=settings.video.max_file_bytes,
            max_width=settings.video.max_width,
            max_height=settings.video.max_height,
            max_duration_s=settings.video.max_duration_s,
        )
        add(
            "demo_video_metadata",
            metadata.frame_count > 0 and metadata.fps > 0,
            metadata.to_dict(),
        )
        if args.full_video and video_fixture is not None:
            report = process_video(
                video_path,
                service=service,
                request=InferenceRequest(
                    crop_key=str(video_fixture["crop"]),
                    marker_side_mm=float(video_fixture["marker_side_mm"]),
                    expected_marker_id=0,
                    demo_policy=True,
                    notes="preflight",
                ),
                settings=settings,
                output_dir=PROJECT_ROOT / "outputs" / "preflight",
                sample_fps=2.0,
            )
            add(
                "demo_video_processing",
                bool(report.output_video)
                and Path(report.output_video).is_file()
                and report.codec.startswith("h264"),
                {
                    "sampled_frames": report.sampled_frames,
                    "codec": report.codec,
                    "output": relative_to_project(report.output_video or ""),
                    "unique_vegetable_count": None,
                    "count_semantics": "per-frame observations",
                },
            )
        else:
            add(
                "demo_video_processing",
                None,
                "Skipped; pass --full-video.",
                required=False,
            )
    else:
        add("demo_video_metadata", False, "Demo MP4 not found.")

    # 6. Optional real camera check.
    if args.camera:
        import cv2

        capture = cv2.VideoCapture(0)
        opened = capture.isOpened()
        ok, frame = capture.read() if opened else (False, None)
        capture.release()
        add(
            "camera_capture",
            bool(opened and ok and frame is not None and frame.size),
            "One frame captured and released."
            if opened and ok
            else "No frame from the default camera.",
            warning=(
                "Camera permission/availability is environment-specific."
                if not (opened and ok)
                else None
            ),
        )
    else:
        add(
            "camera_capture",
            None,
            "Skipped; pass --camera to test hardware.",
            required=False,
        )

    # 7. Video encoder availability.
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        add("ffmpeg", Path(ffmpeg_exe).is_file(), ffmpeg_exe)
    except Exception as exc:
        ffmpeg_exe = None
        add("ffmpeg", False, f"imageio-ffmpeg unavailable: {exc}")

    report = {
        "report_type": "local_demo_preflight",
        "status": "PASS" if not failures else "FAIL",
        "scope": (
            "Local demo environment only. Passing does not establish real-produce "
            "accuracy, measured-threshold validity, or field readiness."
        ),
        "environment": {
            **environment_metadata(),
            "pid": os.getpid(),
            "ui_versions": versions,
        },
        "manifest": relative_to_project(manifest_path),
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "real_thresholds_absent_for": null_thresholds,
        "camera_tested": args.camera,
        "full_video_tested": args.full_video,
    }
    output = write_json(args.output, report)
    print(f"Preflight report written to {output}")
    print(f"Status: {report['status']}; failures={failures or 'none'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
