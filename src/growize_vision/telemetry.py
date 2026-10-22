"""Local health/preflight telemetry. No media leaves the machine."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from .config import PROJECT_ROOT, config_digest, load_inference_settings, load_policies
from .version import __version__


def dependency_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "opencv-python",
        "Pillow",
        "PyYAML",
        "gradio",
        "huggingface-hub",
        "gradio-client",
        "pandas",
        "imageio-ffmpeg",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def camera_probe(index: int = 0) -> dict[str, Any]:
    """Open and release a camera without retaining a frame or image."""
    result: dict[str, Any] = {
        "index": index,
        "opened": False,
        "frame_shape": None,
        "error": None,
    }
    # On Linux cloud containers (e.g. Hugging Face Spaces) without physical V4L2 device
    if not sys.platform.startswith("win"):
        import os
        from pathlib import Path
        if os.environ.get("SPACE_ID") or not Path(f"/dev/video{index}").exists():
            result["note"] = "Cloud container active. Live camera stream received via browser webcam."
            return result

    cap = None
    try:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
        cap = cv2.VideoCapture(index, backend)
        result["opened"] = bool(cap.isOpened())
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                result["frame_shape"] = list(frame.shape)
            else:
                result["error"] = "Camera opened but no frame was returned."
        else:
            result["error"] = "Camera did not open. Check Windows camera permission."
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if cap is not None:
            cap.release()
    return result


def codec_capabilities() -> dict[str, bool]:
    exts = cv2.getBuildInformation()
    return {
        "ffmpeg": "FFMPEG:                      YES" in exts,
        "directshow": "DirectShow:                 YES" in exts,
        "media_foundation": "Media Foundation:           YES" in exts,
        "gstreamer": "GStreamer:                  YES" in exts,
    }


def health_report() -> dict[str, Any]:
    settings = load_inference_settings()
    policies = load_policies()
    output_dir = PROJECT_ROOT / "outputs"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_writable = os.access(output_dir, os.W_OK)
    except OSError:
        output_writable = False
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "software_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cuda_available": _torch_cuda(),
        "packages": dependency_versions(),
        "codecs": codec_capabilities(),
        "camera": camera_probe(0),
        "output_writable": output_writable,
        "free_disk_gb": round(free_bytes / 1e9, 2),
        "config": {
            "inference_version": settings.version,
            "policy_version": policies.version,
            "inference_sha256": settings.config_sha256,
            "policy_sha256": policies.config_sha256,
            "combined_config_sha256": config_digest(
                PROJECT_ROOT / "config" / "inference.yaml",
                PROJECT_ROOT / "config" / "grade_rules.yaml",
                PROJECT_ROOT / "config" / "cameras.yaml",
            ),
        },
        "policy_truth": {
            "approved_thresholds_present": any(
                crop.get("threshold_mm") is not None
                for crop in policies.crops.values()
            ),
            "approval_summary": policies.approval_summary,
            "demo_mode_available": any(
                crop.get("demo_threshold_mm") is not None
                for crop in policies.crops.values()
            ),
        },
        "network_required": False,
    }


def _torch_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
