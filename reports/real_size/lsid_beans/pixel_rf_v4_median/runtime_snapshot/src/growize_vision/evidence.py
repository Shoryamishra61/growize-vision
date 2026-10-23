"""Small shared helpers for evidence-generation scripts.

These scripts intentionally keep every claim tied to an archived JSON report.
They never write a synthetic result into a real-specimen metric field.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy

from growize_vision.version import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "growize_vision": __version__,
        "execution_device": "CPU",
        "cuda_claimed": False,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear-interpolated percentile without a NumPy dependency."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * min(max(fraction, 0.0), 1.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min_ms": None,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "min_ms": round(min(values), 3),
        "mean_ms": round(sum(values) / len(values), 3),
        "median_ms": round(float(numpy.median(values)), 3),
        "p95_ms": round(float(percentile(values, 0.95) or 0.0), 3),
        "max_ms": round(max(values), 3),
    }


def write_json(path: Path | str, payload: dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return str(target)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_to_project(path: Path | str) -> str:
    target = Path(path)
    try:
        return str(target.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(target)


def load_json(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8"))
