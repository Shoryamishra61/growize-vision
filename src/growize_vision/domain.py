"""Typed domain objects shared by every input adapter.

A single result contract is used for image, webcam and video paths.  The
serialised form intentionally excludes raw frames and masks; those are
ephemeral and never written to logs.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Status(str, Enum):
    """Operational state of an inspection.

    Only ACCEPTED observations are allowed to carry a physical size class.
    """

    ACCEPTED = "ACCEPTED"
    NO_OBJECT = "NO_OBJECT"
    NOT_PRODUCE = "NOT_PRODUCE"
    CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"
    POLICY_REQUIRED = "POLICY_REQUIRED"
    OCCLUDED = "OCCLUDED"
    REVIEW_BORDERLINE = "REVIEW_BORDERLINE"
    LOW_IMAGE_QUALITY = "LOW_IMAGE_QUALITY"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INVALID_INPUT = "INVALID_INPUT"
    MULTIPLE_OBJECTS = "MULTIPLE_OBJECTS"


class SizeClass(str, Enum):
    SMALL = "SMALL"
    BIG = "BIG"


class SourceType(str, Enum):
    IMAGE = "image"
    WEBCAM = "webcam"
    VIDEO = "video"
    API = "api"


@dataclass(frozen=True)
class CropOntology:
    key: str
    display_name: str
    aliases: tuple[str, ...]
    physical_unit: str
    metric: str
    metric_description: str
    supported: bool = True


@dataclass
class MediaContext:
    source_type: SourceType = SourceType.IMAGE
    source_id: str = "unknown"
    device: str = "unknown"
    captured_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    frame_id: int | None = None
    timestamp_s: float | None = None
    batch_id: str | None = None
    session_id: str | None = None
    source_sha256: str | None = None


@dataclass
class MarkerDetection:
    marker_id: int
    corners_px: list[tuple[float, float]]
    side_px: float
    area_px: float
    dictionary: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class Calibration:
    valid: bool
    calibration_id: str
    marker_id: int | None = None
    marker_side_mm: float | None = None
    homography: list[list[float]] | None = None
    mm_per_px: float | None = None
    corner_reprojection_rms_px: float | None = None
    marker_measurement_uncertainty_mm: float = 0.5
    source: str = "printed_aruco_marker"
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "calibration_id": self.calibration_id,
            "marker_id": self.marker_id,
            "marker_side_mm": self.marker_side_mm,
            "homography": self.homography,
            "mm_per_px": self.mm_per_px,
            "corner_reprojection_rms_px": self.corner_reprojection_rms_px,
            "marker_measurement_uncertainty_mm": self.marker_measurement_uncertainty_mm,
            "source": self.source,
            "warnings": self.warnings,
            "reasons": self.reasons,
        }


@dataclass
class ImageQuality:
    blur_score: float
    contrast: float
    clipped_low_fraction: float
    clipped_high_fraction: float
    glare_fraction: float
    object_touches_edge: bool = False
    object_area_fraction: float = 0.0
    passed: bool = True
    flags: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blur_score": _round(self.blur_score, 3),
            "contrast": _round(self.contrast, 3),
            "clipped_low_fraction": _round(self.clipped_low_fraction, 5),
            "clipped_high_fraction": _round(self.clipped_high_fraction, 5),
            "glare_fraction": _round(self.glare_fraction, 5),
            "object_touches_edge": self.object_touches_edge,
            "object_area_fraction": _round(self.object_area_fraction, 5),
            "passed": self.passed,
            "flags": list(self.flags),
            "metrics": {k: _round(v, 4) for k, v in self.metrics.items()},
        }


@dataclass
class PhysicalMeasurement:
    """A projected-plane measurement.

    Never a validated 3-D volume or a mass estimate: any ``volume_proxy``
    block is an explicitly-labelled shape-model proxy that grading must
    ignore, and ``centerline`` is diagnostic skeleton information.
    """

    metric: str
    value_mm: float
    uncertainty_mm: float
    lower_mm: float
    upper_mm: float
    unit: str = "mm"
    method: str = "calibrated_planar_homography"
    calibration_id: str | None = None
    coverage_note: str = "engineering_error_bound_not_statistical_95pct"
    secondary: dict[str, float] = field(default_factory=dict)
    centerline: dict[str, Any] | None = None
    volume_proxy: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": _round(self.value_mm, 3),
            "value_mm": _round(self.value_mm, 3),
            "uncertainty_mm": _round(self.uncertainty_mm, 3),
            "lower_mm": _round(self.lower_mm, 3),
            "upper_mm": _round(self.upper_mm, 3),
            "unit": self.unit,
            "method": self.method,
            "calibration_id": self.calibration_id,
            "coverage_note": self.coverage_note,
            "secondary": {k: _round(v, 3) for k, v in self.secondary.items()},
            "centerline": self._round_nested(self.centerline),
            "volume_proxy": self._round_nested(self.volume_proxy),
            "warnings": list(self.warnings),
        }

    @staticmethod
    def _round_nested(block: dict[str, Any] | None) -> dict[str, Any] | None:
        if block is None:
            return None
        rounded: dict[str, Any] = {}
        for key, value in block.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rounded[key] = _round(float(value), 3)
            else:
                rounded[key] = value
        return rounded


@dataclass(frozen=True)
class SizeBand:
    """One ordered size grade band: name plus inclusive lower edge in mm.

    The first band's ``min_mm`` is 0; a band runs from its own ``min_mm``
    (inclusive) up to the next band's ``min_mm`` (exclusive).  Two bands
    reproduce the classic binary SMALL/BIG cut-off exactly.
    """

    name: str
    min_mm: float

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "min_mm": _round(self.min_mm, 3)}


@dataclass(frozen=True)
class GradePolicy:
    crop_key: str
    crop_name: str
    physical_unit: str
    metric: str
    metric_description: str
    threshold_mm: float | None
    comparator: str = "greater_than_or_equal"
    guard_band_mm: float = 0.0
    policy_version: str = "unversioned"
    approval_status: str = "NOT_PROVIDED"
    source: str = "not provided"
    demo: bool = False
    min_visible_fraction: float = 0.98
    bands: tuple[SizeBand, ...] | None = None
    bands_derived: bool = False
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_key": self.crop_key,
            "crop_name": self.crop_name,
            "physical_unit": self.physical_unit,
            "metric": self.metric,
            "metric_description": self.metric_description,
            "threshold_mm": self.threshold_mm,
            "comparator": self.comparator,
            "guard_band_mm": self.guard_band_mm,
            "policy_version": self.policy_version,
            "approval_status": self.approval_status,
            "source": self.source,
            "demo": self.demo,
            "min_visible_fraction": self.min_visible_fraction,
            "bands": [band.to_dict() for band in self.bands] if self.bands else None,
            "bands_derived": self.bands_derived,
            "provenance": dict(self.provenance) if self.provenance else None,
        }


@dataclass
class GradeDecision:
    size_class: SizeClass | None
    status: Status
    reasons: list[str]
    policy: GradePolicy | None = None
    measurement: PhysicalMeasurement | None = None
    threshold_mm: float | None = None
    comparator: str | None = None
    policy_version: str | None = None
    calibration_id: str | None = None
    warning: str | None = None
    band_name: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status == Status.ACCEPTED and (
            self.size_class in (SizeClass.SMALL, SizeClass.BIG)
            or self.band_name is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "size_class": self.size_class.value if self.size_class else None,
            "size_band": self.band_name,
            "status": self.status.value,
            "reasons": list(self.reasons),
            "measurement": self.measurement.to_dict() if self.measurement else None,
            "threshold_mm": self.threshold_mm,
            "comparator": self.comparator,
            "policy_version": self.policy_version,
            "calibration_id": self.calibration_id,
            "policy": self.policy.to_dict() if self.policy else None,
            "warning": self.warning,
        }


@dataclass
class InstanceObservation:
    instance_id: str
    crop_key: str | None
    bbox_px: tuple[int, int, int, int]
    contour_px: list[list[float]]
    mask_area_px: float
    source: str = "opencv_controlled_background"
    tracking_id: str | None = None
    visible_fraction: float = 1.0
    quality_flags: list[str] = field(default_factory=list)
    decision: GradeDecision | None = None
    measurement: PhysicalMeasurement | None = None
    mask: Any = field(default=None, repr=False, compare=False)
    contour_mm: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "crop_key": self.crop_key,
            "bbox_px": list(self.bbox_px),
            "mask_area_px": _round(self.mask_area_px, 2),
            "source": self.source,
            "tracking_id": self.tracking_id,
            "visible_fraction": _round(self.visible_fraction, 4),
            "quality_flags": list(self.quality_flags),
            "decision": self.decision.to_dict() if self.decision else None,
            "measurement": self.measurement.to_dict() if self.measurement else None,
            "contour_points": len(self.contour_px),
        }


@dataclass
class InspectionResult:
    result_id: str
    context: MediaContext
    status: Status
    size_class: SizeClass | None = None
    size_band: str | None = None
    instances: list[InstanceObservation] = field(default_factory=list)
    calibration: Calibration | None = None
    quality: ImageQuality | None = None
    policy: GradePolicy | None = None
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    processing_ms: float = 0.0
    original_shape: tuple[int, int, int] | None = None
    processed_shape: tuple[int, int, int] | None = None
    software_version: str = "0.1.0"
    pipeline_version: str = "opencv-baseline-0.1.0"
    pipeline_sha256: str = ""
    model_sha: str | None = None
    processed_frame_sha256: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    annotated_image: Any = field(default=None, repr=False, compare=False)
    exported_paths: dict[str, str] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    tracking_summary: dict[str, Any] | None = None
    species_prediction: dict[str, Any] | None = None
    appearance_estimate: dict[str, Any] | None = None

    def to_dict(self, include_instances: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "result_id": self.result_id,
            "timestamp": self.timestamp,
            "media": {
                "source_type": self.context.source_type.value,
                "source_id": self.context.source_id,
                "device": self.context.device,
                "captured_at": self.context.captured_at,
                "frame_id": self.context.frame_id,
                "timestamp_s": self.context.timestamp_s,
                "batch_id": self.context.batch_id,
                "session_id": self.context.session_id,
                "source_sha256": self.context.source_sha256,
            },
            "status": self.status.value,
            "size_class": self.size_class.value if self.size_class else None,
            "size_band": self.size_band,
            "accepted": self.status == Status.ACCEPTED
            and (
                self.size_class in (SizeClass.SMALL, SizeClass.BIG)
                or self.size_band is not None
            ),
            "instances": [i.to_dict() for i in self.instances]
            if include_instances
            else [],
            "instance_count": len(self.instances),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "quality": self.quality.to_dict() if self.quality else None,
            "policy": self.policy.to_dict() if self.policy else None,
            "warnings": list(self.warnings),
            "reasons": list(self.reasons),
            "processing_ms": _round(self.processing_ms, 3),
            "timings_ms": {k: _round(v, 3) for k, v in self.timings_ms.items()},
            "original_shape": list(self.original_shape) if self.original_shape else None,
            "processed_shape": list(self.processed_shape) if self.processed_shape else None,
            "software_version": self.software_version,
            "pipeline_version": self.pipeline_version,
            "pipeline_sha256": self.pipeline_sha256,
            "model_sha": self.model_sha,
            "processed_frame_sha256": self.processed_frame_sha256,
            "exported_paths": dict(self.exported_paths),
            "tracking_summary": self.tracking_summary,
            "species_prediction": self.species_prediction,
            "species_prediction_scope": "scene" if self.species_prediction else None,
            "appearance_estimate": self.appearance_estimate,
        }
        return payload


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _round(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
