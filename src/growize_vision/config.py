"""Configuration loading and validation.

Every threshold and quality constant is loaded from a versioned YAML file so an
experiment can be replayed from its recorded configuration hash.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .domain import CropOntology, GradePolicy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "grade_rules.yaml"
DEFAULT_INFERENCE_PATH = PROJECT_ROOT / "config" / "inference.yaml"
DEFAULT_CAMERA_PATH = PROJECT_ROOT / "config" / "cameras.yaml"


class ConfigError(ValueError):
    """Raised when a policy or runtime config would be unsafe to use."""


@dataclass(frozen=True)
class CalibrationSettings:
    dictionary: str = "DICT_4X4_50"
    expected_marker_id: int = 0
    min_marker_side_px: float = 48.0
    max_side_length_ratio: float = 1.35
    min_marker_area_fraction: float = 0.00015
    corner_uncertainty_px: float = 1.0
    boundary_uncertainty_px: float = 1.5
    parallax_uncertainty_mm: float = 2.0
    default_marker_measurement_uncertainty_mm: float = 0.5


@dataclass(frozen=True)
class SegmentationSettings:
    color_distance_min: float = 15.0
    adaptive_margin: float = 7.0
    morph_kernel: int = 5
    min_component_fraction: float = 0.0004
    max_component_fraction: float = 0.72
    max_instances: int = 12
    marker_exclusion_margin_px: int = 10
    use_grabcut_refinement: bool = False
    shadow_value_only_penalty: float = 0.55


@dataclass(frozen=True)
class QualitySettings:
    blur_min: float = 38.0
    min_contrast: float = 16.0
    max_clipped_low_fraction: float = 0.30
    max_clipped_high_fraction: float = 0.12
    max_glare_fraction: float = 0.06
    min_object_area_fraction: float = 0.002
    boundary_touch_px: int = 2


@dataclass(frozen=True)
class VideoSettings:
    max_duration_s: float = 120.0
    max_frames_to_sample: int = 300
    default_sample_fps: float = 2.0
    max_width: int = 1920
    max_height: int = 1080
    max_file_bytes: int = 300_000_000


@dataclass(frozen=True)
class InferenceSettings:
    version: str = "unversioned"
    max_processing_dimension: int = 1280
    max_upload_bytes: int = 25_000_000
    max_image_pixels: int = 40_000_000
    min_object_area_mm2: float = 180.0
    min_object_area_px: float = 220.0
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    quality: QualitySettings = field(default_factory=QualitySettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    demo_enabled_by_default: bool = False
    demo_banner: str = (
        "Demonstration cutoffs only; no ARL-approved thresholds supplied."
    )
    source_path: str | None = None
    config_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "max_processing_dimension": self.max_processing_dimension,
            "max_upload_bytes": self.max_upload_bytes,
            "max_image_pixels": self.max_image_pixels,
            "min_object_area_mm2": self.min_object_area_mm2,
            "min_object_area_px": self.min_object_area_px,
            "calibration": self.calibration.__dict__,
            "segmentation": self.segmentation.__dict__,
            "quality": self.quality.__dict__,
            "video": self.video.__dict__,
            "demo_enabled_by_default": self.demo_enabled_by_default,
            "demo_banner": self.demo_banner,
            "source_path": self.source_path,
            "config_sha256": self.config_sha256,
        }


@dataclass
class PolicyRegistry:
    version: str
    approval_summary: str
    guard_band_mm: float
    crops: dict[str, dict[str, Any]]
    source_path: str | None = None
    config_sha256: str = ""

    @property
    def crop_keys(self) -> list[str]:
        return list(self.crops.keys())

    def ontology(self, crop_key: str) -> CropOntology:
        crop = self._lookup(crop_key)
        return CropOntology(
            key=self._canonical_key(crop_key),
            display_name=str(crop.get("display_name", crop_key)),
            aliases=tuple(crop.get("aliases", []) or []),
            physical_unit=str(crop.get("physical_unit", "unknown")),
            metric=str(crop.get("metric", "")),
            metric_description=str(crop.get("metric_description", "")),
            supported=bool(crop.get("supported_poses", [])) or bool(crop.get("threshold_mm")),
        )

    def policy(self, crop_key: str, demo: bool = False) -> GradePolicy:
        """Return a policy with a null threshold when no approval exists.

        A null threshold is intentional: the grading engine then produces
        POLICY_REQUIRED rather than inventing a cut-off.  Demo thresholds are
        only returned when the caller explicitly requests demo mode.
        """
        canonical = self._canonical_key(crop_key)
        crop = self.crops[canonical]
        if demo:
            threshold = crop.get("demo_threshold_mm")
            if threshold is not None:
                approval = "ASSESSMENT_DEMO_ONLY"
                source = (
                    "Engineering demonstration cut-off stored in grade_rules.yaml; "
                    "not supplied or approved by ARL/Growize."
                )
                policy_version = f"{self.version}/demo"
            else:
                # A crop without a defensible demo threshold (for example
                # coriander) must not advertise a demo policy it does not have.
                approval = "NOT_PROVIDED"
                source = "No ARL-approved or assessment-demo threshold exists for this crop."
                policy_version = self.version
        else:
            threshold = crop.get("threshold_mm")
            approval = (
                "APPROVED_SOURCE" if threshold is not None else "NOT_PROVIDED"
            )
            source = str(crop.get("source", "No ARL-approved threshold supplied."))
            policy_version = self.version

        if threshold is not None:
            threshold = _finite_positive(threshold, f"{canonical}.threshold_mm")

        return GradePolicy(
            crop_key=canonical,
            crop_name=str(crop.get("display_name", canonical)),
            physical_unit=str(crop.get("physical_unit", "unknown")),
            metric=str(crop.get("metric", "")),
            metric_description=str(crop.get("metric_description", "")),
            threshold_mm=float(threshold) if threshold is not None else None,
            comparator=str(crop.get("comparator", "greater_than_or_equal")),
            guard_band_mm=self.guard_band_mm,
            policy_version=policy_version,
            approval_status=approval,
            source=source,
            demo=bool(demo and threshold is not None),
            min_visible_fraction=float(crop.get("min_visible_fraction", 0.98)),
        )

    def _lookup(self, crop_key: str) -> dict[str, Any]:
        canonical = self._canonical_key(crop_key)
        return self.crops[canonical]

    def _canonical_key(self, crop_key: str) -> str:
        if not crop_key:
            raise ConfigError("A crop policy must be selected.")
        needle = crop_key.strip().lower().replace(" ", "_")
        if needle in self.crops:
            return needle
        for key, crop in self.crops.items():
            aliases = [str(a).lower().replace(" ", "_") for a in crop.get("aliases", [])]
            if needle in aliases:
                return key
        raise ConfigError(
            f"Unknown crop policy '{crop_key}'. Supported: {', '.join(self.crops)}"
        )


def _finite_positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(f"{name} must be finite and > 0, got {value!r}")
    return number


def _read_yaml(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise ConfigError(f"Configuration file does not exist: {path}")
    raw = path.read_bytes()
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Top-level YAML object must be a mapping: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def load_policies(path: Path | str = DEFAULT_POLICY_PATH) -> PolicyRegistry:
    path = Path(path)
    data, digest = _read_yaml(path)
    crops = data.get("crops")
    if not isinstance(crops, dict) or not crops:
        raise ConfigError("grade_rules.yaml must define a non-empty crops mapping.")
    guard = data.get("guard_band_mm", 0.0)
    guard = float(guard)
    if not math.isfinite(guard) or guard < 0:
        raise ConfigError("guard_band_mm must be finite and >= 0.")

    for key, crop in crops.items():
        if not isinstance(crop, dict):
            raise ConfigError(f"crop {key} must be a mapping.")
        metric = str(crop.get("metric", ""))
        if metric not in {
            "min_area_rect_short_mm",
            "min_area_rect_long_mm",
            "max_feret_mm",
            "projected_area_cm2",
        }:
            raise ConfigError(f"crop {key} has unsupported metric {metric!r}.")
        for field_name in ("threshold_mm", "demo_threshold_mm"):
            value = crop.get(field_name)
            if value is not None:
                _finite_positive(value, f"{key}.{field_name}")
        comparator = crop.get("comparator", "greater_than_or_equal")
        if comparator not in {
            "greater_than_or_equal",
            "greater_than",
            "less_than",
            "less_than_or_equal",
        }:
            raise ConfigError(f"crop {key} has unsupported comparator {comparator!r}.")
        if crop.get("physical_unit") in {None, "", "unknown"}:
            raise ConfigError(f"crop {key} must define a physical_unit.")

    return PolicyRegistry(
        version=str(data.get("version", "unversioned")),
        approval_summary=str(data.get("approval_summary", "")),
        guard_band_mm=guard,
        crops=crops,
        source_path=str(path),
        config_sha256=digest,
    )


def load_inference_settings(
    path: Path | str = DEFAULT_INFERENCE_PATH,
) -> InferenceSettings:
    path = Path(path)
    data, digest = _read_yaml(path)
    cal = data.get("calibration", {}) or {}
    seg = data.get("segmentation", {}) or {}
    qual = data.get("quality", {}) or {}
    vid = data.get("video", {}) or {}
    demo = data.get("demo", {}) or {}

    settings = InferenceSettings(
        version=str(data.get("version", "unversioned")),
        max_processing_dimension=int(data.get("max_processing_dimension", 1280)),
        max_upload_bytes=int(data.get("max_upload_bytes", 25_000_000)),
        max_image_pixels=int(data.get("max_image_pixels", 40_000_000)),
        min_object_area_mm2=float(data.get("min_object_area_mm2", 180.0)),
        min_object_area_px=float(data.get("min_object_area_px", 220.0)),
        calibration=CalibrationSettings(
            dictionary=str(cal.get("dictionary", "DICT_4X4_50")),
            expected_marker_id=int(cal.get("expected_marker_id", 0)),
            min_marker_side_px=float(cal.get("min_marker_side_px", 48.0)),
            max_side_length_ratio=float(cal.get("max_side_length_ratio", 1.35)),
            min_marker_area_fraction=float(cal.get("min_marker_area_fraction", 0.00015)),
            corner_uncertainty_px=float(cal.get("corner_uncertainty_px", 1.0)),
            boundary_uncertainty_px=float(cal.get("boundary_uncertainty_px", 1.5)),
            parallax_uncertainty_mm=float(cal.get("parallax_uncertainty_mm", 2.0)),
            default_marker_measurement_uncertainty_mm=float(
                cal.get("default_marker_measurement_uncertainty_mm", 0.5)
            ),
        ),
        segmentation=SegmentationSettings(
            color_distance_min=float(seg.get("color_distance_min", 15.0)),
            adaptive_margin=float(seg.get("adaptive_margin", 7.0)),
            morph_kernel=int(seg.get("morph_kernel", 5)),
            min_component_fraction=float(seg.get("min_component_fraction", 0.0004)),
            max_component_fraction=float(seg.get("max_component_fraction", 0.72)),
            max_instances=int(seg.get("max_instances", 12)),
            marker_exclusion_margin_px=int(seg.get("marker_exclusion_margin_px", 10)),
            use_grabcut_refinement=bool(seg.get("use_grabcut_refinement", False)),
            shadow_value_only_penalty=float(seg.get("shadow_value_only_penalty", 0.55)),
        ),
        quality=QualitySettings(
            blur_min=float(qual.get("blur_min", 38.0)),
            min_contrast=float(qual.get("min_contrast", 16.0)),
            max_clipped_low_fraction=float(qual.get("max_clipped_low_fraction", 0.30)),
            max_clipped_high_fraction=float(qual.get("max_clipped_high_fraction", 0.12)),
            max_glare_fraction=float(qual.get("max_glare_fraction", 0.06)),
            min_object_area_fraction=float(qual.get("min_object_area_fraction", 0.002)),
            boundary_touch_px=int(qual.get("boundary_touch_px", 2)),
        ),
        video=VideoSettings(
            max_duration_s=float(vid.get("max_duration_s", 120.0)),
            max_frames_to_sample=int(vid.get("max_frames_to_sample", 300)),
            default_sample_fps=float(vid.get("default_sample_fps", 2.0)),
            max_width=int(vid.get("max_width", 1920)),
            max_height=int(vid.get("max_height", 1080)),
            max_file_bytes=int(vid.get("max_file_bytes", 300_000_000)),
        ),
        demo_enabled_by_default=bool(demo.get("enabled_by_default", False)),
        demo_banner=str(
            demo.get(
                "banner",
                "DEMO THRESHOLDS ACTIVE - engineering demonstration only.",
            )
        ),
        source_path=str(path),
        config_sha256=digest,
    )
    _validate_inference_settings(settings)
    return settings


def _validate_inference_settings(settings: InferenceSettings) -> None:
    positive = {
        "max_processing_dimension": settings.max_processing_dimension,
        "max_upload_bytes": settings.max_upload_bytes,
        "max_image_pixels": settings.max_image_pixels,
        "min_object_area_mm2": settings.min_object_area_mm2,
        "min_object_area_px": settings.min_object_area_px,
        "calibration.min_marker_side_px": settings.calibration.min_marker_side_px,
        "calibration.max_side_length_ratio": settings.calibration.max_side_length_ratio,
        "segmentation.color_distance_min": settings.segmentation.color_distance_min,
        "segmentation.morph_kernel": settings.segmentation.morph_kernel,
        "quality.blur_min": settings.quality.blur_min,
        "quality.min_contrast": settings.quality.min_contrast,
    }
    for key, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ConfigError(f"{key} must be finite and > 0, got {value!r}")
    for key, value in {
        "quality.max_clipped_low_fraction": settings.quality.max_clipped_low_fraction,
        "quality.max_clipped_high_fraction": settings.quality.max_clipped_high_fraction,
        "quality.max_glare_fraction": settings.quality.max_glare_fraction,
        "segmentation.min_component_fraction": settings.segmentation.min_component_fraction,
        "segmentation.max_component_fraction": settings.segmentation.max_component_fraction,
    }.items():
        if not math.isfinite(float(value)) or float(value) < 0 or float(value) > 1:
            raise ConfigError(f"{key} must be between 0 and 1, got {value!r}")


def load_camera_profiles(path: Path | str = DEFAULT_CAMERA_PATH) -> dict[str, Any]:
    data, _ = _read_yaml(Path(path))
    cameras = data.get("cameras", {})
    if not isinstance(cameras, dict):
        raise ConfigError("cameras.yaml must define a cameras mapping.")
    return data


def config_digest(*paths: Path | str) -> str:
    """Hash the ordered config bytes so every result can identify its policy."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()
