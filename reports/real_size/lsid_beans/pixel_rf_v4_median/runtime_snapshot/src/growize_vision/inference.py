"""The single inference entry point used by every UI/input adapter."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    InferenceSettings,
    PolicyRegistry,
    load_inference_settings,
    load_policies,
)
from .domain import (
    Calibration,
    GradeDecision,
    ImageQuality,
    InspectionResult,
    InstanceObservation,
    MediaContext,
    SizeClass,
    Status,
)
from .geometry import MetrologyError, measure_contour
from .grading import evaluate_measurement
from .markers import (
    calibrate_from_marker,
    detect_best_marker,
    marker_mask,
)
from .media import MediaError, image_sha256, resize_for_processing, validate_bgr_frame
from .quality import assess_quality
from .rendering import render_inspection
from .segmentation_cv import segment_foreground
from .version import __version__


@dataclass
class InferenceRequest:
    """User/operator choices that affect interpretation, not the model itself."""

    crop_key: str | None = "potato"
    marker_side_mm: float | None = None
    marker_measurement_uncertainty_mm: float = 0.5
    expected_marker_id: int | None = 0
    demo_policy: bool = False
    segmentation_mode: str = "opencv_baseline"
    notes: str | None = None
    allow_appearance_estimate: bool = False


@dataclass
class InferenceService:
    policies: PolicyRegistry = field(default_factory=load_policies)
    settings: InferenceSettings = field(default_factory=load_inference_settings)
    neural_model: Any = None
    identity_service: Any = None

    def __post_init__(self) -> None:
        self._pipeline_version = "opencv-controlled-mat-0.2.0"
        source_hash = hashlib.sha256()
        for path in sorted(Path(__file__).parent.glob("*.py")):
            source_hash.update(path.name.encode())
            source_hash.update(path.read_bytes())
        self._code_fingerprint = hashlib.sha256(
            (
                f"{__version__}|{self._pipeline_version}|"
                f"{source_hash.hexdigest()}|{self.settings.config_sha256}|{self.policies.config_sha256}"
            ).encode()
        ).hexdigest()

    def inspect_frame(
        self,
        frame: Any,
        context: MediaContext | None = None,
        request: InferenceRequest | None = None,
    ) -> InspectionResult:
        context = context or MediaContext()
        request = request or InferenceRequest()
        started = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []
        result_id = f"insp_{hashlib.sha1(f'{time.time_ns()}|{context.source_id}'.encode()).hexdigest()[:14]}"

        try:
            input_bgr, input_warnings = validate_bgr_frame(
                frame, max_pixels=self.settings.max_image_pixels
            )
            warnings.extend(input_warnings)
        except MediaError as exc:
            return self._invalid_result(result_id, context, str(exc), started)

        original_shape = tuple(int(v) for v in input_bgr.shape)
        if self.identity_service is not None:
            screen_started = time.perf_counter()
            present, reason = self.identity_service.detect_produce_presence(input_bgr)
            timings["input_screen_ms"] = (time.perf_counter() - screen_started) * 1000
            if not present:
                status = Status.NOT_PRODUCE
                if reason == "FRAME_TOO_DARK":
                    status = Status.LOW_IMAGE_QUALITY
                elif reason == "FRAME_TOO_SMALL":
                    status = Status.INVALID_INPUT
                elif reason.startswith("EMPTY_"):
                    status = Status.NO_OBJECT
                elif reason == "NO_PRODUCE_DETECTED":
                    status = Status.OUT_OF_DISTRIBUTION
                refused = InspectionResult(
                    result_id=result_id, context=context, status=status,
                    reasons=[reason], original_shape=original_shape,
                    software_version=__version__, pipeline_version=self._pipeline_version,
                    pipeline_sha256=self._code_fingerprint,
                    timings_ms=timings,
                )
                refused.annotated_image = render_inspection(input_bgr, refused)
                refused.processing_ms = (time.perf_counter() - started) * 1000.0
                return refused
        pre_start = time.perf_counter()
        processing_bgr, resize_scale = resize_for_processing(
            input_bgr, self.settings.max_processing_dimension
        )
        timings["preprocess_ms"] = (time.perf_counter() - pre_start) * 1000.0
        warnings.append(
            f"Processing at {processing_bgr.shape[1]}x{processing_bgr.shape[0]} "
            f"(scale {resize_scale:.3f})."
        ) if resize_scale != 1.0 else None

        if request.segmentation_mode not in {"opencv_baseline", "neural"}:
            return self._invalid_result(
                result_id,
                context,
                f"Unknown segmentation mode {request.segmentation_mode!r}.",
                started,
            )
        if request.segmentation_mode == "neural":
            return self._model_unavailable_result(
                result_id, context, original_shape, started, processing_bgr
            )

        # Marker first: it both calibrates and defines the pixels that must not
        # be treated as produce.
        marker_start = time.perf_counter()
        marker, marker_reasons = detect_best_marker(
            processing_bgr,
            self.settings.calibration,
            expected_id=request.expected_marker_id,
        )
        calibration = (
            calibrate_from_marker(
                processing_bgr.shape,
                marker,
                request.marker_side_mm,
                self.settings.calibration,
                request.marker_measurement_uncertainty_mm,
            )
            if marker is not None
            else Calibration(
                valid=False,
                calibration_id="unavailable",
                reasons=marker_reasons
                or ["No usable calibration marker was found in this frame."],
            )
        )
        timings["calibration_ms"] = (time.perf_counter() - marker_start) * 1000.0
        warnings.extend(calibration.warnings)
        if not calibration.valid:
            warnings.extend(calibration.reasons)

        # Segmentation is still run without calibration so the operator can see
        # the object outline while the calibration instruction is displayed.
        segment_start = time.perf_counter()
        segmentation = segment_foreground(
            processing_bgr,
            calibration if calibration.valid else None,
            marker,
            self.settings.segmentation,
            min_area_mm2=self.settings.min_object_area_mm2,
            min_area_px=self.settings.min_object_area_px,
        )
        timings["segmentation_ms"] = (time.perf_counter() - segment_start) * 1000.0
        warnings.extend(segmentation.warnings)

        marker_mask_arr = marker_mask(
            processing_bgr.shape,
            marker,
            self.settings.segmentation.marker_exclusion_margin_px,
        )
        quality_start = time.perf_counter()
        quality = assess_quality(
            processing_bgr,
            self.settings.quality,
            marker_mask=marker_mask_arr,
            object_mask=segmentation.mask,
            background_busy=segmentation.background_busy,
        )
        timings["quality_ms"] = (time.perf_counter() - quality_start) * 1000.0
        warnings.extend(quality.flags)

        species_prediction = None
        if request.crop_key == "auto":
            identity_started = time.perf_counter()
            if self.identity_service is None:
                from .crop_identity import CropIdentityService

                self.identity_service = CropIdentityService()
            species_prediction = self.identity_service.predict_species(input_bgr)
            # A scene-level suggestion must not assign a species to each separate item.
            effective_crop = (species_prediction.species_key
                              if species_prediction and len(segmentation.objects) == 1 else None)
            request = replace(request, crop_key=effective_crop)
            if effective_crop is None:
                warnings.append("AUTO_CROP_UNRESOLVED: select a crop explicitly or recapture one separated item.")
            timings["crop_identity_ms"] = (time.perf_counter() - identity_started) * 1000

        policy = None
        policy_error: str | None = None
        if request.crop_key:
            try:
                policy = self.policies.policy(
                    request.crop_key, demo=bool(request.demo_policy)
                )
            except Exception as exc:  # ConfigError and unexpected lookup errors
                policy_error = str(exc)
                warnings.append(policy_error)
        if policy and policy.demo:
            warnings.append(self.settings.demo_banner)

        instances = self._build_instances(
            processing_bgr=processing_bgr,
            segmentation=segmentation,
            calibration=calibration,
            quality=quality,
            policy=policy,
            policy_error=policy_error,
            request=request,
            warnings=warnings,
        )

        top_status, top_size, top_reasons = self._summarise(
            instances=instances,
            calibration=calibration,
            quality=quality,
            segmentation=segmentation,
            policy=policy,
            policy_error=policy_error,
        )
        if not instances:
            warnings.append(
                "No object was accepted as a foreground vegetable instance."
            )

        result = InspectionResult(
            result_id=result_id,
            context=context,
            status=top_status,
            size_class=top_size,
            instances=instances,
            calibration=calibration,
            quality=quality,
            policy=policy,
            warnings=_dedupe(warnings),
            reasons=top_reasons,
            original_shape=original_shape,
            processed_shape=tuple(int(v) for v in processing_bgr.shape),
            software_version=__version__,
            pipeline_version=self._pipeline_version,
            pipeline_sha256=self._code_fingerprint,
            timings_ms=timings,
            processed_frame_sha256=image_sha256(processing_bgr),
            species_prediction=asdict(species_prediction) if species_prediction else None,
        )
        if self.identity_service is not None and (species_prediction is not None):
            model_path = (self.identity_service.deep_classifier_path if self.identity_service.is_deep
                          else self.identity_service.classifier_path)
            if model_path.exists():
                result.model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if request.allow_appearance_estimate and not calibration.valid and request.crop_key and len(instances) == 1:
            if self.identity_service is None:
                from .crop_identity import CropIdentityService

                self.identity_service = CropIdentityService()
            estimate = self.identity_service.estimate_appearance_size(input_bgr, request.crop_key)
            if estimate is not None:
                result.appearance_estimate = asdict(estimate)
                result.warnings.append(estimate.disclaimer)
        render_start = time.perf_counter()
        result.annotated_image = render_inspection(processing_bgr, result)
        if result.appearance_estimate is not None:
            result.annotated_image = self.identity_service.render_appearance_overlay(
                input_bgr, estimate, request.crop_key,
            )
        result.timings_ms["render_ms"] = (time.perf_counter() - render_start) * 1000.0
        result.processing_ms = (time.perf_counter() - started) * 1000.0
        return result

    def _build_instances(
        self,
        *,
        processing_bgr: np.ndarray,
        segmentation: Any,
        calibration: Calibration,
        quality: ImageQuality,
        policy: Any,
        policy_error: str | None,
        request: InferenceRequest,
        warnings: list[str],
    ) -> list[InstanceObservation]:
        instances: list[InstanceObservation] = []
        for index, obj in enumerate(segmentation.objects, start=1):
            instance_id = f"obj_{index:03d}"
            visible_fraction = 1.0
            instance_flags = list(obj.flags)
            if obj.touches_edge:
                visible_fraction = 0.60
            if obj.touches_marker:
                visible_fraction = min(visible_fraction, 0.85)
            instance = InstanceObservation(
                instance_id=instance_id,
                crop_key=request.crop_key,
                bbox_px=obj.bbox_px,
                contour_px=obj.contour_px.reshape(-1, 2).astype(float).tolist(),
                mask_area_px=float(obj.area_px),
                source="opencv_controlled_background",
                visible_fraction=visible_fraction,
                quality_flags=instance_flags,
                mask=obj.mask,
            )

            # Priority order is safety-first: no physical millimetre claim
            # without trusted calibration, then quality, then occlusion, then
            # policy, then metrology.
            if not calibration.valid:
                instance.decision = GradeDecision(
                    size_class=None,
                    status=Status.CALIBRATION_REQUIRED,
                    reasons=list(calibration.reasons)
                    or ["Valid marker-based calibration was not available."],
                    policy=policy,
                    policy_version=getattr(policy, "policy_version", None),
                    calibration_id=calibration.calibration_id,
                )
            elif segmentation.background_busy:
                instance.decision = GradeDecision(
                    size_class=None,
                    status=Status.OUT_OF_DISTRIBUTION,
                    reasons=[
                        "Background is not a plain matte station; physical grading "
                        "is outside the validated operating domain."
                    ],
                    policy=policy,
                    policy_version=getattr(policy, "policy_version", None),
                    calibration_id=calibration.calibration_id,
                )
            elif not quality.passed:
                instance.decision = GradeDecision(
                    size_class=None,
                    status=Status.LOW_IMAGE_QUALITY,
                    reasons=list(quality.flags)
                    or ["Image-quality gate failed."],
                    policy=policy,
                    policy_version=getattr(policy, "policy_version", None),
                    calibration_id=calibration.calibration_id,
                )
            elif obj.touches_edge:
                instance.decision = GradeDecision(
                    size_class=None,
                    status=Status.OCCLUDED,
                    reasons=[
                        "Object silhouette touches the frame edge; physical extent "
                        "cannot be inferred from the visible part."
                    ],
                    policy=policy,
                    policy_version=getattr(policy, "policy_version", None),
                    calibration_id=calibration.calibration_id,
                )
            elif obj.touches_marker:
                instance.decision = GradeDecision(
                    size_class=None,
                    status=Status.OCCLUDED,
                    reasons=[
                        "Object mask overlaps or touches the marker exclusion zone; "
                        "move the marker and vegetable apart on the same plane."
                    ],
                    policy=policy,
                    policy_version=getattr(policy, "policy_version", None),
                    calibration_id=calibration.calibration_id,
                )
            elif "LOW_SOLIDITY_POSSIBLY_MERGED" in obj.flags:
                instance.decision = GradeDecision(
                    size_class=None,
                    status=Status.OCCLUDED,
                    reasons=[
                        "Mask is strongly non-convex and may contain touching/merged "
                        "produce or an occlusion; separate items and recapture."
                    ],
                    policy=policy,
                    policy_version=getattr(policy, "policy_version", None),
                    calibration_id=calibration.calibration_id,
                )
            elif policy is None or policy_error:
                instance.decision = GradeDecision(
                    size_class=None,
                    status=Status.POLICY_REQUIRED,
                    reasons=[policy_error or "Select a supported crop policy."],
                    policy=policy,
                    policy_version=getattr(policy, "policy_version", None),
                    calibration_id=calibration.calibration_id,
                )
            elif policy.threshold_mm is None and not policy.demo:
                instance.decision = evaluate_measurement(None, policy, calibration.calibration_id)
                instance.decision.reasons = [
                    "NO_APPROVED_THRESHOLD: no ARL/Growize threshold has been supplied "
                    "for this crop. Physical measurement is withheld in strict mode."
                ]
            else:
                try:
                    measurement, contour_mm = measure_contour(
                        obj.contour_px,
                        calibration,
                        policy.metric,
                        self.settings.calibration,
                    )
                except MetrologyError as exc:
                    instance.decision = GradeDecision(
                        size_class=None,
                        status=Status.REVIEW_BORDERLINE,
                        reasons=[f"METROLOGY_FAILED: {exc}"],
                        policy=policy,
                        policy_version=policy.policy_version,
                        calibration_id=calibration.calibration_id,
                    )
                else:
                    instance.measurement = measurement
                    instance.contour_mm = contour_mm.tolist()
                    instance.decision = evaluate_measurement(
                        measurement, policy, calibration.calibration_id
                    )
                    if (
                        policy.metric == "min_area_rect_long_mm"
                        and measurement.secondary.get("curvature_ratio", 0.0) > 1.45
                    ):
                        instance.quality_flags.append(
                            "HIGH_CURVATURE: straight-axis metric may be biased for bent pods/leaves."
                        )

            if instance.decision and instance.decision.measurement:
                instance.measurement = instance.decision.measurement
            instances.append(instance)
        return instances

    def _summarise(
        self,
        *,
        instances: list[InstanceObservation],
        calibration: Calibration,
        quality: ImageQuality,
        segmentation: Any,
        policy: Any,
        policy_error: str | None,
    ) -> tuple[Status, SizeClass | None, list[str]]:
        if not instances:
            reasons = ["NO_OBJECT: no foreground vegetable instance was detected."]
            if not calibration.valid:
                reasons.append("Calibration marker was also unavailable or invalid.")
            return Status.NO_OBJECT, None, reasons
        if not calibration.valid:
            return (
                Status.CALIBRATION_REQUIRED,
                None,
                ["A valid marker with the user-measured outer side length is required."],
            )
        if segmentation.background_busy:
            return (
                Status.OUT_OF_DISTRIBUTION,
                None,
                ["Busy background: use the plain matte station."],
            )
        if not quality.passed:
            return Status.LOW_IMAGE_QUALITY, None, list(quality.flags)
        if policy is None:
            return (
                Status.POLICY_REQUIRED,
                None,
                [policy_error or "Select an explicit crop policy."],
            )
        if policy.threshold_mm is None and not policy.demo:
            return (
                Status.POLICY_REQUIRED,
                None,
                [
                    "NO_APPROVED_THRESHOLD: physical measurements are withheld until an "
                    "ARL/Growize policy is supplied, or demo mode is explicitly enabled."
                ],
            )
        if len(instances) > 1:
            statuses = [i.decision.status if i.decision else Status.REVIEW_BORDERLINE for i in instances]
            if any(s == Status.OCCLUDED for s in statuses):
                return (
                    Status.MULTIPLE_OBJECTS,
                    None,
                    ["Multiple instances; one or more are occluded. See per-instance rows."],
                )
            if any(s == Status.REVIEW_BORDERLINE for s in statuses):
                return (
                    Status.MULTIPLE_OBJECTS,
                    None,
                    ["Multiple instances; at least one is borderline. See per-instance rows."],
                )
            return (
                Status.MULTIPLE_OBJECTS,
                None,
                ["Multiple vegetables detected; no single aggregate size class is emitted."],
            )
        decision = instances[0].decision
        if decision is None:
            return Status.REVIEW_BORDERLINE, None, ["No decision was produced."]
        if decision.status == Status.ACCEPTED:
            return Status.ACCEPTED, decision.size_class, list(decision.reasons)
        return decision.status, None, list(decision.reasons)

    def _invalid_result(
        self,
        result_id: str,
        context: MediaContext,
        reason: str,
        started: float,
    ) -> InspectionResult:
        result = InspectionResult(
            result_id=result_id,
            context=context,
            status=Status.INVALID_INPUT,
            reasons=[reason],
            software_version=__version__,
            pipeline_version=self._pipeline_version,
            pipeline_sha256=self._code_fingerprint,
        )
        result.processing_ms = (time.perf_counter() - started) * 1000.0
        return result

    def _model_unavailable_result(
        self,
        result_id: str,
        context: MediaContext,
        original_shape: tuple[int, int, int],
        started: float,
        processing_bgr: np.ndarray,
    ) -> InspectionResult:
        result = InspectionResult(
            result_id=result_id,
            context=context,
            status=Status.MODEL_UNAVAILABLE,
            reasons=[
                "The optional learned segmentation model is not installed or has not "
                "passed dataset/licensing gates. The deterministic OpenCV baseline "
                "is available; select it to continue."
            ],
            original_shape=original_shape,
            processed_shape=tuple(int(v) for v in processing_bgr.shape),
            software_version=__version__,
            pipeline_version=self._pipeline_version,
            pipeline_sha256=self._code_fingerprint,
        )
        result.annotated_image = render_inspection(processing_bgr, result)
        result.processing_ms = (time.perf_counter() - started) * 1000.0
        return result


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output
