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
    SourceType,
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
from .scale import calibrate_from_two_point_reference
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
    cross_check_species: bool | None = None  # default: on for still-image sources
    # No printed marker? Mark the two end points (original-image pixel
    # coordinates) of any object of known length and state that length.
    reference_points_px: tuple[float, float, float, float] | None = None
    reference_length_mm: float | None = None
    reference_uncertainty_mm: float = 2.0
    # Station mode for fixed cameras: the first frame that produces a valid
    # calibration (marker or reference length) is remembered; later frames are
    # graded from that calibration without the marker visible.  A visible
    # marker that disagrees with the stored scale triggers recalibration.
    station_mode: bool = False
    station_id: str | None = None


@dataclass
class InferenceService:
    policies: PolicyRegistry = field(default_factory=load_policies)
    settings: InferenceSettings = field(default_factory=load_inference_settings)
    neural_model: Any = None
    identity_service: Any = None
    # Fixed-camera station memory: calibrate once at commissioning, then grade
    # every item without the marker needing to stay visible.
    station_calibrations: dict[str, dict[str, Any]] = field(default_factory=dict)

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
        species_refusal: str | None = None
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
        reference_calibration = None
        if marker is None and request.reference_points_px and request.reference_length_mm:
            # Points are clicked on the original upload; map them into the
            # processed frame's coordinate system before calibrating.
            ax, ay, bx, by = request.reference_points_px
            reference_calibration = calibrate_from_two_point_reference(
                processing_bgr.shape,
                (ax * resize_scale, ay * resize_scale),
                (bx * resize_scale, by * resize_scale),
                request.reference_length_mm,
                self.settings.calibration,
                reference_uncertainty_mm=request.reference_uncertainty_mm,
            )
            if reference_calibration.valid:
                warnings.extend(reference_calibration.warnings)
            else:
                warnings.extend(reference_calibration.reasons)
        if marker is not None:
            calibration = calibrate_from_marker(
                processing_bgr.shape,
                marker,
                request.marker_side_mm,
                self.settings.calibration,
                request.marker_measurement_uncertainty_mm,
            )
        elif reference_calibration is not None and reference_calibration.valid:
            calibration = reference_calibration
        else:
            reasons = list(marker_reasons) or [
                "No usable calibration marker was found in this frame."
            ]
            if not (request.reference_points_px and request.reference_length_mm):
                reasons.append(
                    "Alternatively, mark the two end points of any object of "
                    "known length (ruler, card, coin) and provide that length."
                )
            calibration = Calibration(
                valid=False,
                calibration_id="unavailable",
                reasons=reasons,
            )

        station_key = request.station_id or (
            context.device if context.device not in (None, "", "unknown") else None
        )
        if request.station_mode and station_key:
            calibration = self._apply_station_calibration(
                station_key, calibration, processing_bgr.shape[:2], warnings
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
        elif policy and policy.approval_status == "PUBLISHED_REFERENCE":
            warnings.append(
                "PUBLISHED_REFERENCE threshold: literature/market reference, "
                "not validated on the local population."
            )

        if (
            request.crop_key
            and len(segmentation.objects) == 1
            and (
                request.cross_check_species
                if request.cross_check_species is not None
                else context.source_type == SourceType.IMAGE
            )
        ):
            mismatch = self._cross_check_species(input_bgr, request.crop_key)
            if mismatch:
                message, refuse = mismatch
                warnings.append(message)
                if refuse:
                    species_refusal = message

        instances = self._build_instances(
            processing_bgr=processing_bgr,
            segmentation=segmentation,
            calibration=calibration,
            quality=quality,
            policy=policy,
            policy_error=policy_error,
            request=request,
            warnings=warnings,
            species_refusal=species_refusal,
        )

        top_status, top_size, top_reasons = self._summarise(
            instances=instances,
            calibration=calibration,
            quality=quality,
            segmentation=segmentation,
            policy=policy,
            policy_error=policy_error,
        )
        top_band: str | None = None
        if len(instances) == 1 and instances[0].decision and instances[0].decision.band_name:
            top_band = instances[0].decision.band_name
            if top_status is Status.ACCEPTED:
                top_reasons.append(f"SIZE_BAND:{top_band}")
        if not instances:
            warnings.append(
                "No object was accepted as a foreground vegetable instance."
            )

        result = InspectionResult(
            result_id=result_id,
            context=context,
            status=top_status,
            size_class=top_size,
            size_band=top_band,
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
        species_refusal: str | None = None,
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
            elif "SEGMENTATION_DISAGREEMENT" in obj.flags:
                instance.decision = GradeDecision(
                    size_class=None,
                    status=Status.REVIEW_BORDERLINE,
                    reasons=[
                        "SEGMENTATION_DISAGREEMENT: two independent segmentation "
                        "methods produced different silhouettes; lighting or "
                        "background may be non-standard. Recapture on the matte "
                        "mat or grade this item manually."
                    ],
                    policy=policy,
                    policy_version=getattr(policy, "policy_version", None),
                    calibration_id=calibration.calibration_id,
                )
            elif species_refusal:
                instance.decision = GradeDecision(
                    size_class=None,
                    status=Status.REVIEW_BORDERLINE,
                    reasons=[species_refusal],
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
            elif policy.threshold_mm is None and not policy.bands and not policy.demo:
                instance.decision = evaluate_measurement(None, policy, calibration.calibration_id)
                instance.decision.reasons = [
                    "NO_THRESHOLD_POLICY: no ARL-approved or published-reference "
                    "threshold exists for this crop. Physical measurement is "
                    "withheld in strict mode."
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
                [
                    "Physical scale is missing: place the printed marker beside the "
                    "item, or mark the two end points of any object of known "
                    "length and provide that length.",
                    *calibration.reasons,
                ],
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
        if policy.threshold_mm is None and not policy.bands and not policy.demo:
            return (
                Status.POLICY_REQUIRED,
                None,
                [
                    "NO_THRESHOLD_POLICY: physical measurements are withheld until an "
                    "approved or published-reference policy exists, or demo mode is "
                    "explicitly enabled."
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

    def _apply_station_calibration(
        self,
        station_key: str,
        calibration: Calibration,
        frame_shape: tuple[int, int],
        warnings: list[str],
    ) -> Calibration:
        """Remember, reuse, and drift-check a fixed-camera station's scale.

        Commissioning: the first frame with a valid calibration stores it.
        Production frames: the stored scale is reused (same camera geometry)
        when no marker is visible, so items can cover the marker freely.  A
        visible marker that disagrees with the stored scale by more than 2%
        means the camera or mat moved: the station recalibrates and says so.
        """
        cached = self.station_calibrations.get(station_key)
        if calibration.valid:
            if cached is not None and cached["calibration"].mm_per_px:
                fresh = float(calibration.mm_per_px or 0.0)
                stored = float(cached["calibration"].mm_per_px)
                if fresh > 0 and abs(fresh - stored) / stored > 0.02:
                    warnings.append(
                        "STATION_CALIBRATION_DRIFT: the visible marker disagrees "
                        "with the stored station scale by more than 2% (camera or "
                        "mat moved). The station calibration was updated."
                    )
            self.station_calibrations[station_key] = {
                "calibration": replace(calibration),
                "shape": tuple(frame_shape),
            }
            return calibration
        if cached is None or tuple(cached["shape"]) != tuple(frame_shape):
            return calibration
        reused = replace(cached["calibration"])
        reused.warnings = list(reused.warnings) + [
            "STATION_CALIBRATION_REUSED: scale comes from this fixed camera's "
            "stored calibration (marker not visible in this frame). Commission "
            "with the marker visible periodically to re-verify."
        ]
        warnings.append(
            "Using stored station calibration; the marker was not visible."
        )
        return reused

    def _cross_check_species(
        self, frame: np.ndarray, crop_key: str
    ) -> tuple[str, bool] | None:
        """Warn — and above the configured confidence, refuse — on crop mismatch.

        A low-confidence disagreement stays advisory; a high-confidence one
        means the operator is probably applying the wrong crop's rule, so the
        grade is withheld for review rather than emitted with the wrong
        threshold.
        """
        try:
            canonical = self.policies.canonical_key(crop_key)
        except Exception:
            return None
        if self.identity_service is None:
            from .crop_identity import CropIdentityService

            self.identity_service = CropIdentityService()
        if not getattr(self.identity_service, "is_classifier_available", False):
            return None
        prediction = self.identity_service.predict_species(frame)
        if prediction is None or prediction.species_key == canonical:
            return None
        refusal_knob = self.settings.species_mismatch_refusal_confidence
        refuse = bool(refusal_knob > 0 and prediction.confidence >= refusal_knob)
        message = (
            f"SPECIES_MISMATCH_CANDIDATE: operator selected {canonical!r} but the "
            f"classifier suggests {prediction.species_key!r} "
            f"(confidence {prediction.confidence:.2f}). "
            + (
                "Confidence meets the refusal threshold; grading is withheld for review."
                if refuse
                else "Verify the crop; grading currently follows the operator selection."
            )
        )
        return message, refuse

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
