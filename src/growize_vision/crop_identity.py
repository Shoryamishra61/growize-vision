"""Optional crop identity and explicitly uncalibrated image-space sizing.

Physical metrology belongs to the calibrated inference engine. These appearance
comparisons cannot establish millimeters, physical size, or biological provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from growize_vision.evidence import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIER_PATH = PROJECT_ROOT / "config" / "models" / "vegetable_species_classifier.pkl"
DEFAULT_DEEP_CLASSIFIER_PATH = PROJECT_ROOT / "config" / "models" / "species_classifier_v3.pkl"
DEFAULT_STATS_PATH = PROJECT_ROOT / "config" / "vegetable_population_statistics.json"


@dataclass
class SpeciesPrediction:
    species_key: str
    species_name: str
    confidence: float
    all_probabilities: dict[str, float]


@dataclass
class AppearanceSizeEstimate:
    species_key: str
    pixel_area: float
    major_axis_px: float
    ref_median_area_px: float
    ref_median_axis_px: float
    area_ratio_to_median: float
    appearance_size_class: str  # "SMALL", "BIG", "BORDERLINE"
    disclaimer: str = "UNCALIBRATED_APPEARANCE_ESTIMATE: Normalized 224-pixel footprint versus training-image median. Framing, zoom and distance change this estimate; no physical size or agricultural standard is inferred."


class CropIdentityService:
    def __init__(
        self,
        classifier_path: Path | str | None = None,
        stats_path: Path | str | None = None,
        deep_classifier_path: Path | str | None = None,
    ):
        self.classifier_path = Path(classifier_path or DEFAULT_CLASSIFIER_PATH)
        self.allow_classic_fallback = classifier_path is not None
        self.deep_classifier_path = Path(deep_classifier_path or DEFAULT_DEEP_CLASSIFIER_PATH)
        self.stats_path = Path(stats_path or DEFAULT_STATS_PATH)
        self.model = None
        self.deep_model = None
        self.backbone = None
        self.deep_transform = None
        self.is_deep = False
        self.classes = []
        self.class_to_idx = {}
        self.stats = {}
        self._load()

    def _load(self) -> None:
        # Load Deep Vision Model if available (MobileNetV3 backbone + logistic classifier)
        if self.deep_classifier_path.exists():
            try:
                import timm
                import torch
                import torchvision.transforms as T
                with open(self.deep_classifier_path, "rb") as f:
                    deep_data = pickle.load(f)
                self.deep_model = deep_data["classifier"]
                self.model_manifest_sha256 = deep_data.get("manifest_sha256")
                self.classes = [str(label) for label in self.deep_model.classes_]
                if self.classes != [str(label) for label in deep_data["classes"]]:
                    raise ValueError("Classifier probability columns do not match artifact classes")
                self.confidence_floor = float(deep_data.get("confidence_floor", 0.65))
                if not 0 < self.confidence_floor <= 1:
                    raise ValueError("Invalid confidence threshold")
                if "backbone_file" in deep_data:
                    backbone_path = (self.deep_classifier_path.parent / deep_data["backbone_file"]).resolve()
                    if backbone_path.parent != self.deep_classifier_path.parent.resolve():
                        raise ValueError("Backbone path must be beside the classifier")
                    if hashlib.sha256(backbone_path.read_bytes()).hexdigest() != deep_data["backbone_sha256"]:
                        raise ValueError("Backbone checksum mismatch")
                    self.backbone = timm.create_model(deep_data["backbone_name"], pretrained=False, num_classes=0)
                    self.backbone.load_state_dict(torch.load(backbone_path, map_location="cpu", weights_only=True))
                else:
                    self.backbone = timm.create_model(deep_data["backbone_name"], pretrained=True, num_classes=0)
                self.backbone.eval()
                self.deep_transform = T.Compose([
                    T.Lambda(self._resize_species_image) if deep_data.get("format_version") == 2 else T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                self.is_deep = True
                logger.info("Loaded species classifier with %d classes", len(self.classes))
            except Exception as e:
                logger.warning(f"Could not load deep vision classifier: {e}")

        # Fallback to standard classifier if deep model not available
        if not self.is_deep and self.allow_classic_fallback and self.classifier_path.exists():
            try:
                with open(self.classifier_path, "rb") as f:
                    data = pickle.load(f)
                    self.model = data.get("model")
                    self.classes = data.get("classes", [])
                    self.class_to_idx = data.get("class_to_idx", {})
            except Exception as e:
                logger.warning(f"Could not load species classifier from {self.classifier_path}: {e}")

        if self.stats_path.exists():
            try:
                with open(self.stats_path, encoding="utf-8") as f:
                    self.stats = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load population statistics from {self.stats_path}: {e}")

    @property
    def is_classifier_available(self) -> bool:
        return self.is_deep or (self.model is not None)

    @staticmethod
    def _resize_species_image(image):
        from PIL import Image

        return Image.fromarray(cv2.resize(np.asarray(image), (224, 224), interpolation=cv2.INTER_AREA))

    @staticmethod
    def detect_produce_presence(img_bgr: np.ndarray) -> tuple[bool, str]:
        """Screen obvious blank/stroke inputs; passing does not prove produce presence."""
        if img_bgr is None or img_bgr.size == 0:
            return False, "EMPTY_FRAME"

        h, w = img_bgr.shape[:2]
        if h < 20 or w < 20:
            return False, "FRAME_TOO_SMALL"
        if max(h, w) > 960:
            scale = 960 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
            h, w = img_bgr.shape[:2]

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        h_ch = hsv[:, :, 0]

        mean_val = float(np.mean(val))
        if mean_val < 20.0:
            return False, "FRAME_TOO_DARK"

        std_gray = float(np.std(gray))
        sat_p95 = float(np.percentile(sat, 95))
        chroma = np.sqrt((lab[:, :, 1].astype(float) - 128.0) ** 2 + (lab[:, :, 2].astype(float) - 128.0) ** 2)
        chroma_p95 = float(np.percentile(chroma, 95))

        # A uniformly colored surface is not evidence of a produce instance.
        if std_gray < 8.0 and float(np.max(np.std(img_bgr, axis=(0, 1)))) < 8.0:
            return False, "EMPTY_UNIFORM_SCENE"

        # Edge-Case Gate 1: Non-Produce / Document / Text / Signature Filter
        # Paper / white background detection:
        white_paper = (gray >= 165) & (sat <= 50)
        white_fraction = float(np.mean(white_paper))

        if white_fraction >= 0.40:
            ink_mask = ((gray < 160) | (sat > 60)).astype(np.uint8)
            total_fg = int(np.sum(ink_mask > 0))
            if total_fg < 150:
                return False, "NON_PRODUCE_EMPTY_PAPER"

            cnts, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return False, "NON_PRODUCE_DOCUMENT"

            c = max(cnts, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            rect_width, rect_height = cv2.minAreaRect(c)[1]
            rect_area = float(rect_width * rect_height)
            # Axis-aligned fill falls sharply when a slender pod is rotated.
            fill_ratio = area / rect_area if rect_area > 0 else 0.0

            hull = cv2.convexHull(c)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area > 0 else 0.0

            # Rejection rule A: Thin line strokes, hollow bounding box, or handwriting
            if fill_ratio < 0.25 or solidity < 0.35:
                return False, f"NON_PRODUCE_DOCUMENT_OR_TEXT (fill={fill_ratio:.2f}, solidity={solidity:.2f})"

            # Rejection rule B: Artificial blue/cyan ballpoint ink
            b_ch = img_bgr[:, :, 0].astype(np.int16)
            r_ch = img_bgr[:, :, 2].astype(np.int16)
            # Text elsewhere in the frame must not veto the primary object.
            primary_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(primary_mask, [c], -1, 1, cv2.FILLED)
            fg_idx = (primary_mask > 0) & (ink_mask > 0)
            blue_ink = (b_ch > r_ch + 10) & (h_ch >= 95) & (h_ch <= 140) & fg_idx
            blue_ratio = float(np.sum(blue_ink)) / float(max(1, np.sum(fg_idx)))
            if blue_ratio >= 0.15:
                return False, f"NON_PRODUCE_BLUE_INK (blue_ratio={blue_ratio:.2f})"

        # If spatial contrast exists, check for solid organic contour
        if std_gray >= 12.0:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_pixels = float(h * w)
            large_cnts = [c for c in cnts if cv2.contourArea(c) >= 0.015 * total_pixels]
            if large_cnts:
                return True, "FOREGROUND_CANDIDATE"

        # If high chromatic saturation exists (e.g. green okra, red tomato, orange carrot):
        if sat_p95 >= 30.0 or chroma_p95 >= 15.0:
            return True, "CHROMATIC_CANDIDATE"

        return False, "NO_PRODUCE_DETECTED"

    def extract_features(self, img_bgr: np.ndarray) -> np.ndarray | None:
        if img_bgr is None or img_bgr.size == 0:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        
        total_pixels = float(img_bgr.shape[0] * img_bgr.shape[1])
        if total_pixels == 0:
            return None

        h_hist = cv2.calcHist([img_hsv], [0], None, [16], [0, 180]).flatten() / total_pixels
        s_hist = cv2.calcHist([img_hsv], [1], None, [8], [0, 256]).flatten() / total_pixels
        v_hist = cv2.calcHist([img_hsv], [2], None, [8], [0, 256]).flatten() / total_pixels

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_ratio = 0.0
        aspect = 1.0
        solidity = 0.0
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            area_ratio = area / total_pixels
            x, y, w, h = cv2.boundingRect(c)
            aspect = float(w) / h if h > 0 else 1.0
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0

        means = np.concatenate([img_rgb.mean(axis=(0, 1)), img_hsv.mean(axis=(0, 1)), img_lab.mean(axis=(0, 1))]) / 255.0
        stds = np.concatenate([img_rgb.std(axis=(0, 1)), img_hsv.std(axis=(0, 1)), img_lab.std(axis=(0, 1))]) / 255.0

        return np.concatenate([h_hist, s_hist, v_hist, means, stds, [area_ratio, aspect, solidity]])

    def predict_species(
        self, img_bgr: np.ndarray, *, confidence_floor: float | None = None
    ) -> SpeciesPrediction | None:
        """Return a supported model prediction, or abstain when it is unavailable.

        Confidence is the classifier score, not a guarantee of correctness or
        proof that an object is produce. Geometry must never rewrite that score.
        ``confidence_floor`` may be lowered by callers for explicitly labelled
        low-confidence automation; it never raises the stored floor.
        """
        if (
            not isinstance(img_bgr, np.ndarray)
            or img_bgr.dtype != np.uint8
            or img_bgr.ndim != 3
            or img_bgr.shape[2] != 3
            or img_bgr.size == 0
            or not self.is_classifier_available
        ):
            return None
        is_present, _ = self.detect_produce_presence(img_bgr)
        if not is_present:
            return None

        try:
            if self.is_deep:
                if self.deep_model is None or self.backbone is None:
                    return None
                import torch
                from PIL import Image

                image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                tensor = self.deep_transform(image).unsqueeze(0)
                with torch.inference_mode():
                    features = self.backbone(tensor).cpu().numpy()
                probabilities = self.deep_model.predict_proba(features)[0]
            else:
                if self.model is None:
                    return None
                features = self.extract_features(img_bgr)
                if features is None:
                    return None
                probabilities = self.model.predict_proba([features])[0]
            return self._prediction_from_probabilities(probabilities, confidence_floor)
        except Exception:
            # A broken optional provider must leave calibrated/manual inspection usable.
            logger.exception("Species inference failed; withholding automatic crop selection")
            return None

    def _prediction_from_probabilities(
        self, probabilities: np.ndarray, confidence_floor: float | None = None
    ) -> SpeciesPrediction | None:
        probabilities = np.asarray(probabilities, dtype=float)
        if (
            probabilities.shape != (len(self.classes),)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)
        ):
            logger.warning("Species provider returned invalid probabilities")
            return None
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        floor = getattr(self, "confidence_floor", 0.65)
        if confidence_floor is not None:
            floor = min(float(confidence_floor), floor)
        if confidence < floor:
            return None
        species = str(self.classes[index])
        if species == "Unknown":
            return None
        return SpeciesPrediction(
            species_key=species.lower(),
            species_name=species.replace("_", " "),
            confidence=confidence,
            all_probabilities={str(label): float(probabilities[i])
                               for i, label in enumerate(self.classes)},
        )

    @staticmethod
    def appearance_contour(img_bgr: np.ndarray) -> np.ndarray | None:
        """Extract one complete silhouette in a fixed 224-pixel reference frame."""
        from .config import SegmentationSettings
        from .segmentation_cv import segment_foreground

        if img_bgr is None or img_bgr.size == 0:
            return None
        normalized = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        segmentation = segment_foreground(
            normalized, None, None, SegmentationSettings(),
            min_area_mm2=0.0, min_area_px=350.0,
        )
        # The busy flag guards millimetre grading; for an appearance ratio a
        # single clean, edge-free, convex-enough object is the real gate.
        if len(segmentation.objects) != 1:
            return None
        obj = segmentation.objects[0]
        if obj.touches_edge or "LOW_SOLIDITY_POSSIBLY_MERGED" in obj.flags:
            return None
        return obj.contour_px

    def estimate_appearance_size(
        self, img_bgr: np.ndarray, species_name: str | None = None
    ) -> AppearanceSizeEstimate | None:
        if img_bgr is None or img_bgr.size == 0:
            return None
        if not self.detect_produce_presence(img_bgr)[0]:
            return None
        if species_name is None:
            prediction = self.predict_species(img_bgr)
            species_name = prediction.species_name if prediction else None
        if not species_name:
            return None
        target = species_name.strip().title().replace(" ", "_")
        reference = self.stats.get(target, {})
        # Legacy statistics mix image resolutions and cannot support a comparison.
        if reference.get("reference_size_px") != 224 or reference.get("method") != "controlled_background_v2":
            return None
        median_area = float(reference.get("area_median", 0))
        median_axis = float(reference.get("axis_median", 0))
        if not np.all(np.isfinite([median_area, median_axis])) or min(median_area, median_axis) <= 0:
            return None
        contour = self.appearance_contour(img_bgr)
        if contour is None:
            return None
        area = float(cv2.contourArea(contour))
        major_axis = float(max(cv2.minAreaRect(contour)[1]))
        ratio = area / median_area
        return AppearanceSizeEstimate(
            species_key=target.lower(), pixel_area=area, major_axis_px=major_axis,
            ref_median_area_px=median_area, ref_median_axis_px=median_axis,
            area_ratio_to_median=ratio,
            appearance_size_class="BIG" if ratio >= 1.0 else "SMALL",
        )

    def render_appearance_overlay(
        self, img_bgr: np.ndarray, est: AppearanceSizeEstimate, species_name: str | None = None
    ) -> np.ndarray:
        canvas = img_bgr.copy()
        contour = self.appearance_contour(img_bgr)
        if contour is None:
            return canvas
        scale = np.array([canvas.shape[1] / 224, canvas.shape[0] / 224])
        contour = np.rint(contour * scale).astype(np.int32)
        cv2.drawContours(canvas, [contour], -1, (0, 110, 45), 2, cv2.LINE_AA)
        label = f"APPEARANCE ONLY: {est.appearance_size_class} {est.species_key}"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(canvas, label, (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas
