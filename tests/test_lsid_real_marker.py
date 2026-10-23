"""Marker and oracle-geometry checks on an unchanged real physical capture.

No generated photographs/marker fixtures. This is not automatic segmentation or
universal size accuracy; the manually annotated leaf mask is explicitly oracle input.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest

from growize_vision.config import CalibrationSettings
from growize_vision.geometry import measure_contour
from growize_vision.markers import calibrate_from_marker, detect_best_marker

ROOT = Path(__file__).resolve().parents[1] / "data/external/lsid_beans_f42hwwrpgn_v3/pilot"


@pytest.fixture
def physical_leaf_capture():
    image_path = ROOT / "leaf_001_001.jpg"
    mask_path = ROOT / "leaf_001_001_segmentation.raw"
    annotation_path = ROOT / "leaf_001_001_annotation.xml"
    if not all(path.exists() for path in [image_path, mask_path, annotation_path]):
        pytest.skip("Licensed original LSID real pilot has not been acquired")
    return (
        cv2.imread(str(image_path)),
        np.fromfile(mask_path, dtype=np.uint8).reshape(512, 512),
        ET.parse(annotation_path),
    )


def test_real_custom_marker_requires_its_explicit_family(physical_leaf_capture):
    image, _, _ = physical_leaf_capture
    marker, _ = detect_best_marker(image, CalibrationSettings(), expected_id=0)
    assert marker is None
    marker, reasons = detect_best_marker(
        image, CalibrationSettings(dictionary="LSID_BEANS_3X3"), expected_id=0
    )
    assert marker is not None and not reasons
    assert marker.dictionary == "LSID_BEANS_3X3"


def test_real_leaf_oracle_area_has_physical_units(physical_leaf_capture):
    image, mask, annotation = physical_leaf_capture
    settings = CalibrationSettings(dictionary="LSID_BEANS_3X3")
    marker, _ = detect_best_marker(image, settings, expected_id=0)
    marker_side_mm = float(annotation.findtext(".//marker-real-area")) ** 0.5 * 10
    calibration = calibrate_from_marker(image.shape, marker, marker_side_mm, settings)
    contours, _ = cv2.findContours(
        (mask == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    measurement, _ = measure_contour(
        max(contours, key=cv2.contourArea), calibration, "projected_area_cm2", settings
    )
    independent_trace_area = float(annotation.findtext(".//leaf-area"))
    assert measurement.unit == "cm2"
    assert measurement.lower_mm <= independent_trace_area <= measurement.upper_mm
    assert measurement.coverage_note == "engineering_error_bound_not_statistical_95pct"


def test_real_capture_feature_extraction_does_not_generate_or_mutate_views(
    physical_leaf_capture, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts.benchmark_lsid_size import pixel_features

    image, _, _ = physical_leaf_capture
    original = image.copy()
    features = pixel_features(image)
    assert features.shape == (512 * 512, 18)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))
    assert np.array_equal(image, original)


def test_real_ambiguous_leaf_does_not_promote_small_background_fragment(monkeypatch):
    import pickle

    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project))
    from growize_vision.markers import marker_mask
    from scripts.benchmark_lsid_size import predicted_leaf_mask

    source = ROOT.parent / "LSID-Beans/168/images/leaf_168_001.jpg"
    checkpoint = project / "reports/real_size/lsid_beans/pixel_rf_v1/candidate.pkl"
    if not source.exists() or not checkpoint.exists():
        pytest.skip("Original real validation capture and frozen RF baseline are not available")
    image = cv2.imread(str(source))
    with checkpoint.open("rb") as handle:
        model = pickle.load(handle)["model"]
    settings = CalibrationSettings(dictionary="LSID_BEANS_3X3")
    marker, _ = detect_best_marker(image, settings, expected_id=0)
    predicted = predicted_leaf_mask(image, model, marker_mask(image.shape, marker, 8))
    assert not np.any(predicted)


def test_published_annotation_suffix_variants_preserve_capture_identity(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts.fetch_lsid_beans import annotation_capture_id

    assert annotation_capture_id("leaf_001_001_annotation.xml") == "leaf_001_001"
    assert annotation_capture_id("leaf_002_001_annotation_annotation.xml") == "leaf_002_001"
    with pytest.raises(ValueError, match="XML"):
        annotation_capture_id("leaf_002_001.jpg")


def test_real_scalar_estimate_is_labelled_and_matches_marker_area_ratio(
    physical_leaf_capture, monkeypatch
):
    from dataclasses import replace

    from growize_vision.config import load_inference_settings
    from growize_vision.domain import GradePolicy

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts.benchmark_lsid_size import physical_grade

    image, human_mask, annotation = physical_leaf_capture
    settings = load_inference_settings()
    settings = replace(
        settings, calibration=replace(settings.calibration, dictionary="LSID_BEANS_3X3")
    )
    marker, _ = detect_best_marker(image, settings.calibration, expected_id=0)
    calibration = calibrate_from_marker(image.shape, marker, 50, settings.calibration)
    original_homography = np.array(calibration.homography)
    policy = GradePolicy(
        crop_key="bean_leaf",
        crop_name="Bean leaf",
        physical_unit="single_leaf",
        metric="projected_area_cm2",
        metric_description="One leaf projected area",
        threshold_mm=25.451243842364534,
        approval_status="EMPIRICAL_REFERENCE",
    )
    result = physical_grade(
        image,
        (human_mask == 1).astype(np.uint8),
        marker,
        calibration,
        policy,
        settings,
        "marker_area_scalar",
    )
    contours, _ = cv2.findContours(
        (human_mask == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    expected = cv2.contourArea(max(contours, key=cv2.contourArea)) * 50**2 / marker.area_px / 100
    assert result["measurement"]["value"] == pytest.approx(expected, abs=0.001)
    assert result["measurement"]["method"] == "marker_area_weak_perspective_estimate"
    assert any("unobserved" in warning for warning in result["measurement"]["warnings"])
    assert np.array_equal(np.array(calibration.homography), original_homography)
