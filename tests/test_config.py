"""Policy and runtime-config validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from growize_vision.config import (
    ConfigError,
    config_digest,
    load_camera_profiles,
    load_inference_settings,
    load_policies,
)


def write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def policy_payload(**crop_overrides) -> dict:
    crop = {
        "display_name": "Potato",
        "aliases": ["aloo"],
        "physical_unit": "single_tuber",
        "metric": "min_area_rect_short_mm",
        "metric_description": "test",
        "threshold_mm": None,
        "demo_threshold_mm": 60.0,
        "comparator": "greater_than_or_equal",
    }
    crop.update(crop_overrides)
    return {
        "version": "test-policy-1",
        "approval_summary": "No approved thresholds.",
        "guard_band_mm": 0.5,
        "crops": {"potato": crop},
    }


def test_project_policies_carry_published_reference_provenance() -> None:
    policies = load_policies()
    assert policies.version
    assert policies.approval_summary
    for key in policies.crop_keys:
        policy = policies.policy(key)
        assert policy.demo is False
        if policy.threshold_mm is None:
            # Only crops with no defensible size definition or station fit stay
            # refused: coriander (undefined unit), pumpkin/ash gourd (exceed mat).
            assert key in {"coriander", "pumpkin", "ash_gourd"}
            assert policy.approval_status == "NOT_PROVIDED"
            assert policy.bands is None
            continue
        assert policy.threshold_mm is not None and policy.threshold_mm > 0
        assert policy.approval_status == "PUBLISHED_REFERENCE"
        assert policy.provenance is not None
        assert policy.provenance["source"].strip()
        assert policy.provenance["status"] == "PUBLISHED_REFERENCE"
        assert policy.provenance["definition"].strip()
        assert policy.source == policy.provenance["source"]


def test_threshold_without_provenance_is_rejected(tmp_path) -> None:
    payload = policy_payload(threshold_mm=55.0)
    path = write_yaml(tmp_path / "no_provenance.yaml", payload)
    with pytest.raises(ConfigError, match="threshold_provenance"):
        load_policies(path)
    payload["threshold_provenance"] = {"source": " ", "status": "PUBLISHED_REFERENCE"}
    path = write_yaml(tmp_path / "blank_provenance.yaml", payload)
    with pytest.raises(ConfigError, match="threshold_provenance"):
        load_policies(path)


def test_published_size_bands_load_for_graded_crops() -> None:
    policies = load_policies()
    expected_edges = {
        "potato": [0.0, 40.0, 60.0],
        "onion": [0.0, 40.0, 60.0],
        "tomato": [0.0, 47.0, 67.0],
        "cucumber": [0.0, 150.0, 200.0],
        "capsicum": [0.0, 64.0, 76.0],
        "beetroot": [0.0, 40.0, 60.0],
    }
    for key, edges in expected_edges.items():
        policy = policies.policy(key)
        assert policy.bands is not None, key
        assert [band.min_mm for band in policy.bands] == edges, key
    for key in ("okra", "spinach"):
        assert policies.policy(key).bands is None, key


def test_demo_thresholds_are_opt_in_and_labelled() -> None:
    policies = load_policies()
    strict = policies.policy("potato")
    demo = policies.policy("potato", demo=True)
    assert strict.threshold_mm == 50.0
    assert strict.approval_status == "PUBLISHED_REFERENCE"
    assert demo.threshold_mm == 60.0
    assert demo.demo is True
    assert demo.approval_status == "ASSESSMENT_DEMO_ONLY"
    assert "not supplied or approved" in demo.source
    assert demo.policy_version.endswith("/demo")


def test_coriander_has_no_strict_or_demo_threshold() -> None:
    policies = load_policies()
    assert policies.policy("coriander").threshold_mm is None
    demo = policies.policy("coriander", demo=True)
    assert demo.threshold_mm is None
    assert demo.demo is False
    assert demo.approval_status == "NOT_PROVIDED"
    assert "No ARL-approved or assessment-demo threshold" in demo.source


def test_alias_lookup_is_case_and_space_insensitive() -> None:
    policies = load_policies()
    assert policies.policy("aloo").crop_key == "potato"
    assert policies.policy("  ALOO  ").crop_key == "potato"
    assert policies.policy("ladyfinger").crop_key == "okra"
    with pytest.raises(ConfigError, match="Unknown crop policy"):
        policies.policy("dragonfruit")


def test_ontology_reports_supported_and_unsupported_crops() -> None:
    policies = load_policies()
    assert policies.ontology("potato").supported is True
    assert policies.ontology("coriander").supported is False
    assert policies.ontology("palak").key == "spinach"


def test_policy_registry_loads_from_custom_file(tmp_path) -> None:
    path = write_yaml(tmp_path / "rules.yaml", policy_payload())
    policies = load_policies(path)
    assert policies.guard_band_mm == 0.5
    assert policies.config_sha256
    assert policies.crop_keys == ["potato"]
    assert policies.policy("aloo").crop_name == "Potato"


def test_policy_file_missing_or_malformed_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_policies(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("[not: a: mapping", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_policies(bad)
    non_mapping = tmp_path / "list.yaml"
    non_mapping.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_policies(non_mapping)
    empty = write_yaml(tmp_path / "empty.yaml", {"crops": {}})
    with pytest.raises(ConfigError, match="non-empty crops"):
        load_policies(empty)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update({"guard_band_mm": -0.1}),
        lambda payload: payload["crops"]["potato"].update({"metric": "weight_guess"}),
        lambda payload: payload["crops"]["potato"].update({"threshold_mm": -2}),
        lambda payload: payload["crops"]["potato"].update({"threshold_mm": "TBD"}),
        lambda payload: payload["crops"]["potato"].update({"comparator": "approx"}),
        lambda payload: payload["crops"]["potato"].pop("physical_unit"),
        lambda payload: payload["crops"].update({"potato": "not-a-mapping"}),
    ],
)
def test_unsafe_policy_configs_are_rejected(tmp_path, mutator) -> None:
    payload = policy_payload()
    mutator(payload)
    path = write_yaml(tmp_path / "rules.yaml", payload)
    with pytest.raises(ConfigError):
        load_policies(path)


def test_runtime_settings_load_and_validate() -> None:
    settings = load_inference_settings()
    assert settings.max_processing_dimension > 0
    assert settings.max_upload_bytes >= settings.max_image_pixels // 100
    assert settings.calibration.dictionary == "DICT_4X4_50"
    assert len(settings.config_sha256) == 64
    assert settings.to_dict()["version"] == settings.version
    assert settings.demo_enabled_by_default is False


def test_invalid_runtime_settings_are_rejected(tmp_path) -> None:
    path = write_yaml(tmp_path / "inference.yaml", {"max_processing_dimension": 0})
    with pytest.raises(ConfigError, match="max_processing_dimension"):
        load_inference_settings(path)
    path = write_yaml(
        tmp_path / "quality.yaml",
        {"quality": {"max_glare_fraction": 1.5}},
    )
    with pytest.raises(ConfigError, match="max_glare_fraction"):
        load_inference_settings(path)


def test_camera_profiles_load() -> None:
    data = load_camera_profiles()
    assert "cameras" in data
    assert isinstance(data["cameras"], dict)


def test_config_digest_is_order_sensitive_and_content_sensitive(tmp_path) -> None:
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    first.write_text("a: 1\n", encoding="utf-8")
    second.write_text("b: 2\n", encoding="utf-8")
    assert config_digest(first, second) == config_digest(first, second)
    assert config_digest(first, second) != config_digest(second, first)
    second.write_text("b: 3\n", encoding="utf-8")
    assert config_digest(first, second) != config_digest(first, tmp_path / "a.yaml")


def test_project_demo_size_bands_load_for_potato() -> None:
    policies = load_policies()
    strict = policies.policy("potato")
    demo = policies.policy("potato", demo=True)
    assert strict.bands is not None  # published-reference bands (40/60 mm)
    assert [b.min_mm for b in strict.bands] == [0.0, 40.0, 60.0]
    assert demo.bands is not None and len(demo.bands) == 3
    assert [b.name for b in demo.bands] == ["small", "medium", "large"]
    assert [b.min_mm for b in demo.bands] == [0.0, 45.0, 65.0]
    assert demo.demo is True
    assert demo.approval_status == "ASSESSMENT_DEMO_ONLY"


def test_size_bands_validation_rejects_broken_tables(tmp_path) -> None:
    broken = [
        ("single_band", [{"name": "small", "min_mm": 0}]),
        ("duplicate_names", [{"name": "a", "min_mm": 0}, {"name": "A", "min_mm": 10}]),
        ("first_not_zero", [{"name": "a", "min_mm": 5}, {"name": "b", "min_mm": 10}]),
        ("not_ascending", [{"name": "a", "min_mm": 0}, {"name": "b", "min_mm": 50}, {"name": "c", "min_mm": 30}]),
        ("bad_edge", [{"name": "a", "min_mm": 0}, {"name": "b", "min_mm": "big"}]),
        ("empty_name", [{"name": "", "min_mm": 0}, {"name": "b", "min_mm": 10}]),
    ]
    for label, bands in broken:
        path = write_yaml(tmp_path / f"{label}.yaml", policy_payload(size_bands=bands))
        with pytest.raises(ConfigError):
            load_policies(path)


def test_strict_size_bands_load_when_approved(tmp_path) -> None:
    bands = [
        {"name": "small", "min_mm": 0},
        {"name": "medium", "min_mm": 45},
        {"name": "large", "min_mm": 65},
    ]
    path = write_yaml(
        tmp_path / "bands.yaml",
        policy_payload(
            size_bands=bands,
            threshold_mm=45.0,
            approval_status="APPROVED_SOURCE",
            threshold_provenance={
                "status": "APPROVED_SOURCE",
                "source": "unit-test buyer specification",
            },
        ),
    )
    policies = load_policies(path)
    strict = policies.policy("potato")
    assert strict.bands is not None and len(strict.bands) == 3
    assert strict.approval_status == "APPROVED_SOURCE"


def test_registry_covers_twenty_plus_gradeable_crops() -> None:
    policies = load_policies()
    gradeable = [k for k in policies.crop_keys if policies.policy(k).threshold_mm is not None]
    refused = {k for k in policies.crop_keys if policies.policy(k).threshold_mm is None}
    assert len(gradeable) >= 21
    assert refused == {"coriander", "pumpkin", "ash_gourd"}
    # Hindi/common aliases resolve to the right policy.
    for alias, key in [
        ("baingan", "brinjal"),
        ("karela", "bitter_gourd"),
        ("turai", "ridge_gourd"),
        ("lauki", "bottle_gourd"),
        ("tindora", "ivy_gourd"),
        ("parwal", "pointed_gourd"),
        ("gajar", "carrot"),
        ("hari_mirch", "green_chilli"),
        ("shakarkand", "sweet_potato"),
        ("shimla_mirch", "capsicum"),
        ("chukandar", "beetroot"),
        ("shalgam", "turnip"),
    ]:
        assert policies.policy(alias).crop_key == key


def test_all_gradeable_crops_carry_provenance() -> None:
    policies = load_policies()
    for key in policies.crop_keys:
        policy = policies.policy(key)
        if policy.threshold_mm is None:
            continue
        assert policy.approval_status == "PUBLISHED_REFERENCE", key
        assert policy.provenance and policy.provenance["source"].strip(), key
        assert policy.provenance["definition"].strip(), key


def test_pumpkin_and_ash_gourd_are_honestly_unsupported() -> None:
    policies = load_policies()
    for key in ("pumpkin", "ash_gourd"):
        policy = policies.policy(key)
        assert policy.threshold_mm is None
        assert policies.ontology(key).supported is False
