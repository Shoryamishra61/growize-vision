"""Small checks for ingestion, grouping, and deterministic training views."""

import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["../escape.jpg", "/absolute.jpg", "C:/escape.jpg", "safe/../../escape.jpg", "..\\escape.jpg"])
def test_archive_traversal_is_rejected(name):
    validate = runpy.run_path(str(ROOT / "scripts/fetch_species_source.py"))["validate_member"]
    with pytest.raises(ValueError):
        validate(name)


def test_archive_relative_members_are_allowed():
    validate = runpy.run_path(str(ROOT / "scripts/fetch_species_source.py"))["validate_member"]
    validate("Vegetable Images/Okra/image.jpg")


def test_group_split_is_stable_and_all_members_follow_group():
    split = runpy.run_path(str(ROOT / "scripts/build_species_manifest.py"))["grouped_split"]
    assert [split("capture_1")] * 10 == [split("capture_1") for _ in range(10)]
    assert set(split(f"capture_{i}") for i in range(200)) == {"train", "validation", "test"}


def test_training_view_is_reproducible_and_preserves_input():
    augment = runpy.run_path(str(ROOT / "scripts/train_species_v2.py"))["augment"]
    image = np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    original = image.copy()
    assert np.array_equal(augment(image, 42), augment(image, 42))
    assert np.array_equal(image, original)


def test_full_rotation_view_is_reproducible_bounded_and_preserves_input():
    augment = runpy.run_path(str(ROOT / "scripts/train_species_v2.py"))["augment_full"]
    image = np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    original = image.copy()
    transformed = augment(image, 42)
    assert transformed.shape == (224, 224, 3)
    assert transformed.dtype == np.uint8
    assert np.array_equal(transformed, augment(image, 42))
    assert np.array_equal(image, original)


def test_perceptual_group_key_is_rotation_invariant():
    phash = runpy.run_path(str(ROOT / "scripts/build_species_manifest.py"))["perceptual_hash"]
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8))
    def canonical(frame):
        return min(phash(frame.rotate(angle, expand=True)) for angle in (0, 90, 180, 270))
    assert canonical(image) == canonical(image.rotate(90, expand=True))


def test_training_command_completes_evaluation_after_saving_checkpoint(tmp_path, monkeypatch):
    """Exercise orchestration with dummy embeddings, not image-accuracy evidence."""
    import json
    import sys
    from types import SimpleNamespace

    import torch
    from sklearn.linear_model import LogisticRegression

    module = runpy.run_path(str(ROOT / "scripts/train_species_v2.py"))
    namespace = module["main"].__globals__
    classes = ["Bean", "Okra", "Unknown"]
    embeddings = np.eye(3)
    incumbent = LogisticRegression().fit(embeddings, classes)
    fake = SimpleNamespace(is_deep=True, backbone=torch.nn.Identity(), deep_model=incumbent, classes=classes)
    monkeypatch.setitem(namespace, "CropIdentityService", lambda: fake)

    def dummy_extract(rows, service, cache_name, run_dir, **kwargs):
        assert isinstance(run_dir, Path)
        return np.stack([embeddings[classes.index(row["label"])] for row in rows])

    monkeypatch.setitem(namespace, "extract", dummy_extract)
    rows = [{"path": f"{split}_{label}.png", "label": label, "split": split,
             "group": f"{split}_{label}", "source": "synthetic_test", "sha256": "0" * 64}
            for split in ["train", "validation", "test"] for label in classes]
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(sys, "argv", ["train_species_v2.py", "--run-dir", str(tmp_path)])
    module["main"]()
    report = json.loads((tmp_path / "experiment.json").read_text())
    assert report["status"] == "EVALUATED_NOT_PROMOTED"
    assert (tmp_path / "candidate/deep_vegetable_classifier.pkl").exists()
    assert len((tmp_path / "test_predictions.jsonl").read_text().splitlines()) == 3
