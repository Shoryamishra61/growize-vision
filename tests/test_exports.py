"""Export integrity and audit-provenance tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from growize_vision.exports import (
    export_inspection,
    export_video_report,
    flatten_result_rows,
    write_batch_csv,
)
from growize_vision.inference import InferenceRequest
from growize_vision.media import load_image_file


def inspect(service, demo_dir: Path, filename: str, crop: str = "potato"):
    decoded = load_image_file(demo_dir / filename)
    return service.inspect_frame(
        decoded.bgr,
        request=InferenceRequest(
            crop_key=crop,
            marker_side_mm=50.0,
            expected_marker_id=0,
            demo_policy=True,
        ),
    )


def test_inspection_export_writes_png_json_csv_and_paths(service, demo_dir, tmp_path) -> None:
    result = inspect(service, demo_dir, "potato_big_synthetic.png")
    paths = export_inspection(
        result, result.annotated_image, output_root=tmp_path, stem="potato_big"
    )
    assert set(paths) == {"annotated_image", "json", "csv"}
    for path in paths.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0
    assert Path(paths["json"]).parent == Path(paths["csv"]).parent
    assert result.exported_paths["json"] == paths["json"]

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["status"] == "ACCEPTED"
    assert payload["size_class"] == "BIG"
    assert payload["export"]["annotated_image"] == paths["annotated_image"]
    assert payload["export"]["json"] == paths["json"]
    assert payload["export"]["csv"] == paths["csv"]
    assert payload["exported_paths"]["json"] == paths["json"]
    assert len(payload["pipeline_sha256"]) == 64
    assert payload["model_sha"] is None
    assert "ARL-approved" in payload["export"]["notice"]

    with Path(paths["csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ACCEPTED"
    assert row["size_class"] == "BIG"
    assert row["crop_key"] == "potato"
    assert row["policy_is_demo"] == "True"
    assert row["calibration_valid"] == "True"
    assert row["value_mm"]
    assert row["model_sha"] == ""
    assert row["pipeline_sha256"] == payload["pipeline_sha256"]
    assert "SMALL" not in row["size_class"]


def test_no_object_export_writes_header_only_csv(service, demo_dir, tmp_path) -> None:
    result = inspect(service, demo_dir, "empty_mat_synthetic.png")
    paths = export_inspection(result, result.annotated_image, output_root=tmp_path)
    with Path(paths["csv"]).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    assert len(rows) == 1
    assert "result_id" in rows[0]


def test_batch_csv_unions_mixed_row_keys(service, demo_dir, tmp_path) -> None:
    accepted = inspect(service, demo_dir, "potato_small_synthetic.png")
    empty = inspect(service, demo_dir, "empty_mat_synthetic.png")
    rows = flatten_result_rows([accepted, empty])
    assert len(rows) == 2
    assert rows[0]["size_class"] == "SMALL"
    assert rows[1]["status"] == "NO_OBJECT"
    output = tmp_path / "batch.csv"
    write_batch_csv([accepted, empty], output)
    with output.open(newline="", encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 2
    assert exported[0]["size_class"] == "SMALL"
    assert exported[1]["status"] == "NO_OBJECT"


def test_video_report_csv_header_is_union_of_mixed_rows(tmp_path) -> None:
    frame_rows = [
        {"result_id": "r1", "frame_id": 0, "status": "NO_OBJECT", "size_class": None},
        {
            "result_id": "r2",
            "frame_id": 1,
            "status": "ACCEPTED",
            "size_class": "BIG",
            "value_mm": 72.1,
            "instance_id": "obj_001",
        },
    ]
    payload = {"status": "COMPLETED_WITH_REVIEW", "summary": {"unique_vegetable_count": None}}
    exports = export_video_report(
        report_payload=payload,
        frame_rows=frame_rows,
        output_root=tmp_path,
        stem="mixed",
    )
    assert set(exports) == {"json", "csv"}
    with Path(exports["csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["status"] == "NO_OBJECT"
    assert rows[1]["size_class"] == "BIG"
    assert "instance_id" in rows[0]
    written = json.loads(Path(exports["json"]).read_text(encoding="utf-8"))
    assert written["exports"]["csv"] == exports["csv"]
    assert written["summary"]["unique_vegetable_count"] is None


def test_export_stem_traversal_cannot_escape_output_root(service, demo_dir, tmp_path) -> None:
    result = inspect(service, demo_dir, "potato_small_synthetic.png")
    paths = export_inspection(
        result,
        result.annotated_image,
        output_root=tmp_path,
        stem="../../evil name",
    )
    for path in paths.values():
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())
        assert ".." not in Path(path).parts
    assert Path(paths["annotated_image"]).name.startswith("evil_name")


def test_exported_json_has_no_mask_arrays(service, demo_dir, tmp_path) -> None:
    result = inspect(service, demo_dir, "okra_small_synthetic.png", crop="okra")
    paths = export_inspection(result, result.annotated_image, output_root=tmp_path)
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert "instances" in payload
    assert payload["instances"][0]["decision"]["size_class"] == "SMALL"
    for instance in payload["instances"]:
        assert "mask" not in instance
        assert "contour_px" not in instance
        assert "contour_points" in instance


def test_export_can_omit_the_annotated_image(service, demo_dir, tmp_path) -> None:
    result = inspect(service, demo_dir, "potato_small_synthetic.png")
    paths = export_inspection(result, None, output_root=tmp_path)
    assert "annotated_image" not in paths
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["export"]["annotated_image"] is None
