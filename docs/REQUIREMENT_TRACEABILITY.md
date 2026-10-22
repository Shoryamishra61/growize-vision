# Requirement traceability

Source requirement set: `Growize_48h_Vegetable_Grading_Master_Spec.md` §3.
Statuses: **PASS** (implemented and tested), **PARTIAL**, **BLOCKED**,
**NOT IMPLEMENTED**, **NOT EVALUATED**. A unit test does not make a real-data or
device requirement pass.

## Functional requirements

| ID | Implementation | Tests / evidence | Status | Limitation |
|---|---|---|---|---|
| FR-001 | `media.py:load_image_file` | `tests/test_media.py` JPEG/PNG/WebP cases | **PASS** | Tested codecs are OpenCV-supported; exotic variants refused |
| FR-002 | `apps/gradio_app.py` webcam tab; `tracking.py` | `reports/preflight.json` camera capture | **PARTIAL** | Device frame captured; live browser streaming not recorded |
| FR-003 | `video.py:process_video`; MP4 UI tab | `tests/test_video.py`, `reports/preflight.json` H.264 output | **PARTIAL** | Programmatic video **PASS**; browser upload **NOT EVALUATED** |
| FR-004 | `config.py`, `media.py`, `video.py` bounds | `tests/test_media.py`, `tests/test_video.py` | **PASS** | Limits configurable; no fuzzing beyond tests |
| FR-005 | `media.py` EXIF/alpha/colour/scale | `tests/test_media.py` | **PASS** | Embedded ICC profiles not colour-managed |
| FR-006 | `media.py`, `video.py` bounded errors | `tests/test_media.py` corrupt/oversized/empty | **PASS** | Malicious codec exploit testing out of scope |
| FR-007 | Responsive Gradio layout | Manual layout only | **BLOCKED** | No physical phone round-trip |
| FR-008 | None | None | **BLOCKED** | HTTPS/secure-origin and device test absent |
| FR-009 | UI stop/pause; video cancellation | `tests/test_video.py` cancellation cleanup | **PASS** | Browser interaction not re-tested |
| FR-010 | None | None | **NOT IMPLEMENTED** | P1 batch mode |
| FR-011 | `segmentation_cv.py` | `tests/test_inference.py` accepted fixtures | **PASS** | Controlled matte background only |
| FR-012 | `inference.py` empty-object branch | empty-mat fixture/test | **PASS** | Shadow-only scenes may be ambiguous |
| FR-013 | edge/solidity/marker overlap checks | occlusion and edge-clipped tests | **PASS** | Severe self-occlusion not recoverable |
| FR-014 | connected components + count | multiple-object tests | **PASS** | Touching objects may merge and be refused |
| FR-015 | marker mask, background model | marker/geometry tests | **PASS** | Cluttered background can defeat model |
| FR-016 | None; operator crop selection | config tests | **NOT IMPLEMENTED** | No auto species claim; explicit fallback |
| FR-017 | `config.PolicyRegistry` | `tests/test_config.py` | **PASS** | Operator can select a wrong crop key |
| FR-018 | live 3-of-5 stability; video per-frame | `tests/test_tracking.py` | **PARTIAL** | No validated unique-object tracking in video |
| FR-019 | `markers.detect_best_marker` ArUco | marker detection tests | **PASS** | Synthetic/rendered markers; real print not evaluated |
| FR-020 | ID/quad/area/side validation | `tests/test_markers.py` | **PASS** | Extreme blur/glare can hide marker |
| FR-021 | homography projection | `tests/test_geometry.py` | **PASS** | Planar assumption; lens distortion not corrected |
| FR-022 | `inference` requires valid calibration for mm | no-marker tests | **PASS** | — |
| FR-023 | `CALIBRATION_REQUIRED` precedence | `tests/test_inference.py` | **PASS** | — |
| FR-024 | calibration ID, marker side, homography, error terms | export/JSON tests | **PASS** | Error bound is engineering, not statistical |
| FR-025 | `PhysicalMeasurement` named metric/unit; mask area separate | geometry/export tests | **PASS** | Straight-axis proxy for curved objects |
| FR-026 | `grading.evaluate_measurement` guard band | `tests/test_grading.py` | **PASS** | Threshold values themselves are `null` |
| FR-027 | `scripts/evaluate_baseline.py` real measurement check | `reports/baseline_evaluation.json` | **BLOCKED** | No caliper truth or real specimens |
| FR-028 | per-crop `physical_unit`; coriander disabled | `tests/test_config.py` | **PASS** config / **NOT EVALUATED** real | No validated leafy/bunch truth |
| FR-029 | `SizeClass`, `GradeDecision.accepted` | inference/UI tests | **PASS** | — |
| FR-030 | policy metric/threshold/comparator | config + grading tests | **PASS** | Real thresholds not supplied |
| FR-031 | `pipeline_sha256`, `model_sha=null`, policy version | `tests/test_exports.py`, `tests/test_inference.py` | **PASS** | No model hash because no model |
| FR-032 | `Status` + `reasons` separate from class | status tests | **PASS** | — |
| FR-033 | `tracking.LiveSession` 3-of-5 | `tests/test_tracking.py` | **PASS** unit / **NOT EVALUATED** camera soak | No field flicker study |
| FR-034 | `quality.assess_quality` gate | blur fixture/test | **PASS** | Thresholds are engineering defaults |
| FR-035 | `exports.py` per-object CSV | `tests/test_exports.py` | **PASS** | — |
| FR-036 | `rendering.py`, `video.py` H.264 | export/video tests, failure gallery | **PASS** | Browser download re-test pending for video |
| FR-037 | `VideoReport.summary` per-frame counts | video tests | **PASS** per-frame / **NOT EVALUATED** unique objects | No validated tracker |
| FR-038 | No appearance-only grading branch | code audit | **PASS by absence** | — |
| FR-039 | `dataset.py` provenance/licence/hash validator | `tests/test_dataset.py` | **PASS** tooling / **BLOCKED** real data | Empty template only |
| FR-040 | grouped deterministic splitter | leakage/group tests | **PASS** tooling / **BLOCKED** real split | No real manifest |
| FR-041 | baseline scripts; training refusal | `reports/training_blocked.json`, latency report | **PARTIAL** / training **BLOCKED** | No learned model |
| FR-042 | `evaluate_baseline.py` confusion/MAE | `reports/baseline_evaluation.json` | **NOT EVALUATED** | No real labels |
| FR-043 | demo labels, cards, report language | manual audit | **PASS** | Must remain enforced in future edits |
| FR-044 | None | None | **NOT IMPLEMENTED** | P1 crop information |
| FR-045 | None | None | **NOT IMPLEMENTED** | P2 telemetry; prohibited claim area |
| FR-046 | None | None | **NOT IMPLEMENTED** | P2 agronomy advisory |
| FR-047 | JSON audit, no raw masks/faces | export tests | **PASS** minimal / retention **NOT IMPLEMENTED** | No deletion policy |
| FR-048 | README, docs, cards, traceability | repository audit | **PASS** | Browser evidence caveats recorded |
| FR-049 | neural mode → `MODEL_UNAVAILABLE`; CPU baseline | inference test | **PASS** | — |
| FR-050 | bounded errors; safe paths | media/video/config tests | **PARTIAL** | Full-disk and camera-disconnect injection not run |

## Non-functional requirements

| Area | Implementation | Evidence | Status |
|---|---|---|---|
| Local-first privacy | localhost default; no cloud calls | code audit, README privacy note | **PASS** |
| Determinism | deterministic OpenCV path | `tests/test_inference.py` repeat check | **PASS** |
| Media safety | size/pixel limits, safe stems, traversal checks | `tests/test_media.py`, `tests/test_exports.py` | **PASS** |
| Availability | explicit refusals, no forced grade | inference tests | **PASS** |
| Performance | measured CPU latency, no SLA | `reports/latency_report.json` | **MEASURED** |
| Accessibility | text status + colour; keyboard controls | manual only | **PARTIAL** |
| Reproducibility | pinned deps, hashes, manifests | `pyproject.toml`, reports | **PASS** |
| Model provenance | no model claimed | `model_sha=null`, model card | **PASS** |
| Data provenance | validator rejects unknown licence/consent | `tests/test_dataset.py` | **PASS** tooling; real data absent |
| Claim discipline | synthetic labels and blocked statuses | report audit | **PASS** |

## Evidence index

- Automated suite: `reports/pytest_full.txt`
- Synthetic contract: `reports/synthetic_evaluation.json`
- CPU latency: `reports/latency_report.json`
- Preflight: `reports/preflight.json`
- Failure gallery: `reports/failure_gallery/index.md`
- Dataset gate: `reports/dataset_validation.json`
- Real evaluation: `reports/baseline_evaluation.json` (`BLOCKED`)
- Training: `reports/training_blocked.json` (`BLOCKED`)
- Mandatory handoff: `reports/DEMO_REPORT.md`
- Raw outputs: `test_evidence/` (index; some captures predate the final docs)
