# Software requirements specification — delivered implementation

This is the delivered SRS, not the original planning prompt. Requirement IDs match
the master spec. Statuses are `PASS`, `PARTIAL`, `BLOCKED`, `NOT IMPLEMENTED`, or
`NOT EVALUATED`; evidence paths are relative to the repository root.

## 1. Runtime and interfaces

| ID | Requirement | Implementation | Status / evidence |
|---|---|---|---|
| FR-001 | JPEG/PNG/WebP ingestion | `src/growize_vision/media.py` | **PASS** — `tests/test_media.py` |
| FR-002 | Browser webcam snapshots/stream | `apps/gradio_app.py` live tab, `tracking.py` | **PASS (device capture)** — preflight opened the USB camera; browser session in `reports/DEMO_REPORT.md` |
| FR-003 | MP4 ingestion | `src/growize_vision/video.py` | **PASS** — `tests/test_video.py`, preflight H.264 output |
| FR-004 | File-size/resolution/duration limits | `media.py`, `video.py`, config YAML | **PASS** — `tests/test_media.py`, `tests/test_video.py` |
| FR-005 | EXIF/colour/alpha/scale normalisation | `media.py` | **PASS** — `tests/test_media.py` |
| FR-006 | Corrupt/unsupported/oversized refusal | `media.py`, `video.py` | **PASS** — `tests/test_media.py` |
| FR-007 | Responsive mobile image upload | Gradio responsive layout | **PARTIAL** — layout works; no physical phone round-trip, so **BLOCKED** as device evidence |
| FR-008 | Phone live camera over HTTPS | None | **BLOCKED** — no HTTPS/device test |
| FR-009 | Stop webcam/video | UI pause/cancel; video cancel check | **PASS** — `tests/test_video.py`, UI wiring |
| FR-010 | Multi-image batch | None | **NOT IMPLEMENTED** (P1) |

## 2. Segmentation and identity

| ID | Requirement | Implementation | Status / evidence |
|---|---|---|---|
| FR-011 | Isolated vegetable segmentation | `segmentation_cv.py` | **PASS** — synthetic controlled-mat tests |
| FR-012 | Empty scene without invention | segmentation + inference status | **PASS** — `NO_OBJECT` fixture/test |
| FR-013 | Flag clipping/occlusion | edge + marker + solidity checks | **PASS** — `OCCLUDED` fixture/tests |
| FR-014 | Multiple instances | connected components + status | **PASS** — multiple-object tests |
| FR-015 | Exclude marker/tray/shadow | marker mask + background model | **PASS** — relevant unit tests; real clutter **NOT EVALUATED** |
| FR-016 | Crop identity model | None; operator selects crop | **NOT IMPLEMENTED** — explicit fallback |
| FR-017 | User-selected crop policy | `PolicyRegistry` | **PASS** — `tests/test_config.py` |
| FR-018 | Trustworthy video tracking | Live-only stability; video is per-frame | **PARTIAL** — per-frame counts; no unique-object claim |

## 3. Calibration and measurement

| ID | Requirement | Implementation | Status / evidence |
|---|---|---|---|
| FR-019 | Known-size fiducial detection | `markers.py` ArUco | **PASS** — generated/perspective marker tests |
| FR-020 | Validate dictionary/ID/geometry/size | `detect_markers`, `calibrate_from_marker` | **PASS** — wrong-ID, side, area, tiny-marker tests |
| FR-021 | Perspective correction | homography | **PASS** — geometry/marker tests |
| FR-022 | Physical metric only with scale | `geometry.measure_contour` + status order | **PASS** — no-marker refusal test |
| FR-023 | `CALIBRATION_REQUIRED` instead of mm | `inference.py` | **PASS** — fixture/test |
| FR-024 | Record calibration source/error budget | `Calibration.to_dict`, uncertainty fields | **PASS** — export/JSON tests |
| FR-025 | Named metric, units, mask area separate | `PhysicalMeasurement`, `InstanceObservation` | **PASS** — geometry/export tests |
| FR-026 | Guard band around threshold | `grading.py` | **PASS** — boundary tests |
| FR-027 | Caliper/ruler validation on new devices | `evaluate_baseline.py` | **BLOCKED** — no measured specimens |
| FR-028 | Explicit leafy unit policy | per-crop `physical_unit`; coriander disabled | **PASS** for config; real leafy validation **NOT EVALUATED** |

## 4. Grading and results

| ID | Requirement | Implementation | Status / evidence |
|---|---|---|---|
| FR-029 | Only SMALL/BIG accepted | `SizeClass`, `GradeDecision.accepted` | **PASS** — inference/UI tests |
| FR-030 | Crop-specific metric/threshold/comparator | `config.py`, `grading.py` | **PASS** — config/threshold tests |
| FR-031 | Store policy/model/method versions | result JSON/CSV | **PASS** — `pipeline_sha256`, `model_sha=null`, policy version |
| FR-032 | Reason separate from class | `Status`, `reasons` | **PASS** — status tests |
| FR-033 | Avoid live grade flicker | `tracking.py` 3-of-5 stabilization | **PASS** — `tests/test_tracking.py`; real camera soak **NOT EVALUATED** |
| FR-034 | No grade on blur/quality failure | `quality.py` gate | **PASS** — blur fixture/test |
| FR-035 | Per-item table export | `exports.py` CSV | **PASS** — `tests/test_exports.py` |
| FR-036 | Annotated still/video | `rendering.py`, `video.py` | **PASS** — gallery, H.264 test |
| FR-037 | Video crop/grade aggregation | `VideoReport.summary` | **PASS** for per-frame counts; no unique-object claim |
| FR-038 | Appearance-only grade clearly uncalibrated | Not offered | **PASS by absence** — no appearance branch |

## 5. Learning, operations, and data

| ID | Requirement | Implementation | Status / evidence |
|---|---|---|---|
| FR-039 | Manifest provenance/licence | `dataset.py` validator | **PASS** — `tests/test_dataset.py` |
| FR-040 | Group integrity across splits | grouped deterministic splitter | **PASS** — leakage tests |
| FR-041 | Baseline training/evaluation logged | Baseline evaluated; no training | **PARTIAL** — baseline CPU latency/synthetic contract; training **BLOCKED** |
| FR-042 | Per-crop/size confusion tables | `evaluate_baseline.py` | **NOT EVALUATED** — no real data |
| FR-043 | No synthetic evidence presented as real | UI labels, reports, cards | **PASS** — manual audit of every artifact |
| FR-044 | Curated crop information separate | None | **NOT IMPLEMENTED** (P1) |
| FR-045 | Farm/soil telemetry | None | **NOT IMPLEMENTED** (P2) |
| FR-046 | Qualified agronomic advisory | None | **NOT IMPLEMENTED** (P2) |
| FR-047 | Concise session log, no raw media/PII | JSON audit, no raw mask/frame export | **PASS** — export tests; retention policy not implemented |
| FR-048 | README/setup/demo/disclosures | `README.md`, `docs/`, `reports/DEMO_REPORT.md` | **PASS** |
| FR-049 | Operate without optional model/GPU | `segmentation_mode=neural` returns `MODEL_UNAVAILABLE` | **PASS** — inference test |
| FR-050 | Fail safely on camera/codec/disk/config | Bounded errors; disk-full path not injected | **PARTIAL** — config/media/video failures tested; full-disk/manual disconnect **NOT EVALUATED** |

## 6. Non-functional requirements

| Area | Requirement | Status |
|---|---|---|
| Determinism | Same frame and config produce the same decision | **PASS** — inference determinism test |
| Local-first privacy | No cloud upload, localhost default | **PASS** — code audit; no network calls in inference path |
| Security | Path traversal and unsafe filenames blocked | **PASS** — media/export tests |
| Accessibility | Keyboard-reachable Gradio controls, text status plus colour | **PARTIAL** — no formal WCAG audit |
| Performance budget | Report measured CPU latency, no SLA | **PASS** — 116.0 ms median / 124.3 ms p95 on this workstation |
| Reproducibility | Pinned dependencies, config digests, manifests | **PASS** — `pyproject.toml`, digests in reports |
| Model provenance | No model claimed | **PASS** — `model_sha=null`, model card says absent |
| Data provenance | Real data requires licence/consent/measurement | **PASS** validator; real data absent |
| Unsupported claims | No yield/soil/nutrition/robotics claims | **PASS** — UI/report audit |

## 7. State contract

`InspectionResult.status` is one of the operational states listed in the README.
`size_class` is non-null only when `status == ACCEPTED`. This is enforced in
`domain.py`, `grading.py`, and tested in `tests/test_inference.py`.

## 8. Known deviations

- `model_sha` exists in the JSON contract for provenance compatibility but is
  `null`; the deterministic pipeline hash is recorded as `pipeline_sha256`.
- The sample app uses the same process for webcam and image; no multi-user session
  isolation or authentication is implemented. Keep it localhost-only.
- Video samples frames at a configured rate; counts are per-frame observations, not
  physical items.
