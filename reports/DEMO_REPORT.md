# Growize Vision — mandatory evidence report

**Assessment:** 48-hour calibrated vegetable size grading  
**Build date:** 2026-10-22  
**Report timestamp:** 2026-10-22T17:12:22Z  
**Deadline:** not supplied in the workspace; the brief describes a 48-hour assessment  
**Release decision:** **YELLOW**

## 1. Start/end timestamp and deadline

| Item | Value |
|---|---|
| Build session | 2026-10-22; exact start was not instrumented |
| Earliest archived report timestamp | **Not captured**; see individual report files |
| Report generation | 2026-10-22T17:12:22Z |
| Deadline | **Not supplied**; assessment context is 48 hours |
| Time remaining / schedule claim | **Not asserted** — no deadline artifact exists |

## 2. Working functionality and exact device/browser tested

| Functionality | Status | Exact environment/scope |
|---|---|---|
| Still image upload, segmentation, calibration, grading | **PASS** | Desktop browser session against `127.0.0.1:7860`; synthetic potato fixture produced `ACCEPTED`/`BIG`, 72.3 ± 2.6 mm; PNG/JSON/CSV downloads appeared |
| Demo policy refusal | **PASS** | Same desktop browser session; after unchecking demo mode the result panel showed `POLICY_REQUIRED` / `NO_APPROVED_THRESHOLD` with no size class |
| Laptop webcam device capture | **PASS** | OpenCV preflight opened the local camera and captured one frame; exact device name not logged |
| Browser live webcam streaming | **NOT EVALUATED** | Browser harness became unavailable after server restart; no successful live session recorded |
| MP4 programmatic processing | **PASS** | H.264/libx264, 6 sampled frames, no audio, annotated export and per-frame CSV in preflight |
| Browser MP4 upload | **NOT EVALUATED** | No successful live browser upload session after harness failure |
| Phone gallery upload | **BLOCKED** | No physical phone test |
| Phone live HTTPS camera | **BLOCKED** | No secure origin/device test |
| Multi-image batch | **NOT IMPLEMENTED** | Out of delivered scope |
| Trained segmentation model | **NOT AVAILABLE** | No weights, training run, or model evaluation |

Test environment: Windows 10, Python 3.10.11, CPU-only, RAM constrained
(~1.0–1.7 GB free during browser work), OpenCV 4.11.0, NumPy 2.2.6, Gradio 4.36.1,
FastAPI 0.112.2, Starlette 0.37.2, Pydantic 2.8.2, imageio-ffmpeg 0.6.0. The browser
product/version was not captured by the harness; do not infer a specific browser.

## 3. Dataset counts by unique physical specimen, crop, grade, source, and split

| Dimension | Count |
|---|---:|
| Real unique physical specimens | **0** |
| Real images | **0** |
| Real measured labels | **0** |
| Real crops represented | **0** |
| Real `SMALL` / `BIG` references | **0 / 0** |
| Real train / val / test split | **not created** |
| Synthetic still fixtures | 8 |
| Synthetic video fixtures | 1 MP4 |
| Synthetic contract checks | 8 passed |
| Licensed/consented real sources | **none** |

Synthetic fixtures cover potato and okra only, plus refusals (no marker, empty mat,
blur, edge clipping). They are **not** a sample of real produce and are excluded from
all accuracy claims. See `docs/DATASET_CARD.md` and
`data/dataset_manifest_template.json`.

## 4. Policy version and threshold origin

| Field | Value |
|---|---|
| Config file | `config/grade_rules.yaml` |
| Policy version | See loaded result `policy.version`; real thresholds remain `null` |
| Real threshold origin | **NOT SUPPLIED** by ARL/Growize |
| Demo thresholds | Engineering demonstration only, stamped `ASSESSMENT_DEMO_ONLY` |
| Comparator | `>=` (configured and validated) |
| Guard band | 0.5 mm default, configured per policy |
| Strict-mode behaviour | `POLICY_REQUIRED` / `NO_APPROVED_THRESHOLD`; no size class |
| Demo-mode behaviour | May assign `SMALL`/`BIG` using demo cutoff, clearly labelled |

No real threshold is invented or inferred. Coriander has no safe single-leaf/bunch
metric and no demo threshold.

## 5. Completed / failed / skipped / blocked test counts

| Level | Completed PASS | Failed | Skipped | Blocked / not evaluated |
|---|---:|---:|---:|---:|
| Unit/integration (`pytest`) | 164 | 0 | 0 | — |
| ML evaluation | 0 | 0 | 0 | 2 major tasks (real baseline, model comparison) |
| E2E browser — image | 1 | 0 | 0 | — |
| E2E browser — strict policy | 1 | 0 | 0 | — |
| E2E browser — webcam live | 0 | 0 | 0 | 1 |
| E2E browser — MP4 upload | 0 | 0 | 0 | 1 |
| E2E phone | 0 | 0 | 0 | 2 |
| Synthetic contract | 8 | 0 | 0 | — |
| Synthetic latency | 80 observations | 0 | 0 | — |
| Failure gallery (synthetic) | 4 | 0 | 0 | real gallery blocked |
| Dataset tooling | 2 (empty-template validator, leakage tests) | 0 | 0 | real data blocked |

`pytest` suite: **164 passed in 10.20 s** at the last captured run before the final
docs (`reports/pytest_full.txt`). Ruff: **All checks passed**. The final post-doc
capture was blocked by a shell harness failure; docs-only edits do not affect tests.
Test IDs and commands are mapped in `docs/REQUIREMENT_TRACEABILITY.md` and
`docs/VALIDATION_PLAN.md`.

## 6. Confusion matrix, measurements, time, uncertainty, confidence intervals

### Real produce

**NOT EVALUATED.** There are no real specimens, so there is no confusion matrix, no
balanced accuracy, no per-crop result, no measurement MAE, and no confidence
interval. These fields are `null` in `reports/metrics/final_results.json`.

### Synthetic contract and geometry

- Contract checks: **8/8 passed**.
- Geometry observations: 4 vector-defined fixtures.
- Median absolute geometry delta: **0.179 mm**.
- Maximum absolute geometry delta: **0.268 mm**.
- These values verify the transform/contour code path only. They are not real
  measurement accuracy and have no real confidence interval.

### Time / latency

- Protocol: 80 timed observations, one frame at a time, decode excluded.
- Median: **116.0 ms**; p95: **124.3 ms**; mean: **116.0 ms**.
- Scope: CPU-only build workstation; not throughput or field latency.

### Uncertainty

The application reports an engineering error envelope (marker measurement, corner
localisation, mask boundary, parallax). It is **not** a statistical 95% confidence
interval. The system refuses (`REVIEW_BORDERLINE`) when the envelope touches the
threshold.

## 7. Reproducibility commands, environment, artifact hashes

```powershell
python -m pip install -e .
python -m pytest -q
python -m ruff check .
python scripts\evaluate_synthetic.py
python scripts\benchmark_latency.py --runs 10 --warmup 2
python scripts\demo_preflight.py --full-video --camera
python scripts\build_failure_gallery.py
python scripts\finalize_evidence.py
python apps\gradio_app.py
```

- Environment: Windows 10, Python 3.10.11, CPU-only.
- Dependency pins: `pyproject.toml`.
- Artifact hashes: **BLOCKED in the final minutes** because the shell harness stopped
  responding. `scripts/build_checksums.py` is provided to generate
  `CHECKSUMS.sha256`; it was not successfully run after the final docs.
- Config digests: recorded in each result's policy/calibration provenance.
- Git commit: **none** at report generation; the workspace was not a Git repository.

## 8. Screenshots and recorded demo artifacts

| Artifact | Path / status |
|---|---|
| Failure gallery, 4 synthetic panels | `reports/failure_gallery/index.md` |
| Annotated image demo | Browser session; exported PNG/JSON/CSV (temporary Gradio files) |
| Video export | `reports/preflight.json` records H.264, 6 frames, no audio |
| Webcam capture | `reports/preflight.json` records one frame |
| Browser screenshots | **NOT ARCHIVED** — browser harness became unavailable after initial image/strict tests |
| Recorded webcam/image/video demo MP4s | **NOT PRODUCED** — no recorder and no completed browser video/webcam session |

No screenshot is fabricated or substituted from another session.

## 9. Honesty statement: demo vs tested vs blocked vs unvalidated

- **Demo:** synthetic fixtures and demo thresholds; labelled as assessment-only.
- **Tested:** 164 automated tests; 8 synthetic contract checks; one browser image
  session; strict-policy refusal; one-frame camera capture; programmatic H.264 video.
- **Blocked:** real measured dataset; approved thresholds; phone upload/live camera;
  full-disk/camera-disconnect injection; browser live webcam and MP4 UI sessions.
- **Not evaluated:** real-produce accuracy, confusion matrix, MAE, coverage, field
  robustness, unique-object video counts, accessibility certification.
- **Not trained:** no model, no weights, no checkpoints, no model accuracy.

## 10. Release decision and criteria

**YELLOW.**

Green is not justified: real-data validation, approved thresholds, and full browser
coverage are absent. Red is too strong for the shipped scope: the deterministic
baseline runs, the binary/refusal contract is tested, the app starts locally, and
the synthetic geometry and media paths pass. YELLOW means usable as a controlled
engineering demo with explicit refusals, not as a production or field grader.

## 11. Final acceptance matrix

Score convention for this evidence table: **1.0** = verified for the stated scope,
**0.5** = logic/tooling verified but the real-device/real-data claim is blocked or
not evaluated, **0.0** = not satisfied or not evaluated. This is an evidence score,
not an official external rubric.

| Domain | Score | Status | Evidence | Blocking? |
|---|---:|---|---|---|
| Evaluator scope locked to binary size | 1.0 | **PASS** | Tests + UI contract | No |
| Physical metric defined per crop | 1.0 | **PASS** (metric names/units) | `config/grade_rules.yaml`, tests | No |
| Real crop thresholds defined/approved | 0.0 | **BLOCKED** | Thresholds `null`; ARL approval absent | **Yes** |
| Webcam device capture | 1.0 | **PASS** | Preflight one-frame capture | No |
| Webcam browser start/stop | 0.0 | **NOT EVALUATED** | Browser harness unavailable | **Yes** for E2E claim |
| Photo upload and orientation | 1.0 | **PASS** | Automated media tests + browser image session | No |
| MP4 upload and result export | 0.5 | **PARTIAL** | Programmatic H.264/CSV PASS; browser UI NOT EVALUATED | **Yes** for E2E claim |
| Marker/scale validity | 0.5 | **PASS** synthetic | Marker/wrong-ID/missing tests; real print not evaluated | Real print **yes** |
| Threshold and guard band | 1.0 | **PASS** logic | Grading tests; real values `null` | No for logic; **yes** for real policy |
| At least one measured small/big pair | 0.0 | **BLOCKED** | No real measured specimens | **Yes** |
| Dataset manifest/provenance | 0.5 | **PASS** tooling / **BLOCKED** real | Validator tests; empty template | **Yes** for real data |
| Grouped split leakage safety | 0.5 | **PASS** tooling / **BLOCKED** real | Group split tests | **Yes** for real data |
| Frozen real test evaluation | 0.0 | **NOT EVALUATED** | No data | **Yes** |
| Model baseline comparison | 0.0 | **BLOCKED** | No model | **Yes** if model claimed |
| README/docs/packaging | 1.0 | **PASS** | This repository | No |
| Final demo rehearsal | 0.5 | **PARTIAL** | Script + local evidence; no recorded rehearsal | **Yes** for demo claim |

**Total evidence score: 8.0 / 16 (50%).** The missing half is concentrated in real
data, approved policy, model comparison, and unrecorded browser/device sessions.

## 12. Exact required test scenarios

| IDs | Scenario | Status | Note |
|---|---|---|---|
| T-ING-01–16 | Decode, corrupt/empty/oversized, alpha/EXIF/scale/aspect | **PASS** | `tests/test_media.py` |
| T-MOB-01–03 | Mobile image upload | **BLOCKED** | No physical phone |
| T-MOB-04–07 | Live phone camera | **BLOCKED** | No HTTPS/device |
| T-WEB-01–04 | Webcam snapshots/streaming | **PARTIAL** | Device capture PASS; live browser NOT EVALUATED |
| T-WEB-05 | Permission denied | **NOT EVALUATED** | Not injected |
| T-VID-01–08 | Video metadata, sampling, limits, overlay, export, cancel, audio | **PASS** programmatic | `tests/test_video.py`, preflight |
| T-VID-09–10 | Corrupt/mobile video | **PARTIAL** | Corrupt handled in tests; mobile not tested |
| T-SEC-01–05 | Limits, traversal, no remote fetch, localhost, claims | **PASS** | Media/export tests + code audit; camera privacy not instrumented |
| T-CAL-01–08 | Known length, perspective, marker errors, no mm without scale | **PASS** synthetic | Real ruler/print BLOCKED |
| T-SEG-01–05 | Mask, empty, occlusion, marker/blob confusion | **PASS** synthetic | Real clutter NOT EVALUATED |
| T-GRADE-01–08 | Threshold, guard band, binary-only, reasons | **PASS** | `tests/test_grading.py`, `tests/test_inference.py` |
| T-DEF-01–05 | Optional defects | **NOT IMPLEMENTED** | Out of scope |
| T-ROB-01–05 | Lighting, angle, blur, background, unknown crop | **PARTIAL** | Blur/unknown crop tests; real lighting/angle NOT EVALUATED |
| T-PERF-01–03 | Latency/throughput | **MEASURED** | 116.0/124.3 ms; no SLA |
| T-UI-01–05 | Input, error clarity, colour+text, keyboard | **PARTIAL** | Browser image observed; no formal usability/WCAG audit |
| T-DATA-01–05 | Manifest, leakage, provenance, synthetic separation | **PASS** tooling | Real data BLOCKED |
| T-ML-01–05 | Honest baseline/model comparison | **BLOCKED** | No model/data |

## 13. Evidence packet index

| Path | Purpose |
|---|---|
| `README.md` | Start, status, commands, scope |
| `reports/pytest_full.txt` | Last archived automated test output |
| `reports/synthetic_evaluation.json` | 8/8 synthetic contract checks and geometry |
| `reports/latency_report.json` | 80-observation CPU latency |
| `reports/preflight.json` | Environment, video, camera preflight |
| `reports/failure_gallery/index.md` + PNGs | Synthetic refusal gallery |
| `reports/metrics/final_results.json` | Real metrics null/not evaluated; synthetic/latency scope noted |
| `reports/metrics/specimen_predictions.csv` | Header-only; no real predictions exist |
| `reports/metrics/condition_slices.csv` | Header-only; no real slices exist |
| `reports/baseline_evaluation.json` | BLOCKED real-data gate |
| `reports/training_blocked.json` | BLOCKED training record |
| `reports/EXPERIMENTS.md` | Experiment log |
| `reports/DEMO_REPORT.md` | This mandatory report |
| `CHECKSUMS.sha256` | **BLOCKED** in final capture; generator provided as `scripts/build_checksums.py` |
| `docs/REQUIREMENT_TRACEABILITY.md` | Requirement → code → test → evidence |
| `docs/KNOWN_LIMITATIONS.md` | Full disclosure |
| `test_evidence/README.md` | Raw-output index and capture limitation |

## 14. Honest limitations and disclosure

1. No learned model was trained; no weights or checkpoints exist.
2. No real measured dataset was supplied; real accuracy and MAE are not evaluated.
3. No ARL-approved threshold exists; real cutoffs are `null`.
4. Synthetic fixtures verify code paths, not real-produce performance.
5. Browser live webcam and MP4 upload were not completed after the harness failed.
6. Phone gallery/live camera paths are blocked.
7. Planar geometry cannot recover height, volume, weight, or hidden extent.
8. Video counts are per-frame observations, not unique vegetables.
9. The uncertainty interval is an engineering bound, not a statistical confidence
   interval.
10. The app is localhost-only and unauthenticated; do not expose it to a network.
11. The release is YELLOW, not production-ready, not field-validated, and not a
    buyer-compliance or safety certification.
12. Final JUnit/checksum capture was blocked by a shell harness failure. The last
    observed final-code run was 164 passed and Ruff clean.
13. If the evaluator wants real-image or model evidence, provide licensed, measured,
    consented images and a caliper protocol; otherwise those items remain blocked.
