---
title: Growize Vision
emoji: 🥦
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: apache-2.0
---

# Growize Vision

**Calibrated, refusal-first SMALL/BIG vegetable size grading for a controlled matte station.**
One shared inference engine serves still-image upload, laptop webcam inspection, and MP4
upload; accepted observations receive exactly one of two labels, `SMALL` or `BIG`.

> ## Evidence status — read this first
>
> - **Physical size-model release is blocked.** No measured station reference and
>   independent test set are available. No universal size accuracy is established.
> - Working definition: below the same-kind measured training median is SMALL;
>   at or above it is BIG, with refusal for uncertain measurements.
> - Historical species checkpoints used augmented training views. The simplified
>   interface does not load them, apply appearance sizing, or enable demo cutoffs.
> - Current training/evaluation requires reviewed original real photographs.
>   Synthetic/composited images and generated augmentation are excluded.
> - A separate licensed RGB-D broccoli benchmark with measured diameters was
>   acquired for external validation only; it has not produced a sizing result.
> - See [current evidence and limits](reports/real_size/STATUS.md) and
>   [industrial sizing research](reports/real_size/INDUSTRIAL_SIZING_RESEARCH.md), plus the
>   [real reference data contract](data/real_size/README.md). Older sections below
>   describe underlying engine capabilities, not a validated size-model release.

## What the app does

The interface contains only image upload/clipboard, camera, video, and output.
There is no crop selector, appearance-size guess, or demo-size mode. Missing
physical scale or reference data produces a refusal instead of a size label.

The underlying controlled-station engine can, when supplied a validated policy
and calibration:

1. Decode and normalise the image, webcam frame, or sampled video frame.
2. Detect a printed ArUco marker and validate its geometry, ID, and user-measured
   outer side length.
3. Segment the foreground against the controlled background and exclude the marker.
4. Project the silhouette onto the marker plane and compute a named physical metric.
5. Apply the crop policy with a guard band around the threshold.
6. Emit `SMALL`, `BIG`, or an explicit non-acceptance state with a reason and audit
   record.

Only `SMALL` and `BIG` are accepted size classes. `NO_OBJECT`,
`CALIBRATION_REQUIRED`, `POLICY_REQUIRED`, `OCCLUDED`, `REVIEW_BORDERLINE`,
`LOW_IMAGE_QUALITY`, `OUT_OF_DISTRIBUTION`, `MODEL_UNAVAILABLE`,
`INVALID_INPUT`, and `MULTIPLE_OBJECTS` are operational states, never a third grade.

## Quick start (Windows, verified)

```powershell
cd "C:\code_shit\projects\agriculture vegetables"
python -m pip install -e .
python -m pytest tests/test_real_photo_contract.py tests/test_config.py tests/test_grading.py -q
python apps\gradio_app.py
```

Open <http://127.0.0.1:7860>. The server binds to `127.0.0.1` by default, has no
authentication and no cloud upload path, and must not be exposed to an untrusted
network. Environment overrides: `GROWIZE_HOST`, `GROWIZE_PORT`.

Verified stack: Python 3.10.11, OpenCV 4.11.0, NumPy 2.2.6, Gradio 4.36.1,
FastAPI 0.112.2, Starlette 0.37.2, Pydantic 2.8.2, imageio-ffmpeg 0.6.0.
See `pyproject.toml` for the pinned dependencies.

## What `SMALL` / `BIG` means physically

There is no universal vegetable size rule. Each crop policy names a physical unit,
a metric, a threshold, a comparator, and a guard band:

| Crop | Physical unit | Metric | Real cut-off | Demo cut-off |
|---|---|---|---|---:|---:|
| Potato | single tuber | projected minimum-area-rectangle short side | `null` | 60 mm |
| Onion | single bulb | projected minimum-area-rectangle short side | `null` | 55 mm |
| Tomato | single fruit | projected minimum-area-rectangle short side | `null` | 58 mm |
| Okra | single pod | projected minimum-area-rectangle long side | `null` | 100 mm |
| Cucumber | single fruit | projected minimum-area-rectangle long side | `null` | 150 mm |
| Spinach | single leaf | projected minimum-area-rectangle long side | `null` | 100 mm |
| Coriander | undefined leaf/bunch | projected area | `null` | none |

The real threshold is applied only when ARL/Growize supplies and approves it. Demo
mode uses `demo_threshold_mm` and stamps every result and export with
`ASSESSMENT_DEMO_ONLY`. The comparator is `>=`; the 0.5 mm guard band means an
uncertainty interval that touches the threshold yields `REVIEW_BORDERLINE` rather
than a forced class. See `config/grade_rules.yaml` and `docs/CALIBRATION.md`.

## Supported inputs and tested scope

| Path | Status | Tested scope |
|---|---|---|
| JPEG / PNG / WebP upload | **PASS** | Unit tests plus browser upload, inspection, PNG/JSON/CSV download |
| Laptop webcam, localhost | **PASS (device capture)** | One-frame OpenCV capture in preflight; live UI path implemented; see `reports/DEMO_REPORT.md` for the browser-session result |
| MP4 upload | **PASS** | Programmatic H.264 export, no audio, per-frame labels; browser UI smoke test recorded in `reports/DEMO_REPORT.md` |
| Phone gallery upload | **BLOCKED** | No physical phone/network round-trip in this environment |
| Phone live camera over HTTPS | **BLOCKED** | No secure origin/device test; `getUserMedia` needs HTTPS outside localhost |
| Multi-image batch mode | **NOT IMPLEMENTED** | P1 scope, not shipped |
| Trained segmentation model | **NOT SHIPPED** | No licensed measured data, QA'd masks, or compatible GPU |
| Autonomous sorter / actuator | **OUT OF SCOPE** | No hardware, safety case, or field validation |

The segmentation baseline requires a plain matte background. Cluttered farm scenes
are outside the validated domain and produce `OUT_OF_DISTRIBUTION` when detected.

## Architecture

```text
image / webcam / MP4
        │
        ▼
 media decode + normalisation + size limits
        │
        ▼
 quality gate ──────────────► LOW_IMAGE_QUALITY
        │
        ▼
 ArUco detection + measured-side validation
        │ invalid ───────────► CALIBRATION_REQUIRED
        ▼
 foreground segmentation + marker exclusion
        │ empty ─────────────► NO_OBJECT
        │ edge/overlap ──────► OCCLUDED / MULTIPLE_OBJECTS
        ▼
 projected-plane metrology + uncertainty envelope
        │
        ▼
 crop policy (metric, threshold, guard band)
        │ no approval ───────► POLICY_REQUIRED
        ▼
 SMALL / BIG / REVIEW_BORDERLINE
        │
        ▼
 annotated image + JSON audit + CSV + optional H.264 video
```

Every input adapter calls `InferenceService.inspect_frame`; there is no separate
webcam or video grading logic. See `docs/ARCHITECTURE.md`.

## Current verification and data status

Real-only bean-leaf research now includes612 independently measured leaves from306 plants.
All three segmentation/size candidates failed real validation and remain outside the app.
See [current experiment state](reports/real_size/lsid_beans/current_state.json).
The frozen best candidate made5 wrong decisions among67 accepted clean held-out captures
(120 total,55.83% coverage). It was rejected for release; this is not universal size accuracy.


- Focused numerical/provenance/UI-empty-state checks: **62 passed**.
- Original measured station reference and test manifest: **empty / BLOCKED**.
- Size-model accuracy and physical measurement error: **not evaluated**.
- External real broccoli benchmark: **20 RGB/depth sets, 94 diameter annotations**,
  acquired, checksummed and decoded only. Reserved for validation, not reference fitting.
- Historical species checkpoints and synthetic reports are retained for provenance;
  they do not satisfy the current original-real-photo-only release requirement.
- The active simplified UI does not load a species model or enable physical grades
  without the required scale and measured reference. Camera produce trials and real
  produce video grading remain unverified.

Current receipts: [size status](reports/real_size/STATUS.md),
[physical validation](reports/real_size/validation.json),
[external source](reports/real_size/external_validation_source.json),
and [UI checks](reports/real_size/ui_checks.json).

## Current commands

```powershell
python -m pytest tests/test_real_photo_contract.py tests/test_config.py tests/test_grading.py -q
python scripts/build_size_reference.py --manifest data/real_size/manifest.jsonl
python scripts/validate_real_size.py --manifest data/real_size/manifest.jsonl
python apps/gradio_app.py
```

The two size-data commands currently exit BLOCKED because the station manifest is
empty. Follow [the real-photo contract](data/real_size/README.md) to supply independent
measured specimens. Do not generate, train on, or evaluate synthetic/composited images.
The older synthetic demo, benchmark and gallery scripts are historical tooling,
not the current training, evaluation or reviewer workflow.

## Repository layout

```text
apps/gradio_app.py                 local Gradio UI
src/growize_vision/                core engine and contracts
config/                            policy, inference, camera settings
assets/                            printable ArUco marker
data/demo/                         synthetic fixtures + manifest
data/dataset_manifest_template.json
scripts/                           evidence, validation, split, training refusal
tests/                             164 automated tests
reports/                           raw evidence, metrics, failure gallery
docs/                              PRD, SRS, cards, calibration, traceability
```

## Limitations and next steps

The full list is in `docs/KNOWN_LIMITATIONS.md`. The shortest version:

- No real measured produce, no real accuracy, no real threshold approval.
- One visible object on a plain matte plane only; no clutter, occlusion, stacking,
  hidden extent, weight, volume, soil, nutrition, or yield claims.
- Single overhead planar homography cannot recover the top surface of a tall
  object; parallax/height bias is bounded by an engineering envelope, not a
  statistical 95% interval.
- To move forward: print and physically measure the marker, collect 60–100 measured
  specimens with repeat caliper readings and consent/licence metadata, freeze the
  ARL policy, run the grouped frozen test, and only then consider a learned
  segmenter trained on QA'd masks from that licensed dataset.

## Documentation

- `docs/PRD.md`, `docs/SRS.md` — delivered scope and requirements
- `docs/ARCHITECTURE.md` — modules, contracts, status precedence
- `docs/CALIBRATION.md` — marker, plane geometry, uncertainty
- `docs/DATASET_CARD.md`, `docs/MODEL_CARD.md` — honest data/model disclosures
- `docs/VALIDATION_PLAN.md` — what was tested, what remains blocked
- `docs/REQUIREMENT_TRACEABILITY.md` — requirement → code → test → evidence
- `docs/KNOWN_LIMITATIONS.md` — non-goals and failure modes
- `reports/DEMO_REPORT.md` — mandatory evaluator evidence report
- `reports/EXPERIMENTS.md` — baseline vs absent model, no fabricated comparison
