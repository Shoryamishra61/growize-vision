# Product requirements — delivered scope

**Product:** Growize Vision local vegetable size-grading station
**Build:** 0.1.0, 22 October 2026
**Source of truth:** `Growize_48h_Vegetable_Grading_Master_Spec.md` and
`Growize_Vision_Research_PRD_SRS.md`. This document records what was actually
delivered, not aspirations.

## Problem

Growize/ARL needs a repeatable way to decide whether a visible vegetable is
`SMALL` or `BIG` under a defined crop policy, with a physical basis and an explicit
refusal when scale, quality, or policy is not trustworthy. Raw pixel size or an
uncalibrated photo classifier cannot support that decision.

## Users

- **Station operator:** selects crop, enters the measured marker side, captures an
  image or webcam frame, and acts on the displayed class or refusal.
- **QA/engineering reviewer:** inspects exported JSON/CSV, calibration trace,
  quality flags, policy version, and reason codes.
- **Data/ML engineer:** uses the manifest validator and grouped splitter to build a
  leakage-safe measured dataset before any training claim.

## Product goals

1. Accept only `SMALL` or `BIG` as physical size classes.
2. Require a measured ArUco reference for any millimetre claim.
3. Refuse rather than guess on missing scale, no object, blur, clipping, occlusion,
   multiple objects, unsupported background, absent policy, or absent threshold.
4. Use one inference contract for image, webcam, and video.
5. Make every decision auditable: policy version, calibration ID, metric,
   uncertainty, reasons, and software/pipeline hashes.
6. Keep the release path CPU-only, deterministic, local, and independent of neural
   weights.

## In scope (delivered)

- JPEG/PNG/WebP decode with EXIF orientation, alpha compositing, grayscale
  normalisation, byte/pixel limits, and safe filenames.
- Browser image upload, laptop webcam tab, and MP4 upload in a Gradio UI.
- ArUco marker detection, geometry validation, measured-side validation, and
  projected-plane homography.
- Controlled-background segmentation with marker exclusion and quality gates.
- Crop policies for potato, onion, tomato, okra, cucumber, spinach, and coriander,
  with real thresholds `null` and separate demo cut-offs where defensible.
- Guard-band grading and explicit non-acceptance states.
- PNG/JSON/CSV exports; H.264 MP4 annotated output; per-frame CSV; no audio.
- Live 3-of-5 stability display; no unique-object claim for video.
- Manifest validator, grouped splitter, synthetic contract evaluator, CPU latency
  benchmark, preflight, failure gallery, and training-refusal stub.

## Out of scope / not delivered

- Trained segmentation or classification model; no weights, training run, or model
  accuracy.
- Real measured dataset; no real-produce accuracy, MAE, or coverage.
- ARL-approved thresholds; real policies remain `TBD`.
- Autonomous sorting, conveyor integration, actuators, or safety certification.
- Weight, volume, density, soil, NPK, pH, pesticide, nutrition, yield, or defect
  claims.
- Universal farm-scene detection, arbitrary backgrounds, or hidden-extent recovery.
- Phone live camera over HTTPS and phone gallery upload as tested paths.
- Multi-image batch mode.

## Success criteria and current state

| Criterion | Target | Current |
|---|---|---|
| Binary contract | Accepted outputs exactly SMALL/BIG | **PASS** (tests + UI) |
| Physical basis | No mm without valid measured marker | **PASS** (tests + synthetic) |
| Refusal behaviour | Explicit statuses with reasons | **PASS** (8/8 synthetic checks) |
| Input coverage | Image, laptop webcam, MP4 | Image **PASS**; webcam device **PASS**; UI/video smoke evidence in `reports/DEMO_REPORT.md` |
| Automated quality | Full suite green | **PASS**, 164 tests |
| Real measured validation | Frozen measured specimens | **NOT EVALUATED** |
| Approved policy | ARL thresholds | **BLOCKED** — not supplied |
| Trained model comparison | Baseline vs model on holdout | **BLOCKED** — no model/data/GPU |

Release state: **YELLOW**. A functional calibrated pilot exists; it is not a
production or field-validated grader.
