# Model card

## 1. Model status: no learned model

| Field | Value |
|---|---|
| Model name | **None shipped** |
| Version | N/A |
| Weights / checkpoint | **None** |
| `model_sha` in results | `null` |
| Training run | **NOT RUN** |
| Training data | **NONE** |
| Evaluation data | **NONE** |
| Real-produce accuracy | **NOT EVALUATED** |
| GPU | Not used; CPU-only baseline |
| Release path | `opencv_baseline` |

`scripts/train_segmentation.py` is a deliberate refusal. It exits `BLOCKED` and
writes `reports/training_blocked.json`. No metric in this repository may be
interpreted as model performance.

## 2. Baseline that is shipped

The release path is a deterministic computer-vision pipeline, not a neural network:

1. Bounded media decode and normalisation.
2. Quality gate (blur, contrast, clipping, glare, object area).
3. ArUco marker detection and measured-side validation.
4. Homography from marker plane to millimetres.
5. Foreground segmentation against a controlled matte background, with marker
   exclusion and connected-component filtering.
6. Projected contour geometry (minimum-area rectangle, Feret, area).
7. Crop policy + guard band → `SMALL`, `BIG`, or a refusal state.
8. Annotated export and JSON/CSV audit.

There are no learned weights, no stochastic inference, and no GPU dependency.
`pipeline_sha256` identifies the code/config combination; it is **not** a model hash.

## 3. Why no model was trained

Training a credible segmenter requires all of:

- a licensed, consented, physically measured real-produce dataset;
- QA-approved masks with inter-annotator review;
- a leakage-safe grouped split fixed before training;
- an approved crop policy to validate against;
- a compatible training device and enough time for a reproducible run.

None of the first four exist in this workspace, and the environment is CPU-only for
Torch. Training on synthetic blobs or unlicensed web images would produce numbers
that look impressive and mean nothing. That would violate the assessment's evidence
rules, so the model was not trained.

## 4. Intended use

- Demonstrate a calibrated binary size decision on a controlled matte station.
- Verify that missing scale, quality failure, occlusion, unknown policy, and missing
  approval produce explicit refusals.
- Provide a reproducible baseline for a future measured-data comparison.

## 5. Out-of-scope use

- Any claim of real-produce accuracy, field robustness, or buyer compliance.
- Universal crop recognition; the operator selects the crop.
- Weight, volume, density, hidden extent, defects, pesticides, nutrition, soil, yield.
- Autonomous sorting, conveyor actuation, or safety certification.
- Phone live camera over plain HTTP; secure-origin and device testing are absent.
- Any image without a visible, measured marker on the same plane.

## 6. Measured evidence (baseline only)

| Metric | Value | Scope |
|---|---|---|
| Automated tests | 164 passed | Full repository suite |
| Synthetic contract checks | 8/8 passed | Status/refusal and known geometry |
| Synthetic geometry delta | median 0.179 mm, max 0.268 mm | 4 vector-defined fixtures |
| CPU latency | median 116.0 ms, p95 124.3 ms | 80 single-frame observations, decode excluded |
| Real balanced accuracy | **NOT EVALUATED** | No real data |
| Real measurement MAE | **NOT EVALUATED** | No real data |
| Real per-crop confusion | **NOT EVALUATED** | No real data |
| Model vs baseline comparison | **BLOCKED** | No model |

Synthetic numbers are engineering checks. They must not be quoted as accuracy.

## 7. Failure modes and mitigations

| Failure | Behaviour |
|---|---|
| No marker / bad marker | `CALIBRATION_REQUIRED`, no mm |
| Missing approved threshold | `POLICY_REQUIRED`, no mm in strict mode |
| Blur/glare/clipping | `LOW_IMAGE_QUALITY` |
| Object touches frame/marker | `OCCLUDED` |
| Multiple blobs | `MULTIPLE_OBJECTS`, per-instance rows |
| Busy background | `OUT_OF_DISTRIBUTION` |
| Bent pod / straight-axis bias | Warning flag; review recommended |
| Neural mode selected | `MODEL_UNAVAILABLE` |
| Tall object / parallax | Engineering uncertainty bound, not a 3-D correction |
| Real distribution shift | **Unknown** — no real evaluation |

## 8. Ethical, legal, and operational notes

- Local-only processing; no cloud inference or third-party upload.
- No personal data should be captured; faces/identity documents are unnecessary.
- No protected attribute or person-level inference is performed.
- Real data collection requires consent, licence, and retention review.
- The system can refuse; it must not be tuned to force a grade on invalid captures.
- A human remains responsible for the physical measurement and any operational use.

## 9. Promotion criteria for a future model

A model may replace or augment the baseline only after:

1. the measured dataset and licences pass the validator;
2. masks pass QA and the split is frozen and leakage-free;
3. the model is evaluated on sealed real specimens against the approved policy;
4. per-crop confusion, measurement error, refusal coverage, and subgroup results are
   reported with confidence intervals;
5. the baseline comparison is reproducible from a clean checkout;
6. a model card records weights, hashes, training data, preprocessing, and limits.
