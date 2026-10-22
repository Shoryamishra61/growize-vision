# Validation plan and current results

## 1. Rules

- A test is `PASS` only when the command or manual step actually ran and the
  expected assertion was observed.
- Unavailable hardware, data, or approvals are `BLOCKED`, not `PASS`.
- Synthetic fixtures prove code paths and geometry, never real-produce accuracy.
- Real metrics require unique physically measured specimens, a frozen policy, and a
  sealed test split.
- No number is copied from a requirement or target into an evidence field.

## 2. Test levels and commands

| Level | Command | Evidence | Current |
|---|---|---|---|
| Unit/integration | `python -m pytest -q` | `reports/pytest_full.txt` | **164 passed** |
| Lint | `python -m ruff check .` | command output | **PASS** |
| Synthetic contract | `python scripts\evaluate_synthetic.py` | `reports/synthetic_evaluation.json` | **8/8 PASS** |
| CPU latency | `python scripts\benchmark_latency.py --runs 10 --warmup 2` | `reports/latency_report.json` | **MEASURED** |
| Environment/media preflight | `python scripts\demo_preflight.py --full-video --camera` | `reports/preflight.json` | **PASS** |
| Failure gallery | `python scripts\build_failure_gallery.py` | `reports/failure_gallery/index.md` | **4 cases PASS** |
| Dataset gate | `python scripts\validate_dataset.py --manifest <path>` | `reports/dataset_validation.json` | Template **PASS as empty template**; real data **BLOCKED** |
| Group split | `python scripts\group_split.py --manifest <path>` | `reports/group_split_report.json` | Real data **BLOCKED**; empty template correctly rejected |
| Real baseline evaluation | `python scripts\evaluate_baseline.py --manifest <path> --marker-side-mm <mm>` | `reports/baseline_evaluation.json` | **BLOCKED** |
| Training | `python scripts\train_segmentation.py` | `reports/training_blocked.json` | **BLOCKED** by design |
| Browser image upload | manual | `reports/DEMO_REPORT.md` | **PASS** |
| Browser strict mode | manual | `reports/DEMO_REPORT.md` | **PASS** |
| Browser MP4 upload | manual | `reports/DEMO_REPORT.md` | See report |
| Browser webcam live | manual | `reports/DEMO_REPORT.md` | See report |
| Phone gallery upload | manual | `reports/DEMO_REPORT.md` | **BLOCKED** |
| Phone live camera | manual | `reports/DEMO_REPORT.md` | **BLOCKED** |

## 3. Automated coverage map

| Area | Tests | What is proven |
|---|---|---|
| Geometry | `tests/test_geometry.py` | Homography scale/perspective/rotation, units, area, uncertainty, invalid/degenerate inputs |
| Markers | `tests/test_markers.py` | Generated/perspective detection, wrong ID, missing/tiny/implausible marker, mask/geometry reasons |
| Media | `tests/test_media.py` | JPEG/PNG/WebP, EXIF, alpha, grayscale, clipping, non-finite pixels, byte/pixel limits, corrupt/unsupported, safe filenames |
| Config | `tests/test_config.py` | Null real thresholds, demo thresholds, coriander no-demo, aliases, comparator/metric validation, digests, camera profiles |
| Inference | `tests/test_inference.py` | All required statuses, strict/demo policy, missing marker/scale, unknown crop, neural unavailable, multiple objects, JSON contract, determinism |
| Exports | `tests/test_exports.py` | PNG/JSON/CSV paths, empty/mixed batch CSV, video union schema, traversal safety, no raw masks |
| Video | `tests/test_video.py` | Bounds, H.264 playback, no audio, per-frame counts, cancellation cleanup, rewrite, contact sheet |
| Tracking | `tests/test_tracking.py` | 3-of-5 stability, review clears stability, session counters |
| Dataset | `tests/test_dataset.py` | Provenance rejection, duplicate hashes, leakage detection, deterministic grouped splits, lot integrity, synthetic exclusion |
| Grading | `tests/test_grading.py` | Below/at/above threshold, guard band, invalid policy, no forced class |

Coverage is behavioural, not line-count based. Missing real-device and real-data
paths are listed as blocked rather than inferred from unit tests.

## 4. Physical validation protocol (not yet executed)

For each crop with an approved policy:

1. Assign `specimen_id`, `lot`, and `capture_session` before capture.
2. Measure the policy metric twice with a ruler/caliper; record observer, tool,
   values, and disagreement.
3. Record the measured ArUco outer square and the 100 mm verification line.
4. Capture at least two lighting conditions and two positions/angles per specimen.
5. Freeze the policy and split before examining test outcomes.
6. Run one primary image per specimen through `evaluate_baseline.py`.
7. Report:
   - unique specimens and images;
   - accepted, refused, and ungraded counts;
   - confusion matrix on accepted observations;
   - coverage (accepted / evaluable);
   - measurement MAE/median/max against caliper truth;
   - status reasons and failure gallery;
   - results by crop, size, lighting, angle, and device where support allows.
8. Never tune on the sealed test set. If a test case is opened for debugging, mark
   the round compromised and collect a new sealed set.

Minimum credible pilot: 60–100 measured specimens, with at least 15–20 sealed test
specimens and 5+ per class per evaluated crop. This is a planning threshold, not a
result.

## 5. Acceptance gates

| Gate | Requirement | Current |
|---|---|---|
| 0 Scope | Binary size grading; no unqualified extras | **PASS** |
| 1 Environment | App starts; camera/video dependencies checked | **PASS** (preflight) |
| 2 Data integrity | Provenance/licence/measurement validator; no leakage | **PASS** tooling; real data **BLOCKED** |
| 3 Calibration truth | Known-length tests; wrong/missing marker rejected | **PASS** synthetic; real ruler **BLOCKED** |
| 4 Inference E2E | Valid → grade; invalid → explicit review | **PASS** synthetic + UI |
| 5 Input coverage | Image, laptop webcam, MP4 | Image **PASS**; webcam device **PASS**; UI/video see report |
| 6 Frozen evaluation | Independent real specimens; per-crop metrics | **NOT EVALUATED** |
| 7 Reliability | Critical invalid inputs rejected | **PASS** automated; real stress grid **NOT EVALUATED** |
| 8 Packaging | README, versions, commands, credits, limits | **PASS** |
| 9 Final demo | Rehearsed 3-minute path with fallback | **PARTIAL** — script and local fallback exist; no recorded rehearsal |

Release state: **YELLOW**. Gates 6 and the real portions of 2/3/5/7 are not
satisfied. That blocks any production or accuracy claim.

## 6. Stop conditions

Stop and report `FAIL`/`BLOCKED` if:

- a millimetre value appears without valid calibration;
- a non-`ACCEPTED` result carries a size class;
- synthetic fixtures are counted as real accuracy;
- a threshold is invented or an approval is implied;
- a real test split leaks a lot/session/specimen across splits;
- the app exposes an unauthenticated non-local interface;
- a model is described as trained when no training run and weights exist.
