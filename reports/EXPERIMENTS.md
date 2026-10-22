# Experiments and evidence log

All timestamps are UTC from archived reports. Synthetic results are labelled as such
and are not real-produce accuracy.

## E0 — Environment and media preflight

- **Command:** `python scripts\demo_preflight.py --full-video --camera`
- **Result:** `PASS`
- **Evidence:** `reports/preflight.json`
- **Scope:** dependency imports, config load, demo asset hashes, synthetic contract,
  H.264/libx264 encode, 6 sampled video frames, one laptop webcam frame.
- **Not covered:** live browser streaming, phone camera, long-run stability.

## E1 — Deterministic OpenCV baseline

- **Path:** `opencv_baseline`, no weights, CPU only.
- **Synthetic contract:** 8/8 image checks passed.
- **Synthetic geometry:** median absolute delta 0.179 mm, max 0.268 mm over four
  vector-defined fixtures.
- **Evidence:** `reports/synthetic_evaluation.json`.
- **Interpretation:** plumbing and geometry code work on known synthetic inputs. No
  real-produce accuracy claim is possible.

## E2 — CPU latency

- **Command:** `python scripts\benchmark_latency.py --runs 10 --warmup 2`
- **Protocol:** 80 timed observations, one frame at a time, synthetic fixture,
  decode excluded.
- **Result:** median 116.0 ms, p95 124.3 ms, mean 116.0 ms.
- **Evidence:** `reports/latency_report.json`.
- **Interpretation:** workstation-specific single-frame latency, not throughput or a
  field SLA.

## E3 — Learned segmentation / model comparison

- **Status:** `BLOCKED`.
- **Why:** no licensed measured dataset, no QA'd masks, no approved policy, no
  compatible GPU training environment, and no time for a reproducible training run.
- **Evidence:** `reports/training_blocked.json`.
- **What was not produced:** weights, checkpoints, training curves, validation
  metrics, model card performance claims.
- **What would be required:** measured licensed data, annotation QA, frozen grouped
  split, training run with environment/seed capture, sealed evaluation, baseline
  comparison, and a model card update.

## E4 — Real measured baseline evaluation

- **Status:** `BLOCKED`.
- **Why:** the real manifest is empty; no physically measured specimens or approved
  thresholds exist.
- **Evidence:** `reports/baseline_evaluation.json`.
- **What was not produced:** confusion matrix, balanced accuracy, MAE, coverage,
  per-crop slices, confidence intervals.

## E5 — Browser and device smoke tests

- **Image upload/demo policy:** `PASS` in a desktop browser session; potato fixture
  produced `ACCEPTED`/`BIG`, 72.3 ± 2.6 mm, with PNG/JSON/CSV downloads.
- **Strict mode:** `POLICY_REQUIRED` with `NO_APPROVED_THRESHOLD` observed after
  unchecking demo mode.
- **Laptop camera:** one-frame OpenCV capture `PASS` in preflight; live browser
  streaming **NOT EVALUATED** after the browser harness became unavailable.
- **MP4 UI:** programmatic H.264/video tests `PASS`; browser upload **NOT EVALUATED**.
- **Phone:** gallery upload and live HTTPS camera **BLOCKED**.
- **Evidence:** `reports/DEMO_REPORT.md`.

## E6 — Dataset gates

- Empty template validator: **PASS as an empty template only**; it is not real data.
- Empty template grouped split: correctly **rejected** (no groups).
- Real manifest: **BLOCKED** — absent.
- Evidence: `reports/dataset_validation.json`, `reports/group_split_report.json`
  (the latter may be absent if the split script was not re-run; the rejection was
  observed).

## E7 — Failure gallery

- Four synthetic refusal cases rendered; command exit 0; all expected states matched.
- Evidence: `reports/failure_gallery/index.md` and PNG panels.
- Real failure gallery: **BLOCKED**.

## E8 — Final capture attempt

- The foreground shell harness stopped returning output after the app server restart.
  As a result, the final post-doc automated run, JUnit XML, and repository checksum
  file could **not** be regenerated in the final minutes.
- The final code revision (including `pipeline_sha256`) did pass **164 tests in
  10.20 s** and Ruff before these documentation-only edits.
- This is recorded as a tooling limitation, not a passing final capture.

## Summary table

| Experiment | Status | Evidence class |
|---|---|---|
| E0 Preflight | **PASS** | Environment/media |
| E1 Synthetic baseline contract | **PASS** | Synthetic only |
| E2 CPU latency | **MEASURED** | Synthetic, workstation |
| E3 Model training/comparison | **BLOCKED** | No data/model |
| E4 Real baseline evaluation | **BLOCKED** | No measured specimens |
| E5 Browser/device smoke | **PARTIAL** | Browser image/strict; device capture |
| E6 Dataset gates | **PASS** tooling / **BLOCKED** real | Template only |
| E7 Failure gallery | **PASS** synthetic / **BLOCKED** real | Synthetic only |
| E8 Final evidence capture | **BLOCKED** | Shell harness failure |
