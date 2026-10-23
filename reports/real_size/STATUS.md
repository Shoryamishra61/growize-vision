# Current size-grading evidence

**Not released. The requested reliable vegetable/leaf size model is incomplete.**

Working definition: SMALL below the measured training-specimen median of the same
kind of item; BIG at or above it, with no label when uncertainty crosses the boundary.
One leaf and a bunch remain different units.

## Latest median-based rerun and factory research

The factory methods, thesis/papers, capture geometry limitations and dataset access
audit are in [INDUSTRIAL_SIZING_RESEARCH.md](INDUSTRIAL_SIZING_RESEARCH.md).
Factory sizing establishes physical dimensions with mechanical or calibrated sensor
measurements; population averages do not establish the scale of an arbitrary image.

V4 was freshly trained on the same406 original real captures, without augmentation.
Only the reference statistic changed. The training median is24.933cm2; the medians
within its SMALL and BIG groups are21.733cm2 and28.2cm2, respectively. These are
bean-leaf reference values, not potato dimensions or all-vegetable averages.

Validation84 captures:37 accepted,34 correct,3 wrong;91.89% accepted accuracy and
44.05% coverage. The post-test diagnostic excludes the known pilot plant and has
120 captures:67 accepted,60 correct,7 wrong;89.55% accepted accuracy and55.83%
coverage. **This test fold was already observed during V3 and is not a fresh
independent test for V4.**

Source images, segmentation IoU and every physical measurement record match V3
exactly on validation and test. Area MAE remains2.4908cm2 on64 validation measurements
and2.5821cm2 on98 diagnostic-test measurements. A changed median boundary does not
correct measurement errors; mean and median scores also use different true labels.

71 focused numerical/provenance/real-photo tests passed, with5 dependency deprecation
warnings. Ruff lint and formatting checks passed on the changed code. No synthetic
image suite was run. No model was promoted and no interface changes were made in this
research follow-up.

Receipts: `lsid_beans/pixel_rf_v4_median/experiment_plan.json`, `validation_summary.json`,
`test_summary.json`, source/checkpoint hashes and logs. The latest diagnostic still
fails the requested reliability and cross-vegetable coverage.

## Earlier mean-based real-only experiments

LSID-Beans v3:612 original physical captures of612 measured leaves from306 plants.
The source authors cropped/resized the camera images to512px. No generated views,
composites or augmentation were used in these experiments. Source hashes and annotations
were checked, and every selected image/mask decoded. Source:
https://data.mendeley.com/datasets/f42hwwrpgn/3 .

406 training leaves,84 validation leaves,122 canonical test leaves; related leaves from
one plant stay together. The reference average is25.451243842364534cm2, from training
specimens only. Three candidates were evaluated on validation before the best was frozen.

| Candidate | Validation accepted | Wrong accepted | Accepted accuracy |
|---|---:|---:|---:|
| V1, largest complete component/homography |53/84|11|79.25%|
| V2, dominant-component refusal/homography |44/84|6|86.36%|
| V3, same segmentation/scalar area estimate |37/84|2|94.59%|

The frozen V3 was evaluated once on the canonical test. Removing the previously
inspected development pilot's plant leaves120 clean held-out captures:
**67 accepted,62 correct,5 wrong;92.54% accepted accuracy,55.83% coverage**.
Mean absolute area error was2.5821cm2 on98 measurable captures. The engineering bounds
covered independent truth in only61.22% of those captures. These are failures of the
requested reliability, not a release qualification or a universal vegetable score.

Human-mask diagnostics also fail some size decisions, so segmentation alone cannot
solve marker/leaf pose and depth mismatch. The scalar method is explicitly labelled a
weak-perspective estimate. Padding the uncertainty to cover all observed validation
errors would leave only2/84 labels; this was rejected as a complete solution.

No candidate is loaded by the interface. Historical augmented species checkpoints,
generated marker composites, and synthetic demos are excluded from the current
training/evaluation/release path. The active UI remains capture inspection/refusal mode.

## Other evidence and remaining dependencies

The separate published broccoli source has20 original RGB/depth sets and94 caliper-linked
diameter annotations. It was acquired/checksummed/decoded, not trained or scored.
Its depth-sensor capture is not the current ordinary RGB station.

The intended station manifest `data/real_size/manifest.jsonl` remains empty. There are
no independent measured examples from the user's actual setup spanning the requested
vegetable/leaf types. Original photos must show a trusted scale and have independently
recorded measurements/specimen identities, with distinct reference and held-out items.
There is also no verified real produce video or live-camera physical trial.

Focused checks:62 passed. These cover provenance, numerical policy, source parsing,
real marker detection, real-image feature extraction, scalar geometry units, selection
regression, and UI empty states. They do not establish model accuracy. The historical
synthetic image suite was not rerun under the no-synthetic-images instruction.

The minimal local UI is at http://127.0.0.1:7861/ . It contains image upload/paste,
camera/video inputs and output only, and explicitly says grading is not released.

Detailed receipts: `lsid_beans/current_state.json`, `lsid_beans/source_integrity.json`,
`lsid_beans/pixel_rf_v3/test_summary.json`, `lsid_beans/pixel_rf_v3/test_predictions.jsonl`,
`lsid_beans/pixel_rf_v3/uncertainty_feasibility.json`, and `ui_checks.json`.
