# Negative results

## LSID_REAL_SIZE_001 — not promoted

- Source: LSID-Beans v3, 612 unchanged real captures/612 measured leaves/306 plants.
- Fit: 406 leaves; validation: 84 leaves; canonical test remains unevaluated.
- Pixel random forest: 30 trees, max depth14, minimum leaf10, seed42; no augmentation.
- Training-specimen average: 25.451243842364534 cm2.
- Validation automatic sizing: 53/84 accepted, 42 correct, 11 wrong; accepted accuracy79.2453%.
- Human-mask geometry diagnostic: 53/84 accepted, 44 correct, 9 wrong; not model accuracy.
- Engineering interval covered independent area in only50% of the 74 oracle measurements.
- Cause split: tiny background/other-leaf components selected after a larger ambiguous
  edge component was discarded; marker-plane projection also overestimates some field leaves.
- Validation-only comparison: scalar marker-area scaling MAE2.5908cm2 vs homography4.2903cm2
  on the same74 human masks. This is diagnostic, not a promoted measurement method.
- Next experiments: fix dominant-component refusal; separately test measurement-model
  robustness with real validation residuals. Do not tune on canonical test or use oracle
  masks as automatic input. No change to the active UI's model-release refusal.

Receipts: reports/real_size/lsid_beans/pixel_rf_v1/validation_summary.json and
reports/real_size/lsid_beans/validation_geometry_diagnostic.json.

## LSID_REAL_SIZE_002 — not promoted

Only the dominant-component selection changed. A larger ambiguous leaf is no longer
replaced by a tiny unrelated complete fragment. Pixel features, training data, model
parameters, reference average and metrology are unchanged.

Validation84 captures:44 accepted,38 correct,6 wrong (all SMALL->BIG).
The five wrong BIG->SMALL decisions from V1 disappeared; four previously correct
SMALL outputs also became refusals. Overall correct fraction fell from50% to45.24%
as coverage decreased. This is a justified ambiguity refusal, not a broad accuracy win.
Physical projection/pose errors remain. Canonical test still not used; app model unchanged.

## LSID_REAL_SIZE_003 — frozen test failed; not promoted

Only the area measurement method changed from V2: marker-area weak-perspective
scalar estimation, labelled as approximate. Segmentation, reference average and
training data were unchanged. Validation:37 accepted/84,35 correct,2 wrong.
The scalar method reduced area MAE on the same64 measurable validation captures
from4.0721cm2 to2.4908cm2. Bounds still covered truth in only60.94% of those captures.

The frozen candidate was evaluated once on122 test leaves. After excluding the
known development pilot's plant,120 clean captures:67 accepted,62 correct,5 wrong,
55.83% coverage,92.54% accepted accuracy. Test area MAE2.5821cm2 on98 measurable
captures; engineering bound truth coverage61.22%. No promotion or test-set tuning.

Padding bounds to cover all observed validation errors left only2/84 labels;
that apparent zero-error result was rejected as a complete solution. The unresolved
release dependency is independent, measured representative captures from the actual
setup and other requested vegetable/leaf types. No further source-test adaptation.

## LSID_REAL_SIZE_004_MEDIAN — user-defined target rerun; not promoted

The user explicitly requested a median-based size definition and industrial research.
The reference changes from training mean25.4512cm2 to training median24.933cm2.
Class medians21.733/28.2cm2 are descriptive anchors. All406 original training captures,
RF parameters, segmentation and scalar geometry are unchanged; no synthetic views.

Validation:37/84 labels,34 correct,3 wrong. Source-test rerun is a post-test diagnostic
because V3 outcomes were already observed. Excluding pilot plant1:67/120 labels,
60 correct,7 wrong. Measurements and segmentation IoU exactly match V3; MAE is
unchanged. Different target labels mean score changes are not evidence of a better
or worse dimensional estimator. No test-informed tuning or promotion followed.

Industry research supports calibrated controlled capture, physical dimension choice
and independent metrology. A potato study's reported100% on a controlled experiment
is not our result. Its data is request-only; 3DPotatoTwin file access returned401;
the open MSU sweetpotato source lacks individual image-to-root measurement mapping.
These were not relabelled as verified size training/evaluation data.
