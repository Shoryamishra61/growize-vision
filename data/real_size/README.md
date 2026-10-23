# Original real-photo size validation

The manifest is intentionally empty. There are no verified measured station training/test
specimens in this manifest. A separate published RGB-D broccoli benchmark is
reserved for external validation and is not a training reference. Do not copy generated marker composites from
`data/calibrated_real_samples`, demo images, publisher augmentations, or generated
training views into this dataset.

Each JSONL row must contain:
- `path`, `sha256`: unchanged original photograph under the dataset root and file hash.
- `specimen_id`, `capture_id`, `split`: physical item identity, capture identity,
  and frozen train/validation/test split. All captures of one specimen stay together.
- `is_original_capture: true`, `synthetic: false`, `augmented: false`.
- `source`, `license`, `reviewed_by`: acquisition rights and actual provenance reviewer.
- `crop_key`, `physical_unit`, `metric`: the policy and matching physical dimension being measured.
- `measured_value`, `measurement_unit` (mm or cm2), `measurement_reviewed_by`: independent ruler/caliper measurement
  and actual reviewer; never an image-derived number or a filename-based label.
- `expected_size` is optional: if present, it must agree with the frozen training-specimen median.
- `marker_side_mm`, `marker_id`: measured printed marker side and visible marker ID.

Keep the entire separated item and marker visible on the same flat surface.
Leaves require an agreed unit (one leaf or a bunch) and metric; these are not
interchangeable with pod length. Record actual recaptures for lighting, camera
angle and distance; do not manufacture transformed evaluation images.

The working rule is below the same-kind training median = SMALL, at or above = BIG.
The median of each resulting class is reported as a descriptive size anchor;
these two anchors do not define image scale or replace the median boundary.
Equal measurements belong to BIG, so tied values can prevent an exact 50/50 split.
The existing configured metric and item unit must match the measurements.
Build references from training specimens, then evaluate the independent test specimens:

    python scripts/build_size_reference.py --manifest data/real_size/manifest.jsonl
    python scripts/validate_real_size.py --manifest data/real_size/manifest.jsonl

The runner checks provenance and split identity, refuses missing measured reference averages, and
reports refusals, coverage, confusion and measurement error. This evaluates
metrology using recorded crop policies, not automatic species recognition.
A finite zero-error test does not prove universal 100% accuracy.
