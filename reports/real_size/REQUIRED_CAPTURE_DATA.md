# Data needed to resume physical-size qualification

The current real-only experiment is not a reliable size-model release. Five of67
accepted held-out leaf decisions were wrong. It cannot validate arbitrary vegetable
types or the user's intended camera setup.

Supply original photographs or real camera/video captures from the intended setup,
with paired independent measurements. Do not paste a marker into an existing photo,
generate images, or create augmented test views.

For each physical item, preserve:

- item ID and type; a single leaf and a bunch are different units;
- original file and capture/camera identity, with repeat captures linked to the same item;
- a measured scale reference visible alongside it, and capture geometry that supports
  the proposed measurement (especially marker/object depth and plane alignment);
- independent ruler/caliper dimensions for that item, or a separately measured leaf
  area; measurement unit, method, reviewer and relevant pose;
- reference/training and held-out test items that are physically distinct, including
  the intended range of small and big specimens;
- representative poor, ambiguous and non-produce inputs from the same setup, including
  real cases with the reference present, so refusal behavior can be tested honestly.

The working size rule remains the average of the same kind of item. Reference averages
must come from measured reference specimens, and never from held-out test items.
The schema and size validation commands are in `data/real_size/README.md`.

With those data, the next work is source/identity verification, frozen splits,
measurement calibration, per-type sizing validation, and real camera/video replay.
A finite zero-error test can support a bounded operating claim, not universal100% accuracy.
