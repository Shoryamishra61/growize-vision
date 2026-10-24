# Growize Vision — Vegetable Identification & Size Grading Showcase

Every image below is a **real application output**: the photograph was run
through the deployed app pipeline (auto-detected crop, no scale, no manual
inputs) and the picture shown is exactly what the app displays, with each
detected item outlined and labelled. Orange = SMALL, amber = MEDIUM,
green = BIG.

**How to read a result.** The title names the vegetable and the size mix
(`POTATO · 2 BIG`); each item carries its grade on the image
(`SMALL (1)`, `MEDIUM (2)`, `BIG (3)`); the message states the measured size
in millimetres (estimated scale) or, when nothing measurable was possible,
that sizes are relative to the batch.

---

## Single vegetables — identified and graded with zero inputs

### Okra (lady's finger)
![Okra single](assets/okra_single.png)
**Result:** `Okra — BIG (3)` — Okra: 330 ± 99 mm → BIG.
Graded against okra's published reference cut (120 mm; derived bands
96 / 144 mm) using estimated scale.

### Potato
![Potato single](assets/potato_single.png)
**Result:** `Potato — BIG (3)` — Potato: 172 ± 52 mm → BIG.
Potato trade bands: < 40 mm small, 40–60 mm medium, > 60 mm large.

### Tomato
![Tomato single](assets/tomato_single.png)
**Result:** `Tomato — BIG (3)` — measured against tomato's published bands.

### Green / French bean
![Bean single](assets/bean_single.png)
**Result:** `Green / French bean — BIG (3)` — 262 ± 79 mm → BIG.

### Green chilli
![Chili single](assets/chili_single.png)
**Result:** `Green chilli — BIG (3)` — 204 ± 61 mm → BIG.

### Cucumber (uncalibrated estimate)
![Cucumber single](assets/cucumber_single.png)
**Result:** `Estimated BIG (uncalibrated)` — the cucumber's footprint is
1.48× the typical cucumber in the real-photo reference set. When no scale
exists at all, crops with real-photo references still give an
appearance-based estimate, clearly labelled as uncalibrated.

---

## Batches and bunches — every item graded

### Okra overlap (two crossing pods)
![Okra overlap](assets/okra_overlap.png)
**Result:** `Okra · 1 MEDIUM · 1 BIG` — overlapping produce separated into
two graded items.

### Potato batch
![Potato batch](assets/potato_batch.png)
**Result:** `Potato · 2 BIG` — two genuinely big potatoes both grade BIG;
the absolute band comparison never forces a small one into the photo.

### Leafy bunches (spinach, coriander)
Leafy greens arrive as bunches, so the **bunch is the grading unit** and is
never split. Put one bunch in the photo for an estimated grade, or several
bunches in one frame for a sharp relative split (1 = smallest bunch,
3 = biggest) with no scale needed.

---

## Identification accuracy (species model v6)

Retrained on 28,611 cleaned photos (21k Kaggle + 12.7k field photos from
the j33g3nsm9k_v1 dataset, cross-label duplicates excluded; 1,909
byte-identical files that the public dataset ships in both its Okra and
Tomato folders were removed before training).

**Frozen test split (4,108 photos, never seen in training):**

| Vegetable | Test accuracy | | Vegetable | Test accuracy |
|---|---|---|---|---|
| Bean | 100% (534/536) | | Cabbage | 99% (160/161) |
| Bitter gourd | 99% | | Capsicum | 100% |
| Bottle gourd | 100% | | Carrot | 100% |
| Brinjal | 100% | | Cauliflower | 100% |
| Broccoli | 100% | | Chili | 99% (393/398) |
| **Cucumber** | **99%** | | **Okra** | **92% (424/462)** |
| Onion | 100% | | Papaya | 100% |
| Potato | 86% (588/684) | | Pumpkin | 99% |
| Radish | 100% | | Spinach | 86% |
| Tomato | 99% (195/197) | | | |

**Overall: 96.4% accuracy, 0.98 macro-F1.**

**Live head-to-head vs the previous model (30 fresh field photos per
class, never used in any training):**

| Class | Previous model | v6 |
|---|---|---|
| Okra | 3/30 | **27/30** |
| Chili | 0/30 (class did not exist) | **29/30** |
| Bean | 29/30 | 30/30 |
| Cucumber | 30/30 | 30/30 |
| Potato | 29/30 | 28/30 |
| Tomato | 28/30 | 26/30 |

## Grading correctness checks (automated tests, 326 green)

- Two genuinely big potatoes (measured > 60 mm each) both grade BIG —
  a relative split that would force one SMALL is wrong, and the test
  forbids it.
- Three nearly identical potatoes all receive the same grade with an
  explicit UNIFORM_BATCH note.
- Edge-clipped, soft-focus and busy-background photos grade with named
  warnings instead of refusing (estimate flows).
- Merged piles of single-unit produce are split into individual items;
  leafy bunches are never split.
- Strict marker/station metrology keeps every quality gate and the
  interval-in-band discipline.

## Honest limitations

- Millimetre values without any scale come from an assumed camera distance
  (±30%): grades are right when the photo is taken from roughly a
  forearm's distance; borderline items say so instead of guessing.
- Pod counts in a tightly pressed pile depend on visible boundaries —
  spread the batch loosely for exact counts.
- Identification leans on the crop's photo domain; the dataset's own
  "tomato" folder contained only 16 real tomato photos, so tomato
  recognition rests on the Kaggle domain.
- The strict-refusal discipline (published thresholds, provenance-gated
  rules) is untouched: nothing in this showcase invents a standard.
