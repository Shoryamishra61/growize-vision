# Calibration and measurement

## 1. Principle

No millimetre value is produced without a validated scale. The scale comes from a
printed ArUco marker whose **outer black square** is physically measured by the
operator with a ruler or caliper. The nominal PDF size is never trusted.

The marker and the vegetable must lie on the same flat matte plane. The system
computes a planar homography from the four marker corners to a metric square, then
projects the segmented silhouette through that homography. The result is a
**projected-plane silhouette measurement**, not a 3-D surface measurement.

## 2. Printable reference

- Asset: `assets/aruco_marker_0_50mm.png`
- Dictionary: `DICT_4X4_50`
- Expected ID: `0`
- Nominal outer black square: 50 mm
- Footer: a 100 mm verification line with 0/50/100 mm ticks

Print at **100% / actual size** with no "fit to page" scaling. Measure both:

1. the outer black square side, and
2. the 100 mm verification line.

Use the measured outer square in the app. If the two checks disagree materially,
the printer scaled the page; reprint rather than compensating in software.

Regenerate if needed:

```powershell
python scripts\create_marker.py --id 0 --side-mm 50 --dpi 300
```

## 3. Capture SOP

1. Use a plain matte tray with no printed pattern, glare, or strong colour cast.
2. Place the marker flat, fully visible, and at least one marker-width from the
   vegetable. Do not overlap the object and marker.
3. Place one vegetable flat on the same plane. Keep the whole silhouette inside the
   frame, including leaves/stems.
4. Use an overhead or modestly tilted view. Avoid a tall object hiding its own base
   or casting a hard shadow across the marker.
5. Focus and expose so the marker and object edges are sharp. Avoid clipping.
6. Enter the **measured** marker side in millimetres and the measurement
   uncertainty from the ruler/caliper.
7. Inspect the overlay: marker outline, object mask, measured axis, value,
   uncertainty, cutoff, and status.
8. If the status is not `ACCEPTED`, follow the reason; do not override it.

## 4. Validation rules

Calibration is rejected when:

- no marker is found;
- the marker ID is not the expected ID;
- the marker outline is not a plausible quadrilateral;
- the marker is too small in pixels or too small a fraction of the frame;
- opposite/adjacent side lengths are implausible for a square;
- the supplied physical side length is missing, non-finite, or non-positive;
- homography degenerates or reprojection error is unacceptable.

Rejections produce `CALIBRATION_REQUIRED` with reasons. They never produce a
millimetre value or a size class.

## 5. Metrics

| Metric | Definition | Unit |
|---|---|---|
| `min_area_rect_short_mm` | Shorter side of the minimum-area rectangle around the projected contour | mm |
| `min_area_rect_long_mm` | Longer side of that rectangle; straight axis, not a curved centreline | mm |
| `max_feret_mm` | Maximum caliper distance between projected contour points | mm |
| `centerline_length_mm` | Curved centreline: Zhang–Suen skeleton path with terminal-radius compensation | mm |
| `projected_area_cm2` | Projected contour area on the marker plane | cm² |

Mask area in pixels is retained separately and is not a physical grade.
`curvature_ratio` warns when a straight-axis metric may be biased for bent pods or
leaves.

The curved-centreline metric rasterises the calibrated silhouette, thins it to a
medial axis, breaks thinning loops at inner bends, prunes skeleton spurs shorter
than `calibration.centerline_prune_mm`, and sums the leaf-to-leaf path in
millimetres, adding both terminal silhouette radii. Skeletons that remain
branched or cyclic are refused (`AMBIGUOUS_SKELETON_*`) and surface as
`REVIEW_BORDERLINE`; no straight-axis fallback is substituted. An elongated
silhouette also carries a `volume_proxy_cm3` block (frustum integration along the
projected centreline). It is an explicitly-labelled slender-axisymmetric proxy
per Huynh et al. (2020), not a validated 3-D volume, not a mass, and never a
grading input.

## 6. Uncertainty envelope

The reported interval is an **engineering error bound**, not a statistical 95%
confidence interval. It combines:

- marker measurement uncertainty supplied by the operator;
- corner localisation uncertainty in pixels;
- mask boundary localisation uncertainty in pixels;
- a configured parallax bound for object height above the plane.

The planar scale error propagates with distance from the marker. For area, the
linear envelope is converted into a cm² envelope and the result explicitly says so.
The interval is used conservatively: if it touches the threshold, the result is
`REVIEW_BORDERLINE`.

## 7. Grading rule

```text
m - u >= T   -> BIG
m + u <  T   -> SMALL
otherwise    -> REVIEW_BORDERLINE
```

`m` is the measured metric, `u` the uncertainty, `T` the crop threshold. A configured
guard band widens the review zone. Real thresholds are `null` until ARL approves
them; demo thresholds are labelled `ASSESSMENT_DEMO_ONLY`.


### Threshold provenance

Strict-mode thresholds and bands are PUBLISHED_REFERENCE values compiled from
citable standards, trade practice, and research variety data — see
`docs/SIZE_REFERENCE_SOURCES.md` for the full source table. The config loader
rejects any `threshold_mm` without a `threshold_provenance` block, and every
accepted decision carries `PUBLISHED_REFERENCE_NOT_POPULATION_VALIDATED` plus
a warning, because literature values are not local-population measurements.
Measured specimen medians (`EMPIRICAL_REFERENCE`) and ARL approvals
(`APPROVED_SOURCE`) override them whenever they exist.

### Multi-band grades

A crop may define ordered size bands (`size_bands`, or `demo_size_bands` for the
explicitly-labelled demonstration policy) instead of — or alongside — a single
binary cut-off. Band *i* covers `[min_mm_i, min_mm_{i+1})`; the first band starts
at 0 and the last is unbounded. The measurement interval must sit entirely
inside one band after the guard band is applied to every edge, otherwise the
result is `REVIEW_BORDERLINE` with `UNCERTAINTY_OR_GUARD_BAND_CROSSES_BAND_EDGE`.
The first band maps to SMALL and the last to BIG so a two-band table reproduces
the binary rule exactly; middle bands (for example `medium`) are reported as
`size_band` with no binary size class. Band tables are validated at config load
(unique names, ascending edges, first edge at 0) and follow the same approval
doctrine as thresholds: shipped demo bands are not AGMARK/ICAR size codes.

### Segmentation cross-check

When the background-difference method is primary (marker visible, matte mat),
an independent chromatic-energy Otsu mask is computed and compared by IoU
(`segmentation.ensemble_min_iou`, default 0.80). Below the threshold every
instance is flagged `SEGMENTATION_DISAGREEMENT` and refused for review: one of
the two methods was fooled by lighting or background, and neither is trusted
silently. Setting the knob to 0 disables the cross-check.

## 8. What calibration cannot do

- Recover height, volume, weight, density, or hidden extent.
- Correct a warped lens without intrinsics calibration.
- Make a cluttered background valid.
- Make a moving/blurred object measurable.
- Turn an unmeasured print into a traceable ruler.
- Validate a different camera without a new physical check.

Real-device accuracy remains **NOT EVALUATED** because no measured specimens were
available. The synthetic geometry check (median 0.179 mm, max 0.268 mm over four
vector-defined fixtures) proves the transform and contour code path, not real-produce
accuracy.
