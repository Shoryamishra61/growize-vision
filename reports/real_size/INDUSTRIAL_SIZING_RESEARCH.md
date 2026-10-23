# Vegetable sizing: industrial evidence and revised experiment

Research checked 2026-10-23. This review uses manufacturer documentation, original
research and an institutional thesis. No synthetic images were acquired or used.

## What factories actually measure

| Method | Physical information | Consequence for this project |
|---|---|---|
| Adjustable rollers/screens | Whether produce passes a known opening; width/diameter bands | No vegetable recognition network is needed to establish scale. Crop and commercial grade bands are configured. |
| Controlled optical station | Silhouette/dimensions under a fixed, calibrated camera arrangement; multiple views when required | A calibrated capture station is the closest low-cost software equivalent. |
| Weight grading | An electronic weighing cell measures each item | Image-derived weight is an estimate, not a weighing-cell reading. |
| Colour/multispectral inspection | Surface or internal defect signals from suitable sensors and views | Defect quality and physical size are separate tasks. |
| RGB-D or stereo | Depth plus camera calibration establishes metric geometry | Ordinary uploaded RGB photographs do not contain this sensor information. |

[Tong's lift roller documentation](https://tongengineering.com/us/product/lift-roller-graders/)
describes width/diameter sizing with adjustable upper rollers and configurable grade
bands. These bands depend on crop and contract; they are not population medians.
[Tong/MAF optical documentation](https://tongengineering.com/product/maf-potato-onion-optical-sorting/)
describes LED illumination, HD/multispectral cameras, rotating produce for multiple
views, dimensional analysis, and separate electronic weighing cells. Manufacturer
accuracy language is not an independently reproduced benchmark for this application.

There is a documented progression rather than evidence for one single invention:
a [1996 potato inspection station](https://pure.psu.edu/en/publications/an-automated-inspection-station-for-machine-vision-grading-of-pot/)
already combined a chamber, camera, conveyor and sorter. Moving-potato results were
worse than stationary results, illustrating the importance of capture conditions.
The [2014 Radboud mobile-grading thesis](https://theses.ubn.ru.nl/items/883df431-5378-4111-8c48-54fe84526ec3)
investigated width, height and square-mesh size with 2D/3D vision; its methods include
a marker board, camera calibration and multi-view reconstruction. Its full PDF was
accessible intermittently; its indexed methods were checked, not independently reproduced.

A [2025 multiangle potato study](https://doi.org/10.1016/j.atech.2025.100832)
used physical measurements from 23 cultivars and controlled top/side photographs.
It reports 100% classification in its experiment using top and side widths, while
also calling for further cultivar-specific validation. That finite experiment is
evidence for the approach under its conditions, not proof of universal accuracy.
Its data-availability statement requires requesting the data; no request was sent.

## Correct interpretation of the requested median rule

For each crop, physical item unit and metric, collect independent measurements of
training specimens. Freeze their median as the boundary. SMALL is below it; BIG
is at or above it. Compute the median within each resulting class as a typical
small/big example. Ties belong to BIG, so the counts need not be exactly equal.
Do not use the midpoint between those two class medians as an unannounced substitute
boundary. A class label describes relative size within that measured sample, not
an official commercial grade or a universal biological average.

This definition fits the user's latest instruction. A mean and a median can differ
substantially for skewed distributions. Reference values from test items must not
enter either statistic. Multiple photographs of one item count as one specimen.
Pod length, potato width, individual leaf area and bunch footprint remain different
quantities. A single threshold pooled across these would have no physical meaning.

## Where the earlier approach goes wrong

1. **Population size does not establish photographic scale.** In the pinhole model,
   image length is proportional to focal length times physical length divided by
   camera distance. A crop median supplies a prior, not the unknown camera distance.
   Enlarging an image does not reveal the object's physical dimensions.
2. **A planar marker corrects its own plane.** A raised potato or tilted/wavy leaf
   can occupy another plane. For a simple overhead camera 400 mm above a reference
   plane, an object contour 50 mm nearer the camera is magnified by 400/350, about
   14.3%. This is an illustrative geometry calculation, not a measured station result.
3. **Projected area differs from flattened leaf area.** A perfect segmentation mask
   cannot recover unobserved tilt, folding or hidden surface. Existing human-mask
   diagnostics also contain grading errors.
4. **Recognition accuracy is not size accuracy.** Correctly naming okra does not
   reveal its length. Defect labels such as good/bad or maturity are not size labels.
5. **Refusing almost everything is not success.** Report correct/wrong decisions,
   coverage and dimensional error together. An uncertainty bound must be validated;
   the current engineering envelopes are not calibrated confidence intervals.

## Minimum justified pipeline

Original capture → produce instance/identity → metric scale from a calibrated fixed
station, a suitable coplanar reference, or calibrated depth → dimension → comparison
with the frozen same-type training median → SMALL/BIG, with review for unresolved
measurement or identity. Keep the user's input/output interface simple; capture
calibration and reference distributions belong inside the station setup.

For loose leaves, flattening on a calibrated plane directly addresses a known error.
For thick produce, use calibrated height/depth or validate a fixed camera's height
error over the full working range. Do not silently deploy a plane-only estimate as
a validated 3D measurement. These capture changes require physical validation and
have not been implemented as imaginary camera/sensor capabilities.

Falsification checks are actual recaptures of the same measured specimen at different
distances, positions and orientations; unseen physical specimens; multiple cultivars;
and real negatives/overlap. No digitally transformed evaluation views are needed.

## Dataset audit and executed rerun

- [3DPotatoTwin](https://huggingface.co/datasets/UTokyo-FieldPhenomics-Lab/3DPotatoTwin)
  pairs actual RGB-D capture with physical ground measurements. Public metadata was
  inspected at revision `e208723b6b7d62bf02da1722cb05538fe4996486`. It is gated;
  an unauthenticated file request returned HTTP 401. No access terms were accepted,
  files downloaded, or account/contact information shared.
- [MSU sweetpotato dataset](https://zenodo.org/records/18100484), CC BY 4.0,
  provides real conveyor images/video. Its own usage notes say individual image
  instances are not mapped to specific physical roots. Population measurements do
  not provide individual size truth. The 6.8 GB archive was not downloaded or used
  as a size benchmark. Splitting adjacent video frames would also leak specimens.
- Existing [LSID-Beans v3](https://data.mendeley.com/datasets/f42hwwrpgn/3) provides
  real leaf photographs and independent leaf measurements. V4 was freshly fitted on
  the same 406 original training captures, without augmentation. Only the reference
  statistic changed from mean to median; segmentation and geometry stayed fixed.
  Source-author cropping/resizing remains disclosed. This is bean-leaf sizing with
  recorded identity, not a potato model or automatic all-vegetable recognition.

Plan and receipts: `lsid_beans/pixel_rf_v4_median/`. Its validation score and test
diagnostic are reported in `STATUS.md`. The source test was already observed during
V3, so the new test rerun is explicitly a post-test diagnostic. Switching target
definitions changes true SMALL/BIG labels; scores under the two definitions do not
by themselves establish an improvement in measurement quality.

No candidate has been promoted into the interface. The missing qualification is
representative real measured capture data across the intended vegetables and actual
capture setup, rather than another arbitrary cutoff or a larger recognition model.
