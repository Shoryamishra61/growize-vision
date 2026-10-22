Growize Vision

Research backed product requirements, software specification and validation plan for ARL farm and machine vision

Prepared for the Vision and Autonomous Systems Associate assessment at AgResearch Labs and Growize Farms. Prepared by Shoryakumar Mishra. Version 1.0. Research cut-off 22 October 2026.

Decision: build a calibrated, species-aware instance-segmentation and measurement system. The required output is SMALL or BIG for each eligible vegetable. Unknown inputs, missing scale, borderline dimensions and unsafe frames must produce a review state instead of a confident but unsupported answer. Auxiliary modules for quality, nutrition and crop conditions must never delay or falsify the primary grading decision.

Document status: proposed implementation and test contract. None of the accuracy, latency or readiness targets below is a measured achievement; they require real labelled ARL images, physical ground truth and hardware trials. Exact grade thresholds and output action rules need ARL sign-off.

# 1 Executive product definition

## 1.1 Problem statement

ARL needs an operator-facing, upload-capable and machine-compatible vision system that inspects harvested vegetables, recognises supported crop classes, estimates the applicable physical dimension when measurement is valid, and grades each item as SMALL or BIG against an approved threshold for that crop and market. The system must remain observable, recover safely from missing or ambiguous inputs, and improve through labelled field feedback.

## 1.2 Research conclusions

- Sizing studies emphasise calibrated scale, object distance, pose and occlusion. Pixels alone do not define centimetres \[1, 2, 3\].
- A multi-view RGB-D sweetpotato conveyor prototype explicitly combined segmentation, tracking, dimensions and surface grading; a reported model-specific grading accuracy was 88.1% for its largest model, not universal field accuracy \[3\].
- PlantVillage illustrates why attractive controlled-background validation can misrepresent farm performance; a 2026 cross-domain benchmark quantifies this reliability gap \[4, 5\].
- Fast instance-segmentation candidates exist, but crop-specific benchmarking, postprocessing and licensing matter more than headline benchmark scores \[6, 7\].
- Soil properties and weather can be attached from location-aware sources, but mapped or modelled estimates are not measured farm-lab values \[8, 9\].

## 1.3 Engineering truths

| Question                                          | Answer used in this specification                                                                                                        |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Can one normal photo reveal exact size?           | No. Exact metric sizing needs a known reference in the object plane, a validated fixed rig, stereo or depth and calibration.             |
| Can every plant be forced into two labels?        | No. Add REVIEW, UNKNOWN and MEASUREMENT_UNAVAILABLE internally; only valid measured items enter two physical size bins.                  |
| Are small and big globally defined?               | No. Metric, cultivar, harvest stage, buyer requirement and crop-specific threshold govern labels.                                        |
| Can a harvested leaf reveal soil pH or NPK?       | No. Lab tests or farm sensors are authoritative; geographic soil maps are approximate contextual estimates.                              |
| Can one image show total yield or hidden bruises? | Not reliably. Yield requires plot area, time, crop counts, measured mass and harvest records; internal defects may need non-RGB sensing. |

# 2 Product requirements document

## 2.1 Stakeholders and users

| User                          | Task                                                                     | Primary success signal                     |
| ----------------------------- | ------------------------------------------------------------------------ | ------------------------------------------ |
| Farm or packhouse operator    | Capture one item or a moving batch; see final grade and recapture alerts | Consistent actions with few rechecks       |
| R&D vision engineer           | Inspect masks, dimensions, versions and failure logs                     | Reproducible experiments and debuggability |
| Quality manager               | Audit lot grading, borderline items and defect evidence                  | Traceable batch reports                    |
| Agronomist                    | Join harvest records with sensors and weather                            | Provenance-backed agronomic context        |
| Embedded or robotics engineer | Use timestamped detection and route commands                             | Deterministic safe machine integration     |

## 2.2 Release scope and priority

| Priority             | Capabilities                                                                                                                                      | Release boundary                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| P0 assessment core   | Image upload, live camera, crop class, per-item mask, calibrated measurement, SMALL/BIG, REVIEW, debug overlay, batch CSV/JSON, metrics dashboard | Must work end to end for approved crops and capture setup  |
| P1 practical quality | Visible defects and leaf health triage, sort-lot QC, multi-view tracking, machine event export                                                    | Only after labels and separate defect validation           |
| P2 agronomy context  | Crop knowledge cards, weather and soil records, yield analytics, husbandry suggestions                                                            | Advisory and provenance layer, not image-inferred facts    |
| Future               | Robotic actuator closed loop, harvest drones, multispectral internal-quality sensors                                                              | Requires dedicated safety engineering and field acceptance |

## 2.3 Supported crop ontology

Pilot coverage: potato, onion, okra (ladyfinger/bhindi), cucumber, tomato, spinach or palak (separately tagged if botanical species differs), and coriander. Final exact crop/cultivar list is set from ARL inventory. Mixed vegetables, empty scenes, unripe produce, leaves not intended as produce, non-vegetables and unknown crops must be included as negatives. Botanical identity should not be invented from ambiguous visual appearance.

| Crop form         | Measurement convention                                                                        | Definition issue requiring owner sign-off                              |
| ----------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Potato and onion  | Projected transverse diameter in mm; assess pose bias or multi-view                           | Round/oblong cultivar, diameter vs mass purchase grade                 |
| Okra and cucumber | Visible full-pod longest centreline or calibrated major-axis length in mm; width as secondary | Curvature, tip visibility, harvest market maturity                     |
| Tomato            | Maximum equatorial diameter in mm under calibrated pose                                       | Cultivar differences; cherry tomato vs standard tomato                 |
| Spinach or palak  | Individual intact leaf blade length and optionally leaf area, only if laid flat               | Single leaf vs harvested bunch must be defined                         |
| Coriander         | Leaf or standardized flat-laid bunch projected area, or measured mass from a scale            | A loose bunch has no defensible fixed size from one uncalibrated image |

## 2.4 User stories and journeys

1\. As an operator, I select or confirm the current crop specification and camera profile, then inspect a calibration status indicator before starting a batch.

2\. For an uploaded image, the system returns all separate visible instances with species, measurement and grading status, and clearly requests a known-size reference or calibrated rig for true physical grading.

3\. For live input, the system assigns persistent object IDs and emits one final event per physical item, with a fail-safe review route if timing or confidence is unsafe.

4\. As a quality manager, I compare lot counts, distribution, manual corrections, defect evidence and threshold-version history before exporting a report.

5\. As an agronomist, I inspect crop guidance linked to a specified farm, growth stage, soil test and weather period; missing data display as unavailable.

## 2.5 Non-goals and prohibited claims

- No guarantee of 100% accuracy, unlimited crop recognition or correct performance in conditions absent from validation.
- No automated pesticide safety, pathogenic disease confirmation, hidden rot, Brix, nutrient, freshness, shelf-life or soil laboratory assay from a normal RGB image.
- No yield prediction presented as measured production without crop density, counted plants/fruit, area and recorded weights.
- No actuator or reject gate driven directly from the web interface without a verified physical safety interlock.

# 3 Software requirements specification

## 3.1 Functional requirements

| ID    | Requirement                                                                                                                | Acceptance evidence                       |
| ----- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| FR-01 | Accept authenticated JPEG/PNG uploads and supported camera streams; validate dimensions, decode integrity and size limits. | Format, corrupt-file and rate-limit tests |
| FR-02 | Process zero, one or multiple items; assign stable per-item IDs for video.                                                 | Crowded frames and tracking tests         |
| FR-03 | Return species from an approved versioned vocabulary and UNKNOWN for unsupported cases.                                    | Open-set and class confusion matrix       |
| FR-04 | Return per-instance contour/mask and pixel-space geometry.                                                                 | mAP mask, boundary and regression tests   |
| FR-05 | Load camera intrinsics, distortion, depth/homography or fiducial scale with calibration metadata.                          | Known-object measurement trial            |
| FR-06 | Compute approved dimensional feature and uncertainty; never report calibrated mm for an invalid setup.                     | Metrology uncertainty tests               |
| FR-07 | Grade with species- and customer-specific, versioned thresholds; borderline grades go to REVIEW.                           | Threshold boundary unit tests             |
| FR-08 | Provide overlay, species, length/width/area, units, grade, confidence, reasons, timestamp and version.                     | UI/API contract snapshot tests            |
| FR-09 | Support single-image and live video paths with equivalent grading semantics.                                               | Golden sample parity tests                |
| FR-10 | Generate lot-level counts, item CSV, JSON and a digest of review cases.                                                    | Export integrity tests                    |
| FR-11 | Retain image and sensor lineage, label source, model hash and operator changes.                                            | Audit replay tests                        |
| FR-12 | Provide annotation workflow and manual override with user/reason/time log.                                                 | Reviewer acceptance tests                 |
| FR-13 | Expose health metrics for acquisition, inference, depth validity, calibration drift and queue lag.                         | Synthetic telemetry and alert tests       |
| FR-14 | Optionally classify surface defects with separately trained classes and UNKNOWN when blocked.                              | Defect confusion and occlusion tests      |
| FR-15 | Attach nutrition and growing data only via source-linked crop knowledge or explicit farm records.                          | Missing-data and provenance checks        |
| FR-16 | Support dry-run sorter event messages with idempotency keys, sequence and expiry.                                          | Duplicate/reorder/fault injection         |
| FR-17 | Allow operator to switch to manual inspection without dropping original lot identity.                                      | Safe-degradation drill                    |

## 3.2 Nonfunctional requirements

| ID     | Requirement and proposed target                                                                                                                                                                | How to verify                                                       |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| NFR-01 | Pilot on valid, in-distribution accepted items: species macro F1 >= 0.95; size grade accuracy >= 0.95 and >= 0.90 on specified lighting-shift set. These are proposed go/no-go targets.        | Locked site, day, camera and lot holdouts; 95% confidence intervals |
| NFR-02 | Provisional metric tolerance: calibrated planar measurements <= 3 mm median absolute error for bulb crops and <= 5 mm for longer pods; replace by crop-specific tolerances from ARL metrology. | Blind caliper/ruler reference with repeat operators                 |
| NFR-03 | End-to-end p95 <= 250 ms for a single 640 px inference on selected pilot device; machine latency must satisfy belt time budget.                                                                | On-device benchmark after warm-up including capture/pre/post        |
| NFR-04 | \>= 99% batch event integrity in a controlled soak; unsafe hardware inputs result in STOP/REVIEW. Higher availability target requires site SLA.                                                | 24 h soak, watchdog and power-loss replay                           |
| NFR-05 | Minimum 95% automated coverage only if simultaneous error target is achieved; otherwise safe rejection is preferable.                                                                          | Risk-coverage curve                                                 |
| NFR-06 | Version and hash for every model, threshold table, calibration file, dataset and output.                                                                                                       | Audit reproducibility and exact replay                              |
| NFR-07 | Storage minimization: encrypted transport, bounded retention and restricted access; image exports opt in.                                                                                      | Threat review and access tests                                      |
| NFR-08 | Operator UI remains usable without network; optional weather/soil services fail independently.                                                                                                 | Offline and API-outage tests                                        |

## 3.3 State and exception contract

A grade is valid only when species is supported, instance visibility and segmentation pass, the physical scale is trusted, and the dimension uncertainty interval does not straddle the configured threshold. Otherwise response.state is one of UNKNOWN_SPECIES, RECAPTURE, CALIBRATION_REQUIRED, OCCLUDED, REVIEW_BORDERLINE, SENSOR_FAILURE or MANUAL_REVIEW. A missing weather feed must never change the size grade.

Output contract, one item: item_id, batch_id, frame_ts, crop_code, species_probability, mask_reference, metric_type, measured_value, measurement_unit, uncertainty_95, size_threshold, size_label, decision_state, quality_flags, calibration_id, threshold_version, model_sha, reasons and processing_ms. Batch records also store total seen, total accepted, total review, device ID and source metadata. Never use a generic scalar confidence as proof that mm is correct.

# 4 Size definition and camera metrology

## 4.1 Small and big must be crop-relative

ARL should define a grade_rules.yaml-like reviewed table keyed by species, cultivar/market, harvest form and specification version. Each row contains metric name, unit, threshold T, reference standard or buyer contract, valid pose, calibration mode, uncertainty tolerance, effective date and approver. Indian fruit and vegetable grading rules provide commodity-specific precedents, but they are not a universal binary cutoff \[10\].

Example policy, without inventing thresholds: an accepted potato with diameter d and calibrated 95% uncertainty half-width u is BIG only if d - u >= T_potato; it is SMALL only if d + u < T_potato; otherwise REVIEW_BORDERLINE. A cucumber uses its own length threshold T_cucumber and visibility rules. All threshold values remain TBD until the evaluator supplies grade criteria.

## 4.2 Acquisition configurations

| Mode                        | Sizing capability                                                                  | Required controls                                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| A fixed overhead RGB rig    | Accurate 2D planar measurement for suitable pose and geometry                      | Intrinsic calibration, rigid standoff, colour-stable lighting, calibrated ground-plane homography and reference checks     |
| B RGB-D or stereo rig       | Depth-aided 3D points and object distance with surface/occlusion limits            | Registered aligned RGB-depth, confidence maps, drift and range validation; validate in sunlight and reflective/wet produce |
| C user upload with fiducial | Approximate metric size when fiducial is known and coplanar with produce           | Printed or machined verified fiducial dimensions, detected corners and perspective rectification                           |
| D upload without scale      | Crop identity and visible appearance only; not a reliable physical SMALL/BIG grade | Show MEASUREMENT_UNAVAILABLE or an explicitly labelled unverified appearance-only exploratory estimate                     |

## 4.3 Measurement algorithm

1\. Calibrate lens intrinsics and distortion using OpenCV chessboard or ChArUco and record reprojection residuals, per-camera IDs and calibration date \[11\].

2\. Undistort, check image sharpness/exposure/glare, and align colour and depth or rectify the fixed conveyor plane.

3\. Run instance segmentation. Compare mask against optical boundaries; flag truncation, severe occlusion or object touching the frame edge.

4\. Map valid segmentation contour points into world coordinates using a calibrated planar homography, or depth plus pinhole intrinsics: X = (u-cx)Z/fx and Y = (v-cy)Z/fy.

5\. Compute physical shape features from geometry: diameter for bulb crops, full axis or smoothed centreline for pod crops, projected leaf area for a flattened individual leaf. Avoid using axis-aligned bounding box size as exact length for diagonal or curved produce.

6\. Propagate measurement uncertainty from marker detection, pose, depth noise, segmentation boundary and reference caliper variation; validate empirical error bands per crop and camera.

7\. Evaluate the crop threshold with interval logic, suppress duplicate tracks and emit one immutable final decision.

OpenCV primitives: calibrateCamera, undistort/remap, findChessboardCorners or ChArUco, findHomography/warpPerspective, findContours, contourArea, minAreaRect and connectedComponents. Classical HSV, adaptive thresholding and morphology form an interpretable baseline under controlled backgrounds; they are not relied on for arbitrary farm scenes \[11, 12\].

# 5 Machine learning model selection

## 5.1 Recommended experimental ladder

| Candidate                               | Use                                               | Selection rationale and caveat                                                                                                                                |
| --------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OpenCV background separation + geometry | Mandatory low-compute baseline in controlled tray | Interpretable; inexpensive, but fragile to shadows, soil, overlapping leaves and similar backgrounds                                                          |
| YOLO26n-seg or small variant            | Primary real-time research candidate              | Train crop-specific instance masks; choose based on actual ARL p95 latency and field risk, not only generic COCO mAP; check AGPL/Enterprise obligations \[6\] |
| Torchvision Mask R-CNN                  | Independent deep learning baseline                | Well documented supervised instance masks; likely slower on constrained edge, licensing and exact model weight terms require review \[13\]                    |
| SAM 2                                   | Assisted offline annotation and tricky-mask audit | Excellent promptable mask assist, not proof of vegetable class/size and not necessarily edge-efficient \[14\]                                                 |
| RGB-D geometry and multiview tracking   | Upgrade for 3D/hidden contour ambiguity           | Adds sensor/hardware complexity; supported by vegetable sizing and grading literature \[2, 3\]                                                                |

Final model is selected by a predefined scorecard on ARL-held-out captures: accepted grade error and safety, size measurement error, per-species recall, false sorting rate, field shift and edge latency. The model choice remains provisional until these measurements exist. A direct binary classifier may serve as a baseline, but its labels couple size and camera distance and can spuriously learn background or scale cues.

## 5.2 Model components

- Instance segmentation network: categories for the approved produce ontology plus background; optionally a separate head or post-crop species classifier if visually similar classes need it.
- Deterministic metrology service: geometry, units, calibrated uncertainty, policy thresholds and review states; kept independent of learned crop probabilities.
- Input-quality and out-of-distribution gate: blur, exposure, glare, scale visibility, unsupported class, lens obstruction and confidence risk.
- Optional defect detection: independently labelled surface damage, mould-visible, discolouration, insect damage and torn leaves; outputs are observations, never laboratory disease diagnoses.
- Tracking and lot aggregation: stable IDs, entrance/exit line triggers, duplicate suppression and lot provenance.

## 5.3 Training recipe

Start pretrained and fine-tune with field data; freeze a seed, code commit, package lock and dataset version. Compare input resolutions 640/960, augmentations and model sizes via controlled experiments. Use exposure and colour-temperature changes, mild motion blur, rotations, shadows, varying backgrounds, occlusion/composition, realistic noise and wet/dirty produce. Restrict augmentations that alter physical label semantics; never resize an object and reuse its mm ground truth without recalculating world geometry. Keep calibration overlays and printed fiducials out of shortcuts by randomising their position and appearance appropriately.

Loss and experiment reporting: classwise AP and mask AP; confusion by class; mask IoU/boundary error; measurement MAE, median, p90 and bias; grade classwise precision/recall; risk versus accepted coverage; and hardware p50/p95. Tune confidence/abstention rules on calibration data, not final test data. Modern networks may be miscalibrated, so verify rather than trust softmax as correctness probability \[15, 16\].

# 6 Field data and annotation specification

## 6.1 Capture sampling and size references

Preferred dataset is proprietary ARL imagery from its actual cameras, crops, belts, lighting, farms and postharvest stages. AgML catalogues useful public agricultural datasets and PlantVillage can help transfer learning, but neither provides ARL-specific physical size labels by default \[4, 17\]. A preliminary acquisition target is 300 to 600 distinct measured instances per crop for feasibility, then 1,000 or more per crop if error curves require it; these are planning numbers, not statistical guarantees. Collect minority boundary sizes deliberately.

| Stratum       | Minimum deliberate coverage                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------------ |
| Crop identity | Each approved crop, cultivars, maturity range, organ/bunch definition, organic dirt or residual soil   |
| Lighting      | Direct sun, overcast, shade, dusk/LED, mixed light, backlight, shadow, wet glare, low exposure         |
| Scene         | Tray, conveyor, soil/leaf clutter, reflective metal, hands/tools, empty negatives, camera jitter       |
| Geometry      | Small and big extremes, threshold borderline, rotated, curved, partial, occluded, touching and crowded |
| Acquisition   | Different sessions, lots, farms/rooms, camera units, lenses, sensor temperatures, lens contamination   |
| Environmental | Humidity, dust, moisture on produce, processing speed, motion blur, rain-associated field wetness      |

## 6.2 Label schema and human ground truth

- Each instance: image/frame ID, lot ID, crop code, variety if known, polygon or RLE mask, occlusion fraction/flags, full visibility flag, crop stage and review status.
- Physical dimensions: caliper or ruler method, instrument ID, timestamp, two repeat measurements where feasible, operator, crop-axis protocol and unit; calibrated weight scale only when weight is required.
- Capture lineage: camera serial, lens, exposure, pixel resolution, height, calibration/version, fiducial visibility, lighting stratum and weather if directly observed.
- Quality annotations: surface-visible defect type and severity, affected area, visibility, lab-validated diagnosis only if externally established.
- Agreement: double-label at least a stratified audit sample, resolve disagreements in adjudication, and report inter-annotator differences and mask boundary variance.

## 6.3 Leakage-safe split

Split by physical item identity and lot/session, never random frames of the same item. Reserve entire capture dates, camera units and at least one external setting as holdout where possible. A planning split of 70/15/15 at the group level is acceptable only if it preserves independent groups; the true external field test remains sealed until final. Keep synthetic stress tests and curated edge cases as additional named suites, not substitutes for an independent test.

## 6.4 Data provenance and legal diligence

Record source URL, license, consent/ownership, crop classes, country/site, annotation rules and permissible commercial use for each external dataset and pretrained weight. SAM 2 code and checkpoints are released by Meta under Apache 2.0, while Ultralytics publishes AGPL-3.0 and Enterprise paths; ARL legal/engineering must review deployment constraints before embedding a chosen stack \[6, 14\].

# 7 Deployment and systems architecture

## 7.1 Processing path

Camera or upload gateway -> validated frame + metadata -> undistortion and scale check -> segment and identify instances -> compute world geometry with uncertainty -> crop rule engine -> review/sort output -> operator UI, batch audit and optional machine event. Weather, soil and knowledge lookup run asynchronously and cannot change measurement validity.

## 7.2 Local application and services

| Service                   | Responsibility                                                                     | Failure response                            |
| ------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------- |
| Capture and upload        | RTSP/USB/CSI input, still upload, timestamps and pixel-format normalization        | Pause line or request recapture             |
| Inference worker          | Versioned model, CPU/GPU execution, masks and class logits                         | Circuit break and REVIEW                    |
| Metrology and policy      | Calibration, world points, dimensional metric, threshold and uncertainty           | CALIBRATION_REQUIRED                        |
| Tracking and event bridge | Object ID, timestamped belt position, actuator scheduling, event deduplication     | Suppress unsafe actuator commands           |
| Dashboard and audit API   | Overlays, per-item JSON, lot summaries, corrections and health                     | Read-only degraded mode                     |
| Farm context              | Soil lab tests, sensors, weather, crop information and guidance with source stamps | Unavailable badge without affecting grading |

## 7.3 Live edge budget

Before any actuator integration, measure exposure + transfer + preprocessing + inference + postprocessing + communication + actuation trigger against available transit time, approximately sensor-to-sorter distance divided by belt speed minus safety margin. Enforce one final command per tracked vegetable, expiry before the diverter, state machine interlocks, emergency stop and dry-run simulation. Network failures, out-of-order frames and duplicate events default to the reject/manual route, never arbitrary SMALL/BIG.

Start with a laptop CPU/GPU assessment mode. If edge hardware is available, benchmark on its exact image dimensions and power mode; NVIDIA Jetson Orin Nano Super is one option, but its published TOPS figures are not an application-latency guarantee. Use on-device TensorRT/ONNX benchmarking and compare quantized against full precision accuracy \[18, 19\].

## 7.4 Interface design

- Landing: clear camera/live tab and upload tab, species list, active crop-rule version and calibration status.
- Inspection view: original/overlay toggle, cropped masks, full image, per-item species and confidence, mm + uncertainty, grade or reason for review.
- Batch view: counts by size, species, defects, review rate, unknown rate, selected camera/lot, export and correction log.
- Research view: raw mask, per-stage time, image QA flags, dimension derivation and reference measurements.
- Agronomy tab: farm and crop context, current sensors and lab values with timestamps, estimated regional products distinguished visually from sensor measurements.

# 8 Agronomy intelligence and yield extension

## 8.1 Data truth hierarchy

| Information                        | Defensible input                                                                        | Displayed status                             |
| ---------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------- |
| Vegetable category and outline     | Image + crop-specific validation                                                        | Observed by vision                           |
| Dimensions                         | Image + calibrated rig/reference or depth                                               | Measured estimate with uncertainty           |
| Weight                             | Scale measurement; separately calibrated statistical volume-mass model only as estimate | Measured or model estimate                   |
| Visible condition                  | Annotated surface observations with partial visibility                                  | Visible observation only                     |
| Soil pH, EC, NPK, organic matter   | Timestamped soil analysis / sensor records; SoilGrids broad regional covariates         | Farm measurement or mapped regional estimate |
| Rain, temperature, solar radiation | Farm weather station, IMD or NASA POWER temporal/context data                           | Measured station or provider estimate        |
| Nutritional facts                  | Versioned crop-specific public composition entry such as USDA FoodData Central          | Reference values per 100 g; not sample assay |
| Yield                              | Field area, stand density, harvest counts, aggregate measured mass over time            | Observed or forecast with stated model       |
| Improvement advice                 | Crop/cultivar/stage, soil tests, local weather, irrigation and expert-reviewed guidance | General advisory; needs agronomist review    |

ISRIC SoilGrids supplies 250 m modelled soil maps with uncertainty, which are not suitable for assigning a lab-grade nutrient value to an individual plant. Its REST service is reported temporarily paused; design a cache/alternate data path rather than a blocking call \[8\]. NASA POWER provides agroclimatology history and observations/model products for a location; use local farm telemetry when available \[9\]. FoodData Central is appropriate for reference food composition, not personalised medical benefit claims \[20\]. ICAR and Indian horticultural extension are more appropriate than generic image-language generation for crop practice guidance \[21\].

## 8.2 Agronomy cards

An optional crop card can show common name, botanical name only when taxonomically unambiguous, edible part, common uses, selected public nutrition values and source, broad cultivation requirements and postharvest notes. Any soil, irrigation or fertiliser recommendation must use actual location, crop variety, growth stage and soil/water records, carry an information-source date, and be reviewed locally. Do not infer nutritional benefit from physical size or visual colour alone.

## 8.3 Yield analytics separation

In the packhouse, report throughput, accepted kilograms if weigh-scale data exist, grading distribution and quality-related rejection. In preharvest contexts, build a separate crop-count/density monitoring project using plot maps, sampling, longitudinal observations and weather, with confidence intervals. A photographed harvested potato does not reveal field yield.

# 9 Testing and statistical acceptance

## 9.1 Validation hierarchy

1\. Unit tests: threshold equality, interval overlap, missing/zero scale, calibration expiry, class mapping, contour morphology, unit conversions and unknown output states.

2\. Metrology bench: rulers, certified check objects, repeated caliper measurement, varied positions/heights/angles, thermal drift and depth-sensor reflection.

3\. Model evaluation: held-out species accuracy, detection and mask quality, physical dimension error, calibration curves and risk-coverage by condition.

4\. Stress test: blur, motion, very dark and overexposed images, counterlight, water droplets, camera blockage, occlusion, clutter, repeated items, background confounding and malicious/corrupt upload.

5\. Systems integration: live stream reconnection, throughput burst, dropped frames, timestamp skew, false double-trigger, sort timing and dry-run replay.

6\. Field trial: independent new-day, new-lot and new-camera captures with technician-measured truth and non-negotiable review of every unsafe sort.

7\. Shadow mode: live predictions with no autonomous actuation until observed errors, controls and fault handling are accepted.

## 9.2 Metric and stratification matrix

| Metric           | Proposed gate                                                                                         | Stratified reporting                                  |
| ---------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Crop recognition | Macro F1 >= 0.95 on approved test; explicit unknown handling                                          | Per species, cultivar, setting and domain             |
| Masks            | Mask AP50, AP50:95, boundary IoU; no universal one-number gate                                        | Thin leaves, stems, adjacent items, occluded          |
| Size estimation  | Crop-specific MAE, bias, p90 error and 95% measurement interval coverage                              | Crop, measured size bin, pose, sensor                 |
| Size decision    | \>= 95% accuracy among accepted valid in-distribution items, with class precision and false-sort cost | Near threshold, each crop, lighting, camera           |
| Abstention       | Report coverage versus risk and false automatic release; thresholds tuned before holdout              | Unknown, no-scale, occluded, severe blur              |
| Runtime          | p50/p95/p99 full pipeline; queue lag and memory                                                       | Device, batch size, resolution, power and temperature |
| Machine events   | No duplicate accepted event and no actuation when watchdog fails                                      | Dropped frames, reboot, network, clock drift          |

All thresholds above are assessment proposals pending ARL risk tolerance, crop grades and camera hardware. Report sample counts and interval estimates, not bare percentages: for example, use Wilson 95% intervals for binomial success and bootstrap by independent lot for correlated instances. Require minimum strata counts or mark the evidence insufficient. One hundred observed successes do not prove 100% true accuracy.

## 9.3 Definition of done for the core assessment

- A clean environment can reproduce installation, dataset splits, training, inference, metrics and a model hash from documented steps.
- A real uploaded crop image produces item masks, species prediction and an honest metric-size result only when scale is valid.
- A live camera replay and physical camera demonstration produce the same per-item decision contract; unsafe frames enter REVIEW.
- At least two crop forms (one bulb, one pod or leaf) receive real manual metric ground truth; full crop pilot follows as labelled data allows.
- A locked independent holdout report includes per-crop confusion matrices, mm residual plots, failure gallery and stratified slices.
- Threshold policies are documented, versioned and operator-approved; equality and uncertainty tests pass.
- User-facing UI displays crop and units, calibration, model version, reviewed exceptions and exported batch log.
- A demo recorded with unedited raw frames shows correct handling of dark, bright, blurred, occluded, unsupported and no-scale inputs.
- No unverified soil, defect, nutrition or yield claim is shown as an image-derived measurement.
- All external dataset/model licenses, citations, and field deployment limitations are disclosed.

# 10 Execution plan and engineering ownership

| Phase                             | Work package                                                                                                                        | Evidence produced                                   |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| D0 to D1 discovery                | Confirm crop set, exact grade thresholds, form of spinach/coriander, camera placement, hardware, constraints and evaluation metric. | Signed assumptions and data collection protocol     |
| D2 to D3 calibration and baseline | Build photo/live intake, intrinsic+plane calibration, reference objects, OpenCV mask/geometry baseline.                             | Calibration certificate, benchmark on sample images |
| D4 to D7 dataset and segmentation | Collect real measured crops, mask annotation QA, train YOLO26n-seg and Mask R-CNN comparison.                                       | Dataset card, frozen splits, initial model card     |
| D8 to D10 integration             | Metrology engine, rule-version registry, uncertainty/review, upload and video UI, lot JSON/CSV.                                     | End-to-end demo and automated tests                 |
| D11 to D13 reliability            | Stress matrix, camera/domain holdouts, on-device profiling, unknown cases, fault injection.                                         | Error report, latency and coverage curves           |
| D14 to D15 acceptance             | Shadow-mode trial, failures addressed, demo recording and reproducibility run.                                                      | Final assessment dossier and limitations            |
| Beyond pilot                      | Expanded crops, defect labels, RGB-D/multi-view, actuators and agronomy integration.                                                | Separate milestone approvals                        |

## 10.1 Repository layout and required evidence

Recommended folders: configs/crops and cameras; data/README and data_cards; src/capture, segmentation, metrology, policies, tracking, API and agronomy; tests/unit, integration, metrology, fixtures, edge and regression; scripts/train, evaluate, calibrate, export and benchmark; reports/metrics and field_trials; docs/architecture, model_card, runbook and safety. Keep large datasets out of git, pin versions, and ensure every report names the exact code and dataset revision.

## 10.2 Demonstration sequence

1\. Show crop grading policy and a measured ruler/reference setup, including the camera calibration record.

2\. Upload potato and okra samples of known measured size and show instance masks, mm, thresholds and SMALL/BIG.

3\. Upload an uncalibrated arbitrary image; show honest CALIBRATION_REQUIRED instead of inventing centimetres.

4\. Run the live camera at varying light and show continuous per-item IDs, latency and borderline REVIEW.

5\. Show batch statistics and manual correction log; then the independent held-out validation report and failure gallery.

6\. Show a soil/weather/nutrition card explicitly labelled as contextual and source-linked, while explaining the input provenance.

7\. Trigger a camera-loss or low-confidence condition and demonstrate safe no-sort/manual-review behaviour.

# 11 Risk register and decision log

| Risk                                     | Failure impact                                   | Mitigation                                                          |
| ---------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------- |
| Absent metric scale or wrong camera pose | Systematic incorrect SMALL/BIG                   | Calibration gate, fiducial, known rig, optional depth and REVIEW    |
| Leaf bunch not defined as item           | Contradictory ground truth                       | Specify single leaf vs weighed bunch protocol before annotation     |
| Crop domain shift                        | False classes under farm lighting                | Real farm holdouts, active learning and environmental tests         |
| Boundary sizes near T                    | Unstable automatic grades                        | Uncertainty interval and review band                                |
| Mask truncation and overlap              | Wrong size and double count                      | Occlusion reason, multi-view and persistent IDs                     |
| RGB-only hidden quality inference        | False freshness or safety claims                 | Visible-only quality labels and external assays                     |
| External API outages                     | Misleading or blank agronomy panel               | Cache, timestamps, provider metadata, independent failure           |
| YOLO licensing misunderstanding          | Potential commercial incompatibility             | Legal review, enterprise license or suitable permissive alternative |
| Premature robot actuation                | Misdirected produce or machinery safety incident | Shadow mode, interlock, watchdog, E-stop and physical sign-off      |

## 11.1 Unresolved questions for Nicholas or Nikhil

1\. Is the required unit an individual leaf, bunch, harvested pod, tuber or mixed container? Is small/big based on length, diameter, mass, surface area or market grade for each crop?

2\. Which 3 to 7 crops are definitely in assessment scope, and what exact approved threshold and borderline handling applies to each?

3\. Is the judge expecting reliable arbitrary phone-photo grading without a reference, or can uploads require a fiducial and live mode use a fixed camera?

4\. What cameras, depth sensors, GPU/Jetson, conveyor speed, illumination and space are actually available? Is physical actuation expected?

5\. Is the primary evaluation held-out image accuracy, measured mm error, business sorting consistency, throughput, or a combination?

6\. Can ARL provide actual lot samples, a digital caliper/weigh scale, field imagery and a farm agronomist to validate thresholds?

7\. Must data remain on farm, and may model code/weights be distributed under an open-source license?

8\. What deadline, minimum demonstrable crop count and expected deliverables (repository, UI, recording, hardware test) apply?

# 12 Evidence and reference bibliography

These sources support design decisions, not any unmeasured performance assertion for ARL. Peer-reviewed works and official primary manuals take priority; dataset and licensing details must be rechecked during implementation.

\[1\] Neupane, Pereira, Koirala and Walsh. Fruit Sizing in Orchard: A Review from Caliper to Machine Vision with Deep Learning. Sensors 2023, 23(8), 3868. <https://pmc.ncbi.nlm.nih.gov/articles/PMC10144371/>

\[2\] Vegetable Size Measurement Based on Stereo Camera and Keypoints Detection. Sensors 2022. <https://pmc.ncbi.nlm.nih.gov/articles/PMC8877767/>

\[3\] Prototyping and evaluation of a novel machine vision system for real-time, automated quality grading of sweetpotatoes. Computers and Electronics in Agriculture, 2024. <https://www.sciencedirect.com/science/article/pii/S0168169924002175>

\[4\] Mohanty, Hughes and Salathe. Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science, 2016. <https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full>

\[5\] Quantifying the reliability gap in cross-domain plant disease classification. Frontiers in Plant Science, 2026. <https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1826962/full>

\[6\] Ultralytics YOLO26 official model documentation and licensing, accessed 2026-10-22. <https://docs.ultralytics.com/models/yolo26> and <https://www.ultralytics.com/license>

\[7\] Automatic Cucumber Maturity Classification and Size Estimation via RGB-D Vision and YOLOv8n. Horticulturae, 2026. <https://www.mdpi.com/2311-7524/12/10/1242>

\[8\] ISRIC SoilGrids information and access advisories. <https://docs.isric.org/globaldata/soilgrids/> and <https://isric.org/explore/soilgrids/>

\[9\] NASA POWER Daily API, Agroclimatology Community. <https://power.larc.nasa.gov/docs/services/api/temporal/daily/>

\[10\] Government of India. Fruits and Vegetables Grading and Marking Rules, 2004 and amendments. <https://upload.indiacode.nic.in/showfile?actid=AC_CEN_23_31_00011_193701_1535099362507&filename=fruits_and_vegetables_grading_and_marking_rules%2C_2004.pdf&type=rule>

\[11\] OpenCV official camera calibration and ChArUco documentation. <https://docs.opencv.org/5.0/tutorials/calib3d/camera_calibration/camera_calibration.html> and <https://github.com/opencv/opencv/blob/5.x/doc/tutorials/objdetect/aruco_calibration/aruco_calibration.markdown>

\[12\] OpenCV contour methods documentation. <https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html>

\[13\] PyTorch Torchvision. Object Detection and Instance Segmentation Finetuning Tutorial. <https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial>

\[14\] Ravi et al. SAM 2 Segment Anything in Images and Videos. Meta Research, 2024; official code and license. <https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/> and <https://github.com/facebookresearch/sam2>

\[15\] Guo et al. On Calibration of Modern Neural Networks. ICML/PMLR 2017. <https://proceedings.mlr.press/v70/guo17a.html>

\[16\] Geifman and El-Yaniv. Selective Classification for Deep Neural Networks. NeurIPS 2017. <https://arxiv.org/abs/1705.08500>

\[17\] Project AgML. Agricultural datasets and official repository. <https://project-agml.github.io/datasets> and <https://github.com/Project-AgML/AgML>

\[18\] NVIDIA. Jetson Orin Nano Super official product and guide. <https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/>

\[19\] NVIDIA TensorRT. Performance Benchmarking using trtexec. <https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/benchmarking.html>

\[20\] USDA. FoodData Central API and public-domain food composition data. <https://fdc.nal.usda.gov/api-guide>

\[21\] ICAR. Okra Integrated Crop Management; TNAU Post Harvest Technology of Common Vegetables. <https://epubs.icar.org.in/index.php/IndHort/article/view/115798> and <https://agritech.tnau.ac.in/postharvest/pht_tech_comn_vegetble.html>

\[22\] Postharvest Fruit Grading Technologies and Equipment A Review, 2026. <https://pmc.ncbi.nlm.nih.gov/articles/PMC13606810/>

\[23\] Three-dimensional analysis of fruit and vegetables: Sensors and methods, 2026. <https://www.sciencedirect.com/science/article/pii/S016816992600092X>

\[24\] Design and Experimentation of a Machine Vision-Based Cucumber Quality Grader, 2024. <https://pubmed.ncbi.nlm.nih.gov/38397583/>