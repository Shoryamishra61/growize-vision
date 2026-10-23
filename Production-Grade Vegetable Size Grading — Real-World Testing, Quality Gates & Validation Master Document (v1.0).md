# Production-Grade Vegetable Size Grading

## Real-World Testing, Quality Gates & Validation Master Document

Version 1.0 \| 23 October 2026 \| Status: DRAFT — NOT RELEASE APPROVED

Purpose: Turn a camera-based BIG/SMALL vegetable classifier trained or prototyped with synthetic/generated imagery into a defensible, auditable, continuously operable production system.

Document owner: \[UNASSIGNED\] \| QA owner: \[UNASSIGNED\] \| Site: \[UNSPECIFIED\] \| Model/build: \[UNSPECIFIED\]

Important: This is a newly authored master document, not a recovered copy of the inaccessible Deep Research file. No real-world results have been supplied; all test outcomes remain NOT RUN.

# 1\. Executive decision and non-negotiable principles

The current system must be considered a prototype until it has passed measurement, classification, system integration, and live-site acceptance testing on independently collected real produce. Passing synthetic-image tests is evidence that software executes; it is not evidence of production accuracy.

The system must not silently label every item BIG or SMALL. When the crop is unknown, an object is occluded, the image is unusable, the item lies within the uncertainty band around a threshold, or a hardware fault occurs, it must emit REVIEW/REJECT/UNKNOWN and route safely.

“100% error-free” is not a defensible promise. Even zero observed errors on a finite sample cannot establish zero future errors. Define acceptable risk, test it with confidence intervals, and monitor after deployment.

Release is BLOCKED until crop-specific definitions, real ground truth, acceptance limits, physical sort verification, and named approvers are complete. Numeric examples in this document are illustrations, not validated site specifications.

# 2\. Scope, output contract and crop policy

In scope: one or more overhead/side cameras; controlled illumination; optional encoder, trigger, weighing scale, multi-view or depth sensor; segmentation/detection; camera-to-mm measurement; crop identification or operator-selected crop; BIG/SMALL/REVIEW decision; conveyor tracking; actuator/sorter; logging; 24/7 monitoring.

Out of scope unless separately validated: ripeness, disease, defects, nutritional quality, exact weight from a single 2D photo, hidden dimensions, and legally certified commercial grading.

Output record for each item: unique\_item\_id; timestamp; camera\_id; model\_version; calibration\_id; crop\_type; crop\_confidence; segmentation\_quality; measured\_metric; measured\_value\_mm\_or\_cm2\_or\_g; measurement\_uncertainty; crop\_threshold; margin\_to\_threshold; predicted\_grade; decision\_reason; actuator\_target; actual\_bin\_or\_reject; image\_hash; operator\_override.

Crop catalog must explicitly distinguish potato, sweet potato, onion, tomato, cucumber, brinjal/eggplant, okra/bhindi, bottle gourd/lauki, bitter gourd/karela, ridge gourd/turai, green chili, cabbage/cauliflower, and leafy greens (spinach/palak, coriander/dhaniya, fenugreek/methi, amaranth/chaulai) only where each is supported by real validation.

For leafy produce define the physical unit: single leaf, stem, bunch, tray, or weighed portion. “Size of spinach” is otherwise ambiguous. Define grade by a measurable property (e.g., bunch projected area with controlled compression, stem length, bunch mass), and validate that property. Do not reuse round-vegetable diameter rules for leafy greens.

For elongated produce use a crop-specific combination such as length and width, not a generic bounding-box area. For near-spherical produce define maximum diameter, minimum diameter, equivalent diameter, or weight; the metric must match the business contract.

# 3\. Freeze the product requirements before tuning

REQ-01 \[UNSPECIFIED\] Supported crop types, varieties, and whether crop identity is selected by an operator or predicted automatically.

REQ-02 \[UNSPECIFIED\] Unit of grading (individual piece, bunch, pack); whether damaged, cut, wet, or soil-covered produce is supported.

REQ-03 \[UNSPECIFIED\] Ground-truth size metric and unit for each crop; example: potato maximum Feret diameter in mm, okra length in mm, spinach bunch mass in g.

REQ-04 \[UNSPECIFIED\] Crop-specific BIG/SMALL threshold T and source of authority (customer contract, applicable standard, or locked representative reference distribution).

REQ-05 \[UNSPECIFIED\] Borderline policy: guard band δ in physical units and handling of REVIEW.

REQ-06 \[UNSPECIFIED\] False-BIG and false-SMALL cost, allowable rates, minimum decision coverage, and target confidence.

REQ-07 \[UNSPECIFIED\] Maximum measurement MAE, 95th-percentile absolute error, systematic bias, and repeatability.

REQ-08 \[UNSPECIFIED\] Required throughput in items/min, end-to-end latency, conveyor speed, spacing, uptime and planned maintenance.

REQ-09 \[UNSPECIFIED\] Allowed lighting/temperature/humidity/dust/water conditions, supported camera distance and placement.

REQ-10 \[UNSPECIFIED\] Physical reject/sorter error budget, fail-safe bin, operator intervention and emergency-stop behavior.

REQ-11 \[UNSPECIFIED\] Real-site pilot duration, number of lots/days/operators, required sample counts and statistical confidence.

REQ-12 \[UNSPECIFIED\] Owner, approver, version control, data retention, privacy and security policy.

Decision record: no threshold, guard band or release metric may be inferred from synthetic examples or changed after the blind test set is inspected. Any such change creates a new model/configuration requiring a fresh holdout evaluation.

# 4\. System architecture and evidence chain

Recommended flow: item spacing → photoeye/trigger → fixed exposure and synchronized lighting → image capture → crop type verification → instance detection/segmentation → geometric calibration → physical dimension estimation → uncertainty and quality checks → crop-specific grading → REVIEW if ambiguous → item-to-conveyor tracking → actuator command → physical-bin verification → persistent event log.

Keep four separately measurable outcomes: (A) image/object detection, (B) physical dimension estimation, (C) grade decision, and (D) actual bin placement. A correct vision prediction does not prove a correct mechanical sort.

Use deterministic crop-specific rules after measurement where possible; model confidence is not a substitute for dimensional uncertainty. For high-value or variable-height produce, evaluate stereo, depth, multi-view rotation, or a checkweigher instead of assuming one overhead camera measures hidden 3D size.

Traceability chain for every validation item: lot\_id → item\_id → independently measured ground truth → raw image/video → annotation version → calibration version → prediction → physical destination → adjudication → test report.

# 5\. Real-data acquisition and sampling protocol

Synthetic images may be used for pipeline smoke tests, stress generation and pretraining, but must not be counted as independent real-world accuracy evidence. Capture real produce from the actual target camera, optics, belt, illumination and mechanical layout.

Sampling dimensions: crop and variety; small/borderline/large size; farm/supplier; harvest and storage age; season; day/night and shift; wet/dry/soil; clean/damaged; natural color variation; rotation and orientation; single/adjacent/touching pieces; motion blur; belt speed; illumination variation; camera and lens; operator and line; background and tray.

Create a collection manifest per image or video segment: source\_site, farm\_or\_supplier, lot\_id, harvest\_date\_if\_known, capture\_date, shift, camera\_id, exposure, lens, camera\_height, light\_configuration, belt\_speed, crop, item\_id, ground\_truth\_method, annotator, annotation\_status, consent/licensing status.

Capture size-balanced samples, deliberately oversampling near-threshold items and known difficult conditions. Separately collect a representative production-frequency sample for unbiased prevalence-weighted performance. Do not evaluate only attractive, centered, isolated vegetables.

Recommended initial engineering targets (not proof of statistical adequacy): 100–200 real items per crop for calibration/debugging, then an independently sourced holdout sized from the required error-rate confidence bound. More are needed for rare conditions, numerous crops, very low permitted error, or a model with many tunable parameters.

Prevent leakage: split by farm/lot/day/capture session where practical, not random adjacent video frames. All frames and crops from the same physical vegetable must stay in one split. Keep a final untouched test set from different collection sessions; never tune on it.

Data partitions: DEVELOPMENT for algorithm iteration; CALIBRATION for optical geometry and reference measurements; VALIDATION for choosing thresholds and guard bands; BLIND TEST for final claims; SHADOW PRODUCTION for real operational confirmation.

Evidence: signed sampling plan, raw files with hashes, manifest, versioned annotation exports, split assignment script, contamination audit and blind-set access log.

# 6\. Ground-truth measurement and annotation SOP

Ground truth must be measured independently of the vision prediction. Choose calibrated digital calipers or a suitable gauge for length/diameter, a calibrated scale for mass, and a standardized fixture for bunch dimensions. Document instrument accuracy, resolution, traceable checks and environmental conditions.

Define exactly where to measure: e.g., maximum Feret length along a specified axis, maximum width perpendicular to it, or a standardized 3D caliper dimension. The protocol must state whether stems, roots, protrusions, damaged regions and leaf tips count.

Assign permanent IDs before capture. For each item: measure twice with reorientation; if disagreement exceeds the approved repeatability limit, obtain a third measurement and adjudicate. A subset must be measured by a second operator blind to the first and to the model.

Run a gauge repeatability/reproducibility (Gage R\&R) study across operators, instruments and crops. Ground-truth error should be materially smaller than the allowable model error. Otherwise the validation label itself is too noisy to support the claim.

Segmentation annotation classes: target item, background/belt, touching or overlapping item, stem/leaf exclusion if applicable, occlusion, blur, unknown crop, invalid capture. Define when an image is ungradable rather than forcing a mask.

Grade ground truth is computed from the independent physical measurement and the frozen crop rule, not copied from a human guess about whether the image “looks big.”

Disagreements: adjudicator records original values, reason, final measurement, date and reviewer. Never quietly delete hard samples. Exclusions require pre-registered criteria and counts.

# 7\. Camera calibration, geometry and measurement uncertainty

Lock focus, aperture, focal length, zoom, resolution, exposure and camera mount. Use a checkerboard/ChArUco or equivalent reference for intrinsic calibration and distortion; calibrate the image-to-conveyor plane using a known ruler/marker or homography at the measurement plane.

A local planar approximation is mm\_per\_pixel \= known\_reference\_mm / observed\_reference\_pixels. Then length\_mm ≈ measured\_pixels × mm\_per\_pixel. This is only valid for the calibrated plane and suitable imaging geometry; perspective, lens distortion, object height and tilt create bias.

Illustration only: a 100 mm planar ruler spanning 500 pixels gives 0.2 mm/pixel. A 180-pixel projected width corresponds to 36 mm on that plane. If the vegetable surface is 40 mm above the belt, a perspective camera may have a different scale; 36 mm is not automatically its true diameter.

Test reference objects at multiple positions across the full field of view, orientations, heights and belt speeds. Compare measured physical dimensions with certified gauges. Report position-dependent bias, distortion residuals and uncertainty. Recalibrate after camera movement, lens change, belt height change or impact.

Uncertainty contributors: reference accuracy; intrinsic and extrinsic calibration; segmentation boundary error; object height/parallax; perspective; motion blur; illumination/shadows; ground-truth instrument; crop shape and orientation; frame-to-frame variation.

Define a conservative expanded uncertainty U (using an approved uncertainty method) and a guard band around T. A simple initial decision policy is SMALL when estimate \+ U \< T; BIG when estimate − U \> T; otherwise REVIEW. This may be too conservative or insufficient depending on the true error distribution; validate coverage and calibration empirically.

Do not convert an object’s 2D pixel area directly to mass/volume without an independently validated crop-specific model. For curled leaves and piled bunches, overhead area is particularly unreliable.

Calibration checks: start of shift, after maintenance, after any camera/lighting relocation, and at defined intervals. Persist calibration ID, reference results, and pass/fail status. On failed check, stop auto-sorting or divert to REVIEW.

# 8\. Thresholds, median grading and boundary policy

The median is a reasonable definition of “large relative to this reference population,” but not an objective commercial size standard. It changes with the sampled farm, season, variety and collection bias; it must not be recomputed opportunistically from live incoming items unless the business explicitly wants a moving threshold.

Procedure: select a representative and documented reference population for each crop/variety; measure the agreed metric physically; compute the median; obtain stakeholder approval; freeze threshold T and its version; choose a guard band based on measured error and business risk; validate on new lots.

If a customer or applicable standard specifies minimum dimensions or weight grades, that approved criterion takes precedence over an arbitrary median. Some crops may need more than two grades; do not collapse them without stakeholder authorization.

Boundary example: T \= 50 mm and validated uncertainty U \= 2 mm. A 47 mm estimate can be SMALL; 53 mm can be BIG; estimates between 48 and 52 mm are REVIEW under the simple rule. This is an illustrative policy, not a site-certified tolerance.

Report performance twice: all-item system performance including REVIEW/UNKNOWN, and conditional BIG/SMALL performance on automatically accepted items. High accuracy achieved by rejecting nearly everything is not an acceptable system without a minimum coverage requirement.

Keep separate threshold sets by crop and, where justified, variety or customer contract. Wrong-crop routing must never silently use another crop’s threshold.

# 9\. Test strategy and coverage levels

L0 Static checks: code linting, reproducible environment, pinned dependencies, configuration schema, unit tests, image decode safety, model checksum, calibration file validation.

L1 Algorithm unit tests: crop selection, segmentation masks, geometry functions, threshold equality, uncertainty propagation, rounding, missing values, confidence gating and deterministic replay.

L2 Bench metrology: certified reference objects and real vegetables at multiple field positions, rotations, heights and exposures; compare with physical measurements.

L3 Integration: camera trigger, item ID tracking, frame timestamps, conveyor encoder, model inference, actuator timing, bin sensors and persistent logs.

L4 Fault injection: dropped frames, camera disconnect, corrupted calibration, wrong model, slow inference, missing crop ID, light failure, power cycling, actuator jam and network/storage loss.

L5 Real-world blind test: held-out produce from unseen lots/days, measured by independent operators, with no threshold changes after unblinding.

L6 Shadow mode: operate on the real line without directing product; compare predicted bins with independently inspected outcomes across all shifts.

L7 Controlled production pilot: physical sorting with reject lane, manual oversight, documented stop conditions and lot-level reconciliation.

L8 Sustained production acceptance: continuous operation over the approved duration, monitored accuracy, throughput, availability, drift, rejections, safe recovery and change control.

# 10\. Detailed test-case catalog — vision and metrology

TC-V01 Nominal single centered potato: correct crop, single mask, physical dimension within approved tolerance, grade matches physical truth.

TC-V02 Smallest supported item: detected without disappearing into background or being filtered by minimum contour size.

TC-V03 Largest supported item: fully within field of view; no clipping or saturation; correct grade.

TC-V04 Threshold exactly equal to T: deterministic, documented tie policy; do not rely on float rounding.

TC-V05 Just below and above threshold: guard-band behavior correct; no unqualified confident label.

TC-V06 One item at each image corner and edge: metrology error within position-specific tolerance.

TC-V07 Rotations 0°, 45°, 90°, 135°: physical metric invariant to allowed orientation within tolerance.

TC-V08 Height variation within supported crop range: perspective/parallax error within tolerance or diverted.

TC-V09 Elongated okra/cucumber angled relative to image axes: length from major axis/contour, not naive axis-aligned box.

TC-V10 Curved chili/gourd: define arc length versus straight end-to-end length; implementation matches requirement.

TC-V11 Irregular sweet potato/potato: stable dimension under reorientation; no single-view overclaim about hidden 3D size.

TC-V12 Two separated items: two unique IDs, two masks, no merged measurement.

TC-V13 Touching items: reliably split or REVIEW; never treat merged contour as one huge vegetable.

TC-V14 Overlapping items: detect occlusion and reject or validated multi-view handling.

TC-V15 Partially outside frame: reject as incomplete, never auto-grade.

TC-V16 Stem, root, leaf tip and shadow: inclusion/exclusion follows crop measurement SOP.

TC-V17 Dirty, wet, dusty, glossy or damaged surfaces: correct segmentation or explicit REVIEW.

TC-V18 Similar-color belt/background: segmentation quality and false detections meet limits.

TC-V19 Background-only frame: no phantom item and no actuator firing.

TC-V20 Foreign object/hand/tool: UNKNOWN or safe stop; never assigned a vegetable grade.

TC-V21 Unsupported crop or wrong crop ID: no silent cross-crop threshold use.

TC-V22 Green leafy single leaf vs bunch vs overlapping pile: correct grading unit enforced.

TC-V23 Leafy greens with wilting/curling/compression: measurement consistency evaluated; divert if not observable.

TC-V24 Motion blur across speed range: meet dimensional error or reject, with latency accounted for.

TC-V25 Bright/dim and warm/cool illumination within spec: stable measurement and grade.

TC-V26 Light flicker, hard shadows, sunlight leakage: detect unacceptable conditions or meet tolerance.

TC-V27 Auto-exposure/white-balance drift: fixed settings or demonstrated robustness; no hidden threshold drift.

TC-V28 Image compression, noise, resolution change: unsupported input blocked; supported variation tested.

TC-V29 Camera moved by small known offset: calibration validity alarm and fail-safe operation.

TC-V30 Lens change or focus shift: recalibration required and verified before resuming.

TC-V31 Known reference artifact at multiple distances: measure true scale and field distortion.

TC-V32 Repeated captures of the same item: quantify repeatability, grade stability and boundary flipping.

TC-V33 Same item on different cameras: quantify cross-camera bias and transfer calibration.

TC-V34 Segmentation model low confidence: REVIEW with reason code, not forced BIG/SMALL.

TC-V35 Missing/corrupt image, NaN, infinity or negative measurement: safe error path and structured log.

TC-V36 Adversarial-but-real arrangements (stacked produce, mixed lots, belt debris): explicit support or safe reject.

# 11\. Detailed test-case catalog — system, safety and operations

TC-S01 Trigger-to-image alignment: no double capture or skipped item at minimum/maximum spacing.

TC-S02 Conveyor encoder and item tracking: prediction remains associated with correct physical vegetable.

TC-S03 Actuator timing at all supported speeds: commanded grade reaches correct physical bin.

TC-S04 Two closely spaced different grades: no cross-contamination or command swap.

TC-S05 Inference latency spike: bounded queue, no stale actuator command, safe divert.

TC-S06 Dropped frame or duplicate message: idempotent item IDs and fail-safe handling.

TC-S07 Camera unplugged/black frame: alarm and stop/divert, no assumed size.

TC-S08 Lighting controller failure: invalid frames rejected and alarm raised.

TC-S09 Calibration file missing, expired or mismatched to camera: automatic grading blocked.

TC-S10 Wrong model or threshold version: deployment blocked; version appears in every record.

TC-S11 Actuator stuck/jammed: downstream verification detects mismatch and stops line as specified.

TC-S12 Full bin, reject lane jam or product pileup: sensors and alarms prevent uncontrolled overflow.

TC-S13 Emergency stop and restart: no unintended actuation; tracked items reconciled.

TC-S14 Power loss/recovery: safe state, no corrupted event history, deliberate operator restart.

TC-S15 Network outage: defined local operation or safe stop; queued logs reconciled without duplication.

TC-S16 Storage full/DB unavailable: no silent loss of traceability; policy-driven safe action.

TC-S17 Clock drift between camera/PLC/edge computer: timestamps checked and synchronized.

TC-S18 Operator changes crop/threshold mid-lot: access control, audit log and controlled cutover.

TC-S19 Security: unauthorized config/model edits denied; signed or hashed artifacts verified.

TC-S20 Continuous shift test: morning, evening and night performance compared.

TC-S21 Temperature, vibration and dust over run duration: focus and calibration remain stable or alarms fire.

TC-S22 Cleaning/maintenance: equipment safe, calibration rechecked, operator sign-off captured.

TC-S23 Operator override: reason, item IDs and physical disposition logged.

TC-S24 Software rollback: restore last approved model/config without breaking item tracking.

TC-S25 End-of-lot reconciliation: input count \= BIG \+ SMALL \+ REVIEW \+ documented losses; discrepancies investigated.

TC-S26 Human safety and machinery guarding: approved site safety inspection and emergency-stop test; ML accuracy never replaces machine safety engineering.

# 12\. Metrics, formulas and statistical proof

Measurement: MAE \= mean(\|predicted\_size − physical\_ground\_truth\|); bias \= mean(predicted\_size − physical\_ground\_truth); RMSE \= sqrt(mean(squared\_error)); P95 absolute error; worst observed error; repeatability standard deviation; calibration drift over time. Report by crop, size bin, position, height, lighting, lot and shift.

Classification confusion matrix for items with unambiguous physical ground truth: true SMALL predicted SMALL/BIG/REVIEW; true BIG predicted SMALL/BIG/REVIEW. Report false-BIG rate, false-SMALL rate, precision/recall per grade, balanced accuracy and coverage. Explicitly state denominator and treatment of REVIEW.

End-to-end physical sorting: physical\_bin\_accuracy \= correctly routed items / all incoming eligible items; wrong-bin rate; reject/unknown rate; missed-item rate; throughput; P50/P95/P99 latency; uptime; mean time to recover; jam count; audit completeness.

Evaluate threshold-adjacent items separately (for example within a predeclared band around T). Also report all-item performance with representative size prevalence. Do not mask errors by pooling easy crops with difficult ones.

Use confidence intervals for error proportions (e.g., Wilson or exact binomial) and pre-register one-sided upper bounds where risk requires them. Approximate “rule of three”: if 0 errors in 300 independent trials, the 95% upper error bound is roughly 1%; if 0 in 3,000, roughly 0.1%. Independence and sampling assumptions matter; many video frames of one potato are not 3,000 independent vegetables.

Example sample planning for a binary failure with zero observed failures: n ≈ 3 / maximum\_acceptable\_failure\_rate for a one-sided 95% bound. This is a rough design aid, not a substitute for exact planning, subgroup coverage or a statistically justified acceptance test.

A claim of 99.9% end-to-end correctness needs far more independent items and stronger sampling than a 100-item demonstration. If the line sorts multiple crop classes, the critical class/condition must have adequate evidence; overall accuracy alone is not enough.

Model uncertainty calibration: evaluate whether high-confidence predictions are actually more reliable. Distinguish classification confidence from physical measurement uncertainty and mechanical-routing risk.

# 13\. Release gates — all must pass or have approved exception

G0 Requirements frozen. PASS: signed crop list, metric definitions, T, guard band, supported operating envelope, risk limits, reject policy, and site safety requirements. FAIL: any critical \[UNSPECIFIED\] value remains.

G1 Data governance. PASS: real-data manifest, licenses/permissions, hashes, split audit, no item/session leakage, sufficient coverage and locked blind set. FAIL: synthetic-only accuracy evidence or contaminated holdout.

G2 Ground-truth reliability. PASS: instrument calibration, measurement SOP, inter/intra-operator agreement and adjudication within approved limits. FAIL: label noise greater than allowed model error.

G3 Metrology. PASS: all crop-specific size MAE/P95/bias/repeatability and field-position checks meet preapproved values; uncertainty quantified. FAIL: mm-per-pixel guessed from arbitrary scene.

G4 Crop identification and image quality. PASS: wrong-crop, unknown, touching, occluded and low-quality cases handled to risk limits. FAIL: unsupported item confidently graded.

G5 Blind grade validation. PASS: crop-by-crop false-BIG/false-SMALL upper confidence bounds, accuracy and minimum coverage meet signed limits on untouched real data. FAIL: tuning after test unblinding.

G6 Boundary safety. PASS: guard-band decisions stable; borderline items routed to REVIEW at required rate; no unapproved tie behavior. FAIL: hidden boundary flips.

G7 Integration and physical sort. PASS: tracked item IDs, actuator timing and actual-bin accuracy meet signed limits at required speeds. FAIL: only offline image predictions were tested.

G8 Fault tolerance and machine safety. PASS: all critical faults cause approved safe state; emergency-stop, guards, interlocks, rollback and alarm escalation verified. FAIL: dangerous uncontrolled actuation.

G9 24/7 environmental endurance. PASS: representative multi-shift, multi-lot pilot meets drift, uptime, throughput and accuracy limits. FAIL: single short demo or one lighting condition only.

G10 Operations and observability. PASS: SOPs, calibration schedule, dashboard, audit logs, alerts, operator training, incident process and retention verified. FAIL: no way to detect degradation.

G11 Security/change control. PASS: controlled releases, signed config, least privilege, reproducible build, rollback and traceable threshold changes. FAIL: silent live configuration edits.

G12 Final acceptance. PASS: QA, metrology owner, line operations, product owner and safety authority sign GO; evidence pack archived; approved crops/sites/configurations explicitly listed. FAIL: missing signature or unresolved blocker.

Exception rule: no waiver for machinery safety or for absence of real-world proof of the claimed capability. Any approved non-safety exception must include scope, compensating control, expiration, owner and re-test date.

# 14\. Acceptance criteria worksheet — fill and approve before blind test

AC-01 Supported crop/variety and grading unit: \[UNSPECIFIED\]

AC-02 Ground-truth metric and threshold T per crop: \[UNSPECIFIED\]

AC-03 Measurement MAE ≤ \[UNSPECIFIED\] mm or g; P95 error ≤ \[UNSPECIFIED\]; bias within ±\[UNSPECIFIED\].

AC-04 False-BIG rate upper 95% confidence bound ≤ \[UNSPECIFIED\]; false-SMALL bound ≤ \[UNSPECIFIED\].

AC-05 Automatically graded coverage ≥ \[UNSPECIFIED\]%; REVIEW rate ≤ \[UNSPECIFIED\]% at the approved risk level.

AC-06 Physical wrong-bin rate upper confidence bound ≤ \[UNSPECIFIED\]; item miss/duplicate ≤ \[UNSPECIFIED\].

AC-07 Throughput ≥ \[UNSPECIFIED\] items/min; P99 end-to-end latency ≤ \[UNSPECIFIED\] ms; uptime ≥ \[UNSPECIFIED\]%.

AC-08 Camera calibration residual, reference-object error, shift drift and max permitted height variation: \[UNSPECIFIED\].

AC-09 Real test sample size, lots, farms, days, shifts, independent items and blind holdout plan: \[UNSPECIFIED\].

AC-10 Maximum allowed unsafe fault outcome: zero tolerated in mandatory fault-injection tests; site safety engineering approval required.

Optional planning illustration ONLY: a team might initially target ≤2 mm MAE on a particular rigid crop, ≥99% conditional grade accuracy away from the threshold and ≥95% coverage, but these values are not universal, are not verified, and must not be copied into release criteria without feasibility and risk review.

# 15\. Real-world pilot protocol and stop conditions

Phase A — Metrology bench: measure 20–30 calibrated reference shapes at multiple image locations and heights; verify geometric model. Then measure real crop specimens spanning the range. Do not proceed if calibration is unstable.

Phase B — Independent blind trial: seal the approved model and thresholds; collect and physically measure a pre-registered sample from unseen lots/days. One operator runs the line; another independently measures the ground truth; QA computes metrics without changing settings.

Phase C — Shadow line: process real produce and log decisions while humans or the existing sorter control routing. Include every shift and realistic faults. Review wrong-crop, uncertain, boundary and no-detection cases.

Phase D — Supervised physical pilot: enable sorting only for approved crops and conditions; route all uncertainty to reject; reconcile bins with ground truth; monitor per-lot quality and mechanical errors.

Phase E — Sustained acceptance: run continuously for the signed duration; log shift checks, calibration drift, alarms, cleaning, restarts, environment and operator interventions.

Stop immediately for: emergency-stop/guard failure; uncontrolled actuator; camera/calibration mismatch; wrong-crop auto-grading; item tracking mismatch; error/false-grade rates exceeding control limits; unexpected drop in coverage; repeated lost records; inability to route rejects; unapproved threshold change.

Recovery requires incident ID, root cause, corrective action, targeted regression tests and authorized restart. Never erase failed runs from the evidence pack.

# 16\. A realistic 24-hour engineering sprint — not a production certification

Hour 0–2: freeze one crop, one measurement metric, a provisional physical threshold and a safe REVIEW state; inventory camera/lighting/conveyor constraints.

Hour 2–4: mount fixed camera and diffuse light; lock exposure/focus; capture calibration grid and known-size reference objects.

Hour 4–6: implement image capture, distortion correction, background segmentation and connected-component filtering; store raw images and metadata.

Hour 6–8: implement physical dimension estimation, crop-specific grading, uncertainty band, and deterministic error handling.

Hour 8–10: physically measure a small real-vegetable development set; compare predicted dimensions, inspect residuals and fix obvious geometry bugs.

Hour 10–12: test corner positions, rotations, height changes, touching pieces, bad lighting and blurred images; enforce REVIEW where unsupported.

Hour 12–14: implement item IDs, camera/PLC trigger timing, logging, actuator simulation and reject logic; unit-test thresholds and boundary equality.

Hour 14–16: run controlled conveyor tests with manual truth and actual-bin reconciliation; measure latency and misses.

Hour 16–18: collect a fresh, separate real mini-holdout from another lot/session; run without tuning; produce confusion matrix and measurement errors.

Hour 18–20: fault-inject unplugged camera, missing calibration, jam simulation, power restart and storage failure; verify fail-safe behavior.

Hour 20–22: create operator calibration/cleaning checklist, monitoring dashboard and rollback instructions; document all unresolved risks.

Hour 22–24: assemble evidence pack, label result “PROTOTYPE / PILOT ONLY,” issue NO-GO for production unless all formal gates have independently passed.

A 24-hour sprint can produce a credible demonstrator and initial evidence. It cannot guarantee 100% accuracy or replace adequate multi-lot, multi-shift, statistically powered production qualification.

# 17\. Operator standard operating procedure — each shift

☐ Confirm approved crop, lot, grading unit, threshold version, model version and calibration ID displayed.

☐ Inspect lens, mount, light diffuser, belt cleanliness, reference target and camera focus.

☐ Run known-dimension check pieces at center and corners; record actual/predicted readings and pass/fail.

☐ Verify trigger, encoder, photoeye, conveyor speed, reject lane and actuator self-test.

☐ Send known SMALL, known BIG and borderline/invalid test objects; confirm correct physical destinations.

☐ Confirm clock synchronization, disk space, network status, event logging and alarm channels.

☐ Record ambient lighting, temperature, dust/water conditions and any maintenance since prior shift.

☐ Start lot only when all mandatory checks pass and an authorized operator signs.

☐ Sample and physically measure items during production at the preapproved frequency.

☐ At shift end reconcile item counts, exceptions, wrong bins, overrides, alarms and calibration drift.

☐ Stop or divert if any check fails; create incident and requalify before restart.

# 18\. Continuous monitoring, drift, maintenance and revalidation

Dashboard by crop, camera, line, lot and shift: input count; BIG/SMALL/REVIEW/UNKNOWN counts; confidence/uncertainty distribution; measurement distribution; crop mix; image brightness/blur; reference-object error; threshold margin; latency; frame drops; physical-bin mismatch; operator overrides; uptime and alarms.

Detect drift from new farms, seasons, varieties, lighting, dust, camera shifts, belt wear, model updates and calibration changes. A change in BIG/SMALL ratio alone may reflect real crop distribution, not model failure; investigate against physical sample checks.

Maintain a rolling real-ground-truth audit sample, enriched near thresholds and for rejected items. Use separate representative sampling to estimate live error rates without selection bias.

Change control: new crop, threshold, camera, light, conveyor speed, lens, model weights, segmentation logic or actuator timing triggers documented impact assessment and relevant regression gates. Major changes require new blind test and supervised pilot.

Rollback to last approved model/configuration on performance regression. Retain raw evidence for incidents according to policy; preserve reproducibility and privacy controls.

# 19\. Evidence pack and traceability templates

EVID-01 Requirements and approved acceptance criteria, signed and versioned.

EVID-02 Crop grading specification: crop/variety, physical metric, units, T, guard band, REVIEW rule, exclusions and approver.

EVID-03 Camera/lighting/PLC architecture, wiring, safety interlocks and bill of materials.

EVID-04 Calibration report: equipment serials, target dimensions, residuals, field-position map, uncertainty and expiry.

EVID-05 Ground-truth SOP, instrument certificates, Gage R\&R report and adjudication records.

EVID-06 Dataset manifest, raw hashes, split script, annotation guide, source/lot/shift coverage and leakage audit.

EVID-07 Versioned code, environment lockfile, model checksum, thresholds, config and build instructions.

EVID-08 Test-run log: run\_id, TC\_id, crop, lot, item count, build/config, environment, expected/observed, PASS/FAIL/NOT RUN, defect link and tester.

EVID-09 Blind validation report: full confusion matrix, measurement metrics, confidence intervals, per-crop/condition slices, exclusions and raw predictions.

EVID-10 Integration report: latency distributions, item tracking, mechanical bin counts, actuator verification and fault injection.

EVID-11 Pilot report: start/end, lots, shifts, uptime, drift, overrides, incidents, recovery, recalibration and signed observations.

EVID-12 Release decision: gate matrix, unresolved defects, exceptions, risk acceptance, named approvers and rollback version.

Suggested test-run row: RUN-001 \| TC-V05 \| crop=potato \| lot=L07 \| item=I042 \| truth=49.3 mm \| prediction=49.8 mm \| T=50.0 mm \| U=1.2 mm \| expected=REVIEW \| actual=REVIEW \| PASS \| camera=C1 \| calibration=CAL-03 \| build=B17 \| tester=\[NAME\].

The row above is a format example with invented illustrative numbers, not a record of an executed test.

# 20\. Defect severity and triage rules

P0 CRITICAL: machine safety hazard, uncontrolled actuator, wrong-item routing from tracking mismatch, calibration bypass or systematic dangerous misclassification. Stop line; no waiver.

P1 HIGH: breach of false-grade acceptance limit, undetected crop confusion, missing audit trail, frequent wrong-bin sort, sustained metrology drift. Block production and require root-cause analysis.

P2 MEDIUM: excessive REVIEW, noncritical performance degradation, intermittent recoverable frame drops, operator usability defects. Fix before acceptance if coverage/uptime targets are affected.

P3 LOW: cosmetic dashboard issue or documentation typo with no safety/quality impact. Track with owner and due date.

Every defect: reproducible steps, raw item IDs/images, environment, expected/actual, severity, owner, fix commit, regression test and closure evidence.

# 21\. Pseudocode contract for safe grade decisions

INPUT: image, crop\_id, camera\_id, timestamp, calibration\_version, approved\_thresholds.

ASSERT camera/config/calibration match; otherwise SAFE\_REJECT("CONFIG\_INVALID").

DETECT and SEGMENT candidate instances; if image invalid, crop unknown, occluded, merged or out of frame: SAFE\_REJECT(reason).

FOR each isolated item: compute physical size using validated crop metric and camera geometry; compute uncertainty U; verify supported size, height and image quality.

LOOKUP T for exact crop/variety/contract version; if missing: SAFE\_REJECT("THRESHOLD\_MISSING").

IF size \+ U \< T: decision \= SMALL.

ELSE IF size − U \> T: decision \= BIG.

ELSE: decision \= REVIEW.

WRITE immutable event with item\_id, crop, size, U, T, model/calibration versions and decision.

TRACK item through conveyor; verify actuator command and actual physical destination; on mismatch alarm and stop/divert.

Important: This is logic-level pseudocode, not production-ready control software. Safety PLC/interlocks and verified hardware timing remain mandatory.

# 22\. Final GO / NO-GO release checklist

☐ G0 All crop definitions, requirements, thresholds, uncertainty bands and risk limits signed.

☐ G1 Independent real-data holdout collected, sealed and leakage-free; synthetic-only claim removed.

☐ G2 Physical measurement instruments and ground-truth reliability qualified.

☐ G3 Camera calibration and metrology pass over full operating envelope.

☐ G4 Crop identification, unknowns, occlusions and leafy-unit definitions validated.

☐ G5 Per-crop blind grade error bounds and minimum coverage meet approved limits.

☐ G6 Borderline decisions and REVIEW routing verified.

☐ G7 Physical sorting and tracking verified at required throughput and latency.

☐ G8 Safety interlocks, fault injection, fail-safe, rollback and recovery pass.

☐ G9 Multi-lot/multi-shift endurance and drift controls pass.

☐ G10 Operator SOP, monitoring, audit logging, incident response and maintenance in place.

☐ G11 Configuration security and change management verified.

☐ G12 QA, metrology, operations, product and safety authorities sign final approval.

Release verdict: \[NO-GO — pending real-world validation and completed criteria\]

Authorized crops/sites/builds: \[NONE YET APPROVED\]

Open critical issues: \[TO BE POPULATED\]

Evidence pack location: \[UNSPECIFIED\]

QA signature/date: \[UNSIGNED\] \| Metrology signature/date: \[UNSIGNED\] \| Operations signature/date: \[UNSIGNED\] \| Product signature/date: \[UNSIGNED\] \| Safety signature/date: \[UNSIGNED\]

# 23\. Reference and verification register

This document is a practical validation specification, not a peer-reviewed literature review. The blocked original Deep Research report was not available for source verification. Do not treat any paper, industrial performance claim or grading standard as cited unless the original source has been checked.

Source families to consult during design sign-off: applicable customer grading specifications and Indian AGMARK/BIS or commodity-specific standards where relevant; ISO/IEC 17025 for calibration laboratory competence; ISO 5725 for accuracy/trueness/precision concepts; ISO 12100 for machinery risk assessment; IEC 60204-1 for machinery electrical equipment; ISO 13849-1 or IEC 62061 for safety-related controls, as applicable to the machine and jurisdiction.

For machine vision and statistical methods, review official OpenCV camera calibration documentation; recognized metrology uncertainty guidance such as the JCGM Guide to the Expression of Uncertainty in Measurement; and suitable binomial confidence-interval literature. Applicability and edition must be confirmed by qualified site engineers.

For potato and sweet-potato sorter benchmarks, retrieve actual primary studies, patents, OEM technical specifications and test conditions before citing measured performance. Vendor marketing accuracy claims are not interchangeable with independently verified end-to-end physical-bin accuracy.

# 24\. Document usage and sign-off

This document becomes an authoritative project control document only after the responsible team completes all \[UNSPECIFIED\] fields, approves the numerical acceptance criteria before testing, executes the tests, attaches evidence and signs the final gate. Until then it is a comprehensive test plan, not a certification or proof that the current system works in production.

End of document.

