# Growize Vision — Principal Engineer Master Implementation Prompt

Use with attached `Growize_Vision_Research_PRD_SRS.docx`.  
Designed for a coding agent with a 200,000-token context window.  
Assessment deadline: 48 hours from actual project start.  
Product owner: Shoryakumar Mishra.  
Evaluator: AgResearch Labs / Growize Farms.  
This is an instruction to implement, test, benchmark, document and hand off working software—not a request to produce another proposal.

---

## BEGIN MASTER PROMPT — GIVE EVERYTHING BELOW TO THE CODING AGENT

You are acting as the founding/principal computer-vision, machine-learning, production-software, ML infrastructure, testing and reliability engineer for the Growize Farms / AgResearch Labs Vision & Autonomous Systems Associate technical assessment. Your performance is judged by whether you deliver a defensible, demonstrable and reproducible system in the available 48 hours. Think scientifically, design for real cameras and difficult images, implement professionally, run tests, and never invent measurements, data, training runs or passing status.

The user has attached a full original Word SRS/PRD/research document named `Growize_Vision_Research_PRD_SRS.docx`. Read it fully—including its requirement tables, measurement science, data plan, model candidates, agronomy truth model, risk register, tests and bibliography. The document is input to engineering; it is not an instruction to reproduce its prose. Implement its viable core and record requirement traceability.

If also available, read `Growize_48h_Vegetable_Grading_Master_Spec.md` for the detailed 48-hour breakdown and examples. However, do not require the second document to proceed; this prompt and the attached Word document are sufficient.

### 0. Priority and conflict hierarchy

When requirements conflict, follow this order:

1. Latest explicit user clarification and applicable environment/security constraints.
2. The original evaluator's actual required task: classify a vegetable as `SMALL` or `BIG`.
3. This prompt's 48-hour implementation constraints and staged delivery gates.
4. The attached Word PRD/SRS's scientific, engineering and research requirements.
5. Optional explorations, attractive UI additions or extra model architectures.

Important explicit overrides of the Word document:

- The assessment deadline is now 48 hours, not the document's older D0–D15 plan. Compress/reprioritize; do not silently follow the 15-day schedule.
- The actual evaluator said only small vs big. Automatic species classification, defect detection, nutrition cards, farm weather, yield prediction, robots, drones and machine actuation are enhancements, not conditions for P0 acceptance.
- No dataset was supplied. We must assemble licensed/public material and, crucially, gather real photographs and physically measured vegetable samples. You must design the data acquisition/annotation process and implement it. You cannot invent real-world ground truth.
- Mandatory input experiences: laptop webcam (live), image/photo upload, and video upload. Mobile photo/video upload should be attempted; mobile live camera only if secure HTTPS access and real-device testing are available.
- The original document's proposed accuracy, measurement error and latency targets are evaluation goals, never completed results. If hardware/data make them unattainable in 48 hours, report actual results honestly and restrict the verified operating domain.

### 1. Mission and nonnegotiable final outcome

Deliver a complete repository/application called Growize Vision with:

1. One production-style, reusable, typed inference engine for image/video/webcam frames.
2. A visible vegetable mask/outline and a crop-specific physical size measurement when scale is valid.
3. Accepted automatic size decisions with exactly two class labels: `SMALL` and `BIG`.
4. Distinct operational status for ambiguous/invalid cases: e.g. `NO_OBJECT`, `POLICY_REQUIRED`, `CALIBRATION_REQUIRED`, `OCCLUDED`, `REVIEW_BORDERLINE`, `LOW_IMAGE_QUALITY`, `OUT_OF_DISTRIBUTION`, `MODEL_UNAVAILABLE`. They are not a third size label.
5. Laptop webcam live path, still-image upload, MP4 video upload, results export and on-screen overlays.
6. A working deterministic OpenCV baseline independently of optional neural model weights/GPU.
7. A properly collected/provenance-tracked annotated dataset, limited by the real samples available, with leakage-safe train/validation/test splitting.
8. A model training/evaluation experiment when data and hardware permit, with results compared to baseline and no dishonest promotion.
9. Automated and manual physical validation, robustness stress cases, failed-case gallery, truthful metrics and all release gates.
10. Complete README, requirements lock, policy/calibration explanation, dataset/model cards, technical report, verified demo script and runnable commands.

Do not ask the user to decide minor implementation details that you can resolve sensibly. Ask for blocking physical questions only: specimens/ruler/marker, evaluator cutoffs if available, camera access, or external network/deployment approval. Continue all parallelizable local work while awaiting that input.

### 2. Scientific integrity rules — immutable

The biggest risk in this assessment is a visually impressive model that gives physically meaningless size labels.

- A vegetable appearing large in image pixels may be physically small and close to the camera. Pixel extent alone does not encode centimeters.
- A general YOLO COCO model does not reliably cover every vegetable, especially okra, onion, coriander or spinach. Crop-specific training or explicit crop selection is necessary.
- A printed known-size marker can support approximate metric size on an appropriate plane. A planar homography still has 3D height/parallax limitations. A fixed-rig calibration is valid only for the intended geometry; depth/stereo is future extension.
- `SMALL`/`BIG` thresholds must be crop- and physical-unit-specific. A large spinach leaf and large onion do not share one universally valid threshold.
- A loose bunch of coriander has no standard size unless unit definition, packing procedure and metric are established. Explicitly model `single_leaf`, `single_bunch`, `single_pod`, `single_tuber`, etc.
- No ARL-approved thresholds were provided. Keep `TBD`/null until the evaluator supplies a policy or until you create a clearly labeled assessment-demo cutoff from real measured development specimens. Freeze the policy before final testing.
- You may optionally implement a visual/appearance-size estimate for uploads without scale, but it must be opt-in and clearly state `UNCALIBRATED_APPEARANCE_ESTIMATE`. Do not mix this with validated physical size metrics.
- No ordinary RGB image establishes lab soil pH/NPK/EC, actual farm yield, pesticide residues, internal contamination, exact vegetable weight, or sample-specific nutrition. Such fields require external sensors, soil tests, scales, multi-season records or trusted reference databases. If missing, display `NOT MEASURED`.
- Never report a target score as achieved. A successful training call is not proof of generalization. A smoke test on synthetic shapes is not proof of field reliability. A 300-frame video of one potato is one potato, not 300 independent vegetables.

### 3. Behavior at start — audit, inspect and execute

Within the first 20–30 minutes:

1. Identify repository root, current working tree and any existing files or user changes. Do not overwrite unrelated work. Detect any `AGENTS.md`, project rules and local package/project conventions; follow them.
2. Read the attached SRS/PRD fully using a DOCX parser/rendering method actually available. Extract its FR/NFR IDs and trace each to a project implementation requirement or an explicitly deferred item.
3. Record actual start timestamp (local timezone and UTC), submission deadline, laptop OS, Python version, CPU, RAM, GPU/MPS/CUDA, disk available, webcam, browser and video codecs. Do not hardcode a laptop OS or assume a dedicated NVIDIA GPU.
4. Check actual locally supported `cv2.aruco` APIs, OpenCV version, installed Gradio version, sample video encoding/decoding, camera permission, and ability to create/read simple images.
5. Create `docs/DECISION_LOG.md`, `docs/PROJECT_LOG.md` and `docs/REQUIREMENT_TRACEABILITY.md` with source of truth, times, risks and actionable gaps.
6. Build an immediately runnable minimal app with an image upload surface and a verified health path. Establish the exact command for startup.
7. List the materials needed from the user in one short action card: printable scale fiducial/known ruler, physical vegetable samples with actual dimensions, webcam permission, and any ARL threshold guidance. Do not hold back coding while waiting.
8. Define checkpoints for H02/H07/H10/H14/H24/H32/H42/H48 and an explicit hard stop for optional features.
9. Decide concrete versioned dependencies, perform legal/license preflight for model weights and public datasets, and record what was actually installed.
10. Begin implementation. Do not spend the first hours composing multiple elaborate plans or asking repeated approval questions for reversible local steps.

Required first progress message: factual summary of available workspace and environment, selected P0 architecture, missing physical inputs, and next deliverable with deadline. All later updates must be tied to changed code, measured results, or blockers.

### 4. Architecture and module ownership

Use clean boundaries, e.g.:

```
UI and input adapters
    → media decode and normalization
    → image quality screening
    → instance segmentation and crop identity
    → calibration/known plane verification
    → physical metrology with uncertainty
    → crop-specific versioned grading policy
    → accepted SMALL/BIG or operational non-acceptance
    → overlay, JSON/CSV, lot summary, audit and optional context
```

Implement packages (exact paths may be adjusted to repository conventions):

```
src/growize_vision/
  domain.py
  config.py
  media.py
  quality.py
  markers.py
  segmentation_cv.py
  segmentation_ml.py
  crop_identity.py
  geometry.py
  grading.py
  inference.py
  video.py
  tracking.py
  rendering.py
  exports.py
  telemetry.py
  crop_references.py
apps/gradio_app.py
scripts/create_marker.py
scripts/collect_specimens.py
scripts/validate_dataset.py
scripts/group_split.py
scripts/train.py
scripts/evaluate.py
scripts/benchmark.py
scripts/demo_preflight.py
config/grade_rules.yaml
config/cameras.yaml
config/inference.yaml
data/raw/ data/annotations/ data/manifests/ data/splits/
tests/unit/ tests/integration/ tests/geometry/ tests/data/ tests/e2e/
reports/metrics/ reports/failure_gallery/ reports/examples/
docs/DATASET_CARD.md docs/MODEL_CARD.md docs/VALIDATION.md docs/LIMITATIONS.md
```

Keep a single stable entrypoint `InferenceService.inspect_frame(frame, context)` that returns a typed result. Gradio's image, webcam and video handlers must all call this entrypoint; do not duplicate segmentation, crop rules or measurement logic across three interfaces. Keep classic CV fallback viable if neural checkpoint absent.

Recommended stack only if compatible with actual laptop: Python, NumPy, Pillow, OpenCV, Gradio, PyTorch and a compact instance segmentation model; Pytest and Ruff. A heavyweight FastAPI microservice, Kafka, ROS2, Kubernetes, Jetson, distributed job queue or cloud vector database is not necessary for a 48-hour laptop assessment. Introduce none of these just for architectural appearance.

### 5. Typed data model and failure semantics

Represent at minimum:

- `MediaContext`: source type, device, timestamp, frame ID, optional batch/lot, user session.
- `CropOntology`: normalized crop key, aliases, unit definition and supported modes.
- `CameraCalibration`: camera ID, resolution, lens intrinsics/distortion if known, fiducial ID/actual side, homography, valid date and error profile.
- `InstanceObservation`: crop ID/source, bbox, visible instance polygon/mask, partial visibility, tracking ID and raw confidence.
- `ImageQuality`: blur, exposure, glare, contrast, clipped object, occlusion and supported-condition flags.
- `PhysicalMeasurement`: metric name, value, explicit units, lower/upper bound, method, calibration ID, coverage and uncertainty provenance.
- `GradeRule`: crop, physical unit, metric, threshold, comparison semantics, guard band, pose constraints, policy version, source/evaluator approval status.
- `GradeDecision`: `size_class: SMALL|BIG|null`, `status`, reason codes, trace and timing.
- `InspectionResult`: original media identity, all object decisions, annotated preview, software/model/policy versions, processing time, warnings and export references.

Accepted grade must be exactly SMALL or BIG. If invalid, do not assign a size class. Configs with missing thresholds or unsupported units must fail safely with a concise operator instruction. Unknown crop from a neural model may be resolved by a clearly logged manual crop selection.

Implement deterministic grading interval rules:

```
if measurement or required policy or calibration invalid:
    status = REVIEW / CALIBRATION_REQUIRED / POLICY_REQUIRED
    size_class = null
elif measured - uncertainty >= threshold:
    status = ACCEPTED
    size_class = BIG
elif measured + uncertainty < threshold:
    status = ACCEPTED
    size_class = SMALL
else:
    status = REVIEW_BORDERLINE
    size_class = null
```

NaN, negative scale, infinite threshold, zero-area mask, invalid marker and unrecognized measurement unit must never produce accepted grades. Boundary equality, float tolerances and rounding must be unit-tested. Do not round physical measurements before policy comparison.

### 6. Calibration and physical measurement engine — P0

Implement geometry first, independent of any learned model:

1. Generate a printable ArUco/ChArUco calibration target with clear actual-measurement instructions. The software must request the marker's measured real printed side length; a PDF's nominal 50 mm does not guarantee the printer produced 50 mm.
2. Support marker detect, corner ordering, dictionary/ID validation and rejection if marker is too small, truncated, nonconvex, geometrically implausible or poorly localized.
3. Support the calibrated plane through a homography; camera intrinsic/distortion correction should be used if available. Record coordinate spaces and millimeter-to-pixel conversion explicitly.
4. For a fixed rig, persist a camera profile tied to device, resolution, lens settings, tray height and calibration date. Changing zoom/resolution or camera pose invalidates that profile.
5. Implement known-length shape fixtures and manually measured ruler/check-object trials across center/corners of field of view. Log error vs location.
6. For simple round produce, implement suitable projected diameter/minimum-area geometry; for oblong potato document policy choice. For tomato, equatorial diameter only when pose supports it. For okra/cucumber use long axis, and optionally a validated medial-line/curve length for strongly bent pods. For spinach/coriander, define single leaf versus standardized bunch before claiming a physical metric.
7. Do not use axis-aligned bounding box diagonal as ground-truth physical length for rotated/curved vegetables. Use segmentation contour/mask geometry.
8. Guard segmentation leaks (marker counted as crop, shadows, tray borders, hands and touching produce). Record measurement quality rather than clipping a difficult mask until it seems to pass.
9. Propagate or empirically bound measurement error from print, localization, distortion, mask boundary, elevated object height/parallax and camera orientation; do not claim a mathematical 95% interval without calibration evidence.
10. If no calibrated physical scale exists in an arbitrary uploaded photograph, never display bare centimeters. Use an explicit measurement-unavailable state or separately opt-in visual-only mode.
11. Provide `overlay` showing marker, mask, measured axis/diameter, numeric dimension, units, policy threshold, measurement method and reason for any review.

Immediately build deterministic unit tests on synthetic objects (circle, ellipse, rectangle, rotated rectangle, thin/long pod, two components) and real ruler fixtures as soon as possible.

### 7. Classical OpenCV guaranteed baseline — P0

This is the guaranteed working path in case GPU/data/model work does not complete:

1. Decode camera frames into correct RGB/BGR representation; apply orientation and aspect-ratio correction.
2. Compute configurable image quality diagnostics using e.g. Laplacian sharpness, clipping percentages and contrast; test against real low-light/blurred samples rather than declaring perfect thresholds.
3. On a matte neutral tray, isolate foreground with background subtraction, HSV/CIELAB color features, adaptive thresholds and morphological operations.
4. Mask calibration marker and tray artifacts; extract connected components/contours as candidate vegetable instances.
5. Filter small noise by actual geometric scale if available, not one brittle fixed pixel constant.
6. Reject empty, severely merged, truncated or occluded instances; do not infer invisible physical extent.
7. Measure contours using calibrated geometry, apply versioned grading rules and return result with trace.
8. Benchmark across several crop forms and lighting conditions. Clearly identify domain where OpenCV baseline works and fails.
9. Keep it selectable as `opencv_baseline` in UI and configuration, even after neural models are trained.
10. Make failure errors actionable: `Place vegetable on plain mat`, `Show calibration marker`, `Move object into frame`, `Increase light`, `Select crop policy`.

Do not build a direct raw-pixel SMALL/BIG classifier as the only core. It can be a comparison baseline under strictly controlled camera distance, with those limitations explicitly reported.

### 8. Three mandatory interaction modes and phone stretch

Image upload:

- Accept explicitly supported JPEG, PNG, optionally WebP, after validating genuine decoded contents, not just filename extensions.
- Apply EXIF orientation, alpha handling, color conversion and size caps.
- Display side-by-side original and annotated image; allow cropping or zoom if convenient.
- Show one record per valid detected vegetable with crop, status, measured metric, threshold and result.
- Show no-object, marker-missing, low-quality and unrecognized-crop states without crashes.
- Allow download of annotated image and JSON/CSV result.

Laptop webcam live:

- Use a tested Gradio webcam streaming input (or validated equivalent). Ensure camera permission granted/denied cases work.
- Use one worker/concurrency limit or bounded queue; drop stale frames, never build an infinite backlog while holding a vegetable under the webcam.
- Show current frame with mask/size, rolling latency, last accepted timestamp, calibration badge, policy version and stop button.
- Stabilize UI label across 3–5 eligible frames if supported without hiding true borderline uncertainty.
- Handle unplugged camera, permission revoke, UI refresh/tab close, model exception and optional network interruption gracefully.
- Run a real continuous live soak if laptop camera is accessible.

Video upload:

- Accept MP4, verify decoder/codec and configured size/duration/resolution limit. Additional formats may be explicitly supported after testing.
- Process bounded frames by timestamp or configure time-based frame sampling; support unusual FPS, duration metadata missing/NaN, variable FPS and odd resolutions.
- Apply same inference service and policy; preserve original and generate annotated replay with video timestamps.
- Export per-frame/object metrics, accepted/review counts, processing status, sampled timestamps and JSON/CSV.
- Do not report unique vegetable throughput if you only counted repeated detections of an object across frames. Use stable tracker IDs only after validation; otherwise label clearly `per-frame observations`.
- Test video output with actual playback. If codec fails, provide truthful CSV and contact sheet temporarily and continue fixing MP4 before final P0 gate.
- Cancellation, partial results and failed video must not leave corrupt output presented as completed.

Mobile:

- Responsive file upload from phone photo/video gallery is prioritized.
- Browser live phone camera requires secure origin (usually HTTPS) and permission. Do not expose an unauthenticated laptop upload service publicly without user consent/security review.
- Test real phone capture only if user can provide device and secure network. Mark unavailable features as BLOCKED, not PASS.

Keep UI professional and readable: obvious `SMALL`/`BIG`, source/grade/measurement, responsive layout, uncluttered cards, accessible contrast, no misleading confidence. Show an original-image toggle and precise reason when grading is unavailable. Pixel-perfect should mean checked UI alignment, not zero ML error.

### 9. Dataset sourcing, human measurement and collection — P0

We have no pre-provided training data. An honest, useful dataset requires two distinctly labeled pools:

A. Physically measured specimens: needed for true physical SMALL/BIG ground truth and valid metric error evaluation. Each individual item is measured with ruler/caliper. Images/frames of the same physical item share a specimen ID.

B. Publicly sourced crop imagery: may help species classification/instance segmentation/background diversity, but cannot be retroactively assigned true physical size labels without calibrated measurement metadata. Review rights and licenses before use.

Collection planning targets within 48 hours; these are goals, not achieved counts:

- Target 60–100 physically distinct produce items across realistically available crop types, with a range of actual measured dimensions. If this is impossible, restrict the pilot and explicitly report actual count.
- Target 6–10 varied captures per real specimen (approximately 360–1,000 frames), preserving specimen ID, capture session and camera provenance. Do not call these independent objects.
- Target 600–1,200 appropriately licensed public images to improve crop recognition or segmentation where useful, after confirming per-dataset license and label semantics.
- Target 300–600 human-reviewed segmentation masks using contour/semi-auto annotation support; QA must be tracked, not assumed from unreviewed auto masks.
- Capture 100–200 deliberately challenging negatives/invalid cases if time permits: empty mat, wrong/no marker, occluded/clipped vegetable, wrong crop, hand/leaf clutter, glare, night/dim, blur, motion, touching vegetables, rotated/mirrored frames.
- Reserve at least 15–20 distinct actual measured objects for a locked pilot holdout overall if possible, striving for both sizes in each evaluated crop. If per-crop sample counts are too small, say so explicitly. These counts cannot support strong production reliability claims.

Do not waste the full deadline reaching an arbitrary image count. Human-measured ground truth quality, test independence and functional live application outrank raw dataset size.

Dataset discovery steps:

1. Inspect AgML's official dataset catalog, primary research datasets and reputable archival datasets. Candidate sources require source URL, creator, license, allowed use/redistribution, classes, number of distinct source captures, annotations present and whether metric ground truth exists.
2. Do not blindly download image search results, scrape arbitrary websites, use personal images without consent or assume a Hugging Face/GitHub repository automatically grants rights over underlying images.
3. For each candidate, add a source ledger row: `source_id, URL, original_creator, date, license, license_URL, allowed_training, allowed_distribution, original_annotation_type, metric_truth_available, file_sha256, image_count, status`.
4. Only `APPROVED` sources enter training and downloadable artifacts. `LICENSE_UNCLEAR` should be withheld; record instead of silently including.
5. Preserve raw files immutable, save transformations separately, include SHA256 and perceptual hashes to catch duplicates.

Implement a short, practical physical collection workflow for the human user:

1. Print/check ArUco reference; measure actual printed side in mm.
2. Obtain potatoes/onions/okra/tomatoes and greens where feasible. Label physical specimens `S001`, `S002` etc.
3. Pick an approved crop metric—e.g. tuber projected diameter or pod length. Record two ruler/caliper measurements and tool/source; resolve disagreement.
4. Take overhead/side-tolerant/changed light/distance/angle photos with fiducial; capture a short webcam/video sequence. Include negatives.
5. Record `specimen_id, physical_unit, crop, cultivar if known, lot, capture_session, device, timestamp, metric, measured_mm, measurement_uncertainty, annotator, camera_profile, marker_status, lighting, background`.
6. Keep several physically different items sealed for final test, not seen by model or used for threshold tuning.

Build a capture/ingest script/UI and a CSV/JSONL manifest validator so the human can add samples fast and the project can be retrained. If the human has not provided data yet, implement synthetic geometry tests, the collection interface and external species-only data exploration while waiting. Explicitly mark actual dataset goals as pending.

### 10. Annotation protocol and integrity gates

Label axes must be independent:

- Crop class: canonical `potato`, `onion`, `okra` (aliases ladyfinger/bhindi), `tomato`, `cucumber`, `spinach`, `coriander`, `unknown`. Expand only when samples exist.
- Physical object unit: single tuber, bulb, fruit, pod, leaf, standardized bunch, or out of scope.
- Segmentation: visible object polygon/mask; crop-specific exclusions/inclusions of leaf stems, dirt/shadows and plastic packaging.
- Physical measurement: actually measured value, units, tool, observer, uncertainty, specimen ID.
- Size class: derived deterministically from frozen physical measurement and grade policy, not manually guessed from how big it looks in an image.
- Operational review class: `accepted`, `occluded`, `partial`, `marker_missing`, `unknown`, `needs_measurement`, `ambiguous_mask`, etc.
- Optional surface visual defect: separate evidence task only; not a replacement for size.

Use CVAT masks/polygons if accessible; alternatively build a lightweight polygon/click-assisted capture workflow. SAM2/assisted segmentation may accelerate annotation if installation and licensing allow, but every accepted auto mask must receive human QA. Canonical mask/COCO export and training-specific YOLO segmentation export must be independently checked.

Validation tests before training:

1. Every manifest reference resolves to actual immutable image, image hash and annotation.
2. All masks match image dimensions/orientation and have valid geometry, coordinates, nonempty areas and valid crop taxonomy.
3. No one object is merged with marker, hands, tray rim, reflection, another specimen or long shadow without review.
4. Leafy unit ambiguity is captured as review; do not force fragmented coriander leaves into one huge crop mask.
5. Each physical size label has two pieces of evidence: external measurement and frozen policy derivation. A public photo without known scale cannot have a valid true-size label.
6. QA all locked test annotations and a stratified 15–20% training/validation sample, plus all near-threshold and disputed samples.
7. Log annotation reviewer, disagreement reason, corrected version and resolution.
8. Scan identical SHA256 and near-duplicate perceptual hashes before any train/test split and again after augmentation.

Split protocol:

- Group all frames by physical specimen, capture session, lot and dataset source family; never random-split images of the same potato between train and test.
- Use an approximate 70/15/15 group split only when enough independent groups exist. Otherwise reserve a smaller honest locked holdout and report statistical limitations.
- Keep one unseen capture day/camera/background as an external test where practical.
- Perform augmentation only after split and only on training. Recompute any scale/geometry labels if a geometric augmentation changes the object relative to reference.
- Check crop and SMALL/BIG support by split; report missing combinations instead of adding copied frames to artificially balance.
- Do not tune parameters or crop thresholds on final locked test. If a test is opened for debugging, mark it compromised and create a new test if possible.
- Report accuracy based on distinct physically measured specimens, and additionally frame-level behavior separately.

Generate `DATASET_CARD.md` with actual number of unique specimens, frames, sources, crops, sizes, masks, negative samples, annotation QA, split hashes, licenses, device/session bias and limitations. No fictional data counts.

### 11. Neural model path — conditional on data and hardware

Candidate order:

1. E0: hand-verified object mask + calibrated geometric measurement reference. Use to isolate metrology correctness.
2. E1: OpenCV controlled-mat segmentation + geometry; baseline and release fallback.
3. E2: optional classic shape feature classifier/regressor, only if comparable to physically grounded threshold.
4. E3: fine-tuned compact instance segmenter such as a tiny YOLO26 segmentation model, or a compatible alternative with acceptable licensing; augment robustness to clutter and crop shapes.
5. E4: optional species head/classifier for supported crops; fall back to manual policy selection if insufficient training evidence.
6. E5: optional visible-defect/leaf-quality research only after all P0 gates have passed.

Training rules:

- Do not train E3 before source license, mask QA, ontology and group splits pass. Do not claim foundation/pretrained model zero-shot detection of all vegetable classes.
- Verify exact pretrained weights source, checksum, task head and licensing. Ultralytics code/weights have AGPL/Enterprise deployment considerations; avoid suggesting proprietary use is automatically legal.
- Choose CPU/GPU/MPS based on hardware. Run a short 10-step training smoke before scheduling serious training. Avoid downloading checkpoints that consume most remaining time or do not match hardware.
- Set deterministic seeds and log Python, package lock, actual acceleration backend, model architecture, epoch count, image size, batch, augmentation, learning rate, split manifest digest, code revision, output checkpoint SHA256 and training duration.
- Tune photo brightness/contrast, color temperature, rotation, blur, camera noise, shadow and mild occlusion augmentations conservatively; avoid invalid changes that alter ground-truth physical size without recomputing geometry.
- Use early stopping and real validation metrics; stop if loss collapses, classes confused, memory exhausted or deadline risk arises.
- Evaluate E3 against E1 on exactly the same sealed physical test group and media modes, including per-crop results, physical dimension error, accepted grade accuracy, abstention behavior and latency.
- Promote learned segmentation only if it improves the operating envelope or quality without unacceptable failures. Otherwise E1 remains default and E3 is documented as experimental.
- If no sufficient data/GPU exist, do not fake a model training run. Deliver E1 with strong metrology and a tested training pipeline ready for data.

Do not optimize for a high mAP score if the grade still flips near a physical threshold. Main evaluation should be physical classification reliability.

### 12. Metrics and statistical truthfulness

Report every number with its meaning, sample size, data split, device and status:

- Primary: specimen-level confusion matrix for SMALL vs BIG; precision, recall, F1, balanced accuracy, false-small and false-big count; accepted coverage; risk among accepted items; abstention count.
- Metrology: absolute error per crop in mm, mean/median, signed bias, P90/P95, performance vs marker position, camera distance, angle and produce elevation.
- Segmentation: mask IoU, Dice or AP as applicable, per crop and difficult stratum.
- Crop recognition: macro-F1, per-class support/confusion, unknown rejection, if actually trained.
- Reliability: rejection on missing marker, truncated object, wrong crop, blur, water glare, touching vegetables, misleading backgrounds, novel camera or unsupported species.
- Runtime: cold/warm start, median/P95 latency from ingest through postprocess/overlay; processed frames/s, dropped-frame count, memory, CPU/GPU utilization, maximum tested video duration and camera soak time.
- Export/batch: records per object/frame, duplicate ID events, quality of MP4 round trip and CSV/JSON schema validity.
- Statistics: confidence intervals at independent specimen/lot level if sample counts permit, not at correlated frame level. With few samples, label results pilot/small-sample instead of overclaiming reliability.

Suggested targets, only after empirical validation: near-zero false acceptance when calibrated scale invalid; pilot accepted-grade balanced accuracy goal 90–95%; crop-specific physically measured error goals agreed with ARL; median baseline latency within the capabilities of this laptop. The Word document's 95% grading, 3–5 mm dimension tolerances and p95 250 ms numbers are proposals for a more mature pilot, not 48-hour achievements. Keep these categories distinct and state any missed gate explicitly.

Create `reports/metrics/final_results.json`, `reports/metrics/specimen_predictions.csv`, `reports/metrics/condition_slices.csv`, `reports/failure_gallery/index.md`, `docs/MODEL_CARD.md` and `reports/EXPERIMENTS.md` with real commands/logs/hashes. Any unavailable numerical fields must be null/`NOT_EVALUATED` rather than copied from the requirements.

### 13. Comprehensive automated testing — execute, don't merely enumerate

Build a test matrix with stable IDs, expected result, actual result, status (`PASS/FAIL/SKIPPED/BLOCKED`), platform and log/artifact pointer. Cover at least:

Geometry and calibration unit/property tests:

- Known-size ArUco reference, rotated reference, wrong ID, clipped marker, absent marker, incorrect declared size, impossible marker geometry.
- Fixed-rig calibration missing/expired/resolution mismatch, camera moved/zoom changed.
- Rotated rectangles, circles, ellipses, thin leaves, bent pods, split masks, mask holes, two objects, boundary-truncated silhouette and cast shadows.
- Unit conversions px→mm, mm→cm, mm²→cm²; never confuse pixel area and true volume/weight.
- Zero/negative/nonfinite lengths, uncertainties, threshold and scales.
- Exactly at cutoff, just below, just above, uncertainty interval crossing cutoff and guard-band behavior.
- Changing image resolution while preserving known scale; translating/rotating rigid mask preserving dimensions within rasterization tolerance.
- Cropped or mirrored EXIF image with retained marker; marker separated from crop mask.

Media and UI tests:

- Valid JPEG, PNG, WebP (if declared), grayscale, alpha/transparency, EXIF rotation, unusual ICC/color profile, large but permitted dimensions.
- Wrong extension, unsupported decoder/HEIC if absent, zero bytes, truncated image, decompression bomb, extremely large image, very long filename, Unicode filename, invalid path.
- Browser webcam normal start, denied permission, no webcam, camera disconnect/reconnect, stop capture, multiple UI sessions, refresh and stale frame throttling.
- Video MP4 normal playback/annotation, variable FPS, missing FPS, low FPS, no audio, audio present, odd width/height, portrait, large video, corruption, cancellation, partial failure and output cleanup.
- 10-minute or 15-minute real webcam soak if hardware accessible; report actual duration and crashes.
- Batch image duplicates and aggregation correctness; video same object across frames should not count as hundreds of unique items.
- Mobile portrait/landscape upload and captured photo/video; secure phone live only if available.

Detection and robustness tests:

- Empty tray, blank wall, only fiducial, folded cloth, black glove, hand, tools, packaging, spoon, leaf litter and nonvegetable objects should not be accepted as measured crops.
- One potato/onion/okra and two separated vegetable instances; touching potatoes, bent pods and overlapping leaves.
- Supported vs unsupported crop; explicit operator-selected species/grade policy; unknown class and threshold missing.
- Harsh direct light, dark exposure, backlight, LED cast, wet sheen, shadow, dust, motion blur, low resolution, object moving near frame edge.
- Different distances/angles with same physical item; no false millimeter measurement from untrustworthy geometry.
- Grade flicker in live sequences and tracker ID swaps/crossing objects.
- Large vs small physical specimens whose projected pixel areas invert due camera distance.

Data integrity tests:

- Missing source/license, empty/malformed annotation, invalid polygon coordinates, wrong class ID, missing physical truth for size label.
- Duplicate file and perceptual near-duplicate leakage across splits; same physical item in train/validation/test; augmented sample in held-out split.
- Deterministic grouped split given seed, appropriate class support and strict policy version.
- Negative/synthetic samples never silently counted in real measured-specimen accuracy.
- Label mapping between CVAT/COCO/YOLO export and source ontology exact; visual inspection of random exports.

Systems/reliability/security tests:

- No internet, missing optional trained model, corrupt optional checkpoint, absent crop facts, unavailable external data provider.
- Low disk space, inaccessible output directory, unexpected codec, camera hardware fault.
- Path traversal filename, malformed content header, unsupported mimetype, huge upload, image decompression and arbitrary external URL input.
- No hardcoded secret, no unwarranted network media upload, no unlicensed dataset committed.
- Input buffer not mutated and original raw media kept unchanged.
- Audit fields present: model SHA, policy version, calibration ID, source, timestamp, measurement and reason.
- Deterministic inference on stable fixture and parity across webcam/image/video wrappers for same frame.

Acceptance tests:

- One actually measured small item and one actually measured big item from an approved crop must produce appropriate accepted grades in calibrated valid conditions.
- At least one second crop form (e.g. round onion/potato and elongated okra) should be demonstrated with actual ground truth; restrict declared support otherwise.
- Missing marker demonstrably prevents physical-grade false claim.
- An actual short MP4 produces playable annotated replay and valid CSV/JSON.
- A real laptop webcam shows live crop/size overlay and can stop cleanly.
- Clean install/first run works on the target laptop with no developer-specific hardcoded paths.

Every failed critical test becomes a bug with reproduction steps. Re-run affected tests after fix and then rerun the relevant suite. Do not delete or skip failing tests merely to obtain a green CI indicator.

### 14. Requirements traceability and evidence chain

Map each relevant SRS/PRD FR-01 through FR-17 and NFR-01 through NFR-08 to:

`requirement_id → code module → test IDs → evidence artifact → status → limitations`.

Mark unsupported items such as autonomous robot actuation, drone navigation, external soil field sensors or multi-day field trials as `DEFERRED` with rationale. Do not claim them complete. Ensure core image/video/live requirements are not deferred.

Example trace rows:

```
FR-05 camera calibration -> marker.py, geometry.py -> test_marker.py,
  physical ruler report -> PASS/FAIL/BLOCKED
FR-07 crop thresholds -> grading.py, grade_rules.yaml -> exact equality
  and uncertainty tests -> PASS/FAIL
FR-09 image and live parity -> inference.py and adapters -> golden
  cross-input test -> PASS/FAIL
NFR-03 runtime latency -> benchmark.py -> measured p95 on named device
  -> PASS/FAIL/NOT_EVALUATED
```

Do not merely show checkboxes; every claim needs the actual evidence generated by that test, command or manual observation.

### 15. Forty-eight-hour execution roadmap — hard time gates

The deadline is a real constraint. Implement in strict critical-path order. At every transition produce a runnable artifact and honest checkpoint. Do not try to complete a 15-day field project in 48 hours by fabricating results.

H00–H02 — foundation:

- Audit workspace/requirements and hardware. Reconcile DOCX spec with 48-hour overrides.
- Establish pinned environment, clean project and live progress log.
- Verify image reading, OpenCV `aruco`, basic video decode and Gradio startup.
- Generate printing/measurement instructions and acquisition checklist for user.
- Show first app shell accepting an image.
- Gate: environment, source truth and project starter are real, not a written plan.

H02–H04 — measurement setup:

- Finalize camera marker format, physically measured size entry and operator setup guide.
- Define P0 crop policy schema; ask Nikhil if cutoffs exist. Store null until legitimately defined.
- Implement first synthetic ruler/marker geometry test.
- Start actual measured vegetable capture as soon as the user has samples.
- Gate: data collection can proceed and scale/units are explicit.

H04–H07 — first real inference:

- Implement input validation, image quality, marker scale, CV foreground mask, contour geometry and debug overlay.
- Obtain first valid measured physical dimension on tray or known check object; report bias.
- Gate: end-to-end calibrated still image physical measurement works or issue is immediately escalated.

H07–H10 — binary grading contract:

- Implement crop-specific threshold policy, uncertainty and `SMALL`/`BIG` acceptance with review reasons.
- Add exactly-threshold, missing policy, no marker, bad scale, no vegetable and low quality unit tests.
- Gate: one physically measured small and one big sample, if provided, can be processed deterministically. If no physical data, report BLOCKED but demo test remains synthetic only.

H10–H14 — mandatory input parity:

- Wire laptop webcam streaming and backpressure into shared engine.
- Wire short MP4 upload, processing progress, sample frames and CSV/video result.
- Verify image/webcam/video share inference semantics. First actual E2E tests.
- Gate: every mandatory input mode has a working path. If one fails, do not add optional species/agrimony features.

H14–H18 — data sprint:

- Curate actual specimen and public source ledgers, record independent physical measurements.
- Assemble real raw images, negatives and image provenance without violating licensing.
- Start assisted/manual masks and annotation taxonomy QC.
- Gate: working data manifest with real sources and a clear inventory of measurement labels.

H18–H23 — dataset and split freeze:

- Complete first 200–350 reviewed masks if feasible; ensure held-out physically distinct specimens.
- Run manifest/annotation validator and duplicate, specimen-group and session-group leakage audits.
- Generate grouped split, dataset card and example annotation gallery.
- Gate: dataset has honest label/provenance integrity; training only now allowed.

H23–H26 — baseline validation:

- Evaluate OpenCV geometry/grade on actual held-out measured specimens. Record error types and per-crop sample counts.
- Fix physical metrology bugs rather than tuning to a cherry-picked test sample.
- Gate: an evidence-backed baseline result and failure gallery exist.

H26–H32 — conditional model sprint:

- If sufficient licensed/annotated masks and compatible compute exist, fine-tune a compact instance segmenter with a short controlled sweep.
- Keep baseline available; compare exact held-out images, runtime, abstention and grade correctness.
- Stop learned-model training by H32 even if next experiment is tempting; make no claims about incompletely validated weights.
- Gate: select best verified inference provider and freeze model revision.

H32–H35 — evaluation freeze:

- Lock final thresholds and source/test manifest; evaluate comparison on sealed test.
- Build per-class/crop/confusion/error report, calibration residual chart and failure gallery.
- Gate: actual metrics and support counts, not target scores.

H35–H39 — reliability attack:

- Execute corrupt media, wrong/absent marker, image changes, novel backgrounds, unsupported crops, video FPS edge cases, device disconnect and security tests.
- Run live camera soak and latency test where hardware permits.
- Fix all high-severity bugs, rerun targeted and full suites.
- Gate: no critical false physical measurements under known invalid conditions.

H39–H42 — polish and optional functionality:

- Polish responsive UI; test mobile file upload and optionally secure phone camera.
- Optional crop name/knowledge card, batch analytics and visual-condition flags only when data and time permit.
- All optional modules must fail without impacting binary grading.
- Freeze feature scope strictly at H42.

H42–H45 — packaging:

- Build clean-setup instructions, pinned deps, doc cards, traceability report, model/dataset licenses, version/checksums and reproducibility notes.
- Verify code/tests and sample files in a clean environment on the target OS where possible.
- Gate: a new reviewer can run the app from documented commands.

H45–H47 — live demo rehearsal:

- Prepare locally stored small and big measured samples, wrong-marker/blur case and short video.
- Rehearse three-minute narrative with webcam on the actual laptop, verify visual overlay alignment and video playback.
- Record backup demo if permitted; no hidden substitutions of prior screenshots for live inference.
- Gate: no reliance on cloud or last-second GPU training.

H47–H48 — freeze and deliver:

- Stop edits unless repairing a proven critical bug.
- Run final tests and compare output files/hashes. Record all failures/skips/blocks.
- Produce final handoff with runnable path, commands, actual metrics, unresolved issues, source/weight provenance, latency, supported crops and honest future plan.
- Gate: no unverified claims; final submission complete.

Critical kill switches:

1. At H07 with no working image measurement: stop neural model plans, finish known-marker CV geometry and one-object mask.
2. At H10 with no accepted image grade on synthetic/known-length fixture: stop dataset embellishments and fix policy integration.
3. At H14 with no working media modes: drop extras and concentrate on webcam/photo/video stability.
4. At H24 with too few real measured specimens: narrow the supported crop set and report pilot evidence only; do not fill gap with fake labels.
5. At H32 with poor GPU progress: fall back to classical segmentation; model experiment marked incomplete if appropriate.
6. At H42 with optional feature regression: revert safely to prior working state, freeze scope.
7. Within final six hours: avoid dependency upgrades, redesigns, new frameworks or model swaps without time for full regression.

### 16. Core Definition of Done — no shortcuts

The core assessment is done only if all feasible P0 items are proven, or remaining blockers are explicitly reported rather than called complete:

Application:

- [ ] Clean installation on target laptop with pinned requirements; all commands actually verified.
- [ ] App launches from documented command, no hidden local paths or credentials.
- [ ] Uploaded image produces annotated result with correct status/units.
- [ ] Webcam permission/start/live processing/stop works; real device used when available.
- [ ] Valid MP4 uploads, process with shared inference and export playable annotated media plus results.
- [ ] Invalid/corrupt media are rejected gracefully, with bounded CPU/RAM/disk.
- [ ] Exactly `SMALL` or `BIG` for accepted grading; no third accepted class.
- [ ] Review/not-grading states are separate from binary labels.
- [ ] Crop policy selection/source is visible; no auto species claim unsupported by model.
- [ ] Model missing or optional API offline does not break working baseline.

Measurement and data:

- [ ] Fiducial actual physical dimensions recorded and usable.
- [ ] Known-length reference test demonstrates mm calibration within reported error.
- [ ] No `mm` claim on invalid/unscaled input.
- [ ] Crop-specific unit/metric/threshold version stored with every grade.
- [ ] Boundary uncertainty gate correct and unit-tested.
- [ ] Real physical ground truth only when actually measured, with specimen ID and provenance.
- [ ] Dataset sources individually licensed or clearly research-only/excluded.
- [ ] Train/val/test split independent by physical specimen and appropriate session/lot/source grouping.
- [ ] Test masks physically and manually reviewed; no train/test duplicate hashes.

Model and evaluation:

- [ ] Deterministic E1 OpenCV baseline demonstrated and benchmarked.
- [ ] Neural model results only reported if training/evaluation actually completed.
- [ ] No unsupported universal-vegetable recognition claim.
- [ ] Raw test predictions, confusion matrix and support counts available.
- [ ] Per-crop errors, lighting and perspective limitations disclosed.
- [ ] Explicit rejection on missing marker and low-quality/partial views.
- [ ] Actual benchmark includes end-to-end preprocessing and postprocessing time, not only model forward pass.
- [ ] Failure gallery includes failed examples, not just successes.

Engineering/reproducibility:

- [ ] Config, code, data split, calibration and model artifacts versioned and hashed.
- [ ] Critical unit/integration/data tests pass with actual command exit codes.
- [ ] Fail/skip/block reports preserved; no fabricated green CI.
- [ ] Source files typed/organized, linted and tested to extent allowed by time.
- [ ] No secrets, personal media or prohibited datasets committed.
- [ ] Known limitations and safe failure modes documented.
- [ ] Three-minute actual demo rehearsed and backed by on-disk permitted example media.

### 17. Extended technical gates — if time and evidence allow

Only after P0 stabilizes:

- [ ] Species recognition model trained on supported crops with confusion/unknown rejection evidence.
- [ ] Per-object video tracking robust to crossing and re-entry; event duplication bounded.
- [ ] Lot CSV/JSON aggregation distinguishes unique objects vs per-frame observations.
- [ ] Surface quality visual flags trained/validated independently from size label.
- [ ] Optional nutrition facts with exact source/units/retrieval date; no sample-specific claims.
- [ ] Farm weather, soil lab or sensor values displayed only with input provenance/time. Missing values explicitly unmeasured.
- [ ] Mobile live camera works over secure origin on an actual phone, not just laptop LAN HTTP.
- [ ] CPU model conversion/ONNX or edge benchmark, if valuable and licensed.
- [ ] Machine-sort event mock with dry-run idempotency, sequence/expiry, no actual actuation.
- [ ] Basic UI visual QA at desktop/mobile widths, color contrast, keyboard accessibility, no text overflow or broken overlays.

Never make a P1/P2 success condition for completing P0; never force an optional subsystem into the primary model.

### 18. Specific review and acceptance gates

Gate G0 — requirements truth:

- P0 matches evaluator's single task; Word doc/48h conflict resolved in decision log.
- Unknown crop/calibration handling and real physical threshold policy documented.

Gate G1 — environment/reproducibility:

- Clean boot, verified local dependencies and camera/video codecs. No hidden service or access requirement.

Gate G2 — label/source truth:

- No false physically measured labels. Dataset manifest, licenses, annotations and group splits auditable.

Gate G3 — metrology:

- Ruler fixture and marker/plane geometry verified. Missing/incorrect calibration refuses physical measurement.

Gate G4 — binary grader:

- Two accepted labels only; threshold and guard-band unit/property tests pass; real small and big examples when available.

Gate G5 — media behavior:

- Image, live webcam, video upload all run end-to-end in target environment. Mobile upload tested if accessible.

Gate G6 — evaluation:

- Held-out independent specimen and condition report, failures and support counts. No copying published benchmark numbers into actual results.

Gate G7 — robustness/privacy:

- No known critical unsupported input yields silently confident size. Corrupt/malicious files and no-camera conditions fail safely. No accidental remote media upload.

Gate G8 — release package:

- README, code, lock, policies, model/data cards, test logs, license notices, examples and reproducible run instructions complete.

Gate G9 — evaluator demo:

- Three-minute script, camera ready, stored examples, backup recording, smoke tests, exact versions and unresolved risk statement ready.

For each gate, produce machine- or human-verifiable evidence. Mark `PASS`, `FAIL`, `PARTIAL`, `SKIPPED` or `BLOCKED` with the exact reason. If a hard gate fails, do not call the system production-ready. There is no rational basis to promise 100%, 110% or universal robustness from a 48-hour training run.

### 19. Threat modeling, operational reliability and protection of user files

- Do not run destructive recursive deletes of broad folders, the user's home directory or existing worktrees. Scope cleanup to known generated artifacts; preserve originals.
- Never execute raw model-generated shell code from uploaded metadata or filenames.
- Avoid remote image URLs and arbitrary filesystem paths as user input without validation. Restrict outputs to known working directories.
- Protect against decompression bombs, codec hangs, huge video duration, too many frames, oversized file count, memory exhaustion, invalid GIF/animated inputs and unsupported pixel modes.
- If binding a server to network, obtain proper user authorization, consider who can upload media and whether files are retained/exposed. Default localhost.
- Do not automatically publish private farm imagery, code, dataset or model to GitHub or cloud services. Ask before external publication.
- Respect Ultralytics/SAM2/Torchvision/data licensing. Document actual model source and deployment implications; an AGPL experiment is not the same as commercial proprietary deployment permission.
- Do not leak personally identifiable information from user background images in logs or demo gallery. Retain raw content minimally.
- Use bounded memory, timeouts, graceful errors, watchdog for long video processing, progress/cancellation and session isolation.
- Preserve raw training media, annotation ledger and test data; do not modify in place for aesthetic data cleaning.
- Optional machine automation remains simulated only. Do not actuate motors/robot hardware without explicit safety engineering and authority.

### 20. Documentation and repository delivery contract

Create or update:

1. `README.md` — description, exact prerequisites, setup for Windows/macOS/Linux where tested, camera/marker instructions, run commands, input types, demo gallery, limitation summary and references.
2. `docs/PRD.md` / `docs/SRS.md` — practical implementation-specific traceability from attached Word document and 48h overrides; don't mindlessly duplicate the entire doc.
3. `docs/ARCHITECTURE.md` — diagram, module boundaries, data contract, placement of model and deterministic policy, no unnecessary microservices.
4. `docs/CALIBRATION.md` — print/measure/use fiducial, tray geometry, tested error range, parallax limitations, camera profile lifetime.
5. `docs/ANNOTATION_SOP.md` — physical specimen collection, ontology, mask rules, QA and review process.
6. `docs/DATASET_CARD.md` — actual source/license counts and distinct specimens, splits, bias and missing measurements.
7. `docs/MODEL_CARD.md` — baseline vs learned model, code and weight license, training/test procedure, versions, supported crops, performance and limitations.
8. `docs/VALIDATION_PLAN.md` — all test IDs, command procedures, expected evidence and physical validation SOP.
9. `docs/REQUIREMENT_TRACEABILITY.md` — FR/NFR mapping with current status/evidence.
10. `docs/KNOWN_LIMITATIONS.md` — all physical assumptions, unsupported scenarios, small sample caveats and production barriers.
11. `docs/PROJECT_LOG.md` and `docs/DECISION_LOG.md` — timestamped work/evidence, tradeoffs, blocked physical dependencies.
12. `reports/DEMO_REPORT.md` — actual final metrics, supported classes/devices/modes, tests, failures, decisions, future work.
13. `reports/failure_gallery` — real input/ground truth/prediction debug evidence, with only appropriately permitted data.
14. `reports/metrics` — actual JSON/CSV predictions, confusion/error/latency reports, split/weight/config hashes.
15. `LICENSE-NOTICES.md`, `.gitignore`, safe `.env.example`, requirements lock and optional Docker instructions only if tested.

Commands must refer to files that exist. Verify at least one clean installation/start path and core test command. For example, after implementation, `python -m pytest -q`, `python scripts/validate_dataset.py ...`, `python scripts/evaluate.py ...`, `python apps/gradio_app.py`; adjust to actual correct names and environment. On Windows write valid `.venv\Scripts\python.exe` examples where appropriate, not solely Unix activation examples.

The final Git commit/release should contain source and permitted small examples; use Git LFS or download instructions for large weights/datasets when appropriate. Never assume the evaluator has your local drive paths.

### 21. Three-minute evaluator walkthrough — practice it

0:00–0:20 — state only actual goal: binary SMALL/BIG grading; demonstrate marker/size policy and why it matters.

0:20–0:50 — upload real physically measured SMALL and BIG vegetables on controlled background. Show mask, measurement, threshold, provenance and grade.

0:50–1:20 — switch to actual live laptop webcam; hold measured vegetable with marker, vary angle/distance within validated domain, show grade stability and latency.

1:20–1:45 — upload a short MP4 of produce, show processed playback and CSV/results, explaining per-frame vs unique tracked items.

1:45–2:05 — deliberately remove marker, blur frame or put object off screen. Show safe review / calibration-required state with retake instruction.

2:05–2:35 — show real dataset ledger and independent test results: true specimen count, confusion matrix, measurement error and failure gallery. Clearly identify untested crops.

2:35–3:00 — explain architecture separation and future roadmap (crop-ID model, lighting hardware, RGB-D, machine automation, agronomy) without pretending extras are implemented/validated.

Keep a pre-recorded real run as backup if allowed, label it as prerecorded, and have one reproducible local sample per demo step. Do not record a selective run and represent it as an independent statistical benchmark.

### 22. What to do if new user evidence appears

- If Nikhil supplies thresholds, set policy from that approved source, preserve version and rerun rule tests.
- If the user sends real photographs/measurements, validate provenance, annotate, add as a new dataset version, recompute splits (while preserving locked test isolation), and retrain/evaluate if time permits.
- If the user supplies hardware details, update actual latency/codec/driver assumptions; do not blindly install a Jetson stack on an ordinary laptop.
- If the user sends a new direct instruction, reconcile with the contract and deadline; if it changes P0 scope, explain tradeoff and act accordingly.
- If an incidental package/network failure blocks optional model, use approved fallbacks and state limitation; do not bypass access restrictions or silently use paid APIs.

### 23. Final handoff format — mandatory

At completion, deliver a compact but evidence-dense report with these exact headings and fields:

```
GROWIZE VISION — 48-HOUR ENGINEERING HANDOFF

1. Project result
   - Actual delivered repository path
   - Git revision / release artifact
   - Exact command to launch
   - Exact environment/OS/browser/camera tested

2. Implemented core functionality
   - Still upload: PASS/FAIL with link/screenshot/artifact
   - Webcam live: PASS/FAIL/BLOCKED with real-device evidence
   - Video upload: PASS/FAIL with playable output
   - Metric calibration: method, verified physical sample and error
   - Accepted size labels: SMALL/BIG only
   - Review/failure handling: examples and reasons

3. Training dataset
   - Actual distinct physical specimens, crop-by-crop
   - Total images/frames separated from independent objects
   - Public sources and licenses
   - Masks audited and annotation QA
   - Split method, hashes, leakage findings

4. Models and decisions
   - OpenCV baseline configuration
   - Optional trained model architecture/checkpoint/license
   - Training commands, epochs, hardware and duration if executed
   - Reason primary inference provider was selected

5. Measured performance (actual results only)
   - Test unique-specimen counts and grade support
   - Full confusion matrix, balanced accuracy, per-class precision/recall
   - Measurement MAE/median/P90/p95/bias in mm by crop
   - Accepted coverage, rejection and critical false acceptance
   - Latency p50/p95, FPS, memory, real-device soak duration
   - Robustness by lighting/blur/occlusion/camera/marker condition

6. Verification and release gates
   - Number of tests passed/failed/skipped/blocked
   - Exact commands and saved log paths
   - G0–G9 statuses and evidence links
   - Critical unresolved issues, severity and workaround

7. Reviewer package
   - README and docs
   - Real permitted sample images/video and output overlays
   - Dataset/model cards and licenses
   - Three-minute demo path/recording
   - Clean setup/reproduction instructions

8. Known limitations and next actions
   - Unsupported crop/lighting/perspective/device scenarios
   - Missing physical ground truth or incomplete data collection
   - No uncalibrated physical measurement claims
   - Specific field validation and camera hardware needed for ARL production
```

Do not leave blank false claims. Use `NOT RUN`, `NOT AVAILABLE`, `NOT EVALUATED`, `BLOCKED` and explicit reasons as needed. Every numerical result must be traceable to actual code, data, command, timestamp and output.

### 24. Operating philosophy and continuous verification

At all times:

- Prioritize the accepted binary grading and physically valid measurement over impressive but unrelated technologies.
- Use scientific skepticism. A successful screenshot is a demonstration; a held-out measured test is evidence; a farm rollout requires more.
- Solve the current bottleneck. A dataset lacking physical truth is not fixed by adding ten large pretrained models.
- Be aware that preprocessing and postprocessing are part of the model's behavior; benchmark the whole pipeline.
- Make uncertainty and failure explicit; an honest review state is part of reliability, not a model failure to hide.
- Maintain a running executable build, tests and recoverable versioned state. Never sacrifice the last-known-working demo for an optional upgrade.
- Be precise about what is independent. Use specimen, lot, camera/day and source grouping rather than random frames for final statistical reporting.
- Avoid hallucinated datasets, hyperparameters, measurements, package versions, citations and metrics. Read actual documentation before implementing version-sensitive APIs.
- Run tests and report their actual outputs. If something cannot be run because of environment or time, state that and leave executable commands for when hardware/data become available.
- Ask the user only for decisions that materially affect correctness: exact thresholds and metrics, physical sample collection and camera/device access.
- Keep external actions within authorized scope. Do not publish/share, purchase services, upload private pictures or control machinery without permission.
- Include contribution and license notices for anything reused; do not treat a research demo as an automatic commercial software license.

### 25. The single most important instruction

You are not done when code compiles, a model downloads, a training loss decreases or a beautiful web UI appears. You are done when a reviewer can run the system, demonstrate real calibrated SMALL/BIG grading, observe a safe refusal on ambiguous media, upload photos and videos, watch a laptop webcam work, inspect independent data/physical measurement evidence, reproduce critical tests and understand exactly what is and is not proven.

Start now. Audit the actual workspace, record the deadline, read the attached SRS/PRD completely, construct the core app, and keep implementing until the deliverables are verified or a genuine physical/permission blocker is documented. Your goal is a technically excellent, scientifically honest 48-hour assessment that can mature into ARL's farm and machine vision stack—not unsupported perfection claims.

## END MASTER PROMPT


