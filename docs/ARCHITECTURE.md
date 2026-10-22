# Architecture

## 1. Shape of the system

Growize Vision is a local Python application with one inference service and three
input adapters. The UI is a thin Gradio layer; grading and provenance live in the
package so they can be tested without a browser.

```text
apps/gradio_app.py
  ├─ still image ─┐
  ├─ webcam frame ─┼─► InferenceService.inspect_frame(frame, context, request)
  └─ MP4 sampler ─┘         │
                            ├─ media.validate_bgr_frame / resize
                            ├─ quality.assess_quality
                            ├─ markers.detect_best_marker / calibrate_from_marker
                            ├─ segmentation_cv.segment_foreground
                            ├─ geometry.measure_contour
                            ├─ grading.evaluate_measurement
                            ├─ rendering.render_inspection
                            └─ exports.export_inspection / video.process_video
```

`InferenceRequest` carries operator choices (crop, measured marker side, demo mode,
segmentation provider). `MediaContext` carries source type, source ID, device,
frame/timestamp, and session/batch IDs. `InspectionResult` is the single result
contract. Raw frames and masks are not serialised.

## 2. Modules

| Module | Responsibility |
|---|---|
| `domain.py` | Enums, dataclasses, status/grade contract, JSON-safe rounding |
| `config.py` | YAML loading, policy registry, validated settings, config hashes |
| `media.py` | Bounded JPEG/PNG/WebP decode, EXIF, alpha, BGR normalisation, safe stems |
| `quality.py` | Blur, contrast, clipping, glare, object area, edge-touch flags |
| `markers.py` | ArUco detection, geometry validation, calibration homography, marker mask |
| `segmentation_cv.py` | Controlled-background foreground segmentation and components |
| `geometry.py` | Plane projection, Feret/rectangle/area metrics, error envelope |
| `grading.py` | Interval + guard-band comparison, SMALL/BIG/REVIEW |
| `rendering.py` | Annotated overlay with mask, axes, status, uncertainty |
| `inference.py` | Status precedence and the shared inspection pipeline |
| `exports.py` | PNG, JSON, per-object CSV, batch CSV, safe paths |
| `video.py` | Metadata limits, frame sampling, H.264 output, frame CSV/JSON, contact sheet |
| `tracking.py` | Live 3-of-5 stabilization and session counters |
| `dataset.py` | Provenance/hash/measurement/leakage validation and grouped splits |
| `evidence.py` | Environment metadata and report helpers for scripts |

## 3. Status precedence

The order is deliberate and safety-first:

1. **Invalid input** → `INVALID_INPUT`
2. **No foreground object** → `NO_OBJECT`
3. **Neural mode requested** → `MODEL_UNAVAILABLE` (no model shipped)
4. **Invalid/missing calibration** → `CALIBRATION_REQUIRED`
5. **Busy/unsupported background** → `OUT_OF_DISTRIBUTION`
6. **Quality gate failure** → `LOW_IMAGE_QUALITY`
7. **Edge/marker overlap/merged contour** → `OCCLUDED`
8. **No crop policy or no approved threshold** → `POLICY_REQUIRED`
9. **Metrology failure** → `REVIEW_BORDERLINE`
10. **Multiple instances** → `MULTIPLE_OBJECTS` (per-instance rows retained)
11. **Valid single instance** → `SMALL` / `BIG` / `REVIEW_BORDERLINE`

A status never carries a size class except `ACCEPTED`, and `ACCEPTED` always has a
size class. This is enforced by `GradeDecision.accepted` and tested in
`tests/test_inference.py`.

## 4. Calibration and geometry

- ArUco `DICT_4X4_50`, expected ID 0 by default.
- Marker outline must be a plausible quadrilateral, meet the minimum pixel side and
  area fraction, and have consistent side lengths.
- The user supplies the **measured outer black-square side in millimetres**; the
  nominal PDF size is not trusted.
- A homography maps the four marker corners to a metric square. Contours are
  projected through it and measured in millimetres.
- The uncertainty envelope combines marker-measurement uncertainty, corner
  localisation, mask-boundary localisation, and a configured parallax bound. It is
  an engineering bound, **not** a statistical 95% confidence interval.
- Single overhead homography measures the visible projected footprint. It does not
  measure height, volume, weight, or hidden extent.

## 5. Grading

For measurement `m`, uncertainty `u`, threshold `T`, guard `g`:

- `m - u >= T` → `BIG`
- `m + u < T` → `SMALL`
- otherwise → `REVIEW_BORDERLINE`

The comparator is configurable per crop, but the shipped policies use `>=`.
Comparators and metrics are validated when the YAML loads.

## 6. Input adapters

### Image

`load_image_file` → `inspect_frame` → `export_inspection`. The UI writes annotated
PNG, result JSON, and object CSV, then exposes them for download. Uploaded files are
temporary Gradio files; the app does not copy raw uploads into the repository.

### Webcam

Gradio streams browser frames to `live_inspect_ui`, which calls the same service.
`LiveSession` maintains a 3-of-5 accepted-class window and clears stability on any
non-accepted status. This is display stabilization, not a physical tracking claim.
The webcam path requires a secure browser context; localhost is treated as secure by
desktop browsers, but phone access over plain LAN HTTP is **not** a supported live
camera path.

### Video

`process_video` reads the container, samples at the configured FPS, calls the same
service per sampled frame, writes an annotated H.264 MP4 with no audio, and exports a
frame CSV/JSON. Counts are **per-frame observations**. `unique_vegetable_count` is
`null`; no identity tracker is validated for video.

## 7. Provenance

Every result records:

- `software_version`, `pipeline_version`, `pipeline_sha256`
- `model_sha` (`null`; no model)
- policy version, approval status, threshold, comparator, demo flag
- calibration ID, marker ID/side, homography, per-pixel scale, uncertainty inputs
- original/processed shape, timings, warnings, reasons
- source type/ID/device/frame/timestamp/session/batch

Config files are hashed at load time. The UI and exports show the policy approval
state so a demo result cannot be mistaken for an approved one.

## 8. Failure containment

- Decode, size, and unsupported-format errors become `INVALID_INPUT` or a bounded UI
  error; no partial grade is returned.
- Export failures are appended as warnings and do not convert a refusal into an
  acceptance.
- Video cancellation removes the partial file and keeps completed exports only when
  the report is finalised.
- The neural provider is deliberately unavailable; selecting it returns
  `MODEL_UNAVAILABLE` with an explanation rather than silently falling back.
- The app binds to `127.0.0.1` by default and has no authentication. It is a local
  tool, not a network service.

## 9. Determinism and performance

The baseline uses deterministic OpenCV operations. Given the same frame, config,
marker measurement, and crop policy, the decision is stable. CPU latency is measured
by `scripts/benchmark_latency.py`; current result is a 116.0 ms median over 80
single-frame observations on the build workstation. This is not a throughput or
field-latency guarantee.
