# Known limitations and non-goals

## 1. Evidence limitations (blocking)

1. **No trained model.** No weights, checkpoint, training run, or model accuracy
   exist. `model_sha` is `null`; only `pipeline_sha256` identifies the code/config.
2. **No real measured dataset.** No real vegetable image was used as accuracy
   evidence. Real unique-specimen counts are zero.
3. **No ARL-approved thresholds.** Real cutoff values are `null`. Demo cutoffs are
   assessment-only and must not be quoted as buyer or agronomic standards.
4. **No real-produce accuracy, confusion matrix, MAE, coverage, or field report.**
   These are `NOT EVALUATED`, not zero and not passing.
5. **Synthetic fixtures are not evidence of accuracy.** They test geometry and the
   refusal contract only.
6. **No model comparison.** Baseline-versus-model is `BLOCKED` because no model
   exists.

## 2. Measurement limitations

7. **Planar silhouette only.** A single overhead homography measures the projected
   footprint. It cannot recover height, volume, weight, density, or hidden extent.
   Tall objects have parallax bias bounded only by an engineering envelope.
8. **Not a statistical confidence interval.** The reported uncertainty combines
   configured terms; it is not a calibrated 95% interval from repeated trials.
9. **Straight-axis proxy.** Minimum-area-rectangle long side is not the curved
   centreline length of a bent pod or leaf. A curvature warning may appear, but no
   geodesic correction is implemented.
10. **Print/measurement risk.** A scaled printer, thick label, warped paper, or
    careless caliper reading corrupts every millimetre downstream. The 100 mm
    verification line is a check, not an automatic correction.
11. **No lens-intrinsics calibration.** Strong wide-angle distortion is not modelled.
    Use a modest focal length and keep the marker/object central.
12. **Marker/object separation required.** Overlap, marker on the object, or a marker
    outside the frame forces refusal.
13. **Physical-unit ambiguity for leaves/bunches.** Coriander has no safe size metric
    and therefore no demo threshold. Spinach leaf extent is defined, but real leafy
    validation is absent.

## 3. Input and environment limitations

14. **Controlled matte background required.** Clutter, patterned trays, soil, strong
    shadows, and background objects can cause `OUT_OF_DISTRIBUTION`, wrong
    segmentation, or refusal. Real farm scenes are out of scope.
15. **One visible object.** Multiple objects yield `MULTIPLE_OBJECTS`; stacked or
    touching vegetables are not separated reliably.
16. **No unique-object tracking in video.** Video counts are sampled **per-frame
    observations**. `unique_vegetable_count` is `null`.
17. **Browser webcam live path not fully validated in this environment.** The USB
    camera opened and produced a frame in preflight, but the browser harness became
    unavailable before a live-stream session could be recorded. Treat live webcam UI
    as `NOT EVALUATED` at browser level.
18. **Browser MP4 upload not fully validated.** Programmatic MP4 decode, H.264
    output, frame CSV, and no-audio checks pass. The Gradio file-upload interaction
    was not re-tested in a live browser after the harness failure.
19. **Phone paths are blocked.** Phone gallery upload was not tested on a physical
    device. Live phone camera over HTTPS is unsupported without a secure origin and
    device validation; plain LAN HTTP will not get camera permission.
20. **Batch upload not implemented** (P1 scope).
21. **No real camera soak test.** Long-run memory, USB disconnect, sleep/resume, and
    thermal behaviour are not evaluated.
22. **Disk-full handling not injected.** Export errors are bounded, but a truly full
    disk was not tested.

## 4. Application and operations limitations

23. **Localhost-only, no authentication.** Do not bind to a public interface. There
    is no user isolation, session authentication, or rate limiting.
24. **No persistence/database.** Exports live in temporary Gradio files or the
    configured report directory. Retention and deletion policy are not implemented.
25. **No accessibility certification.** Controls are keyboard-reachable and status is
    text plus colour, but no WCAG audit was performed.
26. **No actuator or safety case.** The app is decision support only. It does not
    control a conveyor, diverter, or sorter.
27. **Latency is workstation-specific.** Median 116.0 ms and p95 124.3 ms over 80
    single-frame observations, decode excluded. This is not a throughput or
    field-latency SLA.
28. **No statistical uncertainty on synthetic checks.** Four geometry fixtures and
    eight contract cases are too few for distributional claims.
29. **No agronomy, soil, nutrition, pesticide, yield, or food-safety output.** These
    claims are prohibited, not merely unimplemented.
30. **Final automated evidence was not re-captured after the shell harness stopped
    responding.** The final code revision passed 164 tests and Ruff before the docs
    were added; docs do not affect tests. Treat `reports/pytest_full.txt` as the
    last archived run, not a post-doc capture.

## 5. Release judgement

**YELLOW.** The calibrated baseline and refusal contract are functional and tested
on synthetic/controlled inputs. It is **not** validated on real produce, has no
approved thresholds, and has no learned model. Any production, buyer-compliance,
accuracy, or field-readiness claim is unsupported.

## 6. What would remove the blocking limitations

1. Supply a licensed, consented, physically measured real dataset with QA'd masks.
2. Freeze and approve crop policies and thresholds with ARL/Growize.
3. Run the sealed grouped test and publish per-crop accuracy, coverage, MAE, and
   failure gallery with confidence intervals.
4. Record successful browser image, webcam, and MP4 sessions on the target laptop.
5. Test a physical phone on HTTPS if mobile live camera is required.
6. Only then train and compare a segmenter under the model-card promotion rules.
