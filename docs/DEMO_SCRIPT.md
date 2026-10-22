# Three-minute demo script

## Before the demo

```powershell
cd "C:\code_shit\projects\agriculture vegetables"
python scripts\demo_preflight.py --full-video --camera
python apps\gradio_app.py
```

Open <http://127.0.0.1:7860>. Keep `reports/DEMO_REPORT.md` and
`reports/failure_gallery/index.md` open in another window. Say the evidence status
out loud before showing any result:

> No trained model exists, no real measured dataset was supplied, and real
> thresholds are TBD. This is a calibrated engineering demo, not a validated
> agricultural grader.

## Demo path

### 0:00–0:30 — Scope and policy

1. Point to the persistent demo-policy banner.
2. Open **Evidence & limits** or the README status block.
3. State: accepted outputs are only `SMALL`/`BIG`; everything else is a refusal.
4. State: the real threshold is `null`; the visible number is a demo cutoff.

### 0:30–1:20 — Image path, demo policy

1. Keep **Use assessment-demo cutoffs** checked.
2. Upload `data/demo/potato_big_synthetic.png`.
3. Select potato, marker side 50 mm, expected marker ID 0.
4. Click **Inspect image**.
5. Point to:
   - `ACCEPTED` and `BIG`;
   - measured value 72.3 mm with uncertainty;
   - marker overlay and measured axis;
   - `ASSESSMENT_DEMO_ONLY` policy stamp;
   - pipeline SHA and JSON/CSV downloads.
6. Repeat with `data/demo/okra_small_synthetic.png`; show the long-axis metric.

### 1:20–1:50 — Strict policy refusal

1. Clear the image or keep the current one.
2. Uncheck the demo-policy checkbox.
3. Inspect again.
4. Show `POLICY_REQUIRED` / `NO_APPROVED_THRESHOLD` and confirm that no size class
   is assigned. Do not treat this as a crash; it is the correct refusal.

### 1:50–2:20 — Calibration and quality refusal

1. Re-enable demo mode.
2. Upload `data/demo/no_marker_synthetic.png`; show `CALIBRATION_REQUIRED` and no
   millimetres.
3. Upload `data/demo/blurred_synthetic.png`; show `LOW_IMAGE_QUALITY`.
4. Open `reports/failure_gallery/index.md`; explain the four rendered synthetic
   cases (missing marker, empty mat, blur, edge clipping).

### 2:20–2:45 — Video path (if browser works)

1. Open the **Video** tab.
2. Upload `data/demo/synthetic_potato_video.mp4`.
3. Keep sampling default; process.
4. Show the annotated MP4 and frame CSV.
5. State clearly: counts are **sampled per-frame observations**, not unique
   vegetables. No multi-object tracker is validated.

If browser file upload fails or the harness is unavailable, fall back to
`reports/preflight.json` and the programmatic video test output. Say **“browser
video upload not evaluated”**; do not imply it passed. Never claim a live webcam
session that was not recorded.

### 2:45–3:00 — Close and next step

1. Open `reports/DEMO_REPORT.md`.
2. State the blocked items: real measured dataset, approved thresholds, trained
   model, phone live camera.
3. State the next physical step: print/measure the marker, collect 60–100 measured
   specimens, freeze the policy, run the sealed grouped test.
4. Close the app with `Ctrl+C`.

## Fallback matrix

| Failure | Fallback | Honest wording |
|---|---|---|
| Browser won't load | Show README + JSON evidence | “Browser UI unavailable in this session” |
| Gradio image upload fails | Use unit/export evidence | “UI upload not evaluated now” |
| Webcam unavailable | Show preflight device result and limitation | “Device capture passed; live UI not evaluated” |
| MP4 UI fails | Show `tests/test_video.py` + preflight report | “Programmatic video passed; browser upload blocked” |
| Camera busy | Close other apps, retry once | “Camera unavailable, not passed” |
| Disk full | Stop; do not delete user files | “Export blocked by disk space” |

## Do-not-say list

- “Trained on real vegetables.”
- “95% accurate.”
- “Validated for potatoes/okra/…” on real produce.
- “Measures weight/volume.”
- “Production ready.”
- “Unique vegetables counted in this video.”
- “Phone camera supported.”
- “These demo cutoffs are ARL standards.”
