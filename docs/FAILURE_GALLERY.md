# Failure gallery

The reproducible gallery is generated at
[`reports/failure_gallery/index.md`](../reports/failure_gallery/index.md) by
`python scripts\build_failure_gallery.py`.

## What it contains

| Fixture | Expected state | What it demonstrates |
|---|---|---|
| `no_marker_synthetic` | `CALIBRATION_REQUIRED` | No scale is invented; no millimetres are emitted |
| `empty_mat_synthetic` | `NO_OBJECT` | Empty scenes are refused, not graded |
| `blurred_synthetic` | `LOW_IMAGE_QUALITY` | Quality failure prevents a forced class |
| `edge_clipped_synthetic` | `OCCLUDED` | A clipped silhouette is not measured |

Each PNG is a three-panel diagnostic: original input, annotated decision, and
machine-readable expected/actual status, reasons, object count, and policy state.

## Important scope

Every case is **synthetic** and labelled `SYNTHETIC - NOT REAL PRODUCE`. These
images verify refusal precedence and drawing; they are not a sample of real
failures and provide no accuracy estimate. A real-produce failure gallery is
`BLOCKED` until licensed measured specimens and the approved policy are available.

## Regenerate

```powershell
python scripts\build_failure_gallery.py
```

The command exits non-zero if any fixture's actual state differs from its manifest
expectation.
