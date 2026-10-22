# Report metrics

This directory contains real-data metric templates and explicitly scoped engineering
results.

- `final_results.json` — real metrics are `null` / `not_yet_evaluated`; synthetic
  contract and latency values are explicitly scoped.
- `specimen_predictions.csv` — header-only. No real specimen prediction exists. Do
  not populate with synthetic rows.
- `condition_slices.csv` — header-only. No real lighting/angle/crop slices exist.

Synthetic results are not real accuracy. Model metrics are absent because no model
was trained. See `reports/DEMO_REPORT.md` for the mandatory evidence report.
