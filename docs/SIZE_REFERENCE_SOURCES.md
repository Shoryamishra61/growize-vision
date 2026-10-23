# Crop size references — sources and interpretation

**Version:** reference-2026-10-24.1 · **Policy file:** `config/grade_rules.yaml`
(policy-2026-10-24.1)

Strict-mode thresholds and size bands are **PUBLISHED_REFERENCE** values
compiled from citable standards, trade practice, and research variety data.
They make the grader work out of the box, and every accepted decision carries
the reason `PUBLISHED_REFERENCE_NOT_POPULATION_VALIDATED` plus a warning,
because a number from the literature is not a measurement of *your* farm's
population. When an ARL/Growize approval, an AGMARK grade document, or a buyer
specification arrives, replace the value and provenance in one place — the
config loader refuses any threshold that lacks a provenance block.

## Reference table

| Crop | Grading metric | Binary cut | Bands (mm) | Evidence tier | Source |
|---|---|---:|---|---|---|
| Potato | min width (mm) | 50 | 0 / 40 / 60 | Industry trade practice | Processing grade 40–60 mm = medium, >60 mm = large ([Intermag crop guide](https://intermag.eu/crop-guides-potato)); 40/60 classing is common trade practice |
| Onion | min width (mm) | 50 | 0 / 40 / 60 | Industry trade practice (India) | Rotary-grader classes: small <40 mm, medium 40–60 mm, large >60 mm (PMC onion-grader study); mandi practice reports >60 mm premium |
| Tomato | min width = equatorial proxy (mm) | 57 | 0 / 47 / 67 | Official standard | [CODEX STAN 293](https://www.fao.org/input/download/standards/11013/CXS_293e.pdf) round-tomato size scale: ≤47, 47–57, 57–67, 67–82, 82–102, >102 mm; 57 mm = M/B code boundary; bands group codes (≤47 / 47–67 / >67) |
| Okra | curved centreline length (mm) | 120 | — (binary) | Research variety data | IIVR varieties marketable at 8–10 cm ([IIVR okra](https://icariivr.org.in/okra-2)); Arka Anamika pods 13.8–16.1 cm ([CCARI DSS](https://ccari.res.in/dss/okra.html)); export grade 6–8 cm. 120 mm separates standard tender pods from long-pod variety types |
| Cucumber | projected long axis (mm) | 150 | 0 / 150 / 200 | Official standard + research | USDA slicing minimum length 6 in ≈ 152 mm ([USDA cucumber standard](https://www.ams.usda.gov/grades-standards/cucumber-grades-and-standards)); slicing harvest 15–20 cm; North-American slicing types 20–30 cm (Grumet et al. 2022) |
| Spinach | single flat leaf blade length (mm) | 120 | — (binary) | Horticulture reference (weakest tier) | Mature leaves 10–15 cm (4–6 in); savoy averages 12–15 cm ([UW Extension](https://hort.extension.wisc.edu/articles/spinach-spinacia-oleracea)); baby leaf <7–10 cm |
| Coriander | — | none | none | No defensible single definition | Requires a defined bunch + weighing; grading stays `POLICY_REQUIRED` by design |
| Brinjal | max projected span (mm) | 120 | — (binary) | Research variety data | Indian evaluations: fruit length 4.9–17.2 cm (mean ~9.7); long types 16–27 cm; 120 mm splits standard vs long types |
| Capsicum | min width (mm) | 70 | 0 / 64 / 76 | Industry trade practice | US market grade edges (2.5 in / 3 in); Indian blocky varieties 4.5–7.3 cm diameter ([US Foods](https://www.usfoods.com/tools-tips-and-ideas/tools/resource-centre/produce-resources/peppers)) |
| Cabbage | head diameter (mm) | 180 | — (binary) | Research variety data | Marketable heads 20.3–21.5 cm; smaller grades 12–15.6 cm (Wszelaki et al. 2003, HortScience) |
| Cauliflower | curd diameter (mm) | 150 | — (binary) | Official standard + research | USDA market minimum 4 in (102 mm); harvest-ready 6–8 in (15–20 cm) |
| Beetroot | root diameter (mm) | 50 | 0 / 40 / 60 | Official standard + research | USDA minimum 1.5 in (38 mm); fresh market ~50 mm; SA Class 1 min 40 mm ([USDA beets](https://www.ams.usda.gov/grades-standards/beets-grades-and-standards)) |
| Turnip | root diameter (mm) | 60 | — (binary) | Industry trade practice (weakest) | [UNECE FFV-59](https://unece.org/fileadmin/DAM/trade/agr/standard/fresh/FFV-Std/English/59_RootandTubercleVeg.pdf) sets no mandatory size; typical market 5–8 cm |
| Carrot | root length (mm) | 170 | — (binary) | Official standard + research | [UNECE FFV-10](https://unece.org/sites/default/files/2023-12/FFV10_Carrots_2023_e.pdf) main-crop min 20 mm diameter; [USDA](https://www.ams.usda.gov/grades-standards/topped-carrots-grades-and-standards) length ≥ 5 in (127 mm); commercial 15–20 cm |
| Green chilli | curved pod length (mm) | 90 | — (binary) | Research variety data | Market varieties 6–11 cm; IIVR export grade 7–9 cm; LCA 625 9–11 cm |
| Bitter gourd | curved fruit length (mm) | 250 | — (binary) | Research variety data | [TNAU](https://agritech.tnau.ac.in/horticulture/horti_vegetables_bittergourd.html) preferred market 20–25 cm; hybrids 29–33 cm |
| Ridge gourd | curved fruit length (mm) | 250 | — (binary) | Research variety data | [IIVR](https://icariivr.org.in/ridge-gourd-2) fruits 20–30 cm (to 40 cm trained) |
| Bottle gourd | curved fruit length (mm) | 350 | — (binary) | Research variety data | [IIVR](https://icariivr.org.in/bottle-gourd-2) ~30 cm varieties; commercial hybrids 35–50 cm; mat limit 500 mm |
| Green/French bean | curved pod length (mm) | 140 | — (binary) | Research variety data | IIVR pole pods 14–15 cm; market 12–15 cm (guar/cluster pods similar) |
| Ivy gourd | curved fruit length (mm) | 50 | — (binary) | Research variety data | Tender harvest 4–5 cm; mature 2.5–6 cm |
| Pointed gourd | curved fruit length (mm) | 100 | — (binary) | Research variety data | Marketable 8–12 cm (range 5–15 cm) |
| Sweet potato | root length (mm) | 180 | — (binary) | Research variety data | Genotype range 8.9–24.8 cm ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9582706)); commercial 15–20 cm |
| Pumpkin / ash gourd | — | none | none | Exceeds station | Fruits exceed the measurement mat; larger station required before any size claim |

**Registry totals:** 24 crops — 21 gradeable with published references, 3 honestly
refused (coriander: undefined unit; pumpkin, ash gourd: exceed the mat).

## Interpretation rules encoded in the config

1. **Metric must match the evidence.** CODEX/USDA diameters are widths, so the
   potato/onion/tomato policies grade on `min_area_rect_short_mm` (projected
   minimum axis); okra grades on pod length (`centerline_length_mm`); cucumber
   on projected long axis; spinach on a flat-laid single leaf blade.
2. **Bands come from the source's own classes** (onion 40/60 grader classes,
   CODEX code groups, USDA minimum vs long types) — not from splitting a range
   into thirds.
3. **Binary cut placement is stated in `basis`.** Midpoints (potato/onion) or
   standard code boundaries (tomato 57 mm) are chosen where the source gives a
   class interval, and the choice is written down next to the number.
4. **Tier labels rank the evidence**: OFFICIAL_STANDARD > INDUSTRY_TRADE_PRACTICE
   ≈ OFFICIAL_STANDARD_PLUS_RESEARCH > RESEARCH_VARIETY_DATA >
   HORTICULTURE_REFERENCE. Spinach's horticulture tier is the weakest and is
   flagged as such in its provenance block.

## What these numbers are not

- Not AGMARK grade documents (AGMARK per-commodity rules exist for e.g.
  onion/tomato; the official DMI texts should replace these references when
  obtained — see the APEDA/AGMARK portal).
- Not validated on any local population; a median from one market is not a
  universal boundary (the project's own doctrine).
- Not variety-, season-, or buyer-aware. Cultivars shift these distributions
  (e.g., long-pod okra varieties vs 8–10 cm market types, which is exactly why
  okra's cut sits at 120 mm between the two clusters).

Measured ground truth always wins: when reviewed original photos with physical
measurements exist, `scripts/validate_real_size.py` overrides these references
with train-split specimen medians under `EMPIRICAL_REFERENCE` labelling.
