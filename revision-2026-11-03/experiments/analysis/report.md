# Revision held-out replication and exploratory H2 extension

Generated 2026-11-03T21:27:24+07:00

All frozen original source modules remained unchanged during this revision. Blender build matches the historical runs.
Historical full module fingerprints differ for perception.py, pipeline.py and ui.py, but the active computation is unchanged: exact historical-hash reconstruction proves perception function bodies equal, both runs supply explicit weights/strength, and pipeline/ui functions are not invoked. See active_code_equivalence.json.
The sole retained target was 5%; capture-only intermediate targets in simplify_levels do not alter its collapse sequence.

| Asset | source tris | QEM / PERCEPTUAL / DECIMATE tris | QEM IoU | PERCEPTUAL IoU | delta IoU | delta symmetric ROI Chamfer |
|---|---:|---:|---:|---:|---:|---:|
| nefertiti | 99938 | 4996 / 4996 / 4996 | 0.997294177 | 0.997442158 | +0.000147981 | +2.2812215e-06 |
| lucy | 99970 | 4998 / 4998 / 4998 | 0.992088817 | 0.993721736 | +0.001632919 | -5.19978687e-05 |
| happy-buddha | 98601 | 4929 / 4929 / 4929 | 0.993230037 | 0.994538436 | +0.001308399 | -7.79587493e-05 |

H1 new set (n=3): mean delta +0.00102976627; HL delta +0.00109942449; two-sided exact p=0.25; W/T/L=2/1/0 at IoU tolerance .0005. No CI is reported.

H2 historical original n=7 remains p=0.46875. New-only H2 (descriptive n=3) p=0.5, mean delta -4.25584655e-05.

Exploratory pooled H2 (n=10): mean delta +4.8736964e-06; HL delta -3.03502675e-06; CI [-3.78387639e-05, +5.44417776e-05], attained level 0.951171875; exact two-sided p=1.0; W/T/L=5/0/5 with 1% relative tolerance.

- H1 new n=3 cannot reach two-sided p<.05; smallest possible p=.25.
- New source filenames/content were absent from the recorded prior local project; lifetime exposure cannot be proven.
- Added assets are all sculpted statues; diversity is limited.
- New ROI is a fixed upper-quarter geometric proxy, not validated semantic importance.
- Pooled H2 uses seven historical original pairs plus three new pairs; full module fingerprints differ but the active function bodies and explicit arguments are verified equivalent.
- Historical perception.py was reconstructed to its exact recorded SHA-256: only tuning comments and unused default weights differ. pipeline.py/ui.py functions are not invoked by the harness. See active_code_equivalence.json.
- Pooled H2 mixes original ROI rules and new upper-quarter ROI; it is exploratory and does not replace original confirmatory H2.
- Quality-pass processing times are not a controlled repeated-timing measurement.
