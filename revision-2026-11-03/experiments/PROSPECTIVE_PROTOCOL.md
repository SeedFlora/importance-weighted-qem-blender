# Prospective held-out replication protocol

Recorded 2026-11-03T21:19:52+07:00 (Asia/Bangkok, UTC+07:00), before new-source preprocessing or simplification.

Assets selected: Nefertiti, Lucy and Happy Buddha, pinned at common-3d-test-models commit 8a4f8642acaf43f9cd7b67858a1502e1055ef202.
No instance or mention was found in the preceding project's recorded files (prior_inventory.json).
This audit establishes no recorded use in this local project; the authors' lifetime exposure cannot be established from local storage.
The set is a new revision-time hold-out, not a retrospectively uncontaminated original study.
All three are sculpted statues; the added-set diversity is limited.

## Frozen method and preprocessing

Original source modules and harness remain unchanged; hashes in frozen_code_hashes.json.
Invoke the original scripts/run_experiment.py against a separate revision manifest.
Original make_dataset.py utilities may be imported without editing them to build only the revision assets.
Use the original preprocessing: OBJ Y-up to Blender Z-up (x,y,z) -> (x,-z,y), loose-geometry removal, bbox centring, longest side 2, fixed 100000-triangle cap, triangulation, final normalisation.
Inspect source-only orientation and dimensions before runs. Any additional fixed Euler rotation must be recorded in orientation.json before the first simplification run.
Apply the cap identically as shared source preprocessing; no source preprocessing is selected from comparative outcomes.

Freeze lambda=12; weights saliency=0, silhouette=1, sharp=0, visibility=0; quadric mode.
No protection input, no camera, no seam/material preservation.
Retained ratio 0.05 only; sample count 4000; metric seed 20260921; silhouette evaluation resolution 512; original 13-view IoU metric.
No sweep, tuning, ablation or method/code adjustment.
Run main methods DECIMATE, QEM and PERCEPTUAL once per asset. DECIMATE is descriptive.
Quality-pass times are not a controlled timing benchmark. Existing sequential repeated-timing evidence remains the cost evidence.

## Fixed ROI before outcomes

For every source, LOD_EvalROI selects vertices with z >= z_min + 0.75*(z_max-z_min), after upright orientation and normalisation.
The rule is geometric and identical for all three assets. It is an upper-quarter proxy, not a validated semantic-importance annotation.
ROI is used only for evaluation and is never supplied to simplification.
Do not change it after viewing method outcomes; do not omit a result because of its sign.

## Locked analysis and interpretation

H1 replication: PERCEPTUAL minus QEM 13-view IoU at ratio 0.05 on NEW n=3 only; two-sided exact Wilcoxon signed-rank test, alpha .05.
Report raw pairs, deltas, Hodges-Lehmann shift and win/tie/loss with original IoU tolerance .0005. No confidence interval for n=3.
Minimum attainable two-sided p with n=3 is .25, so this set cannot establish significant confirmatory evidence. Describe direction and magnitude only.
H2: symmetric normalized ROI Chamfer. Report new n=3 descriptively, then pooled original seven ROI assets + three new assets (n=10), two-sided exact Wilcoxon, as exploratory analysis.
Pooled H2 combines old manually geometric semantic ROI rules with the new upper-quarter proxy, further limiting its interpretation. Do not relabel the original n=7 result as confirmatory n=10.
Keep original study tests/results separate. Original H1/H2 Holm family is unchanged.
Report source and actual output triangle budgets, geometry and ROI integrity checks, code/environment fingerprints, all failures and all assets.
