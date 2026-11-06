# Prospective compatibility addendum: original ROI assets remeasurement

Recorded 2026-11-03T21:23:36+07:00 before any remeasurement of the original ROI assets.

The old recorded source fingerprint differs from current frozen code for perception.py, pipeline.py and ui.py. Current code is identical to the existing dist/smart_lod-0.2.0.zip, but exact historical source is not locally archived. No source file will be reverted or adjusted. Geometry, QEM, metrics, raster and experiment harness hashes match the historical runs.

For coherent exploratory pooling, rerun all seven original ROI assets (beetle, max_planck, igea, ogre, bunny, armadillo, horse) with the SAME frozen current code and parameters as the revision hold-out: ratio .05, lambda12, silhouette1, other weights0, sample4000, seed20260921, metric raster512, seams unlocked, no camera or manual protection. The original dataset and ROI groups remain unchanged.

Use a separate output folder old_roi_remeasured. Main methods only; no tuning and no timing claims. Historical n7 remains a separate published result; pooled n10 uses remeasured original7 + new3, explicitly exploratory because of prior exposure, nonuniform ROI definitions and the code-version boundary. Compare remeasured metrics with historical values and retain all differences or failures. No results are omitted based on direction.

New-set H1 still uses only the three newly acquired meshes. The prospective method/asset/ROI plan remains unchanged. Some new-set outcomes were already produced when the fingerprint mismatch was found; this addendum changes the audit/remeasurement plan only, not the frozen method or new-set analysis.
