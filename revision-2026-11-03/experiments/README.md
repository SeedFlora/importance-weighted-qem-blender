# Additional three-asset experiment (3 November 2026)

This is the recorded Nefertiti, Lucy and Happy Buddha evaluation. The configuration was fixed before these comparisons: silhouette-only weighting, strength 12, 5% budget, 4,000 metric samples, seed 20260921 and 512px masks. The new ROI is the upper quarter of upright Z extent and is used only for measurement.

Read `PROSPECTIVE_PROTOCOL.md`, `protocol_sha256.json`, `orientation.json` and `frozen_code_hashes.json` first. The protocol and frozen code-hash files are unchanged historical records. `analysis/summary.json`, `heldout_pairs.csv` and `pooled_H2_pairs.csv` distinguish the three new H1 pairs from the exploratory ten-pair H2 pool.

The three new silhouette gains have exact two-sided p=0.25 (the minimum at n=3), so they are not a significant confirmation. Global Chamfer and sampled Hausdorff worsen on all three. The pooled H2 result remains unsupported (p=1.00) and mixes historical ROI definitions with the new upper-quarter rule.

`COMPATIBILITY_ADDENDUM.md` records a contemplated remeasurement that did not occur; `COMPATIBILITY_RESOLUTION.md` supersedes it after historical-hash reconstruction. `active_code_equivalence.json` documents equality of the active computation despite historical full-file fingerprints differing. Do not remove this distinction.

Public environment/asset-record derivatives omit argv and replace personal project paths with repository-relative paths. No numeric measurement, code fingerprint, hardware/software version or seed was changed. `PUBLICATION_MANIFEST.json` at the repository root records original and public hashes. Hashes embedded in historical integrity files refer to the original local records, not automatically to these public derivatives.

The full prior filename inventory is withheld because it names private browser/identity/thesis files. `prior_inventory.public.json` identifies its source hash and the redaction count. Neither the original inventory nor its public derivative establishes authors' lifetime exposure to these common models.

Source OBJ/Blender files, LOD outputs and source-orientation renders are intentionally absent. `external/SOURCES.json` retains pinned URL/hash/credit provenance. Collection attribution is not a licence grant.

For a read-only check use `python scripts/verify_publication.py` from the repository root. For the original analysis, work in a disposable copy and run `python revision-2026-11-03/experiments/analyze_replication.py`; it writes analysis files and checksums. Keep this directory depth unchanged because its scripts resolve the project root through parent directories. All 18 frozen code files are provided byte-for-byte.

Do not run `prepare_replication.py` over this archived experiment: it deliberately refuses to overwrite its prospective protocol. For mesh reconstruction, acquire each file named in `external/SOURCES.json`, verify its recorded SHA256, then run `build_heldout_dataset.py` in a disposable copy with Blender. The recorded source orientations are in `orientation.json`. No full mesh experiment is run as part of this publication preparation.
