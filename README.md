# Importance-weighted QEM for Blender: research supplement

Repository: https://github.com/SeedFlora/importance-weighted-qem-blender

Code, protocols, asset provenance and numeric results for **Importance-Weighted Quadric Error Metrics for Perceptual Level-of-Detail Generation in Blender**, by Rafael Cristan Lijaya and Budi Juarto.

The training-free method multiplies each accumulated QEM vertex quadric by `1 + lambda * importance`. Its default importance signal measures occlusion-aware outer silhouettes over 64 hemisphere views. The unweighted control uses the same engine with lambda zero. This repository preserves the evidence behind the paper; it contains no manuscript or third-party meshes.

## What the results support

The original evaluation has nine assets excluded from parameter tuning. At the 5% budget, weighted QEM raises 13-view silhouette IoU on all nine (exact p=0.00390625, Holm p=0.0078125; median silhouette-error reduction 12.7%). Three gains exceed the descriptive tie tolerance. The original ROI Chamfer hypothesis is unsupported (seven assets, p=0.46875). Development profiling and code-review exposure limit the original test's confirmatory interpretation; the recorded protocols retain that history.

Three additional assets—Nefertiti, Lucy and Happy Buddha—were evaluated after locking the method and protocol. All gain silhouette IoU, but n=3 gives p=0.25, the smallest attainable two-sided value, so this is not a significant confirmation. Global Chamfer and sampled Hausdorff worsen on all three. The exploratory ten-pair ROI pool gives five improvements and five regressions, p=1.00. It mixes the original ROI rules with the additional assets' fixed upper-quarter rule and does not replace the original seven-pair test.

These are objective mesh measurements. No participant ratings were collected, and silhouette fidelity does not establish viewer preference. The painted-mask results are secondary and circular because the supplied protection mask is also the evaluated region. The original serial repeated-timing study supplies the processing-cost evidence; additional-asset quality-pass times are observational only.

## Files and evidence

| Location | Contents |
|---|---|
| `smart_lod/` | Eight original research modules and the Blender extension manifest, version 0.2.0 |
| `scripts/` | Ten original frozen acquisition, preprocessing, experiment, analysis, extension/demo and study-tool scripts; one new read-only publication verifier |
| `tests/` | Existing synthetic core/statistical tests and optional Blender integration suites; no participant data |
| `dataset/manifest.json` | Original 12-asset preprocessing, split, geometry-hash and ROI metadata |
| `dataset/external/SOURCES.json` | Original external source URLs, pinned hashes and credits, without geometry |
| `results/tuning/` | Original protocol/amendment, mechanical selector, selection output and five configurations' numeric records; `archive_pre_review/` is explicitly superseded tuning |
| `results/evaluation/PREREGISTRATION.md` | Original internally recorded hypotheses, amendment and known contamination disclosure |
| `results/evaluation/quality_A`, `quality_B`, `quality_C` | 162 quality records plus three asset records each: main, ablation and strength sweep |
| `results/evaluation/timing/` | 324 timing records including warm-ups, plus nine asset records; separate from quality-pass times |
| `results/evaluation/explore_W1/` | Exploratory old-weight comparison, not used to choose the reported configuration |
| `results/evaluation/mask_assisted*` | Numeric secondary painted-mask comparisons and their summaries |
| `results/evaluation/analysis/` | Original numeric result tables and report; chart/model renders are omitted |
| `revision-2026-11-03/experiments/` | Prospective additional-asset protocol, orientation, frozen code hashes, compatibility audit, source provenance, raw held-out records and derived paired analyses |
| `docs/RESEARCH.md` | Original technical guide; read it with the final amended protocols and the evidence distinctions above |
| `docs/USER_STUDY.csv` | Header-only schema template; it contains no ratings |
| `PUBLICATION_MANIFEST.json` | File-level original/public SHA256 and explicit public-derivative changes |

Asset inputs, `.obj`/`.blend` files, LOD outputs, previews/model renders, the manuscript, thesis, identity/author files, browser/tool logs, administrative evidence and private study directories are intentionally absent. No cloud service or API key is required.

## Integrity and public derivatives

The committed `.gitattributes` uses `* -text` so clones preserve archive bytes regardless of `core.autocrlf`; do not renormalise frozen research files.

All 18 code files listed in `frozen_code_hashes.json`, the extension manifest, numeric analysis tables, asset manifests and frozen protocol/hash records retain their original bytes. Original raw and environment files remain local. Their public JSON/JSONL derivatives remove recorded `argv` and replace personal absolute project paths with repository-relative paths; numeric measurements, hardware/software facts, seeds, source hashes and experiment settings are unchanged. `PUBLICATION_MANIFEST.json` records both source and public hashes and each affected field.

Historical integrity/checksum files are retained as records of the original archive. Embedded hashes can therefore refer to original metadata, private filenames or intentionally non-redistributed mesh inputs; they are not a manifest of files currently distributed here. The new publication manifest describes the actual public files. This is an explicit relocation/redaction record, not a claim that historical source hashes equal public derivative hashes.

The full pre-experiment filename inventory is not published because it names private identity/browser/thesis files. Its labelled `prior_inventory.public.json` derivative retains an original-record hash and redaction count. The protocol's no-prior-recorded-use claim has its original local scope; no claim about lifetime author exposure is made. Checksums are integrity aids, not a trusted timestamp service. No archive DOI is claimed.

## Check the numeric archive without Blender

The recorded environment was Blender 5.1.2 (build `ec6e62d40fa9`), Python 3.13.9 and NumPy 2.3.4 on Windows 11, Intel Core i7-12700H, 20 logical cores. Original repeated timing used one warm-up and three serial repetitions in randomised method order. These device/implementation-specific measurements compare Python/NumPy QEM with Blender's compiled Decimate.

Use Python 3.10 or newer with NumPy. SciPy and Matplotlib are optional in the original analysis code (SciPy is needed to reproduce its recorded SciPy implementation path). A local check is:

```sh
python scripts/verify_publication.py
```

The verifier checks distributed file hashes, all 18 frozen code hashes, the prospective protocol hash, experiment settings and the original/additional/pooled test summaries. It reads the archive and produces console output only: no downloads, Blender runs or archive overwrites.

For the existing plain Python geometry suites:

```sh
python scripts/run_tests.py --skip-blender --only test_qem test_raster test_perception
```

The statistical suite also contains a toy Blender harness test that can auto-detect a local Blender installation; it is separate from the no-Blender verifier. The remaining Blender suites can be run with `python scripts/run_tests.py --blender /path/to/blender` when that validation is intended.

## Reproduce analysis or meshes

Work in a disposable copy so generated files do not overwrite the archived records. Preserve the directory structure: the additional-experiment scripts resolve the root through their parent directories, and their hash check expects all ten original scripts and eight original modules.

Recompute the original result tables into a new directory:

```sh
python scripts/analyze_results.py results/evaluation/quality_A results/evaluation/quality_B results/evaluation/quality_C results/evaluation/timing --out reproduction/original-analysis --holm-metrics iou_multiview,roi_chamfer_sym_n --holm-baselines QEM --holm-ratios 0.05
```

Recompute the additional three-pair H1 and exploratory pooled H2 using the archived raw records:

```sh
python revision-2026-11-03/experiments/analyze_replication.py
```

The unchanged historical additional-experiment script was recorded on Windows and uses backslash-separated frozen-hash paths. Run it on Windows; on POSIX, use the portable read-only publication verifier instead unless you create and disclose a separate path-normalising wrapper. The historical script is retained byte-for-byte.

The latter writes new analysis files and checksums in that disposable copy. Its source-code freeze and explicit parameter checks remain active. The archived compatibility documents explain the historical/full-file fingerprint difference rather than concealing it.

For original mesh reconstruction, acquire the original sources with `python scripts/fetch_assets.py`, then use Blender to run `scripts/make_dataset.py`; see `dataset/README.md`. For additional inputs, acquire the three URLs from `revision-2026-11-03/experiments/external/SOURCES.json`, verify each SHA256, place them at the listed filenames under that `external/` directory and run `build_heldout_dataset.py` in Blender. `orientation.json` records the source-only rotations. Rebuilding can change timestamps or Blender-file serialisation; compare source/geometry hashes and use the recorded Blender build for matching preprocessing.

`prepare_replication.py` is the original prospective-record creation tool. It deliberately refuses to overwrite the archived protocol; do not use it to recreate a fictional historical record. `run_all.sh` preserves the original evaluation commands and output locations; choose fresh outputs for a new run. Mesh and timing experiments have not been rerun for this public release.

## Licence and third-party sources

The existing project code licence is **GPL-3.0-or-later**, as specified in `LICENSE`, module headers and `smart_lod/blender_manifest.toml`. This release does not assign a new licence to third-party meshes. The upstream collection's attributions and URLs are provenance, not a redistribution grant. Review the original providers' terms before acquiring, using or redistributing their geometry or renders. No third-party geometry or model renders are included here.
