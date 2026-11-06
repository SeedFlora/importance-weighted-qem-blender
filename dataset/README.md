# Original asset provenance

`manifest.json` is the unchanged original 12-asset manifest: three tuning assets (car, suzanne and spot), nine evaluation assets, preprocessing settings, geometry hashes, ROI rules and split membership. `external/SOURCES.json` records source URLs, source SHA256, credits and the pinned upstream collection commit. It contains metadata only; external geometry is not included.

The original acquisition/build commands, run from a disposable copy of the repository, are:

```sh
python scripts/fetch_assets.py
blender --background --factory-startup --python-exit-code 1 --python scripts/make_dataset.py
blender --background --factory-startup --python-exit-code 1 --python scripts/make_dataset.py -- --verify-only
```

A rebuild can update generated manifests/timestamps; compare geometry/source hashes rather than assuming byte-identical Blender files. The original experiments used Blender 5.1.2, so rebuilding with another version may change preprocessing.

The ten external meshes and their renders are not redistributed. Their collection credits do not grant a redistribution licence; obtain and assess each original provider's terms before use. GPL-3.0-or-later applies to the project code, not to third-party meshes. Car is procedural and Suzanne is a Blender primitive, but no geometry is included in this numeric/code supplement. The three additional assets have separate provenance under `revision-2026-11-03/experiments/external/SOURCES.json`.

The full ROI/preprocessing specification is in the manifest. `docs/RESEARCH.md` is retained as a historical technical guide; the root README and recorded protocols distinguish final original settings from the additional experiment and from superseded tuning.
