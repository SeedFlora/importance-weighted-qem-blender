"""Secondary, descriptive experiment: does the manual protection mask work?

Pre-registered in results/evaluation/PREREGISTRATION.md (Amandemen 1). On the
character_face evaluation assets, PERCEPTUAL is run once without and once with a
manual mask equal to the evaluation ROI (``LOD_EvalROI``), which simulates a user
painting the face with *Paint Important Areas*. Because the protected region is
also the measured region, the design is circular by construction: it shows that
the manual control works and what it costs elsewhere, not automatic perception.

Run (project root):
  blender --background --factory-startup --python-exit-code 1 --python scripts/run_mask_experiment.py -- \
      --out results/evaluation/mask_assisted
"""
import argparse
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smart_lod import geometry, metrics, perception, qem  # noqa: E402

ROI_GROUP = 'LOD_EvalROI'
KEYS = ('chamfer_mean_normalized', 'hausdorff_sampled_normalized', 'silhouette_iou_multiview_mean',
        'roi_chamfer_symmetric_normalized', 'nonroi_chamfer_ref_to_candidate_normalized')


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--assets', default='max_planck,igea,ogre')
    parser.add_argument('--ratios', default='0.5,0.2,0.05')
    parser.add_argument('--strength', type=float, default=12.0)
    parser.add_argument('--weights', default='saliency=0,silhouette=1,sharp=0')
    parser.add_argument('--samples', type=int, default=4000)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20260921)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'dataset' / 'manifest.json').read_text(encoding='utf-8'))
    assets = manifest['assets']  # {asset_id: entry}
    entries = assets if isinstance(assets, dict) else {entry['id']: entry for entry in assets}
    ratios = [float(r) for r in args.ratios.split(',')]
    weights = {name: float(value) for name, value in (item.split('=') for item in args.weights.split(','))}
    rows = []
    for asset_id in args.assets.split(','):
        entry = entries[asset_id]
        bpy.ops.wm.open_mainfile(filepath=str(ROOT / 'dataset' / entry['file']), load_ui=False)
        source = bpy.data.objects[entry['object']]
        mesh, vertices, faces = geometry.snapshot(source, bpy.context.evaluated_depsgraph_get())
        locked, marked_sharp = geometry.feature_constraints(mesh, False)  # primary protocol: seams unlocked
        roi = np.clip(geometry.group_weights(mesh, source, ROI_GROUP), 0.0, 1.0)
        targets = [max(4, int(round(len(faces) * ratio))) for ratio in ratios]
        auto, _ = perception.compute_importance(vertices, faces, weights=weights, marked_sharp=marked_sharp)
        masked, _ = perception.compute_importance(vertices, faces, weights=weights, manual=roi,
                                                  marked_sharp=marked_sharp)
        conditions = {'QEM': (None, 0.0), 'PERCEPTUAL': (auto, args.strength),
                      'PERCEPTUAL_MASK': (masked, args.strength)}
        for name, (importance, strength) in conditions.items():
            clock = time.perf_counter()
            levels = qem.simplify_levels(vertices, faces, targets, importance, strength, locked)
            seconds = time.perf_counter() - clock
            for level, (ratio, result) in enumerate(zip(ratios, levels), 1):
                values = metrics.evaluate_meshes(vertices, faces, result.vertices, result.faces,
                                                 sample_count=args.samples, resolution=args.resolution,
                                                 seed=args.seed, roi_weights=roi, multiview=True)
                row = dict(asset_id=asset_id, condition=name, level=level, ratio=ratio,
                           target_triangles=targets[level - 1], actual_triangles=int(len(result.faces)),
                           roi_vertices=int(np.count_nonzero(roi)), pass_seconds=seconds,
                           **{key: values.get(key) for key in KEYS})
                rows.append(row)
                print(json.dumps(row), flush=True)
        bpy.data.meshes.remove(mesh)
    (out / 'raw.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    # Relative change of each condition vs QEM at the same asset and level.
    lines = ['# Eksperimen sekunder: bantuan mask manual (deskriptif, sirkular secara desain)', '',
             'Perubahan relatif terhadap QEM pada aset dan rasio yang sama (negatif = lebih kecil).', '',
             '| Aset | Rasio | Kondisi | Segitiga | ROI Chamfer sim. | non-ROI Chamfer | Chamfer global | IoU 13-view |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    base = {(r['asset_id'], r['level']): r for r in rows if r['condition'] == 'QEM'}
    for row in rows:
        if row['condition'] == 'QEM':
            continue
        b = base[(row['asset_id'], row['level'])]
        rel = lambda key: f"{100 * (row[key] / b[key] - 1):+.1f} %"
        lines.append(f"| {row['asset_id']} | {row['ratio']:g} | {row['condition']} | {row['actual_triangles']} | "
                     f"{rel('roi_chamfer_symmetric_normalized')} | {rel('nonroi_chamfer_ref_to_candidate_normalized')} | "
                     f"{rel('chamfer_mean_normalized')} | "
                     f"{row['silhouette_iou_multiview_mean'] - b['silhouette_iou_multiview_mean']:+.5f} |")
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('MASK_EXPERIMENT_DONE', out)


if __name__ == '__main__':
    main()
