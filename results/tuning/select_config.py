"""Apply the pre-registered tuning rule in PROTOCOL.md to results/tuning/*/raw.jsonl.

Plain CPython + NumPy. Prints one line per (weight set, lambda) and the selection.
QEM reference rows come from W1_default (identical engine run; LAMBDA_0 must equal QEM).
"""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RATIOS = (0.2, 0.05)


def load(path):
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('record_type') == 'quality' and row.get('metrics'):
            rows.append(row)
    return rows


def key(row):
    return row['asset_id'], round(row['ratio'], 6)


qem = {key(r): r for r in load(HERE / 'W1_default' / 'raw.jsonl') if r['condition'] == 'QEM'}
table = []
for run in sorted(p for p in HERE.iterdir() if (p / 'raw.jsonl').is_file()):
    for row in load(run / 'raw.jsonl'):
        if row['experiment'] != 'sweep' or round(row['ratio'], 6) not in RATIOS:
            continue
        base = qem.get(key(row))
        if base is None:
            continue
        m, b = row['metrics'], base['metrics']
        roi = (m.get('roi_chamfer_ref_to_candidate_normalized'), b.get('roi_chamfer_ref_to_candidate_normalized'))
        table.append(dict(
            config=run.name, lam=row['params'].get('strength'), asset=row['asset_id'], ratio=row['ratio'],
            tri_diff=row['actual_triangles'] - base['actual_triangles'],
            d_iou13=m['silhouette_iou_multiview_mean'] - b['silhouette_iou_multiview_mean'],
            d_iou6=m['silhouette_iou_mean'] - b['silhouette_iou_mean'],
            r_sil_err=(1 - m['silhouette_iou_multiview_mean']) / (1 - b['silhouette_iou_multiview_mean']) - 1,
            r_chamfer=m['chamfer_mean_normalized'] / b['chamfer_mean_normalized'] - 1,
            r_haus=m['hausdorff_sampled_normalized'] / b['hausdorff_sampled_normalized'] - 1,
            r_roi=(roi[0] / roi[1] - 1) if None not in roi else np.nan))

groups = {}
for t in table:
    groups.setdefault((t['config'], t['lam']), []).append(t)
print(f"{'config':15s} {'lam':>5s} {'n':>3s} {'dIoU13':>9s} {'dIoU6':>9s} {'Chamfer%':>9s} {'Haus%':>8s} {'ROI%':>7s} "
      f"{'maxCh%@r':>9s} {'SilErr%':>8s} dT ok")
candidates = []
for (config, lam), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    per_ratio_chamfer = [np.mean([r['r_chamfer'] for r in rows if r['ratio'] == q]) for q in RATIOS]
    # Rule from PROTOCOL.md: Chamfer constraint only. Triangle differences are reported, not filtered.
    ok = all(c <= 0.05 for c in per_ratio_chamfer)
    summary = dict(config=config, lam=lam, n=len(rows), d_iou13=np.mean([r['d_iou13'] for r in rows]),
                   d_iou6=np.mean([r['d_iou6'] for r in rows]), chamfer=np.mean([r['r_chamfer'] for r in rows]),
                   haus=np.mean([r['r_haus'] for r in rows]), roi=np.nanmean([r['r_roi'] for r in rows]),
                   max_chamfer=max(per_ratio_chamfer), ok=ok,
                   max_tri_diff=max(abs(r['tri_diff']) for r in rows),
                   sil_err=np.mean([r['r_sil_err'] for r in rows]))
    candidates.append(summary)
    print(f"{config:15s} {lam:5.1f} {len(rows):3d} {summary['d_iou13']:+9.5f} {summary['d_iou6']:+9.5f} "
          f"{100*summary['chamfer']:+8.1f}% {100*summary['haus']:+7.1f}% {100*summary['roi']:+6.1f}% "
          f"{100*summary['max_chamfer']:+8.1f}% {100*summary['sil_err']:+7.1f}% {summary['max_tri_diff']:2d} {'Y' if ok else '-'}")
eligible = [c for c in candidates if c['ok'] and c['lam'] > 0]
positive = [c for c in eligible if c['d_iou13'] > 0]
if positive:
    best = sorted(positive, key=lambda c: (-c['d_iou13'], c['roi']))[0]
    print('\nSELECTED (max dIoU13 subject to Chamfer <= +5% at both ratios):', best['config'], 'lambda', best['lam'])
else:
    print('\nNo eligible configuration improves 13-view IoU; see PROTOCOL.md fallback rule.')
