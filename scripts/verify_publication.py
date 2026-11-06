"""Read-only integrity and numeric checks for the public Smart LOD supplement.

No Blender, downloads, output writes or mesh experiments are invoked.
Dependencies: Python >=3.10 and NumPy; SciPy is optional as in the original analyzer.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'revision-2026-11-03/experiments'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def records(p):
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]

def main():
    manifest=json.loads((ROOT/'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
    checked=0
    for entry in manifest['files']:
        p=ROOT/entry['path']
        if not p.is_file() or digest(p)!=entry['public_sha256']:
            raise RuntimeError('Public file missing or hash changed: '+entry['path'])
        checked+=1
    frozen=json.loads((HERE/'frozen_code_hashes.json').read_text())['sha256']
    for name,expected in frozen.items():
        if digest(ROOT/name.replace('\\','/'))!=expected:
            raise RuntimeError('Frozen research code changed: '+name)
    protocol=json.loads((HERE/'protocol_sha256.json').read_text())['sha256']
    assert digest(HERE/'PROSPECTIVE_PROTOCOL.md')==protocol
    spec=importlib.util.spec_from_file_location('publication_analysis',ROOT/'scripts/analyze_results.py')
    aa=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=aa
    spec.loader.exec_module(aa)
    def pairs(directory):
        rows=[r for r in records(directory/'raw.jsonl') if r.get('record_type')=='quality' and r.get('experiment')=='main' and r['ratio']==.05]
        output=[]
        for asset in dict.fromkeys(r['asset_id'] for r in rows):
            methods={r['method']:r for r in rows if r['asset_id']==asset}
            assert set(methods)=={'QEM','PERCEPTUAL','DECIMATE'}
            output.append((asset,methods))
        return output
    historical=[]
    for name in ['quality_A','quality_B','quality_C']:
        historical.extend(pairs(ROOT/'results/evaluation'/name))
    fresh=pairs(HERE/'heldout')
    assert len(historical)==9 and len(fresh)==3
    for directory in [ROOT/'results/evaluation'/n for n in ['quality_A','quality_B','quality_C']]+[HERE/'heldout']:
        env=json.loads((directory/'environment.json').read_text())
        c=env['config']
        assert env['status']=='completed' and env['blender_version']=='5.1.2'
        assert c['strength']==12 and c['samples']==4000 and c['seed']==20260921 and c['resolution']==512
        assert c['weights']=={'saliency':0,'silhouette':1,'sharp':0,'visibility':0}
        assert not c['preserve_seams'] and c['manual_protection'] is None and c['camera'] is None
    def compare(group,key,long):
        valid=[m for _,m in group if m['QEM']['metrics'].get(long) is not None and m['PERCEPTUAL']['metrics'].get(long) is not None]
        metric=next(m for m in aa.METRICS if m.key==key)
        return aa.paired_comparison([m['PERCEPTUAL']['metrics'][long] for m in valid],
                                   [m['QEM']['metrics'][long] for m in valid],metric,n_boot=0)
    h1=compare(historical,'iou_multiview','silhouette_iou_multiview_mean')
    h2=compare(historical,'roi_chamfer_sym_n','roi_chamfer_symmetric_normalized')
    new=compare(fresh,'iou_multiview','silhouette_iou_multiview_mean')
    pooled=compare(historical+fresh,'roi_chamfer_sym_n','roi_chamfer_symmetric_normalized')
    assert h1['n']==9 and math.isclose(h1['p'],.00390625,abs_tol=1e-12)
    assert h2['n']==7 and math.isclose(h2['p'],.46875,abs_tol=1e-12)
    assert new['n']==3 and math.isclose(new['p'],.25,abs_tol=1e-12)
    assert pooled['n']==10 and math.isclose(pooled['p'],1.,abs_tol=1e-12)
    archived=json.loads((HERE/'analysis/summary.json').read_text())
    for label,result in [('H1_new_only',new),('H2_original_historical_n7_unchanged',h2),('H2_pooled_n10_exploratory',pooled)]:
        for key in ['n','mean_diff','hl_diff','p','wins','ties','losses']:
            expected=archived[label][key]
            assert math.isclose(result[key],expected,rel_tol=1e-10,abs_tol=1e-12),(label,key,result[key],expected)
    for _,m in fresh:
        assert m['PERCEPTUAL']['metrics']['silhouette_iou_multiview_mean']>m['QEM']['metrics']['silhouette_iou_multiview_mean']
        for key in ['chamfer_mean_normalized','hausdorff_sampled_normalized']:
            assert m['PERCEPTUAL']['metrics'][key]>m['QEM']['metrics'][key]
    print(json.dumps({'status':'PASS','manifest_files_checked':checked,'frozen_code_files':len(frozen),
      'prospective_protocol_hash':'PASS','original_H1':{'n':h1['n'],'p':h1['p']},
      'original_H2':{'n':h2['n'],'p':h2['p']},'additional_H1':{'n':new['n'],'p':new['p']},
      'pooled_H2_exploratory':{'n':pooled['n'],'p':pooled['p']},
      'new_assets_all_IoU_positive_and_both_global_distances_worse':True,
      'measurement_runs_or_downloads':False},indent=2))

if __name__=='__main__':
    main()
