"""Locked new-set H1 and exploratory pooled H2; writes only this revision."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,csv,hashlib,importlib.util,sys,itertools
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('original_analysis',ROOT/'scripts/analyze_results.py')
aa=importlib.util.module_from_spec(spec); sys.modules[spec.name]=aa; spec.loader.exec_module(aa)

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(d): return [json.loads(l) for l in (d/'raw.jsonl').read_text('utf-8').splitlines() if l.strip()]
def writecsv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def comparison(rows,key):
    metric=next(m for m in aa.METRICS if m.key==key)
    return aa.paired_comparison([r[key+'_PERCEPTUAL'] for r in rows],
                               [r[key+'_QEM'] for r in rows],metric,n_boot=0)

def main():
    newdir=HERE/'heldout'
    newenv=json.loads((newdir/'environment.json').read_text())
    if newenv['status']!='completed': raise RuntimeError('Do not analyse an incomplete experiment')
    olddirs=[ROOT/'results/evaluation'/s for s in ['quality_A','quality_B','quality_C']]
    errors=[]; allrows=[]; assets=[]
    environments=[]
    equivalence=json.loads((HERE/'active_code_equivalence.json').read_text())
    assert equivalence['all_function_AST_hashes_equal'] and equivalence['whole_module_AST_equal_except_DEFAULT_WEIGHTS']
    for d in [*olddirs,newdir]:
        env=json.loads((d/'environment.json').read_text())
        environments.append({'path':str(d.relative_to(ROOT)),'sha256':digest(d/'environment.json'),'status':env['status'],
                             'source_hashes_equal_to_new':env['source_sha256']==newenv['source_sha256'],
                             'blender_equal_to_new':env['blender_version']==newenv['blender_version'] and env['blender_build_hash']==newenv['blender_build_hash'],
                             'config':env['config']})
        rr=read(d)
        errors.extend([r for r in rr if r['record_type']=='error'])
        if d==newdir: assets=[r for r in rr if r['record_type']=='asset']
        q=[r for r in rr if r['record_type']=='quality' and r['experiment']=='main' and r['ratio']==.05]
        for asset in dict.fromkeys(r['asset_id'] for r in q):
            methods={r['method']:r for r in q if r['asset_id']==asset}
            if set(methods)!={'QEM','PERCEPTUAL','DECIMATE'}: raise RuntimeError(f'Incomplete methods: {asset}')
            label='heldout_new' if d==newdir else 'original_historical'
            row={'asset':asset,'set':label,'category':methods['QEM']['category'],
                 'source_triangles':methods['QEM']['source_triangles'],'target_triangles':methods['QEM']['target_triangles']}
            for method,r in methods.items():
                row['triangles_'+method]=r['actual_triangles']
                row['target_reached_'+method]=r['target_reached']
                for short,long in [('iou_multiview','silhouette_iou_multiview_mean'),('roi_chamfer_sym_n','roi_chamfer_symmetric_normalized'),('chamfer_n','chamfer_mean_normalized'),('hausdorff_n','hausdorff_sampled_normalized')]:
                    row[short+'_'+method]=r['metrics'].get(long)
            for key in ['iou_multiview','roi_chamfer_sym_n','chamfer_n','hausdorff_n']:
                pv,qv=row[key+'_PERCEPTUAL'],row[key+'_QEM']
                row[key+'_delta_P_minus_Q']=None if pv is None or qv is None else pv-qv
            allrows.append(row)
    new=[r for r in allrows if r['set']=='heldout_new']
    roi=[r for r in allrows if r['roi_chamfer_sym_n_QEM'] is not None and r['roi_chamfer_sym_n_PERCEPTUAL'] is not None]
    assert len(new)==3 and len(roi)==10
    if errors: raise RuntimeError(f'Experiment errors must be audited: {errors}')
    if any(not e['blender_equal_to_new'] for e in environments):
        raise RuntimeError('Blender build differs; investigate before pooling')
    # Full hashes differ only in modules audited in active_code_equivalence;
    # both old and new environments must pass the exact frozen arguments.
    for d in [*olddirs,newdir]:
        c=json.loads((d/'environment.json').read_text())['config']
        assert c['strength']==12 and c['weights']=={'saliency':0,'silhouette':1,'sharp':0,'visibility':0}
        assert c['samples']==4000 and c['seed']==20260921 and c['resolution']==512
        assert not c['preserve_seams'] and c['manual_protection'] is None and c['camera'] is None
    frozen=json.loads((HERE/'frozen_code_hashes.json').read_text())['sha256']
    changed={n: digest(ROOT/n) for n,h in frozen.items() if digest(ROOT/n)!=h}
    if changed: raise RuntimeError(f'Frozen original code changed: {changed}')
    out=HERE/'analysis'; out.mkdir(exist_ok=True)
    writecsv(out/'heldout_pairs.csv',new)
    writecsv(out/'all_original_and_new_pairs.csv',allrows)
    writecsv(out/'pooled_H2_pairs.csv',roi)
    h1=comparison(new,'iou_multiview'); h2new=comparison(new,'roi_chamfer_sym_n'); h2pool=comparison(roi,'roi_chamfer_sym_n')
    historical=[r for r in allrows if r['set']=='original_historical' and r['roi_chamfer_sym_n_QEM'] is not None]
    oldh2=comparison(historical,'roi_chamfer_sym_n')
    # Independent enumeration audits the exact rank sign-flip probability.
    def enum_p(rows,key):
        d=np.array([r[key+'_delta_P_minus_Q'] for r in rows]); d=d[d!=0]
        ranks=aa.average_ranks(abs(d)); obs=float(ranks[d>0].sum())
        sums=np.array([sum(r for r,s in zip(ranks,signs) if s) for signs in itertools.product([False,True],repeat=len(d))])
        return float(min(1,2*min(np.mean(sums<=obs),np.mean(sums>=obs))))
    assert h1['p']==enum_p(new,'iou_multiview')
    assert h2pool['p']==enum_p(roi,'roi_chamfer_sym_n')
    integrity={'error_records':len(errors),'new_asset_records':assets,
               'original_and_new_environment_comparison':environments,'original_code_unchanged':not changed,
               'protocol_sha256':digest(HERE/'PROSPECTIVE_PROTOCOL.md'),'orientation_sha256':digest(HERE/'orientation.json'),
               'manifest_sha256':digest(HERE/'manifest.json'),'raw_sha256':digest(newdir/'raw.jsonl')}
    summary={'generated_at':datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds'),
             'H1_new_only':h1,'H2_new_only_descriptive':h2new,'H2_original_historical_n7_unchanged':oldh2,
             'H2_pooled_n10_exploratory':h2pool,'interpretation':[
                 'H1 new n=3 cannot reach two-sided p<.05; smallest possible p=.25.',
                 'New source filenames/content were absent from the recorded prior local project; lifetime exposure cannot be proven.',
                 'Added assets are all sculpted statues; diversity is limited.',
                 'New ROI is a fixed upper-quarter geometric proxy, not validated semantic importance.',
                 'Pooled H2 uses seven historical original pairs plus three new pairs; full module fingerprints differ but the active function bodies and explicit arguments are verified equivalent.',
                 'Historical perception.py was reconstructed to its exact recorded SHA-256: only tuning comments and unused default weights differ. pipeline.py/ui.py functions are not invoked by the harness. See active_code_equivalence.json.',
                 'Pooled H2 mixes original ROI rules and new upper-quarter ROI; it is exploratory and does not replace original confirmatory H2.',
                 'Quality-pass processing times are not a controlled repeated-timing measurement.'],
             'independent_exact_p_enumeration_verified':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
    (out/'integrity.json').write_text(json.dumps(integrity,indent=2),'utf-8')
    lines=['# Revision held-out replication and exploratory H2 extension','',f"Generated {summary['generated_at']}",'',
           'All frozen original source modules remained unchanged during this revision. Blender build matches the historical runs.',
           'Historical full module fingerprints differ for perception.py, pipeline.py and ui.py, but the active computation is unchanged: exact historical-hash reconstruction proves perception function bodies equal, both runs supply explicit weights/strength, and pipeline/ui functions are not invoked. See active_code_equivalence.json.',
           'The sole retained target was 5%; capture-only intermediate targets in simplify_levels do not alter its collapse sequence.',
           '', '| Asset | source tris | QEM / PERCEPTUAL / DECIMATE tris | QEM IoU | PERCEPTUAL IoU | delta IoU | delta symmetric ROI Chamfer |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in new:
        lines.append(f"| {r['asset']} | {r['source_triangles']} | {r['triangles_QEM']} / {r['triangles_PERCEPTUAL']} / {r['triangles_DECIMATE']} | {r['iou_multiview_QEM']:.9f} | {r['iou_multiview_PERCEPTUAL']:.9f} | {r['iou_multiview_delta_P_minus_Q']:+.9f} | {r['roi_chamfer_sym_n_delta_P_minus_Q']:+.9g} |")
    lines += ['',f"H1 new set (n=3): mean delta {h1['mean_diff']:+.9g}; HL delta {h1['hl_diff']:+.9g}; two-sided exact p={h1['p']}; W/T/L={h1['wins']}/{h1['ties']}/{h1['losses']} at IoU tolerance .0005. No CI is reported.",
              '',f"H2 historical original n=7 remains p={oldh2['p']}. New-only H2 (descriptive n=3) p={h2new['p']}, mean delta {h2new['mean_diff']:+.9g}.",
              '',f"Exploratory pooled H2 (n=10): mean delta {h2pool['mean_diff']:+.9g}; HL delta {h2pool['hl_diff']:+.9g}; CI [{h2pool['ci_low']:+.9g}, {h2pool['ci_high']:+.9g}], attained level {h2pool['ci_level']}; exact two-sided p={h2pool['p']}; W/T/L={h2pool['wins']}/{h2pool['ties']}/{h2pool['losses']} with 1% relative tolerance.",
              '',*['- '+s for s in summary['interpretation']]]
    (out/'report.md').write_text('\n'.join(lines)+'\n','utf-8')
    files={str(p.relative_to(HERE)):digest(p) for p in HERE.rglob('*') if p.is_file() and p.name!='checksums.json'}
    (HERE/'checksums.json').write_text(json.dumps(files,indent=2),'utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
