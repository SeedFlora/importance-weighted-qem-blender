"""Acquire untouched asset bytes and freeze a prospective revision protocol.

This script does not invoke either simplification method. All writes are confined
to this revision directory. Source meshes are research measurement inputs only;
their renders and geometry are not redistributed in the paper package.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json, urllib.request, re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TZ = timezone(timedelta(hours=7))
NAMES = ['nefertiti', 'lucy', 'happy-buddha']
REPO = 'alecjacobson/common-3d-test-models'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent':'SmartLOD-scientific-revision/1'})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()

def main():
    if (HERE/'PROSPECTIVE_PROTOCOL.md').exists():
        raise RuntimeError('Archived prospective protocol already exists; do not overwrite the recorded run. Copy scripts to a new run directory for a new replication.')
    stamp = datetime.now(TZ).isoformat(timespec='seconds')
    external = HERE/'external'
    external.mkdir(exist_ok=True)
    commit_data = get(f'https://api.github.com/repos/{REPO}/commits/master')
    (external/'upstream_commit.json').write_bytes(commit_data)
    commit = json.loads(commit_data)['sha']
    readme = get(f'https://raw.githubusercontent.com/{REPO}/{commit}/README.md')
    (external/'upstream_README.md').write_bytes(readme)
    # Search all earlier text/code/data and file names, excluding the revision
    # itself and generic bibliography metadata. This proves no recorded project
    # use, not that no author has ever encountered a common model elsewhere.
    prior = []
    audit_extensions = {'.py','.json','.jsonl','.md','.tex','.csv','.log','.txt','.obj','.blend'}
    for path in ROOT.rglob('*'):
        if not path.is_file() or 'revision-2026-11-03' in path.parts:
            continue
        rel = str(path.relative_to(ROOT))
        if path.suffix.lower() in audit_extensions:
            prior.append({'path':rel,'sha256':sha(path)})
            if any(name.replace('-','') in path.name.lower().replace('-','').replace('_','') for name in NAMES):
                raise RuntimeError(f'New asset already appears as file: {rel}')
            if path.suffix.lower() not in {'.obj','.blend'}:
                text = path.read_text('utf-8',errors='replace').lower()
                hits = [name for name in NAMES if re.search(r'\b'+re.escape(name)+r'\b',text)]
                if hits:
                    raise RuntimeError(f'New asset name already used in project: {rel}: {hits}')
    (HERE/'prior_inventory.json').write_text(json.dumps({'captured_at':stamp,'files':prior},indent=2),'utf-8')
    frozen = {str(p.relative_to(ROOT)):sha(p) for p in [*sorted((ROOT/'smart_lod').glob('*.py')),*sorted((ROOT/'scripts').glob('*.py'))]}
    (HERE/'frozen_code_hashes.json').write_text(json.dumps({'captured_at':stamp,'sha256':frozen},indent=2),'utf-8')
    sources=[]
    for name in NAMES:
        upstream_name = 'happy' if name == 'happy-buddha' else name
        url=f'https://raw.githubusercontent.com/{REPO}/{commit}/data/{upstream_name}.obj'
        data=get(url)
        path=external/f'{name}.obj'
        path.write_bytes(data)
        identity_lines=[line for line in readme.decode().splitlines() if name in line.lower() or (name=='happy-buddha' and 'happy buddha' in line.lower())]
        sources.append({'name':name,'file':str(path.name),'url':url,'sha256':sha(path),'bytes':len(data),'retrieved_at':datetime.now(TZ).isoformat(timespec='seconds'),'upstream_commit':commit,'readme_credit_lines':identity_lines,'collection':f'https://github.com/{REPO}','usage_note':'Measurement only. Collection attribution is not a licence grant; no source geometry or render is redistributed.'})
    (external/'SOURCES.json').write_text(json.dumps({'models':sources},indent=2),'utf-8')
    protocol=f'''# Prospective held-out replication protocol

Recorded {datetime.now(TZ).isoformat(timespec='seconds')} (Asia/Bangkok, UTC+07:00), before new-source preprocessing or simplification.

Assets selected: Nefertiti, Lucy and Happy Buddha, pinned at common-3d-test-models commit {commit}.
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
'''
    (HERE/'PROSPECTIVE_PROTOCOL.md').write_text(protocol,'utf-8')
    (HERE/'protocol_sha256.json').write_text(json.dumps({'recorded_at':datetime.now(TZ).isoformat(timespec='seconds'),'sha256':sha(HERE/'PROSPECTIVE_PROTOCOL.md')},indent=2),'utf-8')
    print(json.dumps({'protocol':str(HERE/'PROSPECTIVE_PROTOCOL.md'),'asset_bytes':[{'name':x['name'],'bytes':x['bytes'],'sha256':x['sha256']} for x in sources],'prior_files':len(prior),'commit':commit},indent=2))

if __name__=='__main__':
    main()
