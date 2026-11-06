"""Revision-only dataset assembly using unmodified original preprocessing."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json, importlib.util, hashlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('original_make_dataset',ROOT/'scripts'/'make_dataset.py')
ds=importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
ds.DATASET=HERE
ds.EXTERNAL=HERE/'external'
ds.ASSET_DIR=HERE/'assets'
ds.MANIFEST=HERE/'manifest.json'
orientation=json.loads((HERE/'orientation.json').read_text('utf-8'))
original_hash=json.loads((HERE/'frozen_code_hashes.json').read_text('utf-8'))['sha256']
for name,expected in original_hash.items():
    actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    if actual != expected: raise RuntimeError(f'Original code changed: {name}')
sources={}
for r in json.loads((HERE/'external'/'SOURCES.json').read_text('utf-8'))['models']:
    sources[r['name']]={**r,'source_credit': '; '.join(r['readme_credit_lines']),
                        'source_link':r['collection'],'first_known_appearance':None}
assets={}
for name in ['nefertiti','lucy','happy-buddha']:
    rotation=orientation['assets'][name]['rotation_after_OBJ_axes_xyz_degrees']
    # After normalisation, the upright Z span may be less than 2; compute the
    # rule threshold from the actual bbox, not from a presumed height.
    # make_dataset normalises the longest side, and these upright statues have
    # height as their longest side (confirmed from source-only dimensions).
    ds.ROI[name]={'description':'Fixed geometric upper-quarter proxy, not validated semantic importance',
                  'union':[{'box':[[-1.01,-1.01,0.5],[1.01,1.01,1.01]]}]}
    asset_spec={'id':name,'obj':name,'category':'sculpted_statue','split':'evaluation',
                'rotate':rotation,'front':'source yaw retained; full multiview evaluation'}
    entry=ds.build_asset(asset_spec,sources)
    low,high=entry['bbox_min'][2],entry['bbox_max'][2]
    if abs(low+1)>1e-4 or abs(high-1)>1e-4:
        raise RuntimeError(f'{name}: expected upright height longest; bbox {low,high}; no experiments may run')
    assets[name]=entry
manifest={'schema_version':1,'generated_by':str(Path(__file__).name),
          'generated_at':datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds'),
          'blender_version':ds.bpy.app.version_string,'assets':assets,
          'splits':{'evaluation':list(assets),'tuning':[]},
          'roi_definition':{'group':'LOD_EvalROI','use':'evaluation only, never simplification input',
                            'rule':'z >= z_min + .75*(z_max-z_min), fixed before method outcomes',
                            'status':'geometric proxy; no independent semantic validation'},
          'protocol_sha256':hashlib.sha256((HERE/'PROSPECTIVE_PROTOCOL.md').read_bytes()).hexdigest()}
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n','utf-8')
print(json.dumps({'manifest':str(HERE/'manifest.json'),'assets':{n:{'triangles':e['triangles'],'roi':e['roi']} for n,e in assets.items()}},indent=2))
