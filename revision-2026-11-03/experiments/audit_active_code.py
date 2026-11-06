"""Resolve historical/current fingerprint differences without changing code."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,json,hashlib,difflib
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
now=datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds')
oldenv=json.loads((ROOT/'results/evaluation/quality_A/environment.json').read_text())
current=(ROOT/'smart_lod/perception.py').read_bytes()
old=(HERE/'historical_perception_reconstruction.py').read_bytes()
oldhash=hashlib.sha256(old).hexdigest()
assert oldhash==oldenv['source_sha256']['smart_lod/perception.py']
def active_ast(code):
    tree=ast.parse(code)
    tree.body=[node for node in tree.body if not (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_WEIGHTS' for t in node.targets))]
    return ast.dump(tree,include_attributes=False)
assert active_ast(current)==active_ast(old)
functions=lambda b:{n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in ast.walk(ast.parse(b)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
assert functions(current)==functions(old)
diff=''.join(difflib.unified_diff(old.decode().splitlines(True),current.decode().splitlines(True),fromfile='historical_hash_matched_perception.py',tofile='current_frozen_perception.py'))
(HERE/'historical_perception_diff.txt').write_text(diff,'utf-8')
evidence={
 'audited_at':now,'historical_environment_sha256':hashlib.sha256((ROOT/'results/evaluation/quality_A/environment.json').read_bytes()).hexdigest(),
 'historical_perception_expected_sha256':oldhash,'historical_perception_reconstructed_sha256':oldhash,
 'current_perception_sha256':hashlib.sha256(current).hexdigest(),'all_function_AST_hashes_equal':True,
 'whole_module_AST_equal_except_DEFAULT_WEIGHTS':True,'function_AST_hashes':functions(current),
 'only_executable_difference':{'historical_DEFAULT_WEIGHTS':{'saliency':.35,'silhouette':.40,'sharp':.25,'visibility':.15},'current_DEFAULT_WEIGHTS':{'saliency':0,'silhouette':1,'sharp':0,'visibility':.15}},
 'call_path':[
 'Original __init__.py has lazy UI imports only inside register/unregister; harness does not register add-on.',
 'run_experiment imports geometry, metrics, perception, qem directly; pipeline.generate and ui functions are not invoked.',
 'parse_args uses DEFAULT_WEIGHTS only when --weights is absent; old and new argv both supply explicit --weights.',
 'Asset.importance supplies weight_sets(args.weights) explicitly to compute_importance; _weights uses DEFAULT_WEIGHTS only for weights=None.',
 'Both runs supply --strength 12, bypassing default_strength reads of ui.py and pipeline.py.',
 'pipeline.py is parsed as AST only for EXTENSION_VERSION metadata. Old and new reported extension version is 0.2.0.',
 'geometry.py, qem.py, metrics.py, raster.py, __init__.py and run_experiment.py SHA-256 equal historical recorded hashes.',
 'simplify_levels capture at intermediate targets does not alter collapse sequence (original source lines 709-747); .05-only output is comparable to .05 snapshot from original three-budget run.'
 ],
 'decision':'Use historical original seven ROI pairs plus new three for exploratory H2. No old remeasurement was started. Compatibility addendum rerun plan is superseded; retain it as audit history.',
 'limits':'Full source fingerprints differ; only active computational path equivalence is established. Reconstruction hash proves bytes match recorded historical source hash, not source timestamp authenticity.'}
(HERE/'active_code_equivalence.json').write_text(json.dumps(evidence,indent=2),'utf-8')
resolution=f'''# Compatibility addendum resolution\n\nRecorded {now}, before starting any original-asset remeasurement.\n\nThe preceding rerun plan is superseded. Historical perception.py was reconstructed exactly from the current source by removing its added tuning rationale comments and restoring the old DEFAULT_WEIGHTS dictionary. Its SHA-256 matches the original environment record: {oldhash}. All function ASTs and the entire module AST excluding that defaults assignment are equal. Both historical and new harness arguments explicitly pass the same weights and strength, so the differing defaults do not enter the active computation. pipeline.py/ui.py functions are not called by this harness. See active_code_equivalence.json and historical_perception_diff.txt.\n\nTherefore use the historical original seven ROI asset pairs and new three for the predeclared exploratory pooled H2. Retain the full fingerprint mismatch disclosure and the ROI/provenance limitations. No method was altered and no remeasurement was started. The new-set H1 test is unchanged.\n'''
(HERE/'COMPATIBILITY_RESOLUTION.md').write_text(resolution,'utf-8')
print(json.dumps({k:v for k,v in evidence.items() if k!='function_AST_hashes'},indent=2))
