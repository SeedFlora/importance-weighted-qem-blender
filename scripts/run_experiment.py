"""Research harness: run DECIMATE / QEM / PERCEPTUAL on the dataset and log raw results.

Run inside Blender (background mode); arguments follow ``--``::

    blender --background --factory-startup --python scripts/run_experiment.py -- \
        --split tuning --out results/tuning_v02

The harness calls the extension modules directly (``geometry``, ``perception``,
``qem``, ``metrics``), not ``pipeline.generate``, so that every condition uses
the same static snapshot and the timing experiment can shuffle method order.
The DECIMATE baseline replicates ``pipeline._decimate`` (Triangulate + Decimate
COLLAPSE with ``use_collapse_triangulate`` on a temporary object). Timing is
like-for-like per method: DECIMATE counts the temporary object, modifier
evaluation and ``new_from_object`` (the LOD is then a Blender mesh); reading it
back into NumPy (``world_arrays``) is needed only to measure it and is timed
apart as ``readback_seconds``, outside ``processing_seconds``. QEM/PERCEPTUAL
count what they need beyond the shared snapshot: the source mesh -> NumPy
conversion (``input_arrays_seconds``), the seam/material lock mask
(``feature_constraints``, ``constraints_seconds``), importance (PERCEPTUAL),
the pass and the NumPy -> mesh rebuild; both extra components are also
reported separately. (Up to schema 1, DECIMATE included the readback and
QEM/PERCEPTUAL excluded the locks and the input conversion.)

Conditions are *automatic*: no manual protection group and no camera, so the
visibility channel is inactive and ``compute_importance`` renormalizes the
remaining weights. UV seams and material borders are locked for QEM and
PERCEPTUAL as in the extension default; Blender's Decimate has no such lock, so
on UV-mapped assets the methods can stop at different triangle counts
(``--no-preserve-seams`` runs the unconstrained comparison). The dataset's
``LOD_EvalROI`` group is read only for the ROI metrics and is never passed to
any simplification method.

Outputs in ``--out``: ``raw.jsonl`` (one JSON record per line; ``record_type``
is ``quality``, ``timing``, ``asset`` or ``error``; every record carries the
``session_id`` that wrote it), ``environment.json`` and, with ``--save-lods``,
``lods/<asset_id>.blend``. ``scripts/analyze_results.py`` turns them into
tables and figures. ``level`` is the 1-based position of the ratio in
``--ratios``; rows also carry the ratio itself.

``--resume`` continues a run only when the result-affecting settings
(``RESUME_KEYS``) and the code fingerprint of every earlier session
(``CODE_KEYS``: sha256 of ``smart_lod/*.py`` and this script, Blender version
and build hash, extension version) match. ``--allow-code-change`` overrides the
code check and records the differences in the session; the analysis then
flags the run as mixed.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix  # noqa: E402

from smart_lod import geometry, metrics, perception, qem  # noqa: E402

SCHEMA_VERSION = 2  # 2: session_id on every record, like-for-like timing components
EXPERIMENTS = ('main', 'ablation', 'sweep', 'timing')
MAIN_METHODS = ('DECIMATE', 'QEM', 'PERCEPTUAL')
ROI_GROUP = 'LOD_EvalROI'
ANCHOR_V01_STRENGTH = 8.0  # the v0.1 extension default for the positional anchor
ABLATION_WEIGHT_SETS = ('no_saliency', 'no_silhouette', 'no_sharp', 'only_saliency', 'only_silhouette')
IDENTITY = Matrix.Identity(4)


def log(message):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {message}', flush=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def json_safe(value):
    """Plain JSON types only; nonfinite floats become None."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


# --- configuration --------------------------------------------------------------------------

def _literal(node):
    try:
        return float(ast.literal_eval(node))
    except (ValueError, TypeError, SyntaxError):
        return None


def default_strength():
    """The extension's default lambda, read from its source instead of duplicated here.

    Order: literal ``default=`` of ``SMARTLOD_Settings.strength`` in ui.py, the
    module constant ``DEFAULT_STRENGTH`` in pipeline.py, the ``strength``
    default of ``qem.simplify``. Returns ``(value, where)``.
    """
    try:
        tree = ast.parse((ROOT / 'smart_lod' / 'ui.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'SMARTLOD_Settings':
                for item in node.body:
                    if (isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                            and item.target.id == 'strength' and isinstance(item.annotation, ast.Call)):
                        for keyword in item.annotation.keywords:
                            if keyword.arg == 'default' and _literal(keyword.value) is not None:
                                return _literal(keyword.value), 'smart_lod/ui.py SMARTLOD_Settings.strength default'
    except (OSError, SyntaxError):
        pass
    try:
        tree = ast.parse((ROOT / 'smart_lod' / 'pipeline.py').read_text(encoding='utf-8'))
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'DEFAULT_STRENGTH' and _literal(node.value) is not None):
                return _literal(node.value), 'smart_lod/pipeline.py DEFAULT_STRENGTH'
    except (OSError, SyntaxError):
        pass
    return float(inspect.signature(qem.simplify).parameters['strength'].default), 'smart_lod/qem.py simplify() default'


def parse_list(text, kind, name):
    try:
        values = [kind(item) for item in text.split(',') if item.strip()]
    except ValueError as exc:
        raise SystemExit(f'--{name}: {exc}')
    if not values:
        raise SystemExit(f'--{name} must not be empty')
    return values


def parse_weights(text):
    weights = {name: 0.0 for name in perception.CHANNELS}
    for item in text.split(','):
        if not item.strip():
            continue
        name, _, value = item.partition('=')
        name = name.strip()
        if name not in weights:
            raise SystemExit(f'--weights: unknown channel {name!r}; use {", ".join(perception.CHANNELS)}')
        try:
            weights[name] = float(value)
        except ValueError:
            raise SystemExit(f'--weights: {item!r} is not name=number')
        if not math.isfinite(weights[name]) or weights[name] < 0:
            raise SystemExit('--weights must be finite and nonnegative')
    return weights


def parse_args(argv):
    parser = argparse.ArgumentParser(prog='run_experiment.py', description=__doc__.split('\n\n')[0])
    parser.add_argument('--manifest', default=str(ROOT / 'dataset' / 'manifest.json'))
    parser.add_argument('--split', required=True, choices=('tuning', 'evaluation', 'all'))
    parser.add_argument('--assets', default='', help='comma-separated asset ids (filter within --split)')
    parser.add_argument('--out', required=True, help='output directory')
    parser.add_argument('--experiments', default=','.join(EXPERIMENTS))
    parser.add_argument('--ratios', default='0.5,0.2,0.05', help='retained triangle fractions, decreasing')
    parser.add_argument('--reps', type=int, default=3, help='timed repetitions per method (timing)')
    parser.add_argument('--warmup', type=int, default=1, help='untimed-in-analysis warm-up repetitions (timing)')
    parser.add_argument('--seed', type=int, default=20260921, help='metric sampling seed and timing-order seed')
    parser.add_argument('--samples', type=int, default=4000, help='surface samples per mesh (metrics)')
    parser.add_argument('--resolution', type=int, default=128, help='silhouette mask resolution (metrics)')
    parser.add_argument('--strength', type=float, default=None, help='lambda for PERCEPTUAL (default: extension)')
    parser.add_argument('--weights', default=None, help="e.g. 'saliency=0.35,silhouette=0.4,sharp=0.25'")
    parser.add_argument('--lambdas', default='0,2,5,10,20,50', help='lambda values for the sweep')
    parser.add_argument('--save-lods', action='store_true', help='write lods/<asset_id>.blend')
    parser.add_argument('--resume', action='store_true', help='skip (asset, condition) groups already in raw.jsonl')
    parser.add_argument('--allow-code-change', action='store_true',
                        help='let --resume continue although code, Blender or extension differ from earlier sessions')
    parser.add_argument('--no-preserve-seams', action='store_true',
                        help='do not lock UV seams/material borders for QEM and PERCEPTUAL (Decimate never locks them)')
    args = parser.parse_args(argv)
    args.experiments = parse_list(args.experiments, str, 'experiments')
    unknown = sorted(set(args.experiments) - set(EXPERIMENTS))
    if unknown:
        raise SystemExit(f'--experiments: unknown {unknown}; use {",".join(EXPERIMENTS)}')
    args.ratios = parse_list(args.ratios, float, 'ratios')
    if not all(1 > a > b > 0 for a, b in zip(args.ratios, args.ratios[1:])) or not 0 < args.ratios[0] < 1:
        raise SystemExit('--ratios must be strictly decreasing within (0, 1)')
    args.lambdas = parse_list(args.lambdas, float, 'lambdas')
    # The engine refuses strength outside [0, qem.MAX_STRENGTH]; refuse here, before any asset work.
    limit = f'[0, {qem.MAX_STRENGTH:g}] (smart_lod.qem.MAX_STRENGTH)'
    if any(not 0 <= x <= qem.MAX_STRENGTH for x in args.lambdas) or len(set(args.lambdas)) != len(args.lambdas):
        raise SystemExit(f'--lambdas must be distinct and finite, within {limit}')
    if args.reps < 1 or args.warmup < 0 or args.samples < 1 or args.resolution < 8 or args.seed < 0:
        raise SystemExit('--reps >= 1, --warmup >= 0, --samples >= 1, --resolution >= 8, --seed >= 0')
    args.strength_source = '--strength'
    if args.strength is None:
        args.strength, args.strength_source = default_strength()
    if not 0 <= args.strength <= qem.MAX_STRENGTH:  # also rejects NaN and inf
        raise SystemExit(f'--strength must be finite, within {limit}')
    args.weights = dict(perception.DEFAULT_WEIGHTS) if args.weights is None else parse_weights(args.weights)
    args.assets = [item.strip() for item in args.assets.split(',') if item.strip()]
    return args


def lambda_label(value):
    return f'LAMBDA_{value:g}'.replace('.', 'p')


class Condition:
    def __init__(self, experiment, name, method, weight_set=None, strength=0.0, weighting='quadric', note=None):
        self.experiment, self.name, self.method = experiment, name, method
        self.weight_set, self.strength, self.weighting, self.note = weight_set, float(strength), weighting, note

    @property
    def key(self):
        """Simplification cache key: identical keys give identical arrays (deterministic engine)."""
        return (self.weight_set, self.strength, self.weighting)

    def params(self, weight_sets, preserve_seams=True):
        if self.method == 'DECIMATE':
            return dict(engine='blender_decimate_collapse', use_collapse_triangulate=True, triangulate_modifier=True)
        params = dict(engine='smart_lod.qem.simplify_levels', weighting=self.weighting, strength=self.strength,
                      weight_set=self.weight_set, locked=f'feature_constraints(preserve_seams={preserve_seams})',
                      manual_protection=None, camera=None)
        if self.weight_set is not None:
            params['weights'] = weight_sets[self.weight_set]
        if self.note:
            params['note'] = self.note
        return params


def build_conditions(args):
    conditions = []
    if 'main' in args.experiments:
        conditions += [Condition('main', 'DECIMATE', 'DECIMATE'), Condition('main', 'QEM', 'QEM'),
                       Condition('main', 'PERCEPTUAL', 'PERCEPTUAL', 'default', args.strength)]
    if 'ablation' in args.experiments:
        conditions += [Condition('ablation', name.upper(), 'PERCEPTUAL', name, args.strength)
                       for name in ABLATION_WEIGHT_SETS]
        conditions.append(Condition(
            'ablation', 'ANCHOR_V01', 'PERCEPTUAL', 'default', ANCHOR_V01_STRENGTH, 'anchor',
            note='v0.1 anchor (positional point-quadric) weighting at the v0.1 default strength 8, fed with the '
                 'CURRENT v0.2 importance; v0.1 importance code (curvature, axis silhouette) differs, so this '
                 'isolates the weighting rule, not the whole v0.1 method'))
        conditions.append(Condition('ablation', 'LAMBDA0', 'PERCEPTUAL', 'default', 0.0,
                                    note='control: importance passed with strength 0; must equal QEM exactly'))
    if 'sweep' in args.experiments:
        conditions += [Condition('sweep', lambda_label(value), 'PERCEPTUAL', 'default', value)
                       for value in args.lambdas]
    return conditions


def weight_sets(base):
    sets = {'default': dict(base)}
    for name in ('saliency', 'silhouette', 'sharp'):
        sets['no_' + name] = {**base, name: 0.0}
        sets['only_' + name] = {channel: float(channel == name) for channel in perception.CHANNELS}
    return {name: sets[name] for name in ('default',) + ABLATION_WEIGHT_SETS}


def effective_weights(weights, camera=None):
    """Weights after compute_importance's rule: only active channels, renormalized to sum 1."""
    active = {name: value for name, value in weights.items()
              if value > 0 and (name != 'visibility' or camera is not None)}
    total = sum(active.values())
    return {name: value / total for name, value in active.items()} if total > 0 else {}


# --- environment ------------------------------------------------------------------------------

def cpu_name():
    try:
        if sys.platform == 'win32':
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        if sys.platform.startswith('linux'):
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
        if sys.platform == 'darwin':
            import subprocess
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or None
    except Exception:  # best effort only
        pass
    return platform.processor() or None


def total_ram_bytes():
    try:
        if sys.platform == 'win32':
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in (
                        'total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual',
                        'avail_virtual', 'avail_extended')]
            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_phys)
        elif hasattr(os, 'sysconf'):
            return int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    except Exception:
        pass
    return None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extension_version():
    info = {}
    try:
        import tomllib
        with open(ROOT / 'smart_lod' / 'blender_manifest.toml', 'rb') as handle:
            info['manifest_version'] = tomllib.load(handle).get('version')
    except Exception as exc:
        info['manifest_version_error'] = str(exc)
    try:
        tree = ast.parse((ROOT / 'smart_lod' / 'pipeline.py').read_text(encoding='utf-8'))
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'EXTENSION_VERSION'):
                info['pipeline_extension_version'] = ast.literal_eval(node.value)
    except Exception:
        pass
    return info


def environment(args, argv):
    code = {path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in sorted((ROOT / 'smart_lod').glob('*.py')) + [Path(__file__).resolve()]}
    return dict(
        blender_version=bpy.app.version_string,
        blender_build_hash=bpy.app.build_hash.decode('ascii', 'replace') if isinstance(bpy.app.build_hash, bytes)
        else str(bpy.app.build_hash),
        python_version=platform.python_version(), numpy_version=np.__version__,
        platform=platform.platform(), machine=platform.machine(), cpu_name=cpu_name(),
        logical_cores=os.cpu_count(), ram_bytes=total_ram_bytes(),
        thread_env={name: os.environ.get(name) for name in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS') if os.environ.get(name)},
        extension=extension_version(), source_sha256=code, argv=list(argv),
    )


def run_config(args, manifest_sha):
    return dict(
        manifest=str(Path(args.manifest).resolve()), manifest_sha256=manifest_sha, split=args.split,
        assets_filter=args.assets, experiments=args.experiments, ratios=args.ratios, reps=args.reps,
        warmup=args.warmup, seed=args.seed, samples=args.samples, resolution=args.resolution,
        strength=args.strength, strength_source=args.strength_source, weights=args.weights,
        effective_weights=effective_weights(args.weights), lambdas=args.lambdas,
        anchor_v01_strength=ANCHOR_V01_STRENGTH, save_lods=args.save_lods,
        roi_group=ROI_GROUP, roi_threshold=0.5, multiview=True, manual_protection=None, camera=None,
        preserve_seams=not args.no_preserve_seams, target_rule='max(4, round(source_triangles * ratio))',
        importance_defaults={name: parameter.default for name, parameter in
                             inspect.signature(perception.compute_importance).parameters.items()
                             if name in ('sharp_angle', 'silhouette_views', 'silhouette_resolution')},
        timing_order_rng='numpy.random.default_rng([seed, asset_index, rep]).permutation(3)',
    )


# Settings that change results; a resumed run must match them.
RESUME_KEYS = ('manifest_sha256', 'ratios', 'seed', 'samples', 'resolution', 'strength', 'weights',
               'reps', 'warmup', 'preserve_seams')
# Code identity of a session; a resumed run must match every earlier session unless --allow-code-change.
CODE_KEYS = ('source_sha256', 'blender_version', 'blender_build_hash', 'extension')


def code_differences(previous_sessions, current):
    """Readable differences in CODE_KEYS between earlier sessions and the current environment."""
    found = []
    for session in previous_sessions:
        name = session.get('session_id')
        for key in CODE_KEYS:
            old, new = session.get(key), current.get(key)
            if old == new:
                continue
            if key == 'source_sha256' and isinstance(old, dict) and isinstance(new, dict):
                found += [f'{path} (session {name})' for path in sorted(set(old) | set(new))
                          if old.get(path) != new.get(path)]
            else:
                found.append(f'{key} {old!r} -> {new!r} (session {name})')
    return list(dict.fromkeys(found))


# --- raw log ----------------------------------------------------------------------------------

class RawLog:
    def __init__(self, path, session_id=None):
        self.path, self.session_id = path, session_id
        if path.exists() and path.stat().st_size:
            with open(path, 'rb') as handle:
                handle.seek(-1, os.SEEK_END)
                needs_newline = handle.read(1) != b'\n'  # a crash can leave a partial last line
        else:
            needs_newline = False
        self.handle = open(path, 'a', encoding='utf-8', newline='\n')
        if needs_newline:
            self.handle.write('\n')

    def write(self, record):
        record = dict(record, session_id=self.session_id) if self.session_id else record
        self.handle.write(json.dumps(json_safe(record), allow_nan=False) + '\n')
        self.handle.flush()

    def close(self):
        self.handle.close()


def completed_groups(path, ratios, timing_rows):
    """(asset, experiment, condition) groups with every ratio logged, and assets with complete timing."""
    quality, timing = {}, {}
    wanted = {round(ratio, 12) for ratio in ratios}
    if path.exists():
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                kind, asset = record.get('record_type'), record.get('asset_id')
                if kind == 'quality':
                    key = (asset, record.get('experiment'), record.get('condition'))
                    quality.setdefault(key, set()).add(round(float(record.get('ratio', -1)), 12))
                elif kind == 'timing':
                    timing.setdefault(asset, set()).add(
                        (record.get('condition'), round(float(record.get('ratio', -1)), 12), record.get('rep')))
    done = {key for key, found in quality.items() if wanted <= found}
    done_timing = {asset for asset, found in timing.items() if len(found) >= timing_rows}
    return done, done_timing


# --- geometry helpers -------------------------------------------------------------------------

def signed_volume(vertices, faces):
    triangles = vertices[faces]
    return float(np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0)


def is_closed(faces):
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(len(counts)) and bool(np.all(counts == 2))


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(mean=None, nonzero_fraction=None, gt_half_fraction=None)
    return dict(mean=float(values.mean()), nonzero_fraction=float(np.mean(values > 0)),
                gt_half_fraction=float(np.mean(values > 0.5)))


def decimate_mesh(source_mesh, ratio):
    """Blender Decimate baseline, replicating v0.1/v0.2 ``pipeline._decimate``."""
    temporary = bpy.data.objects.new('_SmartLOD_temporary', source_mesh.copy())
    bpy.context.scene.collection.objects.link(temporary)
    temporary.matrix_world = IDENTITY
    temporary_mesh = temporary.data
    try:
        temporary.modifiers.new('Triangulate input', 'TRIANGULATE')
        modifier = temporary.modifiers.new('Unweighted Decimate baseline', 'DECIMATE')
        modifier.decimate_type = 'COLLAPSE'
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = temporary.evaluated_get(depsgraph)
        return bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
    finally:
        bpy.data.objects.remove(temporary, do_unlink=True)
        if temporary_mesh.users == 0:
            bpy.data.meshes.remove(temporary_mesh)


def same_arrays(a, b):
    return (np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces)
            and np.array_equal(a.source_faces, b.source_faces))


# --- per-asset state --------------------------------------------------------------------------

class Asset:
    """One opened dataset asset with its snapshot and caches of deterministic results."""

    def __init__(self, asset_id, index, entry, manifest_dir, args):
        self.id, self.index, self.entry, self.args = asset_id, index, entry, args
        self.weight_sets = weight_sets(args.weights)
        clock = time.perf_counter()
        bpy.ops.wm.open_mainfile(filepath=str(manifest_dir / entry['file']), load_ui=False)
        self.open_seconds = time.perf_counter() - clock
        source = bpy.data.objects.get(entry['object'])
        if source is None or source.type != 'MESH':
            raise ValueError(f'object {entry["object"]!r} not found or not a mesh')
        clock = time.perf_counter()
        self.mesh, self.vertices, self.faces = geometry.snapshot(source, bpy.context.evaluated_depsgraph_get())
        self.snapshot_seconds = time.perf_counter() - clock
        self.sha256 = hashlib.sha256(self.vertices.tobytes() + self.faces.tobytes()).hexdigest()
        clock = time.perf_counter()
        self.locked, self.marked_sharp = geometry.feature_constraints(self.mesh, not args.no_preserve_seams)
        self.constraints_seconds = time.perf_counter() - clock
        self.roi = None
        if source.vertex_groups.get(ROI_GROUP) is not None:
            self.roi = np.clip(geometry.group_weights(self.mesh, source, ROI_GROUP), 0.0, 1.0)
        self.source_triangles = len(self.faces)
        self.targets = [max(4, int(round(self.source_triangles * ratio))) for ratio in args.ratios]
        self.closed = is_closed(self.faces)
        self.volume = signed_volume(self.vertices, self.faces) if self.closed else None
        self.importance_cache, self.simplify_cache, self.metric_cache = {}, {}, {}
        self.decimate_cache = {}  # level -> (mesh or None, vertices, faces, seconds, readback_seconds)
        self.checks = {}

    def base_row(self):
        return dict(asset_id=self.id, asset_index=self.index, category=self.entry.get('category'),
                    split=self.entry.get('split'), source_triangles=self.source_triangles,
                    source_vertices=len(self.vertices))

    def importance(self, name):
        if name not in self.importance_cache:
            seconds = {}
            clock = time.perf_counter()
            values, features = perception.compute_importance(
                self.vertices, self.faces, weights=self.weight_sets[name], manual=None, camera=None,
                marked_sharp=self.marked_sharp, timings=seconds)
            total = time.perf_counter() - clock
            self.importance_cache[name] = dict(
                values=values, seconds=total, channel_seconds=seconds, weights=self.weight_sets[name],
                effective_weights=effective_weights(self.weight_sets[name]),
                summary={**{key: summary(value) for key, value in features.items()}, 'combined': summary(values)})
            log(f'  {self.id}: importance[{name}] {total:.2f} s, mean {values.mean():.3f}, '
                f'>0.5 {np.mean(values > 0.5):.3f}')
        return self.importance_cache[name]

    def simplification(self, condition):
        key = condition.key
        if key not in self.simplify_cache:
            weight_set, strength, weighting = key
            importance = None if weight_set is None else self.importance(weight_set)['values']
            clock = time.perf_counter()
            results = qem.simplify_levels(self.vertices, self.faces, self.targets, importance, strength,
                                          self.locked, weighting=weighting)
            seconds = time.perf_counter() - clock
            self.simplify_cache[key] = dict(results=results, pass_seconds=seconds,
                                            computed_for=f'{condition.experiment}/{condition.name}')
            log(f'  {self.id}: {condition.experiment}/{condition.name} pass {seconds:.2f} s -> '
                + ', '.join(str(len(result.faces)) for result in results) + ' tris')
        return self.simplify_cache[key]

    def qem(self):
        return self.simplification(Condition('main', 'QEM', 'QEM'))

    def decimate(self, level, keep_mesh):
        """Decimate LOD of ``level`` from the pristine snapshot mesh (see ``save_lods``)."""
        if level not in self.decimate_cache:
            clock = time.perf_counter()
            mesh = decimate_mesh(self.mesh, self.targets[level - 1] / self.source_triangles)
            seconds = time.perf_counter() - clock
            clock = time.perf_counter()
            vertices, faces = geometry.world_arrays(mesh, IDENTITY)
            readback = time.perf_counter() - clock
            if not keep_mesh:
                bpy.data.meshes.remove(mesh)
                mesh = None
            self.decimate_cache[level] = (mesh, vertices, faces, seconds, readback)
        return self.decimate_cache[level]

    def evaluate(self, cache_key, vertices, faces):
        if cache_key not in self.metric_cache:
            clock = time.perf_counter()
            values = metrics.evaluate_meshes(
                self.vertices, self.faces, vertices, faces, sample_count=self.args.samples,
                resolution=self.args.resolution, seed=self.args.seed, roi_weights=self.roi, multiview=True)
            volume_ratio = None
            if self.volume:
                volume_ratio = signed_volume(np.asarray(vertices, dtype=np.float64), np.asarray(faces)) / self.volume
            self.metric_cache[cache_key] = (values, volume_ratio, time.perf_counter() - clock)
        return self.metric_cache[cache_key]


# --- experiments ------------------------------------------------------------------------------

def quality_rows(asset, condition, args):
    rows = []
    common = dict(record_type='quality', experiment=condition.experiment, condition=condition.name,
                  method=condition.method, params=condition.params(asset.weight_sets, not args.no_preserve_seams))
    if condition.method == 'DECIMATE':
        for level, (ratio, target) in enumerate(zip(args.ratios, asset.targets), 1):
            _, vertices, faces, seconds, readback = asset.decimate(level, keep_mesh=args.save_lods)
            values, volume_ratio, evaluation_seconds = asset.evaluate(('DECIMATE', level), vertices, faces)
            rows.append(dict(
                common, level=level, ratio=ratio, target_triangles=target, actual_triangles=len(faces),
                target_reached=len(faces) <= target, reduction=1 - len(faces) / asset.source_triangles,
                stats=dict(decimate_ratio=target / asset.source_triangles), metrics=values,
                volume_ratio=volume_ratio, reused_from=None, checks={},
                timings=dict(simplification_and_mesh_seconds=seconds, readback_seconds=readback,
                             importance_seconds=0.0, processing_seconds=seconds, evaluation_seconds=evaluation_seconds,
                             source='single quality run (readback excluded); see timing records')))
        return rows
    cached = condition.key in asset.simplify_cache
    simplified = asset.simplification(condition)
    reused_from = simplified['computed_for'] if cached else None
    importance = None
    if condition.weight_set is not None:
        entry = asset.importance(condition.weight_set)
        importance = dict(weight_set=condition.weight_set, seconds=entry['seconds'],
                          channel_seconds=entry['channel_seconds'], effective_weights=entry['effective_weights'],
                          summary=entry['summary'])
    checks = {}
    if condition.method != 'QEM' and condition.strength == 0.0:
        reference = asset.qem()['results']
        equal = all(same_arrays(a, b) for a, b in zip(simplified['results'], reference))
        checks['equals_qem'] = equal
        asset.checks[f'{condition.experiment}/{condition.name}_equals_qem'] = equal
        if not equal:
            log(f'  WARNING {asset.id}: {condition.name} (strength 0) differs from QEM')
    for level, (ratio, target, result) in enumerate(zip(args.ratios, asset.targets, simplified['results']), 1):
        values, volume_ratio, evaluation_seconds = asset.evaluate(condition.key + (level,), result.vertices,
                                                                  result.faces)
        importance_seconds = importance['seconds'] if importance else 0.0
        rows.append(dict(
            common, level=level, ratio=ratio, target_triangles=target, actual_triangles=len(result.faces),
            target_reached=bool(result.stats['target_reached']),
            reduction=1 - len(result.faces) / asset.source_triangles, stats=result.stats, metrics=values,
            volume_ratio=volume_ratio, importance=importance, reused_from=reused_from, checks=checks,
            timings=dict(pass_seconds=simplified['pass_seconds'], level_capture_seconds=result.stats['seconds'],
                         importance_seconds=importance_seconds, constraints_seconds=asset.constraints_seconds,
                         processing_seconds_without_mesh=result.stats['seconds'] + importance_seconds
                         + asset.constraints_seconds,
                         evaluation_seconds=evaluation_seconds,
                         source='single quality run (mesh building and input arrays excluded); see timing records')))
    return rows


def time_method(asset, method, args):
    """One timed run of a main method; returns per-level timing dicts (like-for-like; module docstring)."""
    levels = []
    if method == 'DECIMATE':
        for level, target in enumerate(asset.targets, 1):
            clock = time.perf_counter()
            mesh = decimate_mesh(asset.mesh, target / asset.source_triangles)
            seconds = time.perf_counter() - clock
            clock = time.perf_counter()
            vertices, faces = geometry.world_arrays(mesh, IDENTITY)  # measurement only, not part of the method
            readback = time.perf_counter() - clock
            bpy.data.meshes.remove(mesh)
            cached = asset.decimate_cache.get(level)
            same = None if cached is None else bool(np.array_equal(cached[1], vertices)
                                                     and np.array_equal(cached[2], faces))
            levels.append(dict(actual_triangles=len(faces), matches_quality_run=same,
                               simplification_and_mesh_seconds=seconds, readback_seconds=readback,
                               importance_seconds=0.0, processing_seconds=seconds))
        total = sum(level['processing_seconds'] for level in levels)
        readback_total = sum(level['readback_seconds'] for level in levels)
        for level in levels:
            level.update(method_set_seconds=total, method_set_readback_seconds=readback_total)
        return levels
    clock = time.perf_counter()
    geometry.world_arrays(asset.mesh, IDENTITY)  # source -> arrays; asset.vertices keep the float64 snapshot values
    input_seconds = time.perf_counter() - clock
    clock = time.perf_counter()
    locked, _ = geometry.feature_constraints(asset.mesh, not args.no_preserve_seams)
    constraints_seconds = time.perf_counter() - clock
    importance, importance_seconds, channel_seconds = None, 0.0, {}
    if method == 'PERCEPTUAL':
        clock = time.perf_counter()
        importance, _ = perception.compute_importance(
            asset.vertices, asset.faces, weights=asset.weight_sets['default'], manual=None, camera=None,
            marked_sharp=asset.marked_sharp, timings=channel_seconds)
        importance_seconds = time.perf_counter() - clock
    strength = args.strength if method == 'PERCEPTUAL' else 0.0
    clock = time.perf_counter()
    results = qem.simplify_levels(asset.vertices, asset.faces, asset.targets, importance, strength, locked,
                                  weighting='quadric')
    pass_seconds = time.perf_counter() - clock
    key = ('default', strength, 'quadric') if method == 'PERCEPTUAL' else (None, 0.0, 'quadric')
    cached = asset.simplify_cache.get(key)
    mesh_seconds = []
    for result in results:
        clock = time.perf_counter()
        mesh = geometry.result_mesh('_SmartLOD_timing', result, asset.mesh, asset.vertices, asset.faces, IDENTITY)
        mesh_seconds.append(time.perf_counter() - clock)
        bpy.data.meshes.remove(mesh)
    prerequisites = input_seconds + constraints_seconds + importance_seconds
    set_seconds = prerequisites + pass_seconds + sum(mesh_seconds)
    for index, result in enumerate(results):
        capture = result.stats['seconds']
        same = None if cached is None else same_arrays(result, cached['results'][index])
        levels.append(dict(
            actual_triangles=len(result.faces), matches_quality_run=same, pass_seconds=pass_seconds,
            level_capture_seconds=capture, mesh_seconds=mesh_seconds[index],
            simplification_and_mesh_seconds=capture + mesh_seconds[index], importance_seconds=importance_seconds,
            importance_channel_seconds=channel_seconds or None, input_arrays_seconds=input_seconds,
            constraints_seconds=constraints_seconds, processing_seconds=capture + mesh_seconds[index] + prerequisites,
            method_set_seconds=set_seconds))
    return levels


def timing_rows(asset, args, raw):
    methods = list(MAIN_METHODS)
    for rep in range(args.warmup + args.reps):
        order = [methods[i] for i in np.random.default_rng([args.seed, asset.index, rep]).permutation(len(methods))]
        for position, method in enumerate(order):
            gc.collect()
            levels = time_method(asset, method, args)
            for level, (ratio, target, timing) in enumerate(zip(args.ratios, asset.targets, levels), 1):
                actual = timing.pop('actual_triangles')
                same = timing.pop('matches_quality_run')
                raw.write(dict(asset.base_row(), record_type='timing', experiment='timing', condition=method,
                               method=method, rep=rep, warmup=rep < args.warmup, order=order, position=position,
                               level=level, ratio=ratio, target_triangles=target, actual_triangles=actual,
                               matches_quality_run=same, timings=timing))
            log(f'  {asset.id}: timing rep {rep}{" (warm-up)" if rep < args.warmup else ""} {method} '
                f'set {levels[0]["method_set_seconds"]:.2f} s')


def save_lods(asset, args, out_dir):
    """Write REF__<id> and <METHOD>__L<level>__<id> static world-space meshes to lods/<id>.blend.

    The reference gets its own copy of the snapshot mesh: vertex-group names and
    color attributes live on the mesh, so adding the importance heatmap to
    ``asset.mesh`` itself would leak into Decimate LODs computed from it later
    (e.g. on ``--resume --save-lods``). Only PERCEPTUAL LODs carry the heatmap.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    collections = {}
    for name in ('REF',) + MAIN_METHODS:
        collections[name] = bpy.data.collections.new(f'{name}__{asset.id}')
        scene.collection.children.link(collections[name])
    reference = bpy.data.objects.new(f'REF__{asset.id}', asset.mesh.copy())
    collections['REF'].objects.link(reference)
    reference['smart_lod_reference'] = True
    reference['smart_lod_asset'] = asset.id
    geometry.add_importance(reference, asset.importance('default')['values'])
    if asset.roi is not None and reference.vertex_groups.get(ROI_GROUP) is None:
        group = reference.vertex_groups.new(name=ROI_GROUP)
        group.add(np.flatnonzero(asset.roi > 0).tolist(), 1.0, 'REPLACE')
    conditions = {'QEM': Condition('main', 'QEM', 'QEM'),
                  'PERCEPTUAL': Condition('main', 'PERCEPTUAL', 'PERCEPTUAL', 'default', args.strength)}
    for method in MAIN_METHODS:
        for level, (ratio, target) in enumerate(zip(args.ratios, asset.targets), 1):
            name = f'{method}__L{level}__{asset.id}'
            if method == 'DECIMATE':
                mesh = asset.decimate(level, keep_mesh=True)[0]
                if mesh is None:  # computed earlier without keeping the mesh
                    asset.decimate_cache.pop(level)
                    mesh = asset.decimate(level, keep_mesh=True)[0]
                mesh.name = name
                result = None
            else:
                result = asset.simplification(conditions[method])['results'][level - 1]
                mesh = geometry.result_mesh(name, result, asset.mesh, asset.vertices, asset.faces, IDENTITY)
            obj = bpy.data.objects.new(name, mesh)
            collections[method].objects.link(obj)
            for group_name in (ROI_GROUP, 'LOD_Importance'):  # Decimate carries source layers over
                if obj.vertex_groups.get(group_name) is not None:
                    obj.vertex_groups.remove(obj.vertex_groups[group_name])
            if mesh.color_attributes.get('LOD_Heatmap') is not None:
                mesh.color_attributes.remove(mesh.color_attributes['LOD_Heatmap'])
            if result is not None and method == 'PERCEPTUAL':
                geometry.add_importance(obj, result.importance)
            obj['smart_lod_generated'] = True
            obj['smart_lod_asset'] = asset.id
            obj['smart_lod_method'] = method
            obj['smart_lod_level'] = level
            obj['smart_lod_ratio'] = ratio
            obj['smart_lod_target'] = target
            obj['smart_lod_triangles'] = len(mesh.polygons)
            if method == 'PERCEPTUAL':
                obj['smart_lod_strength'] = args.strength
            obj.hide_set(True)
    path = out_dir / 'lods' / f'{asset.id}.blend'
    path.parent.mkdir(parents=True, exist_ok=True)
    preferences = bpy.context.preferences.filepaths
    versions = preferences.save_version
    preferences.save_version = 0  # no .blend1 backups next to the results
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True, compress=True)
    finally:
        preferences.save_version = versions
    return path


def process_asset(asset_id, index, entry, conditions, args, raw, out_dir, manifest_dir, done, done_timing):
    pending = [c for c in conditions if (asset_id, c.experiment, c.name) not in done]
    want_timing = 'timing' in args.experiments and asset_id not in done_timing
    want_lods = args.save_lods and not (args.resume and (out_dir / 'lods' / f'{asset_id}.blend').exists())
    if not pending and not want_timing and not want_lods:
        log(f'{asset_id}: all groups already in raw.jsonl, skipped (--resume)')
        return True
    started = time.perf_counter()
    log(f'{asset_id}: opening {entry["file"]} ({entry.get("triangles")} triangles, split {entry.get("split")})')
    try:
        asset = Asset(asset_id, index, entry, manifest_dir, args)
    except Exception as exc:
        raw.write(dict(record_type='error', scope='asset', asset_id=asset_id, asset_index=index,
                       category=entry.get('category'), split=entry.get('split'), experiment=None, condition=None,
                       error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
        log(f'{asset_id}: ERROR while opening: {exc}')
        return False
    ok = True
    for condition in pending:
        try:
            for row in quality_rows(asset, condition, args):
                raw.write(dict(asset.base_row(), **row))
                values = row['metrics']
                log(f'  {asset_id}: {condition.experiment}/{condition.name} L{row["level"]} '
                    f'{row["actual_triangles"]}/{row["target_triangles"]} tris, chamfer '
                    f'{values["chamfer_mean_normalized"]:.3e}, H {values["hausdorff_sampled_normalized"]:.3e}, '
                    f'IoU {values["silhouette_iou_mean"]:.4f}/{values["silhouette_iou_multiview_mean"]:.4f}'
                    + (' (reused)' if row.get('reused_from') else ''))
        except Exception as exc:
            ok = False
            raw.write(dict(asset.base_row(), record_type='error', scope='condition', experiment=condition.experiment,
                           condition=condition.name, error=f'{type(exc).__name__}: {exc}',
                           traceback=traceback.format_exc()))
            log(f'  {asset_id}: ERROR in {condition.experiment}/{condition.name}: {exc}')
    determinism = None
    try:
        first = asset.qem()
        clock = time.perf_counter()
        repeat = qem.simplify_levels(asset.vertices, asset.faces, asset.targets, None, 0.0, asset.locked,
                                     weighting='quadric')
        determinism = dict(checked='QEM simplify_levels repeated once', seconds=time.perf_counter() - clock,
                           identical=all(same_arrays(a, b) for a, b in zip(first['results'], repeat)))
        asset.checks['qem_repeat_identical'] = determinism['identical']
        if not determinism['identical']:
            log(f'  WARNING {asset_id}: repeated QEM pass differs')
    except Exception as exc:
        ok = False
        raw.write(dict(asset.base_row(), record_type='error', scope='determinism', experiment=None, condition=None,
                       error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
    if want_timing:
        try:
            timing_rows(asset, args, raw)
        except Exception as exc:
            ok = False
            raw.write(dict(asset.base_row(), record_type='error', scope='timing', experiment='timing', condition=None,
                           error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
            log(f'  {asset_id}: ERROR in timing: {exc}')
    lod_path = None
    if want_lods:
        try:
            lod_path = save_lods(asset, args, out_dir)
            log(f'  {asset_id}: saved {lod_path}')
        except Exception as exc:
            ok = False
            raw.write(dict(asset.base_row(), record_type='error', scope='save_lods', experiment=None, condition=None,
                           error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
            log(f'  {asset_id}: ERROR while saving LODs: {exc}')
    roi_expected = (entry.get('roi') or {}).get('vertices')
    roi_found = None if asset.roi is None else int(np.count_nonzero(asset.roi > 0))
    record = dict(
        asset.base_row(), record_type='asset', file=entry['file'], object=entry['object'],
        geometry_sha256=asset.sha256, geometry_sha256_matches_manifest=asset.sha256 == entry.get('geometry_sha256'),
        roi_vertices=roi_found, roi_vertices_match_manifest=roi_found == roi_expected,
        roi_area_fraction_manifest=(entry.get('roi') or {}).get('area_fraction'),
        locked_vertices=int(np.count_nonzero(asset.locked)),
        marked_sharp_vertices=int(np.count_nonzero(asset.marked_sharp)),
        closed=asset.closed, reference_volume=asset.volume, targets=asset.targets, ratios=args.ratios,
        open_seconds=asset.open_seconds, snapshot_seconds=asset.snapshot_seconds,
        importance={name: {key: value for key, value in entry_.items() if key != 'values'}
                    for name, entry_ in asset.importance_cache.items()},
        determinism=determinism, checks=asset.checks,
        conditions_run=[f'{c.experiment}/{c.name}' for c in pending], timing_run=want_timing,
        lods_file=str(lod_path) if lod_path else None, asset_wall_seconds=time.perf_counter() - started)
    asset.checks['geometry_sha256_matches_manifest'] = record['geometry_sha256_matches_manifest']
    if roi_expected is not None or roi_found is not None:
        asset.checks['roi_vertices_match_manifest'] = record['roi_vertices_match_manifest']
    raw.write(record)
    if not record['geometry_sha256_matches_manifest']:
        log(f'  WARNING {asset_id}: snapshot hash differs from manifest geometry_sha256')
    if record['roi_vertices_match_manifest'] is False and roi_expected is not None:
        log(f'  WARNING {asset_id}: ROI vertex count {roi_found} differs from manifest {roi_expected}')
    log(f'{asset_id}: done in {record["asset_wall_seconds"]:.1f} s')
    checks_ok = all(value is not False for value in asset.checks.values())
    return ok and checks_ok


def main(argv):
    args = parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest_sha = sha256_file(manifest_path)
    order = list(manifest['assets'])
    selected = [a for a in order if args.split == 'all' or manifest['assets'][a].get('split') == args.split]
    if args.assets:
        unknown = sorted(set(args.assets) - set(order))
        if unknown:
            raise SystemExit(f'--assets: not in the manifest: {unknown}')
        outside = sorted(set(args.assets) - set(selected))
        if outside:
            raise SystemExit(f'--assets: not in split {args.split!r}: {outside}')
        selected = [a for a in selected if a in args.assets]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path, env_path = out_dir / 'raw.jsonl', out_dir / 'environment.json'
    config = run_config(args, manifest_sha)
    current = environment(args, argv)
    sessions, changes = [], []
    has_rows = raw_path.exists() and raw_path.stat().st_size
    if has_rows and not args.resume:
        raise SystemExit(f'{raw_path} exists; pass --resume to continue it or choose another --out')
    if args.resume and env_path.exists():
        previous = json.loads(env_path.read_text(encoding='utf-8'))
        mismatched = [key for key in RESUME_KEYS if previous.get('config', {}).get(key) != config[key]]
        if mismatched:
            raise SystemExit(f'--resume: settings differ from the existing run: {mismatched}')
        sessions = previous.get('sessions', [])
        changes = code_differences(sessions or [previous], current)
    elif args.resume and has_rows:
        changes = [f'{env_path.name} missing: code and settings of the existing rows are unknown']
    if changes and not args.allow_code_change:
        raise SystemExit('--resume: code, Blender or extension differ from an earlier session of this run ('
                         + '; '.join(changes) + '). Resuming would pool results of different code versions; use a new '
                         '--out, or pass --allow-code-change to continue anyway (recorded; the analysis flags it).')
    if changes:
        log('WARNING --allow-code-change: ' + '; '.join(changes))
    session = dict(session_id=uuid.uuid4().hex[:12], started_utc=utc_now(), ended_utc=None, status='running',
                   resumed=bool(args.resume and sessions), assets=selected,
                   allow_code_change=bool(args.allow_code_change), code_changes=changes, **current)
    sessions.append(session)

    def write_environment():
        document = dict(schema_version=SCHEMA_VERSION, harness='scripts/run_experiment.py', config=config,
                        **{key: value for key, value in session.items()}, sessions=sessions)
        env_path.write_text(json.dumps(json_safe(document), indent=2), encoding='utf-8')

    write_environment()
    conditions = build_conditions(args)
    timing_rows_expected = (args.warmup + args.reps) * len(MAIN_METHODS) * len(args.ratios)
    done, done_timing = completed_groups(raw_path, args.ratios, timing_rows_expected) if args.resume else (set(), set())
    log(f'run {session["session_id"]}: {len(selected)} assets {selected}, experiments {args.experiments}, '
        f'ratios {args.ratios}, lambda {args.strength:g} ({args.strength_source}), out {out_dir}')
    raw = RawLog(raw_path, session['session_id'])
    all_ok = True
    try:
        for asset_id in selected:
            entry = manifest['assets'][asset_id]
            all_ok &= process_asset(asset_id, order.index(asset_id), entry, conditions, args, raw, out_dir,
                                    manifest_path.parent, done, done_timing)
        session['status'] = 'completed' if all_ok else 'completed_with_errors_or_failed_checks'
    except BaseException:
        session['status'] = 'interrupted'
        raise
    finally:
        raw.close()
        session['ended_utc'] = utc_now()
        write_environment()
        log(f'run {session["session_id"]}: {session["status"]}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    arguments = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    sys.exit(main(arguments))
