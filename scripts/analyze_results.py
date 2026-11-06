"""Turn ``scripts/run_experiment.py`` output into tables, figures and a short report.

    python scripts/analyze_results.py RESULTS_DIR [RESULTS_DIR ...] [--out DIR]

Needs Python >= 3.10 and NumPy. SciPy (Wilcoxon p for tie-free samples) and
Matplotlib (figures) are optional; without them the script uses its own exact
sign-flip test and skips the figures. Every number in the output is computed
from the ``raw.jsonl`` files; ``environment.json`` is read only for machine
information. Output (default ``<first dir>/analysis``): ``report.md``,
``tables/*.csv`` + ``tables/*.md`` and ``figures/*.png``.

Statistics (per ratio and metric, paired by asset):

* difference ``delta = PERCEPTUAL - baseline`` in the metric's own units;
  "improvement" is ``delta`` for higher-is-better metrics (IoU) and ``-delta``
  for lower-is-better ones (distances, time);
* Hodges-Lehmann shift (median of the Walsh averages ``(d_i + d_j) / 2``,
  ``i <= j``) with its exact distribution-free CI, the interval of shifts the
  Wilcoxon signed-rank test does not reject (Walsh averages of order
  ``c + 1`` and ``M - c``, ``M = n(n+1)/2``, ``c`` the largest count with
  ``P(T+ <= c) <= 2.5 %``). The attained level ``1 - 2 P(T+ <= c)`` is
  reported (96.1 % at n = 9, 95.3 % at n = 7); for continuous data without
  ties the CI excludes 0 exactly when the exact two-sided p is at most
  ``1 - level``. It covers the shift of a symmetric paired distribution; it is
  not a CI of the mean. The interval uses all n differences (zero
  differences count as ties at 0) and the untied null distribution, so with
  tied or zero differences it is approximate. No CI for n < 5 (the exact 95 %
  interval needs n >= 6; n = 5 attains at most 93.75 %). Secondary columns
  ``boot_ci_*`` keep the percentile bootstrap of the mean over assets; it is
  nominal and under-covers at this n (about 0.89-0.91 at n = 9 and 0.74-0.76 at n = 3
  on normal data), so it is never the reported CI;
* two-sided Wilcoxon signed-rank p. Zero differences are dropped (Wilcoxon's
  convention). SciPy's exact test is used when SciPy is present and ``|delta|``
  has no ties; otherwise, for n <= 50, the exact sign-flip distribution of the
  tied ranks (a dynamic program over rank sums, equal to enumerating all 2^n
  sign patterns); beyond that a normal approximation. ``p_method`` names it;
* Holm-Bonferroni adjustment over the declared family: ``--holm-metrics`` x
  ``--holm-baselines`` x ``--holm-ratios`` of one table (default: every
  metric, baseline and ratio; each category is its own family). Everything
  else is exploratory. The report gives, per metric, the smallest Holm p the
  family could reach if every test hit its smallest exact p ``2/2^n``, so
  members that can never be significant are named before any data is read;
* matched-pairs rank-biserial correlation of the improvements (+1: PERCEPTUAL
  better on every asset, -1: worse on every asset);
* wins/ties/losses: a tie is ``|delta| <= --tie-iou`` for IoU metrics and
  ``|delta| <= --tie-rel * |baseline|`` for distances and time.

Consistency checks: settings (``CONFIG_KEYS``) must agree between result
directories, and the code fingerprint (sha256 of ``smart_lod/*.py`` and the
harness, Blender version and build hash, extension version) must agree between
every recorded session of every directory; otherwise the report names the
difference, because rows from different code versions would be pooled.
An error record counts as resolved only when the data it prevented exists
(its condition's rows, a complete timing set, a later LOD file or determinism
result, or a later asset record).
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import sys
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

try:  # optional
    import scipy.stats as scipy_stats
except Exception:  # pragma: no cover - depends on the environment
    scipy_stats = None

EXACT_MAX_N = 50  # exact sign-flip counts stay below 2**53 up to here
CI_LEVEL = 0.95
CI_MIN_N = 5  # no CI below this many pairs (per-category n = 3 would give a degenerate interval)
ALPHA = 0.05
METHODS = ('DECIMATE', 'QEM', 'PERCEPTUAL')
CODE_KEYS = ('source_sha256', 'blender_version', 'blender_build_hash', 'extension')  # per-session fingerprint
STYLE = {'QEM': ('#2a78d6', 'o'), 'PERCEPTUAL': ('#eb6834', 's'), 'DECIMATE': ('#1baf7a', '^'),
         'ANCHOR_V01': ('#52514e', 'D')}  # validated categorical slots 1-3 + neutral ink
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    better: str  # 'higher' or 'lower'
    tie: str  # 'absolute' (IoU) or 'relative' (to |baseline|)


METRICS = (
    Metric('chamfer_n', 'Chamfer (norm.)', 'lower', 'relative'),
    Metric('hausdorff_n', 'Hausdorff sampled (norm.)', 'lower', 'relative'),
    Metric('hausdorff_v_n', 'Hausdorff samples+vertices (norm.)', 'lower', 'relative'),
    Metric('iou_axis', 'Silhouette IoU, 6 axis views', 'higher', 'absolute'),
    Metric('iou_multiview', 'Silhouette IoU, 13 views', 'higher', 'absolute'),
    Metric('roi_chamfer_n', 'ROI Chamfer ref->LOD (norm.)', 'lower', 'relative'),
    Metric('roi_hausdorff_n', 'ROI Hausdorff ref->LOD (norm.)', 'lower', 'relative'),
    Metric('roi_chamfer_sym_n', 'ROI Chamfer symmetric (norm.)', 'lower', 'relative'),
    Metric('roi_hausdorff_sym_n', 'ROI Hausdorff symmetric (norm.)', 'lower', 'relative'),
    Metric('nonroi_chamfer_n', 'non-ROI Chamfer ref->LOD (norm.)', 'lower', 'relative'),
    Metric('time_s', 'Time per LOD (s, median of reps)', 'lower', 'relative'),
)
METRIC = {metric.key: metric for metric in METRICS}
QUALITY_METRICS = tuple(metric for metric in METRICS if metric.key != 'time_s')
QUALITY_PATHS = {
    'chamfer_n': ('metrics', 'chamfer_mean_normalized'),
    'hausdorff_n': ('metrics', 'hausdorff_sampled_normalized'),
    'hausdorff_v_n': ('metrics', 'hausdorff_with_vertices_normalized'),
    'iou_axis': ('metrics', 'silhouette_iou_mean'),
    'iou_multiview': ('metrics', 'silhouette_iou_multiview_mean'),
    'roi_chamfer_n': ('metrics', 'roi_chamfer_ref_to_candidate_normalized'),
    'roi_hausdorff_n': ('metrics', 'roi_hausdorff_ref_to_candidate_normalized'),
    'roi_chamfer_sym_n': ('metrics', 'roi_chamfer_symmetric_normalized'),
    'roi_hausdorff_sym_n': ('metrics', 'roi_hausdorff_symmetric_normalized'),
    'roi_chamfer_c2r_n': ('metrics', 'roi_chamfer_candidate_to_ref_normalized'),
    'roi_hausdorff_c2r_n': ('metrics', 'roi_hausdorff_candidate_to_ref_normalized'),
    'nonroi_chamfer_n': ('metrics', 'nonroi_chamfer_ref_to_candidate_normalized'),
    'nonroi_hausdorff_n': ('metrics', 'nonroi_hausdorff_ref_to_candidate_normalized'),
    'volume_ratio': ('volume_ratio',),
    'samples': ('metrics', 'sampling', 'samples_per_mesh'),
    'metric_seed': ('metrics', 'sampling', 'seed'),
    'resolution': ('metrics', 'sampling', 'silhouette_resolution'),
    'strength': ('params', 'strength'),
    'weighting': ('params', 'weighting'),
    'weight_set': ('params', 'weight_set'),
    'stop_reason': ('stats', 'stop_reason'),
    'locked_vertices': ('stats', 'locked_vertices'),
    'equals_qem': ('checks', 'equals_qem'),
    'locked': ('params', 'locked'),
    'evaluation_seconds': ('timings', 'evaluation_seconds'),
}
CONFIG_KEYS = ('ratios', 'seed', 'samples', 'resolution', 'strength', 'weights', 'preserve_seams', 'manifest_sha256')
TIMING_PATHS = {
    'processing_seconds': ('timings', 'processing_seconds'),
    'method_set_seconds': ('timings', 'method_set_seconds'),
    'importance_seconds': ('timings', 'importance_seconds'),
    'pass_seconds': ('timings', 'pass_seconds'),
    'input_arrays_seconds': ('timings', 'input_arrays_seconds'),  # QEM/PERCEPTUAL: source mesh -> arrays
    'constraints_seconds': ('timings', 'constraints_seconds'),  # QEM/PERCEPTUAL: seam/material locks
    'readback_set_seconds': ('timings', 'method_set_readback_seconds'),  # DECIMATE: LODs -> arrays, measurement only
}
TIMING_COMPONENTS = ('input_arrays_seconds', 'constraints_seconds', 'readback_set_seconds')  # per set, reported apart


# --- loading ---------------------------------------------------------------------------------

def dig(record, path):
    for key in path:
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return record


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def ratio_key(value):
    value = number(value)
    return None if value is None else round(value, 12)


def _base(record):
    return dict(asset_id=record.get('asset_id'), category=record.get('category'), split=record.get('split'),
                asset_index=record.get('asset_index'), experiment=record.get('experiment'),
                condition=record.get('condition'), method=record.get('method'), level=record.get('level'),
                ratio=ratio_key(record.get('ratio')), source_triangles=number(record.get('source_triangles')),
                actual_triangles=number(record.get('actual_triangles')),
                target_triangles=number(record.get('target_triangles')), session_id=record.get('session_id'))


def flatten_quality(record):
    row = _base(record)
    if not row['asset_id'] or not row['experiment'] or not row['condition'] or row['ratio'] is None:
        return None
    row['target_reached'] = record.get('target_reached')
    row['reused_from'] = record.get('reused_from')
    for key, path in QUALITY_PATHS.items():
        value = dig(record, path)
        row[key] = value if isinstance(value, (str, bool)) or value is None else number(value)
    if row['source_triangles'] and row['actual_triangles'] is not None:
        row['reduction'] = 1.0 - row['actual_triangles'] / row['source_triangles']
        row['retained'] = row['actual_triangles'] / row['source_triangles']
    else:
        row['reduction'] = row['retained'] = None
    return row


def flatten_timing(record):
    row = _base(record)
    if not row['asset_id'] or not row['condition'] or row['ratio'] is None or record.get('rep') is None:
        return None
    row.update(rep=record.get('rep'), warmup=bool(record.get('warmup')),
               matches_quality_run=record.get('matches_quality_run'))
    for key, path in TIMING_PATHS.items():
        row[key] = number(dig(record, path))
    return row


class Results:
    """Records of one or more result directories; later duplicates replace earlier ones.

    Asset and error records keep their reading order in ``_seq`` and their
    directory index in ``_dir`` (for error resolution).
    """

    def __init__(self):
        self.quality, self.timing, self.assets = {}, {}, defaultdict(list)
        self.errors, self.problems, self.environments = [], [], []
        self.configs = {}  # directory index -> config of its environment.json
        self.duplicates = self.bad_lines = self.sequence = 0
        self.mixed_code = None  # description when sessions ran different code/Blender

    def add_line(self, line, where, directory_index=0):
        line = line.strip()
        if not line:
            return
        self.sequence += 1
        try:
            record = json.loads(line)
        except ValueError:
            self.problems.append(f'{where}: malformed JSON line skipped')
            self.bad_lines += 1
            return
        if not isinstance(record, dict):
            self.problems.append(f'{where}: not a JSON object, skipped')
            self.bad_lines += 1
            return
        kind = record.get('record_type')
        if kind == 'quality':
            row = flatten_quality(record)
            if row is None:
                self.problems.append(f'{where}: quality record without asset/experiment/condition/ratio')
                self.bad_lines += 1
                return
            key = (row['asset_id'], row['experiment'], row['condition'], row['ratio'])
            self.duplicates += key in self.quality
            self.quality[key] = row
        elif kind == 'timing':
            row = flatten_timing(record)
            if row is None:
                self.problems.append(f'{where}: timing record without asset/condition/ratio/rep')
                self.bad_lines += 1
                return
            key = (row['asset_id'], row['condition'], row['ratio'], row['rep'])
            self.duplicates += key in self.timing
            self.timing[key] = row
        elif kind in ('asset', 'error'):
            record.update(_seq=self.sequence, _dir=directory_index)
            if kind == 'asset':
                self.assets[record.get('asset_id')].append(record)
            else:
                self.errors.append(record)
        else:
            self.problems.append(f'{where}: unknown record_type {kind!r}, skipped')
            self.bad_lines += 1

    def rows(self, **filters):
        return [row for row in self.quality.values() if all(row.get(k) == v for k, v in filters.items())]


def code_fingerprint(session):
    """The code/Blender identity of one session and a short id for it."""
    fingerprint = {key: session.get(key) for key in CODE_KEYS}
    return fingerprint, hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode('utf-8')).hexdigest()[:10]


def check_sessions(results):
    """Flag sessions (of any directory) that ran different code, Blender or extension versions."""
    sessions = [(env.get('_directory'), session) for env in results.environments
                for session in (env.get('sessions') or [env])]
    prints = {}
    for directory, session in sessions:
        fingerprint, code_id = code_fingerprint(session)
        prints.setdefault(code_id, (fingerprint, []))[1].append(f'{directory}/{session.get("session_id")}')
    if len(prints) < 2:
        return None
    fingerprints = [fingerprint for fingerprint, _ in prints.values()]
    differing = []
    for key in CODE_KEYS:
        values = [fingerprint.get(key) for fingerprint in fingerprints]
        if all(value == values[0] for value in values):
            continue
        if key == 'source_sha256' and all(isinstance(value, dict) for value in values):
            paths = sorted({path for value in values for path in value})
            differing += [path for path in paths if len({value.get(path) for value in values}) > 1]
        else:
            differing.append(key)
    session_code = {session.get('session_id'): code_fingerprint(session)[1] for _, session in sessions}
    counts = defaultdict(int)
    for row in results.quality.values():
        counts[session_code.get(row.get('session_id'), 'unknown')] += 1
    return (f'code/Blender/extension differ between recorded sessions ({", ".join(differing)}); code versions: '
            + '; '.join(f'`{code_id}` = {", ".join(names)}' for code_id, (_, names) in prints.items())
            + '. Quality rows per code version: ' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items()))
            + ' - rows from different code versions are pooled; rerun or analyze the versions separately')


def load_results(directories):
    results = Results()
    for index, directory in enumerate(directories):
        directory = Path(directory)
        path = directory / 'raw.jsonl'
        if not path.exists():
            results.problems.append(f'{path}: missing')
            continue
        with open(path, encoding='utf-8') as handle:
            for number_, line in enumerate(handle, 1):
                results.add_line(line, f'{path.name}:{number_} ({directory.name})', index)
        env = directory / 'environment.json'
        if env.exists():
            try:
                document = json.loads(env.read_text(encoding='utf-8'))
            except ValueError:
                results.problems.append(f'{env}: unreadable')
                continue
            if isinstance(document, dict):
                document['_directory'] = directory.name
                results.environments.append(document)
                results.configs[index] = document.get('config') or {}
    for key in CONFIG_KEYS:  # merged rows replace each other by (asset, experiment, condition, ratio)
        values = {json.dumps((env.get('config') or {}).get(key), sort_keys=True) for env in results.environments}
        if len(values) > 1:
            results.problems.append(f'setting {key!r} differs between result directories ({", ".join(sorted(values))}); '
                                    'rows with the same asset/condition/ratio replace each other - analyze separately')
    results.mixed_code = check_sessions(results)
    if results.mixed_code:
        results.problems.append(results.mixed_code)
    return results


# --- statistics ------------------------------------------------------------------------------

def average_ranks(values):
    """1-based ranks; tied values share the mean of their ranks."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return values
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts - 1) / 2.0)[inverse.reshape(-1)]


def signed_ranks(differences):
    """Nonzero differences and the average ranks of their absolute values."""
    d = np.asarray(differences, dtype=np.float64).reshape(-1)
    d = d[d != 0]
    return d, average_ranks(np.abs(d))


def wilcoxon_exact_p(differences):
    """Exact two-sided Wilcoxon signed-rank p by sign-flip over the (tied) ranks.

    Under H0 every sign pattern of the nonzero differences is equally likely;
    the distribution of W+ is counted by dynamic programming over twice the
    average ranks (integers). ``p = min(1, 2 * min(P(W+ <= w), P(W+ >= w)))``.
    Returns None when every difference is zero.
    """
    d, ranks = signed_ranks(differences)
    n = len(d)
    if n == 0:
        return None
    if n > EXACT_MAX_N:
        raise ValueError(f'exact sign-flip limited to n <= {EXACT_MAX_N}')
    doubled = np.rint(2.0 * ranks).astype(np.int64)
    total = int(doubled.sum())
    counts = np.zeros(total + 1, dtype=np.float64)
    counts[0] = 1.0
    for rank in doubled:
        shifted = np.zeros_like(counts)
        shifted[rank:] = counts[:total + 1 - rank]
        counts = counts + shifted
    probabilities = counts / counts.sum()
    observed = int(doubled[d > 0].sum())
    lower, upper = probabilities[:observed + 1].sum(), probabilities[observed:].sum()
    return float(min(1.0, 2.0 * min(lower, upper)))


def wilcoxon_normal_p(differences):
    """Two-sided normal approximation with tie correction, no continuity correction."""
    d, ranks = signed_ranks(differences)
    n = len(d)
    if n == 0:
        return None
    _, ties = np.unique(np.abs(d), return_counts=True)
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(ties ** 3 - ties)) / 48.0
    if variance <= 0:
        return 1.0
    z = (ranks[d > 0].sum() - n * (n + 1) / 4.0) / math.sqrt(variance)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def wilcoxon_test(differences, use_scipy=True):
    """Returns ``(p, method)``; see the module docstring for the choice of method."""
    d, _ = signed_ranks(differences)
    n = len(d)
    if n == 0:
        return None, 'all_differences_zero'
    ties = len(np.unique(np.abs(d))) < n
    if use_scipy and scipy_stats is not None and not ties and n <= EXACT_MAX_N:
        try:
            result = scipy_stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided', method='exact')
            return float(result.pvalue), 'scipy_exact'
        except TypeError:  # SciPy without the ``method`` keyword
            pass
    if n <= EXACT_MAX_N:
        return wilcoxon_exact_p(d), 'exact_sign_flip'
    if use_scipy and scipy_stats is not None:
        try:
            result = scipy_stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided', method='approx',
                                          correction=False)
            return float(result.pvalue), 'scipy_normal_approx'
        except TypeError:
            pass
    return wilcoxon_normal_p(d), 'normal_approx'


def rank_biserial(improvements):
    """Matched-pairs rank-biserial correlation (Kerby 2014) over nonzero improvements."""
    d, ranks = signed_ranks(improvements)
    if not len(d):
        return None
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def holm_adjust(pvalues):
    """Holm-Bonferroni adjusted p for a family; None entries are not part of it."""
    present = sorted(((p, index) for index, p in enumerate(pvalues) if p is not None), key=lambda item: item[0])
    adjusted, running, m = [None] * len(pvalues), 0.0, len(present)
    for position, (p, index) in enumerate(present):
        running = max(running, min(1.0, (m - position) * p))
        adjusted[index] = running
    return adjusted


def cluster_bootstrap_ci(values, clusters=None, n_boot=10000, seed=0, level=0.95):
    """Percentile CI of the pooled mean, resampling whole clusters with replacement.

    One value per cluster (the default) is the ordinary bootstrap of the mean.
    Returns ``(None, None)`` with fewer than two clusters.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    clusters = np.arange(len(values)) if clusters is None else np.asarray(clusters)
    if len(clusters) != len(values):
        raise ValueError('values and clusters differ in length')
    labels, inverse = np.unique(clusters, return_inverse=True)
    k = len(labels)
    if k < 2 or n_boot < 1:
        return None, None
    inverse = inverse.reshape(-1)
    sums = np.bincount(inverse, weights=values, minlength=k)
    sizes = np.bincount(inverse, minlength=k).astype(np.float64)
    draws = np.random.default_rng(seed).integers(0, k, size=(int(n_boot), k))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


@functools.lru_cache(maxsize=None)
def signed_rank_null_cdf(n):
    """``P(T+ <= k)`` for k = 0..n(n+1)/2 under H0, untied ranks 1..n (exact counts for n <= EXACT_MAX_N)."""
    total = n * (n + 1) // 2
    counts = np.zeros(total + 1, dtype=np.float64)
    counts[0] = 1.0
    for rank in range(1, n + 1):
        shifted = np.zeros_like(counts)
        shifted[rank:] = counts[:total + 1 - rank]
        counts = counts + shifted
    return tuple(np.cumsum(counts) / counts.sum())


def walsh_averages(differences):
    """Sorted ``(d_i + d_j) / 2`` for all ``i <= j`` (n(n+1)/2 values)."""
    d = np.asarray(differences, dtype=np.float64).reshape(-1)
    i, j = np.triu_indices(len(d))
    return np.sort((d[i] + d[j]) / 2.0)


def hodges_lehmann(differences):
    """Hodges-Lehmann estimate of the paired shift: the median of the Walsh averages."""
    walsh = walsh_averages(differences)
    return float(np.median(walsh)) if len(walsh) else None


def wilcoxon_shift_ci(differences, level=CI_LEVEL, min_n=CI_MIN_N):
    """Exact distribution-free CI for the paired shift, inverted from the Wilcoxon signed-rank test.

    With ``M = n(n+1)/2`` sorted Walsh averages ``A_(1..M)`` and ``c`` the
    largest count with ``P(T+ <= c) <= (1 - level) / 2``, the interval is
    ``[A_(c+1), A_(M-c)]`` and its attained confidence is ``1 - 2 P(T+ <= c)``
    (Hollander & Wolfe, Nonparametric Statistical Methods, 3.6). It is exact
    for continuous data symmetric about the shift; tied or zero differences
    make it approximate. Normal approximation of ``c`` above EXACT_MAX_N.
    Returns ``(low, high, attained_level)``, or ``(None, None, None)`` for
    n < ``min_n`` or when no interval reaches ``level`` (n <= 5 at 95 %).
    """
    d = np.asarray(differences, dtype=np.float64).reshape(-1)
    n = len(d)
    if n < max(1, min_n):
        return None, None, None
    tail_limit, total = (1.0 - level) / 2.0, n * (n + 1) // 2
    if n <= EXACT_MAX_N:
        cdf = np.asarray(signed_rank_null_cdf(n))
        feasible = np.flatnonzero(cdf <= tail_limit * (1 + 1e-12))
        if not len(feasible):
            return None, None, None
        c = int(feasible[-1])
        tail = float(cdf[c])
    else:
        normal = NormalDist(total / 2.0, math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0))
        c = int(math.floor(normal.mean + normal.stdev * NormalDist().inv_cdf(tail_limit) - 0.5))
        if c < 0:
            return None, None, None
        tail = normal.cdf(c + 0.5)
    walsh = walsh_averages(d)
    return float(walsh[c]), float(walsh[total - 1 - c]), 1.0 - 2.0 * tail


def smallest_exact_p(n):
    """Smallest two-sided exact Wilcoxon p attainable with n nonzero, untied differences."""
    return min(1.0, 2.0 / 2 ** n) if n > 0 else None


def improvement(candidate, baseline, better):
    candidate, baseline = np.asarray(candidate, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    return candidate - baseline if better == 'higher' else baseline - candidate


def tolerances(metric, baseline, tie_rel, tie_iou):
    baseline = np.abs(np.asarray(baseline, dtype=np.float64))
    return np.full_like(baseline, tie_iou) if metric.tie == 'absolute' else tie_rel * baseline


def wins_ties_losses(improvements, tolerance):
    improvements = np.asarray(improvements, dtype=np.float64)
    tolerance = np.broadcast_to(np.asarray(tolerance, dtype=np.float64), improvements.shape)
    wins, losses = int(np.sum(improvements > tolerance)), int(np.sum(improvements < -tolerance))
    return wins, len(improvements) - wins - losses, losses


def stable_seed(seed, *parts):
    return [int(seed), zlib.crc32('|'.join(map(str, parts)).encode('utf-8'))]


def paired_comparison(candidate, baseline, metric, clusters=None, n_boot=10000, seed=0, tie_rel=0.01,
                      tie_iou=5e-4, use_scipy=True):
    """Summary of paired values, one per asset (independent pairs).

    ``ci_low``/``ci_high`` are the exact Walsh-average CI of the shift around
    ``hl_diff`` (attained level ``ci_level``); ``boot_ci_*`` is the secondary,
    nominal percentile bootstrap of ``mean_diff`` (``clusters`` groups values
    for it only). Both are None for n < CI_MIN_N.
    """
    candidate, baseline = np.asarray(candidate, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    delta = candidate - baseline
    n = len(delta)
    out = dict(n=n, mean_diff=None, hl_diff=None, ci_low=None, ci_high=None, ci_level=None, boot_ci_low=None,
               boot_ci_high=None, median_diff=None, median_rel_change_pct=None, wins=0, ties=0, losses=0, p=None,
               p_method=None, rank_biserial=None)
    if n == 0:
        return out
    gains = improvement(candidate, baseline, metric.better)
    nonzero = baseline != 0
    out.update(mean_diff=float(delta.mean()), median_diff=float(np.median(delta)), hl_diff=hodges_lehmann(delta),
               median_rel_change_pct=float(np.median(100.0 * delta[nonzero] / np.abs(baseline[nonzero])))
               if nonzero.any() else None)
    out['ci_low'], out['ci_high'], out['ci_level'] = wilcoxon_shift_ci(delta)
    if n >= CI_MIN_N:
        out['boot_ci_low'], out['boot_ci_high'] = cluster_bootstrap_ci(delta, clusters, n_boot, seed)
    out['wins'], out['ties'], out['losses'] = wins_ties_losses(
        gains, tolerances(metric, baseline, tie_rel, tie_iou))
    out['p'], out['p_method'] = wilcoxon_test(delta, use_scipy)
    out['rank_biserial'] = rank_biserial(gains)
    return out


def pareto_mask(points, senses):
    """Non-dominated rows of ``points``; senses: 'min' or 'max' per column."""
    points = np.asarray(points, dtype=np.float64)
    signed = points * np.array([1.0 if sense == 'min' else -1.0 for sense in senses])
    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not np.all(np.isfinite(signed[i])):
            keep[i] = False
            continue
        others = np.delete(signed, i, axis=0)
        others = others[np.all(np.isfinite(others), axis=1)]
        dominated = np.all(others <= signed[i], axis=1) & np.any(others < signed[i], axis=1)
        keep[i] = not dominated.any()
    return keep


# --- tables ----------------------------------------------------------------------------------

def fmt(value, kind='num'):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return '—'
    if kind == 'text':
        return str(value)
    if kind == 'bool':
        return 'yes' if value else 'no'
    if kind == 'int':
        return f'{int(round(value)):,}'
    if kind == 'p':
        return f'{value:.3g}'
    if kind == 'pct':
        return f'{value:+.2f}%'
    if kind == 'signed':
        return f'{value:+.3g}'
    if kind == 'ratio':
        return f'{100 * value:g}%'
    return f'{value:.4g}'


def md_table(columns, rows):
    lines = ['| ' + ' | '.join(header for _, header, _ in columns) + ' |',
             '|' + '|'.join('---' if kind == 'text' else '---:' for _, _, kind in columns) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(fmt(row.get(key), kind).replace('|', '/') for key, _, kind in columns) + ' |')
    return '\n'.join(lines)


def write_table(directory, name, columns, rows, title):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f'{name}.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([key for key, _, _ in columns])
        for row in rows:
            writer.writerow(['' if row.get(key) is None else row.get(key) for key, _, _ in columns])
    (directory / f'{name}.md').write_text(f'# {title}\n\n{md_table(columns, rows)}\n', encoding='utf-8')


def asset_order(results):
    order = {}
    for row in list(results.quality.values()) + list(results.timing.values()):
        order.setdefault(row['asset_id'], (row.get('split') != 'tuning', row.get('asset_index') or 0, row['asset_id']))
    return sorted(order, key=order.get)


def timing_summary(results):
    """Per (asset, method, ratio) and (asset, method) medians/IQR over non-warm-up repetitions.

    The set statistics also carry the median of each per-set component in
    TIMING_COMPONENTS (``<component>_median``) when the records have it.
    """
    per_level, per_set, components, warmups = defaultdict(list), defaultdict(dict), defaultdict(dict), 0
    for row in results.timing.values():
        if row['warmup']:
            warmups += 1
            continue
        if row['processing_seconds'] is not None:
            per_level[(row['asset_id'], row['condition'], row['ratio'])].append(row['processing_seconds'])
        if row['method_set_seconds'] is not None:
            per_set[(row['asset_id'], row['condition'])][row['rep']] = row['method_set_seconds']
        for key in TIMING_COMPONENTS:  # one value per set, repeated on every level
            if row.get(key) is not None:
                components[(row['asset_id'], row['condition'], key)][row['rep']] = row[key]

    def describe(values):
        values = np.asarray(values, dtype=np.float64)
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        return dict(median=float(median), q1=float(q1), q3=float(q3), n=len(values))
    set_times = {key: describe(list(values.values())) for key, values in per_set.items()}
    for (asset, method, key), values in components.items():
        if (asset, method) in set_times:
            set_times[(asset, method)][f'{key}_median'] = float(np.median(list(values.values())))
    return {key: describe(values) for key, values in per_level.items()}, set_times, warmups


def per_asset_table(results, level_times):
    rows = []
    for asset in asset_order(results):
        for method in METHODS:
            for row in sorted(results.rows(asset_id=asset, experiment='main', condition=method),
                              key=lambda r: -r['ratio']):
                time = level_times.get((asset, method, row['ratio']), {})
                rows.append(dict(row, time_median_s=time.get('median'), time_q1_s=time.get('q1'),
                                 time_q3_s=time.get('q3'), time_reps=time.get('n')))
    return rows


PER_ASSET_COLUMNS = [
    ('asset_id', 'asset', 'text'), ('category', 'category', 'text'), ('split', 'split', 'text'),
    ('condition', 'method', 'text'), ('ratio', 'ratio', 'ratio'), ('source_triangles', 'source tris', 'int'),
    ('target_triangles', 'target', 'int'), ('actual_triangles', 'actual', 'int'),
    ('target_reached', 'reached', 'bool'), ('reduction', 'reduction', 'num'), ('chamfer_n', 'Chamfer', 'num'),
    ('hausdorff_n', 'Hausdorff', 'num'), ('hausdorff_v_n', 'Hausdorff +vertices', 'num'),
    ('iou_axis', 'IoU axis', 'num'), ('iou_multiview', 'IoU 13v', 'num'),
    ('roi_chamfer_n', 'ROI Chamfer', 'num'), ('roi_hausdorff_n', 'ROI Hausdorff', 'num'),
    ('roi_chamfer_sym_n', 'ROI Chamfer sym.', 'num'), ('roi_hausdorff_sym_n', 'ROI Hausdorff sym.', 'num'),
    ('nonroi_chamfer_n', 'non-ROI Chamfer', 'num'), ('volume_ratio', 'volume ratio', 'num'),
    ('time_median_s', 'time median s', 'num'), ('time_q1_s', 'time Q1', 'num'), ('time_q3_s', 'time Q3', 'num'),
    ('time_reps', 'reps', 'int')]

PAIRED_COLUMNS = [
    ('ratio', 'ratio', 'ratio'), ('metric', 'metric', 'text'), ('better', 'better', 'text'),
    ('baseline', 'vs', 'text'), ('n', 'n', 'int'), ('mean_diff', 'mean Δ', 'signed'),
    ('hl_diff', 'HL Δ', 'signed'), ('ci_low', 'CI low', 'signed'), ('ci_high', 'CI high', 'signed'),
    ('ci_level', 'CI level', 'num'), ('median_diff', 'median Δ', 'signed'),
    ('median_rel_change_pct', 'median rel.', 'pct'), ('wins', 'W', 'int'), ('ties', 'T', 'int'),
    ('losses', 'L', 'int'), ('p', 'p', 'p'), ('p_method', 'p method', 'text'), ('p_holm', 'p Holm', 'p'),
    ('rank_biserial', 'r_rb', 'signed'), ('unequal_triangles', 'unequal tris', 'int'), ('family', 'family', 'text'),
    ('boot_ci_low', 'boot CI low (nominal)', 'signed'), ('boot_ci_high', 'boot CI high (nominal)', 'signed')]
REPORT_PAIRED_OMIT = ('median_diff', 'median_rel_change_pct', 'p_method', 'boot_ci_low', 'boot_ci_high')


def asset_info(results):
    """asset -> (split, category) from any quality or timing row."""
    info = {}
    for row in list(results.quality.values()) + list(results.timing.values()):
        info.setdefault(row['asset_id'], (row.get('split'), row.get('category')))
    return info


def paired_values(results, split, category, baseline, ratio, metric, level_times):
    """Aligned PERCEPTUAL/baseline values of the main experiment (time: timing records), one per asset."""
    assets, candidate, reference, unequal = [], [], [], 0
    info = asset_info(results)
    for asset in asset_order(results):
        asset_split, asset_category = info[asset]
        if (split != 'all' and asset_split != split) or (category is not None and asset_category != category):
            continue
        p = results.quality.get((asset, 'main', 'PERCEPTUAL', ratio))
        b = results.quality.get((asset, 'main', baseline, ratio))
        if metric.key == 'time_s':
            pv = level_times.get((asset, 'PERCEPTUAL', ratio), {}).get('median')
            bv = level_times.get((asset, baseline, ratio), {}).get('median')
        else:
            pv, bv = (p or {}).get(metric.key), (b or {}).get(metric.key)
        if pv is None or bv is None:
            continue
        assets.append(asset)
        candidate.append(pv)
        reference.append(bv)
        if p and b and p['actual_triangles'] and b['actual_triangles']:
            unequal += abs(p['actual_triangles'] - b['actual_triangles']) > 0.02 * b['actual_triangles']
    return assets, candidate, reference, unequal


def paired_table(results, args, split, category, level_times):
    rows = []
    ratios = sorted({key[3] for key in results.quality if key[1] == 'main'}
                    | {key[2] for key in results.timing}, reverse=True)
    for ratio in ratios:
        for metric in METRICS:
            for baseline in args.baselines:
                assets, candidate, reference, unequal = paired_values(
                    results, split, category, baseline, ratio, metric, level_times)
                if not assets:
                    continue
                summary = paired_comparison(candidate, reference, metric, assets, args.bootstrap,
                                            stable_seed(args.seed, baseline, ratio, metric.key, category),
                                            args.tie_rel, args.tie_iou, not args.no_scipy)
                rows.append(dict(summary, ratio=ratio, metric=metric.key, better=metric.better, baseline=baseline,
                                 category=category or 'all', split=split, unequal_triangles=unequal,
                                 assets=' '.join(assets)))
    members = [in_holm_family(row, args) for row in rows]
    for row, member, adjusted in zip(rows, members, holm_adjust([row['p'] if member else None
                                                                 for row, member in zip(rows, members)])):
        row['p_holm'] = adjusted
        row['family'] = 'holm' if member else 'exploratory'
    return rows


def in_holm_family(row, args):
    """Declared (pre-registered) family: --holm-metrics x --holm-baselines x --holm-ratios."""
    return (row['metric'] in args.holm_metrics and row['baseline'] in args.holm_baselines
            and (args.holm_ratios is None or row['ratio'] in args.holm_ratios))


def ablation_table(results, args, split):
    rows = []
    conditions = sorted({key[2] for key in results.quality if key[1] == 'ablation'})
    ratios = sorted({key[3] for key in results.quality if key[1] == 'ablation'}, reverse=True)
    for condition in conditions:
        for ratio in ratios:
            for metric in QUALITY_METRICS:
                full, variant = [], []
                for asset in asset_order(results):
                    a = results.quality.get((asset, 'ablation', condition, ratio))
                    f = results.quality.get((asset, 'main', 'PERCEPTUAL', ratio))
                    if a is None or f is None or (split != 'all' and a['split'] != split):
                        continue
                    if a.get(metric.key) is None or f.get(metric.key) is None:
                        continue
                    variant.append(a[metric.key])
                    full.append(f[metric.key])
                if not full:
                    continue
                delta = np.asarray(variant) - np.asarray(full)
                gains = improvement(variant, full, metric.better)  # > 0: the variant is better than full
                better, ties, worse = wins_ties_losses(gains, tolerances(metric, full, args.tie_rel, args.tie_iou))
                rows.append(dict(split=split, condition=condition, ratio=ratio, metric=metric.key,
                                 better=metric.better, n=len(delta), mean_delta=float(delta.mean()),
                                 median_delta=float(np.median(delta)), variant_better=better, ties=ties,
                                 variant_worse=worse))
    return rows


ABLATION_COLUMNS = [
    ('condition', 'condition', 'text'), ('ratio', 'ratio', 'ratio'), ('metric', 'metric', 'text'),
    ('better', 'better', 'text'), ('n', 'n', 'int'), ('mean_delta', 'mean Δ vs full', 'signed'),
    ('median_delta', 'median Δ', 'signed'), ('variant_better', 'variant better', 'int'), ('ties', 'ties', 'int'),
    ('variant_worse', 'variant worse', 'int')]


def relative_change(value, reference, metric):
    if value is None or reference is None:
        return None
    if metric.better == 'higher':  # IoU: absolute difference
        return value - reference
    return 100.0 * (value - reference) / reference if reference else None


SWEEP_METRICS = ('chamfer_n', 'hausdorff_n', 'iou_axis', 'iou_multiview', 'roi_chamfer_n', 'roi_chamfer_sym_n')
PARETO_SPLITS = ('tuning',)  # lambda may be chosen only here; other splits get no Pareto flags


def sweep_table(results, split):
    """Mean over assets of each lambda (and reference methods) relative to QEM, per ratio.

    Pareto flags are computed only for PARETO_SPLITS (the tuning split); on
    held-out splits the sweep is exploratory and flagging a "best" lambda
    there would invite post-hoc re-tuning, so the flags stay None.
    """
    sweep = {key[2]: results.quality[key].get('strength') for key in results.quality if key[1] == 'sweep'}
    if not sweep:
        return []
    entries = [('sweep', name, sweep[name]) for name in sorted(sweep, key=lambda n: sweep[n] or 0)]
    entries += [('main', 'PERCEPTUAL', None), ('main', 'DECIMATE', None), ('ablation', 'ANCHOR_V01', None)]
    rows = []
    for ratio in sorted({key[3] for key in results.quality if key[1] == 'sweep'}, reverse=True):
        assets = [a for a in asset_order(results)
                  if all((a, 'sweep', name, ratio) in results.quality for name in sweep)
                  and (split == 'all' or results.quality[(a, 'sweep', entries[0][1], ratio)]['split'] == split)]
        reference = {}
        for asset in assets:
            ref = results.quality.get((asset, 'main', 'QEM', ratio))
            if ref is None:  # lambda = 0 equals QEM (checked by the harness)
                zero = [name for name, value in sweep.items() if value == 0]
                ref = results.quality.get((asset, 'sweep', zero[0], ratio)) if zero else None
            reference[asset] = ref
        assets = [asset for asset in assets if reference[asset] is not None]
        if not assets:
            continue
        block = []
        for experiment, name, strength in entries:
            members = [results.quality.get((asset, experiment, name, ratio)) for asset in assets]
            if any(member is None for member in members):
                continue
            row = dict(split=split, ratio=ratio, entry=f'{experiment}/{name}', strength=strength
                       if strength is not None else members[0].get('strength'), n=len(assets),
                       is_sweep=experiment == 'sweep')
            for key in SWEEP_METRICS:
                # ROI metrics exist only for assets with an ROI group: average over those
                # assets instead of letting one missing value void the whole mean.
                pairs = [(member.get(key), relative_change(member.get(key), reference[asset].get(key), METRIC[key]))
                         for member, asset in zip(members, assets)]
                pairs = [(value, change) for value, change in pairs if value is not None and change is not None]
                row[f'{key}_mean'] = float(np.mean([value for value, _ in pairs])) if pairs else None
                row[f'{key}_change'] = float(np.mean([change for _, change in pairs])) if pairs else None
                row[f'{key}_n'] = len(pairs)
            block.append(row)
        sweep_rows = [row for row in block if row['is_sweep']]
        for name, columns, senses in (('pareto_iou_hausdorff', ('hausdorff_n_change', 'iou_multiview_change'),
                                       ('min', 'max')),
                                      ('pareto_roi_global_chamfer', ('chamfer_n_change', 'roi_chamfer_n_change'),
                                       ('min', 'min'))):
            points = [[np.nan if row[c] is None else row[c] for c in columns] for row in sweep_rows]
            mask = pareto_mask(points, senses) if points and split in PARETO_SPLITS else [None] * len(sweep_rows)
            for row, flag in zip(sweep_rows, mask):
                row[name] = bool(flag) if flag is not None and all(row[c] is not None for c in columns) else None
        rows += block
    return rows


SWEEP_COLUMNS = [
    ('ratio', 'ratio', 'ratio'), ('entry', 'condition', 'text'), ('strength', 'λ', 'num'), ('n', 'n', 'int'),
    ('chamfer_n_change', 'Chamfer vs QEM', 'pct'), ('hausdorff_n_change', 'Hausdorff vs QEM', 'pct'),
    ('iou_axis_change', 'Δ IoU axis', 'signed'), ('iou_multiview_change', 'Δ IoU 13v', 'signed'),
    ('roi_chamfer_n_change', 'ROI Chamfer vs QEM', 'pct'),
    ('roi_chamfer_sym_n_change', 'ROI Chamfer sym. vs QEM', 'pct'),
    ('pareto_iou_hausdorff', 'Pareto IoU/H', 'bool'), ('pareto_roi_global_chamfer', 'Pareto ROI/global', 'bool'),
    ('chamfer_n_mean', 'mean Chamfer', 'num'), ('hausdorff_n_mean', 'mean Hausdorff', 'num'),
    ('iou_multiview_mean', 'mean IoU 13v', 'num')]
SWEEP_REPORT_COLUMNS = 12  # report.md shows up to the Pareto flags


def timing_table(results, set_times, level_times):
    rows = []
    for asset in asset_order(results):
        for method in METHODS:
            stats = set_times.get((asset, method))
            if not stats:
                continue
            sample = next(row for row in results.timing.values()
                          if row['asset_id'] == asset and row['condition'] == method)
            levels = {ratio: level_times[(asset, method, ratio)]['median']
                      for (a, m, ratio) in level_times if a == asset and m == method}
            rows.append(dict(asset_id=asset, split=sample['split'], category=sample['category'], method=method,
                             source_triangles=sample['source_triangles'], set_median_s=stats['median'],
                             set_q1_s=stats['q1'], set_q3_s=stats['q3'], reps=stats['n'],
                             **{f'{key}_median': stats.get(f'{key}_median') for key in TIMING_COMPONENTS},
                             levels=' '.join(f'{100 * r:g}%:{v:.3g}s' for r, v in sorted(levels.items(),
                                                                                         reverse=True))))
    return rows


TIMING_COLUMNS = [
    ('asset_id', 'asset', 'text'), ('split', 'split', 'text'), ('method', 'method', 'text'),
    ('source_triangles', 'source tris', 'int'), ('set_median_s', 'all LODs median s', 'num'),
    ('set_q1_s', 'Q1', 'num'), ('set_q3_s', 'Q3', 'num'), ('reps', 'reps', 'int'),
    ('input_arrays_seconds_median', 'incl. input arrays s', 'num'),
    ('constraints_seconds_median', 'incl. seam/material locks s', 'num'),
    ('readback_set_seconds_median', 'Decimate readback s (excluded)', 'num'),
    ('levels', 'per-LOD median', 'text')]


# --- figures ---------------------------------------------------------------------------------

def make_figures(results, sweep_rows, set_times, out_dir, splits):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import ticker
    except Exception as exc:
        return [], f'Matplotlib unavailable ({exc}); figures skipped.'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for split in splits:
        main = [row for row in results.quality.values()
                if row['experiment'] == 'main' and (split == 'all' or row['split'] == split)]
        categories = sorted({row['category'] for row in main})
        if main:
            keys = ('chamfer_n', 'hausdorff_n', 'iou_axis', 'iou_multiview')
            fig, axes = plt.subplots(len(keys), len(categories), figsize=(3.6 * len(categories), 2.6 * len(keys)),
                                     squeeze=False, constrained_layout=True)
            for column, category in enumerate(categories):
                for line, key in enumerate(keys):
                    axis = axes[line][column]
                    for method in METHODS:
                        color, marker = STYLE[method]
                        rows = [r for r in main if r['category'] == category and r['condition'] == method
                                and r.get(key) is not None and r.get('retained')]
                        by_asset, by_ratio = defaultdict(list), defaultdict(list)
                        for r in rows:
                            by_asset[r['asset_id']].append((r['retained'], r[key]))
                            by_ratio[r['ratio']].append((r['retained'], r[key]))
                        for points in by_asset.values():
                            points.sort()
                            axis.plot(*zip(*points), color=color, alpha=0.25, linewidth=1)
                        if by_ratio:
                            medians = sorted((float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts])))
                                             for pts in by_ratio.values())
                            axis.plot(*zip(*medians), color=color, marker=marker, markersize=6, linewidth=2,
                                      label=method)
                    axis.set_xscale('log')
                    axis.xaxis.set_major_locator(ticker.FixedLocator([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]))
                    axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{100 * v:g}%'))
                    axis.xaxis.set_minor_formatter(ticker.NullFormatter())
                    if key in ('chamfer_n', 'hausdorff_n'):
                        axis.set_yscale('log')
                    axis.set_xlabel('actual / source triangles')
                    axis.set_ylabel(METRIC[key].label, fontsize=8)
                    if line == 0:
                        axis.set_title(category, color=INK)
            axes[0][0].legend(loc='best', fontsize=8)
            fig.suptitle(f'Quality vs actual triangles (split: {split}; thick = median of assets, thin = assets)',
                         color=INK)
            path = out_dir / f'quality_vs_triangles_{split}.png'
            fig.savefig(path, dpi=130)
            plt.close(fig)
            figures.append((path, f'Kualitas terhadap jumlah segitiga aktual, split {split}'))
        block = [row for row in sweep_rows if row['split'] == split]
        ratios = sorted({row['ratio'] for row in block}, reverse=True)
        if ratios:
            panels = (('hausdorff_n_change', 'iou_multiview_change', 'Hausdorff vs QEM (%)', 'Δ IoU 13 views vs QEM'),
                      ('chamfer_n_change', 'roi_chamfer_n_change', 'Chamfer vs QEM (%)', 'ROI Chamfer vs QEM (%)'))
            fig, axes = plt.subplots(2, len(ratios), figsize=(4.2 * len(ratios), 7.0), squeeze=False,
                                     constrained_layout=True)
            for column, ratio in enumerate(ratios):
                rows = [row for row in block if row['ratio'] == ratio]
                for line, (x_key, y_key, x_label, y_label) in enumerate(panels):
                    axis = axes[line][column]
                    axis.axhline(0, color=MUTED, linewidth=0.8)
                    axis.axvline(0, color=MUTED, linewidth=0.8)
                    sweep = [r for r in rows if r['is_sweep'] and r[x_key] is not None and r[y_key] is not None]
                    if sweep:
                        axis.plot([r[x_key] for r in sweep], [r[y_key] for r in sweep], color=STYLE['PERCEPTUAL'][0],
                                  linewidth=1.5, zorder=1, label='PERCEPTUAL λ sweep')
                        flag = 'pareto_iou_hausdorff' if line == 0 else 'pareto_roi_global_chamfer'
                        for r in sweep:
                            filled = bool(r.get(flag))
                            axis.scatter(r[x_key], r[y_key], s=42, marker='s', zorder=2, linewidths=1.5,
                                         edgecolors=STYLE['PERCEPTUAL'][0],
                                         facecolors=STYLE['PERCEPTUAL'][0] if filled else 'white')
                            axis.annotate(f'λ={r["strength"]:g}', (r[x_key], r[y_key]), fontsize=7, color=INK,
                                          textcoords='offset points', xytext=(4, 4))
                    axis.scatter([0], [0], s=42, marker=STYLE['QEM'][1], color=STYLE['QEM'][0], label='QEM', zorder=3)
                    for r in rows:
                        name = r['entry'].split('/')[1]
                        # ANCHOR_V01 is off-scale (hundreds of %) and would hide the sweep; its
                        # values are in the ablation and sweep tables.
                        if name == 'DECIMATE' and r[x_key] is not None and r[y_key] is not None:
                            axis.scatter(r[x_key], r[y_key], s=42, marker=STYLE[name][1], color=STYLE[name][0],
                                         label=name, zorder=3)
                    axis.set_xlabel(x_label)
                    axis.set_ylabel(y_label)
                    if line == 0:
                        axis.set_title(f'ratio {100 * ratio:g}% (n={rows[0]["n"]})', color=INK)
                    if line == 0 and column == 0:
                        axis.legend(loc='best', fontsize=8)
            fig.suptitle(f'λ sweep, mean over assets (split: {split}); filled = Pareto-optimal λ'
                         if split in PARETO_SPLITS else f'λ sweep, mean over assets (split: {split}); exploratory, '
                         'no Pareto selection (λ is frozen from the tuning split)', color=INK)
            fig.text(0.5, 0.005, 'ANCHOR_V01 omitted (off-scale); see the ablation and sweep tables. '
                     'ROI panels average only assets with an ROI.', ha='center', fontsize=8, color=MUTED)
            path = out_dir / f'lambda_pareto_{split}.png'
            fig.savefig(path, dpi=130)
            plt.close(fig)
            figures.append((path, f'Sapuan λ dan front Pareto, split {split}' if split in PARETO_SPLITS
                            else f'Sapuan λ, split {split} (eksploratori, tanpa penanda Pareto)'))
    if set_times:
        fig, axis = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
        for method in METHODS:
            color, marker = STYLE[method]
            points = []
            for (asset, name), stats in set_times.items():
                if name != method:
                    continue
                source = next(r['source_triangles'] for r in results.timing.values() if r['asset_id'] == asset)
                points.append((source, stats['median'], stats['q1'], stats['q3']))
            if points:
                points.sort()
                x, y, q1, q3 = map(np.array, zip(*points))
                axis.errorbar(x, y, yerr=[y - q1, q3 - y], color=color, marker=marker, markersize=7, linestyle='none',
                              elinewidth=1.5, capsize=2, label=method)  # assets are separate points, not a series
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        axis.xaxis.set_minor_formatter(ticker.NullFormatter())
        axis.set_xlabel('source triangles (log)')
        axis.set_ylabel('all LODs, median seconds (IQR bars, log)')
        axis.set_title('Processing time vs source size', color=INK)
        axis.legend(loc='best', fontsize=8)
        path = out_dir / 'timing_vs_source_triangles.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        figures.append((path, 'Waktu pemrosesan terhadap jumlah segitiga sumber'))
    return figures, None


# --- report ----------------------------------------------------------------------------------

REPORT_METRICS = ('chamfer_n', 'hausdorff_n', 'iou_multiview', 'roi_chamfer_n', 'roi_chamfer_sym_n')


def pivot(rows, keys, cell):
    """Markdown grid: one line per ``keys`` combination, one column per report metric."""
    grid, order = {}, []
    for row in rows:
        if row['metric'] not in REPORT_METRICS:
            continue
        key = tuple(row[k] for k in keys)
        if key not in grid:
            grid[key] = {}
            order.append(key)
        grid[key][row['metric']] = cell(row)
        grid[key]['n'] = max(grid[key].get('n', 0), row['n'])
    columns = [(k, k, 'ratio' if k == 'ratio' else 'text') for k in keys] + [('n', 'n (max)', 'int')] + [
        (m, f'{METRIC[m].label} ({METRIC[m].better} better)', 'text') for m in REPORT_METRICS]
    return md_table(columns, [dict(zip(keys, key), **grid[key]) for key in order])


def integrity(results):
    lines = []
    records = {asset: items[-1] for asset, items in results.assets.items()}
    determinism = [r.get('determinism', {}) or {} for r in records.values()]
    checked = [d for d in determinism if 'identical' in d]
    lines.append(f'- Repeated QEM pass identical: {sum(bool(d["identical"]) for d in checked)}/{len(checked)} assets.')
    equals = [row for row in results.quality.values() if row.get('equals_qem') is not None]
    bad = sorted({(r['asset_id'], r['condition']) for r in equals if r['equals_qem'] is False})
    lines.append(f'- λ = 0 control equals QEM: {len(equals) - sum(r["equals_qem"] is False for r in equals)}/'
                 f'{len(equals)} rows' + (f'; MISMATCH {bad}' if bad else '') + '.')
    for key, label in (('geometry_sha256_matches_manifest', 'Snapshot hash equals manifest geometry_sha256'),
                       ('roi_vertices_match_manifest', 'ROI vertex count equals manifest')):
        values = [r.get(key) for r in records.values() if r.get(key) is not None]
        lines.append(f'- {label}: {sum(bool(v) for v in values)}/{len(values)} assets.')
    matches = [r['matches_quality_run'] for r in results.timing.values() if r['matches_quality_run'] is not None]
    lines.append(f'- Timing runs reproduce the quality-run arrays: {sum(bool(v) for v in matches)}/{len(matches)} rows.'
                 if matches else '- Timing runs reproduce the quality-run arrays: not checked (no timing records '
                 'with a matching quality run).')
    return lines


def budget_warnings(results, split):
    lines = []
    for asset in asset_order(results):
        for ratio in sorted({key[3] for key in results.quality if key[0] == asset and key[1] == 'main'}, reverse=True):
            rows = {m: results.quality.get((asset, 'main', m, ratio)) for m in METHODS}
            rows = {m: r for m, r in rows.items() if r is not None and (split == 'all' or r['split'] == split)}
            counts = {m: r['actual_triangles'] for m, r in rows.items() if r['actual_triangles']}
            missed = [m for m, r in rows.items() if r.get('target_reached') is False]
            if (len(counts) > 1 and max(counts.values()) > 1.02 * min(counts.values())) or missed:
                target = next(iter(rows.values()))['target_triangles']
                lines.append(f'- {asset} @ {100 * ratio:g}% (target {fmt(target, "int")}): '
                             + ', '.join(f'{m} {fmt(c, "int")}' for m, c in counts.items())
                             + (f'; target not reached by {", ".join(missed)}' if missed else ''))
    return lines


def environment_lines(results):
    lines = []
    for env in results.environments:
        ext = env.get('extension') or {}
        sessions = env.get('sessions') or []
        lines.append(f'- Run `{env.get("session_id")}` ({env.get("status")}): Blender {env.get("blender_version")}, '
                     f'NumPy {env.get("numpy_version")}, {env.get("cpu_name")}, {env.get("logical_cores")} logical '
                     f'cores, {env.get("platform")}, extension {ext.get("manifest_version")}; '
                     f'{env.get("started_utc")} → {env.get("ended_utc")}; sessions: {len(sessions)}.')
        for session in sessions if len(sessions) > 1 else []:
            lines.append(f'  - session `{session.get("session_id")}` ({session.get("status")}, '
                         f'{session.get("started_utc")}): code `{code_fingerprint(session)[1]}`, Blender '
                         f'{session.get("blender_version")} {session.get("blender_build_hash")}'
                         + (f'; resumed with --allow-code-change over: {"; ".join(session["code_changes"])}'
                            if session.get('code_changes') else '') + '.')
    return lines


def configuration_lines(results):
    """Settings as recorded in the raw rows themselves."""
    rows = list(results.quality.values())
    def values(key, **filters):
        found = {row.get(key) for row in rows if all(row.get(k) == v for k, v in filters.items())
                 and row.get(key) is not None}
        return sorted((int(v) if isinstance(v, float) and v.is_integer() and key != 'ratio' else v for v in found),
                      key=str)
    return [f'- Surface samples per mesh: {values("samples")}; metric seed: {values("metric_seed")}; '
            f'silhouette resolution: {values("resolution")}.',
            f'- λ of main PERCEPTUAL: {values("strength", condition="PERCEPTUAL", experiment="main")}; '
            f'weighting: {values("weighting", condition="PERCEPTUAL", experiment="main")}; '
            f'ratios: {[f"{100 * r:g}%" for r in sorted(values("ratio"), reverse=True)]}.']


def holm_reachability(paired_rows):
    """Per metric of the Holm family: (n values, smallest exact p, best attainable Holm p).

    The best case sets every family member to its smallest exact p ``2/2^n``;
    Holm adjusted p is nondecreasing in every raw p, so no data can do better.
    """
    family = [row for row in paired_rows if row.get('p_holm') is not None]
    best = holm_adjust([smallest_exact_p(row['n']) for row in family])
    summary = {}
    for row, bound in zip(family, best):
        entry = summary.setdefault(row['metric'], dict(n=set(), p_min=1.0, best=1.0))
        entry['n'].add(row['n'])
        entry['p_min'] = min(entry['p_min'], smallest_exact_p(row['n']) or 1.0)
        entry['best'] = min(entry['best'], bound if bound is not None else 1.0)
    return summary


def guardrails(results, paired_rows, args, splits_present, sweep_rows=(), ablation_rows=()):
    lines = []
    if results.mixed_code:
        lines.append('- PERHATIAN: sesi run yang digabung memakai kode/Blender/extension berbeda; baris dari versi '
                     f'kode berbeda tercampur dalam tabel. {results.mixed_code}.')
    family = [row for row in paired_rows if row.get('p_holm') is not None]
    n_eval = max((row['n'] for row in paired_rows), default=0)
    if n_eval:
        declared = (f'metrik {", ".join(args.holm_metrics)}; baseline {", ".join(args.holm_baselines)}; rasio '
                    + (', '.join(f'{100 * r:g}%' for r in args.holm_ratios) if args.holm_ratios else 'semua'))
        lines.append(f'- Uji berpasangan memakai paling banyak n = {n_eval} aset (split `{args.paired_split}`; '
                     f'metrik ROI hanya aset ber-ROI). p eksak terkecil yang mungkin (tanpa ties) = 2/2^n. Famili '
                     f'Holm ({declared}) berisi {len(family)} uji, ambang terkecil {ALPHA:g}/{len(family)} = '
                     f'{ALPHA / max(len(family), 1):.3g}.')
        reach = holm_reachability(paired_rows)
        if reach:
            blocked = [metric for metric, entry in reach.items() if entry['best'] > ALPHA]
            lines.append('- p Holm terbaik yang mungkin per metrik famili (bila setiap uji famili mencapai p eksak '
                         'terkecilnya): ' + '; '.join(
                             f'{metric} (n = {"/".join(map(str, sorted(entry["n"])))}, p min {entry["p_min"]:.3g}) '
                             f'≥ {entry["best"]:.3g}' for metric, entry in reach.items()) + '. '
                         + (f'{", ".join(blocked)} tidak dapat signifikan setelah Holm apa pun datanya; hasil nol di '
                            'sana adalah konsekuensi desain, bukan bukti tidak ada efek. '
                            + ('Perkecil famili (mis. `--holm-baselines QEM --holm-ratios 0.05`) sebelum melihat data, '
                               'atau laporkan sebagai eksploratori.' if len(args.holm_baselines) > 1
                               or len(args.holm_ratios or (0, 0)) > 1 else 'Laporkan sebagai eksploratori.')
                            if blocked else 'Signifikansi setelah Holm secara prinsip mungkin untuk setiap '
                            'metrik famili, tetapi n tetap kecil.')
                         + ' Laporkan ukuran efek, CI dan W/T/L, bukan hanya p.')
        if args.holm_ratios:
            present = {(row['metric'], row['baseline'], row['ratio']) for row in family}
            missing = [f'{m}/{b}/{100 * r:g}%' for m in args.holm_metrics for b in args.holm_baselines
                       for r in args.holm_ratios if (m, b, r) not in present]
            if missing:
                lines.append('- PERHATIAN: anggota famili Holm yang dideklarasikan tanpa data/p: '
                             f'{", ".join(missing)}; famili yang dikoreksi lebih kecil dari yang didaftarkan.')
        lines.append(f'- CI = interval eksak Walsh/Hodges–Lehmann untuk pergeseran berpasangan (inversi uji Wilcoxon; '
                     f'tingkat tercapai di kolom `CI level`, mis. 96,1 % pada n = 9 dan 95,3 % pada n = 7). Tidak '
                     f'dilaporkan untuk n < {CI_MIN_N}; pada n = 5 interval 95 % tidak mungkin. Kolom `boot_ci_*` di '
                     'tabel CSV adalah bootstrap persentil rata-rata yang nominal dan terlalu sempit pada n sekecil '
                     'ini (cakupan sekitar 0,89–0,91 pada n = 9 untuk data normal); jangan dikutip sebagai CI 95 %.')
    if 'tuning' in splits_present:
        lines.append('- Aset `tuning` dipakai untuk memilih λ dan bobot. Hasilnya tidak boleh dipakai sebagai bukti '
                     'generalisasi; klaim hanya dari split `evaluation` dengan parameter yang dibekukan sebelumnya.')
    held_out = sorted({row['split'] for row in list(sweep_rows) + list(ablation_rows)} - set(PARETO_SPLITS))
    if held_out:
        lines.append(f'- Sapuan λ dan ablation pada split {", ".join(f"`{s}`" for s in held_out)} bersifat '
                     'eksploratori (analisis sekunder). Jangan memilih atau mengubah λ/bobot dari hasil ini; penanda '
                     'Pareto sengaja tidak dihitung di luar split `tuning`. Penalaan ulang setelah melihat data ini '
                     'harus dilaporkan sebagai analisis eksploratori baru.')
    if args.paired_split == 'tuning':
        lines.append('- PERHATIAN: tabel berpasangan di laporan ini dihitung pada split `tuning` (mode uji/smoke). '
                     'Angka p dan CI di sini tidak boleh dikutip sebagai hasil penelitian.')
    lines += [
        '- Metode berbeda dapat berhenti pada jumlah segitiga aktual yang berbeda (kunci seam/UV, batas topologi). '
        'Bandingkan kurva kualitas–segitiga dan daftar anggaran di bawah; hasil yang lebih padat bukan bukti metode '
        'lebih baik.',
        '- Ada trade-off: λ yang menaikkan IoU silhouette dapat menaikkan Hausdorff/Chamfer global. Laporkan kedua '
        'arah; front Pareto hanya deskriptif.',
        '- Chamfer/Hausdorff adalah estimasi berbasis sampel acak (seed tetap); `hausdorff_n` hanya dari sampel dan '
        'dapat meremehkan artefak lokal, `hausdorff_v_n` menambahkan setiap vertex kedua mesh. IoU memakai 6 atau 13 '
        'view ortografik pada raster tetap. ROI `LOD_EvalROI` dianotasi secara geometris oleh satu orang dan hanya '
        'dipakai untuk evaluasi; `roi_*_n` hanya referensi → LOD (tidak melihat geometri berlebih di ROI), '
        '`roi_*_sym_n` dua arah.',
        '- Waktu adalah wall-clock pada satu mesin; implementasi Python (QEM/PERCEPTUAL) vs C (Decimate) mencerminkan '
        'implementasi, bukan kompleksitas algoritmik. Pengulangan waktu bukan sampel independen antar-aset. Waktu '
        'QEM/PERCEPTUAL mencakup konversi mesh sumber ke larik dan kunci seam/material (kolom terpisah); waktu '
        'DECIMATE tidak mencakup pembacaan LOD ke larik (hanya untuk pengukuran; kolom terpisah).',
        '- Asosiasi antar metrik tidak menunjukkan sebab-akibat; ablation hanya menunjukkan perubahan pada data ini.']
    return lines


def error_resolved(results, error):
    """True only when the data the error prevented exists (see the module docstring)."""
    asset, scope = error.get('asset_id'), error.get('scope')
    config = results.configs.get(error.get('_dir'), {})
    ratios = {ratio_key(r) for r in config.get('ratios') or []} - {None}
    later = [r for r in results.assets.get(asset, []) if r.get('_seq', 0) > error.get('_seq', 0)]
    if scope == 'condition':
        group = (asset, error.get('experiment'), error.get('condition'))
        found = {key[3] for key in results.quality if key[:3] == group}
        return bool(found) and (ratios or {key[3] for key in results.quality if key[0] == asset}) <= found
    if scope == 'timing':
        reps = defaultdict(set)
        for (a, method, ratio, rep), row in results.timing.items():
            if a == asset:
                reps[(method, ratio)].add(rep)
        wanted_ratios = ratios or {ratio for _, ratio in reps}
        count = number(config.get('reps')), number(config.get('warmup'))
        wanted = (set(range(int(count[0] + count[1]))) if None not in count
                  else set(range(max((max(v) for v in reps.values()), default=-1) + 1)))
        timed = {rep for (a, _, _, rep), row in results.timing.items() if a == asset and not row['warmup']}
        return bool(timed) and all(wanted <= reps.get((m, r), set()) for m in METHODS for r in wanted_ratios)
    if scope == 'save_lods':
        return any(r.get('lods_file') for r in later)
    if scope == 'determinism':
        return any('identical' in (r.get('determinism') or {}) for r in later)
    if scope == 'asset':
        return bool(later)
    return False


def write_report(path, results, args, tables, paired_rows, category_rows, ablation_rows, sweep_rows, timing_rows,
                 figures, figure_note, splits_present):
    kinds = defaultdict(int)
    for row in results.timing.values():
        kinds['warm-up' if row['warmup'] else 'timed'] += 1
    resolved = [error_resolved(results, e) for e in results.errors]
    unresolved = [e for e, ok in zip(results.errors, resolved) if not ok]
    rel = lambda p: Path(p).relative_to(path.parent).as_posix()  # noqa: E731
    out = [
        '# Analisis hasil Smart LOD', '',
        'Dibuat oleh `scripts/analyze_results.py` dari: ' + ', '.join(f'`{d}`' for d in args.dirs) + '. '
        'Semua angka dihitung dari `raw.jsonl`; `environment.json` hanya untuk informasi mesin.', '',
        '## 1. Data', '',
        f'- Record: {len(results.quality)} quality, {kinds["timed"]} timing (+{kinds["warm-up"]} warm-up, tidak '
        f'dianalisis), {sum(len(v) for v in results.assets.values())} asset, {len(results.errors)} error '
        f'({len(unresolved)} tanpa hasil pengganti). Baris rusak/tidak dikenal: {results.bad_lines}; masalah lain '
        f'(berkas hilang, pengaturan/kode berbeda): {len(results.problems) - results.bad_lines}; '
        f'duplikat yang diganti record terakhir: {results.duplicates}.']
    out += [f'- {problem}' for problem in results.problems[:10]]
    out += [f'- {problem}' for problem in results.problems[10:] if 'differ' in problem]  # never cut off
    listed = unresolved[:10] + [e for e, ok in zip(results.errors, resolved) if ok][:max(0, 10 - len(unresolved))]
    out += [f'- ERROR {e.get("asset_id")} {e.get("scope")} {e.get("experiment")}/{e.get("condition")}: '
            f'{e.get("error")}' + (' (ada hasil pengganti)' if e not in unresolved else ' (TANPA hasil pengganti)')
            for e in listed]
    out += configuration_lines(results) + environment_lines(results)
    asset_rows = []
    for asset in asset_order(results):
        records = results.assets.get(asset, [])
        any_row = next((r for r in list(results.quality.values()) + list(results.timing.values())
                        if r['asset_id'] == asset), {})
        asset_rows.append(dict(asset_id=asset, category=any_row.get('category'), split=any_row.get('split'),
                               source_triangles=any_row.get('source_triangles'),
                               roi=records[-1].get('roi_vertices') if records else None,
                               locked=records[-1].get('locked_vertices') if records else None,
                               wall=sum(number(r.get('asset_wall_seconds')) or 0 for r in records) or None))
    out += ['', md_table([('asset_id', 'asset', 'text'), ('category', 'category', 'text'), ('split', 'split', 'text'),
                          ('source_triangles', 'triangles', 'int'), ('roi', 'ROI vertices', 'int'),
                          ('locked', 'locked vertices', 'int'), ('wall', 'harness wall s', 'num')], asset_rows),
            '', '### Pemeriksaan integritas', ''] + integrity(results)
    budgets = budget_warnings(results, 'all')
    out += ['', '### Anggaran segitiga tidak sama (>2 %) atau target tidak tercapai (main)', '']
    out += budgets or ['- Tidak ada.']
    out += ['', '## 2. Batas interpretasi', ''] + guardrails(results, paired_rows, args, splits_present, sweep_rows,
                                                              ablation_rows)
    out += ['', f'## 3. PERCEPTUAL vs baseline, berpasangan per aset (split `{args.paired_split}`)', '']
    if paired_rows:
        out += ['Δ = PERCEPTUAL − baseline dalam satuan metrik; `better` memberi arah yang baik (`lower`: Δ < 0 '
                'menguntungkan PERCEPTUAL, `higher`: Δ > 0). W/T/L dihitung dari '
                f'sisi PERCEPTUAL dengan toleransi seri {args.tie_rel:g}×|baseline| (jarak, waktu) atau '
                f'{args.tie_iou:g} (IoU). r_rb > 0 berarti PERCEPTUAL lebih baik. HL Δ = estimasi Hodges–Lehmann; '
                f'CI = interval eksak Walsh untuk pergeseran itu (tingkat tercapai di `CI level`; kosong bila n < '
                f'{CI_MIN_N} atau bila 95 % tidak tercapai). Holm atas '
                f'{sum(r.get("p_holm") is not None for r in paired_rows)} uji (metrik {", ".join(args.holm_metrics)}; '
                f'baseline {", ".join(args.holm_baselines)}; rasio '
                + (', '.join(f'{100 * r:g}%' for r in args.holm_ratios) if args.holm_ratios else 'semua')
                + '); baris lain eksploratori (p Holm kosong). Kolom lengkap (median, perubahan relatif, metode p, '
                'bootstrap nominal): `tables/paired_main.md`.', '',
                md_table([c for c in PAIRED_COLUMNS if c[0] not in REPORT_PAIRED_OMIT], paired_rows), '']
    else:
        out += [f'Tidak ada pasangan main PERCEPTUAL/baseline pada split `{args.paired_split}`; uji dilewati.', '']
    out += ['## 4. Per kategori (deskriptif; n per kategori kecil)', '']
    if category_rows:
        out += ['Sel = W/T/L PERCEPTUAL terhadap baseline (menang/seri/kalah, toleransi seperti di atas). Tabel lengkap '
                'dengan Δ, CI, p dan Holm per kategori: `tables/paired_by_category.md`.', '',
                pivot(category_rows, ('category', 'ratio', 'baseline'),
                      lambda r: f'{r["wins"]}/{r["ties"]}/{r["losses"]}'), '']
    else:
        out += ['Tidak ada data.', '']
    out += ['## 5. Ablation (Δ = varian − PERCEPTUAL penuh)', '']
    if ablation_rows:
        out += ['Menonaktifkan satu kanal menormalkan ulang bobot kanal aktif (rata-rata terbobot). ANCHOR_V01 memakai '
                'aturan anchor v0.1 (λ = 8) dengan importance v0.2, bukan metode v0.1 utuh. LAMBDA0 adalah kontrol '
                '(harus identik dengan QEM). Sel = rata-rata Δ antar-aset (jumlah aset varian lebih baik : lebih '
                'buruk). Tabel lengkap: `tables/ablation.md`.', '',
                pivot(ablation_rows, ('split', 'condition', 'ratio'),
                      lambda r: f'{r["mean_delta"]:+.3g} ({r["variant_better"]}:{r["variant_worse"]})'), '']
    else:
        out += ['Tidak ada data ablation.', '']
    out += ['## 6. Sapuan λ', '']
    if sweep_rows:
        out += ['Nilai relatif = rata-rata antar-aset dari perubahan per aset terhadap QEM (λ = 0): persen untuk jarak, '
                'selisih absolut untuk IoU. Pareto dihitung hanya di antara nilai λ sapuan dan hanya pada split '
                '`tuning`, satu-satunya split tempat λ boleh dipilih.', '']
        for split in sorted({r['split'] for r in sweep_rows}):
            out += [f'Split `{split}`' + (':' if split in PARETO_SPLITS else ' (eksploratori; bukan dasar memilih λ, '
                                          'tanpa penanda Pareto):'), '',
                    md_table(SWEEP_COLUMNS[:SWEEP_REPORT_COLUMNS], [r for r in sweep_rows if r['split'] == split]), '']
    else:
        out += ['Tidak ada data sapuan.', '']
    out += ['## 7. Waktu (median antar-pengulangan, tanpa warm-up)', '']
    if timing_rows:
        out += ['Per LOD = biaya mandiri membuat LOD itu: DECIMATE satu run modifier Triangulate + Decimate dan '
                '`new_from_object` (pembacaan LOD ke larik NumPy hanya untuk pengukuran, tidak termasuk; median set '
                'di kolom "Decimate readback"); QEM/PERCEPTUAL konversi mesh sumber ke larik + kunci seam/material '
                '(`feature_constraints`) + waktu pass kumulatif sampai snapshot + rekonstruksi mesh (+ peta '
                'kepentingan untuk PERCEPTUAL). Kolom "incl." adalah median komponen yang sudah termasuk. '
                '"all LODs" = seluruh set LOD satu metode. Metrik tidak termasuk. Record timing lama (tanpa kolom '
                'komponen) memakai definisi sebelumnya: DECIMATE termasuk pembacaan, QEM/PERCEPTUAL tanpa kunci.', '',
                md_table(TIMING_COLUMNS, timing_rows), '']
    else:
        out += ['Tidak ada record timing.', '']
    out += ['## 8. Gambar', '']
    out += [f'- [{caption}]({rel(p)})' for p, caption in figures] or [f'- {figure_note or "Tidak ada gambar."}']
    out += ['', '## 9. Tabel', ''] + [f'- `{rel(t)}` (+ `.csv`)' for t in tables]
    path.write_text('\n'.join(out) + '\n', encoding='utf-8')


# --- main --------------------------------------------------------------------------------------

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Analyze run_experiment.py results into tables, figures and report.md')
    parser.add_argument('dirs', nargs='+', help='result directories containing raw.jsonl')
    parser.add_argument('--out', default=None, help='output directory (default: <first dir>/analysis)')
    parser.add_argument('--baselines', default='QEM,DECIMATE')
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--paired-split', default='evaluation', choices=('evaluation', 'tuning', 'all'),
                        help='split for the paired tests (tuning/all only for checks, never for claims)')
    parser.add_argument('--tie-rel', type=float, default=0.01, help='relative tie tolerance (distances, time)')
    parser.add_argument('--tie-iou', type=float, default=5e-4, help='absolute tie tolerance for IoU')
    parser.add_argument('--holm-metrics', default=','.join(metric.key for metric in METRICS),
                        help='metrics forming the Holm family (pre-register a primary set); others are exploratory')
    parser.add_argument('--holm-baselines', default=None,
                        help='baselines in the Holm family, e.g. QEM (default: every --baselines entry)')
    parser.add_argument('--holm-ratios', default=None,
                        help='ratios in the Holm family, e.g. 0.05 (default: every ratio in the data)')
    parser.add_argument('--no-scipy', action='store_true', help='use only the built-in Wilcoxon')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args(argv)
    args.baselines = [b.strip() for b in args.baselines.split(',') if b.strip()]
    args.holm_metrics = [m.strip() for m in args.holm_metrics.split(',') if m.strip()]
    if not args.holm_metrics or any(m not in METRIC for m in args.holm_metrics):
        parser.error(f'--holm-metrics must be taken from {",".join(METRIC)}')
    unknown = [b for b in args.baselines if b not in METHODS or b == 'PERCEPTUAL']
    if unknown or not args.baselines:
        parser.error(f'--baselines must be taken from DECIMATE,QEM (got {unknown or "nothing"})')
    args.holm_baselines = args.baselines if args.holm_baselines is None else [
        b.strip() for b in args.holm_baselines.split(',') if b.strip()]
    if not args.holm_baselines or any(b not in args.baselines for b in args.holm_baselines):
        parser.error(f'--holm-baselines must be taken from --baselines ({",".join(args.baselines)})')
    if args.holm_ratios is not None:
        try:
            args.holm_ratios = [ratio_key(float(r)) for r in args.holm_ratios.split(',') if r.strip()]
        except ValueError:
            parser.error('--holm-ratios must be comma-separated numbers')
        if not args.holm_ratios or any(r is None or not 0 < r < 1 for r in args.holm_ratios):
            parser.error('--holm-ratios must be retained fractions within (0, 1), e.g. 0.05')
    if args.bootstrap < 0 or args.tie_rel < 0 or args.tie_iou < 0:
        parser.error('--bootstrap, --tie-rel and --tie-iou must be nonnegative')
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    results = load_results(args.dirs)
    if not results.quality and not results.timing:
        print('No quality or timing records found:', *results.problems, sep='\n  ', file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else Path(args.dirs[0]) / 'analysis'
    tables_dir, figures_dir = out / 'tables', out / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    splits_present = sorted({row['split'] for row in results.quality.values() if row['split']})
    level_times, set_times, _ = timing_summary(results)
    tables = []

    def table(name, columns, rows, title):
        write_table(tables_dir, name, columns, rows, title)
        tables.append(tables_dir / f'{name}.md')

    text_keys = ('asset_id', 'category', 'split', 'experiment', 'condition', 'method', 'weighting', 'weight_set',
                 'stop_reason', 'reused_from', 'locked', 'session_id')
    all_columns = [(key, key, 'text' if key in text_keys else 'bool' if key in ('target_reached', 'equals_qem')
                    else 'num') for key in flatten_quality({'asset_id': 'x', 'experiment': 'x', 'condition': 'x',
                                                           'ratio': 1})]
    order = {asset: index for index, asset in enumerate(asset_order(results))}
    all_rows = sorted(results.quality.values(), key=lambda r: (order[r['asset_id']], r['experiment'], r['condition'],
                                                               -r['ratio']))
    table('all_quality_rows', all_columns, all_rows, 'All quality rows (flattened)')
    table('per_asset', PER_ASSET_COLUMNS, per_asset_table(results, level_times), 'Per asset × method × ratio (main)')
    paired_rows = paired_table(results, args, args.paired_split, None, level_times)
    table('paired_main', PAIRED_COLUMNS, paired_rows, f'PERCEPTUAL vs baselines, split {args.paired_split}')
    category_rows = []
    categories = sorted({row['category'] for row in results.quality.values()
                         if row['category'] and (args.paired_split == 'all' or row['split'] == args.paired_split)})
    for category in categories:
        category_rows += paired_table(results, args, args.paired_split, category, level_times)
    table('paired_by_category', [('category', 'category', 'text')] + PAIRED_COLUMNS, category_rows,
          f'PERCEPTUAL vs baselines by category (Holm within category), split {args.paired_split}')
    ablation_rows = [row for split in splits_present for row in ablation_table(results, args, split)]
    table('ablation', [('split', 'split', 'text')] + ABLATION_COLUMNS, ablation_rows, 'Ablation vs full PERCEPTUAL')
    sweep_rows = [row for split in splits_present for row in sweep_table(results, split)]
    table('lambda_sweep', [('split', 'split', 'text')] + SWEEP_COLUMNS, sweep_rows, 'Lambda sweep (mean over assets)')
    timing_rows = timing_table(results, set_times, level_times)
    table('timing', TIMING_COLUMNS, timing_rows, 'Timing, median over non-warm-up repetitions')
    figures, note = ([], 'Gambar dimatikan (--no-figures).') if args.no_figures else make_figures(
        results, sweep_rows, set_times, figures_dir, splits_present)
    write_report(out / 'report.md', results, args, tables, paired_rows, category_rows, ablation_rows, sweep_rows,
                 timing_rows, figures, note, splits_present)
    print(f'Wrote {out / "report.md"}, {len(tables)} tables, {len(figures)} figures'
          + (f' ({note})' if note else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
