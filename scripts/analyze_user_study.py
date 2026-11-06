"""Validate and analyse user-study ratings (system CPython; NumPy required, SciPy optional).

    python scripts/analyze_user_study.py --study study/<name>
    python scripts/analyze_user_study.py --study study/<name> --ratings a.csv b.csv --out somewhere

Defaults under ``--study``: ratings from ``ratings/*.csv``, key from
``private/method_key.csv``, stimuli from ``stimuli.csv``, output to ``analysis/``.
Rating files must carry exactly the header of docs/USER_STUDY.csv.  A rating
row names only its stimulus_id; asset, method code, LOD, target ratio,
triangle count and view set are joined from stimuli.csv (the form never sees
them), and the method from the key.  Key, stimulus list and every row must
carry the same build_id (scripts/make_user_study.py): rows from the form of a
superseded build are rejected, never unblinded with the new key.

Validation: header, study id, build id, participant id format, unknown
stimuli, ratings outside 1..5 (set missing), repeated copies of one stimulus
and incomplete participants (excluded from every table by default;
``--incomplete include`` keeps them).  Repeated copies: the form stamps every
download with export_sequence and exported_at.  A copy whose stamp is not older
than any other copy in both fields supersedes them (a mid-study safety download
followed by the final one); otherwise identical ratings are kept once and
differing ratings are all dropped as a conflict.

Statistics per method x LOD x criterion: n, mean, SD, median and a 95 %
percentile bootstrap CI that resamples participants (clusters) with replacement;
assets are not resampled, so asset-to-asset variation is not in the interval.
Tests use per-participant means over the assets and views of one LOD and
criterion: Friedman across methods (SciPy when installed, else the documented
fallback below, which gives the same statistic and p-value), Kendall's W =
chi2_F / (n (k - 1)) as its effect size, and pairwise Wilcoxon signed-rank
tests with Holm correction, both within each LOD x criterion family (p_holm)
and across all pairwise tests (p_holm_all).  The Wilcoxon p-value is always
computed here: exact over the 2^n sign patterns given the observed average
ranks when n_nonzero <= 50 (zero differences dropped, Wilcoxon's method), else
the tie-corrected normal approximation without continuity correction; this is
what SciPy 1.17 returns for method='exact' without ties, for n <= 13 with ties,
and for method='asymptotic'.  Inter-rater agreement is Kendall's W (judges =
participants) over the non-control stimuli pooled, within each LOD, and within
each asset x LOD.  The pooled value is dominated by the LOD manipulation (and
asset differences) that every participant sees; agreement about the methods is
read from the within-LOD and within asset x LOD values.  Ratings are ordinal;
means are reported alongside medians for convenience, not as interval-scale
claims.
Outputs reveal method names: keep them private until data collection ends.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import re
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:  # optional; every statistic has a NumPy fallback
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - depends on the environment
    _scipy_stats = None

CSV_HEADER = (  # docs/USER_STUDY.csv
    'study_id', 'build_id', 'anonymous_participant_id', 'stimulus_id', 'presentation_order',
    'shape_fidelity_1_5', 'silhouette_fidelity_1_5', 'important_detail_fidelity_1_5', 'overall_quality_1_5',
    'notes', 'export_sequence', 'exported_at')
CRITERIA = ('shape_fidelity_1_5', 'silhouette_fidelity_1_5', 'important_detail_fidelity_1_5', 'overall_quality_1_5')
CRITERION_NAMES = {'shape_fidelity_1_5': 'bentuk', 'silhouette_fidelity_1_5': 'siluet',
                   'important_detail_fidelity_1_5': 'detail penting', 'overall_quality_1_5': 'keseluruhan'}
STIMULUS_COLUMNS = ('stimulus_id', 'asset_id', 'lod_level', 'target_ratio', 'actual_triangles',
                    'view_set_id', 'method_code', 'image', 'build_id')
KEY_COLUMNS = ('study_id', 'build_id', 'method_code', 'method')
METHOD_ORDER = ('PERCEPTUAL', 'QEM', 'DECIMATE')
CONTROL = 'REFERENCE'
PARTICIPANT_PATTERN = re.compile(r'^P[0-9]{2,3}$')
EXACT_LIMIT = 50


class StudyInputError(ValueError):
    """Key or stimulus list unusable: nothing can be analysed."""


class NoRatingsError(ValueError):
    """No rating rows to analyse; nothing is computed or invented."""


# ----------------------------------------------------------------- reading

def decode(data, name):
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise StudyInputError(f'{name}: not UTF-8 text ({exc})') from exc


def _table(text, name, columns):
    lines = [line for line in text.splitlines(keepends=True) if not line.lstrip().startswith('#')]
    reader = csv.DictReader(io.StringIO(''.join(lines)))
    missing = [c for c in columns if c not in (reader.fieldnames or ())]
    if missing:
        raise StudyInputError(f'{name}: missing column(s) {", ".join(missing)}')
    return list(reader)


def load_key(text, name='method_key.csv'):
    """-> (study_id, build_id, {code: method}).  Lines starting with '#' are comments."""
    rows = _table(text, name, KEY_COLUMNS)
    studies, builds = {r['study_id'] for r in rows}, {r['build_id'] for r in rows}
    if len(studies) != 1:
        raise StudyInputError(f'{name}: expected one study_id, found {sorted(studies)}')
    if len(builds) != 1 or not next(iter(builds)):
        raise StudyInputError(f'{name}: expected one non-empty build_id, found {sorted(builds)}')
    key = {}
    for r in rows:
        if r['method_code'] in key or not r['method_code'] or not r['method']:
            raise StudyInputError(f'{name}: empty or repeated method code {r["method_code"]!r}')
        key[r['method_code']] = r['method']
    if len(set(key.values())) != len(key):
        raise StudyInputError(f'{name}: two codes map to the same method')
    return studies.pop(), builds.pop(), key


def load_stimuli(text, key, build_id, name='stimuli.csv'):
    """-> {stimulus_id: stimulus dict}; (asset_id, method_code, lod_level, view_set_id) must be unique
    and every row must belong to build ``build_id``."""
    stimuli, seen = {}, set()
    for r in _table(text, name, STIMULUS_COLUMNS):
        if r['build_id'] != build_id:
            raise StudyInputError(f'{name}: build_id {r["build_id"]!r} of stimulus {r["stimulus_id"]} is not the '
                                  f'key\'s {build_id!r}; key and stimulus list come from different builds')
        try:
            r['lod_level'], r['actual_triangles'] = int(r['lod_level']), int(r['actual_triangles'])
            r['target_ratio'] = float(r['target_ratio'])
        except ValueError as exc:
            raise StudyInputError(f'{name}: bad number in stimulus {r.get("stimulus_id")}') from exc
        if r['method_code'] not in key:
            raise StudyInputError(f'{name}: method code {r["method_code"]} is not in the key')
        k = (r['asset_id'], r['method_code'], r['lod_level'], r['view_set_id'])
        if k in seen or not r['stimulus_id'] or r['stimulus_id'] in stimuli:
            raise StudyInputError(f'{name}: stimulus {r["stimulus_id"]!r} / key {k} is empty or not unique')
        seen.add(k)
        r['method'] = key[r['method_code']]
        stimuli[r['stimulus_id']] = r
    if not stimuli:
        raise StudyInputError(f'{name}: no stimuli')
    return stimuli


def load_ratings(sources):
    """``sources``: iterable of (name, text).  -> (rows, issues); files with a wrong header are skipped."""
    rows, issues = [], []
    for name, text in sources:
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        if header is None:
            issues.append(issue('error', 'empty_file', name, 1, '', 'file is empty; skipped'))
            continue
        if tuple(header) != CSV_HEADER:
            extra = sorted(set(header) - set(CSV_HEADER))
            lost = [c for c in CSV_HEADER if c not in header]
            issues.append(issue('error', 'header_mismatch', name, 1, '',
                                'header differs from docs/USER_STUDY.csv; file skipped'
                                + (f'; missing {lost}' if lost else '') + (f'; unexpected {extra}' if extra else '')
                                + ('' if lost or extra else '; column order differs')))
            continue
        count = 0
        for values in reader:
            line = reader.line_num
            if not any(v.strip() for v in values):
                continue
            if len(values) != len(CSV_HEADER):
                issues.append(issue('error', 'malformed_row', name, line, '',
                                    f'{len(values)} fields instead of {len(CSV_HEADER)}; row skipped'))
                continue
            row = dict(zip(CSV_HEADER, values))
            row['_file'], row['_line'] = name, line
            rows.append(row)
            count += 1
        if not count:
            issues.append(issue('warning', 'no_rows', name, 1, '', 'header only, no ratings'))
    return rows, issues


def issue(severity, code, file, line, participant, message):
    return dict(severity=severity, code=code, file=file, line=line, participant=participant, message=message)


# ----------------------------------------------------------------- validation

def form_order(study_id, participant, stimulus_ids):
    """Presentation order of study/rating_form_template.html: Fisher-Yates over the sorted
    stimulus ids driven by mulberry32 seeded with FNV-1a('<study_id>|<participant>').
    Mirrors the JavaScript bit for bit (ids are ASCII); used only as an integrity check."""
    seed = 0x811C9DC5
    for character in f'{study_id}|{participant}':
        seed = ((seed ^ ord(character)) * 0x01000193) & 0xFFFFFFFF
    state = seed

    def random():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((state ^ (state >> 15)) * (state | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    order = sorted(stimulus_ids)
    for i in range(len(order) - 1, 0, -1):
        j = int(random() * (i + 1))
        order[i], order[j] = order[j], order[i]
    return order


def export_stamp(row):
    """(export_sequence, exported_at) of the download a row came from; None when unreadable."""
    sequence, text = row['export_sequence'].strip(), row['exported_at'].strip()
    try:
        moment = datetime.fromisoformat(text[:-1] + '+00:00' if text.endswith('Z') else text)
    except ValueError:
        return None
    if not sequence.isdigit() or int(sequence) < 1 or moment.tzinfo is None:
        return None
    return int(sequence), moment


def newest_copy(group):
    """The copy of one stimulus whose export is not older than any other copy's in both
    export_sequence and exported_at, where copies with the same stamp agree; else None."""
    if any(r['stamp'] is None for r in group):
        return None
    for r in group:
        (sequence, moment), rating = r['stamp'], tuple(r[c] for c in CRITERIA)
        if all(sequence >= o['stamp'][0] and moment >= o['stamp'][1]
               and (o['stamp'] != r['stamp'] or tuple(o[c] for c in CRITERIA) == rating) for o in group):
            return r
    return None


def validate(rows, study_id, build_id, stimuli, *, incomplete='exclude', control_min=None):
    """-> (records, issues, participants).  ``stimuli`` {stimulus_id: row}; records are unblinded."""
    if incomplete not in ('exclude', 'include'):
        raise ValueError("incomplete must be 'exclude' or 'include'")
    issues, candidates = [], []
    for row in rows:
        where = (row['_file'], row['_line'])
        pid = row['anonymous_participant_id'].strip()
        if row['study_id'].strip() != study_id:
            issues.append(issue('error', 'wrong_study', *where, pid,
                                f'study_id {row["study_id"]!r} is not {study_id!r}; row skipped'))
            continue
        if row['build_id'].strip() != build_id:
            issues.append(issue('error', 'wrong_build', *where, pid,
                                f'build_id {row["build_id"]!r} is not {build_id!r}: rated with the form of another '
                                'build (e.g. before a --force rebuild; its key is in superseded/); row skipped'))
            continue
        if not PARTICIPANT_PATTERN.match(pid):
            issues.append(issue('error', 'invalid_participant', *where, pid,
                                'participant id must be P + 2-3 digits; row skipped'))
            continue
        stimulus = stimuli.get(row['stimulus_id'].strip())
        if stimulus is None:
            issues.append(issue('error', 'unknown_stimulus', *where, pid,
                                f'stimulus_id {row["stimulus_id"]!r} is not in stimuli.csv; row skipped'))
            continue
        stamp = export_stamp(row)
        if stamp is None:
            issues.append(issue('warning', 'invalid_export_stamp', *where, pid,
                                'export_sequence/exported_at unreadable; this copy cannot supersede another'))
        ratings, missing = {}, []  # missing ratings are reported only for the copy that is kept
        for criterion in CRITERIA:
            text = row[criterion].strip()
            if not text:
                ratings[criterion] = None
                missing.append(issue('warning', 'missing_rating', *where, pid, f'{criterion} is empty'))
            elif re.fullmatch(r'[1-5]', text):
                ratings[criterion] = int(text)
            else:
                ratings[criterion] = None
                issues.append(issue('error', 'rating_out_of_range', *where, pid,
                                    f'{criterion}={text!r} is not an integer 1..5; treated as missing'))
        order = row['presentation_order'].strip()
        if not order.isdigit() or int(order) < 1:
            issues.append(issue('warning', 'invalid_order', *where, pid, f'presentation_order {order!r}'))
            order = None
        candidates.append(dict({k: stimulus[k] for k in ('stimulus_id', 'asset_id', 'method_code', 'method',
                                                          'lod_level', 'view_set_id', 'target_ratio',
                                                          'actual_triangles')},
                               participant=pid, presentation_order=int(order) if order else None,
                               notes=row['notes'], export_sequence=stamp[0] if stamp else None,
                               exported_at=stamp[1].isoformat() if stamp else None, stamp=stamp,
                               file=row['_file'], line=row['_line'], missing=missing, **ratings))
    # Repeated copies of one stimulus: the newest export supersedes older ones; copies that cannot be
    # ordered are kept once when their ratings agree and are all dropped when they differ.
    groups = {}
    for record in candidates:
        groups.setdefault((record['participant'], record['stimulus_id']), []).append(record)
    records, superseded = [], {}
    for (pid, sid), group in groups.items():
        chosen = newest_copy(group) if len(group) > 1 else group[0]
        if chosen is None:
            if len({tuple(r[c] for c in CRITERIA) for r in group}) > 1:
                issues.append(issue('error', 'duplicate_conflict', group[0]['file'], group[0]['line'], pid,
                                    f'{len(group)} different ratings for stimulus {sid} and no copy comes from a '
                                    'later export (export_sequence and exported_at); all copies dropped'))
                continue
            chosen, same = group[0], group[1:]
        else:
            same = [r for r in group if r is not chosen and r['stamp'] == chosen['stamp']]
        if same:
            issues.append(issue('warning', 'duplicate_identical', same[0]['file'], same[0]['line'], pid,
                                f'stimulus {sid} rated {len(same) + 1} times identically; kept once'))
        superseded[pid] = superseded.get(pid, 0) + len(group) - 1 - len(same)
        issues.extend(chosen['missing'])
        records.append(chosen)
    for record in candidates:
        del record['stamp'], record['missing']
    participants = {}
    total = len(stimuli)
    for record in candidates:  # every accepted row, before de-duplication
        participants.setdefault(record['participant'], dict(
            participant=record['participant'], files=set(), rows=0, complete_stimuli=0, total_stimuli=total,
            control_overall=[], status='included', reason=''))['files'].add(record['file'])
    for record in records:
        p = participants[record['participant']]
        p['rows'] += 1
        p['complete_stimuli'] += all(record[c] is not None for c in CRITERIA)
        if record['method'] == CONTROL and record['overall_quality_1_5'] is not None:
            p['control_overall'].append(record['overall_quality_1_5'])
    all_ids = list(stimuli)
    for pid, p in sorted(participants.items()):
        if superseded.get(pid):
            issues.append(issue('warning', 'superseded_rows', '', '', pid,
                                f'{superseded[pid]} row(s) replaced by the same stimulus in a later export '
                                '(export_sequence, exported_at); the newest copy is kept'))
        own = [r for r in records if r['participant'] == pid and r['presentation_order']]
        orders = [r['presentation_order'] for r in own]
        if len(orders) != len(set(orders)):
            issues.append(issue('warning', 'duplicate_order', '', '', pid, 'presentation_order repeats'))
        expected = form_order(study_id, pid, all_ids)
        p['order_matches_form'] = bool(own) and all(
            r['presentation_order'] <= len(expected) and expected[r['presentation_order'] - 1] == r['stimulus_id']
            for r in own)
        if own and not p['order_matches_form']:
            issues.append(issue('warning', 'order_mismatch', '', '', pid,
                                'presentation_order differs from the rating form order for this id '
                                '(edited file or another collection method?)'))
        if len(p['files']) > 1:
            issues.append(issue('warning', 'multiple_files', '', '', pid,
                                f'ratings merged from {len(p["files"])} files'))
        reasons = []
        if p['complete_stimuli'] < total:
            reasons.append(f'incomplete ({p["complete_stimuli"]}/{total} stimuli fully rated)')
            issues.append(issue('warning', 'incomplete_participant', '', '', pid, reasons[-1]
                                + ('; excluded' if incomplete == 'exclude' else '; kept (--incomplete include)')))
        control = p['control_overall']
        p['control_mean'] = float(np.mean(control)) if control else None
        if control_min is not None and control and p['control_mean'] < control_min:
            reasons.append(f'control mean {p["control_mean"]:.2f} < {control_min}')
            issues.append(issue('warning', 'control_failed', '', '', pid, reasons[-1] + '; excluded'))
        excluded = (incomplete == 'exclude' and p['complete_stimuli'] < total) or \
            (control_min is not None and control and p['control_mean'] < control_min)
        p['status'] = 'excluded' if excluded else 'included'
        p['reason'] = '; '.join(reasons)
        p['files'] = ';'.join(sorted(p['files']))
        del p['control_overall']
    kept = {pid for pid, p in participants.items() if p['status'] == 'included'}
    records = sorted((r for r in records if r['participant'] in kept),
                     key=lambda r: (r['participant'], r['presentation_order'] or 0, r['stimulus_id']))
    return records, issues, [participants[pid] for pid in sorted(participants)]


# ----------------------------------------------------------------- statistics

def rank_with_ties(values):
    """Average ranks (1-based) and the sizes of tied groups."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values))
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered) != 0) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end + 1) * 0.5
    return ranks, ends - starts


def gamma_q(a, x):
    """Regularized upper incomplete gamma Q(a, x) (series / Lentz continued fraction)."""
    if x <= 0.0:
        return 1.0
    log_front = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        denominator = a
        for _ in range(10000):
            denominator += 1.0
            term *= x / denominator
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - total * math.exp(log_front))
    tiny = 1e-300
    b = x + 1.0 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(log_front) * h


def chi2_sf(x, df):
    return gamma_q(0.5 * df, 0.5 * x) if x > 0 else 1.0


def friedman_fallback(matrix):
    """Friedman chi-square with tie correction (as SciPy): rows are blocks, columns treatments."""
    data = np.asarray(matrix, dtype=float)
    n, k = data.shape
    ranks, ties = np.empty_like(data), 0.0
    for i, row in enumerate(data):
        ranks[i], sizes = rank_with_ties(row)
        ties += float(np.sum(sizes ** 3 - sizes))
    correction = 1.0 - ties / (n * (k ** 3 - k))
    if correction <= 0.0:
        return math.nan, math.nan, ranks.mean(axis=0)
    statistic = (12.0 / (n * k * (k + 1)) * float(np.sum(ranks.sum(axis=0) ** 2)) - 3.0 * n * (k + 1)) / correction
    return statistic, chi2_sf(statistic, k - 1), ranks.mean(axis=0)


def friedman(matrix, use_scipy=True):
    """dict(n, k, statistic, df, p, kendalls_w, mean_ranks, engine, note)."""
    data = np.asarray(matrix, dtype=float)
    n, k = data.shape if data.ndim == 2 else (0, 0)
    result = dict(n=n, k=k, statistic=math.nan, df=k - 1, p=math.nan, kendalls_w=math.nan,
                  mean_ranks=[math.nan] * k, engine='', note='')
    if n < 2 or k < 2:
        result['note'] = 'needs >= 2 participants and >= 2 methods'
        return result
    statistic, p, mean_ranks = friedman_fallback(data)
    result['mean_ranks'] = [float(v) for v in mean_ranks]
    if not math.isfinite(statistic):
        result['note'] = 'every participant gave all methods the same mean; test undefined'
        return result
    engine = 'numpy fallback'
    if use_scipy and _scipy_stats is not None and k >= 3:
        scipy_result = _scipy_stats.friedmanchisquare(*data.T)
        statistic, p, engine = float(scipy_result.statistic), float(scipy_result.pvalue), 'scipy'
    result.update(statistic=statistic, p=p, kendalls_w=statistic / (n * (k - 1)), engine=engine)
    return result


def wilcoxon_signed_rank(x, y, exact_limit=EXACT_LIMIT):
    """Two-sided paired test of x - y; see the module docstring for the method."""
    d = np.round(np.asarray(x, dtype=float) - np.asarray(y, dtype=float), 10)
    result = dict(n_pairs=len(d), n_nonzero=0, statistic=math.nan, r_plus=math.nan, r_minus=math.nan,
                  p=math.nan, rank_biserial=math.nan, method='')
    d = d[d != 0.0]
    n = len(d)
    result['n_nonzero'] = n
    if n == 0:
        result['method'] = 'all differences are zero; not testable'
        return result
    ranks, sizes = rank_with_ties(np.abs(d))
    r_plus, r_minus = float(ranks[d > 0].sum()), float(ranks[d < 0].sum())
    tied = bool(np.any(sizes > 1))
    if n <= exact_limit:
        weights = np.rint(2 * ranks).astype(np.int64)  # average ranks are multiples of 1/2
        counts = np.zeros(int(weights.sum()) + 1, dtype=np.int64)
        counts[0] = 1
        for weight in weights:
            shifted = np.zeros_like(counts)
            shifted[weight:] = counts[:-weight]
            counts += shifted
        observed = int(round(2 * r_plus))
        lower, upper = int(counts[:observed + 1].sum()), int(counts[observed:].sum())
        p = min(1.0, 2.0 * min(lower, upper) / float(2 ** n))
        method = 'exact' + (' (conditional on tied ranks)' if tied else '')
    else:
        mean = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(sizes ** 3 - sizes)) / 48.0
        z = (r_plus - mean) / math.sqrt(variance)
        p = math.erfc(abs(z) / math.sqrt(2.0))
        method = 'normal approximation (tie-corrected, no continuity correction)'
    result.update(statistic=min(r_plus, r_minus), r_plus=r_plus, r_minus=r_minus, p=p, method=method,
                  rank_biserial=(r_plus - r_minus) / (r_plus + r_minus))
    return result


def holm(pvalues):
    """Holm step-down adjusted p-values; NaN entries stay NaN and do not count."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(p), math.nan)
    valid = np.flatnonzero(np.isfinite(p))
    running = 0.0
    for position, index in enumerate(valid[np.argsort(p[valid], kind='mergesort')]):
        running = max(running, min(1.0, (len(valid) - position) * p[index]))
        adjusted[index] = running
    return adjusted


def kendalls_w(matrix):
    """Tie-corrected Kendall's W; rows are judges, columns the ranked objects."""
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        return math.nan
    m, n = data.shape
    ranks, ties = np.empty_like(data), 0.0
    for i, row in enumerate(data):
        ranks[i], sizes = rank_with_ties(row)
        ties += float(np.sum(sizes ** 3 - sizes))
    totals = ranks.sum(axis=0)
    denominator = m * m * (n ** 3 - n) - m * ties
    return 12.0 * float(np.sum((totals - totals.mean()) ** 2)) / denominator if denominator > 0 else math.nan


def cluster_seed(seed, label):
    return np.random.SeedSequence([int(seed) & 0xFFFFFFFF, zlib.crc32(label.encode('utf-8'))])


def clustered_bootstrap_ci(values, clusters, n_boot=10000, seed=0, level=0.95):
    """Percentile CI of the mean, resampling whole clusters with replacement.

    ``seed`` is an int or a numpy SeedSequence.  Returns (nan, nan) with fewer
    than two clusters."""
    values = np.asarray(values, dtype=float)
    labels, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    if len(labels) < 2 or n_boot < 1:
        return math.nan, math.nan
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 2000):
        picks = rng.integers(0, len(labels), size=(min(2000, n_boot - start), len(labels)))
        means[start:start + len(picks)] = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


# ----------------------------------------------------------------- analysis

def method_order(methods):
    methods = set(methods) - {CONTROL}
    return [m for m in METHOD_ORDER if m in methods] + sorted(methods - set(METHOD_ORDER))


def participant_means(records, methods, level, criterion):
    """(participants, n x k matrix) of per-participant means; only participants with every method."""
    sums = {}
    for r in records:
        if r['lod_level'] == level and r['method'] in methods and r[criterion] is not None:
            cell = sums.setdefault(r['participant'], {}).setdefault(r['method'], [0.0, 0])
            cell[0] += r[criterion]
            cell[1] += 1
    participants = sorted(p for p, cells in sums.items() if all(m in cells for m in methods))
    matrix = np.array([[sums[p][m][0] / sums[p][m][1] for m in methods] for p in participants], dtype=float)
    return participants, matrix.reshape(len(participants), len(methods))


def describe(records, method, level, criterion, n_boot, seed):
    chosen = [r for r in records if r['method'] == method and (level == 'all' or r['lod_level'] == level)
              and r[criterion] is not None]
    values = np.array([r[criterion] for r in chosen], dtype=float)
    row = dict(method=method, lod_level=level, criterion=criterion, n_ratings=len(values),
               n_participants=len({r['participant'] for r in chosen}), n_assets=len({r['asset_id'] for r in chosen}),
               mean=math.nan, sd=math.nan, median=math.nan, ci_low=math.nan, ci_high=math.nan)
    if len(values):
        row.update(mean=float(values.mean()), median=float(np.median(values)),
                   sd=float(values.std(ddof=1)) if len(values) > 1 else math.nan)
        low, high = clustered_bootstrap_ci(values, [r['participant'] for r in chosen], n_boot,
                                           cluster_seed(seed, f'{method}|{level}|{criterion}'))
        row.update(ci_low=low, ci_high=high)
    return row


def analyse(records, *, n_boot=10000, seed=20260921, use_scipy=True):
    """All tables from validated records.  Raises NoRatingsError when there is nothing to analyse."""
    if not records:
        raise NoRatingsError('no valid ratings remain after validation; no statistics were computed')
    methods = method_order({r['method'] for r in records})
    levels = sorted({r['lod_level'] for r in records if r['method'] != CONTROL})
    summary = [describe(records, m, level, c, n_boot, seed)
               for level in levels + (['all'] if len(levels) > 1 else []) for c in CRITERIA for m in methods]
    control = [describe(records, CONTROL, 'all', c, n_boot, seed) for c in CRITERIA
               if any(r['method'] == CONTROL for r in records)]
    tests, pairwise, means_rows = [], [], []
    for level in levels:
        for criterion in CRITERIA:
            participants, matrix = participant_means(records, methods, level, criterion)
            for pid, row in zip(participants, matrix):
                means_rows += [dict(participant=pid, lod_level=level, criterion=criterion, method=m, mean=float(v))
                               for m, v in zip(methods, row)]
            test = friedman(matrix, use_scipy) if len(methods) >= 2 else dict(
                n=len(participants), k=len(methods), statistic=math.nan, df=0, p=math.nan, kendalls_w=math.nan,
                mean_ranks=[], engine='', note='needs >= 2 methods')
            tests.append(dict(lod_level=level, criterion=criterion, methods=';'.join(methods),
                              **{k: v for k, v in test.items() if k != 'mean_ranks'},
                              mean_ranks=';'.join(f'{m}={r:.3f}' for m, r in zip(methods, test['mean_ranks']))))
            family = []
            for i, a in enumerate(methods):
                for b in methods[i + 1:]:
                    x, y = matrix[:, methods.index(a)], matrix[:, methods.index(b)]
                    w = wilcoxon_signed_rank(x, y)
                    family.append(dict(lod_level=level, criterion=criterion, method_a=a, method_b=b,
                                       median_difference=float(np.median(x - y)) if len(x) else math.nan,
                                       mean_difference=float(np.mean(x - y)) if len(x) else math.nan, **w))
            for row, adjusted in zip(family, holm([r['p'] for r in family])):
                row['p_holm'] = float(adjusted)
            pairwise += family
    for row, adjusted in zip(pairwise, holm([r['p'] for r in pairwise])):
        row['p_holm_all'] = float(adjusted)
    agreement = agreement_rows(records)
    per_asset = []
    groups = {}
    for r in records:
        for criterion in CRITERIA:
            if r[criterion] is not None:
                groups.setdefault((r['asset_id'], r['method'], r['lod_level'], criterion), []).append(r[criterion])
    for (asset, method, level, criterion), values in sorted(groups.items(), key=lambda item: str(item[0])):
        per_asset.append(dict(asset_id=asset, method=method, lod_level=level, criterion=criterion,
                              n_ratings=len(values), mean=float(np.mean(values)), median=float(np.median(values))))
    return dict(methods=methods, levels=levels, summary=summary, control=control, friedman=tests,
                pairwise=pairwise, agreement=agreement, per_asset=per_asset, participant_means=means_rows,
                n_boot=n_boot, seed=seed, scipy=(_scipy_stats is not None and use_scipy))


def agreement_rows(records):
    """Kendall's W per criterion over the non-control stimuli: pooled (scope 'pooled'), within each
    LOD ('lod') and within each asset x LOD ('asset_lod').  Judges are the participants who rated
    every stimulus of the scope on that criterion.  The pooled W mostly measures the LOD (and asset)
    differences everybody sees; the within scopes measure agreement about the methods."""
    rated = [r for r in records if r['method'] != CONTROL]
    levels, assets = sorted({r['lod_level'] for r in rated}), sorted({r['asset_id'] for r in rated})
    scopes = [('pooled', 'all', 'all', rated)]
    scopes += [('lod', lv, 'all', [r for r in rated if r['lod_level'] == lv]) for lv in levels]
    scopes += [('asset_lod', lv, a, [r for r in rated if r['lod_level'] == lv and r['asset_id'] == a])
               for lv in levels for a in assets]
    rows = []
    for criterion in CRITERIA:
        for scope, level, asset, chosen in scopes:
            stimuli = sorted({r['stimulus_id'] for r in chosen})
            table = {}
            for r in chosen:
                if r[criterion] is not None:
                    table.setdefault(r['participant'], {})[r['stimulus_id']] = r[criterion]
            judges = sorted(p for p, cells in table.items() if len(cells) == len(stimuli))
            matrix = np.array([[table[p][s] for s in stimuli] for p in judges], dtype=float)
            rows.append(dict(criterion=criterion, scope=scope, lod_level=level, asset_id=asset,
                             n_participants=len(judges), n_stimuli=len(stimuli),
                             kendalls_w=kendalls_w(matrix) if len(judges) >= 2 else math.nan))
    return rows


def run(rating_sources, key_text, stimuli_text, *, incomplete='exclude', control_min=None,
        n_boot=10000, seed=20260921, use_scipy=True):
    """In-memory pipeline: -> dict(study_id, key, stimuli, issues, participants, records, results)."""
    study_id, build_id, key = load_key(key_text)
    stimuli = load_stimuli(stimuli_text, key, build_id)
    rating_sources = list(rating_sources)
    if not rating_sources:
        raise NoRatingsError('no rating files were given or found; nothing was analysed')
    rows, issues = load_ratings(rating_sources)
    records, more, participants = validate(rows, study_id, build_id, stimuli, incomplete=incomplete,
                                           control_min=control_min)
    issues += more
    state = dict(study_id=study_id, build_id=build_id, key=key, stimuli=stimuli, issues=issues,
                 participants=participants, records=records, n_rows=len(rows), results=None)
    if records:
        state['results'] = analyse(records, n_boot=n_boot, seed=seed, use_scipy=use_scipy)
    return state


# ----------------------------------------------------------------- output

def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return 'NA'
    return f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def fmt_p(value):
    if value is None or not math.isfinite(value):
        return 'NA'
    return '<0.001' if value < 0.001 else f'{value:.3f}'


def markdown_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    return '\n'.join(lines + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows])


def render_report(state, inputs=None):
    """Indonesian Markdown report; states when nothing could be computed."""
    results, issues, participants = state['results'], state['issues'], state['participants']
    included = [p for p in participants if p['status'] == 'included']
    lines = [f'# Analisis studi pengguna: {state["study_id"]} (build {state.get("build_id", "-")})', '',
             f'Dibuat {datetime.now(timezone.utc).replace(microsecond=0).isoformat()} oleh '
             '`scripts/analyze_user_study.py`. Laporan ini **membuka kode metode** (unblinded); simpan secara '
             'privat sampai pengumpulan data selesai.', '']
    if inputs:
        lines += ['## Berkas masukan', '', markdown_table(
            ['Berkas', 'SHA-256 (12 karakter awal)'], [(f'`{n}`', h[:12]) for n, h in inputs]), '']
    counts = {}
    for item in issues:
        counts[(item['severity'], item['code'])] = counts.get((item['severity'], item['code']), 0) + 1
    lines += ['## Validasi', '',
              f'- Baris rating dibaca: {state["n_rows"]}; baris valid yang dianalisis: {len(state["records"])}.',
              f'- Peserta: {len(participants)} ditemukan, {len(included)} dianalisis, '
              f'{len(participants) - len(included)} dikeluarkan.',
              f'- Stimulus dalam studi: {len(state["stimuli"])}.', '']
    if counts:
        lines += [markdown_table(['Tingkat', 'Kode', 'Jumlah'],
                                 [(s, f'`{c}`', n) for (s, c), n in sorted(counts.items())]),
                  '', 'Rincian lengkap: `validation_issues.csv`.', '']
    else:
        lines += ['Tidak ada masalah validasi.', '']
    if participants:
        lines += [markdown_table(['Peserta', 'Baris', 'Stimulus lengkap', 'Status', 'Alasan'],
                                 [(p['participant'], p['rows'], f'{p["complete_stimuli"]}/{p["total_stimuli"]}',
                                   p['status'], p['reason'] or '-') for p in participants]), '']
    if results is None:
        lines += ['## Hasil', '', '**Tidak ada rating valid.** Tidak ada statistik yang dihitung; '
                  'periksa masalah validasi di atas.', '']
        return '\n'.join(lines)
    methods = results['methods']
    lines += ['## Statistik deskriptif', '',
              f'Rata-rata [IK 95% bootstrap persentil, {results["n_boot"]} ulangan, klaster = peserta, '
              f'seed {results["seed"]}], median, dan n rating. Aset tidak di-resample, sehingga variasi antar-aset '
              'tidak tercakup dalam interval. Skala ordinal 1-5; rata-rata hanya ringkasan.', '']
    for level in results['levels'] + (['all'] if len(results['levels']) > 1 else []):
        rows = []
        for criterion in CRITERIA:
            cells = []
            for method in methods:
                s = next(r for r in results['summary'] if r['method'] == method and r['lod_level'] == level
                         and r['criterion'] == criterion)
                cells.append(f'{fmt(s["mean"])} [{fmt(s["ci_low"])}, {fmt(s["ci_high"])}]; '
                             f'med {fmt(s["median"], 1)}; n={s["n_ratings"]}')
            rows.append([CRITERION_NAMES[criterion]] + cells)
        title = 'semua LOD (gabungan)' if level == 'all' else f'LOD{level}'
        lines += [f'### {title}', '', markdown_table(['Kriteria'] + methods, rows), '']
    if results['control']:
        lines += ['### Stimulus kontrol (Referensi vs Referensi)', '',
                  markdown_table(['Kriteria', 'Rata-rata [IK 95%]', 'n'],
                                 [(CRITERION_NAMES[c['criterion']], f'{fmt(c["mean"])} [{fmt(c["ci_low"])}, '
                                   f'{fmt(c["ci_high"])}]', c['n_ratings']) for c in results['control']]), '']
    engine = 'SciPy' if results['scipy'] else 'fallback NumPy (terdokumentasi dalam skrip)'
    lines += ['## Uji Friedman antar-metode', '',
              f'Data: rata-rata per peserta untuk tiap metode pada satu LOD dan kriteria (hanya peserta yang memiliki '
              f'semua metode). Mesin: {engine}. W = Kendall\'s W = chi2 / (n (k - 1)).', '',
              markdown_table(['LOD', 'Kriteria', 'n', 'k', 'chi2', 'df', 'p', 'W', 'Rata-rata peringkat', 'Catatan'],
                             [(t['lod_level'], CRITERION_NAMES[t['criterion']], t['n'], t['k'], fmt(t['statistic']),
                               t['df'], fmt_p(t['p']), fmt(t['kendalls_w']), t['mean_ranks'] or '-',
                               t['note'] or '-') for t in results['friedman']]), '']
    lines += ['## Wilcoxon signed-rank berpasangan', '',
              'Selisih = metode A - metode B pada rata-rata per peserta. W = min(R+, R-). `p_holm`: koreksi Holm '
              'dalam satu LOD x kriteria; `p_holm_all`: koreksi Holm atas semua uji berpasangan pada tabel ini. '
              'r = korelasi rank-biserial berpasangan (positif berarti A lebih tinggi).', '',
              markdown_table(['LOD', 'Kriteria', 'A', 'B', 'n (non-nol)', 'Median selisih', 'W', 'p', 'p_holm',
                              'p_holm_all', 'r', 'Metode p'],
                             [(w['lod_level'], CRITERION_NAMES[w['criterion']], w['method_a'], w['method_b'],
                               f'{w["n_pairs"]} ({w["n_nonzero"]})', fmt(w['median_difference']), fmt(w['statistic'], 1),
                               fmt_p(w['p']), fmt_p(w['p_holm']), fmt_p(w['p_holm_all']), fmt(w['rank_biserial']),
                               w['method']) for w in results['pairwise']]), '']
    agreement, levels, rows = results['agreement'], results['levels'], []
    for criterion in CRITERIA:
        own = [a for a in agreement if a['criterion'] == criterion]
        cells = [f'{fmt(a["kendalls_w"])} ({a["n_participants"]} p, {a["n_stimuli"]} s)'
                 for a in own if a['scope'] in ('pooled', 'lod')]
        within = [a['kendalls_w'] for a in own if a['scope'] == 'asset_lod' and math.isfinite(a['kendalls_w'])]
        cells.append(f'{fmt(float(np.mean(within)) if within else math.nan)} ({len(within)} kelompok)')
        rows.append([CRITERION_NAMES[criterion]] + cells)
    lines += ['## Kesepakatan antar-peserta', '',
              'Kendall\'s W atas stimulus non-kontrol; penilai = peserta dengan rating lengkap pada cakupan dan '
              'kriteria itu (p = peserta, s = stimulus). **W gabungan didominasi perbedaan LOD dan aset** yang mudah '
              'disepakati semua peserta, sehingga tidak menunjukkan kesepakatan tentang metode. Kesepakatan tentang '
              'metode dibaca dari W di dalam satu LOD dan rata-rata W di dalam aset x LOD (kelompok tanpa W yang '
              'terdefinisi tidak dihitung). Rincian: `agreement.csv`.', '',
              markdown_table(['Kriteria', 'Gabungan'] + [f'Dalam LOD{lv}' for lv in levels]
                             + ['Rata-rata dalam aset x LOD'], rows), '']
    n_included = len(included)
    lines += ['## Batas interpretasi', '']
    if n_included < 10:
        lines += [f'- **Hanya {n_included} peserta dianalisis.** Dengan n pasangan non-nol, p exact Wilcoxon dua sisi '
                  f'terkecil yang mungkin adalah 2/2^n ({fmt_p(2.0 / 2 ** max(n_included, 1))} untuk n = {n_included}); '
                  'pendekatan chi-square pada uji Friedman juga kurang akurat untuk n kecil. Perlakukan hasil ini '
                  'sebagai pilot, bukan kesimpulan.']
    lines += ['- Hasil hanya berlaku untuk aset, view, LOD, dan peserta dalam studi ini; jumlah aset kecil membatasi '
              'generalisasi, dan IK tidak memasukkan variasi antar-aset.',
              '- Nilai p yang tidak signifikan bukan bukti bahwa metode setara. Laporkan ukuran efek dan IK.',
              '- Rata-rata per peserta menyamakan bobot setiap peserta, tetapi menghilangkan variasi dalam peserta.',
              '- Rincian per aset: `per_asset.csv`; data per rating (unblinded): `merged_ratings.csv`.', '']
    return '\n'.join(lines)


def write_csv(path, rows, columns=None):
    columns = columns or (list(rows[0]) if rows else [])
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()})


def write_outputs(state, out, inputs):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.md').write_text(render_report(state, inputs), encoding='utf-8')
    write_csv(out / 'validation_issues.csv', state['issues'],
              ['severity', 'code', 'file', 'line', 'participant', 'message'])
    write_csv(out / 'participants.csv', state['participants'],
              ['participant', 'files', 'rows', 'complete_stimuli', 'total_stimuli', 'control_mean',
               'order_matches_form', 'status', 'reason'])
    written = ['report.md', 'validation_issues.csv', 'participants.csv']
    results = state['results']
    if results is not None:
        write_csv(out / 'merged_ratings.csv', state['records'],
                  ['participant', 'stimulus_id', 'asset_id', 'method_code', 'method', 'lod_level', 'view_set_id',
                   'target_ratio', 'actual_triangles', 'presentation_order', *CRITERIA, 'notes',
                   'export_sequence', 'exported_at', 'file', 'line'])
        for name in ('summary', 'friedman', 'pairwise', 'agreement', 'per_asset', 'participant_means'):
            rows = results[name] + (results['control'] if name == 'summary' else [])
            write_csv(out / f'{name}.csv', rows)
            written.append(f'{name}.csv')
        written.insert(3, 'merged_ratings.csv')
    meta = dict(generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                generated_by='scripts/analyze_user_study.py', python=platform.python_version(),
                numpy=np.__version__, scipy=getattr(sys.modules.get('scipy'), '__version__', None),
                scipy_used=bool(results and results['scipy']), inputs=[dict(file=n, sha256=h) for n, h in inputs],
                n_boot=results['n_boot'] if results else None, seed=results['seed'] if results else None,
                files=written)
    (out / 'analysis_meta.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return written


def rating_paths(items):
    paths = []
    for item in items:
        if item.is_dir():
            paths += sorted(p for p in item.glob('*.csv') if p.is_file())
        elif item.is_file():
            paths.append(item)
        else:
            raise StudyInputError(f'{item} does not exist')
    unique = []
    for path in paths:
        if path.resolve() not in {p.resolve() for p in unique}:
            unique.append(path)
    return unique


def main(argv=None):
    parser = argparse.ArgumentParser(prog='analyze_user_study.py', description=__doc__.split('\n')[0])
    parser.add_argument('--study', type=Path, help='study folder made by make_user_study.py')
    parser.add_argument('--ratings', type=Path, nargs='+', help='rating CSV files or folders (default <study>/ratings)')
    parser.add_argument('--key', type=Path, help='default <study>/private/method_key.csv')
    parser.add_argument('--stimuli', type=Path, help='default <study>/stimuli.csv')
    parser.add_argument('--out', type=Path, help='default <study>/analysis')
    parser.add_argument('--bootstrap', type=int, default=10000, help='bootstrap resamples (default 10000)')
    parser.add_argument('--seed', type=int, default=20260921, help='bootstrap seed')
    parser.add_argument('--incomplete', choices=('exclude', 'include'), default='exclude')
    parser.add_argument('--control-min', type=float,
                        help='exclude participants whose mean overall rating of control stimuli is below this')
    parser.add_argument('--no-scipy', action='store_true', help='use the NumPy fallback even if SciPy is installed')
    args = parser.parse_args(argv)
    if not args.study and not (args.ratings and args.key and args.stimuli and args.out):
        parser.error('give --study, or all of --ratings, --key, --stimuli and --out')
    key_path = args.key or args.study / 'private' / 'method_key.csv'
    stimuli_path = args.stimuli or args.study / 'stimuli.csv'
    out = args.out or args.study / 'analysis'
    try:
        if args.ratings:
            paths = rating_paths(args.ratings)
        elif (args.study / 'ratings').is_dir():
            paths = rating_paths([args.study / 'ratings'])
        else:
            paths = []
        for path in (key_path, stimuli_path):
            if not path.is_file():
                raise StudyInputError(f'{path} not found')
        inputs, sources = [], []
        for path in [key_path, stimuli_path] + paths:
            data = path.read_bytes()
            inputs.append((str(path), hashlib.sha256(data).hexdigest()))
            if path in paths:
                sources.append((path.name, decode(data, path.name)))
        state = run(sources, decode(key_path.read_bytes(), key_path.name),
                    decode(stimuli_path.read_bytes(), stimuli_path.name), incomplete=args.incomplete,
                    control_min=args.control_min, n_boot=args.bootstrap, seed=args.seed,
                    use_scipy=not args.no_scipy)
    except NoRatingsError as exc:
        print(f'Belum ada data rating untuk dianalisis: {exc}. '
              'Tidak ada file yang ditulis dan tidak ada angka yang dibuat.', file=sys.stderr)
        return 2
    except StudyInputError as exc:
        print(f'Masukan tidak valid: {exc}', file=sys.stderr)
        return 1
    written = write_outputs(state, out, inputs)
    errors = sum(1 for i in state['issues'] if i['severity'] == 'error')
    print(f'{len(state["records"])} valid ratings from '
          f'{sum(p["status"] == "included" for p in state["participants"])} participant(s); '
          f'{errors} validation error(s), {len(state["issues"]) - errors} warning(s).')
    print(f'Wrote {", ".join(written)} to {out}')
    if state['results'] is None:
        print('No valid ratings: no statistics were computed (see report.md and validation_issues.csv).',
              file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
