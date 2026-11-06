"""Statistics and loader tests for scripts/analyze_results.py, plus one harness check; synthetic data only.

Run from the repository root:
    python tests/test_analysis.py

Every input below is synthetic and in memory (or in a temporary directory);
no experiment result is read or produced. ``HarnessInBlenderTests`` runs
``scripts/run_experiment.py`` inside Blender on a generated toy asset in a
temporary directory (resume code check, LOD file layers, timing components);
it is skipped when no Blender executable is found (``BLENDER`` environment
variable, the default Windows install path, or ``blender`` on PATH).
"""

import csv
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('analyze_results', ROOT / 'scripts' / 'analyze_results.py')
analysis = importlib.util.module_from_spec(_spec)
sys.modules['analyze_results'] = analysis  # dataclasses resolve their module by name
_spec.loader.exec_module(analysis)


def brute_force_p(differences):
    """Two-sided exact p by enumerating all 2^n sign patterns of the nonzero differences."""
    d = np.asarray(differences, dtype=float)
    d = d[d != 0]
    ranks = analysis.average_ranks(np.abs(d))
    observed = ranks[d > 0].sum()
    sums = np.array([sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product((0, 1), repeat=len(d))])
    lower, upper = np.mean(sums <= observed + 1e-9), np.mean(sums >= observed - 1e-9)
    return min(1.0, 2 * min(lower, upper))


class RankAndWilcoxonTests(unittest.TestCase):
    def test_average_ranks_share_ties(self):
        np.testing.assert_array_equal(analysis.average_ranks([3, 1, 2, 2]), [4, 1, 2.5, 2.5])
        np.testing.assert_array_equal(analysis.average_ranks([5, 5, 5]), [2, 2, 2])
        self.assertEqual(len(analysis.average_ranks([])), 0)

    def test_exact_p_known_small_cases(self):
        self.assertAlmostEqual(analysis.wilcoxon_exact_p([1, 2, 3]), 2 / 8)
        self.assertAlmostEqual(analysis.wilcoxon_exact_p([-1, -2, -3, -4, -5]), 2 / 32)
        # W- = 2 (rank of -2); P(W+ >= 13) = P(W- <= 2) = 3/32 for n = 5.
        self.assertAlmostEqual(analysis.wilcoxon_exact_p([1, -2, 3, 4, 5]), 6 / 32)
        self.assertAlmostEqual(analysis.wilcoxon_exact_p([0, 0, 1, 2, 3]), 2 / 8)  # zeros dropped
        self.assertAlmostEqual(analysis.wilcoxon_exact_p([1, -1]), 1.0)
        self.assertIsNone(analysis.wilcoxon_exact_p([0, 0, 0]))

    def test_exact_p_matches_enumeration_with_ties(self):
        rng = np.random.default_rng(7)
        for _ in range(60):
            n = int(rng.integers(1, 11))
            d = rng.integers(-3, 4, size=n).astype(float)  # many ties and zeros
            with self.subTest(d=d.tolist()):
                expected = None if not np.any(d) else brute_force_p(d)
                got = analysis.wilcoxon_exact_p(d)
                if expected is None:
                    self.assertIsNone(got)
                else:
                    self.assertAlmostEqual(got, expected, places=12)

    def test_exact_p_matches_scipy_without_ties(self):
        if analysis.scipy_stats is None:
            self.skipTest('SciPy not installed')
        rng = np.random.default_rng(11)
        for n in range(1, 21):
            d = rng.normal(0.3, 1.0, size=n)
            with self.subTest(n=n):
                reference = analysis.scipy_stats.wilcoxon(d, zero_method='wilcox', method='exact').pvalue
                self.assertAlmostEqual(analysis.wilcoxon_exact_p(d), float(reference), places=12)

    def test_method_choice_is_documented(self):
        p, method = analysis.wilcoxon_test([0.1, -0.2, 0.3, 0.4], use_scipy=False)
        self.assertEqual(method, 'exact_sign_flip')
        self.assertAlmostEqual(p, analysis.wilcoxon_exact_p([0.1, -0.2, 0.3, 0.4]))
        p, method = analysis.wilcoxon_test([0.1, -0.1, 0.3, 0.4])  # tie in |d|: never SciPy's approximation
        self.assertEqual(method, 'exact_sign_flip')
        self.assertEqual(analysis.wilcoxon_test([0.0, 0.0]), (None, 'all_differences_zero'))
        if analysis.scipy_stats is not None:
            self.assertEqual(analysis.wilcoxon_test([0.1, -0.2, 0.3, 0.4])[1], 'scipy_exact')
        big = np.random.default_rng(3).normal(0.2, 1.0, size=analysis.EXACT_MAX_N + 10)
        p, method = analysis.wilcoxon_test(big, use_scipy=False)
        self.assertEqual(method, 'normal_approx')
        self.assertTrue(0.0 <= p <= 1.0)

    def test_normal_approximation_is_close_to_exact(self):
        d = np.random.default_rng(5).normal(0.25, 1.0, size=40)
        self.assertLess(abs(analysis.wilcoxon_normal_p(d) - analysis.wilcoxon_exact_p(d)), 0.01)
        if analysis.scipy_stats is not None:
            approx = analysis.scipy_stats.wilcoxon(d, method='approx', correction=False).pvalue
            self.assertAlmostEqual(analysis.wilcoxon_normal_p(d), float(approx), places=10)


class AdjustmentAndEffectTests(unittest.TestCase):
    def test_holm_known_values(self):
        adjusted = analysis.holm_adjust([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])
        self.assertEqual(analysis.holm_adjust([None, 0.02, 0.5]), [None, 0.04, 0.5])
        self.assertEqual(analysis.holm_adjust([0.6, 0.7]), [1.0, 1.0])
        self.assertEqual(analysis.holm_adjust([]), [])

    def test_holm_is_monotone_and_not_smaller_than_raw(self):
        p = np.random.default_rng(2).uniform(0, 0.2, size=15).tolist()
        adjusted = analysis.holm_adjust(p)
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(np.array(adjusted)[order]) >= 0))
        self.assertTrue(all(a >= r for a, r in zip(adjusted, p)))

    def test_rank_biserial(self):
        self.assertEqual(analysis.rank_biserial([1, 2, 3]), 1.0)
        self.assertEqual(analysis.rank_biserial([-1, -2, -3]), -1.0)
        self.assertAlmostEqual(analysis.rank_biserial([1, -2, 3]), (1 + 3 - 2) / 6)
        self.assertAlmostEqual(analysis.rank_biserial([0, 1, -2, 3]), (1 + 3 - 2) / 6)  # zeros dropped
        self.assertIsNone(analysis.rank_biserial([0, 0]))

    def test_direction_handling(self):
        chamfer, iou = analysis.METRIC['chamfer_n'], analysis.METRIC['iou_multiview']
        self.assertAlmostEqual(float(analysis.improvement(0.1, 0.2, chamfer.better)), 0.1)
        self.assertAlmostEqual(float(analysis.improvement(0.9, 0.8, iou.better)), 0.1)
        lower = analysis.paired_comparison([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], chamfer, n_boot=200, seed=1)
        self.assertEqual((lower['wins'], lower['ties'], lower['losses']), (3, 0, 0))
        self.assertEqual(lower['rank_biserial'], 1.0)
        self.assertLess(lower['mean_diff'], 0)  # delta stays PERCEPTUAL - baseline
        higher = analysis.paired_comparison([0.95, 0.96], [0.90, 0.91], iou, n_boot=200, seed=1)
        self.assertEqual((higher['wins'], higher['losses']), (2, 0))
        self.assertGreater(higher['mean_diff'], 0)
        worse = analysis.paired_comparison([0.85, 0.86], [0.90, 0.91], iou, n_boot=200, seed=1)
        self.assertEqual((worse['wins'], worse['losses'], worse['rank_biserial']), (0, 2, -1.0))

    def test_tie_tolerances(self):
        chamfer, iou = analysis.METRIC['chamfer_n'], analysis.METRIC['iou_axis']
        result = analysis.paired_comparison([1.005, 0.98, 1.02], [1.0, 1.0, 1.0], chamfer, n_boot=0, tie_rel=0.01)
        self.assertEqual((result['wins'], result['ties'], result['losses']), (1, 1, 1))
        result = analysis.paired_comparison([0.9004, 0.9010], [0.9, 0.9], iou, n_boot=0, tie_iou=5e-4)
        self.assertEqual((result['wins'], result['ties'], result['losses']), (1, 1, 0))
        self.assertIsNone(result['ci_low'])  # n = 2 < CI_MIN_N: no CI
        self.assertIsNone(result['boot_ci_low'])

    def test_empty_comparison(self):
        result = analysis.paired_comparison([], [], analysis.METRIC['hausdorff_n'])
        self.assertEqual(result['n'], 0)
        self.assertIsNone(result['p'])


class BootstrapTests(unittest.TestCase):
    def test_deterministic_for_a_seed(self):
        values = np.random.default_rng(0).normal(size=9)
        first = analysis.cluster_bootstrap_ci(values, n_boot=2000, seed=[1, 2])
        self.assertEqual(first, analysis.cluster_bootstrap_ci(values, n_boot=2000, seed=[1, 2]))
        self.assertNotEqual(first, analysis.cluster_bootstrap_ci(values, n_boot=2000, seed=[1, 3]))

    def test_sanity(self):
        self.assertEqual(analysis.cluster_bootstrap_ci([0.5] * 6, n_boot=500, seed=0), (0.5, 0.5))
        values = np.random.default_rng(4).normal(1.0, 0.1, size=12)
        low, high = analysis.cluster_bootstrap_ci(values, n_boot=4000, seed=0)
        self.assertLess(low, values.mean())
        self.assertGreater(high, values.mean())
        self.assertLess(high - low, 0.2)
        self.assertEqual(analysis.cluster_bootstrap_ci([1.0], n_boot=100, seed=0), (None, None))
        with self.assertRaises(ValueError):
            analysis.cluster_bootstrap_ci([1.0, 2.0], clusters=['a'], n_boot=10)

    def test_whole_clusters_are_resampled(self):
        # Two clusters: pooled means are 0 (a,a), 2 (a,b) or 10 (b,b).
        values, clusters = [0.0, 0.0, 0.0, 0.0, 10.0], ['a', 'a', 'a', 'a', 'b']
        low, high = analysis.cluster_bootstrap_ci(values, clusters, n_boot=4000, seed=0)
        self.assertEqual((low, high), (0.0, 10.0))
        low_iid, high_iid = analysis.cluster_bootstrap_ci(values, None, n_boot=4000, seed=0)
        self.assertLess(high_iid, 10.0)  # resampling single points cannot reach 10

    def test_pareto_mask(self):
        points = [[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [2.0, 0.5], [np.nan, 1.0]]
        mask = analysis.pareto_mask(points, ('min', 'max'))
        self.assertEqual(mask.tolist(), [True, True, True, False, False])


class ShiftIntervalTests(unittest.TestCase):
    """Exact Walsh-average (Hodges-Lehmann) CI that replaced the percentile bootstrap as the reported CI."""

    def test_critical_values_match_the_exact_null_distribution(self):
        # P(T+ <= c) with ranks 1..n: n = 9 -> 10/512 at c = 5 (c = 6 gives 14/512 > 0.025);
        # n = 7 -> 3/128 at c = 2; n = 6 -> 1/64 at c = 0; n = 5 -> 1/32 > 0.025 at c = 0 (no 95 % CI).
        for n, c, tail in ((9, 5, 10 / 512), (7, 2, 3 / 128), (6, 0, 1 / 64)):
            d = np.random.default_rng(n).normal(size=n)
            walsh = analysis.walsh_averages(d)
            self.assertEqual(len(walsh), n * (n + 1) // 2)
            low, high, level = analysis.wilcoxon_shift_ci(d)
            with self.subTest(n=n):
                self.assertEqual((low, high), (walsh[c], walsh[len(walsh) - 1 - c]))
                self.assertAlmostEqual(level, 1 - 2 * tail, places=12)
        d = np.random.default_rng(6).normal(size=6)
        self.assertEqual(analysis.wilcoxon_shift_ci(d)[:2], (d.min(), d.max()))  # n = 6: [min, max] at 96.9 %
        for n in (5, 4, 3, 2, 1, 0):
            self.assertEqual(analysis.wilcoxon_shift_ci(np.arange(1.0, n + 1)), (None, None, None))

    def test_interval_is_the_inverted_signed_rank_test(self):
        # For continuous data theta lies inside the CI exactly when the exact two-sided p of d - theta
        # exceeds 1 - attained level; checked at the endpoints and at random shifts.
        rng = np.random.default_rng(21)
        for n in range(6, 13):
            d = rng.normal(0.3, 1.0, size=n)
            low, high, level = analysis.wilcoxon_shift_ci(d)
            span = high - low
            for theta in (0.0, low - 1e-9 * span, low + 1e-9 * span, high - 1e-9 * span, high + 1e-9 * span,
                          *rng.uniform(low - span, high + span, size=8)):
                with self.subTest(n=n, theta=theta):
                    inside = low <= theta <= high
                    self.assertEqual(inside, analysis.wilcoxon_exact_p(d - theta) > 1 - level + 1e-12)

    def test_coverage_is_at_least_nominal_where_the_bootstrap_was_not(self):
        # Symmetric data around 0.25: the exact interval covers the shift with its attained level
        # (0.961 at n = 9, 0.953 at n = 7); the old percentile bootstrap covered about 0.89-0.91 / 0.87-0.89.
        rng = np.random.default_rng(3)
        for n in (9, 7):
            hits = sum(low <= 0.25 <= high for low, high, _ in
                       (analysis.wilcoxon_shift_ci(rng.normal(0.25, 1.0, size=n)) for _ in range(3000)))
            with self.subTest(n=n):
                self.assertGreater(hits / 3000, 0.94)

    def test_hodges_lehmann_and_large_n(self):
        self.assertEqual(analysis.hodges_lehmann([1.0, 2.0, 3.0]), 2.0)  # Walsh: 1, 1.5, 2, 2, 2.5, 3
        self.assertEqual(analysis.hodges_lehmann([0.0, 10.0]), 5.0)  # Walsh: 0, 5, 10
        self.assertIsNone(analysis.hodges_lehmann([]))
        d = np.random.default_rng(8).normal(size=analysis.EXACT_MAX_N + 20)  # normal approximation of c
        low, high, level = analysis.wilcoxon_shift_ci(d)
        self.assertTrue(low < analysis.hodges_lehmann(d) < high)
        self.assertAlmostEqual(level, 0.95, delta=0.005)

    def test_small_n_has_no_ci(self):
        # The review case: three positive deltas gave a bootstrap "95 % CI" of [min, max] excluding 0
        # while the exact p is 0.25. Now there is no CI below CI_MIN_N pairs.
        result = analysis.paired_comparison([1e-5, 5e-5, 1.5e-4], [0.0, 0.0, 0.0], analysis.METRIC['chamfer_n'],
                                            n_boot=2000, seed=0)
        self.assertEqual((result['ci_low'], result['ci_high'], result['ci_level']), (None, None, None))
        self.assertEqual((result['boot_ci_low'], result['boot_ci_high']), (None, None))
        self.assertAlmostEqual(result['p'], 0.25)
        self.assertAlmostEqual(result['hl_diff'], np.median([1e-5, 3e-5, 8e-5, 5e-5, 1e-4, 1.5e-4]))
        nine = analysis.paired_comparison(np.arange(1.0, 10.0), np.zeros(9), analysis.METRIC['chamfer_n'],
                                          n_boot=500, seed=0)
        self.assertAlmostEqual(nine['ci_level'], 1 - 20 / 512)
        self.assertEqual((nine['ci_low'], nine['ci_high']), (2.5, 7.5))  # 6th and 40th Walsh averages of 1..9
        self.assertIsNotNone(nine['boot_ci_low'])


def quality_record(asset, condition, ratio, chamfer, iou, split='evaluation', category='organic', actual=100,
                   experiment='main', roi=None, strength=None):
    """A synthetic quality record; ``roi`` (a distance) adds every ROI metric key, else they are absent."""
    metrics = dict(chamfer_mean_normalized=chamfer, hausdorff_sampled_normalized=2 * chamfer,
                   hausdorff_with_vertices_normalized=2.5 * chamfer, silhouette_iou_mean=iou,
                   silhouette_iou_multiview_mean=iou,
                   sampling=dict(samples_per_mesh=100, seed=1, silhouette_resolution=32))
    if roi is not None:
        metrics.update(roi_chamfer_ref_to_candidate_normalized=roi, roi_hausdorff_ref_to_candidate_normalized=3 * roi,
                       roi_chamfer_candidate_to_ref_normalized=1.5 * roi,
                       roi_hausdorff_candidate_to_ref_normalized=4 * roi, roi_chamfer_symmetric_normalized=1.25 * roi,
                       roi_hausdorff_symmetric_normalized=4 * roi)
    if strength is None:
        strength = 10.0 if condition == 'PERCEPTUAL' else 0.0
    return dict(record_type='quality', asset_id=asset, asset_index=0, category=category, split=split,
                experiment=experiment, condition=condition, method=condition, level=1, ratio=ratio,
                source_triangles=1000, target_triangles=100, actual_triangles=actual, target_reached=True,
                params=dict(strength=strength, weighting='quadric'), metrics=metrics, volume_ratio=0.99, checks={})


def read_csv(path):
    with open(path, encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_raw(folder, records, environment=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'raw.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    if environment is not None:
        (folder / 'environment.json').write_text(json.dumps(environment), encoding='utf-8')


def evaluation_split(ratios=(0.5, 0.2, 0.05), roi_missing=('e7', 'e8')):
    """9 evaluation assets, 2 without ROI; PERCEPTUAL better than both baselines on every asset and metric."""
    records = []
    for index in range(9):
        asset = f'e{index}'
        for ratio in ratios:
            base = (1 + 0.1 * index) * 1e-3 / ratio
            gain = (1 + index) * 1e-5  # distinct per asset: no ties in |delta|
            roi = None if asset in roi_missing else base
            for condition, shift in (('QEM', 0.0), ('DECIMATE', 0.5 * gain)):
                records.append(quality_record(asset, condition, ratio, base + shift, 0.9 - shift,
                                              roi=None if roi is None else roi + shift))
            records.append(quality_record(asset, 'PERCEPTUAL', ratio, base - gain, 0.9 + gain,
                                          roi=None if roi is None else roi - gain))
    return records


class LoaderTests(unittest.TestCase):
    def test_add_line_robustness(self):
        results = analysis.Results()
        results.add_line('', 'blank')
        results.add_line('{"record_type": "quality", "asset_id": "a"', 'truncated')
        results.add_line('[1, 2]', 'list')
        results.add_line(json.dumps({'record_type': 'mystery'}), 'unknown')
        results.add_line(json.dumps({'record_type': 'quality', 'asset_id': 'a'}), 'incomplete')
        results.add_line(json.dumps({'record_type': 'error', 'scope': 'asset', 'asset_id': 'b', 'error': 'x'}), 'err')
        results.add_line(json.dumps(quality_record('a', 'QEM', 0.2, 1e-3, 0.9)), 'ok')
        results.add_line(json.dumps(quality_record('a', 'QEM', 0.2, 2e-3, 0.8)), 'duplicate')
        self.assertEqual(len(results.quality), 1)
        self.assertEqual(len(results.errors), 1)
        self.assertEqual(len(results.problems), 4)
        self.assertEqual(results.duplicates, 1)
        row = results.quality[('a', 'main', 'QEM', 0.2)]
        self.assertEqual(row['chamfer_n'], 2e-3)  # the later record wins
        self.assertIsNone(row['roi_chamfer_n'])  # missing metric -> None, not 0

    def test_nonfinite_and_bool_values_are_not_numbers(self):
        self.assertIsNone(analysis.number(float('nan')))
        self.assertIsNone(analysis.number(True))
        self.assertIsNone(analysis.number('abc'))
        self.assertEqual(analysis.number('0.5'), 0.5)

    def test_end_to_end_on_synthetic_directory(self):
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            lines = []
            for index in range(6):
                asset = f'synthetic_{index}'
                for ratio in (0.2, 0.05):
                    base = rng.uniform(1e-3, 2e-3) / ratio
                    lines.append(quality_record(asset, 'QEM', ratio, base, 0.95))
                    lines.append(quality_record(asset, 'DECIMATE', ratio, 1.1 * base, 0.94))
                    lines.append(quality_record(asset, 'PERCEPTUAL', ratio, 0.9 * base, 0.96))
            lines.append(dict(record_type='error', scope='condition', asset_id='synthetic_0', experiment='sweep',
                              condition='LAMBDA_2', error='RuntimeError: synthetic'))
            text = '\n'.join(json.dumps(line) for line in lines) + '\n{"record_type": "quality", "asse'
            (folder / 'raw.jsonl').write_text(text, encoding='utf-8')
            out = folder / 'analysis'
            code = analysis.main([str(folder), '--out', str(out), '--no-figures', '--bootstrap', '500'])
            self.assertEqual(code, 0)
            report = (out / 'report.md').read_text(encoding='utf-8')
            self.assertIn('## 3. PERCEPTUAL vs baseline', report)
            self.assertIn('Baris rusak/tidak dikenal: 1', report)
            self.assertIn('ERROR synthetic_0', report)
            with open(out / 'tables' / 'paired_main.csv', encoding='utf-8') as handle:
                rows = list(__import__('csv').DictReader(handle))
            chamfer = [r for r in rows if r['metric'] == 'chamfer_n' and r['baseline'] == 'QEM']
            self.assertEqual(len(chamfer), 2)
            for row in chamfer:
                self.assertEqual((row['n'], row['wins'], row['losses'], row['rank_biserial']), ('6', '6', '0', '1.0'))
                self.assertAlmostEqual(float(row['p']), 2 / 64)
                self.assertGreaterEqual(float(row['p_holm']), float(row['p']))
            # A pre-registered family: only chamfer rows (2 ratios x 2 baselines) are Holm-adjusted.
            code = analysis.main([str(folder), '--out', str(out), '--no-figures', '--bootstrap', '0',
                                  '--holm-metrics', 'chamfer_n'])
            self.assertEqual(code, 0)
            with open(out / 'tables' / 'paired_main.csv', encoding='utf-8') as handle:
                rows = list(__import__('csv').DictReader(handle))
            for row in rows:
                if row['metric'] == 'chamfer_n':
                    self.assertEqual(row['family'], 'holm')
                    self.assertAlmostEqual(float(row['p_holm']), 4 * 2 / 64)
                else:
                    self.assertEqual((row['family'], row['p_holm']), ('exploratory', ''))
            with self.assertRaises(SystemExit):
                analysis.parse_args([str(folder), '--holm-metrics', 'no_such_metric'])

    def test_merging_directories_with_different_settings_is_flagged(self):
        with tempfile.TemporaryDirectory() as folder:
            folders = [Path(folder) / 'a', Path(folder) / 'b']
            for index, path in enumerate(folders):
                path.mkdir()
                (path / 'raw.jsonl').write_text(json.dumps(quality_record('x', 'QEM', 0.2, 1e-3, 0.9)) + '\n',
                                                encoding='utf-8')
                config = dict(seed=1, samples=100, preserve_seams=bool(index))
                (path / 'environment.json').write_text(json.dumps({'config': config}), encoding='utf-8')
            results = analysis.load_results(folders)
            self.assertEqual(results.duplicates, 1)
            self.assertEqual(sum("'preserve_seams' differs" in problem for problem in results.problems), 1)
            self.assertFalse(any("'seed' differs" in problem for problem in results.problems))

    def test_missing_directory_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            results = analysis.load_results([Path(folder) / 'nope'])
            self.assertEqual(len(results.quality), 0)
            self.assertTrue(results.problems)
            self.assertEqual(analysis.main([str(Path(folder) / 'nope'), '--no-figures']), 1)

    def test_new_metric_keys_are_read_and_compared(self):
        results = analysis.Results()
        results.add_line(json.dumps(quality_record('a', 'QEM', 0.2, 1e-3, 0.9, roi=2e-3)), 'x')
        row = results.quality[('a', 'main', 'QEM', 0.2)]
        self.assertAlmostEqual(row['hausdorff_v_n'], 2.5e-3)
        self.assertAlmostEqual(row['roi_chamfer_sym_n'], 2.5e-3)
        self.assertAlmostEqual(row['roi_hausdorff_sym_n'], 8e-3)
        self.assertAlmostEqual(row['roi_chamfer_c2r_n'], 3e-3)
        self.assertAlmostEqual(row['roi_hausdorff_c2r_n'], 8e-3)
        for key in ('hausdorff_v_n', 'roi_chamfer_sym_n', 'roi_hausdorff_sym_n'):
            self.assertEqual(analysis.METRIC[key].better, 'lower')
        results = analysis.Results()
        results.add_line(json.dumps(quality_record('b', 'QEM', 0.2, 1e-3, 0.9)), 'old record')
        self.assertIsNone(results.quality[('b', 'main', 'QEM', 0.2)]['roi_chamfer_sym_n'])  # absent -> None


class HolmFamilyTests(unittest.TestCase):
    """--holm-baselines/--holm-ratios and the per-metric reachability guardrail (review finding 26)."""

    def run_analysis(self, folder, *extra):
        out = folder / 'analysis'
        self.assertEqual(analysis.main([str(folder), '--out', str(out), '--no-figures', '--bootstrap', '200',
                                        *extra]), 0)
        return read_csv(out / 'tables' / 'paired_main.csv'), (out / 'report.md').read_text(encoding='utf-8')

    def test_default_family_blocks_roi_and_declared_family_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_raw(folder, evaluation_split())
            rows, report = self.run_analysis(folder, '--holm-metrics', 'iou_multiview,roi_chamfer_sym_n')
            family = [r for r in rows if r['family'] == 'holm']
            self.assertEqual(len(family), 12)  # 2 metrics x 2 baselines x 3 ratios
            for row in family:
                expected_n = '9' if row['metric'] == 'iou_multiview' else '7'
                self.assertEqual(row['n'], expected_n)
                if row['metric'] == 'roi_chamfer_sym_n':  # 7/7 wins, p = 2/128, best Holm p = 6 * 2/128
                    self.assertAlmostEqual(float(row['p']), 2 / 128)
                    self.assertAlmostEqual(float(row['p_holm']), 6 * 2 / 128)
            self.assertIn('roi_chamfer_sym_n tidak dapat signifikan setelah Holm apa pun datanya', report)
            self.assertIn('roi_chamfer_sym_n (n = 7, p min 0.0156) ≥ 0.0938', report)
            self.assertNotIn('Signifikansi setelah Holm secara prinsip mungkin untuk setiap', report)

            rows, report = self.run_analysis(folder, '--holm-metrics', 'iou_multiview,roi_chamfer_sym_n',
                                             '--holm-baselines', 'QEM', '--holm-ratios', '0.05')
            family = {(r['metric'], r['baseline'], r['ratio']): r for r in rows if r['family'] == 'holm'}
            self.assertEqual(set(family), {('iou_multiview', 'QEM', '0.05'), ('roi_chamfer_sym_n', 'QEM', '0.05')})
            self.assertAlmostEqual(float(family[('iou_multiview', 'QEM', '0.05')]['p_holm']), 2 * 2 / 512)
            self.assertAlmostEqual(float(family[('roi_chamfer_sym_n', 'QEM', '0.05')]['p_holm']), 2 / 128)
            self.assertTrue(all(r['p_holm'] == '' for r in rows if r['family'] == 'exploratory'))
            self.assertIn('Signifikansi setelah Holm secara prinsip mungkin untuk setiap metrik famili', report)
            self.assertIn('baseline QEM; rasio 5%', report)

            _, report = self.run_analysis(folder, '--holm-metrics', 'iou_multiview', '--holm-ratios', '0.3')
            self.assertIn('anggota famili Holm yang dideklarasikan tanpa data/p: iou_multiview/QEM/30%', report)

    def test_reachability_bound(self):
        rows = [dict(metric='iou_multiview', n=9, p_holm=1.0), dict(metric='roi_chamfer_n', n=7, p_holm=1.0),
                dict(metric='chamfer_n', n=9, p_holm=None)]
        reach = analysis.holm_reachability(rows)
        self.assertEqual(set(reach), {'iou_multiview', 'roi_chamfer_n'})
        self.assertAlmostEqual(reach['iou_multiview']['best'], 2 * 2 / 512)
        self.assertAlmostEqual(reach['roi_chamfer_n']['best'], 2 / 128)
        self.assertEqual(analysis.smallest_exact_p(9), 2 / 512)
        self.assertIsNone(analysis.smallest_exact_p(0))

    def test_option_validation(self):
        with self.assertRaises(SystemExit):
            analysis.parse_args(['x', '--baselines', 'QEM', '--holm-baselines', 'DECIMATE'])
        for bad in ('1.5', '0', 'abc', ','):
            with self.subTest(ratios=bad), self.assertRaises(SystemExit):
                analysis.parse_args(['x', '--holm-ratios', bad])
        args = analysis.parse_args(['x', '--holm-ratios', '0.05,0.2'])
        self.assertEqual((args.holm_ratios, args.holm_baselines), ([0.05, 0.2], ['QEM', 'DECIMATE']))
        self.assertIsNone(analysis.parse_args(['x']).holm_ratios)


class ReportTests(unittest.TestCase):
    """Error resolution (29), exploratory held-out sweep (30), `better` column (31), mixed sessions (9)."""

    def test_errors_count_as_resolved_only_with_their_replacement_data(self):
        records, ratios = [], (0.5, 0.2)
        for index in range(5):
            asset = f'e{index}'
            for ratio in ratios:
                records += [quality_record(asset, c, ratio, 1e-3 * (1 + index), 0.9) for c in ('QEM', 'DECIMATE',
                                                                                             'PERCEPTUAL')]
        def timing(asset, rep, warmup=False):
            return [dict(record_type='timing', asset_id=asset, split='evaluation', category='organic',
                         experiment='timing', condition=m, method=m, rep=rep, warmup=warmup, ratio=r, level=1,
                         source_triangles=1000, timings=dict(processing_seconds=1.0, method_set_seconds=2.0))
                    for m in ('DECIMATE', 'QEM', 'PERCEPTUAL') for r in ratios]
        error = lambda asset, scope, **extra: dict(dict(record_type='error', asset_id=asset, scope=scope,  # noqa
                                                        experiment=None, condition=None,
                                                        error='RuntimeError: synthetic'), **extra)
        records += timing('e0', 0, True) + timing('e0', 1) + timing('e1', 0, True) + timing('e1', 1)
        records += timing('e2', 0, True) + timing('e2', 1) + timing('e3', 0, True) + timing('e3', 1)
        records.append(error('e4', 'timing', experiment='timing'))  # no timing rows at all -> unresolved
        records.append(error('e3', 'save_lods'))
        records.append(dict(record_type='asset', asset_id='e3', lods_file=None))  # still no LOD file -> unresolved
        records.append(error('e1', 'save_lods'))
        records.append(dict(record_type='asset', asset_id='e1', lods_file='lods/e1.blend'))  # later file -> resolved
        records.append(error('e2', 'condition', experiment='main', condition='PERCEPTUAL'))  # rows exist -> resolved
        records.append(error('e2', 'condition', experiment='sweep', condition='LAMBDA_2'))  # never ran -> unresolved
        records.append(error('e0', 'determinism'))
        records.append(dict(record_type='asset', asset_id='e0', determinism=dict(identical=True)))  # resolved
        environment = dict(config=dict(ratios=list(ratios), reps=1, warmup=1))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_raw(folder, records, environment)
            results = analysis.load_results([folder])
            verdict = {(e['asset_id'], e['scope'], e.get('condition')): analysis.error_resolved(results, e)
                       for e in results.errors}
            self.assertEqual(verdict, {('e4', 'timing', None): False, ('e3', 'save_lods', None): False,
                                       ('e1', 'save_lods', None): True, ('e2', 'condition', 'PERCEPTUAL'): True,
                                       ('e2', 'condition', 'LAMBDA_2'): False, ('e0', 'determinism', None): True})
            out = folder / 'analysis'
            self.assertEqual(analysis.main([str(folder), '--out', str(out), '--no-figures', '--bootstrap', '0']), 0)
            report = (out / 'report.md').read_text(encoding='utf-8')
            self.assertIn('6 error (3 tanpa hasil pengganti)', report)
            self.assertIn('ERROR e4 timing timing/None: RuntimeError: synthetic (TANPA hasil pengganti)', report)
            time_rows = [r for r in read_csv(out / 'tables' / 'paired_main.csv') if r['metric'] == 'time_s']
            self.assertTrue(time_rows and all(r['n'] == '4' for r in time_rows))  # e4 has no timing

        # An incomplete timing set (rep 1 of PERCEPTUAL missing) does not resolve a timing error either.
        partial = analysis.Results()
        for record in [r for r in timing('e5', 0, True) + timing('e5', 1) if not (r['condition'] == 'PERCEPTUAL'
                                                                                  and r['rep'] == 1)]:
            partial.add_line(json.dumps(record), 'x')
        partial.add_line(json.dumps(error('e5', 'timing')), 'x')
        partial.configs[0] = environment['config']
        self.assertFalse(analysis.error_resolved(partial, partial.errors[0]))

    def test_evaluation_sweep_is_exploratory_without_pareto_flags(self):
        records = []
        for split, prefix in (('tuning', 't'), ('evaluation', 'e')):
            for index in range(3):
                asset = f'{prefix}{index}'
                for ratio in (0.2, 0.05):
                    base = (1 + index) * 1e-3
                    records.append(quality_record(asset, 'QEM', ratio, base, 0.9, split=split, roi=base))
                    records.append(quality_record(asset, 'PERCEPTUAL', ratio, 0.99 * base, 0.901, split=split,
                                                  roi=base))
                    records.append(quality_record(asset, 'DECIMATE', ratio, 1.1 * base, 0.89, split=split, roi=base))
                    for strength, factor in ((0.0, 1.0), (2.0, 0.98), (8.0, 1.03)):
                        records.append(quality_record(asset, f'LAMBDA_{strength:g}', ratio, factor * base,
                                                      0.9 + 0.001 * strength, split=split, experiment='sweep',
                                                      roi=factor * base, strength=strength))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_raw(folder, records)
            results = analysis.load_results([folder])
            tuning = [r for r in analysis.sweep_table(results, 'tuning') if r['is_sweep']]
            held_out = [r for r in analysis.sweep_table(results, 'evaluation') if r['is_sweep']]
            self.assertTrue(tuning and held_out)
            self.assertTrue(all(isinstance(r['pareto_iou_hausdorff'], bool) for r in tuning))
            self.assertTrue(any(r['pareto_iou_hausdorff'] for r in tuning))
            self.assertTrue(all(r['pareto_iou_hausdorff'] is None and r['pareto_roi_global_chamfer'] is None
                                for r in held_out))
            out = folder / 'analysis'
            self.assertEqual(analysis.main([str(folder), '--out', str(out), '--no-figures', '--bootstrap', '0']), 0)
            report = (out / 'report.md').read_text(encoding='utf-8')
            self.assertIn('Sapuan λ dan ablation pada split `evaluation` bersifat eksploratori', report)
            self.assertIn('Split `evaluation` (eksploratori; bukan dasar memilih λ', report)

    def test_report_table_keeps_the_direction_column(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_raw(folder, evaluation_split(ratios=(0.2,)))
            out = folder / 'analysis'
            self.assertEqual(analysis.main([str(folder), '--out', str(out), '--no-figures', '--bootstrap', '0']), 0)
            report = (out / 'report.md').read_text(encoding='utf-8')
            section = report.split('## 3.')[1].split('## 4.')[0]
            header = next(line for line in section.splitlines() if line.startswith('| ratio'))
            self.assertIn('| better |', header)
            self.assertIn('| HL Δ | CI low | CI high | CI level |', header)
            self.assertIn('| 20% | chamfer_n | lower | QEM | 9 |', section)
            self.assertIn('| 20% | iou_multiview | higher | QEM | 9 |', section)

    def test_sessions_with_different_code_are_flagged(self):
        def session(sid, qem_hash, build='abc'):
            return dict(session_id=sid, status='completed', blender_version='5.1.2', blender_build_hash=build,
                        extension=dict(manifest_version='0.2.0'),
                        source_sha256={'smart_lod/qem.py': qem_hash, 'scripts/run_experiment.py': 'h'})
        records = [dict(quality_record('a', 'QEM', 0.2, 1e-3, 0.9), session_id='s1'),
                   dict(quality_record('b', 'QEM', 0.2, 1e-3, 0.9), session_id='s2'),
                   dict(quality_record('c', 'QEM', 0.2, 1e-3, 0.9), session_id='s2')]
        with tempfile.TemporaryDirectory() as folder:
            same, mixed = Path(folder) / 'same', Path(folder) / 'mixed'
            write_raw(same, records, dict(config={}, sessions=[session('s1', 'q1'), session('s2', 'q1')]))
            write_raw(mixed, records, dict(config={}, sessions=[session('s1', 'q1'), session('s2', 'q2', 'def')]))
            self.assertIsNone(analysis.load_results([same]).mixed_code)
            results = analysis.load_results([mixed])
            self.assertIn('(smart_lod/qem.py, blender_build_hash)', results.mixed_code)
            self.assertIn('mixed/s1', results.mixed_code)
            self.assertEqual(sum('differ between recorded sessions' in p for p in results.problems), 1)
            code = {sid: analysis.code_fingerprint(s)[1] for sid, s in (('s1', session('s1', 'q1')),
                                                                        ('s2', session('s2', 'q2', 'def')))}
            self.assertIn(f'{code["s1"]} 1, {code["s2"]} 2'
                          if code['s1'] < code['s2'] else f'{code["s2"]} 2, {code["s1"]} 1', results.mixed_code)
            out = mixed / 'analysis'
            self.assertEqual(analysis.main([str(mixed), '--out', str(out), '--no-figures', '--paired-split', 'all']), 0)
            report = (out / 'report.md').read_text(encoding='utf-8')
            self.assertIn('PERHATIAN: sesi run yang digabung memakai kode/Blender/extension berbeda', report)
            # Two directories whose single sessions ran different code are flagged the same way.
            self.assertIsNotNone(analysis.load_results([same, mixed]).mixed_code)

    def test_timing_components_are_summarized(self):
        records = []
        for rep, (constraints, readback) in enumerate(((0.3, 0.02), (0.1, 0.04), (0.2, 0.03))):
            for method in ('DECIMATE', 'QEM', 'PERCEPTUAL'):
                timings = dict(processing_seconds=1.0 + rep, method_set_seconds=2.0 + rep)
                timings.update(dict(method_set_readback_seconds=readback) if method == 'DECIMATE' else
                               dict(constraints_seconds=constraints, input_arrays_seconds=0.05))
                records.append(dict(record_type='timing', asset_id='a', split='evaluation', category='organic',
                                    condition=method, method=method, rep=rep, warmup=False, ratio=0.2,
                                    source_triangles=1000, timings=timings))
        results = analysis.Results()
        for record in records:
            results.add_line(json.dumps(record), 'x')
        level_times, set_times, _ = analysis.timing_summary(results)
        rows = {r['method']: r for r in analysis.timing_table(results, set_times, level_times)}
        self.assertAlmostEqual(rows['QEM']['constraints_seconds_median'], 0.2)
        self.assertAlmostEqual(rows['PERCEPTUAL']['input_arrays_seconds_median'], 0.05)
        self.assertIsNone(rows['QEM']['readback_set_seconds_median'])
        self.assertAlmostEqual(rows['DECIMATE']['readback_set_seconds_median'], 0.03)
        self.assertIsNone(rows['DECIMATE']['constraints_seconds_median'])
        self.assertAlmostEqual(rows['DECIMATE']['set_median_s'], 3.0)


def find_blender():
    for candidate in (os.environ.get('BLENDER'), r'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe',
                      shutil.which('blender')):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


HARNESS_CHECK = r'''
import hashlib, importlib.util, json, sys, traceback
from pathlib import Path
import bpy

root, work = (Path(p) for p in sys.argv[sys.argv.index('--') + 1:][:2])
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location('run_experiment', root / 'scripts' / 'run_experiment.py')
harness = importlib.util.module_from_spec(spec)
sys.modules['run_experiment'] = harness
spec.loader.exec_module(harness)
from smart_lod import geometry

check = {}
try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0)
    toy = bpy.context.active_object
    toy.name = 'Toy'
    roi = [v.index for v in toy.data.vertices if v.co.z > 0.5]
    toy.vertex_groups.new(name='LOD_EvalROI').add(roi, 1.0, 'REPLACE')
    mesh, vertices, faces = geometry.snapshot(toy, bpy.context.evaluated_depsgraph_get())
    bpy.data.meshes.remove(mesh)
    bpy.ops.wm.save_as_mainfile(filepath=str(work / 'toy.blend'))
    manifest = dict(assets=dict(toy=dict(
        file='toy.blend', object='Toy', split='evaluation', category='organic', triangles=len(faces),
        geometry_sha256=hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest(),
        roi=dict(vertices=len(roi)))))
    (work / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    common = ['--manifest', str(work / 'manifest.json'), '--split', 'evaluation', '--ratios', '0.5,0.2', '--reps', '1',
              '--warmup', '0', '--samples', '200', '--resolution', '32', '--strength', '2']
    first = common + ['--out', str(work / 'run'), '--experiments', 'main,timing']

    def attempt(argv):
        try:
            return dict(code=harness.main(argv))
        except SystemExit as exc:
            return dict(refused=str(exc))

    for name, extra in (('strength_above_limit', ['--experiments', 'main', '--strength', '2e6']),
                        ('lambda_above_limit', ['--experiments', 'sweep', '--lambdas', '0,2e6'])):
        check[name] = attempt(common + ['--out', str(work / name)] + extra)
        check[name + '_out_created'] = (work / name).exists()
    check['first'] = attempt(first)
    check['resume_same_code'] = attempt(first + ['--resume'])
    env_path = work / 'run' / 'environment.json'
    env = json.loads(env_path.read_text(encoding='utf-8'))
    env['sessions'][0]['source_sha256']['smart_lod/qem.py'] = '0' * 64
    env['sessions'][1]['blender_build_hash'] = 'deadbeef'
    env_path.write_text(json.dumps(env), encoding='utf-8')
    check['resume_changed_code'] = attempt(first + ['--resume'])
    check['sessions_after_refusal'] = len(json.loads(env_path.read_text(encoding='utf-8'))['sessions'])
    check['resume_allowed'] = attempt(first + ['--resume', '--allow-code-change'])
    # LODs saved without the main experiment: Decimate is computed after the reference heatmap exists.
    check['lods_run'] = attempt(common + ['--out', str(work / 'lods_run'), '--experiments', 'timing', '--save-lods'])
    bpy.ops.wm.open_mainfile(filepath=str(work / 'lods_run' / 'lods' / 'toy.blend'))
    check['layers'] = {obj.name: dict(colors=sorted(a.name for a in obj.data.color_attributes),
                                      groups=sorted(g.name for g in obj.vertex_groups))
                       for obj in bpy.data.objects}
except Exception:
    check['exception'] = traceback.format_exc()
(work / 'check.json').write_text(json.dumps(check), encoding='utf-8')
'''


@unittest.skipIf(find_blender() is None, 'Blender not found (set BLENDER)')
class HarnessInBlenderTests(unittest.TestCase):
    """run_experiment.py on a toy asset: resume code check (9), LOD layers (27), timing components (28),
    strength limit (12)."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.work = Path(cls.folder.name)
        (cls.work / 'harness_check.py').write_text(HARNESS_CHECK, encoding='utf-8')
        completed = subprocess.run(
            [find_blender(), '--background', '--factory-startup', '--python-exit-code', '1', '--python',
             str(cls.work / 'harness_check.py'), '--', str(ROOT), str(cls.work)],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
        cls.output = completed.stdout[-4000:] + completed.stderr[-4000:]
        path = cls.work / 'check.json'
        cls.check = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def setUp(self):
        self.assertNotIn('exception', self.check, self.check.get('exception'))
        self.assertTrue(self.check, self.output)

    def records(self, name):
        with open(self.work / name / 'raw.jsonl', encoding='utf-8') as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_resume_refuses_changed_code_unless_allowed(self):
        self.assertEqual(self.check['first'], dict(code=0), self.output)
        self.assertEqual(self.check['resume_same_code'], dict(code=0))
        refused = self.check['resume_changed_code'].get('refused', '')
        self.assertIn('smart_lod/qem.py', refused)
        self.assertIn("blender_build_hash 'deadbeef'", refused)
        self.assertEqual(self.check['sessions_after_refusal'], 2)  # a refused resume writes nothing
        self.assertEqual(self.check['resume_allowed'], dict(code=0))
        env = json.loads((self.work / 'run' / 'environment.json').read_text(encoding='utf-8'))
        self.assertEqual(len(env['sessions']), 3)
        self.assertTrue(env['sessions'][-1]['allow_code_change'])
        self.assertTrue(any('smart_lod/qem.py' in change for change in env['sessions'][-1]['code_changes']))
        sessions = {s['session_id'] for s in env['sessions']}
        records = self.records('run')
        self.assertTrue(records and all(r.get('session_id') in sessions for r in records))
        results = analysis.load_results([self.work / 'run'])
        self.assertIn('smart_lod/qem.py', results.mixed_code or '')

    def test_saved_lods_keep_the_heatmap_on_perceptual_only(self):
        self.assertEqual(self.check['lods_run'], dict(code=0), self.output)
        layers = self.check['layers']
        self.assertEqual(sum(name.startswith('DECIMATE__') for name in layers), 2)
        for name, found in layers.items():
            with self.subTest(name=name):
                heat = 'LOD_Heatmap' in found['colors'] and 'LOD_Importance' in found['groups']
                self.assertEqual(heat, name.startswith(('REF__', 'PERCEPTUAL__')))
                self.assertNotIn('LOD_Heatmap', found['colors'] if not heat else [])
                self.assertNotIn('LOD_Importance', found['groups'] if not heat else [])
                self.assertEqual('LOD_EvalROI' in found['groups'], name.startswith('REF__'))

    def test_strength_beyond_the_engine_limit_is_refused_before_any_work(self):
        # qem.simplify_levels raises above qem.MAX_STRENGTH (12); the harness used to accept the value, run
        # every other condition and then log a PERCEPTUAL error per asset.
        for name in ('strength_above_limit', 'lambda_above_limit'):
            with self.subTest(name=name):
                self.assertIn('[0, 1e+06]', self.check[name].get('refused', ''), self.check[name])
                self.assertFalse(self.check[name + '_out_created'])

    def test_timing_components_are_like_for_like(self):
        timing = [r for r in self.records('run') if r['record_type'] == 'timing']
        self.assertEqual(len(timing), 3 * 2)
        for record in timing:
            t = record['timings']
            with self.subTest(method=record['method'], ratio=record['ratio']):
                if record['method'] == 'DECIMATE':
                    self.assertGreater(t['readback_seconds'], 0)
                    self.assertEqual(t['processing_seconds'], t['simplification_and_mesh_seconds'])
                    self.assertNotIn('constraints_seconds', t)
                    same_rep = [r['timings'] for r in timing if r['method'] == 'DECIMATE']
                    self.assertAlmostEqual(t['method_set_seconds'], sum(x['processing_seconds'] for x in same_rep))
                    self.assertAlmostEqual(t['method_set_readback_seconds'],
                                           sum(x['readback_seconds'] for x in same_rep))
                else:
                    self.assertGreater(t['constraints_seconds'], 0)
                    self.assertGreater(t['input_arrays_seconds'], 0)
                    self.assertAlmostEqual(t['processing_seconds'], t['input_arrays_seconds'] + t['constraints_seconds']
                                           + t['importance_seconds'] + t['level_capture_seconds'] + t['mesh_seconds'])
                    self.assertNotIn('readback_seconds', t)
                self.assertTrue(record['matches_quality_run'])


if __name__ == '__main__':
    suite = unittest.TestSuite()
    for case in (RankAndWilcoxonTests, AdjustmentAndEffectTests, BootstrapTests, ShiftIntervalTests, LoaderTests,
                 HolmFamilyTests, ReportTests, HarnessInBlenderTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        sys.exit(1)
