"""User-study tooling tests; plain CPython, synthetic in-memory data only.

    python tests/test_user_study.py

Covers scripts/analyze_user_study.py (validation, build ids, superseding of
repeated downloads, join with stimuli.csv and the method key, aggregation,
clustered bootstrap, Friedman/Wilcoxon fallbacks against SciPy when it is
installed, agreement scopes) and the Blender-free helpers of
scripts/make_user_study.py (blinding codes, ids, build fingerprint, ratio
properties, triangle-budget check, view sets, camera framing, PNG encoding,
form embedding).  Nothing is written to disk.
"""
import csv
import io
import json
import math
from pathlib import Path
import random
import struct
import sys
import unittest
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import analyze_user_study as analysis  # noqa: E402
import make_user_study as maker  # noqa: E402  (bpy is imported lazily)

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency
    scipy_stats = None

STUDY = 's1'
RATIOS = {1: 0.5, 2: 0.2, 3: 0.05}
SCORE = {'PERCEPTUAL': 5, 'QEM': 4, 'DECIMATE': 3, 'REFERENCE': 5}
STAMP = ('1', '2026-09-21T10:00:00.000Z')  # export_sequence, exported_at as the form writes them
# Columns of a rating row (docs/USER_STUDY.csv).
STUDY_COL, BUILD_COL, PID_COL, STIMULUS_COL, ORDER_COL = range(5)
RATING_COLS, NOTES_COL, SEQUENCE_COL, TIME_COL = range(5, 9), 9, 10, 11


def synthetic_study(assets=('a1', 'a2', 'a3'), levels=(2, 3), views=('front',), control=False, seed=7):
    """In-memory key and stimulus list shaped like make_user_study.py output."""
    rng = random.Random(seed)
    codes = maker.assign_method_codes(list(maker.METHODS) + (['REFERENCE'] if control else []), rng)
    plan = [(a, v, lv, m) for a in assets for v in views for lv in levels for m in maker.METHODS]
    plan += [(a, v, 0, 'REFERENCE') for a in assets for v in views] if control else []
    ids = maker.make_stimulus_ids(len(plan), rng)
    stimuli = [dict(stimulus_id=i, asset_id=a, lod_level=lv, target_ratio=maker.format_ratio(RATIOS.get(lv, 1.0)),
                    actual_triangles=4000 if lv == 0 else int(4000 * RATIOS[lv]) - 1, view_set_id=v,
                    method_code=codes[m], image=f'stimuli/{i}.png', method=m)
               for i, (a, v, lv, m) in zip(ids, plan)]
    build = maker.build_fingerprint(STUDY, seed, codes, stimuli)
    for s in stimuli:
        s['build_id'] = build
    key_text = ('# PRIVATE - never give to participants\n'
                + maker.csv_text(('study_id', 'build_id', 'method_code', 'method'),
                                 [dict(study_id=STUDY, build_id=build, method_code=c, method=m)
                                  for m, c in codes.items()]))
    return key_text, maker.csv_text(maker.PUBLIC_COLUMNS, stimuli), stimuli, codes


def rating_row(pid, stimulus, position, ratings, notes='', study=STUDY, stamp=STAMP):
    return [study, stimulus['build_id'], pid, stimulus['stimulus_id'], position, *ratings, notes, *stamp]


def ratings_text(rows, header=analysis.CSV_HEADER):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def participant_rows(pid, stimuli, score=lambda pid, s: (SCORE[s['method']],) * 4, stamp=STAMP):
    """All stimuli in the order the rating form shows them to ``pid``."""
    by_id = {s['stimulus_id']: s for s in stimuli}
    order = analysis.form_order(STUDY, pid, list(by_id))
    return [rating_row(pid, by_id[i], position, score(pid, by_id[i]), stamp=stamp)
            for position, i in enumerate(order, 1)]


def participant_file(pid, stimuli, score=lambda pid, s: (SCORE[s['method']],) * 4, stamp=STAMP):
    return ratings_text(participant_rows(pid, stimuli, score, stamp))


def codes_of(issues):
    return sorted({i['code'] for i in issues})


def decode_png(data):
    """Minimal decoder for the 8-bit RGB PNGs of maker.encode_png (filters 0 and 1)."""
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    position, chunks = 8, {}
    while position < len(data):
        length, tag = struct.unpack('>I4s', data[position:position + 8])
        body = data[position + 8:position + 8 + length]
        crc, = struct.unpack('>I', data[position + 8 + length:position + 12 + length])
        assert zlib.crc32(tag + body) & 0xFFFFFFFF == crc
        chunks[tag] = chunks.get(tag, b'') + body
        position += 12 + length
    width, height, depth, colour = struct.unpack('>IIBB', chunks[b'IHDR'][:10])
    assert (depth, colour) == (8, 2)
    raw = np.frombuffer(zlib.decompress(chunks[b'IDAT']), dtype=np.uint8).reshape(height, width * 3 + 1)
    rows = raw[:, 1:].astype(np.int64)
    for y in range(height):
        if raw[y, 0] == 1:
            for x in range(3, width * 3):
                rows[y, x] = (rows[y, x] + rows[y, x - 3]) & 0xFF
        else:
            assert raw[y, 0] == 0
    return rows.astype(np.uint8).reshape(height, width, 3), chunks


class ContractTests(unittest.TestCase):
    def test_headers_agree_everywhere(self):
        with open(ROOT / 'docs' / 'USER_STUDY.csv', newline='', encoding='utf-8-sig') as handle:
            docs = tuple(next(csv.reader(handle)))
        self.assertEqual(analysis.CSV_HEADER, docs)
        self.assertEqual(maker.CSV_HEADER, docs)
        template = (ROOT / 'study' / 'rating_form_template.html').read_text(encoding='utf-8')
        self.assertEqual(template.count(','.join(docs)), 1)
        self.assertEqual(template.count(maker.TEMPLATE_TOKEN), 1)
        self.assertNotIn('http://', template.replace('http://www.w3.org', ''))
        self.assertNotRegex(template, r'<(script|link)[^>]+(src|href)=["\']https?:')

    def test_template_embedding_is_script_safe_and_round_trips(self):
        template = (ROOT / 'study' / 'rating_form_template.html').read_text(encoding='utf-8')
        _, _, stimuli, _ = synthetic_study()
        stimuli[0]['image'] = 'stimuli/</script><script>alert(1)</script>&\u2028.png'
        data = maker.form_data(STUDY, stimuli[0]['build_id'], stimuli, '2026-09-21T00:00:00+00:00')
        html = maker.fill_template(template, data)
        start = html.index('const STUDY = ') + len('const STUDY = ')
        payload = html[start:html.index(';</script>', start)]
        self.assertNotIn('<', payload)
        self.assertNotIn('\u2028', payload)
        self.assertEqual(json.loads(payload), data)
        self.assertTrue(all(isinstance(v, str) for s in data['stimuli'] for v in s.values()))
        with self.assertRaises(ValueError):
            maker.fill_template(template.replace(maker.TEMPLATE_TOKEN, 'null'), data)
        with self.assertRaises(ValueError):
            maker.fill_template(template.replace(','.join(maker.CSV_HEADER), 'x'), data)

    def test_form_and_export_reveal_no_method_lod_or_triangles(self):
        # Finding 11: the form embedded and exported method code, LOD, target ratio and triangle count.
        _, _, stimuli, _ = synthetic_study()
        data = maker.form_data(STUDY, stimuli[0]['build_id'], stimuli, '2026-09-21T00:00:00+00:00')
        self.assertEqual({key for s in data['stimuli'] for key in s}, {'id', 'image'})
        self.assertEqual(data['build_id'], stimuli[0]['build_id'])
        hidden = ('method', 'asset_id', 'lod_level', 'target_ratio', 'actual_triangles', 'view_set_id')
        for column in hidden:
            self.assertFalse(any(column in c for c in maker.CSV_HEADER), column)
        template = (ROOT / 'study' / 'rating_form_template.html').read_text(encoding='utf-8')
        for word in hidden[1:] + ('method_code', 'csv-text'):  # no metadata fields, no CSV text preview
            self.assertNotIn(word, template)


class MakerHelperTests(unittest.TestCase):
    def test_view_sets(self):
        self.assertEqual(maker.parse_view_sets('front,three_quarter'), [('front',), ('three_quarter',)])
        self.assertEqual(maker.parse_view_sets('front+top, side'), [('front', 'top'), ('side',)])
        for bad in ('nope', 'front,front', 'front+front', ',', ''):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                maker.parse_view_sets(bad)

    def test_method_codes_are_a_seeded_random_bijection(self):
        methods = list(maker.METHODS)
        first = maker.assign_method_codes(methods, random.Random(3))
        self.assertEqual(first, maker.assign_method_codes(methods, random.Random(3)))
        self.assertEqual(sorted(first.values()), ['M1', 'M2', 'M3'])
        seen = {m: set() for m in methods}
        for seed in range(200):
            for method, code in maker.assign_method_codes(methods, random.Random(seed)).items():
                seen[method].add(code)
        self.assertTrue(all(codes == {'M1', 'M2', 'M3'} for codes in seen.values()))

    def test_stimulus_ids(self):
        ids = maker.make_stimulus_ids(500, random.Random(1))
        self.assertEqual(len(set(ids)), 500)
        self.assertTrue(all(len(i) == 6 and i[0] == 'S' and set(i[1:]) <= set(maker.ID_ALPHABET) for i in ids))
        self.assertEqual(ids, maker.make_stimulus_ids(500, random.Random(1)))

    def test_camera_fit_keeps_every_point_inside_and_centred(self):
        rng = np.random.default_rng(5)
        points = rng.normal(size=(4000, 3)) * (1.0, 0.3, 0.6) + (0.2, -0.1, 0.4)
        limit = 1.0 - maker.FRAME_MARGIN
        for name, (azimuth, elevation) in maker.VIEW_PRESETS.items():
            with self.subTest(view=name):
                fit = maker.fit_camera(points, maker.view_direction(azimuth, elevation))
                uv = maker.project(points, fit)
                self.assertLessEqual(np.abs(uv).max(), limit + 1e-9)
                span = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) / 2
                self.assertAlmostEqual(span, limit, delta=1e-6)  # tight on the limiting axis
                self.assertAlmostEqual(uv[:, 0].min() + uv[:, 0].max(), 0.0, delta=1e-9)
                self.assertAlmostEqual(uv[:, 1].min() + uv[:, 1].max(), 0.0, delta=1e-9)
                self.assertTrue(np.all(-(points - fit['location']) @ fit['back'] > fit['clip_start']))
        front = maker.fit_camera(points, maker.view_direction(0, 0))
        self.assertLess(front['location'][1], points[:, 1].min())  # front camera sits on the -Y side

    def test_png_encoding_round_trips(self):
        rgb = np.random.default_rng(2).integers(0, 256, size=(7, 11, 3), dtype=np.uint8)
        data = maker.encode_png(rgb)
        decoded, chunks = decode_png(data)
        np.testing.assert_array_equal(decoded, rgb)
        self.assertEqual(chunks[b'sRGB'], b'\x00')
        self.assertEqual(data, maker.encode_png(rgb))
        try:
            from PIL import Image
        except ImportError:
            return
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(data)).convert('RGB')), rgb)

    def test_compose_and_margins(self):
        size, gap = 10, 2
        left, right = np.full((size, size, 3), 50, np.uint8), np.full((size, size, 3), 200, np.uint8)
        labels = [np.full((3, size, 3), 7, np.uint8), np.full((3, size, 3), 9, np.uint8)]
        image = maker.compose([(left, right), (right, left)], labels, gap)
        self.assertEqual(image.shape, (3 + 2 * size + gap, 2 * size + gap, 3))
        self.assertTrue((image[:3, :size] == 7).all() and (image[:3, size + gap:] == 9).all())
        self.assertTrue((image[3:3 + size, :size] == 50).all() and (image[3:3 + size, size + gap:] == 200).all())
        self.assertTrue((image[3 + size + gap:, :size] == 200).all() and (image[:, size:size + gap] == 255).all())
        alpha = np.zeros((20, 30))
        alpha[5:9, 4:25] = 1.0
        self.assertEqual(maker.content_margins(alpha), (4, 5, 5, 11))
        self.assertIsNone(maker.content_margins(np.zeros((4, 4))))


class StudyBuildTests(unittest.TestCase):
    def test_build_id_changes_with_key_stimuli_and_seed(self):
        _, _, stimuli, codes = synthetic_study()
        build = maker.build_fingerprint(STUDY, 7, codes, stimuli)
        self.assertRegex(build, r'^[0-9a-f]{16}$')
        self.assertEqual(build, maker.build_fingerprint(STUDY, 7, dict(codes), stimuli[::-1]))  # order-free
        swapped = dict(codes, DECIMATE=codes['QEM'], QEM=codes['DECIMATE'])
        changed = [dict(stimuli[0], actual_triangles=stimuli[0]['actual_triangles'] + 1)] + stimuli[1:]
        for other in (maker.build_fingerprint(STUDY, 7, swapped, stimuli),
                      maker.build_fingerprint(STUDY, 8, codes, stimuli),
                      maker.build_fingerprint('s2', 7, codes, stimuli),
                      maker.build_fingerprint(STUDY, 7, codes, changed)):
            self.assertNotEqual(other, build)

    def test_ratio_comes_from_the_harness_property(self):
        # Finding 34: run_experiment.py writes smart_lod_ratio, which was never read.
        ratios = {1: 0.5, 2: 0.2, 3: 0.05}
        harness = {'smart_lod_level': 2, 'smart_lod_ratio': 0.05, 'smart_lod_target': 657}
        self.assertEqual(maker.target_ratio(harness, 'QEM__L2__car', 2, ratios)[:2],
                         (0.05, "object property 'smart_lod_ratio'"))
        self.assertIsNone(maker.target_ratio(harness, 'x', 2, ratios)[2])  # default --ratios: no warning
        self.assertIn('differs from --ratios', maker.target_ratio(harness, 'x', 2, ratios, ratios_given=True)[2])
        self.assertIsNone(maker.target_ratio(dict(harness, smart_lod_ratio=0.2), 'x', 2, ratios, True)[2])
        self.assertEqual(maker.target_ratio({}, 'x', 3, ratios), (0.05, '--ratios', None))
        for bad in (0.0, 1.5, 'abc'):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                maker.target_ratio({'smart_lod_ratio': bad}, 'x', 2, ratios)
        with self.assertRaises(ValueError):
            maker.target_ratio({}, 'x', 4, ratios)
        maker.check_labels(harness, 'QEM__L2__car', 'QEM', 2)
        maker.check_labels({}, 'QEM__L2__car', 'QEM', 2)
        with self.assertRaises(ValueError):
            maker.check_labels(harness, 'QEM__L1__car', 'QEM', 1)
        with self.assertRaises(ValueError):
            maker.check_labels({'smart_lod_method': 'DECIMATE'}, 'QEM__L2__car', 'QEM', 2)

    def test_budget_check_flags_incomparable_triangle_counts(self):
        # Finding 33: smoke spot LOD2 had DECIMATE 292 against QEM/PERCEPTUAL 564 (target 293).
        counts = {('DECIMATE', 1): 1170, ('QEM', 1): 1170, ('PERCEPTUAL', 1): 1170,
                  ('DECIMATE', 2): 292, ('QEM', 2): 564, ('PERCEPTUAL', 2): 564}
        targets = {key: (1171 if key[1] == 1 else 293) for key in counts}
        report = maker.budget_problems(counts, targets)
        self.assertEqual([entry['level'] for entry in report], [1, 2])
        self.assertEqual(report[0]['problems'], [])
        self.assertEqual(len(report[1]['problems']), 2)  # spread and target exceeded
        self.assertIn('PERCEPTUAL 564 > target 293', report[1]['problems'][1])
        self.assertAlmostEqual(report[1]['spread'], (564 - 292) / 292, places=6)
        # One edge collapse (two triangles) or up to the tolerance is comparable; targets are optional.
        near = {('DECIMATE', 2): 292, ('QEM', 2): 293, ('PERCEPTUAL', 2): 294}
        self.assertEqual(maker.budget_problems(near, {k: None for k in near})[0]['problems'], [])
        big = {('DECIMATE', 2): 10000, ('QEM', 2): 10150, ('PERCEPTUAL', 2): 10199}
        self.assertEqual(maker.budget_problems(big, {k: 10200 for k in big})[0]['problems'], [])
        self.assertNotEqual(maker.budget_problems(big, {k: 10200 for k in big}, tolerance=0.01)[0]['problems'], [])
        self.assertNotEqual(maker.budget_problems(big, {k: 9800 for k in big})[0]['problems'], [])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.key_text, self.stimuli_text, self.stimuli, self.codes = synthetic_study()

    def run_rows(self, files, **options):
        return analysis.run(files, self.key_text, self.stimuli_text, n_boot=200, **options)

    def test_clean_study_has_no_errors_and_merges_the_key(self):
        state = self.run_rows([(f'P0{i}.csv', participant_file(f'P0{i}', self.stimuli)) for i in range(1, 4)])
        self.assertEqual(state['issues'], [])
        self.assertEqual(len(state['records']), 3 * len(self.stimuli))
        by_code = {c: m for m, c in self.codes.items()}
        by_id = {s['stimulus_id']: s for s in self.stimuli}
        for record in state['records']:
            self.assertEqual(record['method'], by_code[record['method_code']])
            self.assertEqual(record['overall_quality_1_5'], SCORE[record['method']])
            stimulus = by_id[record['stimulus_id']]  # metadata comes from stimuli.csv, not from the rating row
            for column in ('asset_id', 'method_code', 'lod_level', 'actual_triangles', 'view_set_id'):
                self.assertEqual(record[column], stimulus[column])
            self.assertEqual(record['target_ratio'], float(stimulus['target_ratio']))

    def test_header_bom_and_quoted_notes(self):
        text = participant_file('P01', self.stimuli)
        rows = list(csv.reader(io.StringIO(text)))
        rows[1][NOTES_COL] = 'kiri, "telinga"\nhilang'
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator='\r\n').writerows(rows)
        decoded = analysis.decode(b'\xef\xbb\xbf' + buffer.getvalue().encode('utf-8'), 'P01.csv')
        state = self.run_rows([('P01.csv', decoded)])
        self.assertEqual(codes_of(state['issues']), [])
        self.assertIn('kiri, "telinga"\nhilang', [r['notes'] for r in state['records']])
        wrong = ratings_text([], header=analysis.CSV_HEADER[:-1] + ('comment',))
        swapped = ratings_text([], header=analysis.CSV_HEADER[1::-1] + analysis.CSV_HEADER[2:])
        state = self.run_rows([('bad.csv', wrong), ('swap.csv', swapped), ('P01.csv', text)])
        self.assertEqual([i['file'] for i in state['issues'] if i['code'] == 'header_mismatch'], ['bad.csv', 'swap.csv'])

    def test_row_level_problems_are_reported_not_guessed(self):
        s = self.stimuli
        rows = [rating_row('P01', x, i, (4, 4, 4, 4)) for i, x in enumerate(s, 1)]
        rows[0][STUDY_COL] = 'other_study'
        rows[1][PID_COL] = 'Budi'
        rows[2][STIMULUS_COL] = 'SZZZZZ'
        rows[3][BUILD_COL] = 'ffffffffffffffff'
        rows[4][RATING_COLS[0]], rows[4][RATING_COLS[1]] = '6', '3.5'
        rows[5][RATING_COLS[3]] = ''
        rows[7][SEQUENCE_COL] = 'x'  # unreadable stamp: kept, but it cannot supersede another copy
        rows.append(rows[6][:])  # identical duplicate: kept once
        state = self.run_rows([('P01.csv', ratings_text(rows))], incomplete='include')
        codes = codes_of(state['issues'])
        for code in ('wrong_study', 'invalid_participant', 'unknown_stimulus', 'wrong_build', 'invalid_export_stamp',
                     'rating_out_of_range', 'missing_rating', 'duplicate_identical', 'incomplete_participant'):
            self.assertIn(code, codes)
        record = next(r for r in state['records'] if r['stimulus_id'] == s[4]['stimulus_id'])
        self.assertEqual((record['shape_fidelity_1_5'], record['silhouette_fidelity_1_5']), (None, None))
        self.assertEqual(sum(r['stimulus_id'] == s[6]['stimulus_id'] for r in state['records']), 1)
        self.assertEqual(len(state['records']), len(s) - 4)

    def test_conflicting_duplicates_and_incomplete_participants(self):
        rows = [rating_row('P01', x, i, (4, 4, 4, 4)) for i, x in enumerate(self.stimuli, 1)]
        conflict = rows[0][:]
        conflict[RATING_COLS[3]] = '1'  # same export stamp, different rating: cannot be ordered
        complete = participant_file('P02', self.stimuli)
        state = self.run_rows([('P01.csv', ratings_text(rows)), ('P01b.csv', ratings_text([conflict])),
                               ('P02.csv', complete)])
        self.assertIn('duplicate_conflict', codes_of(state['issues']))
        self.assertIn('multiple_files', codes_of(state['issues']))
        status = {p['participant']: p['status'] for p in state['participants']}
        self.assertEqual(status, {'P01': 'excluded', 'P02': 'included'})
        self.assertEqual({r['participant'] for r in state['records']}, {'P02'})
        kept = self.run_rows([('P01.csv', ratings_text(rows[:5])), ('P02.csv', complete)], incomplete='include')
        self.assertEqual({r['participant'] for r in kept['records']}, {'P01', 'P02'})

    def test_form_order_matches_the_browser(self):
        # Recorded 2026-09-21 in Chrome from a generated rating_form.html (study
        # 'pilot_test', participant P02): the presentation order the form produced.
        ids = ['S2ZJRS', 'S33RG4', 'S395WH', 'S6Z9DK', 'S7BWV2', 'S8S48V', 'S8SUTK', 'SB5PF5', 'SC7EC2',
               'SCC2GC', 'SDY8AM', 'SE9ZJN', 'SF22ZQ', 'SF7D5W', 'SFAKP7', 'SFMJ76', 'SJYZ63', 'SNK7DS',
               'SQ7AMF', 'SRYVCU', 'SVVUBW', 'SW7TM7', 'SXQM7S', 'SZS4JS']
        chrome = ['SFAKP7', 'S33RG4', 'SE9ZJN', 'SRYVCU', 'SZS4JS', 'S395WH', 'SFMJ76', 'SW7TM7', 'SC7EC2',
                  'SXQM7S', 'S6Z9DK', 'SDY8AM', 'SNK7DS', 'SQ7AMF', 'S8SUTK', 'SCC2GC', 'S8S48V', 'SJYZ63',
                  'S2ZJRS', 'S7BWV2', 'SVVUBW', 'SB5PF5', 'SF7D5W', 'SF22ZQ']
        self.assertEqual(analysis.form_order('pilot_test', 'P02', ids[::-1]), chrome)
        self.assertNotEqual(analysis.form_order('pilot_test', 'P03', ids), chrome)
        state = self.run_rows([('P01.csv', participant_file('P01', self.stimuli))])
        self.assertTrue(state['participants'][0]['order_matches_form'])
        shuffled = [rating_row('P01', s, i, (4, 4, 4, 4)) for i, s in enumerate(self.stimuli, 1)]
        state = self.run_rows([('P01.csv', ratings_text(shuffled))])
        self.assertIn('order_mismatch', codes_of(state['issues']))
        self.assertEqual(state['participants'][0]['status'], 'included')  # a warning, not an exclusion

    def test_bad_key_or_stimuli_fail_loudly(self):
        header = 'study_id,build_id,method_code,method\n'
        for bad in ('s1,b1,M1,QEM\ns1,b1,M1,DECIMATE\n', 's1,b1,M1,QEM\ns2,b1,M2,DECIMATE\n',
                    's1,b1,M1,QEM\ns1,b2,M2,DECIMATE\n', 's1,,M1,QEM\n'):
            with self.subTest(key=bad), self.assertRaises(analysis.StudyInputError):
                analysis.load_key(header + bad)
        with self.assertRaises(analysis.StudyInputError):  # key file of the old format (no build_id)
            analysis.load_key('study_id,method_code,method\ns1,M1,QEM\n')
        _, build, key = analysis.load_key(self.key_text)
        with self.assertRaises(analysis.StudyInputError):
            analysis.load_stimuli(self.stimuli_text.replace(',M1,', ',M7,'), key, build)
        with self.assertRaises(analysis.StudyInputError):  # key and stimulus list of different builds
            analysis.load_stimuli(self.stimuli_text, key, 'ffffffffffffffff')

    def test_ratings_from_a_superseded_build_are_rejected_not_unblinded(self):
        # Finding 10: a --force rebuild draws a new key; ratings made with the old form must not be
        # read with it.  Same study id, assets, levels and triangle counts; only seed (key) differs.
        old_key, old_stimuli_text, old, old_codes = synthetic_study(seed=5)
        new_key, new_stimuli_text, new, new_codes = synthetic_study(seed=9)
        self.assertNotEqual(old_codes, new_codes)
        self.assertNotEqual(old[0]['build_id'], new[0]['build_id'])
        files = [(f'P0{i}.csv', participant_file(f'P0{i}', old)) for i in range(1, 4)]
        good = analysis.run(files, old_key, old_stimuli_text, n_boot=50)
        self.assertEqual(good['issues'], [])
        self.assertEqual({r['method']: r['overall_quality_1_5'] for r in good['records']},
                         {m: SCORE[m] for m in maker.METHODS})
        stale = analysis.run(files, new_key, new_stimuli_text, n_boot=50)
        self.assertEqual(stale['records'], [])
        self.assertIsNone(stale['results'])
        self.assertEqual(codes_of(stale['issues']), ['wrong_build'])
        self.assertEqual(len(stale['issues']), 3 * len(old))

    def test_final_download_supersedes_a_mid_study_download(self):
        # Finding 32: a partial "Unduh CSV" mid-study plus the final download excluded the participant.
        final = participant_rows('P05', self.stimuli, stamp=('2', '2026-09-21T10:30:00.000Z'))
        partial = [row[:] for row in final[:2]]
        for row in partial:
            row[SEQUENCE_COL], row[TIME_COL] = '1', '2026-09-21T10:05:00.000Z'
        partial[1][RATING_COLS[2]] = partial[1][RATING_COLS[3]] = ''  # half-rated pair at the first download
        partial[0][RATING_COLS[0]] = '1'  # a rating changed afterwards
        files = [('rating_s1_P05_1.csv', ratings_text(partial)), ('rating_s1_P05_2.csv', ratings_text(final)),
                 ('P01.csv', participant_file('P01', self.stimuli))]
        state = self.run_rows(files)
        self.assertEqual([i for i in state['issues'] if i['severity'] == 'error'], [])
        self.assertEqual(codes_of(state['issues']), ['multiple_files', 'superseded_rows'])
        p05 = next(p for p in state['participants'] if p['participant'] == 'P05')
        self.assertEqual((p05['status'], p05['complete_stimuli']), ('included', len(self.stimuli)))
        kept = {r['stimulus_id']: r for r in state['records'] if r['participant'] == 'P05'}
        self.assertEqual(len(kept), len(self.stimuli))
        self.assertTrue(all(r['export_sequence'] == 2 for r in kept.values()))
        first = kept[final[0][STIMULUS_COL]]
        self.assertEqual(first['shape_fidelity_1_5'], SCORE[first['method']])
        # The order of the two files does not matter.
        self.assertEqual(self.run_rows(files[::-1])['records'], state['records'])

    def test_downloads_that_cannot_be_ordered_conflict(self):
        rows = participant_rows('P05', self.stimuli, stamp=('2', '2026-09-21T10:00:00.000Z'))
        other = [row[:] for row in rows[:1]]
        other[0][SEQUENCE_COL], other[0][TIME_COL] = '1', '2026-09-21T11:00:00.000Z'  # newer time, older number
        other[0][RATING_COLS[3]] = '1'
        state = self.run_rows([('a.csv', ratings_text(rows)), ('b.csv', ratings_text(other))])
        self.assertIn('duplicate_conflict', codes_of(state['issues']))
        self.assertEqual(state['participants'][0]['status'], 'excluded')
        other[0][RATING_COLS[3]] = rows[0][RATING_COLS[3]]  # same ratings: kept once, no conflict
        state = self.run_rows([('a.csv', ratings_text(rows)), ('b.csv', ratings_text(other))])
        self.assertEqual(codes_of(state['issues']), ['duplicate_identical', 'multiple_files'])
        self.assertEqual(state['participants'][0]['status'], 'included')
        self.assertIsNone(analysis.export_stamp(dict(export_sequence='0', exported_at=STAMP[1])))
        self.assertIsNone(analysis.export_stamp(dict(export_sequence='1', exported_at='2026-09-21T10:00:00')))
        self.assertEqual(analysis.export_stamp(dict(export_sequence='3', exported_at=STAMP[1]))[0], 3)

    def test_no_ratings_is_a_clear_error_and_nothing_is_invented(self):
        with self.assertRaises(analysis.NoRatingsError):
            self.run_rows([])
        with self.assertRaises(analysis.NoRatingsError):
            analysis.analyse([])
        header_only = self.run_rows([('P01.csv', ratings_text([]))])
        self.assertIsNone(header_only['results'])
        report = analysis.render_report(header_only)
        self.assertIn('Tidak ada rating valid', report)
        self.assertNotIn('Friedman', report)
        self.assertEqual(analysis.main(['--study', str(ROOT / 'study' / '__missing_study__')]), 1)


class StatisticsTests(unittest.TestCase):
    def test_aggregation_of_known_values(self):
        key_text, stimuli_text, stimuli, _ = synthetic_study(views=('front', 'side'))
        files = [(f'P{i:02d}.csv', participant_file(f'P{i:02d}', stimuli)) for i in range(1, 7)]
        state = analysis.run(files, key_text, stimuli_text, n_boot=300, seed=1)
        results = state['results']
        self.assertEqual(results['methods'], ['PERCEPTUAL', 'QEM', 'DECIMATE'])
        self.assertEqual(results['levels'], [2, 3])
        for row in results['summary']:
            self.assertEqual(row['n_ratings'], 6 * 3 * 2 * (2 if row['lod_level'] == 'all' else 1))
            self.assertEqual((row['n_participants'], row['n_assets']), (6, 3))
            self.assertEqual(row['mean'], SCORE[row['method']])
            self.assertEqual((row['median'], row['sd'], row['ci_low'], row['ci_high']), (row['mean'], 0.0) + (row['mean'],) * 2)
        for test in results['friedman']:
            self.assertEqual((test['n'], test['k']), (6, 3))
            self.assertAlmostEqual(test['kendalls_w'], 1.0)
            self.assertAlmostEqual(test['statistic'], 12.0)
        for row in results['pairwise']:
            self.assertEqual((row['n_pairs'], row['n_nonzero'], row['statistic']), (6, 6, 0.0))
            self.assertAlmostEqual(row['p'], 2 / 2 ** 6)
            self.assertAlmostEqual(row['rank_biserial'], 1.0)
            self.assertAlmostEqual(row['p_holm'], 3 * 2 / 2 ** 6)
        self.assertEqual(len(results['pairwise']), 2 * 4 * 3)
        report = analysis.render_report(state)
        for text in ('Statistik deskriptif', 'Uji Friedman', 'Wilcoxon', 'PERCEPTUAL', 'LOD3'):
            self.assertIn(text, report)
        self.assertNotIn('nan', report.lower().replace('nana', ''))

    def test_participant_means_and_control_stimuli(self):
        key_text, stimuli_text, stimuli, _ = synthetic_study(control=True)
        low_control = lambda pid, s: (1,) * 4 if s['method'] == 'REFERENCE' else (SCORE[s['method']],) * 4  # noqa: E731
        files = [('P01.csv', participant_file('P01', stimuli)), ('P02.csv', participant_file('P02', stimuli)),
                 ('P03.csv', participant_file('P03', stimuli, low_control))]
        state = analysis.run(files, key_text, stimuli_text, n_boot=100)
        results = state['results']
        self.assertNotIn('REFERENCE', results['methods'])
        self.assertEqual(len(results['control']), 4)
        self.assertAlmostEqual(results['control'][3]['mean'], (5 * 3 + 5 * 3 + 1 * 3) / 9)
        participants, matrix = analysis.participant_means(state['records'], results['methods'], 2, CRITERIA_OVERALL)
        self.assertEqual(participants, ['P01', 'P02', 'P03'])
        np.testing.assert_array_equal(matrix, [[5, 4, 3]] * 3)
        strict = analysis.run(files, key_text, stimuli_text, n_boot=100, control_min=4.0)
        self.assertEqual([p['participant'] for p in strict['participants'] if p['status'] == 'excluded'], ['P03'])

    def test_clustered_bootstrap_is_deterministic_and_resamples_clusters(self):
        rng = np.random.default_rng(0)
        values = rng.integers(1, 6, 120).astype(float)
        clusters = np.repeat([f'P{i:02d}' for i in range(12)], 10)
        first = analysis.clustered_bootstrap_ci(values, clusters, 2000, seed=11)
        self.assertEqual(first, analysis.clustered_bootstrap_ci(values, clusters, 2000, seed=11))
        self.assertNotEqual(first, analysis.clustered_bootstrap_ci(values, clusters, 2000, seed=12))
        self.assertLess(first[0], values.mean())
        self.assertGreater(first[1], values.mean())
        seq = analysis.cluster_seed(5, 'QEM|2|shape')
        self.assertEqual(analysis.clustered_bootstrap_ci(values, clusters, 500, seq),
                         analysis.clustered_bootstrap_ci(values, clusters, 500, analysis.cluster_seed(5, 'QEM|2|shape')))
        # Two participants, ten identical ratings each: resampled means are 1, 3 or 5
        # (never the narrow interval an i.i.d. bootstrap of 20 ratings would give).
        low, high = analysis.clustered_bootstrap_ci([5.0] * 10 + [1.0] * 10, ['A'] * 10 + ['B'] * 10, 4000, seed=3)
        self.assertEqual((low, high), (1.0, 5.0))
        self.assertTrue(all(math.isnan(v) for v in analysis.clustered_bootstrap_ci([3.0, 4.0], ['A', 'A'], 100)))

    def test_holm_and_kendalls_w(self):
        np.testing.assert_allclose(analysis.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
        adjusted = analysis.holm([0.2, math.nan, 0.01])
        self.assertTrue(math.isnan(adjusted[1]))
        np.testing.assert_allclose(adjusted[[0, 2]], [0.2, 0.02])
        self.assertAlmostEqual(analysis.kendalls_w([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), 1.0)
        self.assertAlmostEqual(analysis.kendalls_w([[1, 2, 3], [3, 2, 1]]), 0.0)
        rng = np.random.default_rng(4)
        for _ in range(50):
            matrix = rng.integers(1, 6, size=(int(rng.integers(3, 15)), 3)).astype(float)
            test = analysis.friedman(matrix, use_scipy=False)
            if math.isfinite(test['statistic']):
                self.assertAlmostEqual(test['kendalls_w'], analysis.kendalls_w(matrix))
        tied = analysis.friedman(np.full((5, 3), 4.0))
        self.assertTrue(math.isnan(tied['p']))
        self.assertIn('same', tied['note'])

    def test_agreement_is_reported_within_lod_and_asset(self):
        # Finding 35: method-agnostic raters agree trivially that 5% LODs look worse than 20% LODs.
        key_text, stimuli_text, stimuli, _ = synthetic_study()
        rng = np.random.default_rng(3)
        files = [(f'P{i:02d}.csv', participant_file(f'P{i:02d}', stimuli, lambda pid, s: (
            int(rng.integers(4, 6)) if s['lod_level'] == 2 else int(rng.integers(1, 3)),) * 4)) for i in range(1, 9)]
        results = analysis.run(files, key_text, stimuli_text, n_boot=50)['results']
        rows = [a for a in results['agreement'] if a['criterion'] == CRITERIA_OVERALL]
        scopes = [(a['scope'], a['lod_level'], a['asset_id']) for a in rows]
        self.assertEqual(scopes, [('pooled', 'all', 'all'), ('lod', 2, 'all'), ('lod', 3, 'all')]
                         + [('asset_lod', lv, a) for lv in (2, 3) for a in ('a1', 'a2', 'a3')])
        pooled, within = rows[0], rows[1:3]
        self.assertEqual((pooled['n_participants'], pooled['n_stimuli']), (8, 18))
        self.assertEqual([a['n_stimuli'] for a in within], [9, 9])
        self.assertGreater(pooled['kendalls_w'], 0.6)
        self.assertTrue(all(a['kendalls_w'] < 0.35 for a in within))
        by_id = {s['stimulus_id']: s for s in stimuli}
        records = analysis.run(files, key_text, stimuli_text, n_boot=50)['records']
        lod2 = sorted(s for s in by_id if by_id[s]['lod_level'] == 2)
        table = {}
        for r in records:
            table.setdefault(r['participant'], {})[r['stimulus_id']] = r[CRITERIA_OVERALL]
        matrix = [[table[p][s] for s in lod2] for p in sorted(table)]
        self.assertAlmostEqual(within[0]['kendalls_w'], analysis.kendalls_w(matrix))
        report = analysis.render_report(analysis.run(files, key_text, stimuli_text, n_boot=50))
        self.assertIn('W gabungan didominasi perbedaan LOD dan aset', report)
        self.assertIn('Dalam LOD3', report)

    def test_wilcoxon_edge_cases(self):
        zero = analysis.wilcoxon_signed_rank([3, 4, 5], [3, 4, 5])
        self.assertTrue(math.isnan(zero['p']))
        self.assertEqual(zero['n_nonzero'], 0)
        one = analysis.wilcoxon_signed_rank([4.0], [3.0])
        self.assertEqual((one['p'], one['statistic'], one['r_plus']), (1.0, 0.0, 1.0))
        self.assertAlmostEqual(analysis.wilcoxon_signed_rank(np.arange(10) + 1.0, np.zeros(10))['p'], 2 / 2 ** 10)

    @unittest.skipIf(scipy_stats is None, 'SciPy not installed')
    def test_friedman_fallback_matches_scipy(self):
        rng = np.random.default_rng(1)
        compared = 0
        for _ in range(300):
            n, k = int(rng.integers(2, 30)), int(rng.integers(3, 6))
            matrix = rng.integers(1, 6, size=(n, k)).astype(float) / rng.integers(1, 4)
            statistic, p, _ = analysis.friedman_fallback(matrix)
            if not math.isfinite(statistic):
                continue
            reference = scipy_stats.friedmanchisquare(*matrix.T)
            self.assertAlmostEqual(statistic, reference.statistic, delta=1e-9 * max(1.0, reference.statistic))
            self.assertAlmostEqual(p, reference.pvalue, delta=1e-12)
            with_scipy, without = analysis.friedman(matrix), analysis.friedman(matrix, use_scipy=False)
            self.assertEqual((with_scipy['engine'], without['engine']), ('scipy', 'numpy fallback'))
            self.assertAlmostEqual(with_scipy['p'], without['p'], delta=1e-12)
            compared += 1
        self.assertGreater(compared, 250)
        for df in (1, 2, 3, 4, 7, 20, 60):
            for x in (1e-3, 0.4, 1.0, 3.3, 9.0, 25.0, 80.0, 200.0):
                expected = scipy_stats.chi2.sf(x, df)
                self.assertAlmostEqual(analysis.chi2_sf(x, df) / expected, 1.0, delta=1e-10)

    @unittest.skipIf(scipy_stats is None, 'SciPy not installed')
    def test_wilcoxon_matches_scipy_in_each_regime(self):
        rng = np.random.default_rng(2)
        for _ in range(150):  # no ties or zeros: SciPy method='exact'
            n = int(rng.integers(3, 45))
            x, y = rng.normal(size=n), rng.normal(size=n) + 0.3
            ours, ref = analysis.wilcoxon_signed_rank(x, y), scipy_stats.wilcoxon(x, y, method='exact')
            self.assertEqual(ours['method'], 'exact')
            self.assertAlmostEqual(ours['p'], ref.pvalue, delta=1e-12)
            self.assertEqual(ours['statistic'], ref.statistic)
        for _ in range(40):  # ties and zeros, n <= 13: SciPy 'auto' enumerates all sign flips (slow)
            n = int(rng.integers(4, 14))
            x, y = rng.integers(1, 6, n).astype(float), rng.integers(1, 6, n).astype(float)
            if np.all(x == y):
                continue
            ours, ref = analysis.wilcoxon_signed_rank(x, y), scipy_stats.wilcoxon(x, y)
            self.assertAlmostEqual(ours['p'], ref.pvalue, delta=1e-12)
            self.assertEqual(ours['statistic'], ref.statistic)
        for _ in range(60):  # n > 50 with ties: tie-corrected normal approximation
            n = int(rng.integers(55, 150))
            difference = rng.choice([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0], size=n)
            ours = analysis.wilcoxon_signed_rank(difference + 5.0, np.full(n, 5.0))
            ref = scipy_stats.wilcoxon(difference + 5.0, np.full(n, 5.0), method='asymptotic', correction=False)
            self.assertTrue(ours['method'].startswith('normal'))
            self.assertAlmostEqual(ours['p'], ref.pvalue, delta=1e-12)

    def test_fallback_pipeline_equals_scipy_pipeline(self):
        key_text, stimuli_text, stimuli, _ = synthetic_study()
        rng = random.Random(9)
        files = [(f'P{i:02d}.csv', participant_file(f'P{i:02d}', stimuli, lambda pid, s: tuple(
            min(5, max(1, SCORE[s['method']] - rng.randint(0, 2))) for _ in range(4)))) for i in range(1, 9)]
        a = analysis.run(files, key_text, stimuli_text, n_boot=500, seed=4)['results']
        b = analysis.run(files, key_text, stimuli_text, n_boot=500, seed=4, use_scipy=False)['results']
        self.assertEqual(a['summary'], b['summary'])
        for x, y in zip(a['friedman'], b['friedman']):
            self.assertAlmostEqual(x['statistic'], y['statistic'], delta=1e-9)
            self.assertAlmostEqual(x['p'], y['p'], delta=1e-12)
        self.assertEqual(a['pairwise'], b['pairwise'])


CRITERIA_OVERALL = 'overall_quality_1_5'


if __name__ == '__main__':
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(case)
                               for case in (ContractTests, MakerHelperTests, StudyBuildTests, ValidationTests,
                                            StatisticsTests))
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        sys.exit(1)
