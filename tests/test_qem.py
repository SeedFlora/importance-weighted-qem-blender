"""Geometry and algorithm regression tests; runnable without Blender.

Run from the repository root with either interpreter:
    python tests/test_qem.py
    blender --background --factory-startup --python-exit-code 1 --python tests/test_qem.py
"""

import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smart_lod import qem
from smart_lod.qem import WEIGHTING_MODES, simplify, simplify_levels


# Frozen v0.1 engine: qem.py of the v0.1.0 release (dist/archive/smart_lod-0.1.0.zip),
# committed unmodified so the equivalence tests never skip.  SHA-256 of its
# bytes with LF line endings.
V01_PATH = ROOT / "tests" / "reference" / "qem_v01.py"
V01_SHA256 = "8eb7ca7754d2f33747212ae5faf0b802200aebdc0d832a58d1c136c2cd540f11"
ARRAY_FIELDS = ("vertices", "faces", "source_faces", "importance")
_V01 = []


def load_v01():
    if not _V01:
        spec = importlib.util.spec_from_file_location("smart_lod_qem_v01", V01_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses resolve their module
        spec.loader.exec_module(module)
        _V01.append(module)
    return _V01[0]


def grid(size=12):
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, size), np.linspace(-1.0, 1.0, size))
    vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(size * size)))
    faces = []
    for row in range(size - 1):
        for col in range(size - 1):
            a = row * size + col
            faces.extend(((a, a + 1, a + size + 1), (a, a + size + 1, a + size)))
    return vertices, np.asarray(faces, dtype=np.int64)


def octasphere(subdivisions=3):
    vertices = [
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    ]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(subdivisions):
        cache = {}
        subdivided = []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                point = np.asarray(vertices[a]) + np.asarray(vertices[b])
                point /= np.linalg.norm(point)
                cache[key] = len(vertices)
                vertices.append(tuple(point))
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            subdivided.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = subdivided
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


def point_surface_distance(points, corners):
    """Exact distance from each point to the nearest of the (T, 3, 3) triangles."""
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    height = np.einsum("ptk,tk->pt", points[:, None] - a[None], normal)
    foot = points[:, None] - height[..., None] * normal[None]
    inside = np.ones(height.shape, dtype=bool)
    edge_distance = np.full(height.shape, np.inf)
    for u, v in ((a, b), (b, c), (c, a)):
        inside &= np.einsum("ptk,tk->pt", np.cross(v - u, foot - u[None]), normal) >= 0.0
        d = v - u
        t = np.clip(np.einsum("ptk,tk->pt", points[:, None] - u[None], d) / np.einsum("tk,tk->t", d, d), 0.0, 1.0)
        closest = u[None] + t[..., None] * d[None]
        edge_distance = np.minimum(edge_distance, np.linalg.norm(points[:, None] - closest, axis=2))
    return np.where(inside, np.abs(height), edge_distance).min(axis=1)


def volume(vertices, faces):
    tri = vertices[faces]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def uv_sphere(segments=24, rings=16):
    """Blender-style UV sphere and the locks a UV seam gives it.

    feature_constraints locks a UV sphere's seam meridian and both pole fans
    (every pole-fan edge is a UV discontinuity); 63 of 362 vertices here.
    """
    theta = np.pi * np.arange(1, rings) / rings
    phi = 2.0 * np.pi * np.arange(segments) / segments
    ring_points = np.stack(np.broadcast_arrays(
        np.sin(theta)[:, None] * np.cos(phi), np.sin(theta)[:, None] * np.sin(phi), np.cos(theta)[:, None]), axis=-1)
    vertices = np.vstack(([0.0, 0.0, 1.0], ring_points.reshape(-1, 3), [0.0, 0.0, -1.0]))
    south = len(vertices) - 1
    ring = lambda i, j: 1 + (i - 1) * segments + j % segments
    faces = [(0, ring(1, j), ring(1, j + 1)) for j in range(segments)]
    for i in range(1, rings - 1):
        for j in range(segments):
            a, b, c, d = ring(i, j), ring(i + 1, j), ring(i + 1, j + 1), ring(i, j + 1)
            faces += [(a, b, d), (b, c, d)]
    faces += [(south, ring(rings - 1, j + 1), ring(rings - 1, j)) for j in range(segments)]
    locked = np.zeros(len(vertices), dtype=bool)
    locked[[0, south] + [ring(i, 0) for i in range(1, rings)]] = True
    locked[[ring(i, j) for i in (1, rings - 1) for j in range(segments)]] = True
    return vertices, np.asarray(faces, dtype=np.int64), locked


def source_flips(result, vertices, faces):
    """Output faces whose normal has dot <= 0 with their source triangle's normal."""
    normal = lambda tri: np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    source = normal(vertices[faces])[result.source_faces]
    return int(np.count_nonzero(np.einsum("ij,ij->i", normal(result.vertices[result.faces]), source) <= 0.0))


class QEMTests(unittest.TestCase):
    def assert_valid(self, result, input_count):
        self.assertEqual(result.vertices.ndim, 2)
        self.assertEqual(result.vertices.shape[1], 3)
        self.assertEqual(result.faces.shape[1], 3)
        self.assertTrue(np.all(np.isfinite(result.vertices)))
        self.assertTrue(np.all(np.isfinite(result.importance)))
        self.assertEqual(result.importance.shape, (len(result.vertices),))
        self.assertEqual(result.source_faces.shape, (len(result.faces),))
        self.assertEqual(result.stats["output_faces"], len(result.faces))
        self.assertTrue(np.all(result.source_faces >= 0))
        self.assertTrue(np.all(result.source_faces < input_count))
        if len(result.faces):
            self.assertGreaterEqual(int(result.faces.min()), 0)
            self.assertLess(int(result.faces.max()), len(result.vertices))
            corners = result.vertices[result.faces]
            cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
            self.assertTrue(np.all(np.linalg.norm(cross, axis=1) > 1e-12))
            keys = np.sort(result.faces, axis=1)
            self.assertEqual(len(np.unique(keys, axis=0)), len(result.faces))

    def assert_same_result(self, result, expected, ignore_stats=("seconds", "pass_targets")):
        for field in ARRAY_FIELDS:
            actual, wanted = getattr(result, field), getattr(expected, field)
            self.assertEqual(actual.dtype, wanted.dtype, field)
            np.testing.assert_array_equal(actual, wanted, err_msg=field)
        strip = lambda stats: {k: v for k, v in stats.items() if k not in ignore_stats}
        self.assertEqual(strip(result.stats), strip(expected.stats))

    def test_planar_grid_reduces_without_inversion_or_boundary_drift(self):
        vertices, faces = grid()
        result = simplify(vertices, faces, 70)
        self.assert_valid(result, len(faces))
        self.assertTrue(result.stats["target_reached"])
        self.assertLessEqual(len(result.faces), 70)
        self.assertTrue(np.all(result.vertices[:, 2] == 0.0))
        tri = result.vertices[result.faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.assertTrue(np.all(normals[:, 2] > 0.0))
        edges = np.sort(np.concatenate((result.faces[:, [0, 1]], result.faces[:, [1, 2]], result.faces[:, [2, 0]])), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_vertices = np.unique(unique[counts == 1])
        boundary_points = result.vertices[boundary_vertices]
        self.assertTrue(np.all(np.isclose(np.abs(boundary_points[:, :2]), 1.0).any(axis=1)))

    def test_sphere_reduces_preserving_closed_topology_and_normals(self):
        vertices, faces = octasphere(3)
        result = simplify(vertices, faces, 120)
        self.assert_valid(result, len(faces))
        self.assertTrue(result.stats["target_reached"])
        edges = np.sort(np.concatenate((result.faces[:, [0, 1]], result.faces[:, [1, 2]], result.faces[:, [2, 0]])), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue(np.all(counts == 2))
        self.assertEqual(len(result.vertices) - len(unique) + len(result.faces), 2)
        tri = result.vertices[result.faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.assertTrue(np.all(np.einsum("ij,ij->i", normals, tri.mean(axis=1)) > 0.0))

    def test_zero_strength_and_zero_importance_are_exact_baseline(self):
        vertices, faces = octasphere(2)
        weights = np.linspace(0.0, 1.0, len(vertices))
        baseline = simplify(vertices, faces, 70, strength=0.0)
        for weighting in WEIGHTING_MODES:
            with self.subTest(weighting=weighting):
                disabled = simplify(vertices, faces, 70, importance=weights, strength=0.0, weighting=weighting)
                zero_weights = simplify(vertices, faces, 70, importance=np.zeros(len(vertices)),
                                        strength=40.0, weighting=weighting)
                for result in (disabled, zero_weights):
                    np.testing.assert_array_equal(result.vertices, baseline.vertices)
                    np.testing.assert_array_equal(result.faces, baseline.faces)
                    np.testing.assert_array_equal(result.source_faces, baseline.source_faces)

    def test_importance_changes_actual_planar_collapse_cost(self):
        # Anchor mode only: quadric weighting scales plane errors, and every
        # plane error in an exactly planar region is zero, so it cannot protect
        # this grid.  The positional anchor quadric is what makes it work here.
        vertices, faces = grid(15)
        protected = (np.abs(vertices[:, 0]) < 0.45) & (np.abs(vertices[:, 1]) < 0.45)
        weights = protected.astype(float)
        baseline = simplify(vertices, faces, 80, strength=0.0)
        perceptual = simplify(vertices, faces, 80, importance=weights, strength=8.0, weighting="anchor")
        self.assertTrue(perceptual.stats["target_reached"])

        def protected_displacement(result):
            distances = np.linalg.norm(vertices[protected, None, :] - result.vertices[None, :, :], axis=2)
            return float(np.mean(np.min(distances, axis=1)))

        self.assertLess(protected_displacement(perceptual), protected_displacement(baseline) * 0.75)
        self.assertGreater(float(np.max(perceptual.importance)), 0.0)

    def test_quadric_weighting_protects_weighted_curved_region(self):
        vertices, faces = octasphere(4)
        cap = vertices[:, 2] > 0.7
        baseline = simplify(vertices, faces, 256, strength=0.0)
        weighted = simplify(vertices, faces, 256, importance=cap.astype(float), strength=8.0)
        self.assert_valid(weighted, len(faces))
        self.assertEqual(len(weighted.faces), len(baseline.faces))
        # Measured: 39 vs 21 cap vertices, mean cap error 0.0052 vs 0.0079.
        cap_vertices = lambda result: int(np.count_nonzero(result.vertices[:, 2] > 0.7))
        cap_error = lambda result: float(point_surface_distance(vertices[cap], result.vertices[result.faces]).mean())
        self.assertGreater(cap_vertices(weighted), 1.3 * cap_vertices(baseline))
        self.assertLess(cap_error(weighted), 0.85 * cap_error(baseline))

    def test_quadric_weighting_does_not_shrink_convex_surface_like_anchor(self):
        # With uniform importance quadric weighting only rescales every error,
        # while anchoring pulls clusters toward centroids inside the sphere.
        vertices, faces = octasphere(4)
        ones = np.ones(len(vertices))
        reference = volume(vertices, faces)
        ratios = {
            label: volume(result.vertices, result.faces) / reference
            for label, result in (
                ("baseline", simplify(vertices, faces, 256, strength=0.0)),
                ("quadric", simplify(vertices, faces, 256, importance=ones, weighting="quadric")),
                ("anchor", simplify(vertices, faces, 256, importance=ones, weighting="anchor")),
            )
        }
        # Measured: baseline 0.9825, quadric 0.9829, anchor 0.9502.
        self.assertLess(abs(ratios["quadric"] - ratios["baseline"]), 0.005)
        self.assertLess(ratios["anchor"], ratios["baseline"] - 0.02)

    def test_levels_equal_independent_runs(self):
        cases = []
        sphere, sphere_faces = octasphere(3)
        sheet, sheet_faces = grid(12)
        rng = np.random.default_rng(3)
        for vertices, faces in ((sphere, sphere_faces), (sheet, sheet_faces)):
            n, m = len(vertices), len(faces)
            weights = rng.uniform(0.0, 1.0, n) * (rng.uniform(0.0, 1.0, n) < 0.5)
            locks = rng.uniform(0.0, 1.0, n) < 0.05
            # Budgets above the input, duplicated, and unreachable (a surface
            # component always keeps a triangle); unsorted on purpose.
            targets = [m // 5, m + 7, m // 2, 0, m // 5, m, 30]
            for weighting in WEIGHTING_MODES:
                cases += [(vertices, faces, targets, None, 8.0, None, weighting),
                          (vertices, faces, targets, weights, 8.0, None, weighting),
                          (vertices, faces, targets, weights, 3.0, locks, weighting)]
        cases.append((sphere, sphere_faces, [100, 10], None, 8.0, np.ones(len(sphere), bool), "quadric"))
        for vertices, faces, targets, weights, strength, locks, weighting in cases:
            with self.subTest(n=len(vertices), weights=weights is not None, locks=locks is not None,
                              weighting=weighting, targets=targets):
                levels = simplify_levels(vertices, faces, targets, weights, strength, locks, weighting=weighting)
                self.assertEqual(len(levels), len(targets))
                for target, level in zip(targets, levels):
                    single = simplify(vertices, faces, target, weights, strength, locks, weighting=weighting)
                    self.assert_same_result(level, single)
                    self.assertEqual(source_flips(level, vertices, faces), 0)
                    self.assertEqual(level.stats["target_faces"], target)
                    self.assertTrue(level.stats["single_pass"])
                    self.assertEqual(level.stats["pass_targets"], sorted(set(targets)))
                    self.assertEqual(single.stats["pass_targets"], [target])
                # Captures happen from the largest budget down, so seconds do not decrease.
                order = sorted(range(len(targets)), key=lambda i: -targets[i])
                seconds = [levels[i].stats["seconds"] for i in order]
                self.assertEqual(seconds, sorted(seconds))

    def test_duplicate_levels_are_independent_copies(self):
        vertices, faces = octasphere(2)
        first, second = simplify_levels(vertices, faces, np.array([40, 40]))
        self.assert_same_result(first, second, ignore_stats=())
        second.vertices[:] = 0.0
        second.stats["rejected_collapses"]["topology"] = -1
        self.assertFalse(np.all(first.vertices == 0.0))
        self.assertGreaterEqual(first.stats["rejected_collapses"]["topology"], 0)

    def test_progress_is_monotone_ends_at_one_and_never_changes_results(self):
        vertices, faces = octasphere(4)
        calls = []
        watched = simplify_levels(vertices, faces, [1500, 100], progress=calls.append)
        plain = simplify_levels(vertices, faces, [1500, 100])
        for result, expected in zip(watched, plain):
            self.assert_same_result(result, expected)
        collapses = watched[1].stats["collapses"]
        self.assertTrue(watched[1].stats["target_reached"])
        self.assertGreaterEqual(len(calls), 2)
        self.assertLessEqual(len(calls), collapses // 256 + 1)
        self.assertEqual(calls, sorted(calls))
        self.assertTrue(all(0.0 <= fraction <= 1.0 for fraction in calls))
        self.assertEqual(calls[-1], 1.0)
        # Unreachable budget: the final report is the true fraction, below one.
        stuck = []
        simplify(vertices, faces, 10, locked=np.ones(len(vertices), bool), progress=stuck.append)
        self.assertEqual(stuck, [0.0])

    def test_progress_exception_aborts(self):
        vertices, faces = octasphere(4)

        class Cancelled(Exception):
            pass

        def cancel(fraction):
            raise Cancelled(fraction)

        with self.assertRaises(Cancelled):
            simplify_levels(vertices, faces, [1000, 100], progress=cancel)
        with self.assertRaises(Cancelled):
            simplify(vertices, faces, len(faces), progress=cancel)  # the final report

    def assert_same_as_v01(self, result, reference):
        """Bit-for-bit v0.1: arrays and stats, with the guard never firing."""
        new_keys = ("seconds", "single_pass", "pass_targets", "weighting", "strength")
        expected = {**reference.stats, "rejected_collapses": {**reference.stats["rejected_collapses"], "source_flip": 0}}
        self.assert_same_result(result, qem.SimplificationResult(
            reference.vertices, reference.faces, reference.source_faces, reference.importance, expected),
            ignore_stats=new_keys)

    def test_v01_reference_is_the_released_engine(self):
        self.assertTrue(V01_PATH.is_file(), f"missing frozen v0.1 engine {V01_PATH}")
        data = V01_PATH.read_bytes().replace(b"\r\n", b"\n")  # tolerate a CRLF checkout
        self.assertEqual(hashlib.sha256(data).hexdigest(), V01_SHA256)

    def test_anchor_mode_matches_v01_until_the_source_flip_guard_fires(self):
        # The guard is the only v0.2 change to v0.1's collapse decisions, so
        # anchor mode must equal v0.1 bit for bit at every budget the pass
        # reaches before the guard first fires, and at the requested budget
        # when it never fires.  Measured: it fires in 5 of these 27 cases, all
        # 5 final meshes differ from v0.1's, and in 3 of them v0.1's final mesh
        # has a face flipped against its source triangle.
        v01 = load_v01()
        rng = np.random.default_rng(11)
        sphere, sphere_faces = octasphere(3)
        noisy = sphere * (1.0 + 0.05 * rng.normal(size=(len(sphere), 1))) * 37.0 + [100.0, -3.0, 7.0]
        sheet, sheet_faces = grid(15)
        cases = []
        for vertices, faces in ((sphere, sphere_faces), (noisy, sphere_faces), (sheet, sheet_faces)):
            n, m = len(vertices), len(faces)
            weights = rng.uniform(0.0, 1.0, n) * (rng.uniform(0.0, 1.0, n) < 0.5)
            locks = rng.uniform(0.0, 1.0, n) < 0.05
            for target in (m // 2, m // 6, 0):
                cases += [(vertices, faces, target, None, 8.0, None),
                          (vertices, faces, target, weights, 8.0, None),
                          (vertices, faces, target, weights, 3.0, locks)]
        fired_cases = differing_cases = v01_flipped_cases = 0
        for vertices, faces, target, weights, strength, locks in cases:
            with self.subTest(n=len(vertices), target=target, weights=weights is not None, locks=locks is not None):
                # One snapshot per budget: each records the guard count so far.
                budgets = list(range(len(faces), target - 1, -1))
                levels = simplify_levels(vertices, faces, budgets, weights, strength, locks, weighting="anchor")
                fired = [level.stats["rejected_collapses"]["source_flip"] for level in levels]
                self.assertEqual(fired, sorted(fired))
                clean = fired.count(0)  # budgets reached before the guard fired
                reference = v01.simplify(vertices, faces, budgets[clean - 1], weights, strength, locks)
                self.assert_same_as_v01(levels[clean - 1], reference)
                if fired[-1]:
                    fired_cases += 1
                    final = v01.simplify(vertices, faces, target, weights, strength, locks)
                    v01_flipped_cases += source_flips(final, vertices, faces) > 0
                    self.assertEqual(source_flips(levels[-1], vertices, faces), 0)
                    differing_cases += not all(np.array_equal(getattr(levels[-1], field), getattr(final, field))
                                               for field in ARRAY_FIELDS)
        self.assertEqual((fired_cases, differing_cases, v01_flipped_cases), (5, 5, 3))

    def test_no_output_face_points_against_its_source_triangle(self):
        # Review finding: with locked UV-seam vertices, repeated collapses that
        # each turn a face by less than the per-collapse limit folded faces
        # against their input triangles.  Unweighted v0.1, identical to v0.2
        # before the guard, does so here (measured: 2 faces at 144 triangles,
        # min cos -0.12); noise does it without locks (measured: 4 at 256).
        v01 = load_v01()
        seam_sphere, seam_faces, seam_locks = uv_sphere()
        self.assertEqual(int(seam_locks.sum()), 63)
        self.assertGreater(volume(seam_sphere, seam_faces), 0.0)
        rng = np.random.default_rng(1)
        sphere, sphere_faces = octasphere(4)
        noisy = sphere * (1.0 + 0.15 * rng.normal(size=(len(sphere), 1)))
        for vertices, faces, target, locks in ((seam_sphere, seam_faces, 144, seam_locks),
                                              (noisy, sphere_faces, 256, None)):
            with self.subTest(n=len(vertices), locks=locks is not None):
                old = v01.simplify(vertices, faces, target, None, 0.0, locks)
                self.assertGreater(source_flips(old, vertices, faces), 0)
                result = simplify(vertices, faces, target, locked=locks)
                self.assert_valid(result, len(faces))
                self.assertEqual(source_flips(result, vertices, faces), 0)
                self.assertGreater(result.stats["rejected_collapses"]["source_flip"], 0)
                self.assertTrue(result.stats["target_reached"])
                if locks is not None:
                    for point in vertices[locks]:
                        self.assertTrue(np.any(np.all(result.vertices == point, axis=1)))

    def test_locked_positions_survive_exactly(self):
        vertices, faces = octasphere(2)
        vertices = vertices * 123.456789 + np.array([5000.125, -371.27, 41.23])
        locked = np.zeros(len(vertices), dtype=bool)
        locked[[0, 1, 2, 3, 4, 5, 23]] = True
        result = simplify(vertices, faces, 50, locked=locked)
        self.assert_valid(result, len(faces))
        for point in vertices[locked]:
            self.assertTrue(np.any(np.all(result.vertices == point, axis=1)))
        unchanged = simplify(vertices, faces, 10, locked=np.ones(len(vertices), dtype=bool))
        self.assertFalse(unchanged.stats["target_reached"])
        self.assertEqual(unchanged.stats["stop_reason"], "constraints")
        np.testing.assert_array_equal(unchanged.vertices, vertices)
        np.testing.assert_array_equal(unchanged.faces, faces)

    def test_scale_and_translation_invariance(self):
        vertices, faces = octasphere(2)
        weights = np.maximum(0.0, vertices[:, 2])
        result = simplify(vertices, faces, 70, importance=weights)
        moved = simplify(vertices * 1000.0 + [800.0, -600.0, 200.0], faces, 70, importance=weights)
        # Symmetric equal-cost edges can have roundoff-dependent order, so test
        # the resulting geometric quality and count rather than array identity.
        self.assertEqual(len(result.faces), len(moved.faces))
        self.assert_valid(moved, len(faces))
        moved_back = (moved.vertices - [800.0, -600.0, 200.0]) / 1000.0
        self.assertLess(abs(np.mean(np.linalg.norm(result.vertices, axis=1)) - np.mean(np.linalg.norm(moved_back, axis=1))), 0.02)

    def test_degenerate_and_duplicate_faces_are_cleaned_with_source_mapping(self):
        vertices = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0)], dtype=float)
        faces = np.array([(0, 1, 2), (2, 1, 0), (0, 1, 3), (1, 1, 2)], dtype=np.int64)
        result = simplify(vertices, faces, 10)
        self.assertEqual(len(result.faces), 1)
        self.assertEqual(result.stats["removed_invalid_faces"], 3)
        np.testing.assert_array_equal(result.source_faces, [0])
        self.assert_valid(result, len(faces))

    def test_nonmanifold_fan_is_frozen_safely(self):
        vertices = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)], dtype=float)
        faces = np.array([(0, 1, 2), (1, 0, 3), (0, 1, 4)], dtype=np.int64)
        result = simplify(vertices, faces, 1)
        self.assert_valid(result, len(faces))
        self.assertGreaterEqual(result.stats["nonmanifold_vertices"], 2)
        self.assertFalse(result.stats["target_reached"])

    def test_tiny_surface_component_is_not_deleted(self):
        vertices = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=float)
        faces = np.array([(0, 1, 2)], dtype=np.int64)
        result = simplify(vertices, faces, 0)
        self.assertEqual(len(result.faces), 1)
        self.assertFalse(result.stats["target_reached"])
        self.assertEqual(result.stats["stop_reason"], "constraints")

    def test_empty_mesh_and_isolated_locked_vertex(self):
        empty = simplify(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), 0)
        self.assertEqual(empty.vertices.shape, (0, 3))
        self.assertEqual(empty.faces.shape, (0, 3))
        result = simplify(np.array([(2.0, 3.0, 4.0)]), np.empty((0, 3), dtype=np.int64), 0, locked=[True])
        np.testing.assert_array_equal(result.vertices, [[2.0, 3.0, 4.0]])
        levels = simplify_levels(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), [5, 0])
        self.assertEqual([len(level.faces) for level in levels], [0, 0])

    def test_inputs_are_not_mutated_and_results_are_deterministic(self):
        vertices, faces = grid(8)
        weights = np.linspace(0, 1, len(vertices))
        locked = np.zeros(len(vertices), dtype=bool)
        originals = [a.copy() for a in (vertices, faces, weights, locked)]
        for weighting in WEIGHTING_MODES:
            first = simplify(vertices, faces, 35, weights, 8.0, locked, weighting=weighting)
            second = simplify_levels(vertices, faces, [35], weights, 8.0, locked, weighting=weighting)[0]
            for current, previous in zip((vertices, faces, weights, locked), originals):
                np.testing.assert_array_equal(current, previous)
            np.testing.assert_array_equal(first.vertices, second.vertices)
            np.testing.assert_array_equal(first.faces, second.faces)

    def test_invalid_inputs_fail_clearly(self):
        vertices, faces = grid(4)
        bad_cases = [
            (vertices[:, :2], faces, 5, {}),
            (vertices, faces[:, :2], 5, {}),
            (vertices, faces.astype(float), 5, {}),
            (vertices, np.array([[0, 1, 999]]), 5, {}),
            (vertices, np.array([[0, 1, -1]]), 5, {}),
            (vertices * np.nan, faces, 5, {}),
            (vertices, faces, -1, {}),
            (vertices, faces, 1.5, {}),
            (vertices, faces, True, {}),
            (vertices, faces, 5, {"importance": [0.0]}),
            (vertices, faces, 5, {"importance": np.full(len(vertices), 1.1)}),
            (vertices, faces, 5, {"importance": np.full(len(vertices), np.nan)}),
            (vertices, faces, 5, {"locked": [True]}),
            (vertices, faces, 5, {"strength": -1.0}),
            (vertices, faces, 5, {"strength": np.inf}),
            (vertices, faces, 5, {"strength": np.nan}),
            (vertices, faces, 5, {"strength": "strong"}),
            (vertices, faces, 5, {"strength": 2.0e6}),
            (vertices, faces, 5, {"strength": 1.0e103}),
            (vertices, faces, 5, {"weighting": "planar"}),
            (vertices, faces, 5, {"weighting": None}),
            (vertices, faces, 5, {"weighting": ["quadric"]}),
            (vertices, faces, 5, {"progress": 0.5}),
        ]
        for v, f, target, kwargs in bad_cases:
            with self.subTest(target=target, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    simplify(v, f, target, **kwargs)

    def test_invalid_level_targets_fail_clearly(self):
        vertices, faces = grid(4)
        for targets in ([], 5, "5", np.array(5), [5, -1], [5, 1.5], [True], [[5]], None, [5, None]):
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    simplify_levels(vertices, faces, targets)
        with self.assertRaises(ValueError):
            simplify_levels(vertices, faces, [5], weighting="QUADRIC")
        with self.assertRaises(ValueError):
            simplify_levels(vertices, faces, [5], strength=np.nextafter(qem.MAX_STRENGTH, np.inf))

    def test_strength_limit_and_nonfinite_costs(self):
        # Review finding: 1e103 raised OverflowError and 1.7e308 turned NaN
        # costs into 0, so the protected cap became the cheapest region
        # (measured cap vertices: 13 quadric, 3 anchor; 16 unweighted).
        vertices, faces = octasphere(3)
        cap = (vertices[:, 2] > 0.5).astype(float)
        cap_vertices = lambda result: int(np.count_nonzero(result.vertices[:, 2] > 0.5))
        self.assertEqual(qem.MAX_STRENGTH, 1.0e6)
        unweighted = cap_vertices(simplify(vertices, faces, 100, strength=0.0))
        for weighting in WEIGHTING_MODES:
            with self.subTest(weighting=weighting):
                limit = simplify(vertices, faces, 100, cap, qem.MAX_STRENGTH, weighting=weighting)
                self.assert_valid(limit, len(faces))
                self.assertGreater(cap_vertices(limit), unweighted)
                # Past the limit (only by lifting it), overflowing costs are
                # skipped rather than collapsed first: the cap keeps every vertex.
                with mock.patch.object(qem, "MAX_STRENGTH", math.inf), np.errstate(over="ignore"):
                    extreme = simplify(vertices, faces, 100, cap, 1.7e308, weighting=weighting)
                self.assert_valid(extreme, len(faces))
                self.assertEqual(cap_vertices(extreme), int(cap.sum()))
                self.assertEqual(extreme.stats["stop_reason"], "constraints")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QEMTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        sys.exit(1)
