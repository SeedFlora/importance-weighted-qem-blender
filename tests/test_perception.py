"""Perceptual importance channel tests; runnable without Blender.

Run: PYTHONPATH=. python tests/test_perception.py
Camera ray casting needs mathutils and is covered by test_perception_blender.py.
"""

import inspect
import math
import unittest

import numpy as np

from smart_lod import perception
from smart_lod.perception import (DEFAULT_SILHOUETTE_RESOLUTION, DEFAULT_SILHOUETTE_VIEWS, DEFAULT_WEIGHTS,
                                  camera_visibility, compute_importance, mesh_saliency, outer_silhouette,
                                  sharp_features)


def icosphere(subdivisions=3):
    t = (1.0 + 5.0 ** 0.5) / 2.0
    vertices = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
                (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    vertices = [np.asarray(p, dtype=float) / np.linalg.norm(p) for p in vertices]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdivisions):
        cache, subdivided = {}, []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                point = vertices[a] + vertices[b]
                cache[key] = len(vertices)
                vertices.append(point / np.linalg.norm(point))
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            subdivided.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = subdivided
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


def bump(vertices, axis=(0.0, 0.0, 1.0), height=0.3, width=0.25):
    axis = np.asarray(axis) / np.linalg.norm(axis)
    angle = np.arccos(np.clip(vertices @ axis, -1.0, 1.0))
    return vertices * (1.0 + height * np.exp(-(angle / width) ** 2))[:, None]


def weld(points, faces):
    _, first, inverse = np.unique(np.round(points, 9), axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    return points[first[order]], remap[inverse.reshape(-1)][np.asarray(faces)]


def subdivided_cube(n=6):
    """Axis cube [-1, 1]^3 with an n x n grid per side, outward winding, welded."""
    points, faces = [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            u = np.zeros(3)
            u[(axis + 1) % 3] = 1.0
            v = np.cross(normal, u)
            base = len(points)
            for i in range(n + 1):
                for j in range(n + 1):
                    points.append(normal + (2.0 * i / n - 1.0) * u + (2.0 * j / n - 1.0) * v)
            for i in range(n):
                for j in range(n):
                    a = base + i * (n + 1) + j
                    b, c, d = a + (n + 1), a + (n + 1) + 1, a + 1
                    faces.extend(((a, b, c), (a, c, d)))
    return weld(np.asarray(points), faces)


def lathe(profile, segments=48, step=0.08):
    """Closed surface of revolution about +Z; profile runs axis -> axis in (r, z)."""
    profile = np.asarray(profile, dtype=float)
    dense = [profile[0]]
    for a, b in zip(profile[:-1], profile[1:]):
        count = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        dense.extend(a + (b - a) * (k / count) for k in range(1, count + 1))
    dense = np.asarray(dense)
    angles = 2.0 * math.pi * np.arange(segments) / segments
    vertices = [(0.0, 0.0, dense[0, 1])]
    for r, z in dense[1:-1]:
        vertices.extend((r * math.cos(a), r * math.sin(a), z) for a in angles)
    vertices.append((0.0, 0.0, dense[-1, 1]))
    rings, last = len(dense) - 2, len(vertices) - 1

    def ring(i, k):
        return 1 + i * segments + k % segments

    faces = [(0, ring(0, k + 1), ring(0, k)) for k in range(segments)]
    for i in range(rings - 1):
        for k in range(segments):
            a, b, c, d = ring(i, k), ring(i, k + 1), ring(i + 1, k + 1), ring(i + 1, k)
            faces.extend(((a, b, c), (a, c, d)))
    faces.extend((last, ring(rings - 1, k), ring(rings - 1, k + 1)) for k in range(segments))
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    volume = np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
    return vertices, (faces if volume > 0 else faces[:, ::-1].copy())


def tetrahedron():
    vertices = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    return vertices, np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])


def uv_sphere(rings=24, segments=48):
    """Latitude-longitude unit sphere with pole fans, outward winding."""
    t = np.linspace(0.0, math.pi, rings + 1)
    return lathe(np.column_stack((np.sin(t), -np.cos(t))), segments=segments, step=10.0)


def crop(vertices, faces, z_min):
    """Faces whose corners all have z > z_min, with unused vertices removed."""
    faces = faces[(vertices[faces][:, :, 2] > z_min).all(axis=1)]
    used = np.unique(faces)
    remap = np.full(len(vertices), -1)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


def rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    angle = math.radians(degrees)
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


class SaliencyTests(unittest.TestCase):
    def test_invariant_to_uniform_scale_and_translation(self):
        vertices, faces = icosphere(3)
        vertices = bump(vertices)
        reference = mesh_saliency(vertices, faces)
        for scale, offset in ((7.5, (3.0, -2.0, 10.0)), (0.01, (-100.0, 0.0, 5.0))):
            with self.subTest(scale=scale):
                moved = mesh_saliency(vertices * scale + np.asarray(offset), faces)
                np.testing.assert_allclose(moved, reference, atol=1e-5)

    def test_smooth_sphere_low_and_even_bump_is_peak(self):
        vertices, faces = icosphere(4)
        smooth = mesh_saliency(vertices, faces)
        self.assertLess(smooth.max(), 0.1)
        self.assertLess(smooth.std(), 0.02)
        # Without the noise gate the paper normalization stretches noise to [0, 1].
        self.assertAlmostEqual(mesh_saliency(vertices, faces, noise_floor=0).max(), 1.0)
        axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
        bumped = bump(vertices, axis=axis)
        salient = mesh_saliency(bumped, faces)
        angle = np.arccos(np.clip(vertices @ axis, -1.0, 1.0))
        self.assertAlmostEqual(salient.max(), 1.0)
        self.assertLess(angle[np.argmax(salient)], 0.6)
        self.assertGreater(salient[angle < 0.6].mean(), 5.0 * salient[angle > 1.0].mean())
        self.assertLess(np.percentile(salient[angle > 1.0], 95), 0.1)

    def test_flat_open_grid_has_no_boundary_artifact(self):
        x, y = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
        vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(81)))
        faces = [(r * 9 + c, r * 9 + c + 1, r * 9 + c + 10) for r in range(8) for c in range(8)]
        faces += [(r * 9 + c, r * 9 + c + 10, r * 9 + c + 9) for r in range(8) for c in range(8)]
        np.testing.assert_array_equal(mesh_saliency(vertices, faces), 0.0)

    def test_curved_open_cap_has_no_boundary_artifact(self):
        # Half of a unit sphere: true |H| = 1 everywhere, rim included. The
        # one-sided n . L x estimate gave the rim about half of that, and the rim
        # became the strongest saliency feature (max 1 on the border).
        vertices, faces = crop(*icosphere(4), 0.0)
        mesh = perception._Mesh(vertices, faces)
        curvature, _ = perception._mean_curvature(mesh, *perception._laplacian_parts(mesh))
        rim = mesh.border_vertex
        self.assertGreater(rim.sum(), 40)
        np.testing.assert_allclose(np.median(curvature[rim]), np.median(curvature[~rim]), rtol=0.02)
        self.assertLess(mesh_saliency(vertices, faces).max(), 0.1)
        # A real feature on the cap is still the peak and the rim stays low.
        axis = np.array([0.2, -0.3, 0.9]) / np.linalg.norm([0.2, -0.3, 0.9])
        salient = mesh_saliency(bump(vertices, axis=axis), faces)
        self.assertLess(np.arccos(np.clip(vertices[np.argmax(salient)] @ axis, -1.0, 1.0)), 0.6)
        self.assertLess(salient[rim].mean(), 0.1)

    def test_parameter_validation(self):
        vertices, faces = tetrahedron()
        for kwargs in ({'scales': ()}, {'scales': (2, -1)}, {'epsilon_fraction': 0.0},
                       {'epsilon_fraction': float('nan')}, {'noise_floor': -1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                mesh_saliency(vertices, faces, **kwargs)


class SilhouetteTests(unittest.TestCase):
    def test_cup_cavity_is_zero_outer_rim_is_marked(self):
        outer, inner, height, floor = 1.0, 0.7, 2.0, 0.4
        vertices, faces = lathe([(0, 0), (outer, 0), (outer, height), (inner, height),
                                 (inner, floor), (0, floor)])
        silhouette = outer_silhouette(vertices, faces)
        radius = np.linalg.norm(vertices[:, :2], axis=1)
        cavity = (radius < inner + 1e-9) & (vertices[:, 2] < height - 0.25) & (vertices[:, 2] > floor - 1e-9)
        rim = (np.abs(radius - outer) < 1e-9) & (np.abs(vertices[:, 2] - height) < 1e-9)
        self.assertGreater(cavity.sum(), 300)
        self.assertEqual(rim.sum(), 48)
        np.testing.assert_array_equal(silhouette[cavity], 0.0)
        self.assertTrue(np.all(silhouette[rim] > 0.0))
        self.assertGreater(silhouette[rim].mean(), 0.5)

    def test_opposite_views_not_needed_and_parameters_checked(self):
        directions = perception._fibonacci_hemisphere(24)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
        self.assertTrue(np.all(directions[:, 2] > 0.0))
        vertices, faces = tetrahedron()
        for kwargs in ({'view_count': 0}, {'resolution': 4}, {'band_pixels': -1}, {'view_count': 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                outer_silhouette(vertices, faces, **kwargs)

    def test_open_boundary_counts_as_contour(self):
        # A flat square seen from above: its border is the outline of every view.
        x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
        vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(25)))
        faces = [(r * 5 + c, r * 5 + c + 1, r * 5 + c + 6) for r in range(4) for c in range(4)]
        faces += [(r * 5 + c, r * 5 + c + 6, r * 5 + c + 5) for r in range(4) for c in range(4)]
        silhouette = outer_silhouette(vertices, faces)
        border = (np.abs(vertices[:, 0]) == 1) | (np.abs(vertices[:, 1]) == 1)
        self.assertTrue(np.all(silhouette[border] > 0.0))
        np.testing.assert_array_equal(silhouette[~border], 0.0)

    def test_sphere_is_near_uniform_at_any_tessellation(self):
        # Every point of a sphere is on the outline equally often. A binary
        # contour test over a few fixed views gave 0 to 1 stripes (about a third
        # of icosphere(4) at 0, coefficient of variation 0.8).
        for name, (vertices, faces) in (('icosphere3', icosphere(3)), ('icosphere4', icosphere(4)),
                                        ('uv24x48', uv_sphere(24, 48))):
            with self.subTest(mesh=name):
                silhouette = outer_silhouette(vertices, faces)
                self.assertGreater(silhouette.min(), 0.5)
                self.assertLess(silhouette.std() / silhouette.mean(), 0.15)

    def test_rotation_changes_the_score_little(self):
        # The score must describe the shape, not where the fixed views fall on it.
        vertices, faces = icosphere(4)
        blob = bump(bump(vertices, axis=(0.3, -0.5, 0.8)), axis=(-0.7, 0.2, -0.1), height=0.5, width=0.4)
        cup = lathe([(0, 0), (1.0, 0), (1.0, 2.0), (0.7, 2.0), (0.7, 0.4), (0, 0.4)])
        for name, (points, triangles), axis, degrees in (
                ('blob', (blob * [1.4, 1.0, 0.8], faces), (1, 2, 3), 60), ('cup', cup, (1, 0, 0), 30)):
            with self.subTest(mesh=name):
                base = outer_silhouette(points, triangles)
                turned = outer_silhouette(points @ rotation(axis, degrees).T, triangles)
                change = np.abs(turned - base)
                self.assertLess(change.mean(), 0.1)
                self.assertLess(np.mean(change > 0.25), 0.03)

    def test_flat_faces_are_zero_and_creases_high(self):
        # A plane is on the outline only exactly edge-on; the soft contour
        # tolerance shrinks to 0 there instead of aliasing with the views.
        vertices, faces = subdivided_cube(6)
        silhouette = outer_silhouette(vertices, faces)
        on_edge = np.sum(np.isclose(np.abs(vertices), 1.0), axis=1) >= 2
        self.assertGreater(silhouette[on_edge].min(), 0.5)
        np.testing.assert_array_equal(silhouette[~on_edge], 0.0)

    def test_view_defaults_are_the_module_constants(self):
        # Tolerance = sin of the radius of a cap covering 1/24 of the hemisphere (16.6 degrees).
        self.assertAlmostEqual(math.degrees(math.asin(perception._contour_tolerance(24))), 16.6, places=1)
        for function, views, pixels in ((outer_silhouette, 'view_count', 'resolution'),
                                        (compute_importance, 'silhouette_views', 'silhouette_resolution')):
            parameters = inspect.signature(function).parameters
            self.assertEqual(parameters[views].default, DEFAULT_SILHOUETTE_VIEWS)
            self.assertEqual(parameters[pixels].default, DEFAULT_SILHOUETTE_RESOLUTION)
        self.assertIsInstance(DEFAULT_SILHOUETTE_VIEWS, int)
        self.assertIsInstance(DEFAULT_SILHOUETTE_RESOLUTION, int)


class SharpTests(unittest.TestCase):
    def test_cube_edges_are_sharp(self):
        vertices, faces = subdivided_cube(1)
        self.assertEqual((len(vertices), len(faces)), (8, 12))
        np.testing.assert_array_equal(sharp_features(vertices, faces), 1.0)
        np.testing.assert_array_equal(sharp_features(vertices, faces, sharp_angle=100.0), 0.0)

    def test_subdivided_cube_face_interiors_are_zero(self):
        vertices, faces = subdivided_cube(6)
        sharp = sharp_features(vertices, faces)
        on_edge = np.sum(np.isclose(np.abs(vertices), 1.0), axis=1) >= 2
        deep = np.all(np.isclose(np.abs(vertices), 1.0) | (np.abs(vertices) < 1.0 - 4.0 / 6 + 1e-9), axis=1) & ~on_edge
        near = ~on_edge & ~deep
        np.testing.assert_array_equal(sharp[on_edge], 1.0)
        np.testing.assert_array_equal(sharp[near], 0.5)
        np.testing.assert_array_equal(sharp[deep], 0.0)
        self.assertEqual(deep.sum(), 6 * 3 * 3)
        # Inconsistent winding on a flat region must not create creases.
        flipped = faces.copy()
        flipped[::3] = flipped[::3, ::-1]
        np.testing.assert_array_equal(sharp_features(vertices, flipped), sharp)

    def test_marked_sharp_and_open_boundary(self):
        vertices, faces = subdivided_cube(4)
        marked = np.zeros(len(vertices))
        centre = int(np.argmin(np.linalg.norm(vertices - [0.0, 0.0, 1.0], axis=1)))
        marked[centre] = 1.0
        sharp = sharp_features(vertices, faces, marked_sharp=marked)
        self.assertEqual(sharp[centre], 1.0)
        neighbours = np.unique(faces[np.any(faces == centre, axis=1)])
        self.assertTrue(np.all(sharp[neighbours] >= 0.5))
        # Remove one face: its edges become open boundary edges.
        open_faces = faces[np.any(faces == centre, axis=1) == False]  # noqa: E712
        boundary = sharp_features(vertices, open_faces, sharp_angle=180.0)
        np.testing.assert_array_equal(boundary[neighbours[neighbours != centre]], 1.0)


class CombinationTests(unittest.TestCase):
    def setUp(self):
        vertices, faces = icosphere(3)
        self.vertices, self.faces = bump(vertices), faces

    def importance(self, **kwargs):
        return compute_importance(self.vertices, self.faces, silhouette_resolution=128, **kwargs)

    def test_default_weights_and_renormalization(self):
        importance, features = self.importance()
        self.assertEqual(set(features), {'saliency', 'silhouette', 'sharp', 'visibility', 'manual'})
        expected = (DEFAULT_WEIGHTS['saliency'] * features['saliency']
                    + DEFAULT_WEIGHTS['silhouette'] * features['silhouette']
                    + DEFAULT_WEIGHTS['sharp'] * features['sharp']) / 1.0
        np.testing.assert_allclose(importance, expected, atol=1e-12)
        np.testing.assert_array_equal(features['visibility'], 0.0)
        importance, features = self.importance(weights={'saliency': 0.0, 'silhouette': 1.0, 'sharp': 3.0})
        np.testing.assert_array_equal(features['saliency'], 0.0)
        np.testing.assert_allclose(importance, (features['silhouette'] + 3.0 * features['sharp']) / 4.0)
        importance, features = self.importance(weights={'saliency': 2.0})
        np.testing.assert_allclose(importance, features['saliency'])

    def test_visibility_inactive_without_camera(self):
        importance, features = self.importance(weights={'visibility': 1.0})
        np.testing.assert_array_equal(importance, 0.0)
        np.testing.assert_array_equal(features['visibility'], 0.0)
        importance, _ = self.importance(weights={'saliency': 0.0, 'silhouette': 0.0, 'sharp': 0.0, 'visibility': 0.0})
        np.testing.assert_array_equal(importance, 0.0)

    def test_manual_combines_by_maximum(self):
        manual = np.zeros(len(self.vertices))
        manual[:10] = 1.0
        manual[10:20] = 2.0
        manual[20:30] = -1.0
        base, _ = self.importance()
        importance, features = self.importance(manual=manual)
        np.testing.assert_array_equal(importance[:20], 1.0)
        np.testing.assert_array_equal(importance[20:], base[20:])
        self.assertEqual(features['manual'].max(), 1.0)
        self.assertEqual(features['manual'].min(), 0.0)
        importance, _ = self.importance(weights={'sharp': 0.0, 'saliency': 0.0, 'silhouette': 0.0}, manual=manual)
        np.testing.assert_array_equal(importance, np.clip(manual, 0, 1))

    def test_validation(self):
        n = len(self.vertices)
        cases = [{'weights': {'curvature': 1.0}}, {'weights': {'saliency': -0.1}},
                 {'weights': {'sharp': float('nan')}}, {'weights': {'silhouette': float('inf')}},
                 {'weights': {'saliency': 'x'}}, {'manual': np.zeros(n - 1)},
                 {'manual': np.full(n, np.nan)}, {'marked_sharp': np.zeros((n, 2))},
                 {'sharp_angle': float('nan')}, {'sharp_angle': 200.0}, {'sharp_angle': 'x'},
                 {'weights': [('saliency', 1.0)]}, {'silhouette_views': 0}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.importance(**kwargs)
        bad_vertices = self.vertices.copy()
        bad_vertices[0, 0] = np.inf
        for vertices, faces in ((bad_vertices, self.faces), (self.vertices, [[0, 1, n]]),
                                (self.vertices, [[0.5, 1, 2]]), (self.vertices[:, :2], self.faces),
                                (self.vertices, [[0, 1]])):
            with self.assertRaises(ValueError):
                compute_importance(vertices, faces)

    def test_ranges_shapes_and_determinism(self):
        first, features = self.importance(marked_sharp=np.zeros(len(self.vertices)))
        second, again = self.importance(marked_sharp=np.zeros(len(self.vertices)))
        np.testing.assert_array_equal(first, second)
        for name, values in features.items():
            with self.subTest(name=name):
                self.assertEqual(values.shape, (len(self.vertices),))
                self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))
                np.testing.assert_array_equal(values, again[name])
        self.assertTrue(np.all((first >= 0.0) & (first <= 1.0)))
        timings = {}
        self.importance(timings=timings)
        self.assertTrue({'topology', 'saliency', 'silhouette', 'sharp', 'visibility'} <= set(timings))

    def test_camera_frustum_rejection_and_validation_without_blender(self):
        camera = {'type': 'PERSP', 'position': (0.0, -5.0, 0.0), 'forward': (0.0, -1.0, 0.0),
                  'up': (0.0, 0.0, 1.0), 'angle_x': 0.8, 'angle_y': 0.8, 'clip_start': 0.1, 'clip_end': 100.0}
        # Looking away: nothing is in the frustum, so no ray casting (no mathutils) is needed.
        np.testing.assert_array_equal(camera_visibility(self.vertices, self.faces, camera), 0.0)
        ortho = dict(camera, type='ORTHO', forward=(0.0, 1.0, 0.0), ortho_width=1.0, ortho_height=1.0,
                     clip_end=3.0)
        np.testing.assert_array_equal(camera_visibility(self.vertices, self.faces, ortho), 0.0)
        np.testing.assert_array_equal(camera_visibility(self.vertices, self.faces, None), 0.0)
        for broken in ({k: v for k, v in camera.items() if k != 'angle_x'}, dict(camera, type='PANO'),
                       dict(camera, forward=(0.0, 0.0, 1.0)), dict(camera, clip_end=0.05),
                       dict(camera, angle_y=4.0), dict(ortho, ortho_width=0.0),
                       dict(camera, position=(np.nan, 0.0, 0.0))):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                camera_visibility(self.vertices, self.faces, broken)


class TinyMeshTests(unittest.TestCase):
    def check(self, vertices, faces):
        importance, features = compute_importance(vertices, faces, manual=np.zeros(len(vertices)))
        self.assertEqual(importance.shape, (len(vertices),))
        for values in (importance, *features.values()):
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))
        return importance, features

    def test_four_triangle_meshes_do_not_crash(self):
        self.check(*tetrahedron())
        fan = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        self.check(fan, [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
        # Nonmanifold edge (0, 1) shared by three faces, a repeated-index face,
        # a zero-area face and an isolated vertex.
        vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0.5], [9, 9, 9], [2, 0, 0]])
        _, features = self.check(vertices, [[0, 1, 2], [0, 1, 3], [1, 0, 4], [2, 2, 3], [0, 1, 6]])
        for values in features.values():
            self.assertEqual(values[5], 0.0)
        self.check(vertices, np.zeros((0, 3), dtype=np.int64))

    def test_single_scale_and_nonmanifold_saliency(self):
        vertices, faces = tetrahedron()
        values = mesh_saliency(vertices, faces, scales=(3,))
        self.assertTrue(np.all(np.isfinite(values)))


if __name__ == '__main__':
    unittest.main()
