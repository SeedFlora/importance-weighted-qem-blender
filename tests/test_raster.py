"""Run: PYTHONPATH=. python tests/test_raster.py  (plain CPython + NumPy, no Blender)."""

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart_lod import raster


def v01_rasterize(triangles, lower, pixel_scale, resolution):
    """Frozen verbatim copy of v0.1 ``metrics._rasterize``: the equivalence oracle."""
    projected = (triangles - lower) * pixel_scale
    mask = np.zeros((resolution, resolution), dtype=bool)
    minimum = np.ceil(projected.min(axis=1) - 0.5).astype(np.int64)
    maximum = np.floor(projected.max(axis=1) - 0.5).astype(np.int64)
    minimum = np.maximum(minimum, 0)
    maximum = np.minimum(maximum, resolution - 1)
    # Cross(edge, point - edge_start) = a*x + b*y + c.
    edge_start = projected
    edge_end = np.roll(projected, -1, axis=1)
    delta = edge_end - edge_start
    a = -delta[:, :, 1]
    b = delta[:, :, 0]
    c = delta[:, :, 1] * edge_start[:, :, 0] - delta[:, :, 0] * edge_start[:, :, 1]
    signed_area = (
        (projected[:, 1, 0] - projected[:, 0, 0])
        * (projected[:, 2, 1] - projected[:, 0, 1])
        - (projected[:, 1, 1] - projected[:, 0, 1])
        * (projected[:, 2, 0] - projected[:, 0, 0])
    )
    sign = np.sign(signed_area)[:, None]
    a *= sign
    b *= sign
    c *= sign
    valid = np.all(maximum >= minimum, axis=1) & (np.abs(signed_area) > 1e-12)
    single = valid & np.all(maximum == minimum, axis=1)
    if single.any():
        xy = minimum[single] + 0.5
        inside = np.all(a[single] * xy[:, 0:1] + b[single] * xy[:, 1:2] + c[single] >= -1e-9, axis=1)
        hits = minimum[single][inside]
        mask[hits[:, 1], hits[:, 0]] = True
    coordinates = np.arange(resolution, dtype=np.float64) + 0.5
    for index in np.flatnonzero(valid & ~single):
        x0, y0 = minimum[index]
        x1, y1 = maximum[index] + 1
        tile = mask[y0:y1, x0:x1]
        if tile.all():
            continue
        x = coordinates[None, x0:x1]
        y = coordinates[y0:y1, None]
        covered = np.ones(tile.shape, dtype=bool)
        for edge in range(3):
            covered &= a[index, edge] * x + b[index, edge] * y + c[index, edge] >= -1e-9
        tile |= covered
    return mask


def random_soup(rng, count, resolution):
    """Mixed soup in pixel units: tiny, medium, huge, slivers, degenerate, off-frame."""
    parts = []
    centers = rng.uniform(-0.2 * resolution, 1.2 * resolution, (count, 1, 2))
    for scale in (0.3, 1.5, 4.0, 12.0, 0.25 * resolution, 1.5 * resolution):
        parts.append(centers[: count // 6] + rng.normal(0.0, scale, (count // 6, 3, 2)))
    sliver = centers[: count // 6].repeat(3, axis=1)
    sliver[:, 1] += rng.normal(0.0, 0.2 * resolution, (count // 6, 2))
    sliver[:, 2] = sliver[:, 1] + rng.normal(0.0, 1e-3, (count // 6, 2))
    parts.append(sliver)
    collinear = centers[: count // 12].repeat(3, axis=1)
    step = rng.normal(0.0, 5.0, (count // 12, 1, 2))
    collinear[:, 1] += step[:, 0]
    collinear[:, 2] += 2.0 * step[:, 0]
    parts.append(collinear)
    parts.append(centers[: count // 12].repeat(3, axis=1))  # all corners coincide
    # Corners exactly on pixel centres and edges exactly through pixel centres.
    lattice = np.floor(centers[: count // 6]) + 0.5
    lattice = lattice + rng.integers(-3, 4, (count // 6, 3, 2))
    parts.append(lattice)
    soup = np.concatenate(parts)
    flip = rng.random(len(soup)) < 0.5
    soup[flip] = soup[flip][:, ::-1]  # both windings
    return soup


def sphere(subdivisions=3):
    """Icosphere-like closed mesh built by midpoint subdivision (deterministic)."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    vertices = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
             [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
             [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    vertices = [np.array(v, float) / np.linalg.norm(v) for v in vertices]
    for _ in range(subdivisions):
        cache, next_faces = {}, []

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = vertices[i] + vertices[j]
                vertices.append(m / np.linalg.norm(m))
                cache[key] = len(vertices) - 1
            return cache[key]

        for i, j, k in faces:
            a, b, c = midpoint(i, j), midpoint(j, k), midpoint(k, i)
            next_faces += [[i, a, c], [j, b, a], [k, c, b], [a, b, c]]
        faces = next_faces
    return np.array(vertices), np.array(faces)


class RasterEquivalenceTests(unittest.TestCase):
    def assert_equivalent(self, triangles, lower, scale, resolution):
        expected = v01_rasterize(triangles, lower, scale, resolution)
        actual = raster.rasterize(triangles, lower, scale, resolution)
        self.assertEqual(actual.shape, (resolution, resolution))
        self.assertEqual(actual.dtype, np.bool_)
        self.assertTrue(np.array_equal(actual, expected),
                        f"{int(np.count_nonzero(actual != expected))} differing pixels")
        return actual

    def test_random_soups_match_v01(self):
        rng = np.random.default_rng(7)
        for resolution in (8, 32, 64, 128, 257, 512):
            for trial in range(3):
                with self.subTest(resolution=resolution, trial=trial):
                    soup = random_soup(rng, 600, resolution)
                    # Pixel units directly, plus a nontrivial lower/scale mapping.
                    self.assert_equivalent(soup, np.zeros(2), 1.0, resolution)
                    lower = np.array([-3.25, 11.0])
                    self.assert_equivalent(soup / 7.3 + lower, lower, 7.3, resolution)
                    # float32 input promotes exactly as in v0.1.
                    self.assert_equivalent(soup.astype(np.float32), np.zeros(2, np.float32), 1.0, resolution)

    def test_tile_size_boundaries_match_v01(self):
        # Boxes around the batched/loop split and power-of-two padding sizes.
        rng = np.random.default_rng(3)
        resolution = 300
        soups = []
        for width in (1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 63, 64, 65, 100, 250):
            for height in (1, 2, 3, 8, 33, 64, 65, 120):
                origin = rng.uniform(0, resolution - width - 2, (4, 2))
                corners = np.array([[0.1, 0.2], [width + 0.3, 0.6], [0.7, height + 0.4]])
                soups.append(origin[:, None, :] + corners[None])
                soups.append(origin[:, None, :] + corners[None, ::-1])
        self.assert_equivalent(np.concatenate(soups), np.zeros(2), 1.0, resolution)
        # Long axis-aligned slivers: area within the batched limit, one side huge.
        long = []
        for length in (255, 256, 257, 700, 1023, 1024, 1025, 2047):
            for thickness in (0.6, 1.2, 2.1):
                long.append([[1.2, 1.3], [1.2 + length, 1.4], [1.3, 1.3 + thickness]])
                long.append([[1.3, 1.2], [1.3 + thickness, 1.2], [1.4, 1.2 + length]])
        self.assert_equivalent(np.array(long), np.zeros(2), 1.0, 2100)
        self.assertEqual(raster._TILE_SIZES[0], 1)
        self.assertGreaterEqual(raster._TILE_SIZES[-1], raster._BATCH_MAX_PIXELS)

    def test_edge_tolerance_and_degenerate_threshold(self):
        # Needles along a row of pixel centres, isolated so no other triangle
        # covers the same pixels. Offsets give |signed area| on both sides of 1e-12.
        rows = []
        for row, offset in enumerate((1e-15, 1e-14, 5e-14, 1e-12, 1e-11, 1e-6)):
            for flip in (False, True):
                start = np.array([2.5 + (30 if flip else 0), 2.5 + 3 * row])
                needle = np.array([start, start + [20.0, 0.0], start + [20.0, offset]])
                rows.append(needle[::-1] if flip else needle)
        needles = np.array(rows)
        mask = self.assert_equivalent(needles, np.zeros(2), 1.0, 64)
        self.assertTrue(mask[2 + 3 * 4].any() and not mask[2].any())  # 1e-11 drawn, 1e-15 skipped
        # Isolated triangles with corners on pixel centres, so many centres lie on
        # edges. A 1e-12 px jitter moves those edge values just to either side of
        # zero, where only the -1e-9 rule decides coverage. Small triangles use the
        # batched path; the large ones (box above the batched limit) the loop path.
        rng = np.random.default_rng(5)
        cells = np.stack(np.meshgrid(np.arange(8), np.arange(8)), axis=-1).reshape(-1, 1, 2) * 12.0 + 0.5
        lattice = cells + rng.integers(0, 9, (len(cells), 3, 2))
        large = np.array([[[0.5, 0.5], [60.5, 0.5], [0.5, 60.5]], [[70.5, 0.5], [130.5, 60.5], [70.5, 60.5]],
                          [[0.5, 70.5], [61.5, 131.5], [45.5, 70.5]], [[70.5, 70.5], [131.5, 90.5], [80.5, 131.5]]])
        for name, soup, resolution in (("batched", lattice, 100), ("loop", large, 140)):
            for jitter in (0.0, 1e-12):
                jittered = soup + rng.normal(0.0, jitter, soup.shape)
                with self.subTest(path=name, jitter=jitter):
                    self.assert_equivalent(jittered, np.zeros(2), 1.0, resolution)
                    lower = np.array([0.123, -4.56])
                    self.assert_equivalent(jittered / 7.3 + lower, lower, 7.3, resolution)

    def test_empty_and_fully_degenerate_inputs(self):
        empty = np.zeros((0, 3, 2))
        self.assertFalse(self.assert_equivalent(empty, np.zeros(2), 1.0, 16).any())
        flat = np.array([[[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]], [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]])
        self.assertFalse(self.assert_equivalent(flat, np.zeros(2), 1.0, 16).any())
        outside = np.array([[[-9.0, -9.0], [-5.0, -9.0], [-5.0, -5.0]], [[40.0, 1.0], [50.0, 1.0], [45.0, 9.0]]])
        self.assertFalse(self.assert_equivalent(outside, np.zeros(2), 1.0, 16).any())

    def test_covering_triangle_and_pixel_centre_rule(self):
        big = np.array([[[-100.0, -100.0], [300.0, -100.0], [-100.0, 300.0]]])
        self.assertTrue(self.assert_equivalent(big, np.zeros(2), 1.0, 32).all())
        # Right triangle with legs through pixel centres: centres on edges count.
        edge = np.array([[[0.5, 0.5], [3.5, 0.5], [0.5, 3.5]]])
        mask = self.assert_equivalent(edge, np.zeros(2), 1.0, 8)
        self.assertEqual(int(mask.sum()), 10)

    def test_sphere_projections_match_v01(self):
        vertices, faces = sphere(4)  # 5,120 triangles
        for resolution in (32, 128, 512):
            for direction in ((1, 0, 0), (1, 1, 1), (0, 1, -1), (0.3, -0.2, 0.9)):
                with self.subTest(resolution=resolution, direction=direction):
                    u, w = raster.orthographic_basis(direction)
                    points = np.column_stack((vertices @ u, vertices @ w))[faces]
                    lower = points.min(axis=(0, 1))
                    side = float(np.max(points.max(axis=(0, 1)) - lower)) * 1.04
                    mask = self.assert_equivalent(points, lower - 0.02 * side, resolution / side, resolution)
                    self.assertGreater(mask.mean(), 0.6)

    def test_bunny_projections_match_v01_if_available(self):
        """Optional: set SMART_LOD_BUNNY to a Stanford bunny OBJ path to run."""
        import os
        path = os.environ.get("SMART_LOD_BUNNY")
        if not path:
            self.skipTest("SMART_LOD_BUNNY not set")
        vertices, faces = [], []
        with open(path) as handle:
            for line in handle:
                if line.startswith("v "):
                    vertices.append([float(x) for x in line.split()[1:4]])
                elif line.startswith("f "):
                    faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
        vertices, faces = np.array(vertices), np.array(faces)
        for resolution in (32, 64, 128, 256, 512):
            for direction in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (1, -1, 0), (0, 1, -1)):
                with self.subTest(resolution=resolution, direction=direction):
                    u, w = raster.orthographic_basis(direction)
                    points = np.column_stack((vertices @ u, vertices @ w))[faces]
                    low, high = points.min(axis=(0, 1)), points.max(axis=(0, 1))
                    side = float(np.max(high - low)) * 1.04
                    self.assert_equivalent(points, (low + high) * 0.5 - side * 0.5, resolution / side, resolution)


class CanvasTests(unittest.TestCase):
    """``canvas`` selects another pixel index range of the same lattice (reference-anchored framing)."""

    def test_pixels_inside_the_default_canvas_are_bitwise_unchanged(self):
        rng = np.random.default_rng(13)
        for resolution in (8, 64, 257):
            soup = random_soup(rng, 600, resolution)
            lower = np.array([-3.25, 11.0])
            soup = soup / 7.3 + lower
            default = raster.rasterize(soup, lower, 7.3, resolution)
            r = resolution
            for x0, y0, x1, y1 in ((0, 0, r, r), (-7, -3, r + 5, r + 11), (-r, -r, 2 * r, 2 * r), (-1, 0, r, 3 * r)):
                with self.subTest(resolution=resolution, canvas=(x0, y0, x1, y1)):
                    mask = raster.rasterize(soup, lower, 7.3, resolution, (x0, y0, x1, y1))
                    self.assertEqual(mask.shape, (y1 - y0, x1 - x0))
                    self.assertTrue(np.array_equal(mask[-y0:resolution - y0, -x0:resolution - x0], default))

    def test_extra_pixels_match_v01_on_an_exactly_shifted_soup(self):
        # Dyadic coordinates make the integer shift and every edge function exact,
        # so shifting the soup and shifting the canvas must agree pixel for pixel.
        rng = np.random.default_rng(17)
        for resolution in (16, 64, 128):
            soup = np.round(random_soup(rng, 600, resolution) * 1024.0) / 1024.0
            r = resolution
            for x0, y0, x1, y1 in ((-9, -4, r + 6, r + 13), (-r, 3, r, 2 * r)):
                with self.subTest(resolution=resolution, canvas=(x0, y0, x1, y1)):
                    mask = raster.rasterize(soup, np.zeros(2), 1.0, resolution, (x0, y0, x1, y1))
                    side = max(x1 - x0, y1 - y0)
                    expected = v01_rasterize(soup - [x0, y0], np.zeros(2), 1.0, side)[:y1 - y0, :x1 - x0]
                    self.assertTrue(np.array_equal(mask, expected),
                                    f"{int(np.count_nonzero(mask != expected))} differing pixels")

    def test_far_coordinates_are_clamped_before_integer_conversion(self):
        # A huge triangle covering the canvas; v0.1 cast its bounds to int64
        # without clamping, which overflows and dropped it.
        huge = np.array([[[-1e30, -1e30], [1e30, -1e30], [0.0, 1e30]]])
        far = np.array([[[1e30, 1e30], [2e30, 1e30], [1e30, 2e30]]])
        with np.errstate(all="raise"):
            self.assertTrue(raster.rasterize(huge, np.zeros(2), 1.0, 16).all())
            self.assertTrue(raster.rasterize(huge, np.zeros(2), 1.0, 16, (-4, -4, 20, 12)).all())
            self.assertFalse(raster.rasterize(far, np.zeros(2), 1.0, 16, (-4, -4, 20, 12)).any())

    def test_invalid_canvas(self):
        for canvas in ((0, 0, 0, 8), (0, 0, 8, -1), (5, 5, 4, 9)):
            with self.subTest(canvas=canvas), self.assertRaises(ValueError):
                raster.rasterize(np.zeros((1, 3, 2)), np.zeros(2), 1.0, 8, canvas)


class OrthographicBasisTests(unittest.TestCase):
    def test_orthonormal_and_deterministic(self):
        rng = np.random.default_rng(11)
        directions = list(rng.normal(size=(200, 3))) + [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, -1), (1, 1, 1), (0.9, 0.1, 0), (1e-8, 0, 1)]
        for direction in directions:
            d = np.asarray(direction, float) / np.linalg.norm(direction)
            u, w = raster.orthographic_basis(direction)
            again = raster.orthographic_basis(direction)
            self.assertTrue(np.array_equal(u, again[0]) and np.array_equal(w, again[1]))
            frame = np.array([u, w, d])
            self.assertTrue(np.allclose(frame @ frame.T, np.eye(3), atol=1e-12))
            # Right-handed image frame: u x w points along the view direction.
            self.assertTrue(np.allclose(np.cross(u, w), d, atol=1e-12))

    def test_scale_invariant_and_opposite_direction_mirrors(self):
        u, w = raster.orthographic_basis((1, 2, 3))
        u2, w2 = raster.orthographic_basis((10, 20, 30))
        self.assertTrue(np.allclose(u, u2, atol=1e-15) and np.allclose(w, w2, atol=1e-15))
        un, wn = raster.orthographic_basis((-1, -2, -3))
        # Same image plane; the second axis flips, so the silhouette is mirrored.
        self.assertTrue(np.allclose(np.abs(np.array([un, wn]) @ np.array([u, w]).T), np.eye(2), atol=1e-12))

    def test_invalid_direction(self):
        for bad in ((0, 0, 0), (np.nan, 0, 1), (np.inf, 0, 0)):
            with self.subTest(direction=bad), self.assertRaises(ValueError):
                raster.orthographic_basis(bad)


if __name__ == "__main__":
    # Explicit argv so this also runs via ``blender --background --python``.
    result = unittest.main(argv=[sys.argv[0]], exit=False, verbosity=2).result
    sys.exit(0 if result.wasSuccessful() else 1)
