"""Run: blender --background --factory-startup --python tests/test_metrics_blender.py"""

import json
from pathlib import Path
import sys
import unittest

import numpy as np


# The package __init__ imports Blender UI lazily, so this loads only metrics.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart_lod import metrics


def cube():
    vertices = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    return vertices, faces


class CubeCase(unittest.TestCase):
    """Shared unit-cube fixture; defines no tests itself."""

    def setUp(self):
        self.vertices, self.faces = cube()

    def evaluate(self, vertices=None, faces=None, **kwargs):
        return metrics.evaluate_meshes(
            self.vertices, self.faces,
            self.vertices if vertices is None else vertices,
            self.faces if faces is None else faces,
            sample_count=kwargs.pop("sample_count", 1200),
            resolution=kwargs.pop("resolution", 128),
            **kwargs,
        )


class MetricsTests(CubeCase):
    def test_identity_queries_surface_not_other_sample_cloud(self):
        result = self.evaluate()
        self.assertLess(result["chamfer_mean"], 1e-6)
        self.assertLess(result["hausdorff_sampled"], 1e-6)
        self.assertEqual(result["silhouette_iou_mean"], 1.0)
        self.assertEqual(result["polygon_reduction_ratio"], 0.0)
        self.assertEqual(result["triangles_reference"], 12)
        self.assertEqual(result["sampling"]["samples_per_mesh"], 1200)
        json.dumps(result, allow_nan=False)

    def test_disjoint_translation_and_shared_projection_frame(self):
        result = self.evaluate(self.vertices + [2.0, 0.0, 0.0])
        self.assertGreater(result["chamfer_mean"], 1.0)
        self.assertLess(result["chamfer_mean"], 2.0)
        self.assertAlmostEqual(result["hausdorff_sampled"], 2.0, delta=1e-5)
        self.assertEqual(result["silhouette_iou_views"]["+X"], 1.0)
        self.assertEqual(result["silhouette_iou_views"]["+Y"], 0.0)
        self.assertAlmostEqual(result["silhouette_iou_mean"], 1 / 3)

    def test_thinner_geometry_has_expected_axis_silhouettes(self):
        result = self.evaluate(self.vertices * [1.0, 1.0, 0.5])
        self.assertGreater(result["chamfer_mean"], 0.03)
        self.assertAlmostEqual(result["hausdorff_sampled"], 0.25, delta=1e-5)
        self.assertAlmostEqual(result["silhouette_iou_views"]["+X"], 0.5, delta=0.025)
        self.assertEqual(result["silhouette_iou_views"]["+Z"], 1.0)
        self.assertAlmostEqual(result["silhouette_iou_mean"], 2 / 3, delta=0.02)

    def test_repeatability_and_similarity_transform_normalization(self):
        candidate = self.vertices * [1.0, 1.0, 0.5]
        initial = self.evaluate(candidate, seed=17)
        self.assertEqual(initial, self.evaluate(candidate, seed=17))
        transformed = metrics.evaluate_meshes(
            self.vertices * 25 + [1e6, -1e6, 2e6], self.faces,
            candidate * 25 + [1e6, -1e6, 2e6], self.faces,
            sample_count=1200, resolution=128, seed=17,
        )
        self.assertAlmostEqual(initial["chamfer_mean_normalized"], transformed["chamfer_mean_normalized"], places=7)
        self.assertAlmostEqual(initial["chamfer_mean"] * 25, transformed["chamfer_mean"], places=5)
        self.assertEqual(initial["silhouette_iou_views"], transformed["silhouette_iou_views"])

    def test_surface_sampling_is_area_uniform(self):
        # Areas are 0.5 and 2, so the second triangle must receive about 80%.
        triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[10, 0, 0], [12, 0, 0], [10, 2, 0]]], dtype=float)
        samples = metrics._sample_surface(triangles, np.array([0.2, 0.8]), 20000, np.random.default_rng(42))
        self.assertAlmostEqual(float(np.mean(samples[:, 0] > 5)), 0.8, delta=0.01)
        first = samples[samples[:, 0] < 5]
        self.assertTrue(np.all(first[:, 0] + first[:, 1] <= 1.0 + 1e-12))
        self.assertAlmostEqual(float(first[:, 0].mean()), 1 / 3, delta=0.015)

    def test_mixed_degenerate_triangles_are_recorded(self):
        result = self.evaluate(faces=np.vstack([self.faces, [0, 0, 1]]))
        self.assertEqual(result["degenerate_triangles_candidate"], 1)
        self.assertEqual(result["triangles_candidate"], 13)
        self.assertLess(result["chamfer_mean"], 1e-6)

    def test_invalid_geometry_and_parameters(self):
        bad = self.vertices.copy()
        bad[0, 0] = np.nan
        cases = [
            {"vertices": bad}, {"faces": [[0, 1, 99]]}, {"faces": [[0.5, 1, 2]]},
            {"faces": [[0, 0, 0]]}, {"sample_count": 0}, {"resolution": 2}, {"seed": -1},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.evaluate(**case)

    def test_planar_identity_handles_empty_edge_on_silhouette(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        result = metrics.evaluate_meshes(vertices, [[0, 1, 2]], vertices, [[2, 1, 0]], sample_count=100)
        self.assertEqual(result["silhouette_iou_mean"], 1.0)
        self.assertLess(result["chamfer_mean"], 1e-6)


MULTIVIEW_LABELS = ["+X", "+Y", "+Z", "+X+Y+Z", "+X+Y-Z", "+X-Y+Z", "-X+Y+Z",
                    "+X+Y", "+X-Y", "+X+Z", "+X-Z", "+Y+Z", "+Y-Z"]
ROI_DIRECTIONS = ("ref_to_candidate", "candidate_to_ref", "symmetric")
ROI_DISTANCE_KEYS = [f"{region}_{kind}_{direction}{suffix}" for region in ("roi", "nonroi")
                     for kind in ("chamfer", "hausdorff") for direction in ROI_DIRECTIONS
                     for suffix in ("", "_normalized")]


def grid(count=21):
    """Flat (count x count)-vertex grid on [-1, 1]^2 at z = 0."""
    x, y = np.meshgrid(np.linspace(-1, 1, count), np.linspace(-1, 1, count))
    vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    i, j = np.meshgrid(np.arange(count - 1), np.arange(count - 1), indexing="ij")
    a = (i * count + j).ravel()
    faces = np.concatenate([np.column_stack((a, a + 1, a + count + 1)), np.column_stack((a, a + count + 1, a + count))])
    return vertices, faces


def assert_json_safe(test, value):
    json.dumps(value, allow_nan=False)
    if isinstance(value, dict):
        for item in value.values():
            assert_json_safe(test, item)
    elif isinstance(value, list):
        for item in value:
            assert_json_safe(test, item)
    else:
        test.assertIsInstance(value, (bool, int, float, str, type(None)))


class MultiviewTests(CubeCase):
    """Oblique silhouettes; each label also stands for its mirror-identical opposite."""

    def test_identity_is_one_in_all_views_and_directions_are_recorded(self):
        result = self.evaluate()
        self.assertEqual(list(result["silhouette_iou_multiview_views"]), MULTIVIEW_LABELS)
        self.assertEqual(set(result["silhouette_iou_multiview_views"].values()), {1.0})
        self.assertEqual(result["silhouette_iou_multiview_mean"], 1.0)
        directions = result["sampling"]["silhouette_multiview_directions"]
        self.assertEqual(list(directions), MULTIVIEW_LABELS)
        self.assertTrue(np.allclose(np.linalg.norm(list(directions.values()), axis=1), 1.0))
        self.assertTrue(result["sampling"]["silhouette_multiview"])

    def test_axis_entries_equal_axis_views_exactly(self):
        rng = np.random.default_rng(3)
        for candidate in (self.vertices * [1.0, 1.0, 0.5], self.vertices + [0.3, -0.2, 0.1],
                          self.vertices + rng.normal(0.0, 0.08, self.vertices.shape)):
            result = self.evaluate(candidate, resolution=96)
            for label in ("+X", "+Y", "+Z"):
                self.assertEqual(result["silhouette_iou_multiview_views"][label], result["silhouette_iou_views"][label])
            views = list(result["silhouette_iou_multiview_views"].values())
            self.assertEqual(result["silhouette_iou_multiview_mean"], float(np.mean(views)))

    def test_oblique_thin_geometry_follows_projected_area_ratio(self):
        # Nested convex boxes: IoU = projected-area ratio
        # (0.5|nx| + 0.5|ny| + |nz|) / (|nx| + |ny| + |nz|).
        result = self.evaluate(self.vertices * [1.0, 1.0, 0.5], resolution=256)
        views = result["silhouette_iou_multiview_views"]
        for label, expected in [("+X", 0.5), ("+Y", 0.5), ("+Z", 1.0), ("+X+Y", 0.5), ("+X-Y", 0.5)] + [
                (label, 2 / 3) for label in ("+X+Y+Z", "+X+Y-Z", "+X-Y+Z", "-X+Y+Z")] + [
                (label, 0.75) for label in ("+X+Z", "+X-Z", "+Y+Z", "+Y-Z")]:
            self.assertAlmostEqual(views[label], expected, delta=0.02, msg=label)
        # Trend: views containing the thinned axis in the image plane lose most.
        self.assertLess(max(views["+X"], views["+X+Y"]), min(views["+X+Y+Z"], views["-X+Y+Z"]))
        self.assertLess(max(views["+X+Y+Z"], views["+X-Y+Z"]), min(views["+X+Z"], views["+Y-Z"]))
        self.assertAlmostEqual(result["silhouette_iou_multiview_mean"], 2 / 3, delta=0.02)

    def test_oblique_views_detect_shape_hidden_from_axis_views(self):
        # A tetrahedron on alternate cube corners projects to the full square
        # along every axis, but to half the cube's area along the diagonals.
        tetrahedron = np.array([[0, 2, 5], [0, 5, 7], [0, 7, 2], [2, 7, 5]])
        result = self.evaluate(faces=tetrahedron, resolution=256)
        self.assertAlmostEqual(result["silhouette_iou_mean"], 1.0, delta=0.01)
        oblique = [value for label, value in result["silhouette_iou_multiview_views"].items() if len(label) > 2]
        self.assertEqual(len(oblique), 10)
        for value in oblique:
            self.assertAlmostEqual(value, 0.5, delta=0.02)
        self.assertAlmostEqual(result["silhouette_iou_multiview_mean"], 8 / 13, delta=0.02)

    def test_multiview_can_be_disabled(self):
        result = self.evaluate(self.vertices * [1.0, 1.0, 0.5], multiview=False)
        self.assertNotIn("silhouette_iou_multiview_mean", result)
        self.assertNotIn("silhouette_iou_multiview_views", result)
        self.assertFalse(result["sampling"]["silhouette_multiview"])


class RoiTests(CubeCase):
    def test_identity_is_zero_with_expected_region(self):
        weights = (self.vertices[:, 2] > 0).astype(float)
        result = self.evaluate(roi_weights=weights)
        # Top face (2 triangles) plus one triangle per side with two top corners.
        self.assertEqual(result["roi_triangles"], 6)
        self.assertAlmostEqual(result["roi_area_fraction"], 0.5, places=12)
        for key in ROI_DISTANCE_KEYS:
            self.assertLess(result[key], 1e-6, msg=key)
        self.assertEqual(result["sampling"]["roi_notes"], [])
        self.assertEqual(result["sampling"]["roi_samples_per_region"], 1200)
        # The pre-existing key keeps its value; it describes the *_ref_to_candidate keys.
        self.assertIn("reference_to_candidate_only", result["sampling"]["roi_distance_direction"])
        self.assertIn("nearest", result["sampling"]["roi_candidate_to_ref_assignment"])
        # Global candidate samples split by nearest reference face: about the area share.
        self.assertEqual(result["roi_candidate_samples"] + result["nonroi_candidate_samples"], 1200)
        self.assertAlmostEqual(result["roi_candidate_samples"] / 1200, result["roi_area_fraction"], delta=0.06)

    def test_error_is_localized_to_the_region(self):
        vertices, faces = grid()
        radius = np.linalg.norm(vertices[:, :2], axis=1)
        moved = radius < 0.35
        candidate = vertices.copy()
        candidate[moved, 2] += 0.1 * (1.0 - (radius[moved] / 0.35) ** 2)
        # Threshold below 1/3: every triangle touching a moved vertex is in the ROI,
        # so the complement is geometrically unchanged.
        result = metrics.evaluate_meshes(vertices, faces, candidate, faces, sample_count=1500, resolution=64,
                                         roi_weights=moved.astype(float), roi_threshold=0.3)
        for direction in ROI_DIRECTIONS:
            self.assertGreater(result[f"roi_chamfer_{direction}_normalized"], 1e-3, msg=direction)
            self.assertGreater(result[f"roi_hausdorff_{direction}"], 0.05, msg=direction)
            self.assertLess(result[f"nonroi_hausdorff_{direction}"], 1e-6, msg=direction)
            self.assertLess(result[f"nonroi_chamfer_{direction}"], 1e-6, msg=direction)
        self.assertGreater(result["roi_area_fraction"], 0.0)
        self.assertLess(result["roi_area_fraction"], 0.2)
        diagonal = result["reference_bbox_diagonal"]
        self.assertAlmostEqual(result["roi_chamfer_ref_to_candidate"],
                               result["roi_chamfer_ref_to_candidate_normalized"] * diagonal, places=12)

    def test_empty_regions_report_none_not_zero(self):
        # (weights, threshold, region expected to be empty); threshold 0 puts all in the ROI.
        for weights, threshold, empty in (
                (np.zeros(8), 0.5, "roi"), (np.ones(8), 0.5, "nonroi"), (np.zeros(8), 0.0, "nonroi")):
            with self.subTest(weights=weights.tolist(), threshold=threshold):
                result = self.evaluate(self.vertices * [1.0, 1.0, 0.5], roi_weights=weights, roi_threshold=threshold)
                for key in ROI_DISTANCE_KEYS:
                    if key.startswith(empty + "_"):
                        self.assertIsNone(result[key], msg=key)
                    else:
                        self.assertGreater(result[key], 0.0, msg=key)
                self.assertEqual(result["roi_triangles"], 0 if empty == "roi" else 12)
                self.assertEqual(result["roi_area_fraction"], 0.0 if empty == "roi" else 1.0)
                self.assertEqual(result["sampling"]["roi_notes"], [f"{empty}_region_empty_distances_are_none"])
                self.assertEqual(result[f"{empty}_candidate_samples"], 0)
                self.assertEqual(result[f"{'nonroi' if empty == 'roi' else 'roi'}_candidate_samples"], 1200)
                assert_json_safe(self, result)

    def test_candidate_side_spike_inside_roi_is_measured(self):
        # Finding #19: a spike on the candidate inside the ROI hardly moves the
        # reference -> candidate ROI distance; candidate -> reference sees it.
        vertices, faces = grid()
        radius = np.linalg.norm(vertices[:, :2], axis=1)
        spiked = vertices.copy()
        spiked[np.argmin(radius), 2] = 0.5
        result = metrics.evaluate_meshes(vertices, faces, spiked, faces, sample_count=1200, resolution=64,
                                         roi_weights=(radius < 0.35).astype(float), roi_threshold=0.3)
        forward, backward = result["roi_chamfer_ref_to_candidate"], result["roi_chamfer_candidate_to_ref"]
        self.assertGreater(backward, 5.0 * forward)
        self.assertGreater(result["roi_hausdorff_candidate_to_ref"], 5.0 * result["roi_hausdorff_ref_to_candidate"])
        for suffix in ("", "_normalized"):
            self.assertEqual(result[f"roi_chamfer_symmetric{suffix}"], 0.5 * (
                result[f"roi_chamfer_ref_to_candidate{suffix}"] + result[f"roi_chamfer_candidate_to_ref{suffix}"]))
            self.assertEqual(result[f"roi_hausdorff_symmetric{suffix}"], max(
                result[f"roi_hausdorff_ref_to_candidate{suffix}"], result[f"roi_hausdorff_candidate_to_ref{suffix}"]))
        # The spike's footprint lies inside the ROI, so the complement is unaffected.
        for direction in ROI_DIRECTIONS:
            self.assertLess(result[f"nonroi_hausdorff_{direction}"], 1e-6, msg=direction)
        self.assertGreater(result["roi_candidate_samples"], 0)
        self.assertEqual(result["roi_candidate_samples"] + result["nonroi_candidate_samples"], 1200)

    def test_region_without_candidate_samples_reports_none(self):
        # The candidate is a small patch at the opposite corner of the ROI, so
        # no candidate sample has an ROI triangle as its nearest reference face.
        vertices, faces = grid()
        weights = ((vertices[:, 0] > 0.75) & (vertices[:, 1] > 0.75)).astype(float)
        patch = np.array([[-1.0, -1.0, 0.0], [-0.7, -1.0, 0.0], [-1.0, -0.7, 0.0]])
        result = metrics.evaluate_meshes(vertices, faces, patch, [[0, 1, 2]], sample_count=500, resolution=32,
                                         roi_weights=weights)
        self.assertGreater(result["roi_triangles"], 0)
        self.assertEqual(result["roi_candidate_samples"], 0)
        self.assertEqual(result["nonroi_candidate_samples"], 500)
        self.assertGreater(result["roi_chamfer_ref_to_candidate"], 1.0)
        for key in ROI_DISTANCE_KEYS:
            if key.startswith("roi_") and "ref_to_candidate" not in key:
                self.assertIsNone(result[key], msg=key)
            else:
                self.assertIsNotNone(result[key], msg=key)
        self.assertEqual(result["sampling"]["roi_notes"],
                         ["roi_region_has_no_candidate_samples_candidate_to_ref_and_symmetric_are_none"])
        assert_json_safe(self, result)

    def test_invalid_roi_and_option_inputs(self):
        cases = [
            {"roi_weights": np.ones(7)}, {"roi_weights": np.ones((8, 1))}, {"roi_weights": np.full(8, np.nan)},
            {"roi_weights": np.r_[np.ones(7), np.inf]}, {"roi_weights": np.r_[np.ones(7), -0.1]},
            {"roi_weights": np.r_[np.ones(7), 1.5]}, {"roi_weights": ["a"] * 8}, {"roi_weights": [None] * 8},
            {"roi_weights": 0.5}, {"roi_weights": np.ones(8), "roi_threshold": np.nan},
            {"roi_weights": np.ones(8), "roi_threshold": -0.1}, {"roi_weights": np.ones(8), "roi_threshold": 1.1},
            {"roi_weights": np.ones(8), "roi_threshold": True}, {"roi_weights": np.ones(8), "roi_threshold": "0.5"},
            {"roi_threshold": None}, {"multiview": "yes"}, {"multiview": 1},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.evaluate(**case)


class VertexHausdorffTests(CubeCase):
    """Finding #20: ``hausdorff_with_vertices`` also queries every surface vertex."""

    def test_identity_and_thin_box_are_exact(self):
        self.assertLess(self.evaluate()["hausdorff_with_vertices"], 1e-6)
        result = self.evaluate(self.vertices * [1.0, 1.0, 0.5])
        self.assertAlmostEqual(result["hausdorff_with_vertices"], 0.25, delta=1e-5)
        self.assertGreaterEqual(result["hausdorff_with_vertices"], result["hausdorff_sampled"])
        diagonal = result["reference_bbox_diagonal"]
        self.assertAlmostEqual(result["hausdorff_with_vertices"],
                               result["hausdorff_with_vertices_normalized"] * diagonal, places=12)
        self.assertIn("every_vertex", result["sampling"]["hausdorff_with_vertices_definition"])

    def test_spike_tip_on_either_mesh_is_measured(self):
        vertices, faces = grid()
        spiked = vertices.copy()
        spiked[np.argmin(np.linalg.norm(vertices[:, :2], axis=1)), 2] = 0.5
        for reference, candidate in ((vertices, spiked), (spiked, vertices)):
            result = metrics.evaluate_meshes(reference, faces, candidate, faces, sample_count=1200, resolution=32)
            self.assertAlmostEqual(result["hausdorff_with_vertices"], 0.5, delta=1e-5)
            # Area samples thin out toward the apex and miss the tip.
            self.assertLess(result["hausdorff_sampled"], result["hausdorff_with_vertices"] - 1e-3)

    def test_detached_speck_is_measured_but_unused_vertices_are_not(self):
        size = 1e-6
        speck = np.array([[0.0, 0.0, 1.5], [size, 0.0, 1.5], [0.0, size, 1.5]])
        result = self.evaluate(np.vstack([self.vertices, speck]), np.vstack([self.faces, [8, 9, 10]]))
        self.assertAlmostEqual(result["hausdorff_with_vertices"], 1.0, delta=1e-5)
        self.assertLess(result["hausdorff_sampled"], 1e-5)  # the speck is never sampled
        # A far vertex that is unused, or used only by a zero-area triangle, is not surface.
        far = np.vstack([self.vertices, [[9.0, 9.0, 9.0]]])
        for faces in (self.faces, np.vstack([self.faces, [8, 8, 0]])):
            with self.subTest(faces=len(faces)):
                self.assertLess(self.evaluate(far, faces)["hausdorff_with_vertices"], 1e-6)


def v01_frame_iou(reference, candidate, resolution):
    """Frozen copy of the v0.1 (and v0.2.0) union-frame ``_frame_iou``: the framing oracle."""
    lower = np.minimum(reference.min(axis=(0, 1)), candidate.min(axis=(0, 1)))
    upper = np.maximum(reference.max(axis=(0, 1)), candidate.max(axis=(0, 1)))
    side = float(np.max(upper - lower)) * 1.04
    frame_lower = (lower + upper) * 0.5 - side * 0.5
    reference_mask = metrics._rasterize(reference, frame_lower, resolution / side, resolution)
    candidate_mask = metrics._rasterize(candidate, frame_lower, resolution / side, resolution)
    union = int(np.count_nonzero(reference_mask | candidate_mask))
    intersection = int(np.count_nonzero(reference_mask & candidate_mask))
    return float(intersection / union) if union else 1.0


def projected_soups(reference_vertices, vertices, faces):
    """Per multiview label, the (M, 3, 2) projected soup exactly as ``evaluate_meshes`` builds it."""
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    low = reference_vertices.min(axis=0)
    extent = reference_vertices.max(axis=0) - low
    normalized = (np.asarray(vertices, dtype=np.float64) - (low + extent * 0.5)) / float(np.linalg.norm(extent))
    soups = {}
    for label, direction in metrics._MULTIVIEW_DIRECTIONS:
        if np.count_nonzero(direction) == 1:  # axis view: drop the viewing axis
            soups[label] = normalized[faces][:, :, [k for k in range(3) if not direction[k]]]
        else:
            u, w = metrics.orthographic_basis(direction)
            soups[label] = metrics._project(normalized, u, w)[faces]
    return soups


class FramingTests(CubeCase):
    """Finding #18: the silhouette grid is fixed by the reference, not by each candidate.

    v0.1 framed each view on the union of both projected bounds. The values are
    unchanged when the candidate lies within the reference's projected bounds
    (then the union is the reference) and intentionally differ otherwise.
    """

    def views(self, result):
        return dict(result["silhouette_iou_multiview_views"])

    def test_equals_v01_when_candidate_within_reference_bounds(self):
        rng = np.random.default_rng(4)
        tetrahedron = np.array([[0, 2, 5], [0, 5, 7], [0, 7, 2], [2, 7, 5]])
        # Points inside the convex reference project inside it in every view.
        for vertices, faces in ((self.vertices * [1.0, 1.0, 0.5], self.faces), (self.vertices * 0.8, self.faces),
                                (self.vertices * rng.uniform(0.6, 1.0, (8, 3)), self.faces),
                                (self.vertices, tetrahedron)):
            result = self.evaluate(vertices, faces)
            reference = projected_soups(self.vertices, self.vertices, self.faces)
            candidate = projected_soups(self.vertices, vertices, faces)
            for label, value in self.views(result).items():
                self.assertEqual(value, v01_frame_iou(reference[label], candidate[label], 128), msg=label)
            self.assertEqual(result["silhouette_frame_clipped_views"], [])
            self.assertEqual(result["silhouette_multiview_frame_clipped_views"], [])

    def test_every_candidate_is_rasterized_on_the_reference_grid(self):
        calls, original = [], metrics._rasterize

        def recording(triangles, lower, pixel_scale, resolution, canvas=None):
            calls.append((np.asarray(lower).tolist(), pixel_scale, resolution, canvas))
            return original(triangles, lower, pixel_scale, resolution, canvas)

        grids, canvases = [], []
        metrics._rasterize = recording
        try:
            for candidate in (self.vertices * 0.8, self.vertices * [1.3, 1.0, 1.0], self.vertices + [0.2, 0.0, 0.0]):
                calls.clear()
                self.evaluate(candidate)
                self.assertEqual(len(calls), 26)  # 13 views x (reference, candidate)
                grids.append([call[:3] for call in calls])
                canvases.append([call[3] for call in calls])
        finally:
            metrics._rasterize = original
        for grid_calls in grids:
            self.assertEqual(grid_calls[0::2], grid_calls[1::2])  # candidate uses the reference grid
        self.assertEqual(grids[0], grids[1])  # and that grid does not depend on the candidate
        self.assertEqual(grids[0], grids[2])
        self.assertEqual(canvases[0][0::2], [None] * 13)
        self.assertEqual(canvases[0][1::2], [(0, 0, 128, 128)] * 13)
        # The 30% wider box overhangs the 2% margin in x only: +Y (x, z) grows on both sides of x.
        x0, y0, x1, y1 = canvases[1][3]
        self.assertTrue(x0 < 0 and x1 > 128 and (y0, y1) == (0, 128) and abs(x0 + (x1 - 128)) <= 1)

    def test_invisible_far_geometry_does_not_change_iou(self):
        # A triangle too small to cover a pixel centre, beyond the reference.
        # v0.1 rescaled the whole frame to include it; now it only adds empty pixels.
        thin = self.vertices * [1.0, 1.0, 0.5]
        base = self.evaluate(thin)
        speck = np.array([[1.0, 0.0, 0.0], [1.0 + 1e-6, 0.0, 0.0], [1.0, 1e-6, 0.0]])
        vertices, faces = np.vstack([thin, speck]), np.vstack([self.faces, [8, 9, 10]])
        with_speck = self.evaluate(vertices, faces)
        self.assertEqual(self.views(with_speck), self.views(base))
        self.assertEqual(with_speck["silhouette_iou_views"], base["silhouette_iou_views"])
        self.assertEqual(with_speck["silhouette_frame_clipped_views"], [])
        reference = projected_soups(self.vertices, self.vertices, self.faces)
        self.assertNotEqual(v01_frame_iou(reference["+Y"], projected_soups(self.vertices, vertices, faces)["+Y"], 128),
                            v01_frame_iou(reference["+Y"], projected_soups(self.vertices, thin, self.faces)["+Y"], 128))

    def test_overhang_beyond_the_canvas_cap_is_clipped_and_recorded(self):
        # 5 units along +X is far past the 2x canvas cap in every view showing x.
        speck = np.array([[5.0, 0.0, 0.0], [5.0 + 1e-6, 0.0, 0.0], [5.0, 1e-6, 0.0]])
        result = self.evaluate(np.vstack([self.vertices, speck]), np.vstack([self.faces, [8, 9, 10]]))
        self.assertEqual(result["silhouette_frame_clipped_views"], ["+Y", "-Y", "+Z", "-Z"])
        self.assertEqual(result["silhouette_multiview_frame_clipped_views"], MULTIVIEW_LABELS[1:])
        self.assertEqual(set(self.views(result).values()), {1.0})  # the speck covers no pixel centre
        self.assertIn("reference", result["sampling"]["silhouette_frame"])
        self.assertEqual(result["sampling"]["silhouette_projection"],
                         "orthographic_reference_anchored_square_frame_2pct_margin")

    def test_overhanging_candidate_matches_a_generous_canvas(self):
        # Reference-anchored IoU computed independently on a canvas three frames wide.
        candidate = self.vertices * [1.3, 1.0, 0.7] + [0.05, 0.0, 0.0]
        result = self.evaluate(candidate)
        reference = projected_soups(self.vertices, self.vertices, self.faces)
        soups = projected_soups(self.vertices, candidate, self.faces)
        for label, value in self.views(result).items():
            lower, upper = reference[label].min(axis=(0, 1)), reference[label].max(axis=(0, 1))
            side = float(np.max(upper - lower)) * 1.04
            frame_lower = (lower + upper) * 0.5 - side * 0.5
            wide = metrics._rasterize(soups[label], frame_lower, 128 / side, 128, (-128, -128, 256, 256))
            reference_mask = np.zeros_like(wide)
            reference_mask[128:256, 128:256] = metrics._rasterize(reference[label], frame_lower, 128 / side, 128)
            expected = np.count_nonzero(reference_mask & wide) / np.count_nonzero(reference_mask | wide)
            self.assertEqual(value, expected, msg=label)
        # +Y shows (x, z): intersection 1 x 0.7, union 1 + 1.3 x 0.7 - 0.7.
        self.assertAlmostEqual(result["silhouette_iou_views"]["+Y"], 0.7 / 1.21, delta=0.02)


class OptionInvarianceTests(CubeCase):
    def test_options_do_not_change_existing_keys(self):
        candidate = self.vertices * [1.0, 1.0, 0.5] + [0.05, 0.0, 0.0]
        base = self.evaluate(candidate, seed=5, multiview=False)
        full = self.evaluate(candidate, seed=5, roi_weights=(self.vertices[:, 0] > 0).astype(float))
        for key, value in base.items():
            if key != "sampling":
                self.assertEqual(full[key], value, msg=key)
        for key, value in base["sampling"].items():
            if key != "silhouette_multiview":
                self.assertEqual(full["sampling"][key], value, msg=key)

    def test_rng_children_zero_and_one_match_v01_spawn_of_two(self):
        for seed in (0, 42, 2**40):
            two = np.random.SeedSequence(seed).spawn(2)
            four = np.random.SeedSequence(seed).spawn(4)
            for old, new in zip(two, four[:2]):
                self.assertTrue(np.array_equal(np.random.default_rng(old).random(8), np.random.default_rng(new).random(8)))

    def test_determinism_and_json_safety_with_all_options(self):
        rng = np.random.default_rng(9)
        candidate = self.vertices + rng.normal(0.0, 0.05, self.vertices.shape)
        weights = rng.random(8)
        kwargs = dict(seed=11, roi_weights=weights, roi_threshold=0.4, resolution=64)
        first = self.evaluate(candidate, **kwargs)
        self.assertEqual(first, self.evaluate(candidate, **kwargs))
        assert_json_safe(self, first)
        # Other seeds change the ROI sample streams, as expected.
        other = self.evaluate(candidate, **dict(kwargs, seed=12))
        self.assertNotEqual(first["roi_chamfer_ref_to_candidate"], other["roi_chamfer_ref_to_candidate"])


if __name__ == "__main__":
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(case) for case in (
        MetricsTests, MultiviewTests, RoiTests, VertexHausdorffTests, FramingTests, OptionInvarianceTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
