"""Blender integration regression tests for the installable extension.

Run from the repository root:
blender --background --factory-startup --python-exit-code 1 --python tests/test_integration_blender.py
"""

from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
import warnings

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import smart_lod
from smart_lod import pipeline, ui
from smart_lod.geometry import feature_constraints, group_weights, result_mesh, snapshot, world_arrays
from smart_lod.metrics import evaluate_meshes
from smart_lod.perception import (DEFAULT_SILHOUETTE_RESOLUTION, DEFAULT_SILHOUETTE_VIEWS, DEFAULT_WEIGHTS,
                                  camera_from_object, compute_importance)
from smart_lod.pipeline import generate
from smart_lod.qem import simplify

# Stats that legitimately differ between a single-pass level and an independent run.
PASS_ONLY_STATS = {'seconds', 'single_pass', 'pass_targets'}


def source_signature(obj):
    mesh = obj.data
    return {
        "vertices": [tuple(v.co) for v in mesh.vertices],
        "polygons": [tuple(p.vertices) for p in mesh.polygons],
        "materials": [(p.material_index, p.use_smooth) for p in mesh.polygons],
        "uv": [[tuple(loop.uv) for loop in layer.data] for layer in mesh.uv_layers],
        "matrix": [tuple(row) for row in obj.matrix_world],
        "groups": [(group.name, group.index) for group in obj.vertex_groups],
        "weights": [[(group.group, group.weight) for group in vertex.groups] for vertex in mesh.vertices],
        "modifiers": [(modifier.name, modifier.type) for modifier in obj.modifiers],
    }


def scene_state():
    return set(bpy.data.objects.keys()), set(bpy.data.meshes.keys()), set(bpy.data.collections.keys())


def outward_fraction(obj, centre):
    """Share of an object's world triangles whose normal points away from ``centre``."""
    vertices, faces = world_arrays(obj.data, obj.matrix_world)
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    normals = np.cross(b - a, c - a)
    return float(np.mean(np.einsum('ij,ij->i', normals, (a + b + c) / 3 - np.asarray(centre)) > 0))


def signed_volume(obj):
    vertices, faces = world_arrays(obj.data, obj.matrix_world)
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6)


def draw_panel_labels(context):
    """Labels of one SMARTLOD_PT_main.draw. Background mode never draws panels, so a
    recording layout stand-in is used: every layout call returns the recorder."""
    class Recorder:
        def __init__(self):
            self.labels = []

        def __getattr__(self, name):
            return lambda *args, **kwargs: self

        def label(self, text='', **kwargs):
            self.labels.append(text)

    panel = type('PanelStub', (), {'layout': Recorder()})()
    ui.SMARTLOD_PT_main.draw(panel, context)
    return panel.layout.labels


def ngon_grid(name, seed=7):
    """Planar 16x16 grid with a third of its interior edges dissolved (n-gons with
    collinear corners), two materials, mixed smooth flags and UV = local (x, y)."""
    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=16, y_segments=16, size=2.0)
    bm.edges.ensure_lookup_table()
    interior = [edge for edge in bm.edges if not edge.is_boundary]
    chosen = rng.choice(len(interior), size=len(interior) // 3, replace=False)
    bmesh.ops.dissolve_edges(bm, edges=[interior[i] for i in chosen], use_verts=False)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for material in ('Regression_NgonA', 'Regression_NgonB'):
        mesh.materials.append(bpy.data.materials.new(material))
    for polygon in mesh.polygons:
        polygon.material_index = int(polygon.center.x > 0.3)
        polygon.use_smooth = polygon.index % 3 == 0
    layer = mesh.uv_layers.new(name='UVMap')
    for loop in mesh.loops:
        layer.data[loop.index].uv = tuple(mesh.vertices[loop.vertex_index].co)[:2]
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


@contextmanager
def overridden(settings, **values):
    previous = {name: getattr(settings, name) for name in values}
    try:
        for name, value in values.items():
            setattr(settings, name, value)
        yield settings
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        smart_lod.register()
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=12, radius=1.0)
        cls.source = bpy.context.active_object
        cls.source.name = 'Regression_HighPoly'
        cls.source.location = (4.0, -2.0, 1.25)
        cls.source.rotation_euler = (0.22, -0.31, 0.43)
        cls.source.scale = (1.8, 0.7, 1.3)
        for name in ('Regression_Red', 'Regression_Blue'):
            cls.source.data.materials.append(bpy.data.materials.new(name))
        for face in cls.source.data.polygons:
            face.material_index = int(face.center.z > 0)
            face.use_smooth = True
        group = cls.source.vertex_groups.new(name='LOD_Protect')
        group.add([v.index for v in cls.source.data.vertices if v.co.z > 0.6], 0.8, 'REPLACE')
        cls.roi_members = [v.index for v in cls.source.data.vertices if v.co.x > 0.4]
        roi = cls.source.vertex_groups.new(name='LOD_EvalROI')
        roi.add(cls.roi_members, 1.0, 'REPLACE')
        bpy.context.view_layer.update()
        cls.initial_signature = source_signature(cls.source)
        cls.initial_objects = set(bpy.data.objects.keys())
        cls.initial_meshes = set(bpy.data.meshes.keys())
        cls.settings = bpy.context.scene.smart_lod
        cls.settings.ratio_1 = 0.70
        cls.settings.ratio_2 = 0.45
        cls.settings.ratio_3 = 0.20
        cls.settings.sample_count = 200
        cls.settings.resolution = 32
        cls.settings.evaluate = True
        cls.settings.preserve_seams = True
        # Non-default importance settings, so a pipeline ignoring them differs.
        cls.settings.silhouette_views = 12
        cls.settings.silhouette_resolution = 128
        cls.progress_calls = []
        cls.report, cls.outputs = generate(bpy.context, cls.source, cls.settings, compare=True,
                                           progress=lambda *args: cls.progress_calls.append(args))
        cls.encoded_report = json.dumps(cls.report)

    def setUp(self):
        bpy.context.scene['smart_lod_report'] = self.encoded_report
        self.source.hide_set(False)
        for obj in list(bpy.context.selected_objects):
            obj.select_set(False)
        self.source.select_set(True)
        bpy.context.view_layer.objects.active = self.source

    def rows(self, report, method):
        return [row for row in report['results'] if row['method'] == method]

    def recomputed_inputs(self, source):
        """The pipeline's QEM inputs rebuilt independently from the same settings."""
        mesh, vertices, faces = snapshot(source, bpy.context.evaluated_depsgraph_get())
        try:
            locked, marked_sharp = feature_constraints(mesh, self.settings.preserve_seams)
            manual = group_weights(mesh, source, self.settings.protect_group)
        finally:
            bpy.data.meshes.remove(mesh)
        s = self.settings
        importance, _ = compute_importance(
            vertices, faces, weights=dict(saliency=s.saliency_weight, silhouette=s.silhouette_weight,
                                          sharp=s.sharp_weight, visibility=s.visibility_weight),
            manual=manual, sharp_angle=s.sharp_angle, marked_sharp=marked_sharp,
            silhouette_views=s.silhouette_views, silhouette_resolution=s.silhouette_resolution)
        return vertices, faces, locked, importance

    def assert_matches_independent_simplify(self, report, rows, inputs, weighting):
        vertices, faces, locked, importance = inputs
        for row in rows:
            perceptual = row['method'] == 'PERCEPTUAL'
            expected = simplify(vertices, faces, row['target_triangles'],
                                importance=importance if perceptual else None,
                                strength=self.settings.strength if perceptual else 0.0,
                                locked=locked, weighting=weighting)
            obj = bpy.data.objects[row['object']]
            delivered_vertices, delivered_faces = world_arrays(obj.data, obj.matrix_world)
            np.testing.assert_array_equal(delivered_faces, expected.faces)
            # Delivered meshes store float32 coordinates.
            np.testing.assert_allclose(delivered_vertices, expected.vertices, rtol=0, atol=1e-6)
            stats = {k: v for k, v in row['qem_stats'].items() if k not in PASS_ONLY_STATS}
            self.assertEqual(stats, {k: v for k, v in expected.stats.items() if k not in PASS_ONLY_STATS})
            self.assertTrue(row['qem_stats']['single_pass'])
            self.assertEqual(row['qem_stats']['weighting'], weighting)

    def test_all_methods_generate_measured_lods(self):
        self.assertEqual(len(self.outputs), 10)
        self.assertEqual(len(self.report['results']), 9)
        self.assertEqual({row['method'] for row in self.report['results']}, {'DECIMATE', 'QEM', 'PERCEPTUAL'})
        for method in ('DECIMATE', 'QEM', 'PERCEPTUAL'):
            rows = self.rows(self.report, method)
            self.assertEqual([row['lod'] for row in rows], [1, 2, 3])
            for row in rows:
                self.assertGreater(row['actual_triangles'], 0)
                self.assertLess(row['actual_triangles'], row['source_triangles'])
                self.assertEqual(row['actual_triangles'], row['triangles_candidate'])
                self.assertEqual(row['target_reached'], row['actual_triangles'] <= row['target_triangles'])
                self.assertGreaterEqual(row['chamfer_mean'], 0.0)
                self.assertGreaterEqual(row['hausdorff_sampled'], row['chamfer_mean'])
                self.assertTrue(0.0 <= row['silhouette_iou_mean'] <= 1.0)
                self.assertTrue(0.0 <= row['silhouette_iou_multiview_mean'] <= 1.0)
                self.assertEqual(len(row['silhouette_iou_multiview_views']), 13)
                for axis in ('+X', '+Y', '+Z'):
                    self.assertEqual(row['silhouette_iou_multiview_views'][axis], row['silhouette_iou_views'][axis])
                self.assertGreater(row['processing_seconds'], 0.0)
                self.assertEqual(row['sampling']['samples_per_mesh'], 200)
        json.dumps(self.report, allow_nan=False)

    def test_roi_metrics_present_when_source_has_roi_group(self):
        self.assertEqual(self.report['roi_group'], 'LOD_EvalROI')
        self.assertEqual(self.report['roi_vertices'], len(self.roi_members))
        self.assertEqual(self.report['roi_threshold'], 0.5)
        for row in self.report['results']:
            self.assertGreater(row['roi_triangles'], 0)
            self.assertTrue(0.0 < row['roi_area_fraction'] < 1.0)
            for prefix in ('roi', 'nonroi'):
                self.assertIsNotNone(row[f'{prefix}_chamfer_ref_to_candidate'])
                self.assertGreaterEqual(row[f'{prefix}_hausdorff_ref_to_candidate'],
                                        row[f'{prefix}_chamfer_ref_to_candidate'])
                # Both directions and their combination, as the report notes describe.
                for direction in ('candidate_to_ref', 'symmetric'):
                    for kind in ('chamfer', 'hausdorff'):
                        self.assertIsNotNone(row[f'{prefix}_{kind}_{direction}_normalized'])
                self.assertAlmostEqual(row[f'{prefix}_chamfer_symmetric'],
                                       0.5 * (row[f'{prefix}_chamfer_ref_to_candidate']
                                              + row[f'{prefix}_chamfer_candidate_to_ref']), delta=1e-12)
            self.assertGreaterEqual(row['hausdorff_with_vertices'], row['hausdorff_sampled'])
            self.assertEqual(row['sampling']['roi_distance_direction'],
                             'reference_to_candidate_only_roi_defined_on_reference')
        notes = ' '.join(self.report['notes'])
        for key in ('hausdorff_with_vertices', '*_candidate_to_ref', '*_symmetric', 'roi_threshold'):
            self.assertIn(key, notes)

    def test_missing_roi_group_omits_roi_metrics(self):
        with overridden(self.settings, method='QEM', roi_group='No_Such_Group'):
            report, _ = generate(bpy.context, self.source, self.settings)
        self.assertIsNone(report['roi_group'])
        self.assertIsNone(report['roi_vertices'])
        for row in report['results']:
            self.assertFalse(any(key.startswith(('roi_', 'nonroi_')) for key in row))
            self.assertIn('silhouette_iou_multiview_mean', row)

    def test_protection_and_roi_groups_must_differ(self):
        before = scene_state()
        with overridden(self.settings, protect_group='LOD_EvalROI'):
            with self.assertRaisesRegex(ValueError, 'ROI group must not be the protection group'):
                generate(bpy.context, self.source, self.settings)
            # The paint helper refuses before touching groups or the mode; bpy.ops
            # turns the operator's ERROR report into a RuntimeError.
            with self.assertRaisesRegex(RuntimeError, 'cannot be used for protection'):
                bpy.ops.smart_lod.prepare_protection()
        self.assertEqual(before, scene_state())
        self.assertEqual(bpy.context.mode, 'OBJECT')
        self.assertEqual(source_signature(self.source), self.initial_signature)

    def test_report_v02_fields_and_timing_semantics(self):
        report = self.report
        self.assertEqual(report['schema_version'], 2)
        self.assertEqual(report['extension_version'], '0.2.0')
        self.assertEqual(report['methods'], ['DECIMATE', 'QEM', 'PERCEPTUAL'])
        self.assertEqual(report['weighting'], 'quadric')
        self.assertIsNone(report['camera'])
        self.assertIsNone(report['camera_parameters'])
        self.assertEqual(set(report['settings']), set(pipeline.SETTING_NAMES))
        self.assertNotIn('curvature_weight', report['settings'])
        self.assertEqual(report['settings']['weighting'], 'QUADRIC')
        self.assertEqual(set(report['importance_channel_seconds']),
                         {'topology', 'saliency', 'silhouette', 'sharp', 'visibility'})
        self.assertEqual(set(report['importance_summary']),
                         {'saliency', 'silhouette', 'sharp', 'visibility', 'manual', 'combined'})
        self.assertEqual(report['importance_summary']['visibility']['nonzero_fraction'], 0.0)
        self.assertEqual(set(report['method_pass_seconds']), set(report['methods']))
        self.assertTrue(any('13 unique orthographic directions' in note for note in report['notes']))
        self.assertTrue(any('constraint_preparation_seconds' in note for note in report['notes']))
        # The seam/material lock pass is a QEM/PERCEPTUAL prerequisite: charged to both, not to DECIMATE.
        constraints = report['constraint_preparation_seconds']
        self.assertGreater(constraints, 0.0)
        for method in ('QEM', 'PERCEPTUAL'):
            rows = self.rows(report, method)
            for row in rows:
                self.assertAlmostEqual(row['simplification_and_mesh_seconds'],
                                       row['qem_stats']['seconds'] + row['mesh_reconstruction_seconds'], places=12)
                importance = report['importance_preparation_seconds'] if method == 'PERCEPTUAL' else 0.0
                self.assertEqual(row['importance_seconds'], importance)
                self.assertEqual(row['constraint_preparation_seconds'], constraints)
                self.assertAlmostEqual(row['processing_seconds'],
                                       row['simplification_and_mesh_seconds'] + constraints + importance, places=12)
            # Levels come from one pass: cumulative time does not decrease.
            seconds = [row['qem_stats']['seconds'] for row in rows]
            self.assertEqual(seconds, sorted(seconds))
            self.assertGreaterEqual(report['method_pass_seconds'][method],
                                    max(row['simplification_and_mesh_seconds'] for row in rows) + constraints)
        decimate = self.rows(report, 'DECIMATE')
        self.assertAlmostEqual(report['method_pass_seconds']['DECIMATE'],
                               sum(row['simplification_and_mesh_seconds'] for row in decimate), places=9)
        self.assertTrue(all(row['mesh_reconstruction_seconds'] is None for row in decimate))
        for row in decimate:
            self.assertEqual(row['constraint_preparation_seconds'], 0.0)
            self.assertEqual(row['processing_seconds'], row['simplification_and_mesh_seconds'])

    def test_custom_lods_equal_independent_simplify(self):
        inputs = self.recomputed_inputs(self.source)
        vertices, faces = inputs[:2]
        self.assertEqual(self.report['source_sha256'], hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest())
        # The reference stores the importance map at 1/255 precision.
        group = self.outputs[0].vertex_groups['LOD_Importance']
        stored = np.array([next(g.weight for g in v.groups if g.group == group.index)
                           for v in self.outputs[0].data.vertices])
        np.testing.assert_allclose(stored, np.rint(inputs[3] * 255) / 255, rtol=0, atol=1e-6)
        rows = self.rows(self.report, 'QEM') + self.rows(self.report, 'PERCEPTUAL')
        self.assert_matches_independent_simplify(self.report, rows, inputs, 'quadric')
        self.assertEqual(self.rows(self.report, 'QEM')[0]['qem_stats']['strength'], 0.0)
        self.assertEqual(self.rows(self.report, 'PERCEPTUAL')[0]['qem_stats']['strength'], self.settings.strength)

    def test_importance_settings_are_forwarded(self):
        # On this coarse sphere some settings (e.g. silhouette pixels) do not change
        # the result, so check the plumbing directly.
        captured = {}
        original = pipeline.compute_importance

        def recording(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        values = dict(method='QEM', evaluate=False, saliency_weight=0.1, silhouette_weight=0.2, sharp_weight=0.3,
                      visibility_weight=0.4, sharp_angle=33.0, silhouette_views=7, silhouette_resolution=96)
        pipeline.compute_importance = recording
        try:
            with overridden(self.settings, **values):
                report, _ = generate(bpy.context, self.source, self.settings)
        finally:
            pipeline.compute_importance = original
        self.assertEqual(captured['weights'], {name: report['settings'][name + '_weight']
                                               for name in ('saliency', 'silhouette', 'sharp', 'visibility')})
        for name in ('saliency', 'silhouette', 'sharp', 'visibility'):
            self.assertAlmostEqual(captured['weights'][name], values[name + '_weight'], places=6)
        self.assertAlmostEqual(captured['sharp_angle'], 33.0, places=5)
        self.assertEqual((captured['silhouette_views'], captured['silhouette_resolution']), (7, 96))
        self.assertIsNone(captured['camera'])
        self.assertEqual(int(np.count_nonzero(captured['manual'])), report['manually_protected_vertices'])
        self.assertNotIn('silhouette_iou_mean', report['results'][0])

    def test_anchor_weighting_path_runs(self):
        with overridden(self.settings, method='PERCEPTUAL', weighting='ANCHOR'):
            report, outputs = generate(bpy.context, self.source, self.settings)
            self.assertEqual(report['weighting'], 'anchor')
            self.assertEqual(report['settings']['weighting'], 'ANCHOR')
            self.assertEqual(len(outputs), 4)
            self.assert_matches_independent_simplify(report, report['results'],
                                                     self.recomputed_inputs(self.source), 'anchor')

    def test_progress_is_monotone_and_reports_single_pass(self):
        calls = self.progress_calls
        self.assertTrue(calls)
        done = [call[0] for call in calls]
        self.assertEqual(done, sorted(done))
        self.assertTrue(all(0 <= value < 9 and total == 9 for value, total, _, _ in calls))
        self.assertTrue(all(level in (1, 2, 3) for *_, level in calls))
        self.assertEqual([call[2] for call in calls if call[0] == 0], ['Blender Decimate'])
        for name in ('QEM', 'Perceptual QEM'):
            # Start of the pass, at least one call from inside it, then one per level.
            fractional = [call for call in calls if call[2] == name and call[0] != int(call[0])]
            self.assertTrue(fractional, name)

    def test_failed_run_mid_pass_rolls_back(self):
        before = scene_state()
        previous_report = bpy.context.scene['smart_lod_report']
        calls = []

        def fail_inside_pass(done, total, method, level):
            calls.append(done)
            if done != int(done):  # only simplify_levels reports fractions
                raise RuntimeError('Intentional mid-pass interruption')

        with overridden(self.settings, method='QEM'):
            with self.assertRaisesRegex(RuntimeError, 'Intentional mid-pass'):
                generate(bpy.context, self.source, self.settings, progress=fail_inside_pass)
        self.assertEqual(calls[0], 0)
        self.assertEqual(before, scene_state())
        self.assertEqual(bpy.context.scene['smart_lod_report'], previous_report)
        self.assertEqual(source_signature(self.source), self.initial_signature)

    def test_camera_parameters_match_blender_view_frame(self):
        scene = bpy.context.scene
        render = scene.render
        saved = (render.resolution_x, render.resolution_y, render.pixel_aspect_x, render.pixel_aspect_y)
        data = bpy.data.cameras.new('Regression_CameraData')
        camera = bpy.data.objects.new('Regression_Camera', data)
        scene.collection.objects.link(camera)
        camera.location = (1.0, -7.0, 2.5)
        camera.rotation_euler = (1.2, 0.1, -0.3)
        cases = [('PERSP', 'AUTO', 1920, 1080, 1, 1, 0, 0), ('PERSP', 'AUTO', 1080, 1920, 1, 1, 0, 0),
                 ('PERSP', 'HORIZONTAL', 800, 1200, 1, 1, 0, 0), ('PERSP', 'VERTICAL', 1600, 900, 1, 1, 0, 0),
                 ('PERSP', 'AUTO', 1000, 1000, 2, 1, 0, 0), ('PERSP', 'AUTO', 1200, 1000, 1, 1.5, 0, 0),
                 ('PERSP', 'AUTO', 1920, 1080, 1, 1, 0.2, -0.1), ('ORTHO', 'AUTO', 1920, 1080, 1, 1, 0, 0),
                 ('ORTHO', 'VERTICAL', 1920, 1080, 1, 1, 0, 0), ('ORTHO', 'AUTO', 900, 1600, 1, 1, 0, 0.25)]
        try:
            data.sensor_width, data.sensor_height, data.lens, data.ortho_scale = 36.0, 20.0, 42.0, 5.0
            for kind, fit, rx, ry, ax, ay, shift_x, shift_y in cases:
                with self.subTest(kind=kind, fit=fit, size=(rx, ry), aspect=(ax, ay), shift=(shift_x, shift_y)):
                    data.type, data.sensor_fit, data.shift_x, data.shift_y = kind, fit, shift_x, shift_y
                    render.resolution_x, render.resolution_y = rx, ry
                    render.pixel_aspect_x, render.pixel_aspect_y = ax, ay
                    camera.scale = (1.0, 1.0, 1.0)
                    bpy.context.view_layer.update()
                    ours = pipeline.camera_parameters(camera, scene)
                    blender = camera_from_object(camera, scene)  # Camera.view_frame based
                    self.assertEqual(ours['type'], blender['type'])
                    for key in ('position', 'forward', 'up'):
                        np.testing.assert_allclose(ours[key], blender[key], atol=1e-6)
                    for key in ('angle_x', 'angle_y', 'ortho_width', 'ortho_height'):
                        if key in blender:
                            self.assertAlmostEqual(ours[key], blender[key], delta=1e-5 * blender[key])
                    # Rendering ignores the camera object's scale; so do the parameters.
                    camera.scale = (2.0, 0.5, 3.0)
                    bpy.context.view_layer.update()
                    scaled = pipeline.camera_parameters(camera, scene)
                    self.assertEqual(scaled['type'], ours['type'])
                    for key in set(ours) - {'type'}:
                        np.testing.assert_allclose(scaled[key], ours[key], atol=1e-6)
        finally:
            render.resolution_x, render.resolution_y, render.pixel_aspect_x, render.pixel_aspect_y = saved
            bpy.data.objects.remove(camera, do_unlink=True)
            bpy.data.cameras.remove(data)

    def test_camera_visibility_channel_from_scene_camera(self):
        scene = bpy.context.scene
        data = bpy.data.cameras.new('Regression_ViewCameraData')
        camera = bpy.data.objects.new('Regression_ViewCamera', data)
        scene.collection.objects.link(camera)
        camera.location = self.source.location + Vector((0.0, -9.0, 0.0))
        camera.rotation_euler = (self.source.location - camera.location).to_track_quat('-Z', 'Y').to_euler()
        previous_camera = scene.camera
        scene.camera = camera
        bpy.context.view_layer.update()
        try:
            with overridden(self.settings, method='PERCEPTUAL', use_camera=True):
                report, _ = generate(bpy.context, self.source, self.settings)
        finally:
            scene.camera = previous_camera
            bpy.data.objects.remove(camera, do_unlink=True)
            bpy.data.cameras.remove(data)
        self.assertEqual(report['camera'], 'Regression_ViewCamera')
        self.assertEqual(report['camera_parameters']['type'], 'PERSP')
        visible = report['importance_summary']['visibility']['nonzero_fraction']
        # A convex sphere seen from one side: roughly the near half is visible.
        self.assertTrue(0.2 < visible < 0.8, visible)

    def test_original_and_static_reference_are_preserved(self):
        self.assertEqual(source_signature(self.source), self.initial_signature)
        reference = self.outputs[0]
        self.assertIsNot(reference.data, self.source.data)
        self.assertEqual(reference['smart_lod_level'], 0)
        ref_vertices, ref_faces = world_arrays(reference.data, reference.matrix_world)
        source_vertices, source_faces = world_arrays(self.source.data, self.source.matrix_world)
        # Assigning matrix_world decomposes/recomposes Blender float32 transforms.
        np.testing.assert_allclose(ref_vertices, source_vertices, rtol=0, atol=2e-6)
        np.testing.assert_array_equal(ref_faces, source_faces)
        self.assertIsNotNone(reference.data.color_attributes.get('LOD_Heatmap'))
        self.assertIsNotNone(reference.vertex_groups.get('LOD_Importance'))

    def test_transformed_output_metrics_match_delivered_geometry(self):
        reference = self.outputs[0]
        ref_vertices, ref_faces = world_arrays(reference.data, reference.matrix_world)
        expected_diagonal = float(np.linalg.norm(np.ptp(ref_vertices, axis=0)))
        roi = np.zeros(len(ref_vertices))
        roi[self.roi_members] = 1.0
        # Test one delivered mesh from each path, including native Decimate.
        for method in ('DECIMATE', 'QEM', 'PERCEPTUAL'):
            row = next(row for row in self.report['results'] if row['method'] == method and row['lod'] == 1)
            obj = bpy.data.objects[row['object']]
            # Static outputs bake the evaluated world transform, including shear.
            np.testing.assert_allclose(np.asarray(obj.matrix_world), np.eye(4), atol=1e-7)
            vertices, faces = world_arrays(obj.data, obj.matrix_world)
            independently = evaluate_meshes(ref_vertices, ref_faces, vertices, faces, sample_count=200, resolution=32,
                                            roi_weights=roi)
            self.assertAlmostEqual(row['reference_bbox_diagonal'], expected_diagonal, delta=2e-6)
            self.assertAlmostEqual(row['chamfer_mean'], independently['chamfer_mean'], delta=2e-6)
            self.assertAlmostEqual(row['hausdorff_sampled'], independently['hausdorff_sampled'], delta=2e-6)
            self.assertAlmostEqual(row['silhouette_iou_mean'], independently['silhouette_iou_mean'], delta=0.005)
            self.assertAlmostEqual(row['silhouette_iou_multiview_mean'],
                                   independently['silhouette_iou_multiview_mean'], delta=0.005)
            self.assertEqual(row['roi_triangles'], independently['roi_triangles'])
            self.assertAlmostEqual(row['roi_chamfer_ref_to_candidate'],
                                   independently['roi_chamfer_ref_to_candidate'], delta=2e-6)

    def test_uvs_material_slots_and_surface_assignments_survive(self):
        original_layers = [layer.name for layer in self.source.data.uv_layers]
        self.assertTrue(original_layers)
        self.assertGreater(self.report['seam_locked_vertices'], 0)
        self.assertGreater(self.report['manually_protected_vertices'], 0)
        for obj in self.outputs:
            self.assertEqual([layer.name for layer in obj.data.uv_layers], original_layers)
            self.assertEqual(list(obj.data.materials), list(self.source.data.materials))
            self.assertEqual({face.material_index for face in obj.data.polygons}, {0, 1})
            for layer in obj.data.uv_layers:
                self.assertTrue(np.isfinite([tuple(loop.uv) for loop in layer.data]).all())

    def test_json_csv_exports_include_reproducibility_fields(self):
        with tempfile.TemporaryDirectory(prefix='smart_lod_integration_') as directory:
            json_path = Path(directory) / 'report.json'
            csv_path = Path(directory) / 'report.csv'
            self.assertEqual(bpy.ops.smart_lod.export_report(filepath=str(json_path), format='JSON'), {'FINISHED'})
            exported = json.loads(json_path.read_text(encoding='utf-8'))
            self.assertEqual(exported['source_sha256'], self.report['source_sha256'])
            self.assertEqual(len(exported['results']), 9)
            self.assertEqual(bpy.ops.smart_lod.export_report(filepath=str(csv_path), format='CSV'), {'FINISHED'})
            with csv_path.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            self.assertEqual(rows[0]['run_id'], self.report['run_id'])
            self.assertIn('chamfer_mean_normalized', rows[0])
            self.assertIn('sampling', rows[0])
            self.assertEqual(len(json.loads(rows[0]['silhouette_iou_multiview_views'])), 13)
            self.assertEqual(float(rows[0]['roi_chamfer_ref_to_candidate']),
                             self.report['results'][0]['roi_chamfer_ref_to_candidate'])
            self.assertEqual(json.loads(rows[0]['settings_json'])['saliency_weight'],
                             self.report['settings']['saliency_weight'])

    def test_failed_run_rolls_back_generated_collection(self):
        before = scene_state()
        previous_report = bpy.context.scene['smart_lod_report']

        def fail_during_progress(*args):
            raise RuntimeError('Intentional regression-test interruption')

        with self.assertRaisesRegex(RuntimeError, 'Intentional regression-test'):
            generate(bpy.context, self.source, self.settings, progress=fail_during_progress)
        self.assertEqual(before, scene_state())
        self.assertEqual(bpy.context.scene['smart_lod_report'], previous_report)
        self.assertEqual(source_signature(self.source), self.initial_signature)

    def test_failed_object_creation_does_not_orphan_built_mesh(self):
        before = scene_state()
        original_object = pipeline._object

        def fail_for_lods(name, mesh, source, collection, run_id, method, level):
            if level > 0:  # the LOD mesh exists already but is not linked yet
                raise RuntimeError('Intentional object creation failure')
            return original_object(name, mesh, source, collection, run_id, method, level)

        pipeline._object = fail_for_lods
        try:
            for method in ('QEM', 'DECIMATE'):
                with self.subTest(method=method), overridden(self.settings, method=method):
                    with self.assertRaisesRegex(RuntimeError, 'Intentional object creation'):
                        generate(bpy.context, self.source, self.settings)
                    self.assertEqual(before, scene_state())
        finally:
            pipeline._object = original_object
        self.assertEqual(source_signature(self.source), self.initial_signature)

    def test_failed_evaluation_rolls_back_objects_meshes_and_collection(self):
        before = scene_state()
        previous_report = bpy.context.scene['smart_lod_report']
        original_evaluate = pipeline.evaluate_meshes

        def fail_evaluation(*args, **kwargs):
            raise RuntimeError('Intentional regression-test metric failure')

        pipeline.evaluate_meshes = fail_evaluation
        try:
            with self.assertRaisesRegex(RuntimeError, 'Intentional regression-test metric'):
                generate(bpy.context, self.source, self.settings)
        finally:
            pipeline.evaluate_meshes = original_evaluate
        self.assertEqual(before, scene_state())
        self.assertEqual(bpy.context.scene['smart_lod_report'], previous_report)
        self.assertEqual(source_signature(self.source), self.initial_signature)

    def test_no_snapshot_or_decimate_temporaries_leak(self):
        # A fresh run adds exactly its delivered meshes, nothing else.
        meshes_before = set(bpy.data.meshes.keys())
        report, outputs = generate(bpy.context, self.source, self.settings, compare=True)
        added_meshes = set(bpy.data.meshes.keys()) - meshes_before
        self.assertEqual(added_meshes, {obj.data.name for obj in outputs})
        # Across every run in this suite (other tests may have generated more).
        self.assertFalse(any(obj.name.startswith('_SmartLOD_temporary') for obj in bpy.data.objects))
        self.assertFalse(any(mesh.name.endswith('_LOD_snapshot') for mesh in bpy.data.meshes))
        self.assertFalse(any(mesh.users == 0 and mesh.name not in self.initial_meshes for mesh in bpy.data.meshes))

    def test_preview_heatmap_rna_and_visibility(self):
        reference = self.outputs[0]
        view_area = next((area for area in bpy.context.screen.areas if area.type == 'VIEW_3D'), None)
        self.assertIsNotNone(view_area, 'Factory startup should provide a View3D area in background mode')
        with bpy.context.temp_override(area=view_area):
            self.assertEqual(bpy.ops.smart_lod.preview(object_name=reference.name, heatmap=True), {'FINISHED'})
            self.assertEqual(view_area.spaces.active.shading.color_type, 'VERTEX')
            self.assertEqual(reference.data.color_attributes.active_color.name, 'LOD_Heatmap')
        self.assertIs(bpy.context.view_layer.objects.active, reference)
        self.assertFalse(reference.hide_get())
        self.assertTrue(self.source.hide_get())
        self.assertTrue(all(obj.hide_get() for obj in self.outputs[1:]))
        self.assertEqual(bpy.ops.smart_lod.preview(object_name=self.source.name), {'FINISHED'})
        self.assertFalse(self.source.hide_get())
        self.assertTrue(all(obj.hide_get() for obj in self.outputs))

    def test_panel_draws_latest_results(self):
        with overridden(self.settings, show_research=True, show_weights=True):
            labels = draw_panel_labels(bpy.context)
        self.assertTrue(any(text.startswith('IoU axis') and '13-view' in text for text in labels), labels)
        self.assertTrue(any(text.startswith('CD ') and 'HD ' in text for text in labels), labels)
        triangles = len(world_arrays(self.source.data, self.source.matrix_world)[1])
        self.assertIn(f'{triangles:,} base triangles', labels)

    def test_rejected_ratios_do_not_mutate_scene(self):
        before = scene_state()
        with overridden(self.settings, ratio_2=self.settings.ratio_1):
            with self.assertRaisesRegex(ValueError, 'Ratios must decrease'):
                generate(bpy.context, self.source, self.settings)
        self.assertEqual(before, scene_state())
        self.assertEqual(source_signature(self.source), self.initial_signature)

    def test_repeat_generation_keeps_previous_run(self):
        objects_before = set(bpy.data.objects.keys())
        meshes_before = set(bpy.data.meshes.keys())
        with overridden(self.settings, method='DECIMATE'):
            report, outputs = generate(bpy.context, self.source, self.settings)
        self.assertNotEqual(report['run_id'], self.report['run_id'])
        self.assertTrue(objects_before.issubset(set(bpy.data.objects.keys())))
        self.assertTrue(meshes_before.issubset(set(bpy.data.meshes.keys())))
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(set(obj.name for obj in self.outputs + outputs)), 14)
        self.assertEqual(source_signature(self.source), self.initial_signature)

    def test_registration_properties_and_generated_source_guard(self):
        self.assertTrue(hasattr(bpy.types.Scene, 'smart_lod'))
        properties = self.settings.bl_rna.properties
        self.assertEqual(properties['sample_count'].hard_min, 100)
        self.assertNotIn('curvature_weight', properties)
        for channel, weight in DEFAULT_WEIGHTS.items():
            self.assertAlmostEqual(properties[channel + '_weight'].default, weight, places=6)
        self.assertAlmostEqual(properties['strength'].default, pipeline.DEFAULT_STRENGTH, places=6)
        self.assertEqual(properties['weighting'].default, 'QUADRIC')
        self.assertEqual([item.identifier for item in properties['weighting'].enum_items], ['QUADRIC', 'ANCHOR'])
        # The UI defaults are the perception module's defaults (one source of truth).
        self.assertEqual((properties['silhouette_views'].default, properties['silhouette_views'].hard_min,
                          properties['silhouette_views'].hard_max), (DEFAULT_SILHOUETTE_VIEWS, 4, 96))
        self.assertEqual((properties['silhouette_resolution'].default, properties['silhouette_resolution'].hard_min,
                          properties['silhouette_resolution'].hard_max), (DEFAULT_SILHOUETTE_RESOLUTION, 64, 1024))
        self.assertEqual(properties['roi_group'].default, 'LOD_EvalROI')
        self.assertIn('never used for protection', properties['roi_group'].description)
        self.assertTrue(set(pipeline.SETTING_NAMES).issubset(properties.keys()))
        with self.assertRaisesRegex(ValueError, 'original mesh'):
            generate(bpy.context, self.outputs[0], self.settings)

    def test_versions_agree(self):
        with (ROOT / 'smart_lod' / 'blender_manifest.toml').open('rb') as stream:
            manifest = tomllib.load(stream)
        self.assertEqual(manifest['version'], '0.2.0')
        self.assertEqual(pipeline.EXTENSION_VERSION, manifest['version'])
        self.assertEqual('.'.join(map(str, smart_lod.bl_info['version'])), manifest['version'])
        self.assertEqual('.'.join(map(str, ui.bl_info['version'])), manifest['version'])
        self.assertEqual(pipeline.REPORT_SCHEMA_VERSION, 2)

    def test_slot_object_material_override_is_preserved(self):
        source = self.source.copy()
        source.data = self.source.data.copy()
        source.name = 'Regression_ObjectMaterial_HighPoly'
        bpy.context.scene.collection.objects.link(source)
        override = bpy.data.materials.new('Regression_ObjectSlotOverride')
        source.material_slots[0].link = 'OBJECT'
        source.material_slots[0].material = override
        bpy.context.view_layer.update()
        with overridden(self.settings, method='DECIMATE'):
            report, outputs = generate(bpy.context, source, self.settings)
        expected = [slot.material for slot in source.material_slots]
        for obj in outputs:
            self.assertEqual([slot.material for slot in obj.material_slots], expected)

    def test_slot_empty_material_survives_custom_reconstruction(self):
        source = self.source.copy()
        source.data = self.source.data.copy()
        source.name = 'Regression_EmptyMaterial_HighPoly'
        bpy.context.scene.collection.objects.link(source)
        source.material_slots[1].link = 'OBJECT'
        source.material_slots[1].material = None
        bpy.context.view_layer.update()
        with overridden(self.settings, method='QEM'):
            report, outputs = generate(bpy.context, source, self.settings)
        # In Blender 5.1 an OBJECT-linked slot left empty falls back to the mesh
        # data's material, both on the evaluated object and in renders (verified
        # with a Workbench render). Outputs are static snapshots of what renders,
        # so the expectation is the EVALUATED slots, not the raw object slots
        # (which report None). Evaluated slots hold copy-on-evaluation IDs, so
        # compare their originals.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = [slot.material for slot in source.evaluated_get(depsgraph).material_slots]
        expected = [material.original if material else None for material in evaluated]
        self.assertIsNone(source.material_slots[1].material)
        self.assertEqual(expected, list(self.source.data.materials))
        for obj in outputs:
            self.assertEqual([slot.material for slot in obj.material_slots], expected)

    def test_world_transform_with_parent_induced_shear(self):
        parent = bpy.data.objects.new('Regression_ShearedParent', None)
        bpy.context.scene.collection.objects.link(parent)
        parent.scale = (2.0, 0.7, 1.2)
        source = bpy.data.objects.new('Regression_Parented_HighPoly', self.source.data.copy())
        bpy.context.scene.collection.objects.link(source)
        source.parent = parent
        source.rotation_euler = (0.2, 0.4, 0.6)
        source.location = (-3.0, 2.0, 0.0)
        bpy.context.view_layer.update()
        vertices, faces = world_arrays(source.data, source.matrix_world)
        with overridden(self.settings, method='DECIMATE'):
            report, outputs = generate(bpy.context, source, self.settings)
        bpy.context.view_layer.update()
        delivered_vertices, delivered_faces = world_arrays(outputs[0].data, outputs[0].matrix_world)
        np.testing.assert_array_equal(faces, delivered_faces)
        np.testing.assert_allclose(delivered_vertices, vertices, rtol=0, atol=3e-6)

    def test_mirrored_source_keeps_outward_normals_and_visibility(self):
        # A UV sphere is symmetric under x -> -x, so scale (-1, 1, 1) gives the same world
        # geometry; baking the mirror used to deliver every LOD inside-out and invert the
        # front-facing test of the camera visibility channel.
        scene = bpy.context.scene
        centre = Vector((0.5, 0.3, -0.2))
        data = bpy.data.cameras.new('Regression_MirrorCameraData')
        camera = bpy.data.objects.new('Regression_MirrorCamera', data)
        scene.collection.objects.link(camera)
        camera.location = centre + Vector((0.0, -8.0, 0.0))
        camera.rotation_euler = (centre - camera.location).to_track_quat('-Z', 'Y').to_euler()
        previous_camera, scene.camera = scene.camera, camera
        runs = {}
        try:
            for label, scale in (('plain', (1.0, 1.0, 1.0)), ('mirror', (-1.0, 1.0, 1.0)),
                                 ('mirror_stretch', (1.0, -2.0, 0.5))):
                bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=12, radius=1.0, location=centre)
                source = bpy.context.active_object
                source.name = f'Regression_Mirror_{label}_HighPoly'
                source.scale = scale
                bpy.context.view_layer.update()
                with overridden(self.settings, use_camera=True, evaluate=False):
                    runs[label] = generate(bpy.context, source, self.settings, compare=True)
        finally:
            scene.camera = previous_camera
            bpy.data.objects.remove(camera, do_unlink=True)
            bpy.data.cameras.remove(data)

        def fractions(label):
            return {(obj['smart_lod_method'], obj['smart_lod_level']): outward_fraction(obj, centre)
                    for obj in runs[label][1]}

        plain = fractions('plain')
        self.assertEqual(plain[('REFERENCE', 0)], 1.0)
        for label in ('mirror', 'mirror_stretch'):
            with self.subTest(label=label):
                mirrored = fractions(label)
                self.assertEqual(mirrored[('REFERENCE', 0)], 1.0)
                for level in (1, 2, 3):
                    self.assertEqual(mirrored[('DECIMATE', level)], 1.0)
                for key, value in mirrored.items():
                    # Same winding quality as the unmirrored sphere (QEM levels may flip a few faces).
                    self.assertGreaterEqual(value, plain[key] - 0.05, key)
                self.assertTrue(all(signed_volume(obj) > 0 for obj in runs[label][1]))
        visible = {label: runs[label][0]['importance_summary']['visibility']['nonzero_fraction']
                   for label in ('plain', 'mirror')}
        self.assertTrue(0.2 < visible['plain'] < 0.8, visible)
        self.assertAlmostEqual(visible['mirror'], visible['plain'], delta=0.05)

    def test_rotated_ngon_uvs_follow_one_triangulation(self):
        # Re-tessellating n-gons with collinear corners after baking a rotation/scale used to
        # pair barycentrics of one triangulation with the UV loops of another.
        source = ngon_grid('Regression_Ngon_HighPoly')
        source.location = (1.0, -0.5, 0.3)
        source.rotation_euler = (0.7, -0.4, 1.1)
        source.scale = (3.0, 0.4, 1.7)
        bpy.context.view_layer.update()
        self.assertTrue(any(polygon.loop_total > 4 for polygon in source.data.polygons))
        inverse = np.asarray(source.matrix_world.inverted(), dtype=float)

        def local_xy(points):  # the UV of any surface point, as the layer was built
            return (points @ inverse[:3, :3].T + inverse[:3, 3])[:, :2]

        def corner_uvs(mesh):
            uv = np.empty(len(mesh.loops) * 2, dtype=np.float32)
            mesh.uv_layers['UVMap'].data.foreach_get('uv', uv)
            corners = np.empty(len(mesh.loops), dtype=np.int32)
            mesh.loops.foreach_get('vertex_index', corners)
            return uv.reshape(-1, 2), corners

        mesh, vertices, faces = snapshot(source, bpy.context.evaluated_depsgraph_get())
        try:
            # Identity "simplification": each corner keeps its UV, material and smooth flag.
            identity = SimpleNamespace(vertices=vertices, faces=faces, source_faces=np.arange(len(faces)))
            rebuilt = result_mesh('Regression_Ngon_identity', identity, mesh, vertices, faces, Matrix.Identity(4))
            try:
                uv, corners = corner_uvs(rebuilt)
                np.testing.assert_allclose(uv, local_xy(vertices[corners]), rtol=0, atol=1e-5)
                expected = [mesh.polygons[triangle.polygon_index] for triangle in mesh.loop_triangles]
                self.assertEqual([p.material_index for p in rebuilt.polygons], [p.material_index for p in expected])
                self.assertEqual([p.use_smooth for p in rebuilt.polygons], [p.use_smooth for p in expected])
                self.assertEqual({p.material_index for p in rebuilt.polygons}, {0, 1})
                self.assertEqual({p.use_smooth for p in rebuilt.polygons}, {False, True})
            finally:
                bpy.data.meshes.remove(rebuilt)
            meshes_before = set(bpy.data.meshes.keys())
            with self.assertRaisesRegex(ValueError, 'loop triangles'):
                result_mesh('Regression_Ngon_bad', identity, mesh, vertices, faces[:, [1, 2, 0]], Matrix.Identity(4))
            self.assertEqual(set(bpy.data.meshes.keys()), meshes_before)
        finally:
            bpy.data.meshes.remove(mesh)
        with overridden(self.settings, method='QEM', evaluate=False, ratio_1=0.9, ratio_2=0.7, ratio_3=0.5):
            report, outputs = generate(bpy.context, source, self.settings)
        self.assertEqual(report['source_triangles'], len(faces))
        for obj in outputs:
            with self.subTest(obj=obj.name):
                # Planar mesh, UV linear in position: every delivered corner has the UV of its point.
                points = world_arrays(obj.data, obj.matrix_world)[0]
                uv, corners = corner_uvs(obj.data)
                np.testing.assert_allclose(uv, local_xy(points[corners]), rtol=0, atol=1e-4)

    def test_ui_calls_do_not_inherit_compare_or_heatmap(self):
        # 'EXEC_DEFAULT', True runs like a UI click: unset properties are filled from the
        # last call and stored again, unless they are SKIP_SAVE.
        scene = bpy.context.scene
        with overridden(self.settings, method='QEM', evaluate=False):
            self.assertEqual(bpy.ops.smart_lod.generate('EXEC_DEFAULT', True, compare=True), {'FINISHED'})
            self.assertEqual(json.loads(scene['smart_lod_report'])['methods'], ['DECIMATE', 'QEM', 'PERCEPTUAL'])
            # The main button leaves compare unset: it must run the selected method only.
            self.assertEqual(bpy.ops.smart_lod.generate('EXEC_DEFAULT', True), {'FINISHED'})
            report = json.loads(scene['smart_lod_report'])
        self.assertEqual(report['methods'], ['QEM'])
        self.assertEqual(len(report['results']), 3)
        reference = self.outputs[0]
        view_area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        shading = view_area.spaces.active.shading
        with bpy.context.temp_override(area=view_area):
            self.assertEqual(bpy.ops.smart_lod.preview('EXEC_DEFAULT', True, object_name=reference.name,
                                                       heatmap=True), {'FINISHED'})
            self.assertEqual(shading.color_type, 'VERTEX')
            # The eye icons leave heatmap unset: they must show material colours.
            self.assertEqual(bpy.ops.smart_lod.preview('EXEC_DEFAULT', True, object_name=reference.name),
                             {'FINISHED'})
            self.assertEqual(shading.color_type, 'MATERIAL')
        for operator, name in ((bpy.ops.smart_lod.generate, 'compare'), (bpy.ops.smart_lod.preview, 'heatmap'),
                               (bpy.ops.smart_lod.preview, 'object_name')):
            self.assertTrue(operator.get_rna_type().properties[name].is_skip_save, name)

    def test_non_camera_scene_camera_is_rejected_clearly(self):
        scene = bpy.context.scene
        light = bpy.data.lights.new('Regression_SpotData', 'SPOT')
        previous_camera = scene.camera
        for data in (None, light):
            obj = bpy.data.objects.new('Regression_NotACamera', data)
            scene.collection.objects.link(obj)
            scene.camera = obj
            try:
                with self.subTest(type=obj.type), overridden(self.settings, use_camera=True):
                    before = scene_state()
                    with self.assertRaisesRegex(ValueError, 'not a Camera object'):
                        generate(bpy.context, self.source, self.settings)
                    self.assertEqual(before, scene_state())
                    with self.assertRaisesRegex(ValueError, 'not a Camera object'):
                        pipeline.camera_parameters(obj, scene)
                    self.assertIn('Scene camera is not a Camera object', draw_panel_labels(bpy.context))
            finally:
                scene.camera = previous_camera
                bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.lights.remove(light)
        self.assertEqual(source_signature(self.source), self.initial_signature)

    def test_panel_triangle_label_counts_base_mesh_and_flags_modifiers(self):
        source = bpy.data.objects.new('Regression_Subsurf_HighPoly', self.source.data.copy())
        bpy.context.scene.collection.objects.link(source)
        bpy.context.view_layer.objects.active = source
        base = len(world_arrays(source.data, source.matrix_world)[1])
        self.assertIn(f'{base:,} base triangles', draw_panel_labels(bpy.context))
        source.modifiers.new('Regression subdivision', 'SUBSURF').levels = 1
        bpy.context.view_layer.update()
        self.assertIn(f'{base:,} base triangles before modifiers', draw_panel_labels(bpy.context))

    def test_demo_materials_avoid_deprecated_use_nodes(self):
        spec = importlib.util.spec_from_file_location('smart_lod_create_demo', ROOT / 'scripts' / 'create_demo.py')
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            material = demo.material('Regression_DemoMaterial', (0.1, 0.2, 0.3), 0.4, 0.5)
        try:
            self.assertEqual([str(w.message) for w in caught if 'use_nodes' in str(w.message)], [])
            shader = material.node_tree.nodes['Principled BSDF']
            np.testing.assert_allclose(tuple(shader.inputs['Base Color'].default_value), (0.1, 0.2, 0.3, 1.0),
                                       atol=1e-6)
            self.assertAlmostEqual(shader.inputs['Metallic'].default_value, 0.4, places=6)
            self.assertAlmostEqual(shader.inputs['Roughness'].default_value, 0.5, places=6)
        finally:
            bpy.data.materials.remove(material)

    def test_dataset_car_materials_match_the_demo(self):
        # make_dataset.build_car copies create_demo's construction ("keep in sync by hand"); its nested
        # material() must carry the same Blender 5 use_nodes guard, or rebuilding the dataset warns.
        import ast

        def material_source(path, within=None):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            scope = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == within) if within else tree
            function = next(n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == 'material')
            return ast.dump(function)

        self.assertEqual(material_source(ROOT / 'scripts' / 'make_dataset.py', 'build_car'),
                         material_source(ROOT / 'scripts' / 'create_demo.py'))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IntegrationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if hasattr(bpy.types.Scene, 'smart_lod'):
        smart_lod.unregister()
    if not result.wasSuccessful():
        sys.exit(1)
