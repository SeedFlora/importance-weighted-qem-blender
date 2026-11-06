"""Camera visibility tests (need Blender's mathutils BVHTree).

Run: blender --background --factory-startup --python-exit-code 1 --python tests/test_perception_blender.py
"""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from smart_lod.perception import DEFAULT_WEIGHTS, camera_from_object, camera_visibility, compute_importance
from test_perception import icosphere


def persp(position=(0.0, -5.0, 0.0), forward=(0.0, 1.0, 0.0), angle=0.8, **extra):
    camera = {'type': 'PERSP', 'position': position, 'forward': forward, 'up': (0.0, 0.0, 1.0),
              'angle_x': angle, 'angle_y': angle, 'clip_start': 0.1, 'clip_end': 100.0}
    camera.update(extra)
    return camera


def ortho(size=3.0, **extra):
    camera = persp(**extra)
    del camera['angle_x'], camera['angle_y']
    camera.update(type='ORTHO', ortho_width=size, ortho_height=size)
    camera.update(extra)
    return camera


def combine(*parts):
    vertices, faces, offset = [], [], 0
    for v, f in parts:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def segment_distance(points, start, centre):
    """Distance from ``centre`` to each segment ``start -> point``."""
    direction = points - start
    t = np.clip(((centre - start) @ direction.T) / np.einsum('ij,ij->i', direction, direction), 0.0, 1.0)
    return np.linalg.norm(start + t[:, None] * direction - centre, axis=1)


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.vertices, self.faces = icosphere(4)
        self.count = len(self.vertices)

    def test_sphere_front_hemisphere_visible_back_hidden(self):
        y = self.vertices[:, 1]
        for camera in (persp(), ortho()):
            with self.subTest(kind=camera['type']):
                visible = camera_visibility(self.vertices, self.faces, camera)
                self.assertTrue(set(np.unique(visible)) <= {0.0, 1.0})
                np.testing.assert_array_equal(visible[y < -0.3], 1.0)
                np.testing.assert_array_equal(visible[y > -0.1 if camera['type'] == 'PERSP' else y > 0.1], 0.0)

    def test_self_occlusion_by_another_part_of_the_mesh(self):
        small = (self.vertices * 0.3 + [0.0, -2.0, 0.0], self.faces)
        vertices, faces = combine((self.vertices, self.faces), small)
        big = np.arange(len(vertices)) < self.count
        front = big & (vertices[:, 1] < -0.3)
        # Perspective: occluded when the camera->vertex segment passes the small sphere.
        visible = camera_visibility(vertices, faces, persp())
        gap = segment_distance(vertices, np.array([0.0, -5.0, 0.0]), np.array([0.0, -2.0, 0.0]))
        self.assertGreater(np.count_nonzero(front & (gap < 0.25)), 20)
        np.testing.assert_array_equal(visible[front & (gap < 0.25)], 0.0)
        np.testing.assert_array_equal(visible[front & (gap > 0.35)], 1.0)
        # Orthographic: the shadow is a cylinder of radius 0.3 along the view axis.
        visible = camera_visibility(vertices, faces, ortho())
        radius = np.linalg.norm(vertices[:, [0, 2]], axis=1)
        np.testing.assert_array_equal(visible[front & (radius < 0.25)], 0.0)
        np.testing.assert_array_equal(visible[front & (radius > 0.35)], 1.0)
        # Geometry in front of the near clip plane does not occlude.
        occluder = (self.vertices * 0.3 + [0.0, -4.0, 0.0], self.faces)
        vertices, faces = combine((self.vertices, self.faces), occluder)
        visible = camera_visibility(vertices, faces, persp(clip_start=1.5))
        np.testing.assert_array_equal(visible[:self.count][self.vertices[:, 1] < -0.3], 1.0)
        np.testing.assert_array_equal(visible[self.count:], 0.0)
        visible = camera_visibility(vertices, faces, persp(clip_start=0.1))
        self.assertEqual(visible[int(np.argmin(self.vertices[:, 1]))], 0.0)

    def test_out_of_frustum_and_clip_range_are_invisible(self):
        v = self.vertices
        np.testing.assert_array_equal(camera_visibility(v, self.faces, persp(forward=(0.0, -1.0, 0.0))), 0.0)
        np.testing.assert_array_equal(camera_visibility(v, self.faces, persp(clip_end=3.5)), 0.0)
        np.testing.assert_array_equal(camera_visibility(v, self.faces, persp(position=(20.0, -5.0, 0.0))), 0.0)
        narrow = camera_visibility(v, self.faces, persp(angle=0.1))
        centre = (np.abs(v[:, 0]) < 0.1) & (np.abs(v[:, 2]) < 0.1) & (v[:, 1] < -0.9)
        outside = (np.abs(v[:, 0]) > 0.3) | (np.abs(v[:, 2]) > 0.3)
        np.testing.assert_array_equal(narrow[centre], 1.0)
        np.testing.assert_array_equal(narrow[outside], 0.0)
        narrow = camera_visibility(v, self.faces, ortho(size=0.5))
        np.testing.assert_array_equal(narrow[centre], 1.0)
        np.testing.assert_array_equal(narrow[outside], 0.0)
        # Asymmetric frustum: wide horizontally, narrow vertically.
        wide = camera_visibility(v, self.faces, persp(angle_x=1.0, angle_y=0.1))
        np.testing.assert_array_equal(wide[(np.abs(v[:, 2]) > 0.3)], 0.0)
        np.testing.assert_array_equal(wide[(np.abs(v[:, 2]) < 0.1) & (v[:, 1] < -0.5)], 1.0)

    def test_front_facing_uses_face_orientation(self):
        plane = np.array([[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        towards = np.array([[0, 1, 2], [0, 2, 3]])  # normal -Y, facing a camera at -Y
        for camera in (persp(), ortho()):
            with self.subTest(kind=camera['type']):
                np.testing.assert_array_equal(camera_visibility(plane, towards, camera), 1.0)
                np.testing.assert_array_equal(camera_visibility(plane, towards[:, ::-1], camera), 0.0)

    def test_inward_winding_is_treated_as_outward(self):
        # A mirrored object (negative determinant) or flipped normals wind a
        # closed mesh inward; front-facing must not invert (it left only the rim).
        mirrored = self.vertices * [-1.0, 1.0, 1.0]  # same faces: now wound inward
        for camera in (persp(), ortho()):
            expected = camera_visibility(self.vertices, self.faces, camera)
            with self.subTest(kind=camera['type']):
                np.testing.assert_array_equal(camera_visibility(self.vertices, self.faces[:, ::-1], camera), expected)
                visible = camera_visibility(mirrored, self.faces, camera)
                np.testing.assert_array_equal(visible[mirrored[:, 1] < -0.3], 1.0)
                np.testing.assert_array_equal(visible[mirrored[:, 1] > 0.1], 0.0)

    def test_far_from_origin_and_far_camera(self):
        # The float32 BVH made front vertices hit their own faces once the world
        # coordinates were large compared with the mesh (a 1 cm sphere at 14 m).
        y = self.vertices[:, 1]
        cases = [(1.0, (1000.0, 1000.0, 0.0), 5.0), (1.0, (5000.0, 0.0, 0.0), 5.0),
                 (0.01, (10.0, 10.0, 0.0), 5.0), (0.01, (100.0, 0.0, 0.0), 5.0),
                 (1.0, (0.0, 0.0, 0.0), 1e4)]  # last: camera 10 km away
        for radius, offset, distance in cases:
            vertices = self.vertices * radius + np.asarray(offset)
            eye = tuple(np.asarray(offset) + [0.0, -distance * radius, 0.0])
            angle = 2.0 * math.atan(1.5 / distance)
            for camera in (persp(position=eye, angle=angle, clip_start=0.01 * radius, clip_end=2 * distance * radius),
                           ortho(size=3.0 * radius, position=eye, clip_start=0.01 * radius,
                                 clip_end=2 * distance * radius)):
                with self.subTest(radius=radius, offset=offset, distance=distance, kind=camera['type']):
                    visible = camera_visibility(vertices, self.faces, camera)
                    np.testing.assert_array_equal(visible[y < -0.3], 1.0)
                    np.testing.assert_array_equal(visible[y > 0.1], 0.0)

    def test_compute_importance_with_camera(self):
        camera = persp()
        importance, features = compute_importance(self.vertices, self.faces, camera=camera,
                                                  silhouette_resolution=128)
        np.testing.assert_array_equal(features['visibility'], camera_visibility(self.vertices, self.faces, camera))
        total = sum(DEFAULT_WEIGHTS.values())
        expected = sum(DEFAULT_WEIGHTS[k] * features[k] for k in DEFAULT_WEIGHTS) / total
        np.testing.assert_allclose(importance, expected, atol=1e-12)
        _, features = compute_importance(self.vertices, self.faces, camera=camera,
                                         weights={'saliency': 1.0, 'visibility': 0.0})
        np.testing.assert_array_equal(features['visibility'], 0.0)


class BlenderCameraTests(unittest.TestCase):
    def setUp(self):
        import bpy
        self.bpy = bpy
        self.scene = bpy.context.scene
        self.scene.render.resolution_x, self.scene.render.resolution_y = 1920, 1080
        self.scene.render.pixel_aspect_x = self.scene.render.pixel_aspect_y = 1.0
        self.data = bpy.data.cameras.new('PerceptionTestCamera')
        self.object = bpy.data.objects.new('PerceptionTestCamera', self.data)
        self.scene.collection.objects.link(self.object)
        self.object.location = (0.0, -5.0, 0.0)
        self.object.rotation_euler = (math.pi / 2, 0.0, 0.0)
        bpy.context.view_layer.update()

    def tearDown(self):
        self.bpy.data.objects.remove(self.object, do_unlink=True)
        self.bpy.data.cameras.remove(self.data)

    def test_perspective_dict_matches_blender_camera(self):
        self.data.type = 'PERSP'
        self.data.lens, self.data.sensor_width, self.data.sensor_fit = 50.0, 36.0, 'AUTO'
        self.data.clip_start, self.data.clip_end = 0.2, 50.0
        camera = camera_from_object(self.object, self.scene)
        self.assertEqual(camera['type'], 'PERSP')
        np.testing.assert_allclose(camera['position'], (0.0, -5.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(camera['forward'], (0.0, 1.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(camera['up'], (0.0, 0.0, 1.0), atol=1e-6)
        half_x = math.atan(18.0 / 50.0)
        self.assertAlmostEqual(camera['angle_x'], 2 * half_x, places=5)
        self.assertAlmostEqual(camera['angle_y'], 2 * math.atan(math.tan(half_x) * 1080 / 1920), places=5)
        self.assertAlmostEqual(camera['angle_x'], self.data.angle_x, places=5)
        self.assertEqual((camera['clip_start'], camera['clip_end']), (self.data.clip_start, self.data.clip_end))
        vertices, faces = icosphere(3)
        np.testing.assert_array_equal(camera_visibility(vertices, faces, camera),
                                      camera_visibility(vertices, faces, persp(angle_x=camera['angle_x'],
                                                                               angle_y=camera['angle_y'],
                                                                               clip_start=0.2, clip_end=50.0)))

    def test_orthographic_dict_matches_blender_camera(self):
        self.data.type = 'ORTHO'
        self.data.ortho_scale = 4.0
        camera = camera_from_object(self.object, self.scene)
        self.assertEqual(camera['type'], 'ORTHO')
        self.assertAlmostEqual(camera['ortho_width'], 4.0, places=5)
        self.assertAlmostEqual(camera['ortho_height'], 4.0 * 1080 / 1920, places=5)
        vertices, faces = icosphere(3)
        visible = camera_visibility(vertices, faces, camera)
        self.assertGreater(visible.sum(), 0)
        np.testing.assert_array_equal(visible[vertices[:, 1] > 0.1], 0.0)

    def test_camera_object_scale_is_ignored(self):
        # Rendering ignores the camera object's scale; the dict used to scale
        # ortho_width/height with it (scale 2: 8 instead of 4).
        self.data.lens, self.data.sensor_width, self.data.sensor_fit = 50.0, 36.0, 'AUTO'
        self.data.ortho_scale = 4.0
        for kind in ('ORTHO', 'PERSP'):
            self.data.type = kind
            self.object.scale = (1.0, 1.0, 1.0)
            self.bpy.context.view_layer.update()
            reference = camera_from_object(self.object, self.scene)
            for scale in ((2.0, 2.0, 2.0), (2.0, 0.5, 3.0), (-1.0, 1.0, 1.0)):
                self.object.scale = scale
                self.bpy.context.view_layer.update()
                camera = camera_from_object(self.object, self.scene)
                with self.subTest(kind=kind, scale=scale):
                    self.assertEqual(set(camera), set(reference))
                    for key, value in reference.items():
                        if key == 'type':
                            self.assertEqual(camera[key], value)
                        else:
                            np.testing.assert_allclose(camera[key], value, atol=1e-6, err_msg=key)
        self.assertAlmostEqual(reference['angle_x'], self.data.angle_x, places=5)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    for case in (VisibilityTests, BlenderCameraTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
