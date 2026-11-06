"""Create an original procedural car, measured LODs and a comparison render.

blender --background --factory-startup --python-exit-code 1 --python scripts/create_demo.py

The car carries two vertex groups with different roles:
* ``LOD_Protect`` (weight 0.8, upper front: x > 2.18 and z > 0.8) is a manual
  protection input used only by the PERCEPTUAL method.
* ``LOD_EvalROI`` (weight 1.0, both headlamps and the grille, as whole parts) is
  evaluation-only: it drives the roi_*/nonroi_* distances for every method and
  is never read for protection. It overlaps the headlamps in ``LOD_Protect``, so
  the demo's ROI numbers favour PERCEPTUAL by construction; unbiased ROI results
  come from the dataset assets, which have no protection group.
"""
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import smart_lod
from smart_lod.pipeline import generate

EVALUATION_ROI = 'LOD_EvalROI'


def material(name, color, metallic=0.0, roughness=0.35):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    if bpy.app.version < (5, 0, 0):  # 5.x materials always have nodes; use_nodes goes in 6.0
        mat.use_nodes = True
    shader = mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Metallic'].default_value = metallic
    shader.inputs['Roughness'].default_value = roughness
    return mat


parts = []


def finish(obj, name, mat):
    obj.name = name
    obj.data.materials.append(mat)
    for layer in list(obj.data.uv_layers):
        obj.data.uv_layers.remove(layer)
    for face in obj.data.polygons:
        face.use_smooth = True
    parts.append(obj)
    return obj


def box(name, location, dimensions, mat, bevel=0.1):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        modifier = obj.modifiers.new('Rounded bodywork', 'BEVEL')
        modifier.width = bevel
        modifier.segments = 5
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    return finish(obj, name, mat)


def evaluation_roi(obj):
    """Put every vertex of a part in the evaluation ROI; join merges groups by name."""
    obj.vertex_groups.new(name=EVALUATION_ROI).add(range(len(obj.data.vertices)), 1.0, 'REPLACE')
    return obj


def build_car():
    """Build and join the car (geometry unchanged since v0.1) and add its vertex groups."""
    parts.clear()
    paint = material('Lagoon enamel', (0.018, 0.34, 0.40), 0.55, 0.24)
    glass = material('Midnight glass', (0.015, 0.035, 0.055), 0.38, 0.18)
    tire = material('Rubber', (0.018, 0.024, 0.032), 0, 0.75)
    alloy = material('Brushed alloy', (0.44, 0.55, 0.61), 0.8, 0.24)
    lamp = material('Warm headlamp', (0.94, 0.77, 0.35), 0.25, 0.18)
    red = material('Tail lamp', (0.55, 0.025, 0.038), 0.25, 0.25)
    box('Body', (0, 0, 0.72), (4.8, 2.05, 0.78), paint, 0.28)
    cabin = box('Cabin', (-0.25, 0, 1.37), (2.7, 1.82, 0.92), glass, 0.20)
    for vertex in cabin.data.vertices:
        height = max(0, min(1, vertex.co.z / 0.92 + 0.5))
        vertex.co.x *= 1 - height * 0.20
        vertex.co.y *= 1 - height * 0.12
    box('Roof', (-0.25, 0, 1.85), (2.18, 1.60, 0.10), paint, 0.045)
    for y in (-0.87, 0.87):
        box('Window pillar', (-0.20, y, 1.39), (0.10, 0.06, 0.75), paint, 0.02)
        box('Door handle', (-0.65, y*1.18, 1.05), (0.3, 0.055, 0.05), alloy, 0.018)
        box('Mirror', (0.87, y*1.32, 1.25), (0.36, 0.24, 0.19), paint, 0.075)
    evaluation_roi(box('Front grille', (2.39, 0, 0.64), (0.07, 1.1, 0.26), tire, 0.035))
    for y in (-0.73, 0.73):
        evaluation_roi(box('Headlight', (2.29, y, 0.92), (0.19, 0.40, 0.20), lamp, 0.065))
        box('Taillight', (-2.30, y, 0.92), (0.18, 0.37, 0.19), red, 0.06)
    for x in (-1.47, 1.46):
        for y in (-1.00, 1.00):
            bpy.ops.mesh.primitive_torus_add(major_segments=48, minor_segments=12,
                                           location=(x, y, 0.53), rotation=(math.pi/2, 0, 0),
                                           major_radius=0.385, minor_radius=0.15)
            finish(bpy.context.object, 'Tire', tire)
            bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.30, depth=0.065,
                                               location=(x, y*1.14, 0.53), rotation=(math.pi/2, 0, 0))
            hub = finish(bpy.context.object, 'Wheel alloy', alloy)
            bevel = hub.modifiers.new('Wheel rim', 'BEVEL')
            bevel.width = 0.025
            bevel.segments = 3
            bpy.ops.object.modifier_apply(modifier=bevel.name)
    bpy.ops.object.select_all(action='DESELECT')
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    source = bpy.context.object
    source.name = 'Car_HighPoly'
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    group = source.vertex_groups.new(name='LOD_Protect')
    important = [v.index for v in source.data.vertices if v.co.x > 2.18 and v.co.z > 0.8]
    group.add(important, 0.8, 'REPLACE')
    return source


def main():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    smart_lod.register()
    source = build_car()
    settings = bpy.context.scene.smart_lod
    # Only what the demo depends on is set here; importance weights and lambda keep
    # their v0.2 defaults, which the report records under 'settings'.
    settings.weighting = 'QUADRIC'
    settings.protect_group = 'LOD_Protect'
    settings.roi_group = EVALUATION_ROI
    settings.sample_count = 3000
    settings.resolution = 128
    settings.show_research = True
    report, outputs = generate(bpy.context, source, settings, compare=True)
    destination = ROOT / 'examples'
    destination.mkdir(exist_ok=True)
    (destination / 'demo_benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    bpy.ops.smart_lod.export_report(filepath=str(destination / 'demo_benchmark.csv'), format='CSV')
    # A display-only gallery is separate from the measured objects so evaluation
    # retains their original coordinate alignment and provenance.
    gallery = bpy.data.collections.new('Comparison Gallery (display copies)')
    bpy.context.scene.collection.children.link(gallery)
    chosen = [outputs[0]] + [obj for obj in outputs if obj.get('smart_lod_method') == 'PERCEPTUAL']
    white = material('Typography', (0.72, 0.83, 0.88), 0, 0.6)
    for i, original in enumerate(chosen):
        copy = bpy.data.objects.new('Display_' + original.name, original.data)
        gallery.objects.link(copy)
        copy.location = (-3.2 if i % 2 == 0 else 3.2, 2.2 if i < 2 else -2.2, 0)
        copy.rotation_euler.z = math.radians(-12)
        count = report['source_triangles'] if i == 0 else original['smart_lod_triangles']
        curve = bpy.data.curves.new('LOD label', 'FONT')
        curve.body = f'LOD {i}  /  {count:,} tris'
        curve.size = 0.25
        curve.align_x = 'CENTER'
        label = bpy.data.objects.new(curve.body, curve)
        gallery.objects.link(label)
        label.location = (copy.location.x, copy.location.y - 1.65, 0.03)
        label.data.materials.append(white)
    source.hide_render = True
    source.hide_set(True)
    world = bpy.context.scene.world
    world.color = (0.05, 0.05, 0.05)
    bpy.ops.object.camera_add(location=(10, -16, 18))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0, 0.2))-camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 16.4
    bpy.context.scene.camera = camera
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.studiolight_rotate_z = 0.25
    scene.display.shading.color_type = 'MATERIAL'
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = 'BOTH'
    scene.display.shading.background_type = 'WORLD'
    scene.world.color = (0.014, 0.021, 0.036)
    scene.render.resolution_x = 1500
    scene.render.resolution_y = 1100
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(destination / 'lod_preview.png')
    bpy.ops.render.render(write_still=True)
    # Save an uncluttered working file: source visible and gallery disabled.
    gallery.hide_viewport = True
    gallery.hide_render = True
    source.hide_render = False
    source.hide_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    source.select_set(True)
    bpy.context.view_layer.objects.active = source
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_distance = 8
                area.spaces.active.region_3d.view_location = (0, 0, 0.9)
                area.spaces.active.overlay.show_floor = False
                area.spaces.active.show_region_ui = True
    bpy.ops.wm.save_as_mainfile(filepath=str(destination / 'SmartLOD_Demo.blend'))
    print('DEMO_REPORT', json.dumps([dict(method=r['method'], lod=r['lod'], triangles=r['actual_triangles'],
                                           seconds=round(r['processing_seconds'], 3),
                                           iou_axis=round(r['silhouette_iou_mean'], 4),
                                           iou_multiview=round(r['silhouette_iou_multiview_mean'], 4),
                                           roi_chamfer=r.get('roi_chamfer_ref_to_candidate_normalized'))
                                      for r in report['results']]))


if __name__ == '__main__':
    main()
