"""Build the Smart LOD research dataset (run inside Blender, background mode).

From the project root, after ``python scripts/fetch_assets.py``::

    blender --background --factory-startup --python-exit-code 1 --python scripts/make_dataset.py
    ... --python scripts/make_dataset.py -- --only car,bunny      # rebuild a subset, merge manifest
    ... --python scripts/make_dataset.py -- --verify-only         # re-open every .blend and check it
    ... --python scripts/make_dataset.py -- --check-renders DIR   # six ROI-tinted views per asset

Every ``dataset/assets/<id>.blend`` holds exactly one mesh object ``<id>_HighPoly``
with identity transform, Z up, front towards -Y, bounding box centred at the
origin and longest bounding-box side 2.0. All sources are triangulated, so every
method starts from the same triangle mesh. The optional vertex group
``LOD_EvalROI`` (weight 1.0) marks a semantically important region defined by
simple geometric rules; it is for evaluation only and must never be passed to a
simplifier as protection input. ``dataset/manifest.json`` records provenance,
preprocessing, topology and ROI statistics; ``dataset/preview.png`` is a
Workbench contact sheet with the ROI tinted orange.
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'dataset'
ASSET_DIR = DATASET / 'assets'
EXTERNAL = DATASET / 'external'
MANIFEST = DATASET / 'manifest.json'
PREVIEW = DATASET / 'preview.png'
ROI_GROUP = 'LOD_EvalROI'
TRIANGLE_CAP = 100_000
LONGEST_SIDE = 2.0
PROCEDURAL = 'procedural:scripts/make_dataset.py'
OBJ_IMPORT = dict(forward_axis='NEGATIVE_Z', up_axis='Y', use_split_objects=False,
                  use_split_groups=False, validate_meshes=True)
# The importer stores this axis conversion as an object rotation (+90 deg about X,
# with float noise); it is baked exactly instead: Blender (x, y, z) = OBJ (x, -z, y).
OBJ_AXES = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
CATEGORIES = {
    'hard_surface': 'vehicle, product or CAD part with sharp edges',
    'character_face': 'face, bust or character whose face carries meaning',
    'organic': 'scan or animal with a complex silhouette',
}

# rotate: XYZ Euler degrees applied after the import axis conversion (world axes, X then Y then Z).
ASSETS = [
    dict(id='car', category='hard_surface', split='tuning', build='car', rotate=(0, 0, -90),
         front='headlamps and grille; built facing +X in scripts/create_demo.py'),
    dict(id='suzanne', category='character_face', split='tuning', build='suzanne', rotate=(0, 0, 0),
         front='face; the Blender primitive already faces -Y'),
    dict(id='spot', category='organic', split='tuning', obj='spot', rotate=(0, 0, 180), front='head'),
    dict(id='beetle', category='hard_surface', split='evaluation', obj='beetle', rotate=(0, 0, 0),
         subdivide=2, front='bonnet with the two headlamp openings'),
    dict(id='fandisk', category='hard_surface', split='evaluation', obj='fandisk', rotate=(0, 0, 0),
         front='none (CAD part); import orientation kept'),
    dict(id='rocker_arm', category='hard_surface', split='evaluation', obj='rocker-arm', rotate=(0, 0, 0),
         front='none (CAD part); import orientation kept'),
    dict(id='max_planck', category='character_face', split='evaluation', obj='max-planck', rotate=(0, 0, 180),
         front='face'),
    dict(id='igea', category='character_face', split='evaluation', obj='igea', rotate=(0, 0, 0), front='face'),
    dict(id='ogre', category='character_face', split='evaluation', obj='ogre', rotate=(0, 0, 0),
         merge=1e-6, front='face'),
    dict(id='bunny', category='organic', split='evaluation', obj='stanford-bunny', rotate=(0, 0, 60),
         front='gaze of the head; the body axis stays diagonal'),
    dict(id='armadillo', category='organic', split='evaluation', obj='armadillo', rotate=(0, 0, 180),
         front='face'),
    dict(id='horse', category='organic', split='evaluation', obj='horse', rotate=(-90, 0, 180),
         front='head; the source file is Z-up with the head towards +Y'),
]

# Evaluation ROI rules in normalised coordinates. A vertex belongs to the ROI when
# it satisfies any 'union' term and no 'exclude' term. A term is a sphere
# [cx, cy, cz, r], an axis-aligned ellipsoid {center, radii}, a box [min, max] or
# 'components_in_box' [min, max] (every vertex of each connected component whose
# bounding-box centre lies in the box), optionally restricted by a vertex-normal
# filter normal . d >= t. Rules were placed from measured landmarks and checked
# on ROI-tinted renders (front, back, both sides, top, 3/4).
FRONT = [0, -1, 0]
ROI = {
    'car': dict(
        description='separate parts: both headlamps and the grille (component centres at the front) and the '
                    'four tyres and four wheel hubs (component centres at the axles)',
        union=[dict(components_in_box=[[-0.5, -1.01, -0.2], [0.5, -0.85, 0.05]]),
               dict(components_in_box=[[0.35, -0.7, -0.2], [0.5, 0.7, -0.15]]),
               dict(components_in_box=[[-0.5, -0.7, -0.2], [-0.35, 0.7, -0.15]])]),
    'suzanne': dict(
        description='face: eye sockets and eyes, brow, nose and mouth; front-facing part of an ellipsoid '
                    'over the face, ears excluded',
        union=[dict(ellipsoid=dict(center=[0, -0.62, -0.13], radii=[0.64, 0.4, 0.66]), normal=[FRONT, 0.2])]),
    'spot': dict(
        description='head incl. horns and ears: everything in front of y = -0.44 above z = -0.07, plus '
                    'everything in front of y = -0.30 above z = 0.5 (ears, horns, top of the head)',
        union=[dict(box=[[-1.01, -1.01, -0.07], [1.01, -0.44, 1.01]]),
               dict(box=[[-1.01, -1.01, 0.5], [1.01, -0.3, 1.01]])]),
    'beetle': dict(
        description='the two front headlamp openings with their fender surround (the shell has no wheels '
                    'and no grille)',
        union=[dict(sphere=[0.29, -0.9, -0.15, 0.13]), dict(sphere=[-0.29, -0.9, -0.15, 0.13])]),
    'max_planck': dict(
        description='face from brows to chin (eyes, nose, mouth, cheeks); front-facing part of an ellipsoid, '
                    'ears and neck excluded',
        union=[dict(ellipsoid=dict(center=[0, -0.55, -0.02], radii=[0.42, 0.3, 0.5]), normal=[FRONT, 0.0])]),
    'igea': dict(
        description='face from brows to chin (eyes, nose, mouth, cheeks); front-facing part of an ellipsoid, '
                    'hair excluded',
        union=[dict(ellipsoid=dict(center=[0, -0.9, -0.3], radii=[0.42, 0.45, 0.6]), normal=[FRONT, 0.2])]),
    'ogre': dict(
        description='head incl. ears (character): ellipsoid above the neck',
        union=[dict(ellipsoid=dict(center=[0, -0.08, 0.8], radii=[0.23, 0.26, 0.24]))]),
    'bunny': dict(
        description='head incl. both ears: everything above the back (z >= 0.32) plus an ellipsoid over the '
                    'lower head',
        union=[dict(box=[[-1.01, -1.01, 0.32], [1.01, 1.01, 1.01]]),
               dict(ellipsoid=dict(center=[0.03, -0.72, 0.25], radii=[0.4, 0.32, 0.3]))]),
    'armadillo': dict(
        description='head incl. snout and both ears: ellipsoid in front of the shell plus one sphere per ear',
        union=[dict(ellipsoid=dict(center=[0, -0.38, 0.7], radii=[0.26, 0.3, 0.34])),
               dict(sphere=[0.22, -0.33, 0.9, 0.14]), dict(sphere=[-0.22, -0.33, 0.9, 0.14])]),
    'horse': dict(
        description='head incl. ears: everything in front of y = -0.70 above z = 0.30 (throat latch/poll)',
        union=[dict(box=[[-1.01, -1.01, 0.3], [1.01, -0.7, 1.01]])]),
}


# --------------------------------------------------------------------------- utilities

def coordinates(mesh):
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get('co', co)
    return co.reshape(-1, 3)


def vertex_normals(mesh):
    normals = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get('normal', normals)
    return normals.reshape(-1, 3)


def triangles(mesh):
    mesh.calc_loop_triangles()
    tri = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get('vertices', tri)
    return tri.reshape(-1, 3)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def geometry_sha256(obj):
    """Same formula as the report field ``source_sha256`` of smart_lod.pipeline (v0.1)."""
    mesh = obj.data
    vertices = coordinates(mesh)
    transform = np.asarray(obj.matrix_world, dtype=float)
    vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
    faces = triangles(mesh).astype(np.int64)
    return hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()


def component_labels(count, edges):
    """Connected vertex components by vectorised union-find (hooking + pointer jumping)."""
    parent = np.arange(count)
    while True:
        a, b = parent[edges[:, 0]], parent[edges[:, 1]]
        different = a != b
        if not different.any():
            return parent
        np.minimum.at(parent, np.maximum(a, b)[different], np.minimum(a, b)[different])
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped


def rounded(values, digits=6):
    return [round(float(v), digits) for v in values]


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def apply_modifier(obj, kind, **properties):
    modifier = obj.modifiers.new(kind.title(), kind)
    for key, value in properties.items():
        setattr(modifier, key, value)
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.object.modifier_apply(modifier=modifier.name)


# --------------------------------------------------------------------------- sources

def build_car(steps):
    """Procedural car. Construction copied from scripts/create_demo.py (v0.1); keep in sync by hand."""
    parts = []

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
    box('Front grille', (2.39, 0, 0.64), (0.07, 1.1, 0.26), tire, 0.035)
    for y in (-0.73, 0.73):
        box('Headlight', (2.29, y, 0.92), (0.19, 0.40, 0.20), lamp, 0.065)
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
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    steps.append(dict(step='procedural_build', generator=PROCEDURAL + ':build_car',
                      note='construction copied from scripts/create_demo.py v0.1 (22 separate parts: 14 '
                           'bevelled boxes, 4 tori, 4 bevelled cylinders, joined); the demo LOD_Protect '
                           'group is not created; the bevel on the thin grille leaves 24 zero-area polygons '
                           '(48 zero-area triangles), as in the v0.1 demo car'))
    return source


def build_suzanne(steps):
    bpy.ops.mesh.primitive_monkey_add(size=2, calc_uvs=True)
    obj = bpy.context.object
    before = len(triangles(obj.data))
    apply_modifier(obj, 'SUBSURF', subdivision_type='CATMULL_CLARK', levels=2)
    steps.append(dict(step='procedural_build', generator=PROCEDURAL + ':build_suzanne',
                      note='bpy.ops.mesh.primitive_monkey_add(size=2, calc_uvs=True)'))
    steps.append(dict(step='subdivide', method='Subdivision Surface modifier (Catmull-Clark) applied',
                      levels=2, quality=3, uv_smooth='PRESERVE_BOUNDARIES', boundary_smooth='ALL',
                      triangles_before=before, triangles_after=len(triangles(obj.data))))
    return obj


BUILDERS = {'car': build_car, 'suzanne': build_suzanne}


def load_sources():
    path = EXTERNAL / 'SOURCES.json'
    if not path.exists():
        raise SystemExit(f'{path} is missing: run  python scripts/fetch_assets.py  first')
    return {entry['name']: entry for entry in json.loads(path.read_text('utf-8'))['models']}


def import_obj(spec, sources, steps):
    record = sources.get(spec['obj'])
    path = EXTERNAL / f"{spec['obj']}.obj"
    if record is None or not path.exists():
        raise SystemExit(f"{spec['obj']} is not downloaded: run  python scripts/fetch_assets.py")
    digest = sha256_file(path)
    if digest != record['sha256']:
        raise SystemExit(f'{path} sha256 {digest} does not match SOURCES.json {record["sha256"]}')
    obj_vertices = obj_faces = 0
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            obj_vertices += line.startswith('v ')
            obj_faces += line.startswith('f ')
    bpy.ops.wm.obj_import(filepath=str(path), **OBJ_IMPORT)
    objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if len(objects) != 1:
        raise RuntimeError(f"{spec['obj']}: expected one imported mesh, got {len(objects)}")
    obj = objects[0]
    if not np.allclose(np.asarray(obj.matrix_world)[:3, :3], np.asarray(OBJ_AXES), atol=1e-6):
        raise RuntimeError('OBJ importer axis convention changed; update OBJ_AXES')
    obj.matrix_world = Matrix.Identity(4)
    obj.data.transform(OBJ_AXES.to_4x4())
    steps.append(dict(step='obj_import', operator='bpy.ops.wm.obj_import', **OBJ_IMPORT,
                      obj_vertices=obj_vertices, obj_faces=obj_faces, imported_vertices=len(obj.data.vertices),
                      imported_polygons=len(obj.data.polygons),
                      note='the importer keeps only vertices referenced by faces' if
                           len(obj.data.vertices) < obj_vertices else None,
                      axis_conversion='Blender (x, y, z) = OBJ (x, -z, y), baked into vertices'))
    if obj.data.has_custom_normals:
        with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        steps.append(dict(step='clear_custom_normals', note='imported OBJ vn normals are not kept; '
                                                             'shading uses computed normals'))
    return obj


# --------------------------------------------------------------------------- preprocessing

def delete_loose(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    edges = [edge for edge in bm.edges if not edge.link_faces]
    if edges:
        bmesh.ops.delete(bm, geom=edges, context='EDGES')
    verts = [vert for vert in bm.verts if not vert.link_faces]
    if verts:
        bmesh.ops.delete(bm, geom=verts, context='VERTS')
    if edges or verts:
        bm.to_mesh(mesh)
    bm.free()
    return len(verts), len(edges)


def normalize(mesh):
    co = coordinates(mesh)
    lower, upper = co.min(axis=0), co.max(axis=0)
    center = (lower + upper) / 2
    scale = LONGEST_SIDE / float((upper - lower).max())
    mesh.transform(Matrix.Diagonal((scale, scale, scale, 1.0)) @ Matrix.Translation(-Vector(center)))
    mesh.update()
    return dict(step='normalize', bbox_center_before=rounded(center), bbox_size_before=rounded(upper - lower),
                uniform_scale=round(scale, 9), longest_side_after=LONGEST_SIDE)


def merge_by_distance(mesh, distance):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=distance)
    merged = before - len(bm.verts)
    bm.to_mesh(mesh)
    bm.free()
    return merged


def cap_triangles(obj):
    """Blender Decimate (Collapse, triangulate) until at most TRIANGLE_CAP triangles remain."""
    before = len(triangles(obj.data))
    if before <= TRIANGLE_CAP:
        return None
    original = obj.data
    ratio = TRIANGLE_CAP / before
    for attempt in range(1, 9):
        obj.data = original.copy()
        apply_modifier(obj, 'DECIMATE', decimate_type='COLLAPSE', ratio=ratio, use_collapse_triangulate=True)
        after = len(triangles(obj.data))
        if after <= TRIANGLE_CAP:
            break
        rejected, obj.data = obj.data, original
        bpy.data.meshes.remove(rejected)
        ratio *= TRIANGLE_CAP / after * 0.9999
    else:
        raise RuntimeError('Decimate cap did not converge')
    bpy.data.meshes.remove(original)
    return dict(step='decimate_cap', method='Blender Decimate modifier, Collapse, use_collapse_triangulate=True',
                ratio=round(ratio, 9), attempts=attempt, triangles_before=before, triangles_after=after,
                note='all methods share this preprocessed source')


def triangulate(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    polygons = len(bm.faces)
    changed = [face for face in bm.faces if len(face.verts) > 3]
    if changed:
        bmesh.ops.triangulate(bm, faces=changed, quad_method='BEAUTY', ngon_method='BEAUTY')
        bm.to_mesh(mesh)
    triangles_after = len(bm.faces)
    bm.free()
    return dict(step='triangulate', method='bmesh.ops.triangulate(quad_method=BEAUTY, ngon_method=BEAUTY)',
                polygons_before=polygons, polygons_split=len(changed), triangles_after=triangles_after)


def mesh_edges(mesh):
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get('vertices', edges)
    return edges.reshape(-1, 2)


def roi_mask(co, normals, rule, labels):
    def term(entry):
        if 'components_in_box' in entry:
            # Whole connected components whose bounding-box centre lies in the box.
            lower, upper = np.array(entry['components_in_box'][0]), np.array(entry['components_in_box'][1])
            _, inverse = np.unique(labels, return_inverse=True)
            count = inverse.max() + 1
            low = np.full((count, 3), np.inf)
            high = np.full((count, 3), -np.inf)
            np.minimum.at(low, inverse, co)
            np.maximum.at(high, inverse, co)
            centre = (low + high) / 2
            inside = np.all((centre >= lower) & (centre <= upper), axis=1)[inverse]
        elif 'sphere' in entry:
            center, radius = np.array(entry['sphere'][:3]), entry['sphere'][3]
            inside = ((co - center) ** 2).sum(axis=1) <= radius * radius
        elif 'ellipsoid' in entry:
            center, radii = np.array(entry['ellipsoid']['center']), np.array(entry['ellipsoid']['radii'])
            inside = (((co - center) / radii) ** 2).sum(axis=1) <= 1.0
        else:
            lower, upper = np.array(entry['box'][0]), np.array(entry['box'][1])
            inside = np.all((co >= lower) & (co <= upper), axis=1)
        if 'normal' in entry:
            direction = np.array(entry['normal'][0], dtype=float)
            inside &= normals @ (direction / np.linalg.norm(direction)) >= entry['normal'][1]
        return inside

    mask = np.zeros(len(co), dtype=bool)
    for entry in rule['union']:
        mask |= term(entry)
    for entry in rule.get('exclude', []):
        mask &= ~term(entry)
    return mask


def rule_text(rule):
    def fmt(values):
        return '(' + ', '.join(f'{v:g}' for v in values) + ')'

    def term(entry):
        if 'components_in_box' in entry:
            box = entry['components_in_box']
            text = f'connected components with bbox centre in box {fmt(box[0])}..{fmt(box[1])}'
        elif 'sphere' in entry:
            text = f"sphere c={fmt(entry['sphere'][:3])} r={entry['sphere'][3]:g}"
        elif 'ellipsoid' in entry:
            text = f"ellipsoid c={fmt(entry['ellipsoid']['center'])} radii={fmt(entry['ellipsoid']['radii'])}"
        else:
            text = f"box {fmt(entry['box'][0])}..{fmt(entry['box'][1])}"
        if 'normal' in entry:
            text += f" with normal.{fmt(entry['normal'][0])} >= {entry['normal'][1]:g}"
        return text

    text = 'ANY[' + '; '.join(term(entry) for entry in rule['union']) + ']'
    if rule.get('exclude'):
        text += ' AND NOT ANY[' + '; '.join(term(entry) for entry in rule['exclude']) + ']'
    return text


def area_fraction(co, tri, mask):
    """Integral of the linearly interpolated 0/1 ROI weight over the surface / total area."""
    corners = co[tri]
    areas = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    return float((areas * mask[tri].mean(axis=1)).sum() / areas.sum())


def mesh_stats(obj):
    mesh = obj.data
    co = coordinates(mesh)
    tri = triangles(mesh)
    edges = mesh_edges(mesh)
    loop_edges = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get('edge_index', loop_edges)
    faces_per_edge = np.bincount(loop_edges, minlength=len(mesh.edges))
    corners = co[tri]
    area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    return dict(
        triangles=len(tri), vertices=len(co), edges=len(edges), polygons=len(mesh.polygons),
        connected_components=int(np.unique(component_labels(len(co), edges)).size), has_uv=len(mesh.uv_layers) > 0,
        uv_layers=[layer.name for layer in mesh.uv_layers], materials=len(mesh.materials),
        attributes=sorted(a.name for a in mesh.attributes if not a.name.startswith('.')),
        manifold_edge_fraction=round(float((faces_per_edge == 2).mean()), 6),
        boundary_edges=int((faces_per_edge == 1).sum()), nonmanifold_edges=int((faces_per_edge > 2).sum()),
        degenerate_triangles=int((area <= 1e-12).sum()), surface_area=round(float(area.sum()), 6),
        bbox_min=rounded(co.min(axis=0)), bbox_max=rounded(co.max(axis=0)),
        geometry_sha256=geometry_sha256(obj),
    )


# --------------------------------------------------------------------------- build

def build_asset(spec, sources):
    reset()
    steps = []
    if 'build' in spec:
        obj = BUILDERS[spec['build']](steps)
    else:
        obj = import_obj(spec, sources, steps)
    name = f"{spec['id']}_HighPoly"
    obj.name = obj.data.name = name
    mesh = obj.data
    loose_vertices, loose_edges = delete_loose(mesh)
    if loose_vertices or loose_edges:
        steps.append(dict(step='delete_loose', removed_vertices=loose_vertices, removed_edges=loose_edges,
                          note='vertices/edges not used by any face'))
    rotation = Euler([math.radians(angle) for angle in spec['rotate']], 'XYZ').to_matrix().to_4x4()
    mesh.transform(rotation)
    steps.append(dict(step='orient', rotation_euler_xyz_degrees=list(spec['rotate'])))
    steps.append(normalize(mesh))
    if spec.get('merge'):
        merged = merge_by_distance(mesh, spec['merge'])
        steps.append(dict(step='merge_by_distance', distance=spec['merge'], units='normalised',
                          merged_vertices=merged,
                          note='raw file duplicates vertices at identical positions, opening seams'))
    if spec.get('subdivide'):
        before = len(triangles(mesh))
        apply_modifier(obj, 'SUBSURF', subdivision_type='CATMULL_CLARK', levels=spec['subdivide'])
        mesh = obj.data
        steps.append(dict(step='subdivide', method='Subdivision Surface modifier (Catmull-Clark) applied',
                          levels=spec['subdivide'], quality=3, boundary_smooth='ALL',
                          triangles_before=before, triangles_after=len(triangles(mesh)),
                          note='low-poly source turned into a high-poly source'))
    capped = cap_triangles(obj)
    mesh = obj.data
    mesh.name = name
    if capped:
        steps.append(capped)
    steps.append(triangulate(mesh))
    mesh.polygons.foreach_set('use_smooth', np.ones(len(mesh.polygons), dtype=bool))
    steps.append(dict(step='shade_smooth', note='display only; no sharp-edge or seam marks are added'))
    steps.append(normalize(mesh))
    mesh.update()
    rule = ROI.get(spec['id'])
    roi = None
    if rule:
        co, normals = coordinates(mesh), vertex_normals(mesh)
        mask = roi_mask(co, normals, rule, component_labels(len(co), mesh_edges(mesh)))
        if not mask.any():
            raise RuntimeError(f"{spec['id']}: ROI rule selects no vertex")
        obj.vertex_groups.new(name=ROI_GROUP).add(np.flatnonzero(mask).tolist(), 1.0, 'REPLACE')
        roi = dict(group=ROI_GROUP, description=rule['description'], rule=rule_text(rule),
                   rule_spec=dict(union=rule['union'], exclude=rule.get('exclude', [])),
                   vertices=int(mask.sum()), vertex_fraction=round(float(mask.mean()), 6),
                   area_fraction=round(area_fraction(co, triangles(mesh), mask), 6))
    stats = mesh_stats(obj)
    for other in list(bpy.data.objects):
        if other != obj:
            bpy.data.objects.remove(other)
    bpy.data.orphans_purge(do_recursive=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / f"{spec['id']}.blend"
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True, check_existing=False)
    if 'build' in spec:
        source = dict(type='procedural', generator=f"{PROCEDURAL}:build_{spec['build']}")
        license_note = ('Original procedural geometry generated by this project (GPL-3.0-or-later code).'
                        if spec['build'] == 'car' else
                        'Blender built-in Suzanne primitive generated by this script; no external file.')
    else:
        record = sources[spec['obj']]
        source = dict(type='external', name=spec['obj'], url=record['url'], sha256=record['sha256'],
                      credit=record['source_credit'], source_link=record['source_link'],
                      first_known_appearance=record['first_known_appearance'],
                      collection=record['collection'], retrieved_at=record['retrieved_at'])
        license_note = record['usage_note'] + ('. ' + record['note'] if record.get('note') else '')
    entry = dict(
        file=f"assets/{spec['id']}.blend", object=name, category=spec['category'], split=spec['split'],
        source=source, license_note=license_note, preprocessing=steps,
        orientation=dict(up='+Z', front='-Y', front_feature=spec['front'],
                         rotation_euler_xyz_degrees=list(spec['rotate']),
                         applied_after='OBJ import axis conversion' if 'obj' in spec else 'procedural build'),
        **stats, roi=roi)
    print(f"built {spec['id']:11s} {stats['triangles']:>7,d} tris  {stats['vertices']:>6,d} verts  "
          f"ROI {roi['vertices'] if roi else '-'}")
    return entry


# --------------------------------------------------------------------------- rendering

BACKGROUND = (0.035, 0.04, 0.05)
BASE_COLOR = (0.62, 0.64, 0.66, 1.0)
ROI_COLOR = (1.0, 0.42, 0.08, 1.0)
TEXT_COLOR = (1.0, 1.0, 1.0, 1.0)


def camera_basis(direction):
    back = Vector(direction).normalized()
    up = Vector((0, 1, 0)) if abs(back.z) > 0.99 else Vector((0, 0, 1))
    right = up.cross(back).normalized()
    return Matrix((right, back.cross(right), back)).transposed()


def render_setup():
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    shading = scene.display.shading
    shading.light = 'STUDIO'
    shading.color_type = 'VERTEX'
    shading.show_cavity = True
    shading.cavity_type = 'BOTH'
    shading.show_shadows = False
    scene.view_settings.view_transform = 'Standard'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new('Preview background')
    scene.world.color = BACKGROUND
    camera = bpy.data.objects.new('Preview camera', bpy.data.cameras.new('Preview camera'))
    camera.data.type = 'ORTHO'
    camera.data.clip_end = 50
    scene.collection.objects.link(camera)
    scene.camera = camera
    return scene, camera


def colour_mesh(mesh, colors):
    attribute = mesh.color_attributes.new('Preview colour', 'FLOAT_COLOR', 'POINT')
    attribute.data.foreach_set('color', np.asarray(colors, dtype=np.float32).ravel())
    mesh.color_attributes.active_color = attribute
    mesh.color_attributes.render_color_index = mesh.color_attributes.active_color_index


def text_object(body, size, camera, location):
    curve = bpy.data.curves.new('Label', 'FONT')
    curve.body = body
    curve.size = size
    curve.align_x = 'CENTER'
    curve.align_y = 'CENTER'
    holder = bpy.data.objects.new('Label', curve)
    bpy.context.scene.collection.objects.link(holder)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(holder.evaluated_get(depsgraph))
    bpy.data.objects.remove(holder)
    colour_mesh(mesh, np.tile(TEXT_COLOR, (len(mesh.vertices), 1)))
    label = bpy.data.objects.new('Label', mesh)
    bpy.context.scene.collection.objects.link(label)
    label.parent = camera
    label.location = location
    return label


def render_pixels(scene, size, labels, tmp):
    """Render the scene through a temporary PNG; returns RGBA pixels, top row first."""
    width, height = size
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(tmp / 'tile.png')
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(tmp / 'tile.png'), check_existing=False)
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    for item in labels:
        mesh = item.data
        bpy.data.objects.remove(item)
        bpy.data.meshes.remove(mesh)
    return pixels.reshape(height, width, 4)[::-1]


def label_size(label, frame_width, band_height):
    lines = label.split('\n')
    return min(band_height / (len(lines) * 1.3), 0.92 * frame_width / (0.56 * max(map(len, lines))))


def render_tile(scene, camera, obj, direction, size, label='', band=0.0, tmp=None):
    """Render ``obj`` framed from ``direction`` with an optional label band at the bottom."""
    width, height = size
    basis = camera_basis(direction)
    projected = coordinates(obj.data) @ np.asarray(basis)
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    aspect = height / width
    ortho = max((upper[0] - lower[0]) / 0.9, (upper[1] - lower[1]) / (0.9 * (1 - band) * aspect))
    frame_height = ortho * aspect
    centre = (lower + upper) / 2
    camera.data.ortho_scale = ortho
    camera.matrix_world = (Matrix.Translation(basis @ Vector((centre[0], centre[1] - band * frame_height / 2, 20)))
                           @ basis.to_4x4())
    labels = []
    if label:
        text_size = label_size(label, ortho, band * frame_height)
        labels.append(text_object(label, text_size, camera, (0, -frame_height / 2 + band * frame_height / 2, -1)))
    return render_pixels(scene, size, labels, tmp)


def render_text(scene, camera, text, size, tmp):
    width, height = size
    camera.data.ortho_scale = 1.0
    camera.matrix_world = Matrix.Translation((0, -20, 0)) @ camera_basis((0, -1, 0)).to_4x4()
    label = text_object(text, label_size(text, 1.0, 0.6 * height / width), camera, (0, 0, -1))
    return render_pixels(scene, size, [label], tmp)


def save_png(rows, path):
    sheet = np.ascontiguousarray(rows[::-1])
    image = bpy.data.images.new(path.stem, sheet.shape[1], sheet.shape[0], alpha=True)
    image.pixels.foreach_set(sheet.ravel())
    image.filepath_raw = str(path)
    image.file_format = 'PNG'
    image.save()
    bpy.data.images.remove(image)


def display_object(entry):
    """Append the saved asset (verifies the .blend loads) and tint its ROI."""
    path = DATASET / entry['file']
    with bpy.data.libraries.load(str(path), link=False) as (source, target):
        target.objects = [entry['object']]
    obj = target.objects[0]
    if obj is None:
        raise RuntimeError(f"{path} lacks object {entry['object']}")
    bpy.context.scene.collection.objects.link(obj)
    mask = np.zeros(len(obj.data.vertices), dtype=bool)
    group = obj.vertex_groups.get(ROI_GROUP)
    if group:
        for vertex in obj.data.vertices:
            for membership in vertex.groups:
                if membership.group == group.index and membership.weight > 0:
                    mask[vertex.index] = True
    colour_mesh(obj.data, np.where(mask[:, None], ROI_COLOR, BASE_COLOR))
    return obj, mask


def make_preview(manifest, tmp, columns=4, tile=(480, 440)):
    reset()
    scene, camera = render_setup()
    tiles = []
    for asset_id, entry in manifest['assets'].items():
        obj, mask = display_object(entry)
        roi = f"ROI {entry['roi']['vertices']:,} v" if entry['roi'] else 'no ROI'
        label = f"{asset_id}\n{entry['category']} / {entry['split']}\n{entry['triangles']:,} tris / {roi}"
        tiles.append(render_tile(scene, camera, obj, (0.62, -1.0, 0.42), tile, label, band=0.26, tmp=tmp))
        bpy.data.objects.remove(obj)
    blank = np.empty_like(tiles[0])
    blank[:] = tiles[0][0, 0]
    tiles += [blank] * (-len(tiles) % columns)
    rows = [np.concatenate(tiles[i:i + columns], axis=1) for i in range(0, len(tiles), columns)]
    header = render_text(scene, camera, 'Smart LOD dataset  |  orange = LOD_EvalROI (evaluation only, never '
                         'protection input)  |  Workbench, 3/4 front view', (tile[0] * columns, 56), tmp)
    save_png(np.concatenate([header] + rows, axis=0), PREVIEW)
    print('wrote', PREVIEW.relative_to(ROOT))


CHECK_VIEWS = [('front -Y', (0, -1, 0)), ('back +Y', (0, 1, 0)), ('right +X', (1, 0, 0)),
               ('left -X', (-1, 0, 0)), ('top +Z', (0, 0, 1)), ('3/4 front', (0.62, -1.0, 0.42))]


def make_check_renders(manifest, directory, tmp, ids=None, size=(360, 360)):
    directory.mkdir(parents=True, exist_ok=True)
    reset()
    scene, camera = render_setup()
    for asset_id, entry in manifest['assets'].items():
        if ids and asset_id not in ids:
            continue
        obj, mask = display_object(entry)
        tiles = [render_tile(scene, camera, obj, direction, size, f'{asset_id}  {name}', band=0.1, tmp=tmp)
                 for name, direction in CHECK_VIEWS]
        rows = [np.concatenate(tiles[i:i + 3], axis=1) for i in (0, 3)]
        save_png(np.concatenate(rows, axis=0), directory / f'{asset_id}.png')
        bpy.data.objects.remove(obj)
    print('wrote check renders to', directory)


# --------------------------------------------------------------------------- verification

def verify(manifest):
    try:
        sys.path.insert(0, str(ROOT))
        from smart_lod.geometry import snapshot
    except Exception as error:  # the extension package is optional for this check
        snapshot = None
        print('note: smart_lod.geometry unavailable, pipeline hash check skipped:', error)
    failures = []
    for asset_id, entry in manifest['assets'].items():
        bpy.ops.wm.open_mainfile(filepath=str(DATASET / entry['file']), load_ui=False)
        meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        problems = []
        if [obj.name for obj in meshes] != [entry['object']]:
            problems.append(f'mesh objects {[obj.name for obj in meshes]}')
        obj = bpy.data.objects.get(entry['object'])
        if obj is None:
            failures.append((asset_id, problems))
            continue
        if not np.allclose(np.asarray(obj.matrix_world), np.eye(4)):
            problems.append('transform not applied')
        if obj.modifiers:
            problems.append('has modifiers')
        stats = mesh_stats(obj)
        for key in ('triangles', 'vertices', 'connected_components', 'has_uv', 'geometry_sha256'):
            if stats[key] != entry[key]:
                problems.append(f'{key} {stats[key]} != manifest {entry[key]}')
        if stats['polygons'] != stats['triangles']:
            problems.append('not triangulated')
        lower, upper = np.array(stats['bbox_min']), np.array(stats['bbox_max'])
        if np.abs((lower + upper) / 2).max() > 1e-5 or abs((upper - lower).max() - LONGEST_SIDE) > 1e-5:
            problems.append(f'not normalised {lower} {upper}')
        group = obj.vertex_groups.get(ROI_GROUP)
        members, weights = 0, set()
        if group:
            for vertex in obj.data.vertices:
                for membership in vertex.groups:
                    if membership.group == group.index:
                        members += 1
                        weights.add(round(membership.weight, 6))
        expected = entry['roi']['vertices'] if entry['roi'] else 0
        if bool(group) != bool(entry['roi']) or members != expected or (group and weights != {1.0}):
            problems.append(f'ROI group {bool(group)} members {members} weights {sorted(weights)} expected {expected}')
        if len(obj.vertex_groups) != (1 if entry['roi'] else 0):
            problems.append(f'vertex groups {[g.name for g in obj.vertex_groups]}')
        pipeline = 'skipped'
        if snapshot:
            try:
                mesh, vertices, faces = snapshot(obj, bpy.context.evaluated_depsgraph_get())
                bpy.data.meshes.remove(mesh)
                pipeline = hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest() == entry['geometry_sha256']
            except Exception as error:  # signature may change in later extension versions
                print('note: smart_lod.geometry.snapshot check skipped:', error)
            if pipeline is False:
                problems.append('pipeline source_sha256 differs from manifest geometry_sha256')
        print(f"verify {asset_id:11s} objects={len(meshes)} tris={stats['triangles']:>7,d} "
              f"ROI={members:>6,d} pipeline_hash_match={pipeline} {'OK' if not problems else problems}")
        if problems:
            failures.append((asset_id, problems))
    if failures:
        raise RuntimeError(f'dataset verification failed: {failures}')
    print(f"verified {len(manifest['assets'])} assets")


# --------------------------------------------------------------------------- main

def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(prog='make_dataset.py')
    parser.add_argument('--only', help='comma-separated asset ids to rebuild')
    parser.add_argument('--verify-only', action='store_true', help='only re-open and check saved files')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--no-verify', action='store_true')
    parser.add_argument('--check-renders', type=Path, help='directory for six-view ROI renders')
    args = parser.parse_args(argv)
    specs = {spec['id']: spec for spec in ASSETS}
    only = args.only.split(',') if args.only else list(specs)
    unknown = sorted(set(only) - set(specs))
    if unknown:
        parser.error('unknown asset id(s): ' + ', '.join(unknown))
    manifest = json.loads(MANIFEST.read_text('utf-8')) if MANIFEST.exists() else None
    if not args.verify_only:
        sources = load_sources()
        built = {asset_id: build_asset(specs[asset_id], sources) for asset_id in only}
        old = manifest['assets'] if manifest and args.only else {}
        assets = {asset_id: built.get(asset_id, old.get(asset_id)) for asset_id in specs
                  if asset_id in built or asset_id in old}
        manifest = dict(
            schema_version=1, generated_by='scripts/make_dataset.py',
            generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            blender_version=bpy.app.version_string,
            conventions=dict(up='+Z', front='-Y', bbox_center=[0.0, 0.0, 0.0], longest_bbox_side=LONGEST_SIDE,
                             transforms='applied (identity object matrix)', triangulated=True,
                             triangle_cap=TRIANGLE_CAP, object_name='<asset_id>_HighPoly',
                             geometry_sha256='sha256(float64 world vertices + int64 loop-triangle indices), '
                                             'same formula as report source_sha256 in smart_lod.pipeline v0.1'),
            roi_definition=dict(
                group=ROI_GROUP, weight=1.0,
                use='evaluation only; never protection input for any simplification method',
                rule='vertex is inside ANY union term and NOT inside ANY exclude term; coordinates are normalised',
                area_fraction='integral over the surface of the linearly interpolated 0/1 vertex weight '
                              '/ total surface area (triangle area x mean corner membership)'),
            categories=CATEGORIES,
            splits=dict(tuning=[i for i, e in assets.items() if e['split'] == 'tuning'],
                        evaluation=[i for i, e in assets.items() if e['split'] == 'evaluation'],
                        rule='tuning assets choose lambda and channel weights; evaluation assets are never '
                             'used for tuning'),
            assets=assets)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        print('wrote', MANIFEST.relative_to(ROOT), f"({len(assets)} assets)")
    if manifest is None:
        raise SystemExit('dataset/manifest.json is missing; build the dataset first')
    tmp = DATASET / '.render_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        if not args.verify_only and not args.no_preview:
            make_preview(manifest, tmp)
        if args.check_renders:
            make_check_renders(manifest, args.check_renders.resolve(), tmp, set(only) if args.only else None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if args.verify_only or not args.no_verify:
        verify(manifest)


if __name__ == '__main__':
    main()
