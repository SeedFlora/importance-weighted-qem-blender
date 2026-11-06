"""Blender mesh IO. Generated assets are static evaluated snapshots."""
import bpy
import numpy as np
from mathutils import Matrix


def world_arrays(mesh, matrix):
    mesh.calc_loop_triangles()
    vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get('co', vertices)
    vertices = vertices.reshape(-1, 3)
    transform = np.asarray(matrix, dtype=float)
    vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
    faces = np.array([tuple(t.vertices) for t in mesh.loop_triangles], dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def snapshot(source, depsgraph):
    """Evaluated mesh of ``source`` baked to world space, plus its float64/int64 arrays.

    The world matrix is baked once: parenting/constraints can introduce shear,
    which assigning matrix_world to an unparented object would decompose away.
    Baking a mirroring (negative-determinant) matrix reverses the winding, so the
    faces are flipped back afterwards, as Blender's Apply Scale does; outputs
    then keep outward normals. ``vertices`` (the baked positions, float32
    precision as delivered) and ``faces`` (the baked mesh's loop triangles) are
    read after baking, so every method, the metrics and ``result_mesh``'s UV
    transfer share one triangulation of the delivered LOD0.
    """
    evaluated = source.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
    mesh.name = source.name + '_LOD_snapshot'
    try:
        for index, slot in enumerate(evaluated.material_slots):
            if index < len(mesh.materials):
                mesh.materials[index] = slot.material.original if slot.material else None
        determinant = source.matrix_world.determinant()
        if abs(determinant) < 1e-15:
            raise ValueError('Object has zero scale; apply a nonzero scale first')
        mesh.transform(source.matrix_world)
        if determinant < 0:
            mesh.flip_normals()
        mesh.update()
        vertices, faces = world_arrays(mesh, Matrix.Identity(4))
        if len(faces) < 4 or not np.isfinite(vertices).all():
            raise ValueError('Mesh must have at least four finite triangles')
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise
    return mesh, vertices, faces


def group_weights(mesh, source, group_name):
    group = source.vertex_groups.get(group_name)
    weights = np.zeros(len(mesh.vertices), dtype=float)
    if group:
        for vertex in mesh.vertices:
            for membership in vertex.groups:
                if membership.group == group.index:
                    weights[vertex.index] = membership.weight
                    break
    return weights


def feature_constraints(mesh, preserve_seams=True):
    """Lock UV discontinuities and material boundaries, including unmarked seams."""
    locked = np.zeros(len(mesh.vertices), dtype=bool)
    sharp = np.zeros(len(mesh.vertices), dtype=float)
    for edge in mesh.edges:
        if edge.use_edge_sharp:
            sharp[list(edge.vertices)] = 1.0
        if preserve_seams and edge.use_seam:
            locked[list(edge.vertices)] = True
    if not preserve_seams:
        return locked, sharp
    edge_records = {}
    for poly in mesh.polygons:
        loops = list(poly.loop_indices)
        for i, loop in enumerate(loops):
            next_loop = loops[(i + 1) % len(loops)]
            a, b = mesh.loops[loop].vertex_index, mesh.loops[next_loop].vertex_index
            if a > b:
                a, b, loop, next_loop = b, a, next_loop, loop
            signature = (poly.material_index, tuple(
                tuple(round(float(x), 6) for x in layer.data[loop].uv) +
                tuple(round(float(x), 6) for x in layer.data[next_loop].uv)
                for layer in mesh.uv_layers))
            key = (a, b)
            if key in edge_records and edge_records[key] != signature:
                locked[[a, b]] = True
            else:
                edge_records[key] = signature
    return locked, sharp


def _barycentric(points, triangles):
    """``(weights, collinear)``: weights of each point in its triangle's plane (unclamped).

    Collinear triangles (sin^2 of the angle at a below 1e-10), which Blender's
    tessellation of n-gons with collinear corners produces, instead interpolate
    piecewise-linearly between neighbouring corners along the line (clamped to
    its ends), so each corner still maps to itself.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    u, v, w = b - a, c - a, points - a
    uu, uv, vv = np.sum(u*u, axis=1), np.sum(u*v, axis=1), np.sum(v*v, axis=1)
    wu, wv = np.sum(w*u, axis=1), np.sum(w*v, axis=1)
    denominator = uu*vv - uv*uv
    degenerate = ~(denominator > 1e-10 * uu * vv)
    safe = np.where(degenerate, 1, denominator)
    y, z = (vv*wu - uv*wv)/safe, (uu*wv - uv*wu)/safe
    result = np.stack((1-y-z, y, z), axis=1)
    if degenerate.any():
        corners, point = triangles[degenerate], points[degenerate]
        rows = np.arange(len(corners))
        edges = corners[:, [1, 2, 0]] - corners
        direction = edges[rows, np.argmax(np.sum(edges*edges, axis=2), axis=1)]
        order = np.argsort(np.einsum('rij,rj->ri', corners, direction), axis=1)
        s = np.einsum('rij,rj->ri', corners[rows[:, None], order], direction)
        t = np.clip(np.einsum('rj,rj->r', point, direction), s[:, 0], s[:, 2])
        low = (t > s[:, 1]).astype(np.int64)
        span = s[rows, low + 1] - s[rows, low]
        fraction = np.where(span > 0, (t - s[rows, low]) / np.where(span > 0, span, 1), 0.0)
        weights = np.zeros((len(corners), 3))
        weights[rows, order[rows, low]] = 1 - fraction
        weights[rows, order[rows, low + 1]] += fraction
        result[degenerate] = weights
    return result, degenerate


def _corner_sources(result, source_faces, original_vertices, original_faces, tri_loops, tri_polygons):
    """Per output corner: the three source loops it interpolates and their weights.

    These are the loops of the face's source triangle, except for corners off
    the line of a collinear source triangle, which carries no UV information
    there: they use the non-collinear triangle of the same source polygon
    that contains them best (if the polygon has one).
    """
    old_positions = original_vertices[original_faces[source_faces]]
    parts = [_barycentric(result.vertices[result.faces[:, k]], old_positions) for k in range(3)]
    weights = np.stack([part[0] for part in parts], axis=1)
    loops = np.repeat(tri_loops[source_faces][:, None], 3, axis=1)
    flat = np.flatnonzero(parts[0][1])
    if not len(flat):
        return loops, weights
    points = result.vertices[result.faces[flat]]
    size = np.linalg.norm(old_positions[flat] - old_positions[flat][:, [1, 2, 0]], axis=2).max(axis=1)
    rebuilt = np.einsum('fkj,fjd->fkd', weights[flat], old_positions[flat])
    off_line = np.linalg.norm(rebuilt - points, axis=2) > 1e-6 * size[:, None]
    polygon = tri_polygons[source_faces[flat]]
    by_polygon = np.argsort(tri_polygons, kind='stable')
    first = np.searchsorted(tri_polygons[by_polygon], polygon)
    last = np.searchsorted(tri_polygons[by_polygon], polygon, side='right')
    for i, k in zip(*np.nonzero(off_line & (last - first >= 2)[:, None])):
        candidates = by_polygon[first[i]:last[i]]
        candidate_weights, collinear = _barycentric(np.broadcast_to(points[i, k], (len(candidates), 3)),
                                                    original_vertices[original_faces[candidates]])
        score = np.where(collinear, -np.inf, candidate_weights.min(axis=1))
        best = int(np.argmax(score))
        if score[best] > -np.inf:
            weights[flat[i], k], loops[flat[i], k] = candidate_weights[best], tri_loops[candidates[best]]
    return loops, weights


def _get(collection, name, count, dtype):
    values = np.empty(count, dtype=dtype)
    collection.foreach_get(name, values)
    return values


def result_mesh(name, result, original_mesh, original_vertices, original_faces, matrix):
    """Blender mesh for a simplification of ``snapshot`` arrays.

    Each output face takes material, smooth flag and interpolated UVs from its
    source triangle ``result.source_faces``, an index into ``original_faces``,
    which must be ``original_mesh``'s loop triangles as returned by ``snapshot``
    (checked: a different triangulation would pair corners with wrong UVs).
    UVs are interpolated per face, preserving discontinuities between islands;
    see ``_corner_sources`` for collinear source triangles.
    """
    original_mesh.calc_loop_triangles()
    triangles, polygons = original_mesh.loop_triangles, original_mesh.polygons
    tri_vertices = _get(triangles, 'vertices', len(triangles) * 3, np.int32).reshape(-1, 3)
    if tri_vertices.shape != original_faces.shape or not np.array_equal(tri_vertices, original_faces):
        raise ValueError("original_faces are not original_mesh's loop triangles; use snapshot() for both")
    source_faces = np.asarray(result.source_faces, dtype=np.int64)
    tri_loops = _get(triangles, 'loops', len(triangles) * 3, np.int32).reshape(-1, 3)
    tri_polygons = _get(triangles, 'polygon_index', len(triangles), np.int32)
    loops, weights = _corner_sources(result, source_faces, original_vertices, original_faces, tri_loops, tri_polygons)
    inverse = np.asarray(matrix.inverted(), dtype=float)
    local_vertices = result.vertices @ inverse[:3, :3].T + inverse[:3, 3]
    mesh = bpy.data.meshes.new(name)
    try:
        mesh.from_pydata(local_vertices.tolist(), [], result.faces.tolist())
        mesh.update()
        for material in original_mesh.materials:
            mesh.materials.append(material)
        for key, dtype in (('material_index', np.int32), ('use_smooth', bool)):
            mesh.polygons.foreach_set(key, _get(polygons, key, len(polygons), dtype)[tri_polygons[source_faces]])
        for old_layer in original_mesh.uv_layers:
            uv = _get(old_layer.data, 'uv', len(old_layer.data) * 2, np.float32).reshape(-1, 2).astype(float)
            interpolated = np.einsum('fkj,fkjd->fkd', weights, uv[loops]).reshape(-1)
            mesh.uv_layers.new(name=old_layer.name).data.foreach_set('uv', interpolated.astype(np.float32))
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise
    return mesh


def add_importance(obj, values):
    group = obj.vertex_groups.get('LOD_Importance') or obj.vertex_groups.new(name='LOD_Importance')
    # Batch weights at 1/255 precision for faster Blender RNA access.
    bins = np.rint(np.clip(values, 0, 1) * 255).astype(int)
    for value in np.unique(bins):
        indices = np.flatnonzero(bins == value).tolist()
        group.add(indices, float(value)/255, 'REPLACE')
    color = obj.data.color_attributes.get('LOD_Heatmap') or obj.data.color_attributes.new(
        name='LOD_Heatmap', type='FLOAT_COLOR', domain='POINT')
    colors = np.ones((len(values), 4), dtype=np.float32)
    x = np.clip(values, 0, 1)
    colors[:, 0] = np.clip(2*x, 0, 1)
    colors[:, 1] = np.clip(2-2*np.abs(x-0.5)*2, 0, 1)
    colors[:, 2] = np.clip(1-2*x, 0, 1)
    color.data.foreach_set('color', colors.reshape(-1))
