"""Deterministic, dependency-light triangle mesh simplification.

This module deliberately has no Blender dependency.  It implements Garland and
Heckbert style accumulated plane quadrics and an optional *additional* positional
quadric for each important input vertex.  In normalized object coordinates the
extra cost of a cluster at ``p`` is::

    strength * sum(importance[i] * ||p - original_position[i]|| ** 2)

Unlike multiplying a plane error, this term also protects planar regions.  Zero
strength or all-zero importance is exactly the unweighted algorithm, including
tie breaking.  Output importance is the maximum importance in each cluster.

Safety is intentionally conservative: initial nonmanifold vertices are frozen;
the manifold edge-link condition, duplicate-face checks, and normal/area checks
guard every collapse.  Boundary/interior edges cannot collapse, and open boundary
edges are constrained to their segment, with additional boundary plane quadrics.
Locked vertices retain their exact positions.  Consequently a requested triangle
budget is not always reachable.  A collapse removes one or two triangles and can
finish one triangle below the requested budget.  Inspect the returned stats.

Memory is linear in input topology.  A lazy heap, local adjacency updates, and
periodic stale-entry compaction avoid a full-mesh scan after every collapse.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import numbers
import time
from typing import Any

import numpy as np


@dataclass
class SimplificationResult:
    vertices: np.ndarray
    faces: np.ndarray
    source_faces: np.ndarray
    importance: np.ndarray
    stats: dict[str, Any]


# Packed symmetric 4 x 4 matrices: xx xy xz xw yy yz yw zz zw ww.
_QUADRIC_ROWS = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
_QUADRIC_COLS = np.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3])
_AREA_EPS_SQUARED = 1.0e-24
_NORMAL_DOT_MIN_SQUARED = 0.01


def _evaluate(q: np.ndarray, x: float, y: float, z: float) -> float:
    value = (
        q[0] * x * x + 2.0 * q[1] * x * y + 2.0 * q[2] * x * z
        + 2.0 * q[3] * x + q[4] * y * y + 2.0 * q[5] * y * z
        + 2.0 * q[6] * y + q[7] * z * z + 2.0 * q[8] * z + q[9]
    )
    # A sum of squared plane distances is nonnegative; tolerate roundoff.
    return max(0.0, float(value))


def _validate_inputs(vertices, faces, target_count, importance, locked):
    if isinstance(target_count, (bool, np.bool_)) or not isinstance(target_count, numbers.Integral):
        raise ValueError("target_count must be a nonnegative integer")
    if target_count < 0:
        raise ValueError("target_count must be a nonnegative integer")
    try:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces)
    except (TypeError, ValueError) as exc:
        raise ValueError("vertices and faces must be numeric arrays") from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3); triangulate the mesh first")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must contain only finite coordinates")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must contain integer vertex indices")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError("face vertex indices are outside the vertex array")
    faces = faces.astype(np.int64, copy=True)
    n = len(vertices)
    if importance is None:
        importance = np.zeros(n, dtype=np.float64)
    else:
        importance = np.asarray(importance, dtype=np.float64)
        if importance.shape != (n,) or not np.all(np.isfinite(importance)):
            raise ValueError("importance must be a finite array of shape (N,)")
        if np.any(importance < 0.0) or np.any(importance > 1.0):
            raise ValueError("importance values must be in [0, 1]")
        importance = importance.copy()
    if locked is None:
        locked = np.zeros(n, dtype=bool)
    else:
        locked = np.asarray(locked)
        if locked.shape != (n,):
            raise ValueError("locked must have shape (N,)")
        if not np.issubdtype(locked.dtype, np.bool_):
            if not np.issubdtype(locked.dtype, np.number) or not np.all(np.isfinite(locked)):
                raise ValueError("locked must contain boolean or finite numeric values")
        locked = locked.astype(bool, copy=True)
    return vertices.copy(), faces, int(target_count), importance, locked


def simplify(
    vertices,
    faces,
    target_count: int,
    importance=None,
    strength: float = 8.0,
    locked=None,
) -> SimplificationResult:
    """Simplify a triangle mesh without modifying the input arrays.

    ``source_faces[j]`` identifies the original input triangle associated with
    output triangle ``j``.  It can be used to retain face-domain attributes such
    as material indices; it is not a mapping of output corners to original UVs.
    Exact duplicate and geometrically degenerate input triangles are removed.
    Unreferenced vertices are omitted except explicitly locked vertices.

    The result is safe to use even when ``stats['target_reached']`` is false.
    ``stats['stop_reason']`` explains budget completion versus exhaustion of
    allowed collapses.  Distances and positional weights use coordinates centered
    at the bounding-box midpoint and divided by its longest side.
    """
    started = time.perf_counter()
    original, triangles, target, weights, locks = _validate_inputs(
        vertices, faces, target_count, importance, locked
    )
    try:
        strength = float(strength)
    except (TypeError, ValueError) as exc:
        raise ValueError("strength must be finite and nonnegative") from exc
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("strength must be finite and nonnegative")
    n, original_count = len(original), len(triangles)
    if n:
        low, high = original.min(axis=0), original.max(axis=0)
        # Half before adding avoids overflowing the midpoint of large values.
        origin = low * 0.5 + high * 0.5
        extent = high - low
        scale = float(np.max(extent))
        if not math.isfinite(scale):
            raise ValueError("coordinate range is too large; rescale the mesh")
        if scale == 0.0:
            scale = 1.0
        positions = (original - origin) / scale
    else:
        origin = np.zeros(3)
        scale = 1.0
        positions = original.copy()

    source = np.arange(original_count, dtype=np.int64)
    if original_count:
        cross = np.cross(
            positions[triangles[:, 1]] - positions[triangles[:, 0]],
            positions[triangles[:, 2]] - positions[triangles[:, 0]],
        )
        valid = np.einsum("ij,ij->i", cross, cross) > _AREA_EPS_SQUARED
        triangles, source, cross = triangles[valid], source[valid], cross[valid]
        if len(triangles):
            _, keep = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
            keep.sort()
            triangles, source, cross = triangles[keep], source[keep], cross[keep]
    else:
        cross = np.empty((0, 3), dtype=np.float64)
    m = len(triangles)
    alive_faces = np.ones(m, dtype=bool)
    alive_vertices = np.ones(n, dtype=bool)
    vertex_faces: list[set[int]] = [set() for _ in range(n)]
    neighbors: list[set[int]] = [set() for _ in range(n)]
    edge_counts: dict[tuple[int, int], int] = {}
    boundary_degree = np.zeros(n, dtype=np.int32)
    quadrics = np.zeros((n, 10), dtype=np.float64)
    revisions = np.zeros(n, dtype=np.int64)
    exact_vertex_ids = np.arange(n, dtype=np.int64)

    if m:
        normals = cross / np.linalg.norm(cross, axis=1)[:, None]
        planes = np.column_stack((normals, -np.einsum("ij,ij->i", normals, positions[triangles[:, 0]])))
        face_quadrics = planes[:, _QUADRIC_ROWS] * planes[:, _QUADRIC_COLS]
        for corner in range(3):
            np.add.at(quadrics, triangles[:, corner], face_quadrics)
        for face_id, (a, b, c) in enumerate(triangles.tolist()):
            vertex_faces[a].add(face_id)
            vertex_faces[b].add(face_id)
            vertex_faces[c].add(face_id)
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                edge_counts[key] = edge_counts.get(key, 0) + 1
                neighbors[u].add(v)
                neighbors[v].add(u)
        for (a, b), count in edge_counts.items():
            if count == 1:
                boundary_degree[a] += 1
                boundary_degree[b] += 1
                face_id = next(iter(vertex_faces[a] & vertex_faces[b]))
                edge = positions[b] - positions[a]
                boundary_normal = np.cross(edge, normals[face_id])
                normal_length = float(np.linalg.norm(boundary_normal))
                if normal_length > 0.0:
                    boundary_normal /= normal_length
                    plane = np.append(boundary_normal, -np.dot(boundary_normal, positions[a]))
                    q_boundary = 10.0 * plane[_QUADRIC_ROWS] * plane[_QUADRIC_COLS]
                    quadrics[a] += q_boundary
                    quadrics[b] += q_boundary

    # Point quadrics accumulate the displacement of every original important
    # vertex, rather than forgetting past displacement after an edge collapse.
    if strength > 0.0 and np.any(weights):
        point_weight = strength * weights
        quadrics[:, 0] += point_weight
        quadrics[:, 4] += point_weight
        quadrics[:, 7] += point_weight
        quadrics[:, 3] -= point_weight * positions[:, 0]
        quadrics[:, 6] -= point_weight * positions[:, 1]
        quadrics[:, 8] -= point_weight * positions[:, 2]
        quadrics[:, 9] += point_weight * np.einsum("ij,ij->i", positions, positions)

    # Protect nonmanifold edges and bow-tie vertices.  A manifold vertex link is
    # one cycle (interior) or one path (boundary), not several disjoint fans.
    frozen = np.zeros(n, dtype=bool)
    for (a, b), count in edge_counts.items():
        if count > 2:
            frozen[a] = frozen[b] = True
    for v in range(n):
        incident = vertex_faces[v]
        if not incident or frozen[v]:
            continue
        if boundary_degree[v] not in (0, 2):
            frozen[v] = True
            continue
        if len(incident) != len(neighbors[v]) - (1 if boundary_degree[v] else 0):
            frozen[v] = True
            continue
        link: dict[int, list[int]] = {}
        for face_id in incident:
            others = [int(w) for w in triangles[face_id] if w != v]
            u, w = others
            link.setdefault(u, []).append(w)
            link.setdefault(w, []).append(u)
        if any(len(adjacent) > 2 for adjacent in link.values()):
            frozen[v] = True
            continue
        unseen = set(link)
        stack = [next(iter(unseen))]
        while stack:
            w = stack.pop()
            if w in unseen:
                unseen.remove(w)
                stack.extend(link[w])
        if unseen:
            frozen[v] = True

    # Entry fields: error, edge length squared (zero-error tie break), a, b,
    # revisions, and the chosen position.  Pending entries deduplicate work.
    heap: list[tuple] = []
    pending: dict[tuple[int, int], tuple[int, int]] = {}
    rejects = {"topology": 0, "normal_or_area": 0, "duplicate_faces": 0}
    collapses = 0
    remaining = m
    max_heap = 0

    def candidate(a: int, b: int):
        if frozen[a] or frozen[b] or (locks[a] and locks[b]):
            return None
        boundary_a, boundary_b = boundary_degree[a] > 0, boundary_degree[b] > 0
        if boundary_a != boundary_b:
            return None
        if boundary_a and edge_counts[(a, b)] != 1:
            return None
        q = quadrics[a] + quadrics[b]
        pa, pb = positions[a], positions[b]
        if locks[a] or locks[b]:
            point = pa if locks[a] else pb
            x, y, z = map(float, point)
            return _evaluate(q, x, y, z), x, y, z

        ax, ay, az = map(float, pa)
        bx, by, bz = map(float, pb)
        dx, dy, dz = bx - ax, by - ay, bz - az
        x, y, z = (ax + bx) * 0.5, (ay + by) * 0.5, (az + bz) * 0.5
        best = (_evaluate(q, x, y, z), x, y, z)
        for px, py, pz in ((ax, ay, az), (bx, by, bz)):
            cost = _evaluate(q, px, py, pz)
            if cost < best[0]:
                best = cost, px, py, pz

        # Minimum restricted to this edge; valid for boundaries and useful when
        # the full 3 x 3 system is singular on flat surfaces.
        adx = q[0] * dx + q[1] * dy + q[2] * dz
        ady = q[1] * dx + q[4] * dy + q[5] * dz
        adz = q[2] * dx + q[5] * dy + q[7] * dz
        denominator = dx * adx + dy * ady + dz * adz
        if denominator > 1.0e-30:
            t = -(ax * adx + ay * ady + az * adz + q[3] * dx + q[6] * dy + q[8] * dz) / denominator
            t = min(1.0, max(0.0, float(t)))
            px, py, pz = ax + t * dx, ay + t * dy, az + t * dz
            cost = _evaluate(q, px, py, pz)
            if cost < best[0]:
                best = cost, px, py, pz

        if not boundary_a:
            # Explicit symmetric solve avoids creating thousands of tiny NumPy
            # solver work arrays and handles singular quadrics deterministically.
            c00 = q[4] * q[7] - q[5] * q[5]
            c01 = q[2] * q[5] - q[1] * q[7]
            c02 = q[1] * q[5] - q[2] * q[4]
            c11 = q[0] * q[7] - q[2] * q[2]
            c12 = q[1] * q[2] - q[0] * q[5]
            c22 = q[0] * q[4] - q[1] * q[1]
            det = q[0] * c00 + q[1] * c01 + q[2] * c02
            trace = float(q[0] + q[4] + q[7])
            if det > 1.0e-12 * max(trace ** 3, 1.0e-30):
                px = -(c00 * q[3] + c01 * q[6] + c02 * q[8]) / det
                py = -(c01 * q[3] + c11 * q[6] + c12 * q[8]) / det
                pz = -(c02 * q[3] + c12 * q[6] + c22 * q[8]) / det
                if math.isfinite(px) and math.isfinite(py) and math.isfinite(pz):
                    cost = _evaluate(q, px, py, pz)
                    if cost < best[0]:
                        best = cost, float(px), float(py), float(pz)
        return best

    def schedule(a: int, b: int):
        if a > b:
            a, b = b, a
        if not alive_vertices[a] or not alive_vertices[b]:
            return
        key = (a, b)
        revision = (int(revisions[a]), int(revisions[b]))
        if pending.get(key) == revision:
            return
        choice = candidate(a, b)
        if choice is None:
            pending.pop(key, None)
            return
        cost, x, y, z = choice
        delta = positions[a] - positions[b]
        length_squared = float(np.dot(delta, delta))
        heapq.heappush(heap, (cost, length_squared, a, b, revision[0], revision[1], x, y, z))
        pending[key] = revision

    def validate_collapse(a: int, b: int, point: np.ndarray):
        shared = vertex_faces[a] & vertex_faces[b]
        if len(shared) not in (1, 2):
            rejects["topology"] += 1
            return None
        opposite = {int(v) for face_id in shared for v in triangles[face_id] if v != a and v != b}
        if neighbors[a] & neighbors[b] != opposite:
            rejects["topology"] += 1
            return None
        touched = vertex_faces[a] | vertex_faces[b]
        if not touched - shared:
            # Retain a triangle in each surface component instead of silently
            # deleting a tiny disconnected part when its final edge contracts.
            rejects["topology"] += 1
            return None
        surviving = []
        face_keys = set()
        for face_id in sorted(touched - shared):
            tri = [a if int(v) == b else int(v) for v in triangles[face_id]]
            key = tuple(sorted(tri))
            if key in face_keys:
                rejects["duplicate_faces"] += 1
                return None
            face_keys.add(key)
            p0, p1, p2 = (point if v == a else positions[v] for v in tri)
            # Scalar cross/dot operations avoid per-face NumPy cross overhead.
            ux, uy, uz = p1 - p0
            vx, vy, vz = p2 - p0
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            area_squared = nx * nx + ny * ny + nz * nz
            old_normal = cross[face_id]
            old_area_squared = float(np.dot(old_normal, old_normal))
            dot = nx * old_normal[0] + ny * old_normal[1] + nz * old_normal[2]
            if (
                area_squared <= _AREA_EPS_SQUARED or dot <= 0.0
                or dot * dot <= _NORMAL_DOT_MIN_SQUARED * area_squared * old_area_squared
            ):
                rejects["normal_or_area"] += 1
                return None
            surviving.append((face_id, tri, (nx, ny, nz)))
        return touched, shared, surviving

    def remove_face(face_id: int):
        a, b, c = map(int, triangles[face_id])
        vertex_faces[a].discard(face_id)
        vertex_faces[b].discard(face_id)
        vertex_faces[c].discard(face_id)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            count = edge_counts[key]
            if count == 1:
                del edge_counts[key]
                neighbors[u].discard(v)
                neighbors[v].discard(u)
                boundary_degree[u] -= 1
                boundary_degree[v] -= 1
            else:
                edge_counts[key] = count - 1
                if count == 2:
                    boundary_degree[u] += 1
                    boundary_degree[v] += 1

    def add_face(face_id: int, tri):
        a, b, c = tri
        vertex_faces[a].add(face_id)
        vertex_faces[b].add(face_id)
        vertex_faces[c].add(face_id)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            count = edge_counts.get(key, 0)
            edge_counts[key] = count + 1
            if count == 0:
                neighbors[u].add(v)
                neighbors[v].add(u)
                boundary_degree[u] += 1
                boundary_degree[v] += 1
            elif count == 1:
                boundary_degree[u] -= 1
                boundary_degree[v] -= 1

    if remaining > target:
        for a, b in edge_counts:
            schedule(a, b)
    max_heap = len(heap)
    while remaining > target and heap:
        entry = heapq.heappop(heap)
        _, _, a, b, revision_a, revision_b, x, y, z = entry
        key = (a, b)
        if pending.get(key) != (revision_a, revision_b):
            continue
        del pending[key]
        if (
            not alive_vertices[a] or not alive_vertices[b] or key not in edge_counts
            or revisions[a] != revision_a or revisions[b] != revision_b
        ):
            continue
        point = np.array((x, y, z))
        validated = validate_collapse(a, b, point)
        if validated is None:
            continue
        touched, shared, surviving = validated
        affected = neighbors[a] | neighbors[b] | {a, b}
        old_boundaries = {v: bool(boundary_degree[v]) for v in affected}
        old_edges = {(min(u, v), max(u, v)) for u in affected for v in neighbors[u]}
        for face_id in touched:
            remove_face(face_id)
        if np.array_equal(point, positions[a]):
            pass
        elif np.array_equal(point, positions[b]):
            exact_vertex_ids[a] = exact_vertex_ids[b]
        else:
            exact_vertex_ids[a] = -1
        positions[a] = point
        quadrics[a] += quadrics[b]
        weights[a] = max(weights[a], weights[b])
        locks[a] = locks[a] or locks[b]
        alive_vertices[b] = False
        locks[b] = False
        for face_id in shared:
            alive_faces[face_id] = False
        for face_id, tri, normal in surviving:
            triangles[face_id] = tri
            cross[face_id] = normal
            add_face(face_id, tri)
        remaining -= len(shared)
        collapses += 1
        revisions[a] += 1
        revisions[b] += 1
        for v in affected - {a, b}:
            if old_boundaries[v] != bool(boundary_degree[v]):
                revisions[v] += 1
        for edge in old_edges:
            if edge not in edge_counts:
                pending.pop(edge, None)
        # Existing unaffected pending edges are cheap no-ops; rejected local
        # edges are reconsidered because their topology may now be legal.
        local_edges = {(min(u, v), max(u, v)) for u in affected if alive_vertices[u] for v in neighbors[u]}
        for u, v in sorted(local_edges):
            schedule(u, v)
        max_heap = max(max_heap, len(heap))
        if len(heap) > max(10000, 4 * len(pending)):
            heap[:] = [item for item in heap if pending.get((item[2], item[3])) == (item[4], item[5])]
            heapq.heapify(heap)

    output_faces = triangles[alive_faces]
    output_sources = source[alive_faces]
    used = locks.copy()
    if len(output_faces):
        used[output_faces.ravel()] = True
    used &= alive_vertices
    remap = np.full(n, -1, dtype=np.int64)
    remap[used] = np.arange(np.count_nonzero(used), dtype=np.int64)
    output_vertices = positions[used] * scale + origin
    # Preserve input coordinates bit-for-bit when a cluster stayed at an input
    # vertex.  This includes every locked vertex and an unchanged LOD0 mesh.
    exact_ids = exact_vertex_ids[used]
    exact_mask = exact_ids >= 0
    output_vertices[exact_mask] = original[exact_ids[exact_mask]]
    original_locks = np.zeros(n, dtype=bool) if locked is None else np.asarray(locked, dtype=bool)
    stats = {
        "original_faces": original_count,
        "cleaned_faces": m,
        "removed_invalid_faces": original_count - m,
        "output_faces": remaining,
        "target_faces": target,
        "collapses": collapses,
        "target_reached": remaining <= target,
        "stop_reason": "target_reached" if remaining <= target else "constraints",
        "nonmanifold_vertices": int(np.count_nonzero(frozen)),
        "locked_vertices": int(np.count_nonzero(original_locks)),
        "rejected_collapses": rejects,
        "peak_heap_entries": max_heap,
        "normalization_scale": scale,
        "seconds": time.perf_counter() - started,
    }
    return SimplificationResult(
        vertices=output_vertices,
        faces=remap[output_faces],
        source_faces=output_sources,
        importance=weights[used],
        stats=stats,
    )
