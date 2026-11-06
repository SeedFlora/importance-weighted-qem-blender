"""Deterministic, dependency-light triangle mesh simplification.

This module deliberately has no Blender dependency (NumPy only).  It implements
Garland and Heckbert style accumulated plane quadrics with greedy edge
collapse.  Coordinates are first normalized: centered at the bounding-box
midpoint and divided by its longest side.  A per-vertex importance ``I`` in
[0, 1] can steer the result in one of two ways:

``weighting='quadric'`` (default)
    After the face-plane and boundary quadrics are accumulated, each vertex
    quadric is multiplied by ``w_i = 1 + strength * I_i``.  A merged cluster
    sums the quadrics of its members, so the weights carry over, and the new
    position minimizes the weighted quadric.  This only rescales plane errors:
    it cannot protect an exactly planar region, whose plane error is zero
    whatever the weight.  It changes the collapse order and placement only
    where the surface is curved or creased.

``weighting='anchor'``
    The v0.1 weighting, kept for ablation: an additional positional quadric
    for each important input vertex, with extra cost
    ``strength * sum(I_i * ||p - original_position_i|| ** 2)``.  The result is
    bit-for-bit v0.1's whenever ``stats['rejected_collapses']['source_flip']``
    is 0.  Once the source-flip guard (below) rejects a collapse, the result
    can differ; at strength 8 without locks it differed on 10 of the 12
    dataset meshes at 5 % of the triangles.  Anchoring also protects planar
    regions, but it pulls a collapsed vertex toward the centroid of its
    cluster, and that centroid lies inside a convex surface.
    On the demo car at LOD3 this made the result worse than unweighted QEM
    (volume ratio 0.873 vs 0.899, silhouette IoU 0.936 vs 0.956; measured
    before the source-flip guard existed).

Zero strength or all-zero importance is exactly the unweighted algorithm in
both modes, including tie breaking.  Output importance is the maximum
importance in each cluster.  ``strength`` must lie in [0, ``MAX_STRENGTH``].

Safety is intentionally conservative: initial nonmanifold vertices are frozen;
the manifold edge-link condition, duplicate-face checks, and normal/area checks
guard every collapse.  The normal check compares each changed face with its
current normal (at most ~84 degrees of turn per collapse) and also with the
normal of its original input triangle (``source_faces``): a collapse that
would leave any face at dot <= 0 with its source triangle is rejected and
counted in ``stats['rejected_collapses']['source_flip']``.  Without the second
check, which v0.1 lacked, small turns add up to faces folded against their
source, most often around locked seam vertices (22 on spot at 5 % with its
seams locked), but also without locks: up to 10 per LOD on the 12 dataset
meshes, where the check then fires up to 30 times at 50 % of the triangles
and up to 114 times at 5 %.  Hence, up to roundoff in the final rescaling, no
output face points against its source triangle.  Boundary/interior edges
cannot collapse, and open boundary edges are constrained to their segment,
with additional boundary plane quadrics.
Locked vertices retain their exact positions.  Consequently a requested triangle
budget is not always reachable.  A collapse removes one or two triangles and can
finish one triangle below the requested budget.  Inspect the returned stats.

The greedy collapse sequence does not depend on the budget; the budget only
decides when to stop.  ``simplify_levels`` therefore runs one pass down to the
smallest budget and takes a snapshot when each budget is first met.  Every
snapshot is array-identical to an independent ``simplify`` call with that
budget, so a set of LODs costs about as much as the smallest one alone.

Complexity: memory is linear in the input size.  Setup costs O(m log m) for m
triangles (duplicate removal sorts them; the heap is built in linear time).  A
collapse checks and rewires only the one-ring of the merged pair and requeues
the edges around it, O(k^2 + r log H) for valence k, r requeued edges and H heap
entries; stale entries are skipped lazily and compacted periodically, so no
step scans the whole mesh.  The loop is pure Python on float lists and keeps
the v0.1 floating-point operation order, so apart from the source-flip guard
results are unchanged while it runs about three times faster: reducing the
69,451-triangle Stanford bunny to 5 % took 6.6-8.2 s versus 21-25 s for v0.1
(CPython 3.13, development machine shared with other jobs), and 7.5 s versus
25.5 s in Blender 5.1.  These timings predate the guard, which added no
measurable time (11.0-11.5 s with it vs 11.8-13.4 s without, same busy machine).
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import numbers
from operator import add
import time
from typing import Any, Callable, Optional

import numpy as np


WEIGHTING_MODES = ("quadric", "anchor")
# Largest accepted ``strength`` (the UI stops at 100).  On a test sphere the
# protected cap kept 51-52 of 61 vertices at strengths 1e6, 1e30, 1e90 and
# 1e100, and from ~1e102 the weighted quadric arithmetic overflows
# (OverflowError, or NaN costs).
MAX_STRENGTH = 1.0e6


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
_PROGRESS_INTERVAL = 256
_INF = math.inf


def _is_count(value) -> bool:
    return (
        not isinstance(value, (bool, np.bool_)) and isinstance(value, numbers.Integral)
        and value >= 0
    )


def _validate_targets(target_counts) -> list[int]:
    message = "target_counts must be a nonempty sequence of nonnegative integers"
    if isinstance(target_counts, (str, bytes)):
        raise ValueError(message)
    try:
        targets = list(target_counts)
    except TypeError as exc:  # scalars, including 0-d arrays
        raise ValueError(message) from exc
    if not targets or not all(_is_count(t) for t in targets):
        raise ValueError(message)
    return [int(t) for t in targets]


def _validate_inputs(vertices, faces, importance, locked):
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
    return vertices.copy(), faces, importance, locked


def simplify(
    vertices,
    faces,
    target_count: int,
    importance=None,
    strength: float = 8.0,
    locked=None,
    *,
    weighting: str = "quadric",
    progress: Optional[Callable[[float], Any]] = None,
) -> SimplificationResult:
    """Simplify a triangle mesh to ``target_count`` triangles.

    Equivalent to ``simplify_levels(..., [target_count], ...)[0]``.  The input
    arrays are never modified.  ``source_faces[j]`` identifies the original
    input triangle associated with output triangle ``j``.  It can be used to
    retain face-domain attributes such as material indices; it is not a mapping
    of output corners to original UVs.  Exact duplicate and geometrically
    degenerate input triangles are removed.  Unreferenced vertices are omitted
    except explicitly locked vertices.

    The result is safe to use even when ``stats['target_reached']`` is false.
    ``stats['stop_reason']`` explains budget completion versus exhaustion of
    allowed collapses.  See the module docstring for ``weighting``.
    """
    if not _is_count(target_count):
        raise ValueError("target_count must be a nonnegative integer")
    return simplify_levels(
        vertices, faces, [target_count], importance, strength, locked,
        weighting=weighting, progress=progress,
    )[0]


def simplify_levels(
    vertices,
    faces,
    target_counts,
    importance=None,
    strength: float = 8.0,
    locked=None,
    *,
    weighting: str = "quadric",
    progress: Optional[Callable[[float], Any]] = None,
) -> list[SimplificationResult]:
    """Simplify once to several triangle budgets; results follow ``target_counts``.

    One greedy pass runs to the smallest budget.  A snapshot is taken as soon
    as ``remaining <= target`` first holds, or at the end when constraints stop
    the pass earlier.  Each result is array-identical to an independent
    ``simplify`` call with that budget, and so are its stats except for these:
    ``seconds`` is the time from this call's start until that snapshot was
    complete, ``single_pass`` is true, and ``pass_targets`` lists the distinct
    budgets of the pass in ascending order.  Duplicate budgets get independent
    copies of the same snapshot.

    ``progress``, if given, is called with the fraction of the pass completed,
    ``removed triangles / (cleaned triangles - smallest budget)`` clipped to
    [0, 1], at most every 256 collapses and once at the end.  It never affects
    the result; any exception it raises propagates and aborts the pass.
    """
    started = time.perf_counter()
    targets = _validate_targets(target_counts)
    original, triangles, weights, locks = _validate_inputs(vertices, faces, importance, locked)
    message = f"strength must be a finite number in [0, {MAX_STRENGTH:g}]"
    try:
        strength = float(strength)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not 0.0 <= strength <= MAX_STRENGTH:  # also rejects NaN
        raise ValueError(message)
    if not isinstance(weighting, str) or weighting not in WEIGHTING_MODES:
        raise ValueError("weighting must be one of: " + ", ".join(WEIGHTING_MODES))
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")

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
    tris: list[list[int]] = triangles.tolist()
    vertex_faces: list[set[int]] = [set() for _ in range(n)]
    neighbors: list[set[int]] = [set() for _ in range(n)]
    edge_counts: dict[tuple[int, int], int] = {}
    boundary_degree = [0] * n
    quadrics = np.zeros((n, 10), dtype=np.float64)

    if m:
        normals = cross / np.linalg.norm(cross, axis=1)[:, None]
        planes = np.column_stack((normals, -np.einsum("ij,ij->i", normals, positions[triangles[:, 0]])))
        face_quadrics = planes[:, _QUADRIC_ROWS] * planes[:, _QUADRIC_COLS]
        for corner in range(3):
            np.add.at(quadrics, triangles[:, corner], face_quadrics)
        for face_id, (a, b, c) in enumerate(tris):
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

    if strength > 0.0 and np.any(weights):
        if weighting == "quadric":
            # Scaling the accumulated plane and boundary quadrics keeps their
            # minimizer on the surface instead of anchoring it to input points.
            quadrics *= (1.0 + strength * weights)[:, None]
        else:
            # Point quadrics accumulate the displacement of every original
            # important vertex, rather than forgetting past displacement after
            # an edge collapse.
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
    frozen = [False] * n
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
            u, w = [x for x in tris[face_id] if x != v]
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
    frozen_count = sum(frozen)
    locked_count = int(np.count_nonzero(locks))

    # The collapse loop works on Python floats and lists: NumPy scalar indexing
    # and arithmetic dominated the v0.1 run time.  Every expression keeps the
    # v0.1 operation order (np.dot of a 3-vector is the left-to-right sum), so
    # results are bit-for-bit those of the NumPy-scalar implementation.
    Q = quadrics.tolist()
    P = positions.tolist()
    normals_xyz = cross.tolist()
    # Unnormalized normals of the input triangles; normals_xyz drifts with
    # every collapse, so only these catch a face turned over in small steps.
    source_normals = cross.tolist()
    areas = [x * x + y * y + z * z for x, y, z in normals_xyz]
    W = weights.tolist()
    L = locks.tolist()
    alive = [True] * n
    face_alive = [True] * m
    revisions = [0] * n
    exact_vertex_ids = list(range(n))

    # Entry fields: error, edge length squared (zero-error tie break), a, b,
    # revisions, and the chosen position.  Pending entries deduplicate work.
    heap: list[tuple] = []
    pending: dict[tuple[int, int], tuple[int, int]] = {}
    rejects = {"topology": 0, "normal_or_area": 0, "duplicate_faces": 0, "source_flip": 0}
    collapses = 0
    remaining = m
    max_heap = 0
    heappush, heappop = heapq.heappush, heapq.heappop

    def candidate(a: int, b: int):
        """Heap entry for edge ``a < b``, or None when it may not collapse."""
        if frozen[a] or frozen[b]:
            return None
        lock_a, lock_b = L[a], L[b]
        if lock_a and lock_b:
            return None
        boundary = boundary_degree[a] > 0
        if boundary != (boundary_degree[b] > 0):
            return None
        if boundary and edge_counts[(a, b)] != 1:
            return None
        q0, q1, q2, q3, q4, q5, q6, q7, q8, q9 = map(add, Q[a], Q[b])
        # Doubled off-diagonal terms; 2.0 * q is exact, so each cost below is
        # evaluated exactly as q0*x*x + 2*q1*x*y + ... + q9, left to right.
        h1, h2, h3, h5, h6, h8 = 2.0 * q1, 2.0 * q2, 2.0 * q3, 2.0 * q5, 2.0 * q6, 2.0 * q8
        ax, ay, az = P[a]
        bx, by, bz = P[b]
        dx, dy, dz = bx - ax, by - ay, bz - az
        # A sum of squared plane distances is nonnegative, so a negative cost
        # is roundoff and becomes 0.  A non-finite cost (NaN or +-inf, which
        # MAX_STRENGTH rules out for any real mesh) becomes +inf: that position
        # never wins, and an edge whose every position is non-finite is not
        # queued, instead of NaN turning into the cheapest collapse.
        if lock_a or lock_b:
            x, y, z = (ax, ay, az) if lock_a else (bx, by, bz)
            cost = q0*x*x + h1*x*y + h2*x*z + h3*x + q4*y*y + h5*y*z + h6*y + q7*z*z + h8*z + q9
            if not cost > 0.0:
                cost = 0.0 if cost > -_INF else _INF
            if cost == _INF:
                return None
            return (cost, dx * dx + dy * dy + dz * dz, a, b, revisions[a], revisions[b], x, y, z)

        # Candidates in v0.1 order: midpoint, endpoints, the minimum on the
        # edge, then the unconstrained minimum; a later one must be strictly
        # cheaper to win, which fixes the choice between equal costs.
        x, y, z = (ax + bx) * 0.5, (ay + by) * 0.5, (az + bz) * 0.5
        cost = q0*x*x + h1*x*y + h2*x*z + h3*x + q4*y*y + h5*y*z + h6*y + q7*z*z + h8*z + q9
        if not cost > 0.0:
            cost = 0.0 if cost > -_INF else _INF
        c = q0*ax*ax + h1*ax*ay + h2*ax*az + h3*ax + q4*ay*ay + h5*ay*az + h6*ay + q7*az*az + h8*az + q9
        if not c > 0.0:
            c = 0.0 if c > -_INF else _INF
        if c < cost:
            cost, x, y, z = c, ax, ay, az
        c = q0*bx*bx + h1*bx*by + h2*bx*bz + h3*bx + q4*by*by + h5*by*bz + h6*by + q7*bz*bz + h8*bz + q9
        if not c > 0.0:
            c = 0.0 if c > -_INF else _INF
        if c < cost:
            cost, x, y, z = c, bx, by, bz

        # Minimum restricted to this edge; valid for boundaries and useful when
        # the full 3 x 3 system is singular on flat surfaces.
        adx = q0 * dx + q1 * dy + q2 * dz
        ady = q1 * dx + q4 * dy + q5 * dz
        adz = q2 * dx + q5 * dy + q7 * dz
        denominator = dx * adx + dy * ady + dz * adz
        if denominator > 1.0e-30:
            t = -(ax * adx + ay * ady + az * adz + q3 * dx + q6 * dy + q8 * dz) / denominator
            t = t if t > 0.0 else 0.0
            t = t if t < 1.0 else 1.0
            px, py, pz = ax + t * dx, ay + t * dy, az + t * dz
            c = q0*px*px + h1*px*py + h2*px*pz + h3*px + q4*py*py + h5*py*pz + h6*py + q7*pz*pz + h8*pz + q9
            if not c > 0.0:
                c = 0.0 if c > -_INF else _INF
            if c < cost:
                cost, x, y, z = c, px, py, pz

        if not boundary:
            # Explicit symmetric solve avoids tiny NumPy solver work arrays and
            # handles singular quadrics deterministically.
            c00 = q4 * q7 - q5 * q5
            c01 = q2 * q5 - q1 * q7
            c02 = q1 * q5 - q2 * q4
            c11 = q0 * q7 - q2 * q2
            c12 = q1 * q2 - q0 * q5
            c22 = q0 * q4 - q1 * q1
            det = q0 * c00 + q1 * c01 + q2 * c02
            trace = q0 + q4 + q7
            # A float ``**`` raises OverflowError above ~5.6e102.  MAX_STRENGTH
            # keeps that out of reach: a cluster trace is at most
            # (1 + MAX_STRENGTH) * (3 m + 20 boundary edges) + 3 MAX_STRENGTH n,
            # which needs ~1e96 faces or vertices.  ``**`` keeps v0.1 rounding.
            if det > 1.0e-12 * max(trace ** 3, 1.0e-30):
                px = -(c00 * q3 + c01 * q6 + c02 * q8) / det
                py = -(c01 * q3 + c11 * q6 + c12 * q8) / det
                pz = -(c02 * q3 + c12 * q6 + c22 * q8) / det
                if math.isfinite(px) and math.isfinite(py) and math.isfinite(pz):
                    c = q0*px*px + h1*px*py + h2*px*pz + h3*px + q4*py*py + h5*py*pz + h6*py + q7*pz*pz + h8*pz + q9
                    if not c > 0.0:
                        c = 0.0 if c > -_INF else _INF
                    if c < cost:
                        cost, x, y, z = c, px, py, pz
        if cost == _INF:
            return None
        return (cost, dx * dx + dy * dy + dz * dz, a, b, revisions[a], revisions[b], x, y, z)

    def schedule(a: int, b: int):
        """(Re)queue edge ``a < b`` unless its queued entry is still current."""
        key = (a, b)
        stamp = pending.get(key)
        if stamp is not None and stamp[0] == revisions[a] and stamp[1] == revisions[b]:
            return
        entry = candidate(a, b)
        if entry is None:
            if stamp is not None:
                del pending[key]
            return
        heappush(heap, entry)
        pending[key] = (entry[4], entry[5])

    def remove_face(face_id: int, tri):
        a, b, c = tri
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

    def collapse(a: int, b: int, x: float, y: float, z: float) -> bool:
        """Contract b into a at (x, y, z) if every safety check passes."""
        nonlocal remaining, collapses
        faces_a, faces_b = vertex_faces[a], vertex_faces[b]
        shared = faces_a & faces_b
        if len(shared) != 1 and len(shared) != 2:
            rejects["topology"] += 1
            return False
        opposite = set()
        for face_id in shared:
            for v in tris[face_id]:
                if v != a and v != b:
                    opposite.add(v)
        neighbors_a, neighbors_b = neighbors[a], neighbors[b]
        if neighbors_a & neighbors_b != opposite:
            rejects["topology"] += 1
            return False
        rest = faces_a ^ faces_b  # touched faces that survive the collapse
        if not rest:
            # Retain a triangle in each surface component instead of silently
            # deleting a tiny disconnected part when its final edge contracts.
            rejects["topology"] += 1
            return False
        surviving = []
        face_keys = set()
        flipped = False
        for face_id in sorted(rest):
            t0, t1, t2 = tri = tris[face_id]
            renamed = b in tri
            if renamed:
                t0, t1, t2 = [a if v == b else v for v in tri]
            key = tuple(sorted((t0, t1, t2)))
            if key in face_keys:
                rejects["duplicate_faces"] += 1
                return False
            face_keys.add(key)
            p0x, p0y, p0z = (x, y, z) if t0 == a else P[t0]
            p1x, p1y, p1z = (x, y, z) if t1 == a else P[t1]
            p2x, p2y, p2z = (x, y, z) if t2 == a else P[t2]
            ux, uy, uz = p1x - p0x, p1y - p0y, p1z - p0z
            vx, vy, vz = p2x - p0x, p2y - p0y, p2z - p0z
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            area_squared = nx * nx + ny * ny + nz * nz
            ox, oy, oz = normals_xyz[face_id]
            dot = nx * ox + ny * oy + nz * oz
            if (
                area_squared <= _AREA_EPS_SQUARED or dot <= 0.0
                or dot * dot <= _NORMAL_DOT_MIN_SQUARED * area_squared * areas[face_id]
            ):
                rejects["normal_or_area"] += 1
                return False
            if not flipped:
                sx, sy, sz = source_normals[face_id]
                flipped = nx * sx + ny * sy + nz * sz <= 0.0
            surviving.append((face_id, renamed, [t0, t1, t2], (nx, ny, nz), area_squared))
        if flipped:
            # Checked last, so this counts exactly the collapses that every
            # v0.1 check accepts: without any, the result equals v0.1's.
            rejects["source_flip"] += 1
            return False

        # Only the one-ring of b can change boundary status or lose edges;
        # faces of a without b keep their vertices and only get new normals.
        affected = neighbors_a | neighbors_b
        old_boundary = [(v, boundary_degree[v] > 0) for v in neighbors_b if v != a]
        removed = []
        for face_id in shared:
            removed.append(tris[face_id])
            remove_face(face_id, tris[face_id])
            face_alive[face_id] = False
        for face_id, renamed, tri, normal, area_squared in surviving:
            if renamed:
                removed.append(tris[face_id])
                remove_face(face_id, tris[face_id])
                tris[face_id] = tri
                add_face(face_id, tri)
            normals_xyz[face_id] = normal
            areas[face_id] = area_squared
        ax, ay, az = P[a]
        bx, by, bz = P[b]
        if x == ax and y == ay and z == az:
            pass
        elif x == bx and y == by and z == bz:
            exact_vertex_ids[a] = exact_vertex_ids[b]
        else:
            exact_vertex_ids[a] = -1
        P[a] = (x, y, z)
        Q[a] = list(map(add, Q[a], Q[b]))
        if W[b] > W[a]:
            W[a] = W[b]
        L[a] = L[a] or L[b]
        alive[b] = False
        L[b] = False
        remaining -= len(shared)
        collapses += 1
        revisions[a] += 1
        revisions[b] += 1
        for v, was_boundary in old_boundary:
            if (boundary_degree[v] > 0) != was_boundary:
                revisions[v] += 1
        for t0, t1, t2 in removed:
            for u, v in ((t0, t1), (t1, t2), (t2, t0)):
                key = (u, v) if u < v else (v, u)
                if key not in edge_counts:
                    pending.pop(key, None)
        # Existing unaffected pending edges are cheap no-ops; rejected local
        # edges are reconsidered because their topology may now be legal.
        for u in affected:
            if alive[u]:
                for v in neighbors[u]:
                    if u < v:
                        schedule(u, v)
                    elif v not in affected:
                        schedule(v, u)
        return True

    distinct = sorted(set(targets))
    snapshots: dict[int, SimplificationResult] = {}

    def capture(target: int):
        alive_mask = np.array(face_alive, dtype=bool)
        output_faces = np.array(tris, dtype=np.int64).reshape(-1, 3)[alive_mask]
        output_sources = source[alive_mask]
        used = np.array(L, dtype=bool)
        if len(output_faces):
            used[output_faces.ravel()] = True
        used &= np.array(alive, dtype=bool)
        remap = np.full(n, -1, dtype=np.int64)
        remap[used] = np.arange(np.count_nonzero(used), dtype=np.int64)
        output_vertices = np.array(P, dtype=np.float64).reshape(-1, 3)[used] * scale + origin
        # Preserve input coordinates bit-for-bit when a cluster stayed at an input
        # vertex.  This includes every locked vertex and an unchanged LOD0 mesh.
        exact_ids = np.array(exact_vertex_ids, dtype=np.int64)[used]
        exact_mask = exact_ids >= 0
        output_vertices[exact_mask] = original[exact_ids[exact_mask]]
        stats = {
            "original_faces": original_count,
            "cleaned_faces": m,
            "removed_invalid_faces": original_count - m,
            "output_faces": remaining,
            "target_faces": target,
            "collapses": collapses,
            "target_reached": remaining <= target,
            "stop_reason": "target_reached" if remaining <= target else "constraints",
            "nonmanifold_vertices": frozen_count,
            "locked_vertices": locked_count,
            "rejected_collapses": dict(rejects),
            "peak_heap_entries": max_heap,
            "normalization_scale": scale,
            "weighting": weighting,
            "strength": strength,
            "single_pass": True,
            "pass_targets": list(distinct),
        }
        snapshots[target] = SimplificationResult(
            vertices=output_vertices,
            faces=remap[output_faces],
            source_faces=output_sources,
            importance=np.array(W, dtype=np.float64)[used],
            stats=stats,
        )
        stats["seconds"] = time.perf_counter() - started

    smallest = distinct[0]
    span = m - smallest
    waiting = distinct[::-1]  # largest budget first

    def report():
        fraction = 1.0 if span <= 0 else (m - remaining) / span
        progress(min(1.0, max(0.0, fraction)))

    # An independent run with budget t stops right after the collapse that
    # first makes remaining <= t (or never starts when m <= t), so capturing at
    # exactly those moments reproduces it, including its stats.
    while waiting and remaining <= waiting[0]:
        capture(waiting.pop(0))
    if waiting:
        # Pops depend only on the heap contents, so heapify matches v0.1 pushes.
        for a, b in edge_counts:
            entry = candidate(a, b)
            if entry is not None:
                heap.append(entry)
                pending[(a, b)] = (0, 0)
        heapq.heapify(heap)
        max_heap = len(heap)
        while remaining > smallest and heap:
            entry = heappop(heap)
            a, b = entry[2], entry[3]
            key = (a, b)
            if pending.get(key) != (entry[4], entry[5]):
                continue
            del pending[key]
            if (
                not alive[a] or not alive[b] or key not in edge_counts
                or revisions[a] != entry[4] or revisions[b] != entry[5]
            ):
                continue
            if not collapse(a, b, entry[6], entry[7], entry[8]):
                continue
            max_heap = max(max_heap, len(heap))
            if len(heap) > max(10000, 4 * len(pending)):
                heap[:] = [item for item in heap if pending.get((item[2], item[3])) == (item[4], item[5])]
                heapq.heapify(heap)
            while waiting and remaining <= waiting[0]:
                capture(waiting.pop(0))
            if progress is not None and collapses % _PROGRESS_INTERVAL == 0:
                report()
        while waiting:
            capture(waiting.pop(0))
    if progress is not None:
        report()

    results = []
    delivered = set()
    for target in targets:
        result = snapshots[target]
        if target in delivered:
            result = SimplificationResult(
                vertices=result.vertices.copy(),
                faces=result.faces.copy(),
                source_faces=result.source_faces.copy(),
                importance=result.importance.copy(),
                stats={**result.stats, "rejected_collapses": dict(result.stats["rejected_collapses"]),
                       "pass_targets": list(distinct)},
            )
        delivered.add(target)
        results.append(result)
    return results
