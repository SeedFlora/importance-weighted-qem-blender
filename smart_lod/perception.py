"""Per-vertex perceptual importance channels for LOD simplification.

All channels are geometric heuristics computed on the evaluated triangle mesh;
there is no learned or semantic detection (no face/eye detector). Channels:

* ``saliency`` - multi-scale mesh saliency after Lee, Varshney & Jacobs (2005),
  "Mesh Saliency", ACM SIGGRAPH. Differences from the paper are documented in
  :func:`mesh_saliency`.
* ``silhouette`` - how often a vertex lies on or near the *outer* projected
  outline over generic orthographic views, with a soft contour weight matched
  to the view spacing (occlusion-aware only through the union mask).
* ``sharp`` - dihedral creases, open/nonmanifold edges and user-marked sharp
  edges, spread over one ring.
* ``visibility`` - camera frustum, front-facing and self-occlusion test
  (Blender only; needs ``mathutils``).

NumPy only at import time, so the module also runs outside Blender; mathutils
is imported lazily by :func:`camera_visibility`.
"""

import math
import time

import numpy as np

from .raster import orthographic_basis, rasterize


# Chosen on the tuning split only (results/tuning/PROTOCOL.md, Amandemen 1; selection.txt):
# silhouette alone gave the largest 13-view IoU gain within the Chamfer constraint, while
# saliency alone worsened IoU and ROI error there. Saliency and sharp remain available in
# the UI; visibility is used only when a scene camera is enabled.
DEFAULT_WEIGHTS = {'saliency': 0.0, 'silhouette': 1.0, 'sharp': 0.0, 'visibility': 0.15}
CHANNELS = ('saliency', 'silhouette', 'sharp', 'visibility')
# Silhouette channel defaults (the UI properties read these). 64 hemisphere
# views: contour tolerance 10.1 degrees, see outer_silhouette.
DEFAULT_SILHOUETTE_VIEWS = 64
DEFAULT_SILHOUETTE_RESOLUTION = 256
# Silhouette frame side = projected extent times this margin.
_FRAME_MARGIN = 1.04


# ---------------------------------------------------------------- mesh helpers

def _validate(vertices, faces):
    try:
        v = np.asarray(vertices, dtype=np.float64)
        raw = np.asarray(faces)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('vertices/faces must be numeric arrays') from exc
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError('vertices must have shape (N, 3)')
    if not np.isfinite(v).all():
        raise ValueError('vertex coordinates must be finite')
    if raw.size == 0:
        return v, np.zeros((0, 3), dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError('faces must be an (M, 3) triangle index array')
    if raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise ValueError('face indices must be finite integers')
    f = raw.astype(np.int64)
    if not np.equal(f, raw).all():
        raise ValueError('face indices must be integers')
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError('face index is outside the vertex array')
    return v, f


def _vertex_array(values, count, name):
    if values is None:
        return None
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (count,):
        raise ValueError(f'{name} must have shape ({count},), got {a.shape}')
    if not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite')
    return np.clip(a, 0.0, 1.0)


class _Mesh:
    """Shared topology: faces with a repeated index are dropped (they have no
    edges of their own); zero-area faces with distinct indices are kept for
    connectivity but get zero normal and zero weight."""

    def __init__(self, vertices, faces):
        v = vertices
        n = len(v)
        f = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])]
        self.v, self.f, self.n = v, f, n
        cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        self.cross = cross
        self.double_area = np.linalg.norm(cross, axis=1)
        good = self.double_area > 0.0
        self.good = good
        self.normals = np.zeros_like(cross)
        self.normals[good] = cross[good] / self.double_area[good, None]
        # Half-edge h = 3*face + k runs from f[k] to f[k+1].
        start, end = f.ravel(), f[:, [1, 2, 0]].ravel()
        lo, hi = np.minimum(start, end), np.maximum(start, end)
        keys, self.half_edge, self.edge_faces = np.unique(lo * n + hi, return_inverse=True, return_counts=True)
        self.half_edge = self.half_edge.reshape(-1)
        self.half_forward = start < end
        self.edges = np.column_stack((keys // n, keys % n))
        border = self.edges[self.edge_faces != 2].ravel()
        self.border_vertex = np.zeros(n, dtype=bool)
        self.border_vertex[border] = True
        self.used = np.zeros(n, dtype=bool)
        self.used[f.ravel()] = True

    def neighbour_max(self, values):
        """Max of ``values`` over each vertex's 1-ring (``-inf`` without neighbours)."""
        result = np.full(self.n, -np.inf)
        a, b = self.edges[:, 0], self.edges[:, 1]
        np.maximum.at(result, a, values[b])
        np.maximum.at(result, b, values[a])
        return result


# ---------------------------------------------------------------- saliency

def _laplacian_parts(mesh):
    """Clamped cotangent edge weights and barycentric lumped vertex areas."""
    v, f, area2 = mesh.v, mesh.f, mesh.double_area
    p = v[f]
    safe = np.where(mesh.good, area2, 1.0)
    cot = np.empty((len(f), 3))
    for k in range(3):
        u, w = p[:, (k + 1) % 3] - p[:, k], p[:, (k + 2) % 3] - p[:, k]
        cot[:, k] = np.einsum('ij,ij->i', u, w) / safe
    cot = np.where(mesh.good[:, None], np.clip(cot, 0.0, 1e3), 0.0)
    # The angle at corner k is opposite half-edge k+1.
    weights = np.bincount(mesh.half_edge, weights=0.5 * np.roll(cot, 1, axis=1).ravel(),
                          minlength=len(mesh.edges))
    area = np.bincount(f.ravel(), weights=np.repeat(area2 / 6.0, 3), minlength=mesh.n)
    return weights, area


def _laplace(values, edges, weights, n):
    delta = (values[edges[:, 0]] - values[edges[:, 1]]) * weights
    return np.bincount(edges[:, 0], weights=delta, minlength=n) - np.bincount(edges[:, 1], weights=delta, minlength=n)


def _mean_curvature(mesh, weights, area):
    """Unsigned |H| = |n . L x| / (2 A) at vertices with positive area, else 0.

    Using only the normal component removes the tangential term that the
    Laplacian of position has on irregular triangulations. Where the
    area-weighted normal cancels (e.g. two-sided sheets) |L x| is used.

    At open or nonmanifold edge vertices this estimate is biased: L x has a
    large co-normal term there (about 1/h times the curvature term) and the
    one-sided normal is tilted, so part of it leaks into the projection (a
    sphere cap's rim gets about half the true |H| at any resolution), and the
    step to the interior would be the strongest saliency feature. Those
    vertices therefore take the mean |H| of their 1-ring neighbours whose value
    is already known, filled ring by ring from the interior; a component with
    no interior vertex keeps the one-sided estimate.
    """
    n, f = mesh.n, mesh.f
    lx = np.column_stack([_laplace(mesh.v[:, k], mesh.edges, weights, n) for k in range(3)])
    normal = np.column_stack([np.bincount(f.ravel(), weights=np.repeat(mesh.cross[:, k], 3), minlength=n)
                              for k in range(3)])
    length = np.linalg.norm(normal, axis=1)
    total = np.bincount(f.ravel(), weights=np.repeat(mesh.double_area, 3), minlength=n)
    oriented = length > 1e-6 * total
    projected = np.where(oriented, np.abs(np.einsum('ij,ij->i', lx, normal)) / np.where(oriented, length, 1.0),
                         np.linalg.norm(lx, axis=1))
    valid = area > 0.0
    curvature = np.zeros(n)
    curvature[valid] = projected[valid] / (2.0 * area[valid])
    known = valid & ~mesh.border_vertex
    pending = valid & mesh.border_vertex
    a, b = mesh.edges[:, 0], mesh.edges[:, 1]
    while pending.any():
        source = np.where(known, curvature, 0.0)
        total = np.bincount(a, weights=source[b], minlength=n) + np.bincount(b, weights=source[a], minlength=n)
        count = np.bincount(a, weights=known[b], minlength=n) + np.bincount(b, weights=known[a], minlength=n)
        reached = pending & (count > 0)
        if not reached.any():
            break
        curvature[reached] = total[reached] / count[reached]
        known |= reached
        pending &= ~reached
    return curvature, valid


def _heat_smooth(values, t, x0, edges, weights, area, tolerance=1e-8, max_iterations=2000):
    """Solve (M + t L) u = M values with Jacobi-preconditioned CG."""
    n = len(values)
    degree = np.bincount(edges[:, 0], weights=weights, minlength=n) + np.bincount(edges[:, 1], weights=weights, minlength=n)
    diagonal = area + t * degree
    inverse = np.where(diagonal > 0.0, 1.0 / np.where(diagonal > 0.0, diagonal, 1.0), 0.0)
    b = area * values
    b_norm = float(np.linalg.norm(b))
    if b_norm == 0.0:
        return np.zeros(n), 0
    x = x0.copy()
    r = b - (area * x + t * _laplace(x, edges, weights, n))
    z = r * inverse
    p = z.copy()
    rz = float(r @ z)
    iterations = 0
    while iterations < max_iterations and float(np.linalg.norm(r)) > tolerance * b_norm:
        ap = area * p + t * _laplace(p, edges, weights, n)
        pap = float(p @ ap)
        if not pap > 0.0:
            break
        alpha = rz / pap
        x += alpha * p
        r -= alpha * ap
        z = r * inverse
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
        iterations += 1
    return x, iterations


def _saliency(mesh, scales=(2, 3, 4, 5, 6), epsilon_fraction=0.003, noise_floor=0.02, stats=None):
    try:
        scales = tuple(float(k) for k in np.atleast_1d(np.asarray(scales, dtype=np.float64)))
        epsilon_fraction, noise_floor = float(epsilon_fraction), float(noise_floor)
    except (TypeError, ValueError) as exc:
        raise ValueError('saliency parameters must be numbers') from exc
    if not scales or not all(math.isfinite(k) and k > 0 for k in scales):
        raise ValueError('scales must be a nonempty sequence of positive numbers')
    if not (math.isfinite(epsilon_fraction) and epsilon_fraction > 0):
        raise ValueError('epsilon_fraction must be positive')
    if not (math.isfinite(noise_floor) and noise_floor >= 0):
        raise ValueError('noise_floor must be nonnegative')
    n = mesh.n
    zeros = np.zeros(n)
    if not mesh.good.any():
        return zeros
    weights, area = _laplacian_parts(mesh)
    curvature, valid = _mean_curvature(mesh, weights, area)
    clip = float(np.percentile(curvature[valid], 99)) if valid.any() else 0.0
    if not clip > 0.0:
        return zeros
    curvature = np.minimum(curvature, clip)
    used = mesh.v[mesh.used]
    epsilon = epsilon_fraction * float(np.linalg.norm(used.max(axis=0) - used.min(axis=0)))
    smoothed, guess, iterations = {}, curvature, 0
    for sigma in sorted(set(scales) | {2.0 * k for k in scales}):
        s = sigma * epsilon
        guess, count = _heat_smooth(curvature, 0.5 * s * s, guess, mesh.edges, weights, area)
        smoothed[sigma] = guess
        iterations += count
    total = np.zeros(n)
    contrast = 0.0
    for k in scales:
        s = np.where(valid, np.abs(smoothed[k] - smoothed[2.0 * k]), 0.0)
        peak = float(s.max())
        if not peak > 0.0:
            continue
        contrast = max(contrast, peak / clip)
        s /= peak
        # Ties within 1e-8 of the unit peak count as maxima, so exactly symmetric
        # neighbours do not flip with rounding (scale/translation invariance).
        maxima = valid & (s > 0.0) & (s >= mesh.neighbour_max(s) - 1e-8)
        maxima[int(np.argmax(s))] = False
        mean_maximum = float(s[maxima].mean()) if maxima.any() else 0.0
        total += s * (1.0 - mean_maximum) ** 2
    top = float(total.max())
    if not top > 0.0:
        return zeros
    gate = 1.0 if noise_floor == 0 else min(1.0, contrast / noise_floor)
    if stats is not None:
        stats.update(saliency_contrast=contrast, saliency_gate=gate, saliency_cg_iterations=iterations)
    return np.clip(total / top * gate, 0.0, 1.0)


def mesh_saliency(vertices, faces, *, scales=(2, 3, 4, 5, 6), epsilon_fraction=0.003, noise_floor=0.02):
    """Multi-scale mesh saliency in [0, 1] (Lee, Varshney & Jacobs 2005).

    Curvature is the unsigned mean curvature from the clamped (nonnegative)
    cotangent Laplacian with barycentric lumped areas, clipped at its 99th
    percentile. For each ``k`` in ``scales`` with ``sigma = k * eps`` and
    ``eps = epsilon_fraction * bbox diagonal``, ``S_k = |G(sigma) - G(2 sigma)|``.
    Each ``S_k`` is divided by its maximum and multiplied by
    ``(1 - m)^2``, ``m`` the mean of its other local maxima (vertex >= all 1-ring
    neighbours; Itti-style suppression), then the scales are summed and divided
    by the maximum.

    Deviations from the paper, stated plainly:

    * ``G(sigma)`` is not the paper's Euclidean Gaussian-weighted neighbourhood
      average. It is one implicit heat-diffusion step ``(M + t L) u = M H`` with
      ``t = sigma^2 / 2``, which matches the Gaussian's variance but not its
      exact shape, measures distance intrinsically along the surface and
      smooths disconnected components independently.
    * Noise gate: when the strongest per-scale contrast ``max_k max(S_k)`` is
      below ``noise_floor`` times the clipped curvature, the result is scaled by
      ``contrast / noise_floor``. Otherwise a featureless surface (a sphere)
      would have its discretization noise stretched to [0, 1]. ``noise_floor=0``
      gives the ungated paper normalization.

    Vertices without a positive-area incident triangle get 0.
    """
    v, f = _validate(vertices, faces)
    return _saliency(_Mesh(v, f), scales, epsilon_fraction, noise_floor)


# ---------------------------------------------------------------- silhouette

def _fibonacci_hemisphere(count):
    """Near-uniform unit directions with z > 0 (d and -d give mirrored silhouettes)."""
    i = np.arange(count) + 0.5
    z = i / count
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    r = np.sqrt(1.0 - z * z)
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def _contour_tolerance(view_count):
    """``sin(rho)``, ``rho`` the angular radius of one view's share of the hemisphere.

    A spherical cap of area ``2 pi / view_count`` has ``cos(rho) = 1 - 1/view_count``
    (24 views: 16.6 degrees, 64: 10.1, 96: 8.3).
    """
    cosine = 1.0 - 1.0 / view_count
    return math.sqrt(max(0.0, 1.0 - cosine * cosine))


def _dilate(mask, steps):
    """Square (8-neighbour) dilation: pixels within Chebyshev distance ``steps``."""
    for _ in range(steps):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        mask = grown.copy()
        mask[:, 1:] |= grown[:, :-1]
        mask[:, :-1] |= grown[:, 1:]
    return mask


def _dot_gradient(mesh, faces, corner_face, starts):
    """Per vertex with faces, a 3x3 ``G`` such that ``G @ d`` is the least-squares
    gradient of ``n_f . d`` over the incident face centroids, taken in the plane
    of the area-weighted vertex normal (a sphere of radius r: ``d_t / r``; a
    plane: 0; a crease: about the normal jump over the ring size).

    ``corner_face`` indexes ``faces`` for the corners sorted by vertex and
    ``starts`` is the first sorted corner of each vertex.
    """
    count = np.diff(np.r_[starts, len(corner_face)])

    def ring_centred(values):
        return values - np.repeat(np.add.reduceat(values, starts, axis=0) / count[:, None], count, axis=0)

    offset = ring_centred(mesh.v[faces].mean(axis=1)[corner_face])
    normal = np.repeat(np.add.reduceat(mesh.cross[mesh.good][corner_face], starts, axis=0), count, axis=0)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, length, out=np.zeros_like(normal), where=length > 0.0)
    offset -= np.einsum('ij,ij->i', offset, normal)[:, None] * normal
    spread = ring_centred(mesh.normals[mesh.good][corner_face])
    a = np.add.reduceat(offset[:, :, None] * offset[:, None, :], starts, axis=0)
    m = np.add.reduceat(offset[:, :, None] * spread[:, None, :], starts, axis=0)
    return np.linalg.pinv(a, rcond=1e-9, hermitian=True) @ m


def _outer_silhouette(mesh, view_count=DEFAULT_SILHOUETTE_VIEWS, resolution=DEFAULT_SILHOUETTE_RESOLUTION,
                      band_pixels=1):
    if int(view_count) != view_count or view_count < 1:
        raise ValueError('view_count must be a positive integer')
    if int(resolution) != resolution or resolution < 8:
        raise ValueError('resolution must be an integer >= 8')
    if int(band_pixels) != band_pixels or band_pixels < 0:
        raise ValueError('band_pixels must be a nonnegative integer')
    view_count, resolution, band_pixels = int(view_count), int(resolution), int(band_pixels)
    counts = np.zeros(mesh.n)
    faces = mesh.f[mesh.good]
    if not len(faces):
        return counts
    normals = mesh.normals[mesh.good]
    corners = faces.ravel()
    order = np.argsort(corners, kind='stable')
    corner_face = order // 3
    starts = np.flatnonzero(np.r_[True, corners[order][1:] != corners[order][:-1]])
    owner = corners[order][starts]
    in_faces = np.zeros(mesh.n, dtype=bool)
    in_faces[owner] = True
    gradient = _dot_gradient(mesh, faces, corner_face, starts)
    # Open or nonmanifold edge vertices are contour in every view (the surface ends there).
    border = mesh.border_vertex & in_faces
    tolerance = _contour_tolerance(view_count)
    sag = 1.0 - math.sqrt(1.0 - tolerance * tolerance)
    # Inward image offset, on a sphere that fills the frame, of a point whose
    # normal is tolerance-off the contour; the band reaches that far inside.
    reach = band_pixels + int(math.ceil(0.5 * resolution * sag - 1e-9))
    weight = np.zeros(mesh.n)
    for d in _fibonacci_hemisphere(view_count):
        u, w = orthographic_basis(d)
        p = np.column_stack((mesh.v @ u, mesh.v @ w))
        lo, hi = p[in_faces].min(axis=0), p[in_faces].max(axis=0)
        side = float((hi - lo).max()) * _FRAME_MARGIN
        if not side > 0.0:
            continue
        # Soft contour weight (see outer_silhouette); the frame radius is side / 2.
        dots = (normals @ d)[corner_face]
        low, high = np.minimum.reduceat(dots, starts), np.maximum.reduceat(dots, starts)
        gap = np.maximum(np.maximum(low, -high), 0.0)
        slope = np.linalg.norm(gradient @ d, axis=1)
        local = np.minimum(tolerance, np.sqrt(slope * side * sag))
        weight[owner] = np.maximum(local - gap, 0.0) / tolerance
        weight[border] = 1.0
        lower = (lo + hi) / 2.0 - side / 2.0
        scale = resolution / side
        mask = rasterize(p[faces], lower, scale, resolution)
        padded = np.pad(mask, 1)
        interior = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
        band = _dilate(mask & ~interior, reach)
        pixel = np.clip(np.floor((p - lower) * scale).astype(np.int64), 0, resolution - 1)
        counts += np.where(band[pixel[:, 1], pixel[:, 0]], weight, 0.0)
    return counts


def outer_silhouette(vertices, faces, *, view_count=DEFAULT_SILHOUETTE_VIEWS,
                     resolution=DEFAULT_SILHOUETTE_RESOLUTION, band_pixels=1):
    """Score in [0, 1]: how often a vertex is on or near the outer outline over generic views.

    ``view_count`` Fibonacci directions on the upper hemisphere are used (an
    orthographic view and its opposite have mirrored, identical silhouettes).

    Soft contour weight per view ``d``: with ``[lo, hi]`` the range of ``n . d``
    over the vertex's incident face normals and ``gap = max(lo, -hi, 0)`` (0 when
    the faces face both ways, i.e. a contour vertex of the mesh), the weight is
    ``max(tau_v - gap, 0) / tau``. ``tau = sin(rho)`` is matched to the view
    spacing, ``rho`` the angular radius of one view's share of the hemisphere
    (:func:`_contour_tolerance`; 64 views: 10.1 degrees), so every vertex's
    contour band ``|n . d| < tau`` is crossed by several views and the sum is
    smooth over view directions. A binary contour test with few views instead
    gives stripes that depend on where the fixed views fall, i.e. on the mesh's
    orientation and tessellation. ``tau_v = min(tau, sqrt(2 s R (1 - cos rho)))``
    with ``s`` the surface gradient of ``n . d`` (:func:`_dot_gradient`) and
    ``R`` half the view frame: a point ``gap`` off its contour lies about
    ``gap^2 / (2 s)`` inside it in the image, and ``tau_v`` keeps that within the
    band that a tolerance-off point of a sphere filling the frame needs. So
    curved regions and creases (large ``s``) get the full kernel and flat
    regions get 0: a plane is on the outline only exactly edge-on, and a
    narrower kernel than the view spacing would alias again. Vertices on open
    or nonmanifold edges get weight 1 in every view.

    Occlusion: the union of all projected triangles is rasterized in a square
    frame; outline pixels are mask pixels with a 4-neighbour outside the mask
    (the image border counts as outside). A vertex's weight is added only when
    its pixel is within Chebyshev distance ``band_pixels + k`` of an outline
    pixel, ``k = ceil(resolution / 2 * (1 - cos rho))`` (64 views at 256
    pixels: 2). Contour vertices hidden inside the projected mask (a cavity, a
    concave fold) therefore do not count, while holes visible through the object
    (a torus) do. The summed weights are divided by the 90th percentile of the
    positive sums and clipped. The views are generic and are not the six
    evaluation axes of ``metrics``.
    """
    v, f = _validate(vertices, faces)
    counts = _outer_silhouette(_Mesh(v, f), view_count, resolution, band_pixels)
    return _normalize_counts(counts)


def _normalize_counts(counts):
    positive = counts[counts > 0]
    if not len(positive):
        return np.zeros_like(counts)
    return np.clip(counts / float(np.percentile(positive, 90)), 0.0, 1.0)


# ---------------------------------------------------------------- sharp edges

def _angle(sharp_angle):
    try:
        sharp_angle = float(sharp_angle)
    except (TypeError, ValueError) as exc:
        raise ValueError('sharp_angle must be a number') from exc
    if not (math.isfinite(sharp_angle) and 0.0 <= sharp_angle <= 180.0):
        raise ValueError('sharp_angle must be in [0, 180] degrees')
    return sharp_angle


def _sharp(mesh, sharp_angle=45.0, marked_sharp=None):
    sharp_angle = _angle(sharp_angle)
    marked = _vertex_array(marked_sharp, mesh.n, 'marked_sharp')
    flag = mesh.edge_faces != 2
    pairs = np.flatnonzero(mesh.edge_faces == 2)
    if len(pairs):
        order = np.argsort(mesh.half_edge, kind='stable')
        first = np.searchsorted(mesh.half_edge[order], pairs)
        h0, h1 = order[first], order[first + 1]
        n0, n1 = mesh.normals[h0 // 3], mesh.normals[h1 // 3]
        # Same traversal direction means the two faces disagree on orientation.
        flip = np.where(mesh.half_forward[h0] == mesh.half_forward[h1], -1.0, 1.0)
        cosine = np.einsum('ij,ij->i', n0, n1) * flip
        usable = mesh.good[h0 // 3] & mesh.good[h1 // 3]
        flag[pairs] = usable & (cosine <= math.cos(math.radians(sharp_angle)) + 1e-12)
    value = np.zeros(mesh.n)
    value[mesh.edges[flag].ravel()] = 1.0
    if marked is not None:
        value = np.maximum(value, marked)
    # Spread edge evidence over one ring without diluting the evidence itself.
    return np.maximum(value, 0.5 * np.maximum(mesh.neighbour_max(value), 0.0))


def sharp_features(vertices, faces, *, sharp_angle=45.0, marked_sharp=None):
    """Sharp-feature score in [0, 1].

    An edge is sharp when the angle between its two face normals is at least
    ``sharp_angle`` degrees (face orientation is reconciled per edge), or when
    it does not have exactly two faces (open boundary, nonmanifold). Vertices
    on sharp edges get 1, as do vertices set in ``marked_sharp`` (per-vertex,
    e.g. from Blender's "Mark Sharp"); 1-ring neighbours get half of that.
    """
    v, f = _validate(vertices, faces)
    return _sharp(_Mesh(v, f), sharp_angle, marked_sharp)


# ---------------------------------------------------------------- visibility

def _camera(camera):
    try:
        kind = camera['type']
        position = np.asarray(camera['position'], dtype=np.float64).reshape(3)
        forward = np.asarray(camera['forward'], dtype=np.float64).reshape(3)
        up = np.asarray(camera['up'], dtype=np.float64).reshape(3)
        clip_start, clip_end = float(camera['clip_start']), float(camera['clip_end'])
        if kind == 'PERSP':
            extent = float(camera['angle_x']), float(camera['angle_y'])
        elif kind == 'ORTHO':
            extent = float(camera['ortho_width']), float(camera['ortho_height'])
        else:
            raise ValueError("camera['type'] must be 'PERSP' or 'ORTHO'")
    except (KeyError, TypeError) as exc:
        raise ValueError(f'camera dict is incomplete: {exc!r}') from exc
    if not all(np.isfinite(x).all() for x in (position, forward, up, clip_start, clip_end, *extent)):
        raise ValueError('camera values must be finite')
    forward = forward / max(np.linalg.norm(forward), 1e-300)
    up = up - (up @ forward) * forward
    if not np.linalg.norm(forward) > 0.5 or not np.linalg.norm(up) > 1e-9:
        raise ValueError('camera forward/up must be nonzero and not parallel')
    up /= np.linalg.norm(up)
    if not 0.0 <= clip_start < clip_end:
        raise ValueError('camera clip range must satisfy 0 <= clip_start < clip_end')
    if kind == 'PERSP' and not all(0.0 < a < math.pi for a in extent):
        raise ValueError('camera angles must be in (0, pi) radians')
    if kind == 'ORTHO' and not all(e > 0.0 for e in extent):
        raise ValueError('camera ortho size must be positive')
    return kind, position, forward, up, np.cross(forward, up), clip_start, clip_end, extent


def _visibility(mesh, camera):
    result = np.zeros(mesh.n)
    if camera is None:
        return result
    kind, position, forward, up, right, clip_start, clip_end, extent = _camera(camera)
    faces = mesh.f[mesh.good]
    if not len(faces):
        return result
    relative = mesh.v - position
    depth, x, y = relative @ forward, relative @ right, relative @ up
    slack = 1e-9 * max(1.0, float(np.abs(relative).max()))
    inside = (depth >= clip_start - slack) & (depth <= clip_end + slack)
    normals = mesh.normals[mesh.good] * _winding(mesh, faces)
    if kind == 'PERSP':
        inside &= (np.abs(x) <= depth * math.tan(extent[0] / 2.0) + slack)
        inside &= (np.abs(y) <= depth * math.tan(extent[1] / 2.0) + slack)
        facing = np.einsum('ij,ij->i', normals.repeat(3, axis=0), relative[faces.ravel()]) < 0.0
    else:
        inside &= (np.abs(x) <= extent[0] / 2.0 + slack) & (np.abs(y) <= extent[1] / 2.0 + slack)
        facing = np.repeat(normals @ forward < 0.0, 3)
    front = np.bincount(faces.ravel(), weights=facing, minlength=mesh.n) > 0
    candidates = np.flatnonzero(inside & front)
    if not len(candidates):
        return result
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    # The BVH stores float32: build and cast in a frame centred on the mesh, and
    # start each ray where it enters the bounding sphere (nothing occludes before
    # it), so rounding scales with the mesh size, not with its distance from the
    # world origin or from the camera.
    used = mesh.v[mesh.used]
    centre = (used.max(axis=0) + used.min(axis=0)) / 2.0
    radius = float(np.linalg.norm(used - centre, axis=1).max())
    epsilon = max(1e-5 * float(np.linalg.norm(used.max(axis=0) - used.min(axis=0))), 1e-9)
    tree = BVHTree.FromPolygons((mesh.v - centre).tolist(), faces.tolist(), all_triangles=True)
    eye = position - centre
    # Distance (PERSP: along any ray; ORTHO: depth) before which the bounding sphere is not reached.
    entry = (float(np.linalg.norm(eye)) if kind == 'PERSP' else float(-eye @ forward)) - radius - epsilon
    for i in candidates.tolist():
        if kind == 'PERSP':
            ray = relative[i]
            distance = float(np.linalg.norm(ray))
            if distance <= epsilon:
                result[i] = 1.0
                continue
            ray = ray / distance
            # Start on the near clip plane (geometry in front of it is clipped
            # away) or where the ray enters the bounding sphere, if later.
            skip = max(clip_start / max(float(ray @ forward), 1e-12), entry)
            origin = eye + ray * skip
        else:
            ray = forward
            skip = max(clip_start, entry)
            distance = float(depth[i])
            origin = mesh.v[i] - centre - forward * (distance - skip)
        remaining = distance - skip
        if remaining <= epsilon:
            result[i] = 1.0
            continue
        hit, _, _, hit_distance = tree.ray_cast(Vector(origin), Vector(ray), remaining + epsilon)
        result[i] = float(hit is None or hit_distance >= remaining - epsilon)
    return result


def _winding(mesh, faces):
    """-1 when the mesh is wound inward (negative signed volume about its
    area-weighted centroid, e.g. a mirrored object whose faces were not flipped),
    else 1. Open surfaces enclosing no volume (a plane) keep their winding."""
    area = mesh.double_area[mesh.good]
    centroid = (mesh.v[faces].mean(axis=1) * area[:, None]).sum(axis=0) / area.sum()
    p = mesh.v[faces] - centroid
    volume = float(np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()) / 6.0
    return -1.0 if volume < -1e-9 * (0.5 * float(area.sum())) ** 1.5 else 1.0


def camera_visibility(vertices, faces, camera):
    """1 for vertices the camera can see, else 0 (all 0 when ``camera`` is None).

    Visible = inside the frustum (PERSP: within half of ``angle_x``/``angle_y``
    around ``forward`` and inside the clip range; ORTHO: inside the
    ``ortho_width`` x ``ortho_height`` box and clip range) AND front-facing
    (some incident face has ``n . (v - position) < 0``, ORTHO ``n . forward < 0``,
    with ``n`` taken outward: when the whole mesh is wound inward, i.e. its
    signed volume is negative, all normals are flipped for this test; an open
    surface enclosing no volume keeps its winding) AND not self-occluded (a
    ``mathutils`` BVH ray from the near clip plane, or from where the ray
    enters the mesh's bounding sphere if that is later, reaches the vertex; the
    BVH is built in a frame centred on the mesh so float32 rounding does not
    grow with the distance from the world origin).
    Only this mesh occludes; other scene objects and lens shift are ignored.
    Needs Blender's ``mathutils``; ``camera`` is the world-space dict described
    in :func:`camera_from_object`.
    """
    v, f = _validate(vertices, faces)
    return _visibility(_Mesh(v, f), camera)


def camera_from_object(camera_object, scene=None):
    """World-space camera dict for :func:`camera_visibility` from a Blender camera.

    Uses ``Camera.view_frame(scene=scene)`` so sensor fit and render aspect are
    respected. The frame is measured in the camera's local axes (x right, y up,
    looking down -z) and only the direction of the world axes is used, so the
    camera object's scale is ignored, as in rendering (it matches
    ``pipeline.camera_parameters``). A frustum with lens shift is replaced by
    the symmetric frustum that contains it. Panoramic cameras are treated as
    perspective.
    """
    matrix = np.asarray(camera_object.matrix_world, dtype=np.float64)
    data = camera_object.data
    corners = np.asarray([tuple(c) for c in data.view_frame(scene=scene)], dtype=np.float64)
    forward = -matrix[:3, 2] / np.linalg.norm(matrix[:3, 2])
    up = matrix[:3, 1] - (matrix[:3, 1] @ forward) * forward
    up /= np.linalg.norm(up)
    half_x, half_y, depth = np.abs(corners[:, 0]), np.abs(corners[:, 1]), -corners[:, 2]
    camera = {'position': tuple(matrix[:3, 3]), 'forward': tuple(forward), 'up': tuple(up),
              'clip_start': float(data.clip_start), 'clip_end': float(data.clip_end)}
    if data.type == 'ORTHO':
        camera.update(type='ORTHO', ortho_width=2.0 * float(half_x.max()), ortho_height=2.0 * float(half_y.max()))
    else:
        camera.update(type='PERSP', angle_x=2.0 * math.atan(float((half_x / depth).max())),
                      angle_y=2.0 * math.atan(float((half_y / depth).max())))
    return camera


# ---------------------------------------------------------------- combination

def _weights(weights):
    if weights is None:
        return dict(DEFAULT_WEIGHTS)
    if not hasattr(weights, 'keys') or not hasattr(weights, 'get'):
        raise ValueError('weights must be a dict of channel name -> weight')
    unknown = set(weights.keys()) - set(CHANNELS)
    if unknown:
        raise ValueError(f'unknown importance weight(s): {sorted(unknown)}; expected {list(CHANNELS)}')
    result = {}
    for name in CHANNELS:
        try:
            value = float(weights.get(name, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f'weight {name!r} must be a number') from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f'weight {name!r} must be finite and nonnegative')
        result[name] = value
    return result


def compute_importance(vertices, faces, *, weights=None, manual=None, camera=None,
                       sharp_angle=45.0, marked_sharp=None, silhouette_views=DEFAULT_SILHOUETTE_VIEWS,
                       silhouette_resolution=DEFAULT_SILHOUETTE_RESOLUTION, timings=None):
    """Combine the channels into a per-vertex importance in [0, 1].

    ``G = sum_k w_k f_k / sum_k w_k`` over the *active* channels: a channel is
    computed and active only when its weight is > 0, and ``visibility`` only
    when ``camera`` is not None; inactive channels are returned as zeros and do
    not dilute the others (G = 0 when nothing is active). ``weights=None``
    uses ``DEFAULT_WEIGHTS``; a given dict counts missing keys as 0 and rejects
    unknown keys and negative or nonfinite values. ``manual`` (per vertex, e.g. a
    protect vertex group) combines by maximum: ``importance = max(G, manual)``.

    Returns ``(importance, features)`` with feature keys ``saliency``,
    ``silhouette``, ``sharp``, ``visibility`` and ``manual``. If ``timings`` is a
    dict, per-channel wall-clock seconds are stored in it.
    """
    v, f = _validate(vertices, faces)
    w = _weights(weights)
    manual = _vertex_array(manual, len(v), 'manual')
    marked_sharp = _vertex_array(marked_sharp, len(v), 'marked_sharp')
    sharp_angle = _angle(sharp_angle)
    clock = time.perf_counter()
    mesh = _Mesh(v, f)
    seconds = {'topology': time.perf_counter() - clock}
    active = {name: w[name] > 0.0 for name in CHANNELS}
    active['visibility'] &= camera is not None
    features = {}
    compute = {
        'saliency': lambda: _saliency(mesh),
        'silhouette': lambda: _normalize_counts(_outer_silhouette(mesh, silhouette_views, silhouette_resolution)),
        'sharp': lambda: _sharp(mesh, sharp_angle, marked_sharp),
        'visibility': lambda: _visibility(mesh, camera),
    }
    for name in CHANNELS:
        clock = time.perf_counter()
        features[name] = compute[name]() if active[name] else np.zeros(len(v))
        seconds[name] = time.perf_counter() - clock
    total = sum(w[name] for name in CHANNELS if active[name])
    combined = np.zeros(len(v))
    if total > 0.0:
        combined = sum(w[name] * features[name] for name in CHANNELS if active[name]) / total
    features['manual'] = np.zeros(len(v)) if manual is None else manual
    importance = np.clip(np.maximum(combined, features['manual']), 0.0, 1.0)
    if timings is not None:
        timings.update(seconds)
    return importance, features
