"""Orthographic projection helpers and a NumPy triangle rasterizer.

Shared by silhouette evaluation (``metrics``) and silhouette importance
(``perception``). No Blender dependency.
"""

import numpy as np


# Triangles whose clamped pixel bounding box has at most this many pixels are
# evaluated together in padded tiles; larger ones use a per-triangle loop that
# can skip tiles already fully covered. Both paths evaluate the same expression.
_BATCH_MAX_PIXELS = 1024
# Upper bound on tile elements per batched evaluation; keeps each temporary
# array at about 2 MB regardless of mesh size.
_BATCH_CHUNK_ELEMENTS = 1 << 18
# Tile side lengths: exact up to 8, then four steps per octave (<= 25% padding
# per side). Must reach _BATCH_MAX_PIXELS, because a 1 x N sliver is batched.
_TILE_SIZES = np.array(list(range(1, 8)) + [(8 << octave) * step // 4 for octave in range(8) for step in (4, 5, 6, 7)])


def orthographic_basis(direction):
    """Return a deterministic orthonormal image basis ``(u, w)`` for a view direction.

    ``u`` and ``w`` are perpendicular to ``direction`` and to each other.
    Opposite directions produce mirrored images with identical silhouettes.
    """
    d = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(d))
    if not np.isfinite(length) or length == 0.0:
        raise ValueError("view direction must be finite and nonzero")
    d = d / length
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, helper)
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    return u, w


def _fill_batched(mask, rows, minimum, size, a, b, c, origin):
    """OR the coverage of triangles ``rows`` into ``mask`` using padded tiles.

    Bounding boxes are padded to tabulated tile shapes so each group shares one
    broadcast; padding pixels are excluded by the explicit bounding-box test
    (the edge tolerance could otherwise admit centres just outside the box).
    ``origin`` is the pixel index ``(x, y)`` of ``mask[0, 0]``.
    """
    width = mask.shape[1]
    flat = mask.reshape(-1)
    size = size[rows]
    shape = np.searchsorted(_TILE_SIZES, size)
    key = shape[:, 0] * len(_TILE_SIZES) + shape[:, 1]
    order = np.argsort(key, kind="stable")
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    for group in np.split(order, bounds):
        tile_x = int(_TILE_SIZES[shape[group[0], 0]])
        tile_y = int(_TILE_SIZES[shape[group[0], 1]])
        step = max(1, _BATCH_CHUNK_ELEMENTS // (tile_x * tile_y))
        dx = np.arange(tile_x)
        dy = np.arange(tile_y)
        for start in range(0, len(group), step):
            local = group[start:start + step]
            index = rows[local]
            x0 = minimum[index, 0]
            y0 = minimum[index, 1]
            # Same pixel-centre values as the reference loop: integer + 0.5.
            x = ((x0[:, None] + dx) + 0.5)[:, None, :]
            y = ((y0[:, None] + dy) + 0.5)[:, :, None]
            covered = (dx < size[local, 0:1])[:, None, :] & (dy < size[local, 1:2])[:, :, None]
            value = np.empty(covered.shape)
            inside = np.empty(covered.shape, dtype=bool)
            for edge in range(3):
                # (a*x + b*y) + c: the loop path's operation order, so bitwise equal.
                np.add(a[index, edge, None, None] * x, b[index, edge, None, None] * y, out=value)
                np.add(value, c[index, edge, None, None], out=value)
                np.greater_equal(value, -1e-9, out=inside)
                covered &= inside
            hit, row, column = np.nonzero(covered)
            flat[(y0[hit] + row - origin[1]) * width + x0[hit] + column - origin[0]] = True


def rasterize(triangles, lower, pixel_scale, resolution, canvas=None):
    """Rasterize a projected triangle soup, including both winding directions.

    ``triangles`` has shape (M, 3, 2). Points map to pixel units by
    ``(p - lower) * pixel_scale``; pixel ``(i, j)`` has its center at
    ``(i + 0.5, j + 0.5)`` and is covered when that center lies inside or on a
    triangle edge (edge functions >= -1e-9 after orienting each triangle
    counter-clockwise). Triangles with ``|signed pixel area| <= 1e-12`` are
    skipped. Returns a boolean mask indexed ``[y, x]``.

    The mask covers pixels ``0 <= i, j < resolution`` unless ``canvas`` gives
    another integer index range ``(x0, y0, x1, y1)`` (``x0 <= i < x1``,
    ``y0 <= j < y1``, may be negative) on the same lattice; then ``mask[j - y0,
    i - x0]`` is pixel ``(i, j)``. A pixel gets the same value on every canvas
    that contains it, so a larger canvas only adds pixels.

    Small triangles are evaluated in batched NumPy tiles and large ones in a
    per-triangle loop; the result is bitwise identical to evaluating every
    triangle's clamped bounding box one at a time (the v0.1 algorithm).
    """
    x_lo, y_lo, x_hi, y_hi = (0, 0, resolution, resolution) if canvas is None else (int(v) for v in canvas)
    if x_hi <= x_lo or y_hi <= y_lo:
        raise ValueError("canvas must have x1 > x0 and y1 > y0")
    projected = (triangles - lower) * pixel_scale
    mask = np.zeros((y_hi - y_lo, x_hi - x_lo), dtype=bool)
    p0, p1, p2 = projected[:, 0], projected[:, 1], projected[:, 2]
    minimum = np.ceil(np.minimum(np.minimum(p0, p1), p2) - 0.5)
    maximum = np.floor(np.maximum(np.maximum(p0, p1), p2) - 0.5)
    # Far-off coordinates would overflow the int cast. Every value lies within
    # one pixel of [minimum.min(), maximum.max()], so two reductions decide.
    bound = 2.0 ** 62
    if len(minimum) and not (minimum.min() > -bound and maximum.max() < bound):
        minimum, maximum = minimum.clip(-bound, bound), maximum.clip(-bound, bound)
    minimum = np.maximum(minimum.astype(np.int64), (x_lo, y_lo))
    maximum = np.minimum(maximum.astype(np.int64), (x_hi - 1, y_hi - 1))
    signed_area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    valid = (maximum[:, 0] >= minimum[:, 0]) & (maximum[:, 1] >= minimum[:, 1]) & (np.abs(signed_area) > 1e-12)
    if not valid.any():
        return mask
    # Only triangles that can cover a pixel centre need edge functions.
    projected, minimum, maximum = projected[valid], minimum[valid], maximum[valid]
    # Cross(edge, point - edge_start) = a*x + b*y + c.
    edge_start = projected
    edge_end = projected[:, [1, 2, 0]]
    delta = edge_end - edge_start
    a = -delta[:, :, 1]
    b = delta[:, :, 0]
    c = delta[:, :, 1] * edge_start[:, :, 0] - delta[:, :, 0] * edge_start[:, :, 1]
    sign = np.sign(signed_area[valid])[:, None]
    a *= sign
    b *= sign
    c *= sign
    size = maximum - minimum + 1
    batched = size[:, 0] * size[:, 1] <= _BATCH_MAX_PIXELS
    if batched.any():
        _fill_batched(mask, np.flatnonzero(batched), minimum, size, a, b, c, (x_lo, y_lo))
    # Integer-valued floats plus 0.5: the same centre values on any canvas.
    x_centres = np.arange(x_lo, x_hi, dtype=np.float64) + 0.5
    y_centres = np.arange(y_lo, y_hi, dtype=np.float64) + 0.5
    for index in np.flatnonzero(~batched):
        x0, y0 = minimum[index] - (x_lo, y_lo)
        x1, y1 = maximum[index] + 1 - (x_lo, y_lo)
        tile = mask[y0:y1, x0:x1]
        if tile.all():
            continue
        x = x_centres[None, x0:x1]
        y = y_centres[y0:y1, None]
        covered = np.ones(tile.shape, dtype=bool)
        for edge in range(3):
            covered &= a[index, edge] * x + b[index, edge] * y + c[index, edge] >= -1e-9
        tile |= covered
    return mask
