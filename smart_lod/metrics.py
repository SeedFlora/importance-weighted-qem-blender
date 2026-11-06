"""Reproducible geometry and silhouette measurements without changing a scene.

All public distances are in the units of the supplied world-space coordinates.
Chamfer is the mean of the two directed, *unsquared* point-to-surface means;
Hausdorff is a sampled estimate, not the exact continuous Hausdorff distance.
Silhouettes are binary triangle coverage at pixel centers in six orthographic
views. Opposite views have identical IoU and are computed from the same mask.

Silhouette framing (v0.1 and earlier v0.2 builds framed each view on the union
of both meshes' projected bounds, so pixel size and grid origin moved with
every candidate): per view, pixel size and grid origin come from the reference
alone, a square frame 1.04x the reference's larger projected extent, centred
on it. The reference mask is therefore identical for every candidate. A
candidate that overhangs the frame widens the canvas by whole pixels of the
same lattice (at most ``resolution`` extra pixels per axis) instead of
rescaling; anything beyond that cap is not counted and the view is listed as
clipped. When the candidate's projected bounds lie within the reference's,
values equal v0.1; otherwise they differ by grid aliasing (bunny Decimate
LODs at 128 px: up to 0.0017 per view, up to 0.0007 on the six-view mean).

Additions in v0.2 (existing key names unchanged; multiview and ROI are options):

* Multiview silhouette IoU over 13 unique orthographic directions (3 axes,
  4 body diagonals, 6 face diagonals). Each direction also stands for its
  opposite, whose silhouette is a mirror image. Still a small fixed view set,
  not an integral over all view directions.
* Region-of-interest (ROI) distances from per-vertex reference weights. The
  region is defined on the reference; ``*_ref_to_candidate`` samples it
  directly. ``*_candidate_to_ref`` uses the global candidate samples, each
  assigned to the region of its nearest reference face (the same
  nearest-surface correspondence the global distances use), so candidate-side
  artifacts such as spikes inside the ROI are seen. ``*_symmetric`` combines
  both directions like the global Chamfer/Hausdorff.
* ``hausdorff_with_vertices``: the sampled Hausdorff estimate, also maximized
  over every vertex of each mesh's nondegenerate triangles queried against
  the other surface. It catches spike tips and small detached triangles that
  area sampling misses. Still a lower bound on the exact Hausdorff distance:
  the farthest point can also lie inside a face.
"""

import math

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .raster import orthographic_basis
from .raster import rasterize as _rasterize


_VIEW_AXES = (("+X", "-X", 1, 2), ("+Y", "-Y", 0, 2), ("+Z", "-Z", 0, 1))
# Unique directions of the 3x3x3 neighbourhood (d and -d give mirrored silhouettes).
_MULTIVIEW_DIRECTIONS = (
    ("+X", (1, 0, 0)), ("+Y", (0, 1, 0)), ("+Z", (0, 0, 1)),
    ("+X+Y+Z", (1, 1, 1)), ("+X+Y-Z", (1, 1, -1)), ("+X-Y+Z", (1, -1, 1)), ("-X+Y+Z", (-1, 1, 1)),
    ("+X+Y", (1, 1, 0)), ("+X-Y", (1, -1, 0)), ("+X+Z", (1, 0, 1)),
    ("+X-Z", (1, 0, -1)), ("+Y+Z", (0, 1, 1)), ("+Y-Z", (0, 1, -1)),
)
# SeedSequence(seed).spawn(4): children 0/1 are the same streams v0.1 got from spawn(2).
_RNG_CHILDREN = ("reference", "candidate", "roi", "nonroi")
_FRAME_MARGIN = 1.04  # square frame side / larger projected reference extent (2% per side)


def _validate_geometry(vertices, faces, name):
    try:
        vertices = np.asarray(vertices, dtype=np.float64)
        raw_faces = np.asarray(faces)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: vertices/faces must be numeric arrays") from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError(f"{name}: vertices must have shape (N, 3), with N >= 3")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{name}: vertex coordinates must be finite")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or len(raw_faces) == 0:
        raise ValueError(f"{name}: faces must be a nonempty (M, 3) triangle array")
    if raw_faces.dtype.kind not in "iuf":
        raise ValueError(f"{name}: face indices must be integers")
    if not np.isfinite(raw_faces).all():
        raise ValueError(f"{name}: face indices must be finite")
    if np.any(raw_faces < 0) or np.any(raw_faces >= len(vertices)):
        raise ValueError(f"{name}: face index is outside the vertex array")
    faces = raw_faces.astype(np.int64)
    if not np.equal(faces, raw_faces).all():
        raise ValueError(f"{name}: face indices must be integers")
    return vertices, faces


def _validate_roi(roi_weights, vertex_count):
    """Return float64 per-vertex weights within [0, 1], or raise ``ValueError``."""
    try:
        raw = np.asarray(roi_weights)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("roi_weights must be a numeric array") from exc
    if raw.dtype.kind not in "biuf":
        raise ValueError("roi_weights must be a numeric array")
    if raw.shape != (vertex_count,):
        raise ValueError(f"roi_weights must have shape ({vertex_count},): one weight per reference vertex")
    weights = raw.astype(np.float64)
    if not np.isfinite(weights).all():
        raise ValueError("roi_weights must be finite")
    if np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError("roi_weights must lie within [0, 1]")
    return weights


def _surface_data(vertices, faces, name):
    triangles = vertices[faces]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if not np.isfinite(double_areas).all():
        raise ValueError(f"{name}: geometry has an unsupported coordinate range")
    usable = double_areas > 0.0
    if not usable.any():
        raise ValueError(f"{name}: mesh contains no nondegenerate triangles")
    areas = double_areas[usable]
    probabilities = areas / areas.sum()
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name}: geometry has an unsupported area range")
    return faces[usable], triangles[usable], probabilities, int((~usable).sum())


def _sample_surface(triangles, probabilities, sample_count, rng):
    """Area-uniform sampling using square-root barycentric coordinates."""
    selected = rng.choice(len(triangles), size=sample_count, p=probabilities)
    triangle = triangles[selected]
    random = rng.random((sample_count, 2))
    root_u = np.sqrt(random[:, 0:1])
    v = random[:, 1:2]
    return (
        (1.0 - root_u) * triangle[:, 0]
        + root_u * (1.0 - v) * triangle[:, 1]
        + root_u * v * triangle[:, 2]
    )


def _surface_tree(vertices, faces):
    return BVHTree.FromPolygons(vertices.tolist(), faces.tolist(), all_triangles=True)


def _nearest_distances(tree, points, with_faces=False):
    """Distance from each point to the closest point on the tree's triangle surface.

    With ``with_faces`` also return the index of that closest triangle in the
    face list the tree was built from.
    """
    distances = np.empty(len(points), dtype=np.float64)
    faces = np.empty(len(points), dtype=np.int64)
    find_nearest = tree.find_nearest
    for index, point in enumerate(points.tolist()):
        result = find_nearest(Vector(point))
        if result[0] is None or result[3] is None or not math.isfinite(result[3]):
            raise ValueError("Could not find a finite closest surface point")
        distances[index] = result[3]
        faces[index] = result[2]
    return (distances, faces) if with_faces else distances


def _frame_iou(reference, candidate, resolution):
    """IoU of two projected (M, 3, 2) soups on a pixel grid fixed by the reference.

    Returns ``(iou, clipped)``. The square frame (side ``_FRAME_MARGIN`` times
    the reference's larger extent, centred on it) sets pixel size and origin
    and holds the whole reference mask. Pixels the candidate's bounds reach
    outside it are added on the same lattice, at most ``resolution`` per axis;
    when both sides need more, each side gets at least ``resolution // 2``.
    ``clipped`` is True when the candidate's bounds reach beyond the added
    pixels; coverage there is not counted, so the IoU is then an upper bound.
    """
    lower, upper = reference.min(axis=(0, 1)), reference.max(axis=(0, 1))
    side = float(np.max(upper - lower)) * _FRAME_MARGIN
    if not math.isfinite(side) or side <= 0.0:
        raise ValueError("Could not construct a finite silhouette projection")
    frame_lower = (lower + upper) * 0.5 - side * 0.5
    scale = resolution / side
    reference_mask = _rasterize(reference, frame_lower, scale, resolution)
    # Pixel indices whose centres the candidate's bounds reach; the rasterizer
    # computes the same values per triangle, since min/max commute with them.
    reach_low = np.ceil((candidate.min(axis=(0, 1)) - frame_lower) * scale - 0.5)
    reach_high = np.floor((candidate.max(axis=(0, 1)) - frame_lower) * scale - 0.5)
    need_low, need_high = np.maximum(-reach_low, 0.0), np.maximum(reach_high - (resolution - 1), 0.0)
    low = np.minimum(need_low, np.maximum(resolution - need_high, resolution // 2))
    high = np.minimum(need_high, resolution - low)
    clipped = bool(np.any(low < need_low) or np.any(high < need_high))
    (x0, y0), (x1, y1) = low.astype(np.int64), high.astype(np.int64)
    candidate_mask = _rasterize(candidate, frame_lower, scale, resolution, (-x0, -y0, resolution + x1, resolution + y1))
    inner = candidate_mask[y0:y0 + resolution, x0:x0 + resolution]
    intersection = int(np.count_nonzero(reference_mask & inner))
    # Candidate pixels outside the frame add to the union only.
    union = int(np.count_nonzero(reference_mask | inner)) + int(np.count_nonzero(candidate_mask)) - int(
        np.count_nonzero(inner))
    # An edge-on planar surface can have no coverage in either projection.
    return (float(intersection / union) if union else 1.0), clipped


def _silhouette_iou(reference_triangles, candidate_triangles, resolution):
    """Axis-view IoU dict and the list of its labels whose canvas clipped the candidate."""
    result, clipped = {}, []
    for positive, negative, axis_u, axis_v in _VIEW_AXES:
        iou, clip = _frame_iou(
            reference_triangles[:, :, [axis_u, axis_v]], candidate_triangles[:, :, [axis_u, axis_v]], resolution
        )
        result[positive] = iou
        result[negative] = iou
        clipped += [positive, negative] if clip else []
    return result, clipped


def _project(vertices, u, w):
    """Image coordinates (N, 2); explicit sums keep results independent of BLAS kernels."""
    return np.column_stack((
        vertices[:, 0] * u[0] + vertices[:, 1] * u[1] + vertices[:, 2] * u[2],
        vertices[:, 0] * w[0] + vertices[:, 1] * w[1] + vertices[:, 2] * w[2],
    ))


def _multiview_iou(reference_vertices, reference_faces, candidate_vertices, candidate_faces, axis_views,
                   axis_clipped, resolution):
    """IoU per unique direction and clipped labels; axis directions reuse the axis-view results exactly."""
    views, clipped = {}, []
    for label, direction in _MULTIVIEW_DIRECTIONS:
        if label in axis_views:
            views[label], clip = axis_views[label], label in axis_clipped
        else:
            u, w = orthographic_basis(direction)
            views[label], clip = _frame_iou(
                _project(reference_vertices, u, w)[reference_faces],
                _project(candidate_vertices, u, w)[candidate_faces],
                resolution,
            )
        clipped += [label] if clip else []
    return views, clipped


def _roi_distances(weights, threshold, faces, triangles, tree, sample_count, children, diagonal,
                   candidate_distances, candidate_nearest):
    """ROI and complement distances in both directions, plus their symmetric combination.

    ``faces``/``triangles`` are the nondegenerate reference triangles in
    normalized coordinates; ``tree`` is the candidate surface in the same frame.
    ``candidate_distances``/``candidate_nearest`` are the global candidate
    samples' distances to the reference and their nearest reference triangle
    (an index into ``faces``); a sample belongs to the region of that triangle.
    """
    in_roi = weights[faces].mean(axis=1) >= threshold
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    result = {
        "roi_triangles": int(np.count_nonzero(in_roi)),
        "roi_area_fraction": float(double_areas[in_roi].sum() / double_areas.sum()),
    }
    notes = []
    for prefix, region, child in (("roi", in_roi, children[2]), ("nonroi", ~in_roi, children[3])):
        values = dict.fromkeys(("ref_to_candidate", "candidate_to_ref", "symmetric"), (None, None))
        attributed = candidate_distances[region[candidate_nearest]]
        if region.any():
            areas = double_areas[region]
            samples = _sample_surface(triangles[region], areas / areas.sum(), sample_count, np.random.default_rng(child))
            distances = _nearest_distances(tree, samples)
            forward = values["ref_to_candidate"] = (float(distances.mean()), float(distances.max()))
            if len(attributed):
                backward = values["candidate_to_ref"] = (float(attributed.mean()), float(attributed.max()))
                values["symmetric"] = (0.5 * (forward[0] + backward[0]), max(forward[1], backward[1]))
            else:
                notes.append(f"{prefix}_region_has_no_candidate_samples_candidate_to_ref_and_symmetric_are_none")
        else:
            notes.append(f"{prefix}_region_empty_distances_are_none")
        for direction, (chamfer, hausdorff) in values.items():
            result[f"{prefix}_chamfer_{direction}"] = None if chamfer is None else chamfer * diagonal
            result[f"{prefix}_chamfer_{direction}_normalized"] = chamfer
            result[f"{prefix}_hausdorff_{direction}"] = None if hausdorff is None else hausdorff * diagonal
            result[f"{prefix}_hausdorff_{direction}_normalized"] = hausdorff
        result[f"{prefix}_candidate_samples"] = int(len(attributed))
    return result, notes


def evaluate_meshes(
    reference_vertices,
    reference_faces,
    candidate_vertices,
    candidate_faces,
    sample_count=4000,
    resolution=128,
    seed=42,
    *,
    roi_weights=None,
    roi_threshold=0.5,
    multiview=True,
):
    """Measure a candidate against a reference; return only JSON-safe values.

    Parameters are world-space vertex arrays and triangulated index arrays.
    Zero-area triangles are excluded from surface sampling/BVH/rasterization but
    included in the reported input triangle counts. An entirely degenerate mesh,
    nonfinite coordinates or invalid indices raises ``ValueError``.

    Each mesh contributes ``sample_count`` independent area-uniform samples.
    Distances query the other mesh's continuous triangle surface, not its sample
    cloud. Coordinates are normalized before Blender's float32 BVH operations to
    reduce precision loss from scene scale and world-space offsets.
    ``hausdorff_with_vertices`` also queries every vertex of either mesh's
    nondegenerate triangles against the other surface (unused vertices and
    vertices of zero-area triangles only are not surface points).

    Silhouettes use a per-view frame fixed by the reference projection (see the
    module docstring). ``silhouette_frame_clipped_views`` lists the axis views
    whose capped canvas could not hold the candidate's projected bounds (then
    that view's IoU is an upper bound). The cap adds one frame width per axis,
    so this needs an overhang of more than a frame width on one side or half
    of one on both sides.

    ``multiview`` adds ``silhouette_iou_multiview_mean``,
    ``silhouette_iou_multiview_views`` and
    ``silhouette_multiview_frame_clipped_views`` over 13 unique orthographic
    directions, framed like the axis views at the same resolution.

    ``roi_weights`` (shape ``(N_ref,)``, finite, within [0, 1]) marks a reference
    region of interest: nondegenerate reference triangles whose mean corner
    weight is ``>= roi_threshold``. ``sample_count`` area-uniform samples are
    drawn on the ROI and on its complement, and their distances to the
    candidate surface give ``{roi,nonroi}_*_ref_to_candidate`` mean ("chamfer")
    and sampled max ("hausdorff") values. The global candidate samples, each
    assigned to the region of its nearest reference triangle, give
    ``*_candidate_to_ref`` (``*_candidate_samples`` counts them; about the
    region's share of the candidate area, so noisier for small regions).
    ``*_symmetric`` is the mean of the two directed means (chamfer) and the max
    of the two directed maxima (hausdorff). An empty region, or one no
    candidate sample is assigned to, reports ``None`` for the affected keys.

    Keys present without the options, and their values, are unchanged by them.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)) or resolution < 8:
        raise ValueError("resolution must be an integer >= 8")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(multiview, (bool, np.bool_)):
        raise ValueError("multiview must be a bool")
    if (
        isinstance(roi_threshold, (bool, np.bool_))
        or not isinstance(roi_threshold, (int, float, np.integer, np.floating))
        or not 0.0 <= float(roi_threshold) <= 1.0
    ):
        raise ValueError("roi_threshold must be a real number within [0, 1]")
    sample_count, resolution, seed = int(sample_count), int(resolution), int(seed)
    multiview, roi_threshold = bool(multiview), float(roi_threshold)
    ref_vertices, ref_faces = _validate_geometry(reference_vertices, reference_faces, "reference")
    candidate_vertices, candidate_faces = _validate_geometry(candidate_vertices, candidate_faces, "candidate")
    weights = None if roi_weights is None else _validate_roi(roi_weights, len(ref_vertices))
    reference_min = ref_vertices.min(axis=0)
    reference_extent = ref_vertices.max(axis=0) - reference_min
    diagonal = float(np.linalg.norm(reference_extent))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("reference: bounding-box diagonal must be finite and positive")
    origin = reference_min + reference_extent * 0.5
    ref_normalized = (ref_vertices - origin) / diagonal
    candidate_normalized = (candidate_vertices - origin) / diagonal
    if not np.isfinite(candidate_normalized).all():
        raise ValueError("candidate: geometry has an unsupported coordinate range")
    ref_valid_faces, ref_triangles, ref_probabilities, ref_degenerate = _surface_data(ref_normalized, ref_faces, "reference")
    candidate_valid_faces, candidate_triangles, candidate_probabilities, candidate_degenerate = _surface_data(
        candidate_normalized, candidate_faces, "candidate"
    )
    children = np.random.SeedSequence(seed).spawn(len(_RNG_CHILDREN))
    rng_reference, rng_candidate = np.random.default_rng(children[0]), np.random.default_rng(children[1])
    ref_samples = _sample_surface(ref_triangles, ref_probabilities, sample_count, rng_reference)
    candidate_samples = _sample_surface(candidate_triangles, candidate_probabilities, sample_count, rng_candidate)
    candidate_tree = _surface_tree(candidate_normalized, candidate_valid_faces)
    ref_tree = _surface_tree(ref_normalized, ref_valid_faces)
    ref_to_candidate = _nearest_distances(candidate_tree, ref_samples)
    candidate_to_ref, candidate_nearest = _nearest_distances(ref_tree, candidate_samples, with_faces=True)
    chamfer_normalized = float((ref_to_candidate.mean() + candidate_to_ref.mean()) * 0.5)
    hausdorff_normalized = float(max(ref_to_candidate.max(), candidate_to_ref.max()))
    hausdorff_vertices_normalized = float(max(
        hausdorff_normalized,
        _nearest_distances(candidate_tree, ref_normalized[np.unique(ref_valid_faces)]).max(),
        _nearest_distances(ref_tree, candidate_normalized[np.unique(candidate_valid_faces)]).max(),
    ))
    silhouette_views, silhouette_clipped = _silhouette_iou(ref_triangles, candidate_triangles, resolution)
    result = {
        "triangles_reference": int(len(ref_faces)),
        "triangles_candidate": int(len(candidate_faces)),
        "polygon_reduction_ratio": float(1.0 - len(candidate_faces) / len(ref_faces)),
        "reference_bbox_diagonal": diagonal,
        "chamfer_mean": chamfer_normalized * diagonal,
        "chamfer_mean_normalized": chamfer_normalized,
        "hausdorff_sampled": hausdorff_normalized * diagonal,
        "hausdorff_sampled_normalized": hausdorff_normalized,
        "hausdorff_with_vertices": hausdorff_vertices_normalized * diagonal,
        "hausdorff_with_vertices_normalized": hausdorff_vertices_normalized,
        "silhouette_iou_mean": float(np.mean(list(silhouette_views.values()))),
        "silhouette_iou_views": silhouette_views,
        "silhouette_frame_clipped_views": silhouette_clipped,
        "degenerate_triangles_reference": ref_degenerate,
        "degenerate_triangles_candidate": candidate_degenerate,
    }
    sampling = {
        "samples_per_mesh": sample_count,
        "seed": seed,
        "surface_distribution": "triangle_area_uniform_barycentric",
        "distance_target": "continuous_triangle_surface_bvh",
        "chamfer_definition": "mean_of_directed_unsquared_distance_means",
        "hausdorff_is_sampled_estimate": True,
        "hausdorff_with_vertices_definition": (
            "max_of_sampled_hausdorff_and_every_vertex_of_each_mesh_nondegenerate_triangles_to_other_surface"),
        "normalization": "reference_world_bbox_diagonal",
        "silhouette_resolution": resolution,
        "silhouette_views": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
        "silhouette_projection": "orthographic_reference_anchored_square_frame_2pct_margin",
        "silhouette_frame": (
            "per_view_pixel_scale_and_grid_origin_from_reference_projection_only_square_side_1.04x_larger_extent_"
            "centred_candidate_overhang_widens_canvas_by_whole_pixels_same_lattice_at_most_resolution_extra_per_axis_"
            "beyond_is_not_counted_and_view_listed_as_clipped_v0.1_used_union_bounds_frame"),
        "silhouette_coverage": "pixel_center_binary",
        "empty_silhouette_pair_iou": 1.0,
        "rng_streams": "numpy_seedsequence_spawn_children_" + "_".join(
            f"{index}_{name}" for index, name in enumerate(_RNG_CHILDREN)),
        "silhouette_multiview": multiview,
    }
    if multiview:
        views, clipped = _multiview_iou(ref_normalized, ref_valid_faces, candidate_normalized, candidate_valid_faces,
                                        silhouette_views, silhouette_clipped, resolution)
        result["silhouette_iou_multiview_mean"] = float(np.mean(list(views.values())))
        result["silhouette_iou_multiview_views"] = views
        result["silhouette_multiview_frame_clipped_views"] = clipped
        sampling["silhouette_multiview_directions"] = {
            label: (np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)).tolist()
            for label, direction in _MULTIVIEW_DIRECTIONS
        }
        sampling["silhouette_multiview_definition"] = (
            "13_unique_orthographic_directions_3_axes_4_body_diagonals_6_face_diagonals_"
            "opposite_direction_is_mirror_identical_axis_entries_reuse_axis_views"
        )
    if weights is not None:
        roi, notes = _roi_distances(weights, roi_threshold, ref_valid_faces, ref_triangles, candidate_tree,
                                    sample_count, children, diagonal, candidate_to_ref, candidate_nearest)
        result.update(roi)
        sampling.update({
            "roi_threshold": roi_threshold,
            "roi_definition": "nondegenerate_reference_triangles_with_mean_corner_weight_gte_threshold",
            "roi_samples_per_region": sample_count,
            # Unchanged v0.2.0 value; it describes the *_ref_to_candidate keys.
            "roi_distance_direction": "reference_to_candidate_only_roi_defined_on_reference",
            "roi_candidate_to_ref_assignment": (
                "global_candidate_area_samples_each_assigned_to_region_of_its_"
                "nearest_nondegenerate_reference_triangle"),
            "roi_symmetric_definition": (
                "chamfer_half_sum_of_both_directed_means_hausdorff_max_of_both_directed_maxima_none_if_either_is_none"),
            "roi_chamfer_definition": "directed_unsquared_mean_distance",
            "roi_hausdorff_is_sampled_estimate": True,
            "roi_notes": notes,
        })
    result["sampling"] = sampling
    return result
