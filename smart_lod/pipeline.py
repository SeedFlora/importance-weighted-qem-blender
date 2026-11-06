"""Generation, benchmarks and provenance shared by the UI and batch scripts."""
import hashlib
import json
import math
import platform
import re
import time
import uuid
from datetime import datetime, timezone

import bpy
import numpy as np
from mathutils import Matrix

from .geometry import snapshot, world_arrays, group_weights, feature_constraints, result_mesh, add_importance
from .perception import compute_importance
from .qem import WEIGHTING_MODES, simplify_levels
from .metrics import evaluate_meshes

EXTENSION_VERSION = '0.2.0'
REPORT_SCHEMA_VERSION = 2
# Importance multiplier lambda (quadric weight = 1 + lambda * importance). The UI
# property default reads this constant. 12 was selected on the tuning split with the
# pre-registered rule in results/tuning/PROTOCOL.md (Amandemen 1).
DEFAULT_STRENGTH = 12.0
METHOD_NAMES = {'PERCEPTUAL': 'Perceptual QEM', 'QEM': 'QEM', 'DECIMATE': 'Blender Decimate'}
METHOD_TAGS = {'PERCEPTUAL': 'PA', 'QEM': 'QEM', 'DECIMATE': 'DEC'}
ROI_THRESHOLD = 0.5
SETTING_NAMES = (
    'method', 'ratio_1', 'ratio_2', 'ratio_3', 'strength', 'weighting',
    'saliency_weight', 'silhouette_weight', 'sharp_weight', 'visibility_weight', 'sharp_angle',
    'silhouette_views', 'silhouette_resolution', 'protect_group', 'roi_group',
    'preserve_seams', 'use_camera', 'sample_count', 'resolution', 'seed', 'evaluate')
# Share of a custom method's progress span spent in its single simplification
# pass; the rest is split between the three levels' mesh building and metrics.
_PASS_SHARE = 0.5


def settings_dict(settings):
    return {name: getattr(settings, name) for name in SETTING_NAMES}


def camera_parameters(camera_object, scene):
    """World-space camera dict for ``perception.camera_visibility``.

    Follows Blender's view-plane rules: AUTO sensor fit picks the larger of
    ``resolution_x * pixel_aspect_x`` and ``resolution_y * pixel_aspect_y``; the
    sensor size is ``sensor_height`` only for VERTICAL fit. Object scale is
    ignored, as in rendering. Lens shift is replaced by the smallest symmetric
    frustum that contains it; panoramic cameras are treated as perspective.
    Raises ValueError for a non-camera object (any object can be scene.camera).
    """
    if camera_object.type != 'CAMERA':
        raise ValueError(f'The scene camera "{camera_object.name}" ({camera_object.type.lower()}) is not a Camera '
                         'object; make a Camera the scene camera or turn off "Use scene camera visibility"')
    data = camera_object.data
    matrix = np.asarray(camera_object.matrix_world, dtype=np.float64)
    forward = -matrix[:3, 2] / np.linalg.norm(matrix[:3, 2])
    up = matrix[:3, 1] - (matrix[:3, 1] @ forward) * forward
    up /= np.linalg.norm(up)
    render = scene.render
    size_x = render.resolution_x * render.pixel_aspect_x
    size_y = render.resolution_y * render.pixel_aspect_y
    fit = data.sensor_fit
    if fit == 'AUTO':
        fit = 'HORIZONTAL' if size_x >= size_y else 'VERTICAL'
    # Half extents of the view plane in units of the fitted sensor dimension.
    if fit == 'HORIZONTAL':
        half_x, half_y = 0.5, 0.5 * size_y / size_x
    else:
        half_x, half_y = 0.5 * size_x / size_y, 0.5
    half_x += abs(data.shift_x)
    half_y += abs(data.shift_y)
    camera = {'position': matrix[:3, 3].tolist(), 'forward': forward.tolist(), 'up': up.tolist(),
              'clip_start': float(data.clip_start), 'clip_end': float(data.clip_end)}
    if data.type == 'ORTHO':
        camera.update(type='ORTHO', ortho_width=2.0 * half_x * data.ortho_scale,
                      ortho_height=2.0 * half_y * data.ortho_scale)
    else:
        sensor = data.sensor_height if data.sensor_fit == 'VERTICAL' else data.sensor_width
        camera.update(type='PERSP', angle_x=2.0 * math.atan(half_x * sensor / data.lens),
                      angle_y=2.0 * math.atan(half_y * sensor / data.lens))
    return camera


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(mean=0.0, nonzero_fraction=0.0, above_half_fraction=0.0)
    return dict(mean=float(values.mean()), nonzero_fraction=float(np.count_nonzero(values > 0) / len(values)),
                above_half_fraction=float(np.count_nonzero(values >= 0.5) / len(values)))


def _object(name, mesh, source, collection, run_id, method, level):
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.matrix_world = Matrix.Identity(4)
    obj['smart_lod_generated'] = True
    obj['smart_lod_run'] = run_id
    obj['smart_lod_source'] = source.name
    obj['smart_lod_method'] = method
    obj['smart_lod_level'] = level
    obj.hide_render = True
    obj.hide_set(True)
    return obj


def _decimate(mesh, source, collection, ratio, context):
    temporary = bpy.data.objects.new('_SmartLOD_temporary', mesh.copy())
    collection.objects.link(temporary)
    temporary.matrix_world = Matrix.Identity(4)
    temporary_mesh = temporary.data
    try:
        temporary.modifiers.new('Triangulate input', 'TRIANGULATE')
        modifier = temporary.modifiers.new('Unweighted Decimate baseline', 'DECIMATE')
        modifier.decimate_type = 'COLLAPSE'
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        depsgraph = context.evaluated_depsgraph_get()
        evaluated = temporary.evaluated_get(depsgraph)
        return bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
    finally:
        bpy.data.objects.remove(temporary, do_unlink=True)
        if temporary_mesh.users == 0:
            bpy.data.meshes.remove(temporary_mesh)


def generate(context, source, settings, compare=False, progress=None):
    """Create LOD0 plus three LODs per method, measure them and store a report.

    ``progress(done, total, method_name, level)`` is called before each
    Decimate level, at the start of each custom pass, during it (``done`` is
    then fractional and ``level`` is the LOD the pass is working towards,
    estimated from the source triangle count) and before each custom level's
    mesh building. ``done`` never decreases and ends below ``total`` (rows).
    An exception raised by ``progress`` aborts the run and rolls it back.
    """
    if source is None or source.type != 'MESH':
        raise ValueError('Select a mesh object first')
    if source.get('smart_lod_generated'):
        raise ValueError('Select the original mesh, not a generated LOD')
    ratios = [settings.ratio_1, settings.ratio_2, settings.ratio_3]
    if not 1 > ratios[0] > ratios[1] > ratios[2] > 0:
        raise ValueError('Ratios must decrease: 1 > LOD1 > LOD2 > LOD3 > 0')
    weighting = str(settings.weighting).lower()
    if weighting not in WEIGHTING_MODES:
        raise ValueError('Weighting must be QUADRIC or ANCHOR')
    protect_name, roi_name = settings.protect_group.strip(), settings.roi_group.strip()
    if protect_name and protect_name == roi_name:
        raise ValueError('The evaluation ROI group must not be the protection group')
    camera_object = context.scene.camera if settings.use_camera else None
    camera = camera_parameters(camera_object, context.scene) if camera_object else None
    methods = ['DECIMATE', 'QEM', 'PERCEPTUAL'] if compare else [settings.method]
    total = len(methods) * 3
    source_mesh = None
    collection = None
    created_meshes = []
    start = time.perf_counter()

    def report_progress(done, method, level):
        if progress:
            progress(done, total, METHOD_NAMES[method], level)

    try:
        source_mesh, vertices, faces = snapshot(source, context.evaluated_depsgraph_get())
        source_seconds = time.perf_counter() - start
        # Seam/material locks (and marked-sharp edges) are a QEM/PERCEPTUAL prerequisite.
        constraint_start = time.perf_counter()
        locked, marked_sharp = feature_constraints(source_mesh, settings.preserve_seams)
        constraint_seconds = time.perf_counter() - constraint_start
        roi = None
        if roi_name and source.vertex_groups.get(roi_name) is not None:
            roi = np.clip(group_weights(source_mesh, source, roi_name), 0.0, 1.0)
        importance_start = time.perf_counter()
        manual = group_weights(source_mesh, source, protect_name)
        channel_seconds = {}
        importance, features = compute_importance(
            vertices, faces,
            weights=dict(saliency=settings.saliency_weight, silhouette=settings.silhouette_weight,
                         sharp=settings.sharp_weight, visibility=settings.visibility_weight),
            manual=manual, camera=camera, sharp_angle=settings.sharp_angle, marked_sharp=marked_sharp,
            silhouette_views=settings.silhouette_views, silhouette_resolution=settings.silhouette_resolution,
            timings=channel_seconds)
        importance_seconds = time.perf_counter() - importance_start
        run_id = uuid.uuid4().hex[:12]
        base = re.sub(r'(?i)(?:_highpoly|_high_poly|_high|_lod\d+)$', '', source.name)
        collection = bpy.data.collections.new(base + '_SmartLOD_' + run_id[:6])
        context.scene.collection.children.link(collection)
        reference_mesh = source_mesh.copy()
        created_meshes.append(reference_mesh)
        reference = _object(base + '_LOD0', reference_mesh, source, collection, run_id, 'REFERENCE', 0)
        add_importance(reference, importance)
        rows = []
        outputs = [reference]
        targets = [max(4, int(round(len(faces) * ratio))) for ratio in ratios]
        method_pass_seconds = {}

        def deliver(method, level, output_mesh, output_vertices, output_faces, stats, seconds, mesh_seconds,
                    output_importance=None):
            name = f'{base}_{METHOD_TAGS[method]}_LOD{level}' if compare else f'{base}_LOD{level}'
            output = _object(name, output_mesh, source, collection, run_id, method, level)
            # Only PERCEPTUAL carries an importance map; QEM results are unweighted.
            if output_importance is not None and method == 'PERCEPTUAL':
                add_importance(output, output_importance)
            outputs.append(output)
            target, perceptual = targets[level - 1], method == 'PERCEPTUAL'
            constraints = constraint_seconds if method != 'DECIMATE' else 0.0
            row = dict(object=output.name, source=source.name, method=method,
                       lod=level, requested_ratio=float(ratios[level - 1]), target_triangles=target,
                       actual_triangles=len(output_faces), source_triangles=len(faces),
                       target_reached=len(output_faces) <= target,
                       reduction_ratio=1-len(output_faces)/len(faces),
                       simplification_and_mesh_seconds=seconds,
                       mesh_reconstruction_seconds=mesh_seconds,
                       constraint_preparation_seconds=constraints,
                       importance_seconds=importance_seconds if perceptual else 0.0,
                       processing_seconds=seconds + constraints + (importance_seconds if perceptual else 0.0),
                       qem_stats=stats)
            if settings.evaluate:
                evaluation_start = time.perf_counter()
                row.update(evaluate_meshes(vertices, faces, output_vertices, output_faces,
                                           sample_count=settings.sample_count, resolution=settings.resolution,
                                           seed=settings.seed, roi_weights=roi, roi_threshold=ROI_THRESHOLD))
                row['evaluation_seconds'] = time.perf_counter() - evaluation_start
            output['smart_lod_target'] = target
            output['smart_lod_triangles'] = len(output_faces)
            rows.append(row)

        for method in methods:
            base_done = len(rows)
            if method == 'DECIMATE':
                method_pass_seconds[method] = 0.0
                for level, target in enumerate(targets, 1):
                    report_progress(base_done + level - 1, method, level)
                    step_start = time.perf_counter()
                    output_mesh = _decimate(source_mesh, source, collection, target/len(faces), context)
                    created_meshes.append(output_mesh)
                    output_vertices, output_faces = world_arrays(output_mesh, Matrix.Identity(4))
                    seconds = time.perf_counter() - step_start
                    method_pass_seconds[method] += seconds
                    # Modifier evaluation and mesh conversion are not separable here.
                    deliver(method, level, output_mesh, output_vertices, output_faces, {}, seconds, None)
                continue
            perceptual = method == 'PERCEPTUAL'
            report_progress(base_done, method, 1)

            def on_pass(fraction, base_done=base_done, method=method):
                remaining = len(faces) - fraction * (len(faces) - min(targets))
                level = next((level for level, target in enumerate(targets, 1) if remaining > target), 3)
                report_progress(base_done + 3 * _PASS_SHARE * fraction, method, level)

            pass_start = time.perf_counter()
            results = simplify_levels(vertices, faces, targets,
                                      importance=importance if perceptual else None,
                                      strength=settings.strength if perceptual else 0.0,
                                      locked=locked, weighting=weighting, progress=on_pass)
            method_pass_seconds[method] = time.perf_counter() - pass_start + constraint_seconds
            for level, result in enumerate(results, 1):
                report_progress(base_done + 3 * _PASS_SHARE + (1 - _PASS_SHARE) * (level - 1), method, level)
                mesh_start = time.perf_counter()
                output_mesh = result_mesh(base + '_mesh', result, source_mesh, vertices, faces, Matrix.Identity(4))
                created_meshes.append(output_mesh)
                mesh_seconds = time.perf_counter() - mesh_start
                method_pass_seconds[method] += mesh_seconds
                deliver(method, level, output_mesh, result.vertices, result.faces, result.stats,
                        result.stats['seconds'] + mesh_seconds, mesh_seconds, result.importance)
        summary = {name: _summary(values) for name, values in features.items()}
        summary['combined'] = _summary(importance)
        report = dict(schema_version=REPORT_SCHEMA_VERSION, extension_version=EXTENSION_VERSION, run_id=run_id,
                      created_at=datetime.now(timezone.utc).isoformat(),
                      blender_version=bpy.app.version_string, platform=platform.platform(),
                      source=source.name, reference_object=reference.name, collection=collection.name,
                      source_triangles=len(faces), source_vertices=len(vertices),
                      source_sha256=hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest(),
                      source_matrix=[list(row) for row in source.matrix_world], frame=context.scene.frame_current,
                      settings=settings_dict(settings), methods=methods, weighting=weighting,
                      camera=camera_object.name if camera_object else None,
                      camera_matrix=[list(row) for row in camera_object.matrix_world] if camera_object else None,
                      camera_parameters=camera,
                      source_preparation_seconds=source_seconds,
                      constraint_preparation_seconds=constraint_seconds,
                      importance_preparation_seconds=importance_seconds,
                      importance_channel_seconds=channel_seconds,
                      importance_summary=summary,
                      method_pass_seconds=method_pass_seconds,
                      seam_locked_vertices=int(np.count_nonzero(locked)),
                      manually_protected_vertices=int(np.count_nonzero(manual)),
                      roi_group=roi_name if roi is not None else None,
                      roi_vertices=int(np.count_nonzero(roi > 0)) if roi is not None else None,
                      roi_threshold=ROI_THRESHOLD if roi is not None else None,
                      total_wall_seconds=time.perf_counter()-start, results=rows,
                      notes=[
                          'Static snapshot at current frame with the world transform baked in (mirroring transforms '
                          'keep outward winding); rigs, shape keys and animations are not copied.',
                          'QEM output transfers UVs and face materials; other custom attributes are not transferred.',
                          'PERCEPTUAL weighting "quadric" (default) scales each vertex quadric by 1 + strength * '
                          'importance after the plane and boundary quadrics are summed; it cannot protect exactly '
                          'planar regions. "anchor" is the v0.1 positional penalty, kept for ablation; it pulls '
                          'collapsed vertices inside convex surfaces. QEM rows use the same engine unweighted.',
                          'Importance is geometric, not semantic: saliency = multi-scale mesh saliency after Lee, '
                          'Varshney & Jacobs 2005 (heat-diffusion smoothing, noise gate); silhouette = how often a '
                          'vertex lies on or near the outer outline over generic Fibonacci-hemisphere orthographic '
                          'views (independent of the evaluation views); sharp = creases at or above the sharp angle, '
                          'open/nonmanifold and marked-sharp edges, spread one ring; visibility (optional) = camera '
                          'frustum, front-facing (outward normals) and self-occlusion from the near clip plane, '
                          'ignoring other scene objects. Manual protection weights combine by maximum.',
                          'QEM and PERCEPTUAL run one greedy pass per method down to the smallest budget and snapshot '
                          'each LOD; each snapshot equals an independent run to that budget. '
                          'simplification_and_mesh_seconds = the cumulative pass time until that snapshot (the cost of '
                          'an independent run to that budget) plus its mesh reconstruction; for DECIMATE it is one '
                          'independent modifier run plus mesh conversion. constraint_preparation_seconds is the '
                          'seam/material lock extraction (with marked-sharp edges) that QEM and PERCEPTUAL need and '
                          'DECIMATE does not (0 on DECIMATE rows); importance_seconds (importance_preparation_seconds, '
                          'including reading the protection group) is needed by PERCEPTUAL only. processing_seconds = '
                          'simplification_and_mesh_seconds + constraint_preparation_seconds + importance_seconds. '
                          'method_pass_seconds is the wall time of the whole pass plus reconstruction of all levels '
                          'plus constraint_preparation_seconds, without importance (DECIMATE: sum of its three runs). '
                          'The shared snapshot (source_preparation_seconds) and evaluation time are excluded from all '
                          'of these.',
                          'hausdorff_sampled is the maximum over area-weighted surface samples; '
                          'hausdorff_with_vertices also measures every vertex of each mesh to the other surface. Both '
                          'are estimates, not the exact surface maximum. Chamfer is unsquared point-to-surface.',
                          'silhouette_iou_mean uses six fixed orthographic world-axis views; '
                          'silhouette_iou_multiview_* uses 13 unique orthographic directions (3 axes, 4 body '
                          'diagonals, 6 face diagonals). Both are pixel-centre coverage, not perspective renders.',
                          'ROI metrics use the evaluation-only ROI vertex group (never used for protection; present '
                          'only when the source has it): roi_* covers the reference triangles whose mean corner weight '
                          'is >= roi_threshold, nonroi_* the rest. Per region, *_ref_to_candidate measures reference '
                          'samples in the region to the candidate, *_candidate_to_ref the candidate samples whose '
                          'nearest reference face lies in the region, and *_symmetric combines both (chamfer: mean of '
                          'the two means; hausdorff: maximum).',
                          'Safety/seam constraints can prevent targets. Compare actual counts, not only requested ratios.',
                          'Blender Decimate also uses quadrics internally; these are implementation baselines.'
                      ])
        encoded = json.dumps(report, indent=2)
        text = bpy.data.texts.new('SmartLOD_Report_' + run_id + '.json')
        text.write(encoded)
        context.scene['smart_lod_report'] = encoded
        return report, outputs
    except Exception:
        if collection:
            for obj in list(collection.objects):
                mesh = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            bpy.data.collections.remove(collection)
        for mesh in created_meshes:
            try:
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            except ReferenceError:  # already removed with its object above
                pass
        raise
    finally:
        if source_mesh and source_mesh.users == 0:
            bpy.data.meshes.remove(source_mesh)
