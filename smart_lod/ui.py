"""Blender UI and operators. SPDX-License-Identifier: GPL-3.0-or-later."""
bl_info = {
    'name': 'Perceptual Smart LOD', 'author': 'Smart LOD Project',
    'version': (0, 2, 0), 'blender': (4, 2, 0), 'category': 'Object',
    'location': 'View3D > Sidebar > Smart LOD',
    'description': 'Generate perceptually weighted LODs and compare geometry metrics',
}

import csv
import json
from pathlib import Path
import traceback

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ExportHelper

from .perception import DEFAULT_SILHOUETTE_RESOLUTION, DEFAULT_SILHOUETTE_VIEWS, DEFAULT_WEIGHTS
from .pipeline import DEFAULT_STRENGTH, METHOD_NAMES, generate


class SMARTLOD_Settings(bpy.types.PropertyGroup):
    method: EnumProperty(name='Method', items=[
        ('PERCEPTUAL', 'Perceptual QEM', 'Importance-weighted quadric error simplification'),
        ('QEM', 'QEM baseline', 'Same custom engine with perceptual weights disabled'),
        ('DECIMATE', 'Blender Decimate', 'Native unweighted Collapse modifier')])
    ratio_1: FloatProperty(name='LOD1 retain', default=0.50, min=0.001, max=0.999, subtype='FACTOR')
    ratio_2: FloatProperty(name='LOD2 retain', default=0.20, min=0.001, max=0.999, subtype='FACTOR')
    ratio_3: FloatProperty(name='LOD3 retain', default=0.05, min=0.001, max=0.999, subtype='FACTOR')
    strength: FloatProperty(name='Protection strength (λ)', default=DEFAULT_STRENGTH, min=0, max=100,
                            description='Importance multiplier λ: quadric weight = 1 + λ × importance')
    weighting: EnumProperty(name='Weighting', default='QUADRIC', items=[
        ('QUADRIC', 'Weighted quadrics', 'Scale each vertex quadric by 1 + λ × importance (default)'),
        ('ANCHOR', 'Legacy v0.1 positional anchor',
         'λ weights a displacement penalty toward original positions; shrinks convex surfaces (ablation only)')])
    saliency_weight: FloatProperty(name='Saliency', default=DEFAULT_WEIGHTS['saliency'], min=0, max=1,
                                   description='Multi-scale mesh saliency (Lee et al. 2005); geometric, not semantic')
    silhouette_weight: FloatProperty(name='Silhouette', default=DEFAULT_WEIGHTS['silhouette'], min=0, max=1,
                                     description='How often a vertex lies on the outer outline over generic views')
    sharp_weight: FloatProperty(name='Sharp edges', default=DEFAULT_WEIGHTS['sharp'], min=0, max=1,
                                description='Creases at or above the sharp angle, open and marked-sharp edges')
    visibility_weight: FloatProperty(name='Camera visibility', default=DEFAULT_WEIGHTS['visibility'], min=0, max=1,
                                     description='Frustum, front-facing and self-occlusion test from the scene camera')
    sharp_angle: FloatProperty(name='Sharp angle (degrees)', default=45, min=1, max=179)
    silhouette_views: IntProperty(name='Silhouette views', default=DEFAULT_SILHOUETTE_VIEWS, min=4, max=96,
                                  description='Orthographic hemisphere views for the silhouette importance channel')
    silhouette_resolution: IntProperty(name='Silhouette view pixels', default=DEFAULT_SILHOUETTE_RESOLUTION,
                                       min=64, max=1024,
                                       description='Raster resolution of each silhouette importance view')
    protect_group: StringProperty(name='Protection group', default='LOD_Protect',
                                  description='Paint weights on the original mesh: 1 is most important; not a hard lock')
    roi_group: StringProperty(name='Evaluation ROI group', default='LOD_EvalROI',
                              description='Evaluation only: vertex group for ROI distance metrics. '
                                          'It is never used for protection or importance')
    preserve_seams: BoolProperty(name='Lock UV / material seams', default=True,
                                 description='Custom QEM modes lock seam vertices; may stop before target count')
    use_camera: BoolProperty(name='Use scene camera visibility', default=False,
                             description='Add the camera visibility channel from the active scene camera')
    evaluate: BoolProperty(name='Calculate quality metrics', default=True)
    sample_count: IntProperty(name='Surface samples', default=4000, min=100, max=100000)
    resolution: IntProperty(name='IoU resolution', default=128, min=32, max=1024,
                            description='Pixel resolution of the evaluation silhouette views')
    seed: IntProperty(name='Sampling seed', default=42, min=0)
    show_research: BoolProperty(name='Research settings', default=False)
    show_weights: BoolProperty(name='Perceptual weights', default=False)


class SMARTLOD_OT_generate(bpy.types.Operator):
    bl_idname = 'smart_lod.generate'
    bl_label = 'Generate Optimized LOD'
    bl_description = 'Create LOD0 plus three LODs from the selected original mesh'
    bl_options = {'REGISTER', 'UNDO'}
    # SKIP_SAVE: a UI call must not inherit the last run's value (the main button leaves it unset).
    compare: BoolProperty(default=False, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        wm = context.window_manager
        wm.progress_begin(0, 100)
        try:
            report, outputs = generate(context, context.active_object, context.scene.smart_lod,
                                       self.compare, lambda i, n, m, l: wm.progress_update(100*i/n))
            stopped = sum(not row['target_reached'] for row in report['results'])
            message = f"Created {len(outputs)} objects in {report['total_wall_seconds']:.1f}s"
            if stopped:
                message += f'; {stopped} target(s) limited by geometry constraints'
            self.report({'WARNING' if stopped else 'INFO'}, message)
            return {'FINISHED'}
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            wm.progress_end()


class SMARTLOD_OT_protect(bpy.types.Operator):
    bl_idname = 'smart_lod.prepare_protection'
    bl_label = 'Paint Important Areas'
    bl_description = 'Create/select the protection vertex group and enter Weight Paint mode'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and context.active_object is not None and context.active_object.type == 'MESH' and not context.active_object.get('smart_lod_generated')

    def execute(self, context):
        obj = context.active_object
        group_name = context.scene.smart_lod.protect_group.strip() or 'LOD_Protect'
        if group_name == context.scene.smart_lod.roi_group.strip():
            self.report({'ERROR'}, 'The evaluation ROI group cannot be used for protection')
            return {'CANCELLED'}
        context.scene.smart_lod.protect_group = group_name
        group = obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)
        obj.vertex_groups.active_index = group.index
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        self.report({'INFO'}, 'Paint important areas red (1.0), then return to Object Mode')
        return {'FINISHED'}


class SMARTLOD_OT_preview(bpy.types.Operator):
    bl_idname = 'smart_lod.preview'
    bl_label = 'Preview LOD'
    bl_description = 'Show this mesh alone for the latest run in the viewport'
    bl_options = {'UNDO'}
    object_name: StringProperty(options={'SKIP_SAVE'})
    heatmap: BoolProperty(default=False, options={'SKIP_SAVE'})

    def execute(self, context):
        data = context.scene.get('smart_lod_report')
        if not data:
            return {'CANCELLED'}
        report = json.loads(data)
        obj = bpy.data.objects.get(self.object_name)
        if obj is None or obj.name not in context.view_layer.objects:
            self.report({'ERROR'}, 'Object is missing from the current view layer')
            return {'CANCELLED'}
        collection = bpy.data.collections.get(report['collection'])
        if collection:
            for child in collection.objects:
                child.hide_set(child != obj)
        source = bpy.data.objects.get(report['source'])
        if source and source.name in context.view_layer.objects:
            source.hide_set(source != obj)
        for selected in list(context.selected_objects):
            selected.select_set(False)
        obj.hide_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        if context.space_data and context.space_data.type == 'VIEW_3D':
            shading = context.space_data.shading
            shading.type = 'SOLID'
            if self.heatmap and obj.data.color_attributes.get('LOD_Heatmap'):
                obj.data.color_attributes.active_color = obj.data.color_attributes['LOD_Heatmap']
                shading.color_type = 'VERTEX'
            else:
                shading.color_type = 'MATERIAL'
        return {'FINISHED'}


class SMARTLOD_OT_export(bpy.types.Operator, ExportHelper):
    bl_idname = 'smart_lod.export_report'
    bl_label = 'Export Benchmark'
    bl_description = 'Export the latest measured results; JSON includes full reproducibility metadata'
    filename_ext = '.json'
    filter_glob: StringProperty(default='*.json;*.csv', options={'HIDDEN'})
    format: EnumProperty(name='Format', items=[('JSON', 'JSON', 'Full report with settings'), ('CSV', 'CSV', 'One row per method and LOD')])

    def check(self, context):
        ext = '.json' if self.format == 'JSON' else '.csv'
        path = str(Path(self.filepath).with_suffix(ext))
        changed = path != self.filepath
        self.filepath = path
        return changed

    @classmethod
    def poll(cls, context):
        return bool(context.scene.get('smart_lod_report'))

    def execute(self, context):
        report = json.loads(context.scene['smart_lod_report'])
        path = Path(bpy.path.abspath(self.filepath)).with_suffix('.json' if self.format == 'JSON' else '.csv')
        try:
            if self.format == 'JSON':
                path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            else:
                rows = []
                for result in report['results']:
                    row = dict(run_id=report['run_id'], source_sha256=report['source_sha256'],
                               blender_version=report['blender_version'], settings_json=json.dumps(report['settings']))
                    row.update({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in result.items()})
                    rows.append(row)
                fields = list(dict.fromkeys(key for row in rows for key in row))
                with path.open('w', newline='', encoding='utf-8-sig') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            self.report({'INFO'}, f'Saved {path.name}')
            return {'FINISHED'}
        except OSError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class SMARTLOD_PT_main(bpy.types.Panel):
    bl_label = 'Perceptual Smart LOD'
    bl_idname = 'SMARTLOD_PT_main'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Smart LOD'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.smart_lod
        obj = context.active_object
        source_ok = obj is not None and obj.type == 'MESH' and not obj.get('smart_lod_generated')
        box = layout.box()
        box.label(text='01  SOURCE', icon='MESH_DATA')
        box.label(text=obj.name if source_ok else 'Select the original mesh')
        if source_ok:
            # Every polygon has >= 3 corners, so its triangles are its corners - 2. Budgets apply
            # to the evaluated mesh (LOD0 count in the results), which modifiers can change.
            mesh = obj.data
            modifiers = ' before modifiers' if any(m.show_viewport for m in obj.modifiers) else ''
            box.label(text=f'{len(mesh.loops) - 2 * len(mesh.polygons):,} base triangles{modifiers}')
        box.label(text='Static snapshot at current frame', icon='INFO')
        box = layout.box()
        box.label(text='02  LOD BUDGETS', icon='MOD_DECIM')
        box.prop(settings, 'method', text='')
        for name in ('ratio_1', 'ratio_2', 'ratio_3'):
            box.prop(settings, name, slider=True)
        box = layout.box()
        box.label(text='03  IMPORTANT AREAS', icon='GROUP_VERTEX')
        box.prop(settings, 'strength')
        box.prop(settings, 'protect_group')
        box.operator('smart_lod.prepare_protection', icon='BRUSH_DATA')
        if context.mode != 'OBJECT':
            box.label(text='Return to Object Mode to generate', icon='INFO')
        box.prop(settings, 'preserve_seams')
        box.prop(settings, 'use_camera')
        if settings.use_camera and not context.scene.camera:
            box.label(text='No scene camera: channel disabled', icon='INFO')
        elif settings.use_camera and context.scene.camera.type != 'CAMERA':
            box.label(text='Scene camera is not a Camera object', icon='ERROR')
        box.prop(settings, 'show_weights', icon='TRIA_DOWN' if settings.show_weights else 'TRIA_RIGHT', emboss=False)
        if settings.show_weights:
            for name in ('saliency_weight', 'silhouette_weight', 'sharp_weight', 'visibility_weight',
                         'sharp_angle', 'silhouette_views', 'silhouette_resolution'):
                box.prop(settings, name)
        column = layout.column()
        column.enabled = source_ok and context.mode == 'OBJECT'
        column.scale_y = 1.5
        column.operator('smart_lod.generate', icon='MOD_DECIM')
        layout.prop(settings, 'show_research', icon='TRIA_DOWN' if settings.show_research else 'TRIA_RIGHT', emboss=False)
        if settings.show_research:
            box = layout.box()
            box.prop(settings, 'weighting')
            box.prop(settings, 'evaluate')
            if settings.evaluate:
                box.prop(settings, 'sample_count')
                box.prop(settings, 'resolution')
                box.prop(settings, 'seed')
                box.prop(settings, 'roi_group')
                if source_ok and settings.roi_group.strip() and not obj.vertex_groups.get(settings.roi_group.strip()):
                    box.label(text='No ROI group on source: ROI metrics off', icon='INFO')
            col = box.column()
            col.enabled = source_ok and context.mode == 'OBJECT'
            col.operator('smart_lod.generate', text='Compare All 3 Methods', icon='EXPERIMENTAL').compare = True
            box.label(text='Metrics are sampled estimates', icon='INFO')
        encoded = context.scene.get('smart_lod_report')
        if encoded:
            report = json.loads(encoded)
            box = layout.box()
            box.label(text='04  LATEST RESULTS', icon='OUTLINER_COLLECTION')
            row = box.row(align=True)
            row.operator('smart_lod.preview', text='Source').object_name = report['source']
            heat = row.operator('smart_lod.preview', text='Importance', icon='SHADING_RENDERED')
            heat.object_name = report['reference_object']
            heat.heatmap = True
            row = box.row()
            row.label(text=f"LOD0: {report['source_triangles']:,} triangles")
            row.operator('smart_lod.preview', text='', icon='HIDE_OFF').object_name = report['reference_object']
            for result in report['results']:
                col = box.column(align=True)
                row = col.row(align=True)
                row.label(text=f"{METHOD_NAMES[result['method']]} / L{result['lod']}")
                row.operator('smart_lod.preview', text='', icon='HIDE_OFF').object_name = result['object']
                col.label(text=f"{result['actual_triangles']:,} tris  |  {result['processing_seconds']:.2f}s")
                if 'silhouette_iou_mean' in result:
                    text = f"IoU axis {result['silhouette_iou_mean']:.3f}"
                    if 'silhouette_iou_multiview_mean' in result:
                        text += f"  |  13-view {result['silhouette_iou_multiview_mean']:.3f}"
                    col.label(text=text)
                    col.label(text=f"CD {result['chamfer_mean_normalized']:.5f}  |  "
                                   f"HD {result['hausdorff_sampled_normalized']:.4f}")
                if not result['target_reached']:
                    col.label(text='Target limited by constraints', icon='INFO')
                col.separator(factor=0.4)
            box.operator('smart_lod.export_report', icon='EXPORT')


CLASSES = (SMARTLOD_Settings, SMARTLOD_OT_generate, SMARTLOD_OT_protect,
           SMARTLOD_OT_preview, SMARTLOD_OT_export, SMARTLOD_PT_main)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.smart_lod = PointerProperty(type=SMARTLOD_Settings)


def unregister():
    del bpy.types.Scene.smart_lod
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()
