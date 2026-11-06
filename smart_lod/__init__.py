"""Perceptual Smart LOD. SPDX-License-Identifier: GPL-3.0-or-later.

Lazy UI import also permits importing the NumPy-only QEM core outside Blender.
"""
bl_info = {
    'name': 'Perceptual Smart LOD', 'author': 'Smart LOD Project',
    'version': (0, 2, 0), 'blender': (4, 2, 0), 'category': 'Object',
    'location': 'View3D > Sidebar > Smart LOD',
    'description': 'Generate perceptually weighted LODs and compare geometry metrics',
}


def register():
    from . import ui
    ui.register()


def unregister():
    from . import ui
    ui.unregister()
