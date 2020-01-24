#
# Render VRML CAD geometry and trajectories with Blender
#
# The current release of Blender (v2.80 July 2019) is a major upgrade over
# the previous release (v2.79 September 2017).  Unfortunately, VRML
# support is currently somewhat flaky in v2.80 (colors aren't importing).
# This code runs fine on v2.79.
#
# Blender renders the images to PNG.  At 1920x1080, the optimized PNGs
# are around 900kB.  Conversion to JPEG at quality factor 94 gave
# no noticeable JPEG artifacts when viewed at full screen:
#
# > convert design.png -quality 94 design.jpg
#

import bpy
import sys
from math import pi

rad = 1.0
deg = pi/180.0

basename = sys.argv[-1]

bpy.data.objects['Cube'].select = True
bpy.data.objects['Lamp'].select = True
bpy.ops.object.delete()

A = bpy.ops.import_scene.x3d(
    filepath=basename + '.wrl', axis_forward='Z', axis_up='-Y')

root_objects = []
for obj in bpy.context.scene.objects:
    if not obj.parent:
        root_objects.append(obj)

for x in root_objects:
    if x.type == 'MESH':
        x.scale *= 0.001
        bpy.ops.object.transform_apply(
            location=False, rotation=False, scale=True)

obj = bpy.context.scene.camera
# obj.data.type = 'PERSP'
# obj.data.lens = 85.0
obj.data.type = 'ORTHO'
obj.data.ortho_scale = 2.2
obj.location = [3.7, -5.0, 4.0]
obj.rotation_euler = [60*deg, 0.0, 30*deg]

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.world.horizon_color = (0.1, 0.1, 0.1)
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.resolution_percentage = 200
bpy.context.scene.cycles.samples = 1024
bpy.context.scene.cycles.film_transparent = True

# key light
bpy.ops.object.lamp_add(
    type='POINT', radius=0.4,
    view_align=False,
    location=(-0.2, -1.0, 0.2))
lamp = bpy.context.object
lamp.data.use_nodes = True
lamp.data.node_tree.nodes['Emission'].inputs['Strength'].default_value = 50.0

# fill light
bpy.ops.object.lamp_add(
    type='POINT', radius=0.4,
    view_align=False,
    location=(2.0, 0.0, 1.5))
lamp = bpy.context.object
lamp.data.use_nodes = True
lamp.data.node_tree.nodes['Emission'].inputs['Strength'].default_value = 100.0

# back light
bpy.ops.object.lamp_add(
    type='POINT', radius=1,
    view_align=False,
    location=(-0.5, 1.2, 2.0))
lamp = bpy.context.object
lamp.data.use_nodes = True
lamp.data.node_tree.nodes['Emission'].inputs['Strength'].default_value = 80.0

mat_name = 'Scintillator'
mat = bpy.data.materials.new(name=mat_name)
mat.use_nodes = True

matout = mat.node_tree.nodes.get('Material Output')
newmat = mat.node_tree.nodes.new('ShaderNodeMixShader')
mat.node_tree.links.new(matout.inputs[0], newmat.outputs[0])

matout = mat.node_tree.nodes.get('Mix Shader')
matout.inputs[0].default_value = 0.75

newmat = mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
mat.node_tree.links.new(matout.inputs[1], newmat.outputs[0])
newmat.inputs[0].default_value = (0.082, 0.722, 0.859, 1.0)

newmat = mat.node_tree.nodes.new('ShaderNodeBsdfTransparent')
mat.node_tree.links.new(matout.inputs[2], newmat.outputs[0])

for shape_index in [11, 12, 13, 14]:
    A = bpy.data.objects['Shape_IndexedFaceSet.{:03d}'.format(shape_index)]
    A.data.materials[0] = mat

bpy.context.scene.render.image_settings.file_format='PNG'
bpy.context.scene.render.filepath = basename + '.png'
bpy.ops.render.render(write_still=True)
