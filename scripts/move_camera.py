# Move the scene's camera before a render: blender -b scene.blend --python move_camera.py --python render_check.py -- ...
# MOVE="tx,ty,tz,rx,ry,rz": a translation by (tx, ty, tz) scene units along the camera's own right, up and backward
# axes, then a rotation by rx, ry, rz degrees about those axes (Euler XYZ), about the camera's own position. Unset or
# all zeros: the scene's camera. The camera's transform keyframes are removed (classroom keys the camera's location,
# which each render would evaluate again); its lens, sensor and depth of field stay the scene's.
import math, os
import bpy, mathutils

TRANSFORM = ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale",
             "delta_location", "delta_rotation_euler", "delta_rotation_quaternion", "delta_scale")
move = [float(v) for v in os.environ.get("MOVE", "0,0,0,0,0,0").replace(",", " ").split()]
assert len(move) == 6, "MOVE needs six numbers"
if any(move):
    cam = bpy.context.scene.camera
    ad = cam.animation_data
    if ad and ad.action:
        act = ad.action
        bags = [act] if hasattr(act, "fcurves") else [cb for layer in act.layers for strip in layer.strips
                                                      for cb in getattr(strip, "channelbags", [])]
        for bag in bags:
            for fc in [f for f in bag.fcurves if f.data_path in TRANSFORM]:
                bag.fcurves.remove(fc)
    loc, rot, scale = cam.matrix_world.decompose()
    turn = mathutils.Euler([math.radians(a) for a in move[3:]], "XYZ").to_quaternion()
    cam.matrix_world = mathutils.Matrix.LocRotScale(loc + rot @ mathutils.Vector(move[:3]), rot @ turn, scale)
    print("move_camera: " + ",".join(f"{v:g}" for v in move), flush=True)
