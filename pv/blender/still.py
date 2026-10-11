"""Render test stills from a saved .blend in the background (no MCP time-out, no busy UI).

    blender -b work/blender/pv3.blend --python pv/blender/still.py -- <scene> <out_dir> <t1> [<t2> ...]
        [--pct 50] [--samples 16] [--cam x,y,z:tx,ty,tz]

Writes <out_dir>/<scene>_<t>.png, display-referred (AgX), for looking at; the film itself renders EXR.
"""
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
opts = {}
rest = []
k = 0
while k < len(argv):
    if argv[k].startswith("--"):
        opts[argv[k][2:]] = argv[k + 1]
        k += 2
    else:
        rest.append(argv[k])
        k += 1
scene, out = rest[0], rest[1]
times = [float(x) for x in rest[2:]]
sc = bpy.data.scenes[scene]
bpy.context.window_manager.windows[0].scene = sc if bpy.context.window_manager.windows else None
sc.render.resolution_percentage = int(opts.get("pct", 50))
sc.eevee.taa_render_samples = int(opts.get("samples", 16))
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_depth = "8"
sc.render.use_motion_blur = opts.get("blur", "0") == "1"
if "cam" in opts:
    a, b = opts["cam"].split(":")
    pos, tgt = Vector(map(float, a.split(","))), Vector(map(float, b.split(",")))
    cam = sc.camera
    cam.animation_data_clear()
    cam.location = pos
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (tgt - pos).to_track_quat("-Z", "Y")
    if "lens" in opts:
        cam.data.lens = float(opts["lens"])
only = [x for x in opts.get("only", "").split(",") if x]
hide = [x for x in opts.get("hide", "").split(",") if x]
for t in times:
    sc.frame_set(int(round(t * sc.render.fps)))
    for ob in sc.objects:                          # debugging: render only some objects, or hide some
        if ob.type in ("CAMERA", "LIGHT"):
            continue
        if (only and not any(ob.name.startswith(x) for x in only)) or any(ob.name.startswith(x) for x in hide):
            ob.animation_data_clear() if ob.animation_data and False else None
            ob.hide_render = True
    sc.render.filepath = f"{out}/{scene}_{t:06.2f}.png"
    bpy.ops.render.render(write_still=True, scene=sc.name)
    print("still", sc.render.filepath)
