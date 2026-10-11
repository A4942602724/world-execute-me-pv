"""Render a scene's frames to scene-linear EXR, in the background (never through the MCP: no time-outs).

    blender -b work/blender/pv3.blend --python pv/blender/render_frames.py -- <scene> <t0> <t1>
        [--times 31.5,40.2] [--samples 48] [--pct 100] [--force]

Frame f is song time f / 24; output work/blender/frames/<scene>/f_<f:04d>.exr. Existing frames are kept unless
--force, so an interrupted render resumes where it stopped.
"""
import math
import sys
import time
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
opts, rest, k = {}, [], 0
while k < len(argv):
    if argv[k].startswith("--"):
        if argv[k] == "--force":
            opts["force"] = "1"
            k += 1
            continue
        opts[argv[k][2:]] = argv[k + 1]
        k += 2
    else:
        rest.append(argv[k])
        k += 1
scene = rest[0]
sc = bpy.data.scenes[scene]
for win in bpy.context.window_manager.windows:     # frame_set only evaluates the scene a window shows
    win.scene = sc
fps = sc.render.fps
if "times" in opts:
    frames = sorted({int(round(float(x) * fps)) for x in opts["times"].split(",")})
else:
    t0, t1 = float(rest[1]), float(rest[2])
    frames = list(range(int(math.floor(t0 * fps)), int(math.ceil(t1 * fps)) + 1))
out = Path(bpy.path.abspath("//")).parent.parent / "work" / "blender" / "frames" / scene if False else \
    Path(__file__).resolve().parents[2] / "work" / "blender" / "frames" / scene
out.mkdir(parents=True, exist_ok=True)
sc.eevee.taa_render_samples = int(opts.get("samples", 48))
sc.render.resolution_percentage = int(opts.get("pct", 100))
im = sc.render.image_settings
im.file_format = "OPEN_EXR"
im.color_depth = "16"
im.color_mode = "RGBA"
t_start = time.time()
done = 0
for f in frames:
    path = out / f"f_{f:04d}.exr"
    if path.exists() and "force" not in opts:
        continue
    sc.frame_set(f)
    sc.render.filepath = str(path)
    sc.render.use_file_extension = False
    bpy.ops.render.render(write_still=True, scene=sc.name)
    done += 1
    el = time.time() - t_start
    print(f"frame {f} ({f / fps:.2f} s)  {done} done, {el / done:.1f} s/frame", flush=True)
print("finished", scene, done, "frames in", round(time.time() - t_start), "s")
