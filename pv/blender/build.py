"""Rebuild every Blender scene of the film from data and save work/blender/pv3.blend (run inside Blender: MCP).

    import build; build.all()            # or build.all(fresh_her=True) after her figures changed
"""
import importlib

import bpy

import bl_common
import her_art
import scene_finale
import scene_rain

SAVE = bl_common.WORK / "pv3.blend"


def all(fresh_her=False):
    importlib.invalidate_caches()
    for mod in (bl_common, her_art, scene_rain, scene_finale):
        importlib.reload(mod)
    for img in [i for i in bpy.data.images if i.name.startswith(("claude_", "code_atlas"))]:
        img.reload()                                 # the art layers may have been prepared again
    C = bl_common
    for name in [s.name for s in bpy.data.scenes if s.name not in ("rain", "finale")]:
        if len(bpy.data.scenes) > 1 and name != bpy.context.window.scene.name:
            bpy.data.scenes.remove(bpy.data.scenes[name])
    if fresh_her:                                    # her figures changed on disk: drop the cached meshes
        for ob in [o for o in bpy.data.objects if o.name.startswith("her_")]:
            bpy.data.objects.remove(ob, do_unlink=True)
        for me in [m for m in bpy.data.meshes if m.name.startswith("her_")]:
            bpy.data.meshes.remove(me)
        for m in [m for m in bpy.data.materials if m.name.startswith("her_")]:
            bpy.data.materials.remove(m)
    out = [scene_rain.build(), scene_finale.build()]
    for name in [s.name for s in bpy.data.scenes if s.name not in ("rain", "finale")]:
        bpy.context.window.scene = bpy.data.scenes["finale"]
        bpy.data.scenes.remove(bpy.data.scenes[name])
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.curves, bpy.data.lights, bpy.data.cameras,
                  bpy.data.node_groups, bpy.data.images):
        for item in [b for b in block if b.users == 0]:
            block.remove(item)
    bpy.ops.wm.save_as_mainfile(filepath=str(SAVE), copy=True)
    return [(s.name, s.frame_start, s.frame_end, len(s.objects)) for s in out]
