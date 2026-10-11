"""Song data for the Blender scenes: the Blender side has no librosa or kit, so everything it needs is written once
to work/blender/timing.json (and the dot matrix the scanline leaves at 31.1 s to work/blender/dots.npz).

    python pv/blender_data.py
"""
import env

import json

import numpy as np

import kit

OUT = env.ROOT / "work" / "blender"


def timing():
    ne = kit.note_ends()
    lines = []
    for ln in kit.lines():
        chars, k = [], 0
        breaks = set()
        for ch in ln["text"]:
            if ch.isspace():
                breaks.add(k - 1)
            else:
                k += 1
        glyphs = [ch for ch in ln["text"] if not ch.isspace()]
        for k, c in enumerate(ln["chars"]):
            chars.append(dict(c=glyphs[k], s=round(c["s"], 3), e=round(ne[(ln["i"], k)], 3), brk=k in breaks))
        lines.append(dict(i=ln["i"], start=chars[0]["s"], end=chars[-1]["e"], chars=chars))
    a = kit.analysis()
    hits = [round(t, 3) for t, s in kit.hits("kick") if s > 0.6]
    return dict(fps=kit.FPS, lines=lines, downbeats=[round(x, 3) for x in a["downbeats"]], kick=hits,
                sections=a["sections"], duration=a["duration"])


def dots():
    """The LED dot matrix of scene_light at the hit, in Blender coordinates (x, light3d z -> y, light3d y -> z)."""
    import scene_light as SL
    S, T = SL.static(), SL.times()
    xyz = S.dot_xyz
    return dict(xyz=np.stack([xyz[:, 0], xyz[:, 2], xyz[:, 1]], 1).astype(np.float32), rgb=S.dot_rgb.astype(np.float32),
                uv=S.dot_uv.astype(np.float32), cam=np.array([SL.CAM0[0], SL.CAM0[2], SL.CAM0[1]], np.float32),
                plane=np.float32(SL.PLANE), hit=np.float32(T.hit), fov=np.float32(38.0))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    data = timing()
    (OUT / "timing.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / "dots.npz", **dots())
    print("ok", OUT / "timing.json", len(data["lines"]), "lines;", OUT / "dots.npz")
