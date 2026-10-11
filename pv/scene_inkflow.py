"""Scene: she grows up in ink (lyric lines 4 - 5, 16.55 - 31.2 s), the pond melting into it.

A stable-fluids simulation (fluid.py), baked once. The pond painting liquefies at the end of line 3 and, on the drum
at 17 s, bursts from her into swirls of ink and orange. Then the ink flows backwards into her as a child (line 4,
first phrase), dissolves and flows back into her at fourteen (second phrase), dissolves again, and on the last words
of line 5 flows slowly back into her as she is now, landing on the held last note. A shape forming is the same
simulation as that shape dissolving, played backwards.

Where the ink is dense, code scrolls inside it, as inside her silhouette in the pond; where it is thin and moving,
the flow is drawn in code: one glyph per cell, chosen by the direction the water moves.
"""
import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

import fluid
import kit
import lyrics
import scene_pond2 as pond
from kit import H, W

T0, T1 = 16.55, 30.7
PATH = "memory://childhood/ink"
CX, FEET = 0.33 * W, 0.90 * H
FIGS = {"child": ("child_solo", 600), "adult": ("character_claude", 680)}
FLOW = "-\\|/"                              # glyphs by flow direction


def out_cubic(x):
    return 1 - (1 - kit.clamp01(x)) ** 3


@lru_cache(None)
def phrases():
    """Start of each phrase of lines 4 and 5, and of line 5's held last word."""
    L = {ln["i"]: ln for ln in kit.lines()}
    out = []
    for i in (4, 5):
        ln, k = L[i], 0
        out.append(ln["chars"][0]["s"])
        for ch in ln["text"]:
            if ch.isspace():
                out.append(ln["chars"][k]["s"])
            else:
                k += 1
    return sorted(out), L[5]["chars"][-1]["s"]


@lru_cache(None)
def schedule():
    """(name, start, end, mode) for every sequence: mode 'forward' dissolves, 'form' plays it backwards, 'hold' rests
    near the formed shape. Times follow the voice: the forms land on phrases, the last on line 5's held note."""
    (p4a, p4b, p5a, p5b), held = phrases()
    return (("melt", T0, T0 + 2.4, "forward"),
            ("child", p4a + 0.85, p4a + 2.55, "form"), ("child", p4a + 2.55, p5a - 0.15, "hold"),
            ("child", p5a - 0.15, p5a + 2.1, "forward"),
            ("adult", p5a + 1.3, held, "form"), ("adult", held, T1, "hold"))


def melt_dye():
    """The pond as it is at T0, turned into ink and orange dye."""
    img = np.asarray(pond.plate(T0, ghost=lyrics.render("cinema", T0)["ghost"]), np.float32) / 255
    paper = np.array(kit.PAPER, np.float32) / 255
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    k = np.clip((paper @ np.array([0.299, 0.587, 0.114]) - lum) / 0.8 * 1.15 - 0.04, 0, 1)
    o = np.clip(((img[..., 0] - img[..., 2]) - (paper[0] - paper[2])) * 2.2, 0, 1)
    k = k * (1 - o)
    return np.stack([k, o])[:, ::2, ::2].astype(np.float32)


def bake_all():
    S = SimpleNamespace(seq={}, dye0={})
    burst_at = (CX / 4, (FEET - 0.6 * 640) / 4)
    for seed, (name, (pose, height)) in enumerate(FIGS.items(), 11):
        dye0 = fluid.figure_dye(pose, height, CX, FEET)
        S.dye0[name] = dye0
        stir = fluid.Stir(seed, waves=18, wavelength=(25, 110))
        V = fluid.vortices(fluid.edge_points(dye0, 46, 3), 95, 9, 5)
        B = fluid.burst(burst_at, 1.0, 40)

        def forces(t, vel, stir=stir, V=V, B=B):
            if t < 0.12:
                return (V + 60 * B) / 0.12
            return 1.2 * (stir(t, 18.0) - vel)
        S.seq[name] = fluid.bake(name, dye0, 3.6 if name == "adult" else 3.0, forces, eps=6.0,
                                 fade=lambda t: 0.9985, diffuse=0.06)

    dye0 = melt_dye()
    S.dye0["melt"] = dye0
    stir = fluid.Stir(77, waves=20, wavelength=(30, 140))
    V = fluid.vortices(fluid.edge_points(dye0, 120, 9), 80, 10, 9)
    B = fluid.burst((pond.chest()[0] / 4, pond.chest()[1] / 4), 1.0, 70)
    hit = next(tk for tk, _ in kit.hits("kick") if tk > T0 + 0.25) - T0     # the drum at ~17 s
    sink = fluid.torch.zeros((1, 2, fluid.VH, fluid.VW), device=fluid.DEV)
    sink[:, 1] = 10.0                                                    # ink sinks a little

    def forces(t, vel):
        target = stir(t, 6.0 if t < hit else 22.0) + sink
        a = 1.2 * (target - vel)
        if hit <= t < hit + 0.12:
            a = a + (V + 140 * B) / 0.12
        return a
    S.seq["melt"] = fluid.bake("melt", dye0, 2.4, forces, eps=6.0,
                               fade=lambda t: 1.0 if t < hit else 0.996, diffuse=0.06)
    S.hair_pts = {n: np.argwhere(d[1] > 0.5)[:, ::-1].astype(np.float32) * 2 for n, d in S.dye0.items()}
    return S


@lru_cache(None)
def static():
    S = bake_all()
    S.text = kit.text_texture("ink_text", W, 2 * H, size=13, line=16, seed=21)
    S.flow = {c: kit.mono_glyph(c, 15) for c in FLOW}
    return S


def frames_of(t):
    """[(name, frame index (float), weight)] of the sequences showing at t, and the name of the formed figure."""
    out, formed = [], None
    for name, a, b, mode in schedule():
        if not a <= t < b:
            continue
        p = (t - a) / (b - a)
        n = len(static().seq[name][0]) - 1
        if mode == "forward":
            start = 4 if name != "melt" else 0
            idx = (t - a) * kit.FPS if name == "melt" else start + (n - start) * p ** 1.3
            w = 1 - kit.smooth((t - (b - 0.9)) / 0.9) if name != "adult" else 1.0
        elif mode == "form":
            idx = min(n, 2.2 * kit.FPS) * (1 - out_cubic(p))
            w = kit.smooth((t - a) / 0.9)
            if p > 0.85:
                formed = name
        else:                                                          # resting: the edges breathe out and back
            idx, w, formed = 0.6 - 0.6 * math.cos(2 * math.pi * (t - a) / 2.8), 1.0, name
        out.append((name, idx, w))
    return out, formed


def sample(name, idx):
    dye, vel = static().seq[name]
    i = int(idx)
    j, f = min(i + 1, len(dye) - 1), idx - int(idx)
    d = dye[i].astype(np.float32) * (1 - f) + dye[j].astype(np.float32) * f
    v = vel[i].astype(np.float32) * (1 - f) + vel[j].astype(np.float32) * f
    return d / 255, v


def mix(t):
    """The dye (2, DH, DW) and velocity (2, ., .) at t: the weighted mix of the sequences showing."""
    parts, formed = frames_of(t)
    total = sum(w for _, _, w in parts) or 1.0
    d = v = 0
    for name, idx, w in parts:
        dd, vv = sample(name, idx)
        d, v = d + dd * w / total, v + vv * w / total
    return d, v, formed


def flow_glyphs(k, vel):
    """Density of the flow drawn in code: in thin, moving ink, one glyph per 12 x 20 cell by flow direction."""
    S = static()
    cw, ch = 12, 20
    gh, gw = H // ch, W // cw
    dens = k[: gh * ch // 2, : gw * cw // 2].reshape(gh, ch // 2, gw, cw // 2).mean((1, 3))
    vx = np.asarray(Image.fromarray(vel[0]).resize((gw, gh), Image.Resampling.BILINEAR))
    vy = np.asarray(Image.fromarray(vel[1]).resize((gw, gh), Image.Resampling.BILINEAR))
    speed = np.hypot(vx, vy)
    band = np.clip((dens - 0.05) / 0.08, 0, 1) * np.clip((0.55 - dens) / 0.2, 0, 1) * np.clip((speed - 6) / 20, 0, 1)
    im = Image.new("L", (W, H), 0)
    for gy, gx in zip(*np.nonzero(band > 0.05)):
        ang = math.atan2(vy[gy, gx], vx[gy, gx]) % math.pi
        g = S.flow[FLOW[int(round(ang / (math.pi / 4))) % 4]]
        im.paste(Image.new("L", g.size, int(255 * 0.6 * band[gy, gx])), (gx * cw, gy * ch), g)
    return np.asarray(im, np.float32) / 255


def render(d, vel, t):
    """Ink and orange on paper, code scrolling in the dense ink, code tracing the flow in the thin ink."""
    S = static()
    k = np.asarray(Image.fromarray(d[0]).resize((W, H), Image.Resampling.BILINEAR), np.float32)
    o = np.asarray(Image.fromarray(d[1]).resize((W, H), Image.Resampling.BILINEAR), np.float32)
    edge = np.asarray(Image.fromarray(np.hypot(*np.gradient(ndimage.gaussian_filter(d[0], 1.0)))).resize(
        (W, H), Image.Resampling.BILINEAR), np.float32)
    kd = np.clip((1 - np.exp(-2.4 * k)) / (1 - math.exp(-2.4)) + 1.5 * edge, 0, 1) * 0.94
    od = np.clip((1 - np.exp(-2.4 * o)) / (1 - math.exp(-2.4)), 0, 1) * 0.85
    txt = S.text[(np.arange(H) + int(26 * t)) % S.text.shape[0]]
    dense = np.clip((kd - 0.70) / 0.2, 0, 1)
    kd = kd * (1 - 0.42 * txt * dense)
    od = od * (1 - 0.35 * txt * np.clip((od - 0.6) / 0.2, 0, 1))
    kd = np.maximum(kd, flow_glyphs(d[0], vel) * 0.8)
    grain = kit.ink_grain()
    return kit.multiply(kit.paper().copy(), kit.stack((kd * grain, kit.INK), (od * grain, kit.HAIR)))


def push(t):
    """The camera's slow push towards her over the scene."""
    return 1.0 + 0.11 * kit.smooth((t - T0) / (T1 - T0))


def plate(t, ghost=None):
    d, v, _ = mix(t)
    img = render(d, v, t)
    s = push(t)
    img = img.transform((W, H), Image.Transform.AFFINE, (1 / s, 0, CX - CX / s, 0, 1 / s, 0.6 * H - 0.6 * H / s),
                        resample=Image.Resampling.BICUBIC, fillcolor=kit.PAPER)
    if ghost is not None and ghost.any():
        img = kit.multiply(img, kit.to_ink(ghost * 0.6, kit.INK))
    if t < T0 + 0.3:                                                   # the pond hands over without a seam
        w = kit.smooth((t - T0) / 0.3)
        img = Image.blend(pond.plate(t, ghost=ghost), img, w)
    return img


def her_point(t, i, k):
    """Threads land on her hair once she has formed; while she is ink in water there is nothing to hold."""
    _, formed = frames_of(t)
    if formed is None:
        return None
    pts = static().hair_pts[formed]
    if not len(pts):
        return None
    x, y = pts[(i * 31 + k * 17) % len(pts)]
    s = push(t)
    return CX + (x - CX) * s, 0.6 * H + (y - 0.6 * H) * s
