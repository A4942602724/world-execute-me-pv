"""Scene: the pond, ink and code (0 - 16.6 s, lyric lines 1 - 3). Same staging as scene_pond, a different material.

I am the one typing: each sung character appears in the input box at the bottom, and a thread of attention runs
from it to her; when the line is done it is sent. She is still a silhouette, but a terminal is shaped like her: code
scrolls through her, light on dark. The mountains are painted at the ridge and printed at the foot. The ripples are
rings of 0 and 1. Nothing moves on the drums; the cursor blinks on the beat while it waits, as cursors do.
"""
import bisect
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

import kit
import lyrics
import scene_pond as base
from kit import H, W

T0, T1 = base.T0, base.T1
PATH = "memory://childhood/pond"
BOX = (0.27 * W, 0.865 * H, 0.73 * W, 0.945 * H)
TYPE_SIZE = 34
LINES = (1, 2, 3)


@lru_cache(None)
def static():
    S = base.static()
    S.far2 = kit.ascii_mist("far_pond_ascii2", S.far)
    S.mid2 = kit.ascii_mist("mid_pond_ascii2", S.mid)
    S.static_far = kit.to_ink(S.far2, kit.INK_FAR)
    S.static_mid = kit.to_ink(S.mid2, kit.INK_MID)
    S.text = kit.text_texture("her_text", W, 2 * H)
    rng = np.random.default_rng(32)
    S.row_delay = np.repeat(rng.uniform(0, 0.12, H // 16 + 1), 16)[:H, None].astype(np.float32)

    S.a_ink = np.clip(S.fig_ink / 0.92, 0, 1)
    S.a_hair = np.clip(S.fig_hair, 0, 1)
    whole = np.clip(S.a_ink + S.a_hair, 0, 1)
    S.rim = np.clip(whole - ndimage.grey_erosion(whole, size=(5, 5)), 0, 1)
    S.hairness = S.a_hair / (whole + 1e-6)

    font = ImageFont.truetype(kit.F_SONG, TYPE_SIZE)
    S.font, S.hud, S.small = font, ImageFont.truetype(kit.F_MONO, 18), ImageFont.truetype(kit.F_SONG, 26)
    S.layout = {}
    ys, xs = np.nonzero(whole > 0.7)
    rng = np.random.default_rng(31)
    for ln in kit.lines():
        if ln["i"] not in LINES:
            continue
        x, k, pos = BOX[0] + 30, 0, []
        for ch in ln["text"]:
            if ch.isspace():
                x += TYPE_SIZE * 0.6
                continue
            w = font.getlength(ch)
            j = rng.integers(len(xs))
            pos.append((x, w, (float(xs[j]), float(ys[j]))))           # where the character sits, where it reaches her
            x += w
            k += 1
        S.layout[ln["i"]] = pos
    S.targets = {(ln["i"], k): (float(xs[j]), float(ys[j]))           # where each sung character reaches her
                 for ln in kit.lines() for k, j in enumerate(rng.integers(len(xs), size=len(ln["chars"])))}
    return S


def world_to_screen(p, zoom, pan):
    cx, cy = W / 2, H / 2
    return (p[0] - cx + pan[0]) * zoom + cx, (p[1] - cy + pan[1]) * zoom + cy


def ripples(S, t):
    """Rings of 0 and 1 spreading on the water, over a faint line."""
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    act = 0.0
    for e, (t0, x, y, s, v, life, rings) in enumerate(S.ripples):
        a = t - t0
        if not 0 <= a < life:
            continue
        depth = (y - base.SHORE) / (H - base.SHORE)
        asp, vv = 0.27 + 0.10 * depth, v * (0.55 + 0.45 * depth)
        act += s * (1 - a / life)
        n = 96 if v > 400 else 40 if rings > 1 else 24
        for k in range(rings):
            aa = a - 0.18 * k
            if aa <= 0:
                continue
            r = vv * aa ** 0.7
            fade = s * (1 - a / life) ** 1.6 * (1 - 0.3 * k)
            d.ellipse((x - r, y - r * asp, x + r, y + r * asp), outline=int(70 * fade), width=1)
            for i in range(n):
                th = 2 * math.pi * i / n + 0.37 * e + 0.5 * k
                gx, gy = x + r * math.cos(th), y + r * asp * math.sin(th)
                if not (0 <= gx < W and base.SHORE <= gy < H):
                    continue
                size = int(10 + 7 * (gy - base.SHORE) / (H - base.SHORE))
                g = kit.mono_glyph("01"[(i * 7 + e * 3 + k) % 2], size)
                im.paste(Image.new("L", g.size, int(255 * fade)), (int(gx - g.width / 2), int(gy - g.height / 2)), g)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(0.4)), np.float32) / 255 * 0.8, min(act, 1.0)


def printed(S, t):
    """She is printed into the page row by row, top to bottom, the way a terminal prints: 0, 1 or a row mask."""
    p = kit.since(t, S.drop_t + 0.35, 1.2)
    if p <= 0 or p >= 1:
        return float(p >= 1)
    edge = S.fig_top + (base.HER_H + 40) * p
    rows = np.arange(H, dtype=np.float32)[:, None]
    return ((rows + 600 * S.row_delay) < edge).astype(np.float32)


def her(S, t, reveal):
    """Her, as a terminal: dark where she is, light code scrolling through, a firm rim so the shape holds."""
    rows = (np.arange(H) + int(26 * t)) % S.text.shape[0]
    txt = S.text[rows]
    ink = (S.a_ink * (0.92 - 0.45 * txt) + 0.9 * S.rim * (1 - S.hairness)) * reveal
    hair = (S.a_hair * (0.85 - 0.40 * txt) + 0.85 * S.rim * S.hairness) * reveal
    return np.clip(ink, 0, 1), np.clip(hair, 0, 1)


def typed_voice(S, t):
    """(time, x, y, line, char) of the characters typed in the input box, for threads()."""
    out = []
    for ln in kit.lines():
        if ln["i"] in S.layout:
            out += [(c["s"], x + w / 2, BOX[1], ln["i"], k) for k, (c, (x, w, _)) in enumerate(zip(ln["chars"], S.layout[ln["i"]]))]
    return out


def threads(S, t, zoom, pan, voice):
    """A thread of attention from each sung character to a point on her; it lands, then fades."""
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for s, x, y, i, k in voice:
        a = t - s
        if not 0 <= a < 1.8:
            continue
        u = kit.ease_out(a / 0.35)
        fade = 1 - kit.since(t, s + 0.35, 1.4)
        p0 = (x, y)
        p2 = world_to_screen(S.targets[(i, k)], zoom, pan)
        p1 = ((p0[0] + p2[0]) / 2, max(p0[1], p2[1]) + 80)          # it sags like a thread, under the text
        pts = [((1 - v) ** 2 * p0[0] + 2 * (1 - v) * v * p1[0] + v * v * p2[0],
                (1 - v) ** 2 * p0[1] + 2 * (1 - v) * v * p1[1] + v * v * p2[1]) for v in np.linspace(0, u, 28)]
        d.line(pts, fill=int(210 * fade), width=2)
        if u > 0.98:
            r = 3 + 6 * kit.since(t, s + 0.35, 0.6)
            d.ellipse((p2[0] - r, p2[1] - r, p2[0] + r, p2[1] + r), outline=int(240 * fade), width=2)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(0.5)), np.float32) / 255


def beat_index(t):
    return bisect.bisect_right(kit.analysis()["beats"], t)


def ui(S, t):
    """The input box I type into, the cursor, the line being sent, and a quiet status line."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(BOX, radius=20, fill=(250, 248, 243, 236), outline=(200, 194, 183, 255), width=2)
    cy = (BOX[1] + BOX[3]) / 2
    lines = [ln for ln in kit.lines() if ln["i"] in S.layout]
    cur = next((ln for ln in lines if ln["chars"][0]["s"] <= t < ln["end"] + 0.25), None)
    cursor_x, typing = BOX[0] + 30, False
    if cur is None and t < lines[0]["chars"][0]["s"]:
        d.text((BOX[0] + 30, cy), "回覆她……", font=S.font, fill=(168, 162, 152, 255), anchor="lm")
    if cur is not None:
        for c, (x, w, _) in zip(cur["chars"], S.layout[cur["i"]]):
            a = kit.since(t, c["s"], 0.06)
            if a > 0:
                d.text((x, cy), c["c"], font=S.font, fill=(*kit.INK, int(255 * a)), anchor="lm")
                cursor_x, typing = x + w + 2, t < cur["end"]
    for ln in lines:                                                   # sent: it rises out of the box and fades
        u = kit.since(t, ln["end"] + 0.25, 0.45)
        if 0 < u < 1:
            for c, (x, w, _) in zip(ln["chars"], S.layout[ln["i"]]):
                d.text((x, cy - 80 * kit.ease_out(u)), c["c"], font=S.font, fill=(*kit.INK, int(255 * (1 - u))), anchor="lm")
    if typing or beat_index(t) % 2 == 0:
        d.rectangle((cursor_x, cy - 19, cursor_x + 3, cy + 19), fill=(*kit.TERRA, 255))
    return hud(S, t, layer)


def hud(S, t, layer=None):
    """A quiet status line in the corners."""
    layer = layer or Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    sung = sum(1 for ln in kit.lines() for c in ln["chars"] if c["s"] <= t)
    d.text((40, 32), "memory://childhood/pond", font=S.hud, fill=(*kit.GREY, 255))
    d.text((W - 40, 32), f"tokens {sung * 3:04d}  ·  temperature 0.7", font=S.hud, fill=(*kit.GREY, 255), anchor="ra")
    return layer


def camera_at(t):
    """(zoom, pan) of the pond's slow drift at time t."""
    e = kit.smooth(t / T1)                              # a slow crane down and across: the mountains slide behind her
    return 1.06 + 0.08 * e, (40 - 150 * e, 18 - 36 * e)   # zoomed in enough that the pan never shows an edge


def chest():
    """A point on her chest, in world coordinates: where the camera dives in."""
    S = static()
    return base.HER_X, S.fig_top + 0.36 * base.HER_H


def her_point(t, i, k, zoom=None, pan=None):
    """Where sung character k of line i reaches her, on screen."""
    z, p = camera_at(t)
    return world_to_screen(static().targets[(i, k)], z if zoom is None else zoom, p if pan is None else pan)


def plate(t, zoom=None, pan=None, ghost=None):
    """The pond alone: no lyrics, threads or status line. zoom/pan override the camera; ghost (a density, the huge
    pale character of the lyrics) is laid behind the mountains."""
    S = static()
    rev = base.reveal(S, t)
    rev_near = None if rev is None else rev[:, base.PAD:base.PAD + W]
    shown = printed(S, t)
    z, p = camera_at(t)
    zoom, pan = (z if zoom is None else zoom), (p if pan is None else pan)

    img = kit.paper().copy()
    if ghost is not None and ghost.any():                            # behind the mountains
        img = kit.multiply(img, kit.to_ink(ghost, kit.INK))
    for dens, static_img, colour, depth in ((S.far2, S.static_far, kit.INK_FAR, 0.35), (S.mid2, S.static_mid, kit.INK_MID, 0.6)):
        layer = static_img if rev is None else kit.to_ink(dens * rev, colour)
        img = kit.multiply(img, kit.camera(layer, 1 + (zoom - 1) * depth, (pan[0] * depth, pan[1] * depth), origin=(-base.PAD, 0)))

    rip, act = ripples(S, t)
    env_ink, water = base.union(rip, base.reeds(S, t), S.shore) * kit.ink_grain(), S.water
    if rev_near is not None:
        env_ink, water = env_ink * rev_near, water * rev_near
    her_ink, her_hair = her(S, t, shown)
    soft = kit.soft_grain()
    ink = base.union(env_ink, base.wobble(S.refl_ink, t, act) * shown * soft, her_ink, base.drop(S, t))
    hair = base.union(base.wobble(S.refl_hair, t, act) * shown, her_hair, base.petals(S, t))
    near = kit.stack((water, kit.INK_MID), (ink, kit.INK), (hair, kit.HAIR))
    return kit.multiply(img, kit.camera(near, zoom, pan))


def frame(t, style=None):
    """The test renders: style None: the lyrics typed into the input box; "cinema" or "flow": lyrics.py typography."""
    S = static()
    zoom, pan = camera_at(t)
    lyr = lyrics.render(style, t) if style else None
    img = plate(t, ghost=None if lyr is None else lyr["ghost"])
    voice = typed_voice(S, t) if lyr is None else lyr["voice"]
    img = kit.multiply(img, kit.to_ink(threads(S, t, zoom, pan, voice), kit.TERRA))
    if lyr is None:
        return Image.alpha_composite(img.convert("RGBA"), ui(S, t)).convert("RGB")
    img = kit.multiply(img, kit.stack((lyr["ink"], kit.INK), (lyr["code"], kit.GREY), (lyr["accent"], kit.TERRA)))
    return Image.alpha_composite(img.convert("RGBA"), hud(S, t)).convert("RGB")
