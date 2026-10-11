"""Small decorations, kept sparse: an original pixel sprite and a spark.

The sprite is an ink drop that came alive: a terracotta droplet with two dark eyes and, on its tip, the little
sunburst flower she wears in her hair. It appears three times: watching the pond (it hops on each sung syllable of
line 2, when the rings spread), riding the scanline that turns the ink into light, and waving goodbye at the end.
The spark (eight rays, long and short) marks where a thread of attention lands on her, and ticks round in the
status line as the tokens count up. Both are drawn here; film.py decides where.
"""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

import kit
from kit import H, W

GW, GH, PX = 15, 19, 4                    # sprite grid and its pixel size: 60 x 76 px on screen
COL = {"O": (118, 54, 38), "B": (217, 119, 87), "L": (240, 170, 138), "E": (30, 24, 24), "W": (252, 246, 236),
       "F": (246, 176, 72), "C": (200, 96, 40), "K": (150, 70, 48)}
FLOWER = ("F.F.F", ".FCF.", "FCCCF", ".FCF.", "F.F.F")


@lru_cache(None)
def grid(state):
    """The sprite's pixels in a state: idle, blink, look, squash, walk1, walk2, wave."""
    g = np.full((GH, GW), ".", dtype="<U1")
    cx = GW // 2
    tip, cy, r = (8, 13.4, 4.9) if state == "squash" else (7, 13.0, 4.6)
    yy, xx = np.mgrid[0:GH, 0:GW]
    dx, dy = xx - cx, yy - cy
    body = (dx * dx + dy * dy <= r * r) | ((yy >= tip) & (yy < cy) & (np.abs(dx) <= (yy - tip + 0.5) / (cy - tip) * r))
    g[body] = "B"
    edge = body & ~(np.roll(body, 1, 0) & np.roll(body, -1, 0) & np.roll(body, 1, 1) & np.roll(body, -1, 1))
    g[edge] = "O"
    for y, x in ((10, 4), (11, 4), (9, 5)):
        if g[y, x] == "B":
            g[y, x] = "L"
    ey = 11 if state == "look" else 12
    for ex in (5, 8):
        if state == "blink":
            g[ey + 1, ex] = g[ey + 1, ex + 1] = "E"
        else:
            g[ey:ey + 2, ex:ex + 2] = "E"
            g[ey, ex] = "W"
    for fx in {"walk1": (5, 9), "walk2": (4, 8)}.get(state, (5, 8)):
        g[GH - 1, fx:fx + 2] = "K"
    top = tip - 5
    for j, row in enumerate(FLOWER):
        for i, ch in enumerate(row):
            if ch != ".":
                g[top + j, cx - 2 + i] = ch
    if state == "wave":                                                    # a little arm, raised
        g[11, 12], g[10, 13], g[9, 13] = "O", "O", "B"
    return g


@lru_cache(None)
def image(state):
    rgba = np.zeros((GH, GW, 4), np.uint8)
    for ch, c in COL.items():
        rgba[grid(state) == ch] = (*c, 255)
    return Image.fromarray(rgba, "RGBA").resize((GW * PX, GH * PX), Image.Resampling.NEAREST)


def put(img, state, x, feet, alpha=1.0):
    """The sprite with its feet centred at (x, feet)."""
    if alpha <= 0.01:
        return
    im = image(state)
    if alpha < 0.999:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    img.paste(im, (int(round(x - im.width / 2)), int(round(feet - im.height))), im)


def dissolve(img, state, x, feet, age, seed=3):
    """The sprite breaking into its own pixels, which fly apart and fade (age in seconds)."""
    g = grid(state)
    rng = np.random.default_rng(seed)
    d = ImageDraw.Draw(img)
    fade = max(0.0, 1 - age / 0.45)
    if fade <= 0:
        return
    x0, y0 = x - GW * PX / 2, feet - GH * PX
    for (gy, gx) in zip(*np.nonzero(g != ".")):
        vx, vy = rng.normal(0, 260), rng.normal(-180, 200)
        px, py = x0 + gx * PX + vx * age, y0 + gy * PX + vy * age + 400 * age * age
        c = tuple(int(v * fade) for v in COL[g[gy, gx]])
        d.rectangle((px, py, px + PX - 1, py + PX - 1), fill=c)


def spark(draw, cx, cy, r, angle=0.0, value=255, width=2):
    """Eight rays about (cx, cy): four long, four short between them."""
    for k in range(8):
        a = angle + k * math.pi / 4
        rr = r if k % 2 == 0 else 0.55 * r
        draw.line((cx + 0.25 * rr * math.cos(a), cy + 0.25 * rr * math.sin(a), cx + rr * math.cos(a), cy + rr * math.sin(a)),
                  fill=value, width=width)


# ---------------------------------------------------------------- where it appears
@lru_cache(None)
def line2_hops():
    return tuple(c["s"] for ln in kit.lines() if ln["i"] == 2 for c in ln["chars"])


def hop(t, times):
    """(state, lift in px) for hopping on each time in `times`."""
    for s in times:
        a = t - s
        if -0.07 <= a < 0:
            return "squash", 0.0
        if 0 <= a < 0.26:
            return "idle", 16 * 4 * (a / 0.26) * (1 - a / 0.26)
    return None


def blink_or(t, state):
    return "blink" if (t % 3.3) < 0.11 else state


def draw(img, t):
    """Every appearance of the sprite at time t (film.py calls this over the frame)."""
    import scene_light as light
    import scene_pond2 as pond
    if 2.4 <= t < 16.9:                                                   # 1. watching the pond from the shore
        zoom, pan = pond.camera_at(t)
        x, feet = pond.world_to_screen((0.255 * W, pond.base.SHORE + 3), zoom, pan)
        h = hop(t, line2_hops())
        state, lift = h if h else (blink_or(t, "look" if 9.0 < t < 15.6 else "idle"), 0.0)
        a = kit.smooth((t - 2.4) / 0.5) * (1 - kit.smooth((t - 16.5) / 0.35))
        put(img, state, x, feet - lift, a)
        return
    T = light.times()
    if T.scan0 <= t < T.hit + 0.5:                                        # 2. riding the scanline into the light
        p = kit.clamp01((t - T.scan0) / (T.scan1 - T.scan0))
        y = -20 + (H + 40) * light.in_out_cubic(p)
        x = 0.05 * W + 0.12 * W * p                                        # along the empty left of the frame
        feet = min(y - 2, H - 34)
        if t >= T.hit:
            dissolve(img, "look", x, feet, t - T.hit)
        elif p < 1:
            put(img, "walk1" if int(t / 0.12) % 2 else "walk2", x, feet)
        else:
            put(img, blink_or(t, "look"), x, feet)
        return
    import scene_climax as climax
    held = climax.times().held
    if held + 0.4 <= t < climax.T1 - 0.2:                                # 3. waving goodbye
        a = kit.smooth((t - held - 0.4) / 0.3) * (1 - kit.smooth((t - (climax.T1 - 0.55)) / 0.3))
        put(img, "wave" if int((t - held) / 0.22) % 2 else "idle", 0.86 * W, H - 60, a)
