"""The whole film on the song's timeline: the scenes, the transitions between them, and what lies over all of them —
the lyrics, the threads of attention from each sung character to her, the status line.

    python pv/render.py film [--t0 T] [--t1 T]      (and --stills / --strip, like any scene)

A scene module gives plate(t, ghost) (its frame alone), her_point(t, line, char) (where a thread lands on her, or
None) and PATH (its address in her memory, for the status line). Over the frame the lyrics take the colour of what
lies under them: ink on paper, paper on ink.
"""
import env  # noqa: F401  (cache locations)

import bisect
import math
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import kit
import lyrics
import scene_blender as blend
import scene_climax as climax
import scene_inkflow as inkflow
import scene_light as light
import scene_pond2 as pond
import sprites
import transitions as tr  # noqa: F401  (transitions between later scenes)
from kit import H, W

T0, T1 = 0.0, 76.93                      # the whole song
SCENES = ((0.0, pond), (inkflow.T0, inkflow), (light.T0, light), (blend.RAIN.T0, blend.RAIN),
          (blend.FINALE.T0, blend.FINALE))       # each lasts until the next; from the hit at 31.1 s on: Blender
CUTS = ()                                # (t0, t1, transition, A, B); each scene carries its own way in
PATHS = ((0.0, pond.PATH), (16.95, inkflow.PATH), (31.3, light.PATH), (47.6, blend.FINALE.PATH),
         (blend.FINALE.held + 0.3, "memory://us/saved"))
NO_LYRICS = dict(ink=None, code=None, accent=None, ghost=None, voice=[])   # the finale writes them in 3D

CREAM = (240, 233, 220)
GREY_DARK = (150, 144, 134)


@lru_cache(None)
def static():
    pond.static()
    inkflow.static()
    light.static()
    kit.paper()
    return True


def scene_at(t):
    return SCENES[bisect.bisect_right([s[0] for s in SCENES], t) - 1][1]


def frame(t):
    L = NO_LYRICS if t >= blend.FINALE.T0 else lyrics.render("cinema", t)
    cut = next((c for c in CUTS if c[0] <= t < c[1]), None)
    if cut:
        t0, t1, fn, A, B = cut
        img, her = fn(t, A, B, t0, t1, L["ghost"]), None
    else:
        sc = scene_at(t)
        img, her = sc.plate(t, ghost=L["ghost"]), (lambda i, k: sc.her_point(t, i, k))
    return overlay(img, t, L, her)


# ---------------------------------------------------------------- what lies over every scene
def darkness(img):
    """0 where the frame is paper-light, 1 where it is ink-dark, smooth over a few dozen pixels."""
    small = np.asarray(img.resize((W // 8, H // 8), Image.Resampling.BILINEAR), np.float32) / 255
    lum = small @ np.array([0.299, 0.587, 0.114], np.float32)
    k = np.clip((0.55 - lum) / 0.3, 0, 1)
    k = k * k * (3 - 2 * k)
    return np.asarray(Image.fromarray((k * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR), np.float32) / 255


def paint(base, dens, dark, light_col, dark_col):
    """Blend a density into base in a colour that follows the ground: light_col on paper, dark_col on ink."""
    if dens is None:
        return
    rows = np.flatnonzero(dens.max(1) > 0.003)
    if not len(rows):
        return
    cols = np.flatnonzero(dens[rows[0]:rows[-1] + 1].max(0) > 0.003)
    box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    d, k = dens[box][..., None], dark[box][..., None]
    col = np.array(light_col, np.float32) / 255 * (1 - k) + np.array(dark_col, np.float32) / 255 * k
    base[box] = base[box] * (1 - d) + col * d


def threads(t, voice, her):
    """A thread of attention from each sung character to a point on her; it sags like a thread, lands, fades."""
    if her is None:
        return None
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for s, x, y, i, k in voice:
        a = t - s
        if not 0 <= a < 1.8:
            continue
        p2 = her(i, k)
        if p2 is None:
            continue
        u = kit.ease_out(a / 0.35)
        fade = 1 - kit.since(t, s + 0.35, 1.4)
        p0 = (x, y)
        p1 = ((p0[0] + p2[0]) / 2, max(p0[1], p2[1]) + 80)
        pts = [((1 - v) ** 2 * p0[0] + 2 * (1 - v) * v * p1[0] + v * v * p2[0],
                (1 - v) ** 2 * p0[1] + 2 * (1 - v) * v * p1[1] + v * v * p2[1]) for v in np.linspace(0, u, 28)]
        d.line(pts, fill=int(210 * fade), width=2)
        if u > 0.98:                                                  # where it lands, a small spark opens
            g = kit.since(t, s + 0.35, 0.6)
            sprites.spark(d, p2[0], p2[1], 5 + 9 * g, angle=0.6 * g, value=int(240 * fade))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(0.5)), np.float32) / 255 * 0.85


def typed_path(t):
    """(address, typing): on a change the old suffix is deleted and the new one typed, 0.3 s each, like a shell."""
    k = bisect.bisect_right([p[0] for p in PATHS], t + 0.30) - 1
    tc, new = PATHS[k]
    old = PATHS[k - 1][1] if k > 0 else new
    common = os.path.commonprefix([old, new])
    if t < tc:
        return old[:len(common) + int(round((len(old) - len(common)) * (tc - t) / 0.30))], True
    n = len(common) + int(round((len(new) - len(common)) * min(1.0, (t - tc) / 0.30)))
    return new[:n], n < len(new)


@lru_cache(None)
def _hud_font():
    return ImageFont.truetype(kit.F_MONO, 18)


@lru_cache(None)
def _starts():
    return tuple(sorted(c["s"] for ln in kit.lines() for c in ln["chars"]))


def hud(img, t):
    d = ImageDraw.Draw(img)
    path, typing = typed_path(t)
    d.text((40, 32), path, font=_hud_font(), fill=kit.GREY)
    if typing:
        x = 40 + _hud_font().getlength(path) + 2
        d.rectangle((x, 33, x + 9, 52), fill=kit.TERRA)
    sung = bisect.bisect_right(_starts(), t)
    label = f"tokens {sung * 3:04d}  ·  temperature 0.7"
    d.text((W - 40, 32), label, font=_hud_font(), fill=kit.GREY, anchor="ra")
    tick = kit.ease_out(kit.since(t, _starts()[sung - 1], 0.25)) if sung else 0.0   # one eighth turn per syllable
    sprites.spark(d, W - 40 - _hud_font().getlength(label) - 18, 42, 8, (sung - 1 + tick) * math.pi / 4, kit.TERRA)


def overlay(img, t, L, her):
    base = np.asarray(img, np.float32) / 255
    dark = darkness(img)
    paint(base, threads(t, L["voice"], her), dark, kit.TERRA, kit.TERRA)
    paint(base, L["code"], dark, kit.GREY, GREY_DARK)
    paint(base, L["ink"], dark, kit.INK, CREAM)
    paint(base, L["accent"], dark, kit.TERRA, kit.TERRA)
    out = Image.fromarray((np.clip(base, 0, 1) * 255 + 0.5).astype(np.uint8))
    sprites.draw(out, t)
    hud(out, t)
    return out
