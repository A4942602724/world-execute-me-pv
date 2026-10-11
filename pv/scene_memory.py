"""Scene: the log of memories (lyric lines 4 - 5, about 16.6 - 31 s).

We are inside her now, in her memory of the two of us, kept as a log. The cards stand in a column and one is open
at a time: on each phrase of the voice the next card rises into focus and its picture develops from code into ink.
A rail on the left marks every card with its date. Nothing here moves on the drums.
"""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import kit
import vignette as vg
from kit import H, W

T0, T1 = 16.0, 31.2
PATH = "memory://childhood/log"
CX, CY, SPACING = 0.335 * W, 0.50 * H, 455
RAIL_X = CX - vg.CW / 2 - 70
PAPER_CLEAR = (250, 247, 240, 0)


def out_quint(x):
    return 1 - (1 - kit.clamp01(x)) ** 5


@lru_cache(None)
def focus_times():
    """When each card opens: the start of each phrase of lines 4 and 5, then the held last word of line 5."""
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
    out.append(L[5]["chars"][-1]["s"])
    return tuple(sorted(out)[: len(vg.CARDS)])


def focus(t):
    """Which card is open, as a float: it steps up on each phrase, easing over 0.9 s."""
    return sum(out_quint((t - (tk - 0.35)) / 0.9) for tk in focus_times()[1:])


@lru_cache(None)
def static():
    for j in range(len(vg.CARDS)):
        vg.picture(j), vg.ascii_picture(j), vg.develop_field(j), vg.caption(j)
    vg.card_mask()
    return True


def placement(j, f, t):
    """(centre x, centre y, scale, alpha, blur) of card j while card f (a float) is open."""
    a = abs(j - f)
    alpha = 1 - 0.62 * a if a <= 1 else max(0.0, 0.38 * (1.7 - a) / 0.7)
    return CX, CY + (j - f) * SPACING + 3 * math.sin(0.7 * t + 1.3 * j), 1 - 0.16 * min(1.0, a), alpha, 2.2 * min(1.0, a)


def develop(j, t):
    """0 (still code), 1 (ink) or the mask of the ink developing through the code."""
    p = kit.since(t, focus_times()[j] - 0.25, 1.1)
    if p <= 0 or p >= 1:
        return float(p >= 1)
    return np.clip((p * 1.25 - vg.develop_field(j)) / 0.12, 0, 1)


def put(img, card, x, y, s, alpha, blur):
    """Paste an RGBA card centred at (x, y), scaled by s, softened by blur, faded by alpha."""
    w, h = card.size
    ow, oh = int(w * s) + 12, int(h * s) + 12
    X0, Y0 = int(x - ow / 2), int(y - oh / 2)
    a = 1 / s
    im = card.transform((ow, oh), Image.Transform.AFFINE, (a, 0, (X0 - x) * a + w / 2, 0, a, (Y0 - y) * a + h / 2),
                        resample=Image.Resampling.BICUBIC, fillcolor=PAPER_CLEAR)
    if blur > 0.3:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    if alpha < 0.999:
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    img.paste(im, (X0, Y0), im)


@lru_cache(None)
def _rail_font():
    return ImageFont.truetype(kit.F_MONO, 13)


def rail(img, t, f):
    d = ImageDraw.Draw(img)
    d.line((RAIL_X, 90, RAIL_X, H - 90), fill=(214, 207, 195), width=1)
    for j, (_, _, cap) in enumerate(vg.CARDS):
        _, y, _, alpha, _ = placement(j, f, t)
        if not 90 < y < H - 90:
            continue
        is_open = abs(j - f) < 0.5
        r = 5 if is_open else 3
        d.ellipse((RAIL_X - r, y - r, RAIL_X + r, y + r), fill=kit.TERRA if is_open else (176, 169, 158))
        a = max(alpha, 0.35)
        col = tuple(int(p + (g - p) * a) for p, g in zip(kit.PAPER, kit.GREY))
        d.text((RAIL_X - 16, y), cap.split(" · ")[1], font=_rail_font(), fill=col, anchor="rm")


def plate(t, ghost=None):
    """The log alone: no lyrics, threads or status line."""
    img = kit.paper().copy()
    if ghost is not None and ghost.any():
        img = kit.multiply(img, kit.to_ink(ghost, kit.INK))
    f = focus(t)
    rail(img, t, f)
    for j in sorted(range(len(vg.CARDS)), key=lambda j: -abs(j - f)):   # the far ones first
        x, y, s, alpha, blur = placement(j, f, t)
        if alpha <= 0.01:
            continue
        is_open = abs(j - f) < 0.5
        put(img, vg.card(j, develop(j, t), t if is_open else None, is_open), x, y, s, alpha, blur)
    return img


def her_point(t, i, k):
    """Where sung character k of line i reaches her: her hair, in the open card."""
    f = focus(t)
    j = int(round(f))
    x, y, s, alpha, _ = placement(j, f, t)
    pts = vg.picture(j)[2]
    if not len(pts) or alpha < 0.5:
        return None
    px, py = pts[(i * 31 + k * 17) % len(pts)]
    return x + (vg.PAD + px - vg.CW / 2) * s, y + (vg.PAD + py - vg.CH / 2) * s
