"""Transitions between scenes. Each renders a finished frame (PIL RGB) for a time inside its window.

  dive  The camera dives into her silhouette, faster and faster, smeared by its own speed, until we are inside the
        code she is made of; then the next scene opens out of that dark from the centre, its rim bleeding like ink,
        and keeps coming towards us until it is all there is. The camera never stops moving forward, so there is no
        cut anywhere.
"""
import math
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage

import kit
from kit import H, W

AFFINE, BICUBIC = Image.Transform.AFFINE, Image.Resampling.BICUBIC
CENTRE = (W / 2, H / 2)


def out_expo(x):
    x = kit.clamp01(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def scaled(img, s, centre=CENTRE, fill=(255, 255, 255)):
    """img scaled by s about centre (s > 1 comes closer)."""
    cx, cy = centre
    a = 1 / s
    return img.transform((W, H), AFFINE, (a, 0, cx - cx * a, 0, a, cy - cy * a), resample=BICUBIC, fillcolor=fill)


def in_out_cubic(x):
    x = kit.clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def zoom_blur(img, strength, centre=CENTRE):
    """Radial blur: the average of copies scaled up to 1 + strength about centre, enough of them to leave no steps."""
    if strength < 0.004:
        return img
    n = int(np.clip(strength * 70, 8, 28))
    acc = np.zeros((H, W, 3), np.float32)
    for i in range(n):
        acc += np.asarray(scaled(img, 1 + strength * i / (n - 1), centre), np.float32)
    return Image.fromarray((acc / n + 0.5).astype(np.uint8))


@lru_cache(None)
def grid():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    n = ndimage.gaussian_filter(np.random.default_rng(41).standard_normal((H, W)), 70)
    return xx, yy, (n / n.std()).astype(np.float32)


def opening(p, centre=CENTRE, soft=120, wobble=75):
    """Mask of an opening that grows from centre past the corners as p goes 0 -> 1, its rim bleeding like ink."""
    xx, yy, n = grid()
    r = p * (math.hypot(W, H) * 0.62 + wobble + soft)
    return np.clip((r - (np.hypot(xx - centre[0], yy - centre[1]) + wobble * n)) / soft, 0, 1)


@lru_cache(None)
def inner_code():
    """What she is made of, close up: light code on her dark, twice the frame in each direction."""
    dens = kit.text_texture("inner_code", 2 * W, 2 * H, size=26, line=34, seed=5)
    dark, light = np.array((24, 21, 21), np.float32), np.array((226, 219, 205), np.float32)
    return Image.fromarray((dark + (light - dark) * (0.55 * dens)[..., None]).astype(np.uint8))


def inner(s, drift=0.0):
    """The inner code at scale s about the frame centre, drifted up by drift pixels."""
    a = 1 / s
    return inner_code().transform((W, H), AFFINE, (a, 0, W - (W / 2) * a, 0, a, H - (H / 2) * a + drift), resample=BICUBIC)


def dive(t, A, B, t0, t1, ghost=None):
    """A: a scene with camera_at(t), chest() and plate(t, zoom, pan, ghost); B: a scene with plate(t, ghost)."""
    t_cover = t0 + 0.55 * (t1 - t0)                                 # her dark has filled the frame
    t_open = t0 + 0.60 * (t1 - t0)                                  # the next scene starts to open
    if t < t_open:
        u = kit.clamp01((t - t0) / (t_cover - t0))
        z0, p0 = A.camera_at(t)
        e = u ** 2.4
        zoom = z0 * (9.0 / z0) ** e
        q = A.chest()
        aim = (CENTRE[0] - q[0], CENTRE[1] - q[1])                  # pan that brings her chest to the centre
        pan = (p0[0] + (aim[0] - p0[0]) * e, p0[1] + (aim[1] - p0[1]) * e)
        speed = 2.4 * u ** 1.4 * math.log(9.0 / z0) / (t_cover - t0) / kit.FPS   # growth of log zoom per frame
        img = zoom_blur(A.plate(t, zoom=zoom, pan=pan, ghost=ghost), min(0.5, 3.5 * speed))
        a = kit.smooth((t - (t_cover - 0.14)) / 0.2)               # her code, close up
        if a > 0:
            img = Image.blend(img, zoom_blur(inner(1.0 + 0.5 * (1 - a)), 0.05), a)
        return img
    q = kit.clamp01((t - t_open) / (t1 - t_open))
    eo = out_expo(q)
    inside = zoom_blur(inner(1.0 + 1.2 * q, drift=-40 * q), 0.05 * (1 - q))
    dest = scaled(B.plate(t, ghost=ghost), 0.84 + 0.16 * eo, fill=kit.PAPER)
    m = opening(in_out_cubic(q))[..., None]                         # a small hole first, then it gives way
    out = np.asarray(inside, np.float32) * (1 - m) + np.asarray(dest, np.float32) * m
    return Image.fromarray((out + 0.5).astype(np.uint8))
