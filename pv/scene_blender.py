"""The Blender scenes in the film: rain (S1 - S4, from the hit at 31.1 s) and finale (S5 - S11, from 46.2 s).

Blender renders scene-linear EXR frames (work/blender/frames/<scene>/f_####.exr, frame = song time * 24); here they
get what a camera and a colourist would add: bloom from the real highlights, the AgX tone curve (the same look as
Blender's view transform), a whisper of grain, and the white of the drop. The lyric overlay, HUD and sprite are
painted over by film.py as for every scene; from the drop on the lyrics live inside the 3D world instead.
"""
import env  # noqa: F401

import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from PIL import Image

import kit
from kit import H, W

FRAMES = env.ROOT / "work" / "blender" / "frames"

AGX = np.array([[0.842479062253094, 0.0423282422610123, 0.0423756549057051],
                [0.0784335999999992, 0.878468636469772, 0.0784336],
                [0.0792237451477643, 0.0791661274605434, 0.879142973793104]], np.float32)
AGX_INV = np.array([[1.19687900512017, -0.0528968517574562, -0.0529716355144438],
                    [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
                    [-0.0990297440797205, -0.0989611768448433, 1.15107367264116]], np.float32)


def agx(rgb):
    """Scene-linear Rec.709 -> display values, Troy Sobotka's AgX (polynomial fit of the default contrast)."""
    x = np.clip(rgb @ AGX.T, 1e-10, None)
    lo, hi = -12.47393, 4.026069
    x = (np.clip(np.log2(x), lo, hi) - lo) / (hi - lo)
    x2 = x * x
    x4 = x2 * x2
    y = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232
    return np.clip(y @ AGX_INV.T, 0, 1)


def _blur(img, sigma):
    """Gaussian blur of (H, W, 3); wide blurs run at a fraction of the size (same look, far faster)."""
    from scipy import ndimage
    small = max(1, int(sigma // 4))
    if small == 1:
        return ndimage.gaussian_filter(img, (sigma, sigma, 0))
    h, w = img.shape[:2]
    out = []
    for c in range(3):
        sm = np.asarray(Image.fromarray(np.ascontiguousarray(img[..., c])).resize((w // small, h // small),
                                                                                Image.Resampling.BILINEAR))
        sm = ndimage.gaussian_filter(sm, sigma / small)
        out.append(np.asarray(Image.fromarray(sm).resize((w, h), Image.Resampling.BILINEAR)))
    return np.stack(out, -1)


def bloom(lin, gain=0.10):
    """Light that is brighter than white spills: soft-knee threshold at 1, three widths."""
    lum = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    k = np.clip((lum - 0.8) / 1.2, 0, 1)[..., None]
    hot = (lin * k).astype(np.float32)
    return lin + gain * (0.5 * _blur(hot, 6) + 0.3 * _blur(hot, 18) + 0.2 * _blur(hot, 48))


@lru_cache(None)
def _grain():
    rng = np.random.default_rng(3)
    return [rng.normal(0, 1, (H, W)).astype(np.float32) for _ in range(8)]


def load(scene, t):
    """The frame of `scene` nearest to song time t, scene-linear RGB (H, W, 3); black if not rendered (yet)."""
    import OpenEXR
    f = int(round(t * kit.FPS))
    path = FRAMES / scene / f"f_{f:04d}.exr"
    if not path.exists():
        return np.zeros((H, W, 3), np.float32)
    with OpenEXR.File(str(path)) as ex:
        px = ex.channels()["RGBA"].pixels
    rgb = px[..., :3].astype(np.float32)
    if rgb.shape[:2] != (H, W):
        rgb = np.stack([np.asarray(Image.fromarray(rgb[..., c]).resize((W, H), Image.Resampling.BILINEAR))
                        for c in range(3)], -1)
    return rgb


def grade(lin, t, flash=0.0, exposure=0.0, glow=0.10):
    lin = lin * (2.0 ** exposure)
    lin = bloom(lin, glow)
    lin = lin + flash                                    # the white of the drop, in light, before the curve
    out = agx(lin)
    g = _grain()[int(t * kit.FPS) % 8]
    out = np.clip(out + (0.010 * g)[..., None], 0, 1)
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8))


class Rain:
    """S1 - S4 (from the hit at 31.1 s): lyric lines 6 - 10 keep the cinema captions over it."""
    PATH = "runtime://present/rain"

    def __init__(self):
        import scene_light
        self.T0 = scene_light.times().hit
        self.T1 = kit.lines()[10]["start"]                 # L11 starts the finale

    def static(self):
        return True

    def plate(self, t, ghost=None):
        img = grade(load("rain", t), t)
        if ghost is not None and ghost.any():           # the huge pale held character, as in every scene
            a = np.asarray(img, np.float32) / 255
            a = np.clip(a + ghost[..., None] * np.array([0.94, 0.88, 0.80], np.float32) * 0.22, 0, 1)
            img = Image.fromarray((a * 255 + 0.5).astype(np.uint8))
        return img

    def her_point(self, t, i, k):
        return None


class Finale:
    """S5 - S11 (46.2 s - end): the lyrics are 3D type inside the frame, so no captions are painted over it.
    Two flashes of white: the drop, and (softer) her picture breaking into shards on the first syllable of L16.
    From the moment the shards lock into her art the bloom is held down, so her line work stays as drawn."""
    PATH = "runtime://present/now"

    def __init__(self):
        L = {ln["i"]: ln for ln in kit.lines()}
        self.T0 = L[11]["start"]
        self.drop = L[11]["chars"][2]["s"]
        self.shatter = L[16]["chars"][0]["s"]
        self.lock = L[16]["chars"][6]["s"]
        self.held = kit.note_ends()[(16, len(L[16]["chars"]) - 1)]

    def static(self):
        return True

    def plate(self, t, ghost=None):
        a = t - self.drop
        flash = 0.0 if a < -0.04 else 2.6 * math.exp(-max(a, 0.0) / 0.16) if a < 1.2 else 0.0
        b = t - self.shatter
        flash += 0.0 if b < -0.04 or b > 0.6 else 0.9 * math.exp(-max(b, 0.0) / 0.06)   # a glint, not a white-out
        glow = 0.10 - 0.075 * kit.smooth((t - self.lock + 0.3) / 0.8)
        img = grade(load("finale", t), t, flash=flash, glow=glow)
        if t > self.held + 0.3:                          # the end: fade to black after she has gone (on the display
            k = 1.0 - kit.smooth((t - self.held - 0.3) / 1.1)          # values: dimming the light would tint it)
            img = Image.fromarray((np.asarray(img, np.float32) * k + 0.5).astype(np.uint8))
        return img

    def her_point(self, t, i, k):
        return None


RAIN, FINALE = Rain(), Finale()
