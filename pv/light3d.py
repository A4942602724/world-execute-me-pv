"""Light in the dark: a small 3D point renderer for the second half of the film (torch; GPU when it has one).

Everything there is points of light. A Camera projects them in perspective; each point's blur is its circle of
confusion (thin lens) or its own size, whichever is larger, and points are splatted into a few blur levels that are
blurred once each, so out-of-focus light opens into soft discs. Fog dims the far points, bloom lets the bright ones
glow, and a wet floor mirrors whatever stands on it. Points come as (xyz, rgb, size) tensors; the image is light
added onto a warm charcoal ground.
"""
import env  # noqa: F401

import math
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

import kit
from kit import H, W

DEV = os.environ.get("PV_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
LEVELS = (0.0, 1.5, 3.5, 7.0, 14.0, 28.0, 52.0)          # blur radius (px) of each splat level
CREAM = (0.96, 0.91, 0.84)
ORANGE = (0.98, 0.55, 0.24)
TERRA = (0.85, 0.47, 0.34)


def t32(a):
    return torch.as_tensor(np.asarray(a, np.float32), device=DEV)


def handheld(t, amp=0.03):
    """A slow, organic sway for a camera position: three incommensurate sines per axis, none faster than 0.4 Hz,
    so the frame always breathes a little and never follows the beat."""
    waves = ((0.11, 0.7), (0.19, 2.3), (0.33, 4.1))
    axis = lambda k: sum(math.sin(2 * math.pi * f * (1 + 0.13 * k) * t + p * (1 + k)) for f, p in waves) / 3
    return np.array([axis(0), 0.6 * axis(1), axis(2)], np.float32) * amp


class Camera:
    def __init__(self, pos, target, fov=38.0, focus=None, aperture=0.0):
        self.pos, tgt = t32(pos), t32(target)
        fwd = tgt - self.pos
        self.fwd = fwd / fwd.norm()
        right = torch.linalg.cross(t32((0, 1, 0)), self.fwd)            # +x on screen is to the right
        self.right = right / right.norm()
        self.up = torch.linalg.cross(self.fwd, self.right)
        self.f = (H / 2) / math.tan(math.radians(fov) / 2)
        self.focus = float((tgt - self.pos).norm()) if focus is None else focus
        self.aperture = aperture

    def project(self, xyz):
        rel = xyz - self.pos
        x, y, z = rel @ self.right, rel @ self.up, rel @ self.fwd
        z = z.clamp(min=1e-3)
        return W / 2 + self.f * x / z, H / 2 - self.f * y / z, z

    def coc(self, z):
        """Blur radius in pixels of a point at depth z (thin lens)."""
        return self.aperture * self.f * (1 / self.focus - 1 / z).abs()

    def unproject(self, u, v, depth):
        """World points at screen (u, v) and distance `depth` along the view direction."""
        x, y = (u - W / 2) * depth / self.f, -(v - H / 2) * depth / self.f
        return self.pos + self.right * x[:, None] + self.up * y[:, None] + self.fwd * depth


def _gauss(sigma):
    r = int(math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, device=DEV, dtype=torch.float32)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


def blur(img, sigma):
    """Gaussian blur of (C, H, W); large blurs run at reduced size."""
    if sigma < 0.3:
        return img
    c, h, w = img.shape
    s = 1
    while sigma / s > 4 and s < 16:
        s *= 2
    x = F.avg_pool2d(img[None], s) if s > 1 else img[None]
    k = _gauss(sigma / s)
    x = F.conv2d(F.pad(x, (len(k) // 2, len(k) // 2, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, len(k) // 2, len(k) // 2), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    if s > 1:
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x[0]


@lru_cache(None)
def ground():
    """The warm charcoal ground with a vignette and a little grain, (3, H, W)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    top, bottom = np.array((0.090, 0.082, 0.078), np.float32), np.array((0.046, 0.042, 0.040), np.float32)
    g = top[:, None, None] + (bottom - top)[:, None, None] * (yy / H)[None]
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    g = g * (1 - 0.35 * np.clip(r - 0.4, 0, None) ** 1.4)[None]
    grain = ndimage.gaussian_filter(np.random.default_rng(5).standard_normal((H, W)), 0.7)
    return t32(np.clip(g + 0.006 * grain[None], 0, 1))


SHARP = 3                                                 # levels below this are kept at full size, the rest at half


class Frame:
    """Accumulates points for one frame, then resolves them into an image."""

    def __init__(self, cam, fog=0.045):
        self.cam, self.fog = cam, fog
        self.acc = torch.zeros((3, SHARP * H * W), device=DEV)                    # sharp levels, side by side
        self.soft = torch.zeros((3, (len(LEVELS) - SHARP) * (H // 2) * (W // 2)), device=DEV)   # blurry ones, half size

    def add(self, xyz, rgb, size=0.0, gain=1.0):
        """xyz (N, 3), rgb (N, 3) or (3,), size: a minimum blur radius in px (scalar or (N,))."""
        if xyz is None or len(xyz) == 0:
            return
        u, v, z = self.cam.project(xyz)
        rgb = torch.as_tensor(rgb, device=DEV, dtype=torch.float32).expand(len(xyz), 3) * gain
        rgb = rgb * torch.exp(-self.fog * z)[:, None]
        r = torch.maximum(self.cam.coc(z), torch.as_tensor(size, device=DEV, dtype=torch.float32))
        lv = torch.bucketize(r, t32(LEVELS[1:]) - 0.01)
        rgb = rgb * (1 + 0.10 * t32(LEVELS)[lv])[:, None]          # bokeh keeps some of its brightness
        keep = (u > -60) & (u < W + 60) & (v > -60) & (v < H + 60) & (z > 0.05)
        u, v, rgb, lv = u[keep], v[keep], rgb[keep], lv[keep]
        sharp = lv < SHARP
        self._splat(self.acc, u[sharp], v[sharp], rgb[sharp], lv[sharp], W, H)
        # half size: each half-size pixel later spreads over four (resolve divides by 4 to keep the light the same)
        self._splat(self.soft, u[~sharp] / 2, v[~sharp] / 2, rgb[~sharp], lv[~sharp] - SHARP, W // 2, H // 2)

    @staticmethod
    def _splat(acc, u, v, rgb, lv, w_, h_):
        """Bilinear splat: four pixels share each point."""
        x0, y0 = torch.floor(u), torch.floor(v)
        fx, fy = u - x0, v - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = (x0 + dx).long(), (y0 + dy).long()
            ok = (xi >= 0) & (xi < w_) & (yi >= 0) & (yi < h_)
            acc.index_add_(1, lv[ok] * (h_ * w_) + yi[ok] * w_ + xi[ok], (rgb[ok] * w[ok][:, None]).T.contiguous())

    def resolve(self, exposure=1.6, bloom=0.55):
        img = torch.zeros((3, H, W), device=DEV)
        sharp = self.acc.view(3, SHARP, H, W)
        for k in range(SHARP):
            if torch.any(sharp[:, k] > 0):
                img += blur(sharp[:, k], LEVELS[k] / 2)
        soft = self.soft.view(3, len(LEVELS) - SHARP, H // 2, W // 2)
        half = torch.zeros((3, H // 2, W // 2), device=DEV)
        for k in range(len(LEVELS) - SHARP):
            if torch.any(soft[:, k] > 0):
                half += blur(soft[:, k], LEVELS[SHARP + k] / 4)
        img += F.interpolate(half[None], size=(H, W), mode="bilinear", align_corners=False)[0] / 4
        bright = (img - 0.35).clamp(min=0)
        img = img + bloom * (0.6 * blur(bright, 6) + 0.4 * blur(bright, 24))
        light = 1 - torch.exp(-img * exposure)
        g = ground()
        return g + light * (1 - g)


def to_image(rgb):
    """(3, H, W) float tensor -> PIL RGB."""
    from PIL import Image
    a = (rgb.clamp(0, 1) * 255 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


@lru_cache(None)
def art_colours(name, n, height=1.7, depth=0.30, seed=0):
    """The art's own colour (0..1) under each point of figure_points(name, n, ...): for when she is seen in colour."""
    L = np.load(kit.ASSETS / "silhouettes" / f"{name}.npz")
    a = L["alpha"].astype(np.float32)
    rng = np.random.default_rng(seed)                                       # the same draws as figure_points
    ys, xs = np.nonzero(a > 0.5)
    pick = rng.choice(len(xs), size=n, p=a[ys, xs] / a[ys, xs].sum())
    return (L["colour"][ys[pick], xs[pick]].astype(np.float32) / 255)


@lru_cache(None)
def figure_points(name, n, height=1.7, depth=0.30, seed=0):
    """Her as a cloud of n points of light (numpy): xyz with her feet on y = 0 and her centre at x = 0, given depth
    by inflating the silhouette (thickest at its middle), cream where the art is not orange, orange where it is,
    a little brighter where the art is lighter, and brightest along her outline so the silhouette reads.
    Returns (xyz, rgb, phase): phase in [0, 2 pi) lets each point twinkle on its own."""
    L = np.load(kit.ASSETS / "silhouettes" / f"{name}.npz")
    a = L["alpha"].astype(np.float32)
    lum = L["colour"].astype(np.float32).mean(-1) / 255
    _, hair = kit.figure(name, a.shape[0])
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(a > 0.5)
    pick = rng.choice(len(xs), size=n, p=a[ys, xs] / a[ys, xs].sum())
    py, px = ys[pick] + rng.uniform(-0.5, 0.5, n), xs[pick] + rng.uniform(-0.5, 0.5, n)
    rows = np.flatnonzero(a.max(1) > 0.5)
    top, bottom = rows[0], rows[-1]
    s = height / (bottom - top)
    d = ndimage.distance_transform_edt(a > 0.5)
    thick = depth * np.sqrt(d[ys[pick], xs[pick]] / d.max())
    z = thick * np.where(rng.random(n) < 0.8, np.sign(rng.uniform(-1, 1, n)) * rng.uniform(0.75, 1.0, n), rng.uniform(-1, 1, n))
    xyz = np.stack([(px - xs.mean()) * s, (bottom - py) * s, z], 1).astype(np.float32)
    is_hair = hair[ys[pick], xs[pick]] > 0.4
    rim = np.exp(-d[ys[pick], xs[pick]] / (0.012 * (bottom - top)))
    bright = ((0.5 + 0.35 * lum[ys[pick], xs[pick]]) * (0.75 + 0.9 * rim))[:, None]
    rgb = np.where(is_hair[:, None], np.array(ORANGE, np.float32), np.array(CREAM, np.float32)) * bright
    return xyz, rgb.astype(np.float32), rng.uniform(0, 2 * np.pi, n).astype(np.float32)
