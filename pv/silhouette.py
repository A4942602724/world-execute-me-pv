"""Silhouettes from character art: a matte, an ink fill, and a backlit rim that keeps the art's own colours.

    python pv/silhouette.py <image> [<image> ...]   -> assets/silhouettes/<stem>.npz (layers for the renderer)
                                                       and <stem>_preview.png (the layers over a test sky)

The character is never drawn by code: every pixel of her comes from the art. On a white background the matte is the
border flood fill of near-white pixels (the art's outline closes her white dress off from the page); art with a real
alpha channel is used as is.
"""
import env

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

OUT = env.ROOT / "assets" / "silhouettes"


def matte(rgba: np.ndarray, white=226) -> np.ndarray:
    """float alpha in [0, 1]."""
    if rgba[..., 3].min() < 250:                                   # the art already has transparency
        return rgba[..., 3] / 255.0
    rgb = rgba[..., :3].astype(np.int16)
    paper = (rgb.min(-1) > white) & (rgb.max(-1) - rgb.min(-1) < 24)
    labels, _ = ndimage.label(paper, structure=np.ones((3, 3)))
    border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    fg = ~np.isin(labels, border[border > 0])
    fg = ndimage.binary_opening(fg, iterations=1)
    lab, n = ndimage.label(fg)                                     # keep the figure, drop specks
    if n > 1:
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        fg = np.isin(lab, 1 + np.flatnonzero(sizes > 0.002 * fg.size))
    return np.clip(ndimage.gaussian_filter(fg.astype(np.float32), 0.8), 0, 1)


def on_green(rgba: np.ndarray) -> bool:
    """True when the art stands on a chroma-key green ground (judged by the border)."""
    b = np.concatenate([rgba[0, :, :3], rgba[-1, :, :3], rgba[:, 0, :3], rgba[:, -1, :3]]).astype(np.int16)
    g = np.median(b[:, 1] - np.maximum(b[:, 0], b[:, 2]))
    return g > 100


def chroma(rgba: np.ndarray):
    """(alpha, rgb) keyed off a green ground, a colour-difference keyer: alpha from how much greener than red and blue
    a pixel is, scaled against the ground's own greenness (gaps between hair strands come out too); then every edge
    pixel is un-mixed from the green it was blended with, F = (C - (1 - alpha) G) / alpha, so no green fringe is left."""
    rgb = rgba[..., :3].astype(np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    ground = np.median(border, 0)                                   # the green itself
    g = rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])
    g_ground = ground[1] - max(ground[0], ground[2])
    alpha = np.clip(1.04 * (g_ground - g) / g_ground - 0.02, 0, 1)
    fg = alpha > 0.5
    lab, n = ndimage.label(fg)                                     # keep the figure, drop specks
    if n > 1:
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        alpha = alpha * ndimage.binary_dilation(np.isin(lab, 1 + np.flatnonzero(sizes > 0.002 * fg.size)), iterations=2)
    a = alpha[..., None]
    fore = np.clip((rgb - (1 - a) * ground) / np.maximum(a, 0.05), 0, 255)
    # the generator also paints the green into shadows between hair strands; nothing of hers is green (her hair is
    # ~23 deg, her gold ~40 deg), so yellow-green to green hues are ground: clear them, refill colour from around
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1)
    hue = np.where(mx == rgb[..., 1], 120 + 60 * (rgb[..., 2] - rgb[..., 0]) / np.maximum(mx - mn, 1), 0.0)
    step = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))
    green = step((hue - 52) / 10) * (1 - step((hue - 175) / 10)) * step((sat - 0.18) / 0.12)
    w = alpha * (1 - green)
    fill = np.stack([ndimage.gaussian_filter(fore[..., k] * w, 3) for k in range(3)], -1) \
        / np.maximum(ndimage.gaussian_filter(w, 3), 1e-3)[..., None]
    fore = fore * (1 - green[..., None]) + fill * green[..., None]
    alpha = alpha * (1 - green)
    spill = np.maximum(fore[..., 0], fore[..., 2])
    fore[..., 1] = np.minimum(fore[..., 1], spill)                  # whatever green survives the un-mixing
    return np.clip(ndimage.gaussian_filter(alpha, 0.4), 0, 1), np.clip(fore, 0, 255).astype(np.uint8)


def layers(img: Image.Image) -> dict:
    """alpha, rim (0..1, how close to the outline), normal (outward, for directional backlight), colour (the art)."""
    rgba = np.asarray(img.convert("RGBA"))
    if on_green(rgba):
        alpha, rgb = chroma(rgba)
        rgba = np.dstack([rgb, rgba[..., 3]])
    else:
        alpha = matte(rgba)
    inside = alpha > 0.5
    depth = ndimage.distance_transform_edt(inside)                 # pixels from the outline, inside her
    rim = np.exp(-depth / 3.0) * inside
    soft = ndimage.gaussian_filter(alpha, 4)
    gy, gx = np.gradient(soft)
    norm = np.hypot(gx, gy) + 1e-6
    normal = np.stack([-gx / norm, -gy / norm], -1)                # points out of the figure
    return dict(alpha=alpha.astype(np.float16), rim=rim.astype(np.float16), normal=normal.astype(np.float16),
                colour=rgba[..., :3])


def shade(L: dict, light=(-0.6, -0.8), ink=(14, 12, 16), rim_gain=1.6, detail=0.07) -> np.ndarray:
    """RGBA float image of the silhouette lit from behind; light is the 2D direction the light comes from."""
    a = L["alpha"].astype(np.float32)[..., None]
    col = L["colour"].astype(np.float32) / 255
    lx, ly = light
    facing = np.clip(0.25 + 0.75 * (L["normal"][..., 0] * lx + L["normal"][..., 1] * ly), 0, 1)
    rim = (L["rim"].astype(np.float32) * facing)[..., None]
    body = np.array(ink, np.float32) / 255 + detail * col          # a whisper of the art inside the ink
    rgb = body * (1 - rim) + np.clip(col * rim_gain, 0, 1) * rim
    return np.concatenate([rgb, a], -1)


def test_sky(h, w):
    y = np.linspace(0, 1, h)[:, None, None]
    top, mid, low = np.array([0.10, 0.10, 0.13]), np.array([0.85, 0.47, 0.30]), np.array([0.95, 0.80, 0.62])
    sky = np.where(y < 0.62, top + (mid - top) * (y / 0.62) ** 1.6, mid + (low - mid) * ((y - 0.62) / 0.38))
    return np.broadcast_to(sky, (h, w, 3)).copy()


def main(paths):
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (p if p.is_absolute() else env.ROOT / p for p in map(Path, paths)):
        img = Image.open(p)
        L = layers(img)
        np.savez_compressed(OUT / f"{p.stem}.npz", **L)
        fig = shade(L)
        sky = test_sky(*fig.shape[:2])
        halo = ndimage.gaussian_filter(L["alpha"].astype(np.float32), 18)[..., None] * np.array([1.0, 0.62, 0.35])
        bg = np.clip(sky + 0.35 * halo, 0, 1)
        comp = bg * (1 - fig[..., 3:]) + fig[..., :3] * fig[..., 3:]
        side = np.concatenate([np.asarray(img.convert("RGB")) / 255, comp], 1)
        Image.fromarray((side * 255).astype(np.uint8)).save(OUT / f"{p.stem}_preview.png")
        print("ok", OUT / f"{p.stem}.npz", f"coverage {float(L['alpha'].astype(np.float32).mean()):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
