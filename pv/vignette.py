"""Memory cards: small ink pictures of the two of us as children, and the code they are decoded from.

Each card holds a 640 x 360 ink vignette: a setting drawn in ink (the pond, the way home, a night of stars, rain,
osmanthus) with the two children placed in it as silhouettes from the art (never drawn by code). Before a card is
opened it shows as code, an ASCII rendering of the same picture; when it opens, the ink develops through the code.
The open card is alive: its water ripples, its rain falls, its petals drift.
"""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

import kit

VW, VH = 640, 360
PAD, CAP = 12, 46                           # frame padding, caption strip
CW, CH = VW + 2 * PAD, VH + PAD + CAP       # the whole card
FILL = np.array((250, 247, 240), np.float32) / 255
CODE = np.array((112, 106, 98), np.float32) / 255
EDGE = (205, 198, 186)

CARDS = (                                   # (setting, children, caption)
    ("pond", "child_sit", "0001 · 2009-07-14 · pond"),
    ("path", "child_walk", "0137 · 2010-04-02 · the way home"),
    ("night", "child_sit", "0412 · 2011-08-20 · stars"),
    ("rain", "child_walk", "0698 · 2012-06-11 · rain"),
    ("osmanthus", "child_sit", "0943 · 2013-10-11 · osmanthus"),
)


def _ridge(rng, base, amp, freq):
    return base - amp * kit.fbm1d(VW, rng, freq) ** 1.4


def _wash(ridge, fall, strength):
    below = np.arange(VH, dtype=np.float32)[:, None] - ridge[None, :].astype(np.float32)
    d = np.where(below > 0, np.exp(-np.maximum(below, 0) / fall), 0) + 0.35 * np.exp(-np.abs(below) / 1.6) * (below > -1.5)
    return (strength * np.clip(d, 0, 1)).astype(np.float32)


def _streaks(rng, y0, y1, strength):
    s = ndimage.gaussian_filter(rng.standard_normal((VH, VW)), (0.6, 30))
    yy = np.arange(VH)[:, None]
    return (strength * ((yy >= y0) & (yy < y1)) * np.clip(0.5 + s / s.std(), 0, 1.5)).astype(np.float32)


def _draw(fn, blur=0.6):
    im = Image.new("L", (VW, VH), 0)
    fn(ImageDraw.Draw(im))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255


def _stroke(d, pts, w0, w1, fill=255):
    """A tapering brush stroke along a polyline."""
    left, right = [], []
    for k, (x, y) in enumerate(pts):
        x2, y2 = pts[min(k + 1, len(pts) - 1)]
        x1, y1 = pts[max(k - 1, 0)]
        nx, ny = -(y2 - y1), x2 - x1
        n = math.hypot(nx, ny) or 1
        w = (w0 + (w1 - w0) * k / (len(pts) - 1)) / 2
        left.append((x + nx / n * w, y + ny / n * w))
        right.append((x - nx / n * w, y - ny / n * w))
    d.polygon(left + right[::-1], fill=fill)


def _reeds(d, rng, x0, x1, base, n, h):
    for _ in range(n):
        x, hh, lean = rng.uniform(x0, x1), rng.uniform(0.5, 1) * h, rng.uniform(-0.25, 0.35)
        _stroke(d, [(x + hh * lean * u * u, base - hh * u) for u in np.linspace(0, 1, 7)], 2.4, 0.4, 210)


@lru_cache(None)
def kids(name, target_h):
    """(ink, hair) of the two children, cropped to their bounds, scaled so they are target_h tall."""
    a = np.load(kit.ASSETS / "silhouettes" / f"{name}.npz")["alpha"].astype(np.float32)
    rows = np.flatnonzero(a.max(1) > 0.05)
    ink, hair = kit.figure(name, int(round(target_h * a.shape[0] / (rows[-1] - rows[0] + 1))))
    whole = np.clip(ink / 0.92 + hair, 0, 1)
    ys, xs = np.nonzero(whole > 0.05)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return ink[box], hair[box]


def _place(dst, src, cx, bottom):
    h, w = src.shape
    x0, y0 = int(round(cx - w / 2)), int(round(bottom - h))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(VW, x0 + w - sx0), min(VH, y0 + h - sy0)
    region = dst[y0:y1, x0:x1]
    np.maximum(region, src[sy0:sy0 + y1 - y0, sx0:sx0 + x1 - x0], out=region)


@lru_cache(None)
def picture(idx):
    """(ink, hair, her_points) of card idx: densities of its vignette and points on her hair, for threads."""
    setting, who, _ = CARDS[idx]
    rng = np.random.default_rng(100 + idx)
    ink, hair = np.zeros((VH, VW), np.float32), np.zeros((VH, VW), np.float32)
    k_ink, k_hair = kids(who, 250 if who == "child_walk" else 215)
    if setting in ("pond", "osmanthus"):
        horizon = 0.40 * VH if setting == "pond" else 0.46 * VH
        ink += _wash(_ridge(rng, horizon, 0.16 * VH, 1 / 160), 24, 0.30)
        ink += _streaks(rng, horizon + 2, VH, 0.09)
        ink += _draw(lambda d: _reeds(d, rng, 0.02 * VW, 0.16 * VW, VH - 4, 9, 120))
        ink += _draw(lambda d: _reeds(d, rng, 0.86 * VW, 0.98 * VW, VH - 4, 7, 100))
        if setting == "osmanthus":                                   # a branch over them, in flower
            def branch(d):
                pts = [(VW + 10 - 230 * u, -10 + 120 * u ** 1.6 + 12 * math.sin(5 * u)) for u in np.linspace(0, 1, 14)]
                _stroke(d, pts, 9, 2, 230)
                for k in range(3, 13, 3):
                    x, y = pts[k]
                    _stroke(d, [(x - 40 * u, y + 34 * u ** 1.3) for u in np.linspace(0, 1, 6)], 3, 0.6, 210)
            ink += _draw(branch)
            hair += _draw(lambda d: [d.ellipse((x - 2.6, y - 2.6, x + 2.6, y + 2.6), fill=230)
                                     for x, y in zip(rng.normal(VW - 120, 60, 40), rng.normal(70, 28, 40))], 0.4)
        _place(ink, k_ink, 0.5 * VW, VH - 4)
        _place(hair, k_hair, 0.5 * VW, VH - 4)
    elif setting == "path":
        horizon = 0.46 * VH
        ink += _wash(_ridge(rng, horizon, 0.14 * VH, 1 / 140), 22, 0.26)
        yy = np.arange(VH, dtype=np.float32)[:, None]
        ink += 0.05 * np.clip((yy - horizon) / (VH - horizon), 0, 1) ** 1.5
        def path(d):
            for x_bottom, x_far in ((0.22 * VW, 0.485 * VW), (0.78 * VW, 0.515 * VW)):
                _stroke(d, [(x_bottom + (x_far - x_bottom) * u, VH - (VH - horizon) * u) for u in np.linspace(0, 1, 8)], 3, 0.5, 200)
            _reeds(d, rng, 0.0, 0.2 * VW, VH - 2, 10, 110)
            _reeds(d, rng, 0.8 * VW, VW, VH - 2, 10, 110)
        ink += _draw(path)
        _place(ink, k_ink, 0.5 * VW, VH - 8)
        _place(hair, k_hair, 0.5 * VW, VH - 8)
    elif setting == "night":
        horizon = 0.56 * VH
        yy, xx = np.mgrid[0:VH, 0:VW].astype(np.float32)
        sky = (yy < horizon) * (0.18 + 0.40 * np.clip(1 - yy / horizon, 0, 1) ** 0.8)
        water = (yy >= horizon) * 0.22 + _streaks(rng, horizon, VH, 0.08)
        moon = np.hypot(xx - 0.62 * VW, yy - 0.28 * VH)
        glow = np.exp(-moon / 60)
        dens = (sky + water) * (1 - np.clip((36 - moon) / 1.5, 0, 1)) * (1 - 0.55 * glow)
        dens *= 1 - 0.5 * (yy >= horizon) * np.exp(-np.abs(xx - 0.62 * VW) / 10) * (0.6 + 0.4 * np.sin(yy * 0.9))
        stars = _draw(lambda d: [d.text((x, y), c, fill=255, font=ImageFont.truetype(kit.F_MONO, 11))
                                 for x, y, c in zip(rng.uniform(0, VW, 34), rng.uniform(0, horizon - 20, 34),
                                                    rng.choice(list("·+*."), 34))], 0.3)
        ink += np.clip(dens - 0.45 * stars, 0, 1)
        ink += 0.5 * np.exp(-np.abs(yy - horizon) / 1.4)
        _place(ink, k_ink, 0.40 * VW, VH - 4)
        _place(hair, k_hair, 0.40 * VW, VH - 4)
    elif setting == "rain":
        ground = 0.74 * VH
        ink += _wash(_ridge(rng, 0.44 * VH, 0.12 * VH, 1 / 150), 20, 0.16)
        yy = np.arange(VH, dtype=np.float32)[:, None]
        ink += (yy > ground) * 0.08 + 0.4 * np.exp(-np.abs(yy - ground) / 1.3)
        def puddles(d):
            for x, y, r in zip(rng.uniform(40, VW - 40, 6), rng.uniform(ground + 14, VH - 10, 6), rng.uniform(24, 46, 6)):
                d.ellipse((x - r, y - r * 0.22, x + r, y + r * 0.22), outline=150, width=1)
            for x, y in zip(rng.uniform(0, VW, 90), rng.uniform(0, VH, 90)):
                d.line((x, y, x - 5, y + 22), fill=90, width=1)
        ink += _draw(puddles, 0.4)
        _place(ink, k_ink, 0.47 * VW, VH - 6)
        _place(hair, k_hair, 0.47 * VW, VH - 6)
    ink, hair = np.clip(ink, 0, 1), np.clip(hair, 0, 1)
    pts = np.argwhere(hair > 0.5)[:, ::-1].astype(np.float32)        # (x, y) on her hair
    return ink, hair, pts


@lru_cache(None)
def ascii_picture(idx):
    """The same picture as code: one glyph per 6 x 10 cell, darker glyphs where the ink is denser."""
    ink, hair, _ = picture(idx)
    d = np.clip(ink + 0.8 * hair, 0, 1)
    cw, ch = 6, 10
    gh, gw = VH // ch, VW // cw
    cells = d[:gh * ch, :gw * cw].reshape(gh, ch, gw, cw).mean((1, 3))
    out = Image.new("L", (VW, VH), 0)
    for gy, gx in zip(*np.nonzero(cells > 0.03)):
        c = kit.RAMP[1 + min(len(kit.RAMP) - 2, int(min(1.0, cells[gy, gx] * 1.7) * (len(kit.RAMP) - 1)))]
        g = kit.mono_glyph(c, 10)
        out.paste(Image.new("L", g.size, 255), (int(gx * cw), int(gy * ch)), g)
    return np.asarray(out, np.float32) / 255


@lru_cache(None)
def develop_field(idx):
    n = ndimage.gaussian_filter(np.random.default_rng(200 + idx).standard_normal((VH, VW)), 7)
    return ((n - n.min()) / np.ptp(n)).astype(np.float32)


def live(idx, t):
    """(ink, hair) added to the open card at time t: the picture is alive."""
    setting = CARDS[idx][0]
    rng = np.random.default_rng(300 + idx)
    if setting in ("pond", "osmanthus"):
        def rings(d):
            for k in range(3):
                a = (t * 0.55 + k / 3) % 1.0
                r = 8 + 70 * a
                d.ellipse((0.63 * VW - r, 0.55 * VH - r * 0.25, 0.63 * VW + r, 0.55 * VH + r * 0.25),
                          outline=int(170 * (1 - a)), width=1)
        ink = _draw(rings, 0.5)
        hair = np.zeros_like(ink)
        if setting == "osmanthus":
            def fall(d):
                for k in range(10):
                    ph = rng.uniform(0, 1)
                    a = (t * 0.18 + ph) % 1.0
                    x = rng.uniform(VW * 0.45, VW * 0.95) - 120 * a + 8 * math.sin(6 * a + k)
                    y = 60 + (VH - 60) * a
                    d.ellipse((x - 2.4, y - 1.6, x + 2.4, y + 1.6), fill=220)
            hair = _draw(fall, 0.4)
        return ink, hair
    if setting == "rain":
        def drops(d):
            for x0, ph in zip(rng.uniform(0, VW + 60, 70), rng.uniform(0, 1, 70)):
                a = (t * 1.6 + ph) % 1.0
                x, y = x0 - 30 * a, -20 + (VH + 20) * a
                d.line((x, y, x - 4, y + 18), fill=140, width=1)
        ink = _draw(drops, 0.3)
        return ink, np.zeros_like(ink)
    if setting == "night":
        def twinkle(d):
            for x, y, ph in zip(rng.uniform(0, VW, 10), rng.uniform(10, 0.5 * VH, 10), rng.uniform(0, 6.28, 10)):
                a = 0.5 + 0.5 * math.sin(3 * t + ph)
                d.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=int(255 * a))
        return -0.4 * _draw(twinkle, 0.5), np.zeros((VH, VW), np.float32)
    def birds(d):                                                     # the way home: two birds crossing the sky
        for k in range(2):
            a = (t * 0.07 + 0.4 * k) % 1.0
            x, y = -20 + (VW + 40) * a, 0.22 * VH + 14 * k + 4 * math.sin(2 * t + k)
            w = 7 + 2 * math.sin(9 * t + k)
            d.line((x - w, y - 3, x, y, x + w, y - 3), fill=200, width=1)
    ink = _draw(birds, 0.3)
    return ink, np.zeros_like(ink)


@lru_cache(None)
def card_mask():
    """Antialiased rounded-rectangle alpha of a card, and its 1.5 px outline."""
    big = Image.new("L", (CW * 2, CH * 2), 0)
    ImageDraw.Draw(big).rounded_rectangle((0, 0, CW * 2 - 1, CH * 2 - 1), radius=28, fill=255)
    line = Image.new("L", (CW * 2, CH * 2), 0)
    ImageDraw.Draw(line).rounded_rectangle((1, 1, CW * 2 - 2, CH * 2 - 2), radius=27, outline=255, width=3)
    return (np.asarray(big.resize((CW, CH), Image.Resampling.LANCZOS), np.float32) / 255,
            np.asarray(line.resize((CW, CH), Image.Resampling.LANCZOS), np.float32) / 255)


@lru_cache(None)
def caption(idx):
    im = Image.new("L", (CW, CAP), 0)
    ImageDraw.Draw(im).text((PAD + 22, CAP // 2 - 2), CARDS[idx][2], font=ImageFont.truetype(kit.F_MONO, 15), fill=255,
                            anchor="lm")
    return np.asarray(im, np.float32) / 255


def card(idx, develop, t=None, focused=False):
    """RGBA image of card idx. develop: 0 (code), 1 (ink) or a per-pixel mask; t: the open card's time (alive)."""
    ink, hair, _ = picture(idx)
    if t is not None:
        li, lh = live(idx, t)
        ink, hair = np.clip(ink + li, 0, 1), np.clip(hair + lh, 0, 1)
    paint = FILL * (1 - ink[..., None] * (1 - np.array(kit.INK, np.float32) / 255)) \
        * (1 - hair[..., None] * (1 - np.array(kit.HAIR, np.float32) / 255))
    code = FILL * (1 - 0.6 * ascii_picture(idx)[..., None] * (1 - CODE))
    m = develop if np.ndim(develop) else np.full((VH, VW), float(develop), np.float32)
    rgb = np.empty((CH, CW, 3), np.float32)
    rgb[:] = FILL
    rgb[PAD:PAD + VH, PAD:PAD + VW] = code * (1 - m[..., None]) + paint * m[..., None]
    cap = caption(idx)[..., None]
    rgb[PAD + VH:] = rgb[PAD + VH:] * (1 - cap) + (np.array(kit.GREY, np.float32) / 255) * cap
    if focused:                                                       # the open card's dot
        yy, xx = np.mgrid[0:CAP, 0:CW]
        dot = np.clip(4.5 - np.hypot(xx - (PAD + 8), yy - (CAP // 2 - 2)), 0, 1)[..., None]
        rgb[PAD + VH:] = rgb[PAD + VH:] * (1 - dot) + (np.array(kit.TERRA, np.float32) / 255) * dot
    alpha, line = card_mask()
    rgb = rgb * (1 - line[..., None]) + (np.array(EDGE, np.float32) / 255) * line[..., None]
    out = np.dstack([rgb, alpha * 0.96])
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")
