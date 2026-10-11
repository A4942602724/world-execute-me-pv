"""Scene: the pond, 0 - 16.6 s (lyric lines 1 - 3).

An ink drop falls on the first drum hit and the page bleeds open from where it lands. During line 1 dew footsteps
cross the water towards her, one ripple per sung character; during line 2 rings spread one after another in front of
her, one per character; during line 3 a breeze lifts osmanthus petals off her hair and carries them over the pond,
and on the line's last two characters they fall into the water. Only the voice moves things: the drums do not.
"""
import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

import kit
from kit import H, W

T0, T1 = 0.0, 16.6
SHORE = 0.70 * H
HER_X, HER_H = 0.34 * W, 560
POSE = "character_claude"
PAD = 200                                   # far and mid mountains are wider than the frame, for parallax


def vsmooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def lerp(a, b, u):
    return a + (b - a) * u


# ---------------------------------------------------------------- built once per process
@lru_cache(None)
def static():
    S = SimpleNamespace()
    xs = np.arange(W + 2 * PAD) - PAD
    gap = np.abs(xs - HER_X)                # a misty valley behind her, so her silhouette reads
    S.far = kit.mountains("far_pond", W + 2 * PAD, 0.47 * H, 0.26 * H, 70, seed=3, freq=1 / 520) \
        * (0.30 * (0.55 + 0.45 * vsmooth(gap / 600)))[None, :]
    S.mid = kit.mountains("mid_pond", W + 2 * PAD, 0.62 * H, 0.20 * H, 46, seed=5, freq=1 / 380) \
        * (0.52 * (0.25 + 0.75 * vsmooth(gap / 520)))[None, :]

    rng = np.random.default_rng(21)
    yy = np.arange(H, dtype=np.float32)[:, None]
    depth = np.clip((yy - SHORE) / (H - SHORE), 0, 1)
    streak = ndimage.gaussian_filter(rng.standard_normal((H, W)), (0.7, 60))
    water = (yy > SHORE) * (0.04 + 0.10 * depth ** 1.2 + 0.05 * np.clip(streak / streak.std(), -1, 2))
    xs = np.arange(W)
    line = SHORE + 8 * (kit.fbm1d(W, rng, 1 / 300) - 0.5)
    brush = 0.55 + 0.45 * kit.fbm1d(W, rng, 1 / 90)
    taper = vsmooth((xs - 0.02 * W) / 160) * vsmooth((0.80 * W - xs) / 260)
    shore = 0.65 * np.exp(-((yy - line[None, :]) / 2.4) ** 2) * (brush * taper)[None, :]
    shore += 0.25 * np.exp(-((yy - line[None, :] - 6) / 1.6) ** 2) * (brush ** 3 * taper)[None, :]
    S.water = np.clip(water, 0, 1).astype(np.float32)
    S.shore = np.clip(shore, 0, 1).astype(np.float32)

    ink, hair = kit.figure(POSE, HER_H)
    rows = np.flatnonzero((ink + hair).max(1) > 0.3)
    S.feet = SHORE + 6
    x0, y0 = HER_X - ink.shape[1] / 2, S.feet - rows[-1]
    S.fig_ink, S.fig_hair = kit.place(ink, x0, y0), kit.place(hair, x0, y0)
    S.fig_top = y0 + rows[0]

    r = np.arange(H)
    src = (2 * S.feet - r).astype(int)
    ok = (r > S.feet) & (src >= 0) & (src < H)
    fade = np.clip(1 - (r - S.feet) / (0.75 * HER_H), 0, 1) ** 1.3 * 0.30
    S.refl_ink, S.refl_hair = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    S.refl_ink[r[ok]] = S.fig_ink[src[ok]] * fade[ok, None]
    S.refl_hair[r[ok]] = S.fig_hair[src[ok]] * fade[ok, None]
    S.refl_ink = ndimage.gaussian_filter(S.refl_ink, (2.0, 0.8))
    S.refl_hair = ndimage.gaussian_filter(S.refl_hair, (2.0, 0.8))

    S.reveal_noise = ndimage.gaussian_filter(rng.standard_normal((H, W + 2 * PAD)), 40)
    S.reveal_noise /= S.reveal_noise.std()
    n = ndimage.gaussian_filter(rng.standard_normal((H, W)), 10)
    n = (n - n.min()) / np.ptp(n)
    vertical = np.clip((np.arange(H, dtype=np.float32)[:, None] - S.fig_top) / HER_H, 0, 1)
    S.her_field = (0.6 * n + 0.4 * vertical).astype(np.float32)   # she soaks in from the head down, unevenly

    hy, hx = np.nonzero(S.fig_hair[: int(S.fig_top + 0.45 * HER_H)] > 0.5)
    S.petal_source = np.stack([hx, hy], 1)
    S.drop_t = next(tk for tk, _ in kit.hits("kick") if tk > 0.1)
    S.drop_at = (0.53 * W, 0.83 * H)
    S.ripples = ripple_events(S)
    S.reeds = [(0.05 * W + k * 0.012 * W + rng.normal(0, 6), rng.uniform(60, 150), rng.uniform(-0.15, 0.25),
                rng.uniform(0, 6.28)) for k in range(14)]
    S.static_far = kit.to_ink(S.far, kit.INK_FAR)
    S.static_mid = kit.to_ink(S.mid, kit.INK_MID)
    return S


def ripple_events(S):
    """(t0, x, y, strength, speed, life, rings) for every ripple in the scene."""
    L = {ln["i"]: ln for ln in kit.lines()}
    ev = [(S.drop_t, *S.drop_at, 1.0, 560, 3.2, 3)]
    p0, p1 = (0.70 * W, 0.95 * H), (0.41 * W, 0.745 * H)            # line 1: footsteps of dew towards her
    n = len(L[1]["chars"])
    for k, c in enumerate(L[1]["chars"]):
        u = k / (n - 1)
        x = lerp(p0[0], p1[0], u) + 45 * math.sin(u * math.pi) * (1 if k % 2 else -1) * 0.6
        ev.append((c["s"], x, lerp(p0[1], p1[1], u ** 0.8), 0.55, 120, 1.9, 2))
    for k, c in enumerate(L[2]["chars"]):                           # line 2: one ring per character
        ev.append((c["s"], 0.42 * W, 0.80 * H, 0.85, 190 + 30 * k, 2.8, 3))
    S.petals = petal_paths(S, L[3])
    ev += [(p.land_t, p.land_x, p.land_y, 0.35, 80, 1.5, 1) for p in S.petals if p.land_t is not None]
    return sorted(ev)


def petal_paths(S, line):
    rng = np.random.default_rng(9)
    petals = []
    n = len(line["chars"])
    for k, c in enumerate(line["chars"]):
        heavy = k >= n - 2                                          # the last two characters: they fall
        for j in range(6 if heavy else 3):
            x0, y0 = S.petal_source[rng.integers(len(S.petal_source))]
            p = SimpleNamespace(t0=c["s"] + 0.05 * j, x0=float(x0), y0=float(y0), vx=rng.uniform(110, 200),
                                vy=rng.uniform(-70, -30), g=rng.uniform(150, 210) if heavy else rng.uniform(45, 80),
                                ph=rng.uniform(0, 6.28), rot=rng.uniform(0, 6.28), spin=rng.uniform(-4, 4),
                                size=rng.uniform(6, 10), floor=rng.uniform(SHORE + 10, 0.97 * H), land_t=None,
                                floret=rng.random() < 0.4, tumble=rng.uniform(3, 7))
            for step in range(1, 240 * 12):                         # find where it lands
                a = step / 240
                x, y = petal_xy(p, a)
                if y >= p.floor:
                    p.land_t, p.land_x, p.land_y = p.t0 + a, x, y
                    break
            petals.append(p)
    return petals


def petal_xy(p, a):
    return (p.x0 + p.vx * a + 14 * math.sin(2.3 * a + p.ph),
            p.y0 + p.vy * a + 0.5 * p.g * a * a + 8 * math.sin(3.1 * a + p.ph))


# ---------------------------------------------------------------- per frame
def reveal(S, t):
    """Where the page has bled open (world grid, W + 2 PAD wide), or None once it is fully open."""
    if t >= S.drop_t + 2.6:
        return None
    if t < S.drop_t:
        return np.zeros((H, W + 2 * PAD), np.float32)
    r = 2600 * kit.ease_out(kit.since(t, S.drop_t, 2.4))
    yy, xx = np.mgrid[0:H, 0:W + 2 * PAD]
    dist = np.hypot(xx - PAD - S.drop_at[0], yy - S.drop_at[1]) + 220 * S.reveal_noise
    return np.clip((r - dist) / 160, 0, 1).astype(np.float32)


def ripples(S, t):
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    act = 0.0
    for t0, x, y, s, v, life, rings in S.ripples:
        a = t - t0
        if not 0 <= a < life:
            continue
        depth = (y - SHORE) / (H - SHORE)
        asp, vv = 0.27 + 0.10 * depth, v * (0.55 + 0.45 * depth)
        act += s * (1 - a / life)
        for k in range(rings):
            aa = a - 0.18 * k
            if aa <= 0:
                continue
            r = vv * aa ** 0.7
            fade = s * (1 - a / life) ** 1.6 * (1 - 0.3 * k)
            d.ellipse((x - r, y - r * asp, x + r, y + r * asp), outline=int(255 * fade), width=max(1, round(2.6 - 0.6 * k)))
    dens = np.asarray(im.filter(ImageFilter.GaussianBlur(1.0)), np.float32) / 255 * 0.75
    return dens, min(act, 1.0)


def wind(t):
    ln = {l["i"]: l for l in kit.lines()}[3]
    return kit.smooth(kit.since(t, ln["start"] - 0.3, 0.8)) * (1 - kit.smooth(kit.since(t, ln["end"] + 0.3, 1.6)))


def her_reveal(S, t):
    """Her ink soaking into the page after the drop lands: 0, 1 or a per-pixel mask in between."""
    p = kit.since(t, S.drop_t + 0.35, 1.1)
    if p <= 0 or p >= 1:
        return float(p >= 1)
    return np.clip((p * 1.35 - S.her_field) / 0.12, 0, 1)


def drop(S, t):
    """The falling ink drop, before the page opens."""
    im = Image.new("L", (W, H), 0)
    if S.drop_t - 0.5 <= t < S.drop_t:
        u = (t - (S.drop_t - 0.5)) / 0.5
        y = lerp(-30, S.drop_at[1], u * u)
        ImageDraw.Draw(im).ellipse((S.drop_at[0] - 5, y - 9, S.drop_at[0] + 5, y + 7), fill=255)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255 * 0.9


def reeds(S, t):
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    w = wind(t)
    for x, h, lean, ph in S.reeds:
        bend = lean + 0.06 * math.sin(1.3 * t + ph) + w * (0.35 + 0.08 * math.sin(4.1 * t + ph))
        pts_l, pts_r = [], []
        for u in np.linspace(0, 1, 9):
            px = x + h * bend * u * u
            py = SHORE + 4 - h * u * (1 - 0.15 * bend * bend)
            half = 1.8 * (1 - u) + 0.3
            pts_l.append((px - half, py))
            pts_r.append((px + half, py))
        d.polygon(pts_l + pts_r[::-1], fill=200)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255 * 0.85


def petals(S, t):
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for p in S.petals:
        a = t - p.t0
        if a < 0:
            continue
        landed = p.land_t is not None and t >= p.land_t
        if landed:                                                  # floating, drifting, fading
            fl = t - p.land_t
            if fl > 3.0:
                continue
            x, y, rot, alpha = p.land_x + 6 * fl, p.land_y, p.rot + p.spin * (p.land_t - p.t0), 1 - fl / 3.0
        else:
            x, y = petal_xy(p, a)
            rot, alpha = p.rot + p.spin * a, kit.since(t, p.t0, 0.25)
        val = int(235 * alpha)
        if p.floret:                                                # an osmanthus floret: four round lobes
            for q in range(4):
                ang = rot + q * math.pi / 2
                cx, cy, r = x + 0.42 * p.size * math.cos(ang), y + 0.42 * p.size * math.sin(ang), 0.34 * p.size
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=val)
        else:                                                       # a single petal, tumbling as it flies
            flat = 0.8 if landed else abs(math.cos(p.tumble * a))
            maj, mnr = 0.95 * p.size, 0.22 * p.size + 0.35 * p.size * flat
            cr, sr = math.cos(rot), math.sin(rot)
            pts = [(x + maj * math.cos(s) * cr - mnr * math.sin(s) * sr, y + maj * math.cos(s) * sr + mnr * math.sin(s) * cr)
                   for s in np.linspace(0, 2 * math.pi, 14)]
            d.polygon(pts, fill=val)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(0.5)), np.float32) / 255


def wobble(arr, t, act):
    """Rows of the reflection pushed sideways by the water."""
    out = arr.copy()
    y0 = int(SHORE)
    rows = np.arange(y0, H)
    dx = (2.0 + 7 * act) * np.sin(0.11 * (rows - y0) - 3.1 * t) + 1.2 * np.sin(0.031 * (rows - y0) + 1.7 * t)
    cols = np.arange(W)
    src = cols[None, :] - dx[:, None]
    i0 = np.floor(src).astype(int)
    f = (src - i0).astype(np.float32)
    region = arr[y0:]
    a = np.take_along_axis(region, np.clip(i0, 0, W - 1), 1)
    b = np.take_along_axis(region, np.clip(i0 + 1, 0, W - 1), 1)
    out[y0:] = a * (1 - f) + b * f
    return out


def union(*ds):
    out = np.zeros((H, W), np.float32)
    for d in ds:
        out = out + d - out * d
    return out


def frame(t):
    S = static()
    rev = reveal(S, t)
    rev_near = None if rev is None else rev[:, PAD:PAD + W]
    her = her_reveal(S, t)

    zoom = 1.0 + 0.045 * kit.smooth(t / T1)                         # a slow drift; nothing moves on the drums
    pan = (30 - 60 * kit.smooth(t / T1), 0.0)
    img = kit.paper().copy()
    for dens, static_img, colour, depth in ((S.far, S.static_far, kit.INK_FAR, 0.35), (S.mid, S.static_mid, kit.INK_MID, 0.6)):
        layer = static_img if rev is None else kit.to_ink(dens * rev, colour)
        img = kit.multiply(img, kit.camera(layer, 1 + (zoom - 1) * depth, (pan[0] * depth, 0), origin=(-PAD, 0)))

    rip, act = ripples(S, t)
    env_ink, water = union(rip, reeds(S, t), S.shore) * kit.ink_grain(), S.water
    if rev_near is not None:
        env_ink, water = env_ink * rev_near, water * rev_near
    soft = kit.soft_grain()
    ink = union(env_ink, wobble(S.refl_ink, t, act) * her * soft, S.fig_ink * her * soft, drop(S, t))
    hair = union(wobble(S.refl_hair, t, act) * her, S.fig_hair * her, petals(S, t)) * soft
    near = kit.stack((water, kit.INK_MID), (ink, kit.INK), (hair, kit.HAIR))
    img = kit.multiply(img, kit.camera(near, zoom, pan))

    lyr = kit.lyrics_layer(t)
    if lyr is not None:
        img = kit.multiply(img, kit.to_ink(lyr, kit.INK))
    return img
