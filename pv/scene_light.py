"""Scene: the world turns to light (29.9 - 47 s: the bridge out of the ink, then lyric lines 6 - 10).

Digitize. As line 5's held note ends, a scanline sweeps down the ink page and turns everything above it into
light: the paper goes dark and the ink becomes a matrix of glowing dots. In the drums' one silent beat the dots hang
there; on the hit at 31.1 s they lift off the page into depth, the camera starts to move, and they fly together
into her, standing in the rain, made of light (light3d.py renders it: depth of field, fog, a wet mirror floor).
It rains light; each sung syllable drops one bright drop that rings out on the floor. On line 10 she turns her back
and walks away into the dark, out of focus. Nothing moves on the drums except the one hit that frees the light.
"""
import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import kit
import light3d as L3
import lyrics
import scene_inkflow as ink
from kit import H, W

T0, T1 = 29.90, 46.8                  # the climax takes over on the drum at 46.8 s
PATH = "runtime://present/rain"
HER_X = -1.15                         # she stands left of centre, as she did on the page
CAM0 = (0.0, 1.05, -5.2)              # while scanning the camera looks straight ahead: the page is the image plane
PLANE = 5.2
PITCH = 6                             # px between the dots of the matrix
N_HER = 90000


def in_out_cubic(x):
    x = kit.clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def out_cubic(x):
    return 1 - (1 - kit.clamp01(x)) ** 3


@lru_cache(None)
def times():
    L = {ln["i"]: ln for ln in kit.lines()}
    held_end = kit.note_ends()[(5, len(L[5]["chars"]) - 1)]
    hit = next(tk for tk, s in kit.hits("kick") if tk > held_end + 0.9 and s > 0.6)       # the drum after the gap
    return SimpleNamespace(scan0=held_end - 0.06, scan1=held_end + 0.64, hit=hit, formed=hit + 1.75,
                           turn=L[10]["chars"][0]["s"], walk=L[10]["chars"][0]["s"] + 0.7, end=T1)


def rank_match(src, dst):
    """For each dst point, a src index, matched by rank along height so heads fly to heads."""
    order_s = np.argsort(src[:, 1] + 0.15 * src[:, 0])
    order_d = np.argsort(dst[:, 1] + 0.15 * dst[:, 0])
    out = np.empty(len(dst), np.int64)
    out[order_d] = order_s[(np.arange(len(dst)) * len(src)) // len(dst)]
    return out


@lru_cache(None)
def static():
    S = SimpleNamespace()
    T = times()
    # the dot matrix: the ink page at the end of line 5, one dot per 6 px cell where there is ink
    d, _, _ = ink.mix(T.scan0)
    s = ink.push(T.scan0)
    gy, gx = np.mgrid[0:H // PITCH, 0:W // PITCH]
    sx, sy = gx * PITCH + PITCH / 2, gy * PITCH + PITCH / 2
    px = np.clip((ink.CX + (sx - ink.CX) / s) / 2, 0, 959).astype(int)
    py = np.clip((0.6 * H + (sy - 0.6 * H) / s) / 2, 0, 539).astype(int)
    k, o = d[0][py, px], d[1][py, px]
    keep = (k + o) > 0.12
    S.dot_uv = np.stack([sx[keep], sy[keep]], 1).astype(np.float32)
    S.dot_rgb = np.where((o > k)[keep][:, None], L3.ORANGE, L3.CREAM).astype(np.float32) * np.clip(k + o, 0, 1)[keep][:, None]
    cam0 = L3.Camera(CAM0, (CAM0[0], CAM0[1], 0.0), aperture=0.0)
    S.dot_xyz = cam0.unproject(L3.t32(S.dot_uv[:, 0]), L3.t32(S.dot_uv[:, 1]), PLANE).cpu().numpy()

    # her in the rain, and her back turned, as clouds of light; every particle starts at a dot of the page
    for name, pose in (("rain", "umbrella"), ("away", "back_walk")):
        xyz, rgb, ph = L3.figure_points(pose, N_HER, seed=1 if name == "rain" else 2)
        setattr(S, name, (xyz + np.array([HER_X, 0, 0], np.float32), rgb, ph))
    S.src = rank_match(S.dot_xyz, S.rain[0])
    S.away_src = rank_match(S.rain[0], S.away[0])
    rng = np.random.default_rng(8)
    S.jitter = rng.normal(0, 0.006, (N_HER, 3)).astype(np.float32)
    S.swirl = rng.normal(0, 1, (N_HER, 3)).astype(np.float32)
    # the page bends into a relief before it lets go: a smooth depth for every dot
    bend = ndimage.gaussian_filter(rng.standard_normal((H // 8, W // 8)), 9)
    bend = bend / np.abs(bend).max()
    S.relief = bend[(S.dot_uv[:, 1] // 8).astype(int), (S.dot_uv[:, 0] // 8).astype(int)].astype(np.float32)
    # and lets go head first: each particle's start, by the height it is flying to
    y = S.rain[0][:, 1]
    S.stagger = ((y.max() - y) / (y.max() - y.min())).astype(np.float32)
    # a lamp above her: its cone lights the rain, a mist hangs in it, a pool of light lies under it
    S.lamp = np.array([HER_X + 0.25, 4.6, 0.3], np.float32)
    h = rng.uniform(0, 1, 7000) ** 0.7
    ang, rad = rng.uniform(0, 2 * np.pi, 7000), np.sqrt(rng.uniform(0, 1, 7000))
    reach = math.tan(math.radians(15)) * 4.6 * h
    S.mist = np.stack([S.lamp[0] + reach * rad * np.cos(ang), S.lamp[1] - 4.6 * h, S.lamp[2] + reach * rad * np.sin(ang)], 1).astype(np.float32)
    ang, rad = rng.uniform(0, 2 * np.pi, 9000), np.sqrt(rng.uniform(0, 1, 9000)) * 1.35
    S.pool = np.stack([S.lamp[0] + rad * np.cos(ang), np.full(9000, 0.003), S.lamp[2] + rad * np.sin(ang)], 1).astype(np.float32)
    S.pool_w = np.exp(-(rad / 0.8) ** 2).astype(np.float32)
    # the world: rain, motes of dust, a faint grid on the floor
    n = 1500
    S.rain_base = np.stack([rng.uniform(-7, 7, n), rng.uniform(0, 6, n), rng.uniform(-2, 16, n)], 1).astype(np.float32)
    S.rain_speed = rng.uniform(5.5, 7.5, n).astype(np.float32)
    S.motes = np.stack([rng.uniform(-6, 6, 1200), rng.uniform(0, 4, 1200), rng.uniform(-3, 13, 1200)], 1).astype(np.float32)
    S.mote_dir = rng.normal(0, 0.05, (1200, 3)).astype(np.float32)
    lines_x = [np.stack([np.full(400, x), np.zeros(400), np.linspace(-2, 18, 400)], 1) for x in np.arange(-7, 7.01, 0.5)]
    lines_z = [np.stack([np.linspace(-7, 7, 280), np.zeros(280), np.full(280, z)], 1) for z in np.arange(-2, 18.01, 0.5)]
    S.grid = np.concatenate(lines_x + lines_z).astype(np.float32)
    S.drops = [(c["s"], HER_X + rng.uniform(-2.2, 3.2), rng.uniform(-1.4, 2.8)) for ln in kit.lines()
               if 6 <= ln["i"] <= 10 for c in ln["chars"]]
    S.font = ImageFont.truetype(kit.F_MONO, 15)
    S.ink_t = None
    return S


def camera_pose(t):
    """(position, target, focus distance, aperture) of the camera at t (after the hit)."""
    T = times()
    e = in_out_cubic((t - T.hit) / 2.6)
    s = max(0.0, t - T.hit)
    away = in_out_cubic((t - T.walk) / 2.4)                               # the lens opens as she leaves
    yaw = math.radians(16) * e + math.radians(13) * math.sin(2 * math.pi * s / 17) * e   # a slow arc to and fro
    crane = 0.10 + 0.32 * (0.5 - 0.5 * math.cos(2 * math.pi * s / 13))                   # rises and settles
    pivot = np.array([HER_X + 0.9, 0.95, 0.0]) * e + np.array([CAM0[0], CAM0[1], 0.0]) * (1 - e)
    pivot = pivot + np.array([0.0, 0.0, 0.9]) * away                      # follows her a little as she goes
    r = PLANE - 0.6 * e - 0.35 * math.sin(2 * math.pi * s / 21) * e      # drifts in and out
    pos = pivot + np.array([math.sin(yaw) * r, crane * e, -math.cos(yaw) * r]) + L3.handheld(t) * e
    return pos, pivot, r, 0.085 * e * (1 + 1.4 * away)


def camera(t):
    pos, target, focus, aperture = camera_pose(t)
    return L3.Camera(tuple(pos), tuple(target), fov=38, focus=focus, aperture=aperture)


def turn_y(p, theta, cx):
    """Points rotated by theta about the vertical axis through x = cx, z = 0."""
    c, s = math.cos(theta), math.sin(theta)
    x, z = p[:, 0] - cx, p[:, 2]
    return torch.stack([cx + c * x + s * z, p[:, 1], -s * x + c * z], 1)


_last = {}


def her_cloud(t):
    """(xyz, rgb, phase) of her particles at t, as tensors (the last frame's answer is kept: threads ask often)."""
    if t not in _last:
        _last.clear()
        _last[t] = _her_cloud(t)
    return _last[t]


def _her_cloud(t):
    """Positions, colours, phases and how far each particle has come from the page (0 = still a dot, 1 = her)."""
    S, T = static(), times()
    xyz, rgb, ph = (L3.t32(a) for a in S.rain)
    start, dot_col = L3.t32(S.dot_xyz[S.src]), L3.t32(S.dot_rgb[S.src])
    if t < T.hit:
        return start, dot_col, ph, torch.zeros(N_HER, device=L3.DEV)
    bent = start + L3.t32(S.relief[S.src])[:, None] * L3.t32((0.0, 0.0, 1.4)) * out_cubic((t - T.hit) / 0.5)
    e = ((t - (T.hit + 0.35 + 0.55 * L3.t32(S.stagger))) / 0.85).clamp(0, 1)    # head first, a stream
    e = torch.where(e < 0.5, 4 * e ** 3, 1 - (-2 * e + 2) ** 3 / 2)
    pos = bent * (1 - e[:, None]) + xyz * e[:, None] + L3.t32(S.swirl) * 0.25 * torch.sin(math.pi * e)[:, None]
    col = dot_col * (1 - e[:, None]) + rgb * e[:, None]
    if t > T.turn:                                    # she turns her back (a real half turn), then walks away
        a_xyz, a_rgb, _ = (L3.t32(a) for a in S.away)
        idx = torch.as_tensor(S.away_src, device=L3.DEV)
        f = in_out_cubic((t - T.turn) / 0.9)
        th = math.pi * f
        pos = turn_y(pos[idx], th, HER_X) * (1 - f) + turn_y(a_xyz, th - math.pi, HER_X) * f
        col = col[idx] * (1 - f) + a_rgb * f
        w = in_out_cubic((t - T.walk) / 2.4)
        pos = pos + L3.t32((0.35, 0.0, 3.4)) * w
        col = col * (1 - 0.5 * w) * L3.t32((0.92, 0.96, 1.04))
    shimmer = L3.t32(S.jitter) * 1.5 * torch.sin(ph + 2.0 * t)[:, None] * e[:, None]
    return pos + shimmer, col, ph, e


def light(t):
    """The light world at t, as a (3, H, W) tensor."""
    S, T = static(), times()
    cam = camera(t) if t >= T.hit else L3.Camera(CAM0, (CAM0[0], CAM0[1], 0.0), aperture=0.0)
    fr = L3.Frame(cam)
    pos, col, ph, e = her_cloud(t)
    world = kit.smooth((t - T.hit) / 1.2)
    if t < T.hit:                                                          # the dot matrix, hanging in the silence
        tw = 0.9 + 0.1 * torch.sin(ph + 9.0 * t)[:, None]
        fr.add(pos, col * tw, size=1.0, gain=0.42)                         # ~18 particles stack on every dot
        fr.add(pos, col, size=4.0, gain=0.06)
        return fr.resolve()
    tw = 0.8 + 0.2 * torch.sin(ph + 6.0 * t)[:, None]
    fr.add(pos, col * tw * (0.42 + 0.23 * e)[:, None])                     # a dot's brightness becomes hers
    fr.add(pos, col * (0.06 + 0.18 * e)[:, None], size=3.0)
    refl = pos * L3.t32((1, -1, 1))
    fr.add(refl, col * (0.20 * world) * torch.exp(-pos[:, 1:2].clamp(min=0) / 0.9), size=3.0)
    # rain falling forever, fading in, bright where the lamp's cone catches it
    y = (L3.t32(S.rain_base[:, 1]) - L3.t32(S.rain_speed) * t) % 6.0
    base = torch.stack([L3.t32(S.rain_base[:, 0]) - 0.02 * y, y, L3.t32(S.rain_base[:, 2])], 1)
    streak = torch.cat([base + L3.t32((0.004 * k, 0.018 * k, 0)) for k in range(20)])   # dense enough to read as a line
    v = streak - L3.t32(S.lamp)
    cos = -v[:, 1] / v.norm(dim=1).clamp(min=1e-3)
    lit = ((cos - math.cos(math.radians(18))) / (math.cos(math.radians(13)) - math.cos(math.radians(18)))).clamp(0, 1)
    fr.add(streak, L3.t32(L3.CREAM) * (0.24 * world * (1 + 6.0 * lit))[:, None])
    warm = L3.t32((1.0, 0.88, 0.74))
    mist = L3.t32(S.mist) + L3.t32((0.0, -0.12, 0.0)) * math.sin(0.4 * t)
    fr.add(mist, warm, size=3.0, gain=1.1 * world)                         # the beam: a haze that holds the light
    fr.add(L3.t32(S.pool), warm * (0.25 * world) * L3.t32(S.pool_w)[:, None], size=2.0)
    fr.add(L3.t32(S.lamp)[None], warm, size=6.0, gain=40.0 * world)        # the lamp itself, if the frame reaches it
    drift = L3.t32(S.motes) + L3.t32(S.mote_dir) * t
    fr.add(drift, L3.CREAM, gain=1.1 * world)
    fr.add(L3.t32(S.grid), L3.TERRA, gain=0.05 * world)
    # one bright drop per sung syllable, and its ring on the floor
    for s, rx, rz in S.drops:
        a = t - s
        if -0.3 < a < 0:
            u = (a + 0.3) / 0.3
            p = L3.t32([[rx, 2.4 * (1 - u * u) + 0.06 * k, rz] for k in range(6)])
            fr.add(p, L3.CREAM, gain=0.9 * world)
        elif 0 <= a < 1.5:
            ang = torch.linspace(0, 2 * math.pi, 120, device=L3.DEV)
            for lag, g in ((0.0, 1.0), (0.16, 0.6)):
                if a - lag <= 0:
                    continue
                r = 0.06 + 0.95 * math.sqrt(a - lag)
                ring = torch.stack([rx + r * torch.cos(ang), torch.full_like(ang, 0.002), rz + r * torch.sin(ang)], 1)
                fr.add(ring, L3.CREAM, gain=0.9 * g * (1 - a / 1.5) ** 2 * world)
    return fr.resolve()


def scanline(img, y, t):
    """The bright line that turns ink into light, with its readout riding at the right."""
    a = np.asarray(img, np.float32) / 255
    rows = np.arange(H, dtype=np.float32)[:, None]
    glow = 0.9 * np.exp(-((rows - y) / 1.6) ** 2) + 0.35 * np.exp(-((rows - y) / 22) ** 2) * (rows < y)
    col = np.array(L3.TERRA, np.float32) * 0.6 + 0.4
    a = np.clip(a + glow[..., None] * col, 0, 1)
    out = Image.fromarray((a * 255 + 0.5).astype(np.uint8))
    pct = int(100 * kit.clamp01(y / H))
    ImageDraw.Draw(out).text((W - 40, y - 22), f"scan  ·  ink → light  ·  {pct:3d}%", font=static().font,
                             fill=(236, 214, 196), anchor="ra")
    return out


def plate(t, ghost=None):
    T = times()
    lit = light(t)
    if ghost is not None and ghost.any():
        lit = lit + L3.t32(ghost)[None] * L3.t32(L3.CREAM)[:, None, None] * 0.9
    img = L3.to_image(lit)
    if t < T.scan1 + 0.05:                                                 # the scan: light above, ink below
        y = -20 + (H + 40) * in_out_cubic((t - T.scan0) / (T.scan1 - T.scan0))
        below = np.asarray(ink.plate(t, ghost=ghost), np.float32)
        above = np.asarray(img, np.float32)
        m = np.clip((np.arange(H, dtype=np.float32) - y) / 3, 0, 1)[:, None, None]
        img = scanline(Image.fromarray((above * (1 - m) + below * m + 0.5).astype(np.uint8)), y, t)
    return img


def her_point(t, i, k):
    """Threads land on her once she has formed out of the light."""
    T = times()
    if t < T.formed or t > T.walk + 0.6:
        return None
    pos = her_cloud(t)[0]
    idx = (i * 7919 + k * 104729) % N_HER
    u, v, _ = camera(t).project(pos[idx:idx + 1])
    return float(u[0]), float(v[0])
