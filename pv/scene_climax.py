"""Scene: the climax, in the light world (46.8 s to the end; lyric lines 11 - 16).

46.8  The drums come back. The camera rushes through the rain to where she walked; her light bursts and comes back
      together facing us, inside a storm of light orbiting her (line 11).
49.7  Band by band, one band per syllable of line 12, the storm settles into a cocoon around her.
53.0  The cocoon unwinds; its threads straighten into the grid of a go board on the floor and she kneels across it.
      The first phrase of line 13 sets down a stone on each syllable, hers light, mine a ring; on the second phrase
      they rise and are gone.
60.5  She gathers into one point of light, and the point draws her outline while I sing line 14: the pen moves as
      fast as the voice is loud, and a blueprint grows around her. On line 15 her light fills the outline, edge first.
68.0  Mountains of light rise behind her and, one by one, end (line 16). She turns to look back, and on that look,
      the only time in the film, she is in colour. Then she drifts up as light, the sprite waves goodbye, and the
      cursor blinks twice and goes out.
"""
import math
from functools import lru_cache
from types import SimpleNamespace

import contourpy
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

import kit
import light3d as L3
import scene_light as prev
from kit import H, W

T0, T1 = 46.8, 76.93
PATH = "runtime://present/now"
N_HER, N_STORM = 90000, 60000
C = np.array([prev.HER_X + 0.35, 0.0, 3.4], np.float32)      # where she walked to: everything happens here
KNEEL = C + np.array([-0.95, 0.0, 0.0], np.float32)           # at the board's left, facing it (the pose mirrored)
HALF = 0.8                                                     # half the go board's side
STONES = ((9, 9), (10, 10), (8, 10), (10, 8), (11, 11), (8, 8))   # the first phrase of line 13, alternately hers, mine
COCOON = (0.62, 1.08)                                          # its horizontal and vertical radii
in_out, out_cubic = prev.in_out_cubic, prev.out_cubic


def smooth(x):
    return kit.smooth(x)


def in_cubic(x):
    return kit.clamp01(x) ** 3


@lru_cache(None)
def times():
    L = {ln["i"]: ln for ln in kit.lines()}
    ends = kit.note_ends()
    s = lambda i: tuple(c["s"] for c in L[i]["chars"])
    end = lambda i: ends[(i, len(L[i]["chars"]) - 1)]
    T = SimpleNamespace(l11=s(11), l12=s(12), l13=s(13), l14=s(14), l15=s(15), l16=s(16))
    T.rush = T0 + 0.7
    T.board = end(12) - 0.25                 # the cocoon unwinds as line 12's last note fades
    T.board_in = T.board + 1.3
    T.gather = end(13) - 0.15                # she becomes the pen
    T.draw0, T.draw1 = T.l14[0] - 0.15, T.l15[0] - 0.1
    T.fill = T.l15[0]
    T.mtn = end(15) - 0.5
    T.turn0, T.turn1 = T.l16[4] - 0.2, T.l16[5] + 0.1
    T.colour0, T.colour1 = T.l16[5], T.l16[6] + 0.15
    T.held = end(16)
    return T


def rank_match(src, dst):
    return prev.rank_match(src, dst)


def turn_y(p, th):
    """Points turned by th about her own vertical axis (through C)."""
    c, s = math.cos(th), math.sin(th)
    x, z = p[:, 0] - float(C[0]), p[:, 2] - float(C[2])
    return torch.stack([float(C[0]) + c * x + s * z, p[:, 1], float(C[2]) - s * x + c * z], 1)


# ---------------------------------------------------------------- what exists, built once per process
@lru_cache(None)
def static():
    T = times()
    S = SimpleNamespace()
    t32 = L3.t32
    rng = np.random.default_rng(31)
    start = prev.her_cloud(T0)[0].cpu().numpy()                    # the far, turned-away her at the drop
    front, front_rgb, ph = L3.figure_points("character_claude", N_HER, seed=3)
    kneel, kneel_rgb, _ = L3.figure_points("go_sit", N_HER, height=1.05, seed=4)   # kneeling: lower than standing
    look, look_rgb, _ = L3.figure_points("look_back", N_HER, seed=5)
    kneel = kneel * np.array([-1, 1, 1], np.float32)                 # the art faces left; she must face the board
    front, kneel, look = front + C, kneel + KNEEL, look + C
    S.front, S.front_rgb, S.ph = t32(front), t32(front_rgb), t32(ph)
    S.kneel, S.kneel_rgb = t32(kneel), t32(kneel_rgb)
    S.look, S.look_rgb = t32(look), t32(look_rgb)
    S.look_art = t32(L3.art_colours("look_back", N_HER, seed=5))
    S.start = t32(start[rank_match(start, front)])                 # all in the front pose's order
    S.kneel_from = torch.as_tensor(rank_match(front, kneel), device=L3.DEV)
    S.look_from = torch.as_tensor(rank_match(front, look), device=L3.DEV)
    S.swirl = t32(rng.normal(0, 1, (N_HER, 3)))

    # the storm, its cocoon, its board
    r = 0.55 + 1.35 * rng.uniform(0, 1, N_STORM) ** 1.6
    h = rng.uniform(0.0, 2.1, N_STORM)
    S.r, S.h, S.phi = t32(r), t32(h), t32(rng.uniform(0, 2 * np.pi, N_STORM))
    S.omega = t32(1.8 / np.sqrt(r) * np.where(rng.random(N_STORM) < 0.85, 1, -1))
    S.storm_rgb = t32(np.where(rng.random(N_STORM)[:, None] < 0.3, L3.ORANGE, L3.CREAM) * rng.uniform(0.4, 1.0, (N_STORM, 1)))
    band = np.minimum(6, (h / 2.1 * 7).astype(int))
    S.band = torch.as_tensor(band, device=L3.DEV)
    S.lat = t32(np.radians(-72 + 144 * (band + rng.uniform(0, 1, N_STORM)) / 7))
    S.band_t = t32(np.array(T.l12)[band] - 0.1)
    S.band_dir = t32(np.where(band % 2, 1.0, -1.0))
    line = np.arange(N_STORM) % 38
    u = rng.uniform(-HALF, HALF, N_STORM)
    k = (line % 19) / 18 * 2 * HALF - HALF
    gx, gz = np.where(line < 19, k, u), np.where(line < 19, u, k)
    S.grid = t32(np.stack([C[0] + gx, np.full(N_STORM, 0.004), C[2] + gz], 1))
    S.grid_delay = t32(rng.uniform(0, 0.8, N_STORM))                 # threads peel off one after another
    S.hero = torch.as_tensor(np.arange(N_STORM) % 12 == 0, device=L3.DEV)   # the storm's bright streaks
    S.hero_idx = torch.nonzero(S.hero).squeeze(1)

    # stones
    S.sphere = rng.normal(0, 1, (300, 3))
    S.sphere = t32(S.sphere / np.linalg.norm(S.sphere, axis=1, keepdims=True) * 0.034 * rng.uniform(0.3, 1, (300, 1)))
    a = np.linspace(0, 2 * np.pi, 140, endpoint=False)
    S.ring = t32(np.stack([0.036 * np.cos(a), np.zeros(140), 0.036 * np.sin(a)], 1))

    # the pen's path: her outline, then the lines of the drawing inside it, in drawing order
    S.path, S.path_len, S.path_rgb = outline_path()
    # the fill: every particle of hers leaves from the outline point nearest to it, the edge first
    tree = cKDTree(S.path.cpu().numpy()[:, :2])
    dist, idx = tree.query(front[:, :2])
    S.fill_from = S.path[torch.as_tensor(idx, device=L3.DEV)]
    S.fill_depth = t32(np.clip(dist / np.percentile(dist, 97), 0, 1))
    S.draw_energy = draw_energy()

    # the mountains, three ranges behind her, and the image she becomes
    S.mountains = []
    for j in range(3):
        xs = np.linspace(C[0] - 7.5, C[0] + 7.5, 900)
        ys = 0.12 + (0.9 + 0.55 * j) * kit.fbm1d(900, np.random.default_rng(60 + j), 1 / 160) ** 1.3
        S.mountains.append(t32(np.stack([xs, ys, np.full(900, C[2] + 3.5 + 3.2 * j)], 1)))
    S.look_img, S.look_quad = colour_image()
    S.dust = t32(np.stack([rng.uniform(-3, 3, 900) + C[0], rng.uniform(0, 2.6, 900), rng.uniform(-1, 4, 900) + C[2]], 1))
    S.font = ImageFont.truetype(kit.F_MONO, 15)
    return S


def outline_path():
    """(points (N, 3), cumulative drawn length (N,), colour (N, 3)) of the pen's path over her standing pose."""
    L = np.load(kit.ASSETS / "silhouettes" / "character_claude.npz")
    a = L["alpha"].astype(np.float32)
    art = L["colour"].astype(np.float32) / 255
    _, hair = kit.figure("character_claude", a.shape[0])
    ys, xs = np.nonzero(a > 0.5)
    rows = np.flatnonzero(a.max(1) > 0.5)
    top, bottom = rows[0], rows[-1]
    s = 1.7 / (bottom - top)
    q = 4                                                              # trace at a quarter of the art's size
    small = a[: a.shape[0] // q * q, : a.shape[1] // q * q].reshape(a.shape[0] // q, q, a.shape[1] // q, q).mean((1, 3))
    dark = ((art.mean(-1) < 0.30) & (a > 0.5)).astype(np.float32)
    dark = dark[: a.shape[0] // q * q, : a.shape[1] // q * q].reshape(a.shape[0] // q, q, a.shape[1] // q, q).mean((1, 3))
    outer = sorted(contourpy.contour_generator(z=small).lines(0.5), key=len, reverse=True)
    inner = [ln for ln in contourpy.contour_generator(z=ndimage.gaussian_filter(dark, 0.6)).lines(0.35) if len(ln) > 10]
    paths, here = [outer[0]] + [ln for ln in outer[1:] if len(ln) > 8], None
    order, rest = paths[:1], paths[1:] + inner
    here = order[0][-1]
    while rest:                                                        # nearest next line, as a hand would go
        k = min(range(len(rest)), key=lambda i: min(np.hypot(*(rest[i][0] - here)), np.hypot(*(rest[i][-1] - here))))
        ln = rest.pop(k)
        if np.hypot(*(ln[-1] - here)) < np.hypot(*(ln[0] - here)):
            ln = ln[::-1]
        order.append(ln)
        here = ln[-1]
    pts, lens, cols, total = [], [], [], 0.0
    near_hair = ndimage.maximum_filter(hair, 5) > 0.3
    for ln in order:
        px, py = ln[:, 0] * q, ln[:, 1] * q
        world = np.stack([(px - xs.mean()) * s + C[0], (bottom - py) * s, np.full(len(px), C[2])], 1)
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(world, axis=0), axis=1))]
        n = max(2, int(seg[-1] / 0.006))
        at = np.linspace(0, seg[-1], n)
        p = np.stack([np.interp(at, seg, world[:, k]) for k in range(3)], 1)
        hy = np.clip(np.interp(at, seg, py).astype(int), 0, a.shape[0] - 1)
        hx = np.clip(np.interp(at, seg, px).astype(int), 0, a.shape[1] - 1)
        is_hair = near_hair[hy, hx]
        pts.append(p)
        lens.append(total + at)
        cols.append(np.where(is_hair[:, None], L3.ORANGE, L3.CREAM))
        total += seg[-1]
    return L3.t32(np.concatenate(pts)), L3.t32(np.concatenate(lens)), L3.t32(np.concatenate(cols))


def draw_energy():
    """(times, fraction of the outline drawn): the pen moves as fast as the voice is loud, never quite stopping."""
    T = times()
    ts = np.linspace(T.draw0, T.draw1, 400)
    speed = np.array([0.25 + kit.level("rms_vocals", t) for t in ts])
    done = np.r_[0, np.cumsum(speed[1:] * np.diff(ts))]
    return ts, done / done[-1]


def colour_image():
    """Her, looking back, in colour (RGBA, matte from the silhouette), and the world corners of where it stands."""
    L = np.load(kit.ASSETS / "silhouettes" / "look_back.npz")
    a = L["alpha"].astype(np.float32)
    rgba = np.dstack([L["colour"], (np.clip(a, 0, 1) * 255).astype(np.uint8)])
    ys, xs = np.nonzero(a > 0.5)
    rows = np.flatnonzero(a.max(1) > 0.5)
    top, bottom = rows[0], rows[-1]
    s = 1.7 / (bottom - top)
    h, w = a.shape
    corners = np.array([[(0 - xs.mean()) * s + C[0], (bottom - 0) * s], [(w - xs.mean()) * s + C[0], (bottom - h) * s]])
    img = Image.fromarray(rgba, "RGBA")
    if h > 1400:                                                       # sharper than the frame needs, quicker to scale
        img = img.resize((round(w * 1400 / h), 1400), Image.Resampling.LANCZOS)
    return img, corners


# ---------------------------------------------------------------- the camera
def front_pose(yaw, r, h, side=0.75):
    """Camera facing her from angle yaw at distance r, trucked `side` to its right so she stands left of centre
    (the lyrics have the right of the frame)."""
    right = np.array([math.cos(yaw), 0, math.sin(yaw)], np.float32) * side
    pos = C + right + np.array([math.sin(yaw) * r, h, -math.cos(yaw) * r], np.float32)
    return pos, C + right + np.array([0, 0.95, 0], np.float32)


def blend(a, b, e):
    return tuple(np.asarray(x) * (1 - e) + np.asarray(y) * e for x, y in zip(a, b))


def camera(t):
    T = times()
    p0, t0, f0, a0 = prev.camera_pose(T0)
    pos, tgt = front_pose(0.0, 3.0, 1.0)
    rush = (p0, t0, f0, a0)
    face = (pos, tgt, 3.0, 0.035)
    if t < T.rush:
        pose = blend(rush, face, in_out((t - T0) / 0.7))
    else:
        # the storm and the cocoon: a sweeping orbit, pulling out mid-way and craning up to look down on the cocoon
        e = in_out((t - T.rush) / (T.board - T.rush))
        r = 3.0 + 0.6 * math.sin(math.pi * e) + 0.1 * e
        pos, tgt = front_pose(math.radians(80) * e, r, 1.0 + 0.6 * e)
        orbit = (pos, tgt, r, 0.035)
        # the board: a slow dolly in and down towards the stones
        b = in_out((t - (T.board + 1.3)) / (T.gather - T.board - 1.3))
        board = (C + np.array([-0.45 + 0.03 * b, 0.95 - 0.15 * b, -2.7 + 0.22 * b], np.float32),   # along the view only
                 C + np.array([-0.12, 0.35 - 0.05 * b, 0.2], np.float32), 2.65 - 0.2 * b, 0.03)
        # the pen: an arc across her while she is drawn, coming closer
        p = in_out((t - T.gather) / (T.mtn - T.gather))
        pos, tgt = front_pose(math.radians(-12 + 24 * p), 3.3 - 0.4 * p, 1.0 + 0.15 * p)
        pen = (pos, tgt, 3.3 - 0.4 * p, 0.05)
        # the end: low as the mountains rise, craning up to face her square as she looks back, then a slow push
        f = in_out((t - T.mtn) / (T.turn0 - T.mtn))
        g = in_out((t - T.turn0) / (T.held - T.turn0))
        pos, tgt = front_pose(math.radians(-8) * (1 - f), 3.4 - 0.35 * g, 0.75 + 0.3 * f)
        fin = (pos, tgt, 3.4 - 0.35 * g, 0.05)
        pose = orbit
        pose = blend(pose, board, in_out((t - T.board) / 1.6))
        pose = blend(pose, pen, in_out((t - (T.gather - 0.4)) / 1.1))
        pose = blend(pose, fin, in_out((t - T.mtn) / 1.2))
    pos, tgt, focus, ap = pose
    still = smooth((t - T.colour0) / 0.4) * (1 - smooth((t - T.held) / 0.5))   # steady while she is in colour
    pos = np.asarray(pos) + L3.handheld(t) * (1 - 0.8 * still)
    return L3.Camera(tuple(pos), tuple(tgt), fov=38, focus=float(focus), aperture=float(ap))


# ---------------------------------------------------------------- the storm, the cocoon, the board
def storm_at(t):
    """Positions (N_STORM, 3) of the storm's particles at t, and how far each has settled into the cocoon (N, 1)."""
    S, T = static(), times()
    tl = t - T0
    rad = S.r * out_cubic(tl / 0.6) * (1 - 0.22 * kit.since(t, T0 + 0.6, 3.0))
    ang = S.phi + S.omega * tl
    y = S.h + 0.1 * torch.sin(1.7 * tl + S.phi)
    c = L3.t32(C)
    vortex = torch.stack([c[0] + rad * torch.cos(ang), y, c[2] + rad * torch.sin(ang)], 1)
    e = ((t - S.band_t) / 0.7).clamp(0, 1)
    e = torch.where(e < 0.5, 4 * e ** 3, 1 - (-2 * e + 2) ** 3 / 2)[:, None]
    lon = S.phi + 1.1 * S.band_dir * tl
    rx, ry = COCOON
    shell = torch.stack([c[0] + rx * torch.cos(S.lat) * torch.cos(lon), 0.92 + ry * torch.sin(S.lat),
                         c[2] + rx * torch.cos(S.lat) * torch.sin(lon)], 1)
    pos = vortex * (1 - e) + shell * e
    if t > T.board:
        g = ((t - T.board - S.grid_delay) / 1.0).clamp(0, 1)
        g = torch.where(g < 0.5, 4 * g ** 3, 1 - (-2 * g + 2) ** 3 / 2)[:, None]
        pos = pos * (1 - g) + S.grid * g + L3.t32((0.0, 0.2, 0.0)) * torch.sin(math.pi * g)
    return pos, e


def stones(fr, t):
    S, T = static(), times()
    for k, (i, j) in enumerate(STONES):
        s = T.l13[k]
        if t < s - 0.22:
            continue
        x, z = C[0] - HALF + 2 * HALF * i / 18, C[2] - HALF + 2 * HALF * j / 18
        y = 0.04 + 0.56 * (1 - in_out((t - (s - 0.22)) / 0.22))
        rise = T.l13[6 + k % 4]                                       # the second phrase takes them back
        up = in_out((t - rise) / 1.2)
        y += 1.6 * up ** 2
        fade = 1 - up
        if fade <= 0.01:
            continue
        at = L3.t32((x, y, z))
        if k % 2 == 0:                                                # hers: a small sun
            fr.add(S.sphere + at, L3.CREAM, gain=0.9 * fade)
            fr.add(at[None], L3.CREAM, size=5.0, gain=3.0 * fade)
        else:                                                         # mine: a ring
            fr.add(S.ring + at, L3.TERRA, gain=1.6 * fade)
        a = t - s
        if 0 <= a < 0.7:                                              # where it lands, the board rings
            ang = torch.linspace(0, 2 * math.pi, 90, device=L3.DEV)
            rr = 0.03 + 0.24 * math.sqrt(a)
            ring = torch.stack([x + rr * torch.cos(ang), torch.full_like(ang, 0.006), z + rr * torch.sin(ang)], 1)
            fr.add(ring, L3.CREAM, gain=0.8 * (1 - a / 0.7) ** 2)


# ---------------------------------------------------------------- the pen
def drawn(t):
    """Fraction of the outline drawn at t."""
    S, T = static(), times()
    ts, f = S.draw_energy
    return float(np.interp(t, ts, f, left=0.0, right=1.0))


def tip(t):
    S = static()
    target = drawn(t) * float(S.path_len[-1])
    k = int(torch.searchsorted(S.path_len, L3.t32([target]))[0].clamp(max=len(S.path_len) - 1))
    return S.path[k], k


# ---------------------------------------------------------------- her, in each part
def her(t):
    """(positions, colours, gain) of her particles at t; positions None while she is the pen."""
    S, T = static(), times()
    shimmer = 0.009 * torch.sin(S.ph + 2.0 * t)[:, None] * S.swirl
    if t < T.board:                                                   # bursts, then faces us
        burst = (S.start - L3.t32(C + np.array([0, 0.9, 0], np.float32))) * 0.9 * out_cubic((t - T0) / 0.35)
        e = in_out((t - T0 - 0.2) / 0.9)
        pos = (S.start + burst) * (1 - e) + S.front * e + S.swirl * 0.3 * math.sin(math.pi * e)
        inside = 1 - 0.35 * kit.since(t, T.l12[3], 1.5)               # dimmer once the cocoon closes round her
        return pos + shimmer, S.front_rgb, (0.42 + 0.2 * e) * inside
    if t < T.gather:                                                  # kneels across the board
        e = in_out((t - T.board) / 1.2)
        src = S.front[S.kneel_from]
        pos = src * (1 - e) + S.kneel * e + L3.t32((0, 0.35, 0)) * math.sin(math.pi * e)
        return pos + shimmer, S.kneel_rgb, 0.62 * (1 - 0.35 * (1 - e))
    if t < T.draw0 + 0.1:                                             # gathers into the pen's point
        e = in_cubic((t - T.gather) / (T.draw0 + 0.1 - T.gather))
        point = tip(T.draw0)[0]
        pos = S.kneel * (1 - e) + point * e + S.swirl[: N_HER] * 0.2 * math.sin(math.pi * e)
        return pos, S.kneel_rgb, 0.62 * (1 - 0.7 * e)
    if t < T.fill:
        return None, None, 0.0
    if t < T.turn0:                                                   # fills the outline, edge first
        e = ((t - (T.fill + 1.8 * S.fill_depth)) / 0.9).clamp(0, 1)
        e = torch.where(e < 0.5, 4 * e ** 3, 1 - (-2 * e + 2) ** 3 / 2)[:, None]
        pos = S.fill_from * (1 - e) + S.front * e
        return pos + shimmer * e, S.front_rgb * e.sqrt(), 0.65
    f = in_out((t - T.turn0) / (T.turn1 - T.turn0))                   # turns, and looks back
    th = math.pi * f
    pos = turn_y(S.front[S.look_from], th) * (1 - f) + turn_y(S.look, th - math.pi) * f
    k = smooth((t - (T.colour0 - 0.25)) / (T.colour1 - T.colour0 + 0.25))
    col = S.look_rgb * (1 - k) + S.look_art * 1.15 * k
    up = in_cubic((t - T.held) / 1.3)                                 # at the very end she drifts up as light
    pos = pos + L3.t32((0, 0.9, 0)) * up + S.swirl * 0.25 * up
    return pos + shimmer, col, 0.62 * (1 - up)


def her_point(t, i, k):
    """Threads land on her: on her particles, or while she is the pen, on the line being drawn."""
    S, T = static(), times()
    cam = camera(t)
    if T.gather <= t < T.fill:
        if t < T.draw0:
            return None
        _, n = tip(t)
        p = S.path[max(0, n - 1 - (i * 37 + k * 11) % 60)][None]
    else:
        pos, _, gain = her(t)
        if pos is None or gain < 0.2 or t > T.held:
            return None
        p = pos[(i * 7919 + k * 104729) % N_HER][None]
    u, v, _ = cam.project(p)
    return float(u[0]), float(v[0])


# ---------------------------------------------------------------- the frame
def light(t):
    S, T = static(), times()
    cam = camera(t)
    fr = L3.Frame(cam)
    pos, col, gain = her(t)
    if pos is not None:
        tw = 0.8 + 0.2 * torch.sin(S.ph + 6.0 * t)[:, None]
        fr.add(pos, col * tw * gain)
        fr.add(pos, col * (0.4 * gain), size=3.0)
        fr.add(pos * L3.t32((1, -1, 1)), col * (0.18 * gain) * torch.exp(-pos[:, 1:2].clamp(min=0) / 0.9), size=3.0)

    # the rain goes on, stretched into lines while the camera rushes
    rush = math.sin(math.pi * kit.clamp01((t - T0) / 0.7))
    y = (L3.t32(prev.static().rain_base[:, 1]) - L3.t32(prev.static().rain_speed) * t) % 6.0
    base = torch.stack([L3.t32(prev.static().rain_base[:, 0]) - 0.02 * y, y, L3.t32(prev.static().rain_base[:, 2])], 1)
    rain_fade = 1 - smooth((t - T.board) / 2.0) * 0.7
    streak = torch.cat([base + L3.t32((0.004 * k, 0.018 * k, -0.25 * k * rush)) for k in range(20)])
    fr.add(streak, L3.CREAM, gain=0.30 * rain_fade * (1 + 1.5 * rush))
    fr.add(L3.t32(prev.static().grid), L3.TERRA, gain=0.05 * (1 - smooth((t - T.mtn) / 1.5)))

    # the storm, with trails; then the cocoon; then the board's grid
    if t < T.gather + 0.4:
        board_fade = 1 - smooth((t - (T.l13[6] + 0.8)) / (T.gather + 0.4 - T.l13[6] - 0.8))
        ramp = board_fade * kit.since(t, T0, 0.25)
        pos0, e = storm_at(t)
        vortex = 1 - e[:, 0]
        w = torch.where(S.hero, 2.2, 0.10) * vortex + 0.8 * (1 - vortex)   # streaks and a haze; then all of the cocoon
        fr.add(pos0, S.storm_rgb * w[:, None], gain=ramp)
        if t < T.board + 0.5:                                              # long-exposure trails behind the streaks
            hv = vortex[S.hero_idx]
            for k in range(1, 8):
                pk = storm_at(t - 0.022 * k)[0][S.hero_idx]
                fr.add(pk, S.storm_rgb[S.hero_idx] * (hv * 1.7 * (1 - k / 8))[:, None], gain=ramp)
        if t > T.board:
            stones(fr, t)

    # the pen and what it has drawn, fading as her light fills it
    if T.draw0 <= t < T.mtn + 0.5:
        n_drawn = tip(t)[1]
        if n_drawn > 1:
            fade = 1 - smooth((t - (T.fill + 2.0)) / 1.4)
            breath = 0.85 + 0.3 * kit.level("rms_vocals", t)
            fr.add(S.path[:n_drawn], S.path_rgb[:n_drawn], size=1.0, gain=1.1 * fade * breath)
            fresh = max(0, n_drawn - 25)
            fr.add(S.path[fresh:n_drawn], S.path_rgb[fresh:n_drawn], size=2.0, gain=2.0 * fade)
        if t < T.fill:
            p, _ = tip(t)
            fr.add(p[None], L3.CREAM, size=1.5, gain=9.0)
            fr.add(p[None], (1.0, 0.75, 0.55), size=12.0, gain=5.0)
            sparks(fr, t)

    # the mountains rise behind her and end, the far range first
    if T.mtn <= t < T.colour1 + 1.0:
        rise = in_out((t - T.mtn) / 1.0)
        for j, ridge in enumerate(S.mountains):
            end = T.l16[1 + (2 - j)]
            gone = in_out((t - end) / 0.7)
            if gone >= 1:
                continue
            p = ridge * L3.t32((1, rise, 1)) + L3.t32((0, 0.7, 0)) * gone ** 2
            fr.add(p, L3.CREAM, size=1.0, gain=(2.2 - 0.4 * j) * (1 - gone))
            fr.add(p * L3.t32((1, -1, 1)), L3.CREAM, size=3.0, gain=(0.5 - 0.1 * j) * (1 - gone))   # in the wet floor
    fr.add(S.dust + L3.t32((0.03, 0.02, 0.0)) * t, L3.CREAM, gain=0.5 * smooth((t - T.mtn) / 1.5))
    return fr.resolve(), cam


def sparks(fr, t):
    """Sparks thrown off the pen's point: born every 0.05 s, falling, gone in half a second."""
    pts = []
    for b in range(10):
        tb = math.floor(t / 0.05) * 0.05 - 0.05 * b
        a = t - tb
        if a < 0 or tb < times().draw0:
            continue
        p, _ = tip(tb)
        rng = np.random.default_rng(int(tb * 1000))
        v = rng.normal(0, 0.35, (4, 3))
        pts.append(p + L3.t32(v * a + np.array([0, -0.9, 0]) * a * a))
    if pts:
        fr.add(torch.cat(pts), (1.0, 0.8, 0.6), gain=0.9)


def blueprint(img, t, cam):
    """The technical drawing around her while the pen draws: a frame, her height, two notes."""
    T = times()
    a = smooth((t - T.draw0) / 0.6) * (1 - smooth((t - (T.fill + 1.5)) / 1.0))
    if a <= 0.01:
        return
    S = static()
    d = ImageDraw.Draw(img)
    col = tuple(int(c * a) for c in (196, 182, 164))
    p = drawn(t)
    lo, hi = S.path.min(0).values.cpu().numpy(), S.path.max(0).values.cpu().numpy()
    corners = L3.t32([[lo[0], lo[1], C[2]], [hi[0], lo[1], C[2]], [lo[0], hi[1], C[2]], [hi[0], hi[1], C[2]]])
    u, v, _ = cam.project(corners)
    x0, x1, y0, y1 = float(u.min()) - 26, float(u.max()) + 26, float(v.min()) - 20, float(v.max()) + 14
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line((cx, cy, cx + 22 * sx, cy), fill=col, width=1)
        d.line((cx, cy, cx, cy + 22 * sy), fill=col, width=1)
    if p > 0.35:                                                      # her height, dimensioned
        xd = x1 + 26
        d.line((xd, y0 + 20, xd, y1 - 14), fill=col, width=1)
        for yy in (y0 + 20, y1 - 14):
            d.line((xd - 6, yy, xd + 6, yy), fill=col, width=1)
        d.text((xd + 10, (y0 + y1) / 2), "h 1.70", font=S.font, fill=col, anchor="lm")
    if p > 0.6:
        d.line((x0, y0 + 60, x0 - 40, y0 + 34), fill=col, width=1)
        d.text((x0 - 44, y0 + 34), "hair  #E8742A", font=S.font, fill=col, anchor="rm")
    if p > 0.85:
        d.line((x0, (y0 + y1) / 2, x0 - 40, (y0 + y1) / 2 + 26), fill=col, width=1)
        d.text((x0 - 44, (y0 + y1) / 2 + 26), "book  · closed", font=S.font, fill=col, anchor="rm")


def colour(img, t, cam):
    """For the length of one look: her, in colour, over the light."""
    S, T = static(), times()
    k = smooth((t - T.colour0) / (T.colour1 - T.colour0)) * (1 - smooth((t - T.held) / 0.9))
    if k <= 0.01:
        return img
    (xa, ya), (xb, yb) = S.look_quad
    u, v, _ = cam.project(L3.t32([[xa, ya, C[2]], [xb, yb, C[2]]]))
    x0, y0, x1, y1 = float(u[0]), float(v[0]), float(u[1]), float(v[1])
    im = S.look_img.resize((max(1, int(x1 - x0)), max(1, int(y1 - y0))), Image.Resampling.LANCZOS)
    glow = im.getchannel("A").filter(ImageFilter.GaussianBlur(18))
    halo = Image.new("RGB", im.size, (255, 196, 150))
    base = img.copy()
    base.paste(halo, (int(x0), int(y0)), glow.point(lambda v: int(v * 0.35 * k)))
    im.putalpha(im.getchannel("A").point(lambda v: int(v * k)))
    base.paste(im, (int(x0), int(y0)), im)
    return base


def ending(img, t):
    """The cursor blinks twice and goes out, and the frame goes dark."""
    T = times()
    if t < T.held + 0.8:
        return img
    d = ImageDraw.Draw(img)
    on = (T.held + 0.9 <= t < T.held + 1.2) or (T.held + 1.5 <= t < T.held + 1.8)
    d.text((40, H - 70), ">", font=static().font, fill=(150, 144, 134))
    if on:
        d.rectangle((58, H - 70, 67, H - 51), fill=kit.TERRA)
    dark = smooth((t - (T1 - 0.45)) / 0.45)
    if dark > 0:
        img = Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), dark)
    return img


def plate(t, ghost=None):
    lit, cam = light(t)
    if ghost is not None and ghost.any():
        lit = lit + L3.t32(ghost)[None] * L3.t32(L3.CREAM)[:, None, None] * 0.9
    img = L3.to_image(lit)
    blueprint(img, t, cam)
    img = colour(img, t, cam)
    return ending(img, t)
