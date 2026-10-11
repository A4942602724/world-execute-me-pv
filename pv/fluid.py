"""Ink in water: a stable-fluids solver (Stam 1999) on the GPU, used to bake ink sequences.

bake(name, dye0, seconds, forces) runs the fluid forward from an initial dye with two channels (ink, and her orange)
and keeps every frame at 24 fps: the dye at half resolution (uint8) and the velocity at an eighth (float16), in
work/cache/fluid_<name>_{dye,vel}.npy. Played forward the ink dissolves into swirls; played backward the swirls flow
together into the exact shape they started from. The solver: semi-Lagrangian advection, vorticity confinement to
keep the curls alive, and a Jacobi pressure projection with closed walls.
"""
import env  # noqa: F401  (cache locations)

import math

import numpy as np
import torch
import torch.nn.functional as F

import kit

VH, VW = 270, 480                 # velocity grid: a quarter of 1080p, one cell = 4 px
DH, DW = 540, 960                 # dye grid: half of 1080p
SUB = 3                           # solver steps per film frame
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _base(h, w):
    ys, xs = torch.linspace(-1, 1, h, device=DEV), torch.linspace(-1, 1, w, device=DEV)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], -1)[None]


def advect(field, vel, dt, base):
    """Sample field where its contents were dt ago. vel is in velocity-grid cells per second."""
    h, w = field.shape[-2:]
    v = vel if (h, w) == (VH, VW) else F.interpolate(vel, size=(h, w), mode="bilinear", align_corners=True)
    back = base - torch.stack([v[:, 0] * dt * 2 / (VW - 1), v[:, 1] * dt * 2 / (VH - 1)], -1)
    return F.grid_sample(field, back, mode="bilinear", padding_mode="border", align_corners=True)


def _shift(a, dy, dx):
    """a moved by one cell (replicating the edge), for central differences."""
    p = F.pad(a, (1, 1, 1, 1), mode="replicate")
    return p[..., 1 + dy:1 + dy + a.shape[-2], 1 + dx:1 + dx + a.shape[-1]]


def project(vel, iters=60):
    """Make the velocity divergence-free (incompressible water), with no flow through the frame's edges."""
    u, v = vel[:, :1], vel[:, 1:]
    div = 0.5 * (_shift(u, 0, 1) - _shift(u, 0, -1) + _shift(v, 1, 0) - _shift(v, -1, 0))
    p = torch.zeros_like(div)
    for _ in range(iters):
        p = (_shift(p, 0, 1) + _shift(p, 0, -1) + _shift(p, 1, 0) + _shift(p, -1, 0) - div) / 4
    u = u - 0.5 * (_shift(p, 0, 1) - _shift(p, 0, -1))
    v = v - 0.5 * (_shift(p, 1, 0) - _shift(p, -1, 0))
    u[..., :, 0] = u[..., :, -1] = 0
    v[..., 0, :] = v[..., -1, :] = 0
    return torch.cat([u, v], 1)


def vorticity(vel, eps):
    """Vorticity confinement: push along the curls so the swirls do not smear away."""
    u, v = vel[:, :1], vel[:, 1:]
    w = 0.5 * (_shift(v, 0, 1) - _shift(v, 0, -1) - _shift(u, 1, 0) + _shift(u, -1, 0))
    a = w.abs()
    nx, ny = 0.5 * (_shift(a, 0, 1) - _shift(a, 0, -1)), 0.5 * (_shift(a, 1, 0) - _shift(a, -1, 0))
    n = torch.sqrt(nx * nx + ny * ny) + 1e-5
    return eps * torch.cat([ny / n * w, -nx / n * w], 1)


class Stir:
    """Divergence-free stirring: the curl of a sum of slowly drifting sine waves (curl noise)."""

    def __init__(self, seed, waves=14, wavelength=(50, 260)):
        g = torch.Generator().manual_seed(seed)
        self.k = [(2 * math.pi / float(torch.empty(1).uniform_(*wavelength, generator=g)),
                   float(torch.empty(1).uniform_(0, 2 * math.pi, generator=g)),
                   float(torch.empty(1).uniform_(0, 2 * math.pi, generator=g)),
                   float(torch.empty(1).uniform_(0.3, 1.2, generator=g))) for _ in range(waves)]
        ys, xs = torch.meshgrid(torch.arange(VH, device=DEV, dtype=torch.float32),
                                torch.arange(VW, device=DEV, dtype=torch.float32), indexing="ij")
        self.xs, self.ys = xs, ys

    def __call__(self, t, strength):
        """Velocity (1, 2, VH, VW) in cells per second."""
        u = torch.zeros_like(self.xs)
        v = torch.zeros_like(self.xs)
        for kk, ang, ph, om in self.k:
            kx, ky = kk * math.cos(ang), kk * math.sin(ang)
            c = torch.cos(kx * self.xs + ky * self.ys + ph + om * t)
            u += ky * c                                              # psi = sin(.); (u, v) = (dpsi/dy, -dpsi/dx)
            v -= kx * c
        scale = strength / (sum(k[0] for k in self.k) / len(self.k)) / math.sqrt(len(self.k))
        return torch.stack([u, v])[None] * scale


def burst(centre, strength, radius):
    """Outward velocity (1, 2, VH, VW) from centre (in velocity cells)."""
    ys, xs = torch.meshgrid(torch.arange(VH, device=DEV, dtype=torch.float32),
                            torch.arange(VW, device=DEV, dtype=torch.float32), indexing="ij")
    dx, dy = xs - centre[0], ys - centre[1]
    r = torch.sqrt(dx * dx + dy * dy) + 1e-3
    fall = strength * torch.exp(-(r / radius) ** 2)
    return torch.stack([dx / r * fall, dy / r * fall])[None]


def vortices(points, peak, radius, seed):
    """Velocity (1, 2, VH, VW) of Lamb-Oseen vortices at points (velocity cells), alternating spin, each peaking at
    about `peak` cells/s at its core radius: the small curls that roll ink into spirals."""
    g = np.random.default_rng(seed)
    ys, xs = torch.meshgrid(torch.arange(VH, device=DEV, dtype=torch.float32),
                            torch.arange(VW, device=DEV, dtype=torch.float32), indexing="ij")
    u, v = torch.zeros_like(xs), torch.zeros_like(xs)
    for k, (px, py) in enumerate(points):
        rc = radius * g.uniform(0.6, 1.4)
        spin = (1 if k % 2 else -1) * peak * g.uniform(0.6, 1.2) * 1.4 * rc
        dx, dy = xs - px, ys - py
        r2 = dx * dx + dy * dy + 1e-3
        tang = spin * (1 - torch.exp(-r2 / (rc * rc))) / r2
        u -= tang * dy
        v += tang * dx
    return torch.stack([u, v])[None]


def edge_points(dye0, n, seed):
    """n points on the outline of the dye's shape, in velocity cells."""
    a = np.asarray(dye0, np.float32).max(0)[::2, ::2] > 0.3
    edge = a & ~(np.roll(a, 1, 0) & np.roll(a, -1, 0) & np.roll(a, 1, 1) & np.roll(a, -1, 1))
    ys, xs = np.nonzero(edge)
    pick = np.random.default_rng(seed).choice(len(xs), size=min(n, len(xs)), replace=False)
    return list(zip(xs[pick].astype(float), ys[pick].astype(float)))


def _diffuse(dye, k):
    """Ink spreading into the water: a little Laplacian diffusion."""
    return dye + k * (_shift(dye, 0, 1) + _shift(dye, 0, -1) + _shift(dye, 1, 0) + _shift(dye, -1, 0) - 4 * dye)


def bake(name, dye0, seconds, forces, eps=3.0, fade=None, diffuse=0.06):
    """Run the fluid from dye0 (2, DH, DW float in 0..1) and keep every frame. forces(t, vel) -> acceleration.

    fade(t) -> per-step multiplier of the dye (ink thinning out), or None; diffuse: the ink's spread per step.
    Returns (dye, vel) memory-mapped arrays.
    """
    dpath, vpath = kit.CACHE / f"fluid_{name}_dye.npy", kit.CACHE / f"fluid_{name}_vel.npy"
    if dpath.exists() and vpath.exists():
        return np.load(dpath, mmap_mode="r"), np.load(vpath, mmap_mode="r")
    kit.CACHE.mkdir(parents=True, exist_ok=True)
    n = int(round(seconds * kit.FPS)) + 1
    out_d = np.zeros((n, 2, DH, DW), np.uint8)
    out_v = np.zeros((n, 2, VH // 2, VW // 2), np.float16)
    dye = torch.tensor(np.asarray(dye0, np.float32)[None], device=DEV)
    vel = torch.zeros((1, 2, VH, VW), device=DEV)
    bv, bd = _base(VH, VW), _base(DH, DW)
    dt = 1 / (kit.FPS * SUB)
    with torch.no_grad():
        for f in range(n):
            out_d[f] = (dye[0].clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()
            out_v[f] = F.avg_pool2d(vel, 2)[0].half().cpu().numpy()
            for s in range(SUB):
                t = (f * SUB + s) * dt
                vel = vel + (forces(t, vel) + vorticity(vel, eps)) * dt
                vel = project(advect(vel, vel, dt, bv))
                dye = _diffuse(advect(dye, vel, dt, bd), diffuse)
                if fade is not None:
                    dye = dye * fade(t)
    np.save(dpath, out_d)
    np.save(vpath, out_v)
    return np.load(dpath, mmap_mode="r"), np.load(vpath, mmap_mode="r")


def figure_dye(name, height, cx, feet):
    """Initial dye of her silhouette (ink, orange) placed with its centre at cx and its feet at `feet` (1080p px)."""
    ink, hair = kit.figure(name, height)
    rows = np.flatnonzero((ink + hair).max(1) > 0.3)
    x0, y0 = cx - ink.shape[1] / 2, feet - rows[-1]
    full = np.stack([kit.place(ink / 0.92, x0, y0), kit.place(hair, x0, y0)])
    return np.clip(full[:, ::2, ::2] * 0.5 + full[:, 1::2, 1::2] * 0.5, 0, 1).astype(np.float32)
