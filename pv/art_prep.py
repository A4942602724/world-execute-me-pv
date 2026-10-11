"""Her original art (input/character_claude.jpg), prepared as layers for the 3D finale. Nothing of her is drawn:
every layer is the art itself, matted, filtered or cut.

    python pv/art_prep.py          -> assets/art/claude_pre.exr, claude_aux.exr, claude_art.npz, preview_*.png

  pre   RGB = the art's colours pre-compensated for the film's AgX curve (scene_blender.agx), so that as pure light
        in Blender they come out of the grade as the art's own sRGB values; A = the matte.
  aux   R = line art (thin dark lines of the drawing, the edges of its dark fills and the outline);
        G = when the pen reaches each line pixel (0..1; 2 = never): every line grows from the top of its stroke
            network along the lines themselves, like many pens drawing at once;
        B = hair (the orange strands that glow in her silhouette);
        A = distance to the nearest crack between glass shards, in pixels.
  npz   depth (an inflated relief from the matte, metres), the shard polygons (convex, in art pixels) and the
        points that matter (heart, eyes), and the plane: the art stands 1.69 m tall on the floor at y = 0.
"""
import env

import json

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import Voronoi, cKDTree

import scene_blender

OUT = env.ROOT / "assets" / "art"
SRC = env.ROOT / "assets" / "silhouettes" / "character_claude.npz"
HEIGHT_M = 1.69                    # the art's full height in metres (her 1.65 m plus the margins of the page)
HEART = (530, 410)                 # art pixels: her heart (the left side of her chest, the image's right)
EYES = (492, 229)                  # between her eyes
N_SHARDS = 240
APRON = [(415, 575), (580, 575), (580, 760), (565, 1110), (415, 1110)]      # art pixels: the sash's orange panel


# ---------------------------------------------------------------- colour
def agx_inverse(srgb):
    """Scene-linear values whose AgX display (scene_blender.agx) is srgb (display-referred, 0..1). Colours outside
    what AgX can show are pulled in (clipped in its log space), so the result is the closest the grade can give."""
    A, A_inv = scene_blender.AGX, scene_blender.AGX_INV
    lo, hi = -12.47393, 4.026069
    x = np.linspace(0, 1, 8192)
    x2 = x * x
    x4 = x2 * x2
    poly = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232
    poly = np.maximum.accumulate(poly)                                # monotonic, for the inverse look-up
    y = srgb @ np.linalg.inv(A_inv.T)
    y = np.clip(y, poly[0], poly[-1])
    L = np.interp(y, poly, x)
    lin_inset = 2.0 ** (L * (hi - lo) + lo)
    lin = lin_inset @ np.linalg.inv(A.T)
    return np.clip(lin, 0, None).astype(np.float32)


# ---------------------------------------------------------------- matte
def open_holes(alpha, rgb):
    """The page also shows through the loops of her hair: closed islands of pure white, ringed by the dark line of
    a strand (the white of her dress is ringed by soft grey folds instead). They are cut out of the matte."""
    c = (rgb * 255).astype(np.int16)
    paper = (c.min(-1) > 236) & ((c.max(-1) - c.min(-1)) < 14) & (alpha > 0.5)
    lab, n = ndimage.label(paper)
    dark = c.mean(-1) < 90
    holes = np.zeros_like(paper)
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < 40:
            continue
        ring = ndimage.binary_dilation(comp, iterations=4) & ~comp
        if dark[ring].mean() > 0.55:
            holes |= comp
    soft = ndimage.gaussian_filter(ndimage.binary_dilation(holes, iterations=1).astype(np.float32), 0.8)
    return alpha * (1 - soft), int(holes.sum())


# ---------------------------------------------------------------- line art
def line_art(rgb, alpha):
    """Soft line mask: thin dark strokes (black-hat), the edges of dark fills (gradient), and her outline."""
    gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    closed = ndimage.grey_closing(gray, footprint=_disk(3))
    thin = np.clip((closed - gray - 0.10) / 0.22, 0, 1)                 # dark strokes narrower than ~7 px
    dark = ndimage.gaussian_filter((gray < 0.22).astype(np.float32), 0.7)
    edge_dark = np.clip(np.hypot(*np.gradient(dark)) * 3.2, 0, 1)      # where a black fill meets a light area
    a = ndimage.gaussian_filter(alpha, 0.8)
    outline = np.clip(np.hypot(*np.gradient(a)) * 2.6, 0, 1)
    line = np.maximum(np.maximum(thin, edge_dark), outline) * (alpha > 0.02)
    lab, n = ndimage.label(line > 0.25, structure=np.ones((3, 3)))      # drop specks
    sizes = ndimage.sum(np.ones_like(line), lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.flatnonzero(sizes >= 14))
    return (line * ndimage.binary_dilation(keep, iterations=1)).astype(np.float32)


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def pen_times(line, thresh=0.25):
    """When the pen reaches each line pixel, 0..1 (2 where there is no line). Every connected stroke network starts
    at its top (later the lower it starts) and grows along its own pixels at one speed, many pens at once."""
    mask = line > thresh
    h, w = mask.shape
    idx = -np.ones((h, w), np.int64)
    ys, xs = np.nonzero(mask)
    n = len(ys)
    idx[ys, xs] = np.arange(n)
    rows, cols, wts = [], [], []
    for dy, dx, c in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.4142), (1, -1, 1.4142)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 < h) & (x2 >= 0) & (x2 < w)
        j = np.where(ok, idx[np.clip(y2, 0, h - 1), np.clip(x2, 0, w - 1)], -1)
        ok &= j >= 0
        rows.append(np.arange(n)[ok])
        cols.append(j[ok])
        wts.append(np.full(ok.sum(), c))
    lab, nc = ndimage.label(mask, structure=np.ones((3, 3)))
    comp = lab[ys, xs] - 1
    top = ndimage.minimum(ys, comp, range(nc))                              # each network's highest row
    seeds = np.array([np.flatnonzero((comp == c) & (ys == top[c]))[0] for c in range(nc)])
    speed = 1.0                                                             # pixels per unit; scaled away below
    start = (np.asarray(top, np.float64) / h) ** 1.2 * 0.55 * h            # lower networks start later
    src = n                                                                 # a virtual source before every seed
    rows.append(np.full(nc, src))
    cols.append(seeds)
    wts.append(start + 1e-3)
    r, c, wv = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    G = sparse.coo_matrix((np.r_[wv, wv[:len(wv) - nc]], (np.r_[r, c[:len(c) - nc]], np.r_[c, r[:len(r) - nc]])),
                          shape=(n + 1, n + 1)).tocsr()
    dist = csgraph.dijkstra(G, directed=True, indices=src) / speed
    d = dist[:n]
    d = d / np.percentile(d[np.isfinite(d)], 99.5)
    out = np.full((h, w), 2.0, np.float32)
    out[ys, xs] = np.clip(d, 0, 1)
    return out


# ---------------------------------------------------------------- hair
def hair_mask(rgb, alpha, line):
    """Her orange hair (and the sunburst flowers): orange where the drawing is dense with strand lines. The sash and
    the skirt's panels are the same orange but drawn as flat fills, so they stay dark."""
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-4)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue_orange = (r >= g) & (g >= b) & ((g - b) > 0.12 * (r - b)) & ((g - b) < 0.78 * (r - b))
    orange = hue_orange & (sat > 0.42) & (mx > 0.40) & (alpha > 0.5)
    strands = ndimage.uniform_filter((line > 0.25).astype(np.float32), 25)    # density of drawn lines nearby
    near_orange = ndimage.uniform_filter(orange.astype(np.float32), 25)       # mostly orange around here
    hairy = (strands > 0.13) & (near_orange > 0.30)
    hairy = ndimage.binary_closing(hairy, structure=_disk(4))
    hairy = ndimage.binary_opening(hairy, structure=_disk(3))
    lab, n = ndimage.label(hairy)
    sizes = ndimage.sum(np.ones_like(alpha), lab, range(1, n + 1))
    hairy = np.isin(lab, 1 + np.flatnonzero(sizes > 900))
    yy = np.indices(alpha.shape)[0]
    # the broad hair on her head has few lines: take the orange that hangs together with the top of her head
    soft = ndimage.binary_closing(orange, structure=_disk(2))
    lab, n = ndimage.label(soft, structure=np.ones((3, 3)))
    tops = ndimage.minimum(yy, lab, range(1, n + 1))
    sizes = ndimage.sum(np.ones_like(alpha), lab, range(1, n + 1))
    head = np.isin(lab, [i + 1 for i in range(n) if tops[i] < 0.2 * alpha.shape[0] and sizes[i] > 400])
    # the apron: a broad flat orange fill below her waist
    flat = (ndimage.uniform_filter((line > 0.25).astype(np.float32), 45) < 0.07) & orange & (yy > 540)
    flat = ndimage.binary_opening(flat, structure=_disk(5))
    apron = ndimage.binary_dilation(flat, structure=_disk(14))
    from skimage.draw import polygon                                             # the sash and its panel, by hand
    rr, cc = polygon(np.array(APRON)[:, 1], np.array(APRON)[:, 0], alpha.shape)
    apron[rr, cc] = True
    hair = orange & (hairy | head) & ~apron & (yy < 1060)                       # her hair ends above the hem
    return ndimage.gaussian_filter(hair.astype(np.float32), 0.9) * (alpha > 0.02)


# ---------------------------------------------------------------- relief and shards
def relief(alpha, depth_m=0.17):
    """An inflated relief: deepest in the middle of her body, flat at the edges and in thin strands of hair."""
    inside = alpha > 0.5
    d = ndimage.distance_transform_edt(inside)
    d = np.sqrt(d / max(d.max(), 1.0))
    d = ndimage.gaussian_filter(d, 9.0)
    return (d / d.max() * depth_m).astype(np.float32)


def shards(alpha, n=N_SHARDS, seed=7):
    """Voronoi cells over her, clipped to the page: (polygons in art pixels [(x, y), ...], site of each), and the
    distance of every pixel to its cell's nearest edge."""
    h, w = alpha.shape
    rng = np.random.default_rng(seed)
    near = ndimage.binary_dilation(alpha > 0.03, iterations=24)
    ys, xs = np.nonzero(near)
    # blue-noise sites: best of 12 random candidates, farthest from the sites so far
    sites = [np.array([xs[len(xs) // 2], ys[len(ys) // 2]], np.float64)]
    for _ in range(int(n * near.sum() / (alpha > 0.5).sum()) - 1):
        cand = rng.integers(0, len(xs), 12)
        c = np.stack([xs[cand], ys[cand]], 1).astype(np.float64)
        dmin = np.min(np.linalg.norm(c[:, None] - np.array(sites)[None], axis=2), 1)
        sites.append(c[int(np.argmax(dmin))] + rng.uniform(-0.5, 0.5, 2))
    S = np.array(sites)
    # mirror the sites across the page's edges so every cell is bounded by the page
    mir = [S, np.c_[-S[:, 0], S[:, 1]], np.c_[2 * w - S[:, 0], S[:, 1]], np.c_[S[:, 0], -S[:, 1]],
           np.c_[S[:, 0], 2 * h - S[:, 1]]]
    vor = Voronoi(np.concatenate(mir))
    polys, keep_sites = [], []
    tree = cKDTree(S)
    yy, xx = np.mgrid[0:h, 0:w]
    pix = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], 1)
    d, j = tree.query(pix, k=2)
    lab = j[:, 0].reshape(h, w)
    cover = ndimage.maximum(alpha, lab, range(len(S)))
    for i in range(len(S)):
        reg = vor.regions[vor.point_region[i]]
        if -1 in reg or not reg or cover[i] < 0.05:
            continue
        P = vor.vertices[reg]
        P[:, 0] = np.clip(P[:, 0], 0, w)
        P[:, 1] = np.clip(P[:, 1], 0, h)
        ctr = P.mean(0)
        P = P[np.argsort(np.arctan2(P[:, 1] - ctr[1], P[:, 0] - ctr[0]))]
        polys.append(P.astype(np.float32))
        keep_sites.append(S[i])
    s1, s2 = S[j[:, 0]], S[j[:, 1]]
    edge = (d[:, 1] ** 2 - d[:, 0] ** 2) / (2 * np.maximum(np.linalg.norm(s2 - s1, axis=1), 1e-6))
    return polys, np.array(keep_sites, np.float32), np.clip(edge, 0, 64).reshape(h, w).astype(np.float32)


# ---------------------------------------------------------------- code silk
ATLAS_ROWS, ATLAS_COLS, ATLAS_CELL = 32, 160, (16, 32)       # rows of text, characters per row, cell px (w, h)
TOKENS = ("memory.save(us)", "for t in song:", "she.look_back()", "attention(q, k, v)", "temperature=0.7",
          "while True:", "return her", "yield light", "0x7D55", "0x5531", "0x7E6D", "0x611B", "lambda t: t*24",
          "rain.reverse()", "cocoon.close()", "if not lost:", "softmax(x)", "tokens += 3", "{}", "[]", "=>", "!=",
          "&&", "||", "0b1011", "/* hold */", "# breathe", "fn(me)", "ink -> light", "t=47.23s", "await dawn",
          "self.dream", "assert you", "pen.trace()", "glyph[11]", "U+8DB3", "seed=0712", "print(you)")


def code_atlas():
    """Rows of glowing code for the ribbons of the cocoon (white on black, monospace, Consolas)."""
    from PIL import ImageDraw, ImageFont
    cw, ch = ATLAS_CELL
    im = Image.new("L", (ATLAS_COLS * cw, ATLAS_ROWS * ch), 0)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 26)
    rng = np.random.default_rng(12)
    for r in range(ATLAS_ROWS):
        s = ""
        while len(s) < ATLAS_COLS:
            s += TOKENS[rng.integers(len(TOKENS))] + " " * int(rng.integers(1, 3))
        for k, c in enumerate(s[:ATLAS_COLS]):
            d.text((k * cw + cw / 2, r * ch + ch / 2), c, font=f, fill=255, anchor="mm")
    return im


# ---------------------------------------------------------------- write
def write_exr(path, rgba):
    import OpenEXR
    rgba = np.ascontiguousarray(rgba.astype(np.float32))
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, {"RGBA": rgba}) as f:
        f.write(str(path))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(SRC)
    alpha = z["alpha"].astype(np.float32)
    rgb = z["colour"].astype(np.float32) / 255
    alpha, n_holes = open_holes(alpha, rgb)
    print("page showing through her hair, cut out:", n_holes, "px")
    # where the matte is soft the art's pixels are half page: take the colour of the nearest solid pixel instead
    # (her outline is a dark line, so the edge stays dark instead of a pale fringe on a dark world)
    solid = alpha > 0.97
    _, (iy, ix) = ndimage.distance_transform_edt(~solid, return_indices=True)
    rgb = np.where(solid[..., None], rgb, rgb[iy, ix])
    h, w = alpha.shape
    pre = agx_inverse(rgb)
    back = scene_blender.agx(pre)
    err = np.abs(back - rgb)[alpha > 0.5]
    print(f"AgX round trip: mean {err.mean() * 255:.2f}, 99% {np.percentile(err, 99) * 255:.1f} /255; "
          f"max linear {pre.max():.2f}")
    line = line_art(rgb, alpha)
    pen = pen_times(line)
    hair = hair_mask(rgb, alpha, line)
    depth = relief(alpha)
    polys, sites, edge = shards(alpha)
    print(f"line px {int((line > 0.25).sum())}, hair px {int((hair > 0.5).sum())}, shards {len(polys)}")
    write_exr(OUT / "claude_pre.exr", np.concatenate([pre, alpha[..., None]], -1))
    write_exr(OUT / "claude_aux.exr", np.stack([line, pen, hair, edge], -1))
    # her in warm monochrome light (S9), and the rim a light behind her would leave on her silhouette (S5 - S6)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    lo, hi = np.array([0.10, 0.055, 0.035], np.float32), np.array([0.98, 0.84, 0.62], np.float32)
    warm = agx_inverse(lo + (hi - lo) * lum[..., None] ** 0.85)
    inner = ndimage.distance_transform_edt(alpha > 0.5)
    rim = (np.exp(-inner / 5.0) * np.clip(alpha, 0, 1)).astype(np.float32)
    write_exr(OUT / "claude_warm.exr", np.concatenate([warm, rim[..., None]], -1))
    code_atlas().save(OUT / "code_atlas.png")
    meta = dict(size=[w, h], height_m=HEIGHT_M, px_m=HEIGHT_M / h, heart=HEART, eyes=EYES,
                n_shards=len(polys))
    np.savez_compressed(OUT / "claude_art.npz", depth=depth.astype(np.float16), sites=sites,
                        poly_len=np.array([len(p) for p in polys], np.int32), poly_xy=np.concatenate(polys),
                        meta=json.dumps(meta))
    (OUT / "claude_art.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # previews for looking at
    Image.fromarray((255 - line * 255).astype(np.uint8)).save(OUT / "preview_line.png")
    pv = np.where(pen < 1.5, pen, np.nan)
    col = np.zeros((h, w, 3), np.float32)
    ok = np.isfinite(pv)
    col[ok] = np.stack([pv[ok], 1 - np.abs(pv[ok] - 0.5) * 2, 1 - pv[ok]], 1)
    Image.fromarray((col * 255).astype(np.uint8)).save(OUT / "preview_pen.png")
    hv = rgb * 0.25 * alpha[..., None]
    hv = np.where(hair[..., None] > 0.3, rgb, hv)
    Image.fromarray((np.clip(hv, 0, 1) * 255).astype(np.uint8)).save(OUT / "preview_hair.png")
    Image.fromarray((depth / depth.max() * 255).astype(np.uint8)).save(OUT / "preview_depth.png")
    cr = rgb.copy()
    cr[edge < 1.2] = [1.0, 0.8, 0.2]
    cr = cr * np.maximum(alpha[..., None], 0.25)
    Image.fromarray((np.clip(cr, 0, 1) * 255).astype(np.uint8)).save(OUT / "preview_shards.png")
    print("ok", OUT)


if __name__ == "__main__":
    main()
