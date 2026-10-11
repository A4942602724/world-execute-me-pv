"""The cover of the PV 《芊芊》: her in ink on rice paper, only her hair in orange watercolour, standing on still water.
Her hem lets go of the ink and becomes the line art of the drawing, and the lines fall into the water as code.
Nothing of her is drawn: every layer is her original art (input/character_claude.jpg), matted, filtered or cut.

    python pv/cover.py            -> out/cover/cover_16x9.png (1920x1080), cover_16x9_4k.png, cover_3x4.png
"""
import env

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import art_prep

OUT = env.ROOT / "out" / "cover"
FONTS = env.ROOT / "assets" / "fonts"
SERIF = FONTS / "notoseriftc" / "NotoSerifTC-VF.ttf"
KAI = FONTS / "lxgwwenkaitc" / "LXGWWenKaiTC-Regular.ttf"
MONO = "C:/Windows/Fonts/CascadiaMono.ttf"

PAPER = np.array([0.937, 0.910, 0.858], np.float32)
INK = np.array([0.085, 0.080, 0.078], np.float32)
ORANGE = np.array([0.890, 0.420, 0.160], np.float32)
VERMILION = np.array([0.720, 0.180, 0.140], np.float32)
TOKENS = ["0", "1", "0", "1", "{", "}", ";", "//", "<t>", "f(t)", "ink", "0x82", "8A", "v", "q.k", "1"]


# ---------------------------------------------------------------- helpers
def noise(h, w, sigma, rng, octaves=1):
    """Smooth value noise in -1..1 (sum of blurred white noise, finest octave last)."""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = max(sigma / 2 ** o, 0.6)
        k = max(1, int(s // 4))                   # blur at reduced size for big sigmas
        n = rng.standard_normal((h // k + 2, w // k + 2)).astype(np.float32)
        n = ndimage.gaussian_filter(n, s / k)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) if k > 1 else n[:h, :w]
        n /= n.std() + 1e-6
        out += amp * n
        tot += amp
        amp *= 0.5
    return np.clip(out / tot / 2.2, -1, 1)


def over(base, rgb, a):
    a = np.clip(a, 0, 1)[..., None]
    return base * (1 - a) + rgb * a


def multiply(base, rgb, a):
    """Pigment on paper: darkens, never lightens."""
    a = np.clip(a, 0, 1)[..., None]
    return base * (1 - a + a * rgb)


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def text_mask(size, items):
    """items: (xy, text, font, kwargs) drawn white on black; returns 0..1."""
    im = Image.new("L", size, 0)
    d = ImageDraw.Draw(im)
    for xy, text, font, kw in items:
        d.text(xy, text, font=font, fill=255, **kw)
    return np.asarray(im, np.float32) / 255


def font(path, px, weight=None):
    f = ImageFont.truetype(str(path), int(px))
    if weight is not None:
        f.set_variation_by_axes([weight])
    return f


def inky(mask, rng, S, bleed=1.0):
    """Ink set in type on paper: a hair of bleed, a rough edge, uneven density."""
    m = ndimage.gaussian_filter(mask, 0.9 * S * bleed)
    rough = noise(*mask.shape, 1.2 * S, rng)
    m = smooth(0.32, 0.62, m + 0.12 * rough)
    dens = 0.86 + 0.14 * noise(*mask.shape, 18 * S, rng, 2)
    return m * dens


# ---------------------------------------------------------------- her layers
def her_layers():
    z = np.load(env.ROOT / "assets" / "silhouettes" / "character_claude.npz")
    alpha = z["alpha"].astype(np.float32)
    rgb = z["colour"].astype(np.float32) / 255
    alpha, _ = art_prep.open_holes(alpha, rgb)
    solid = alpha > 0.97
    _, (iy, ix) = ndimage.distance_transform_edt(~solid, return_indices=True)
    rgb = np.where(solid[..., None], rgb, rgb[iy, ix])
    line = art_prep.line_art(rgb, alpha)
    hair = art_prep.hair_mask(rgb, alpha, line)
    ys, xs = np.nonzero(alpha > 0.5)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return dict(alpha=alpha, rgb=rgb, line=line, hair=hair, box=box)


def place(layer, art, scale, x0, y0, W, H, interp=cv2.INTER_AREA):
    """Art pixels -> canvas: art (u, v) lands at (x0 + u * scale, y0 + v * scale)."""
    M = np.float32([[scale, 0, x0], [0, scale, y0]])
    return cv2.warpAffine(layer, M, (W, H), flags=interp if scale < 1 else cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------- the page
def paper(H, W, S, rng):
    img = np.ones((H, W, 3), np.float32) * PAPER
    blot = noise(H, W, 160 * S, rng, 3)
    img *= (1 + 0.035 * blot)[..., None]
    fib = np.zeros((H, W), np.float32)
    n = int(2600 * (W * H) / (1920 * 1080))
    pts = rng.random((n, 2)) * [W, H]
    ang = rng.random(n) * np.pi
    ln = (6 + rng.random(n) * 26) * S
    fl = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(fl)
    for (x, y), a, l in zip(pts, ang, ln):
        dx, dy = np.cos(a) * l, np.sin(a) * l
        d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=int(40 + rng.random() * 60), width=max(1, int(S)))
    fib = ndimage.gaussian_filter(np.asarray(fl, np.float32) / 255, 0.6 * S)
    img *= (1 - 0.06 * fib)[..., None]
    img *= (1 + 0.025 * noise(H, W, 1.0 * S, rng))[..., None]
    return img


def mountains(H, W, S, rng, horizon, x_from, x_to, height):
    """Two ranges of distant ink mountains: dry-brush ridges washing out downward."""
    out = np.zeros((H, W), np.float32)
    xs = np.arange(W, dtype=np.float32)
    for k, (amp, dens, off) in enumerate([(1.0, 0.11, 0.0), (0.55, 0.17, 0.0)]):
        r = np.zeros(W, np.float32)
        for o in range(6):
            sig = W / (14 + 10 * k) / 2 ** o
            r += ndimage.gaussian_filter1d(rng.standard_normal(W).astype(np.float32), sig) * sig ** 0.9
        r = (r - r.min()) / (r.max() - r.min())
        r = r ** 1.6
        env_x = smooth(x_from - 0.10 * W, x_from + 0.15 * W, xs) * (1 - smooth(x_to - 0.20 * W, x_to, xs))
        top = horizon - (off * H + amp * height * (0.15 + 0.85 * r)) * env_x
        yy = np.arange(H, dtype=np.float32)[:, None]
        depth = (yy - top[None]) / (height * amp + 1)
        m = (depth > 0) * np.exp(-np.clip(depth, 0, None) * 2.6) * (yy < horizon)
        ridge = np.exp(-((yy - top[None]) / (1.6 * S)) ** 2) * 0.7
        tex = 0.7 + 0.3 * noise(H, W, 4 * S, rng, 2)
        streak = ndimage.gaussian_filter(0.8 + 0.2 * noise(H, W, 1, rng), (5 * S, 0.5))
        mist = smooth(0.0, 0.5, (horizon - yy) / (height + 1))      # mist lies on the water
        out = np.maximum(out, dens * (m * tex + ridge * env_x[None]) * streak * (yy < horizon) * (0.35 + 0.65 * mist))
    return out


# ---------------------------------------------------------------- the cover
def render(W, H, S, layout, seed=11):
    rng = np.random.default_rng(seed)
    L = her_layers()
    bx0, by0, bx1, by1 = L["box"]
    fig_h = layout["fig_h"] * H
    scale = fig_h / (by1 - by0)
    foot_y = layout["water"] * H
    cx = layout["fig_x"] * W
    x0 = cx - (bx0 + bx1) / 2 * scale
    y0 = foot_y - by1 * scale + 4 * S
    P = lambda a: place(a, None, scale, x0, y0, W, H)
    alpha, line, hair = P(L["alpha"]), P(L["line"]), P(L["hair"])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = (yy - (y0 + by0 * scale)) / fig_h                     # 0 at her head, 1 at her feet

    page = paper(H, W, S, rng)
    img = page.copy()
    # distant mountains and the water line
    mt = mountains(H, W, S, rng, foot_y, layout["mt_from"] * W, layout["mt_to"] * W, layout["mt_h"] * H)
    img = multiply(img, INK * 2.2, mt)
    shore = np.exp(-((yy - (foot_y + 0.004 * H)) / (1.1 * S)) ** 2) * 0.10
    shore *= smooth(0.0, 0.25 * W, xx) * (1 - smooth(0.75 * W, W, xx)) * (0.6 + 0.4 * noise(H, W, 40 * S, rng))
    img = multiply(img, INK, shore)

    # her: ink silhouette, the hem letting go into line art
    ragged = noise(H, W, 14 * S, rng, 4)
    let_go = smooth(0.74, 0.90, v + 0.05 * ragged)             # 0 = ink, 1 = only lines
    inner = ndimage.distance_transform_edt(alpha > 0.5)
    wash = 0.86 + 0.14 * (0.5 + 0.5 * noise(H, W, 60 * S, rng, 2))
    edge_dark = 1 + 0.20 * np.exp(-inner / (3 * S))            # pigment pools at the edge of a wash
    gran = 1 + 0.035 * noise(H, W, 1.2 * S, rng)
    ink_a = alpha * (1 - let_go) * np.clip(wash * edge_dark * gran, 0, 1) * 0.97
    hair_a = hair * (1 - let_go)
    # orange watercolour: pooled at the edges, lighter in the middle, a little granulation
    h_in = ndimage.distance_transform_edt(hair > 0.5)
    h_col = ORANGE * (0.92 + 0.10 * noise(H, W, 30 * S, rng, 2))[..., None]
    h_col = h_col * (1 - 0.18 * np.exp(-h_in / (2.5 * S)))[..., None]
    line_a = line * np.clip(0.25 + let_go, 0, 1) * (alpha > 0.02) * 0.9
    fig = np.zeros((H, W, 3), np.float32) + 1                  # what her pigment multiplies the page by
    fig = fig * (1 - ink_a[..., None]) + INK * ink_a[..., None]
    fig = fig * (1 - hair_a[..., None]) + h_col * hair_a[..., None]
    fig = fig * (1 - line_a[..., None] * (1 - INK))
    fig_a = np.clip(np.maximum.reduce([ink_a, hair_a, line_a]), 0, 1)

    # code falling from the lines of her hem into the water
    code = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(code)
    fmono = font(MONO, 11 * S)
    hem = np.nonzero((line > 0.4) & (v > 0.89) & (v < 0.94))
    if len(hem[0]):
        order = rng.permutation(len(hem[0]))
        used = []
        for i in order:
            y, x = hem[0][i], hem[1][i]
            if any(abs(x - u) < 22 * S for u in used):        # one column of falling code per gap
                continue
            used.append(x)
            if len(used) > 14:
                break
            n = 2 + rng.integers(4)
            for j in range(n):
                tok = TOKENS[rng.integers(len(TOKENS))]
                yj = y + (j + 0.6) * 15 * S
                if yj > foot_y - 0.002 * H:
                    break
                d.text((x, yj), tok, font=fmono, fill=int(255 * 0.85 * (1 - j / n)), anchor="mm")
    code = np.asarray(code, np.float32) / 255 * (yy < foot_y + 0.012 * H) * (1 - smooth(0.05, 0.4, alpha))

    # reflection: everything above the water (mountains and her), flipped, broken by ripples
    img = img * fig
    pigment = np.clip(img / page, 0, 1)
    flip_y = 2 * foot_y - yy
    ripple = (np.sin(yy / (2.2 * S) + noise(H, W, 30 * S, rng) * 3) * 3.0 * S
              * smooth(foot_y, foot_y + 0.12 * H, yy) + noise(H, W, 8 * S, rng) * 1.5 * S)
    mx, my = (xx + ripple).astype(np.float32), flip_y.astype(np.float32)
    ref = cv2.remap(pigment, mx, my, cv2.INTER_LINEAR, borderValue=(1, 1, 1))
    ref = ndimage.gaussian_filter(ref, (1.4 * S, 0.6 * S, 0))
    fade = (yy > foot_y) * np.exp(-(yy - foot_y) / (0.09 * H)) * 0.42
    gaps = ndimage.gaussian_filter(smooth(-0.4, 0.4, noise(H, W, 2 * S, rng)), (0.4 * S, 16 * S))
    k = (fade * (0.5 + 0.5 * gaps))[..., None]
    img = img * (1 - k * (1 - ref))

    # rings where she stands
    rings = np.zeros((H, W), np.float32)
    for k, r in enumerate([0.040, 0.072, 0.112, 0.160]):
        rx, ry = r * W, r * W * 0.085
        e = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - foot_y - 0.006 * H) / ry) ** 2)
        rings += np.exp(-((e - 1) * rx / (1.0 * S)) ** 2) * (0.30 - 0.06 * k)
    rings *= (yy > foot_y - 0.003 * H) * (0.5 + 0.5 * smooth(-0.2, 0.5, noise(H, W, 10 * S, rng)))
    img = multiply(img, INK, rings)

    img = multiply(img, INK * 2.5, code * 0.55)

    # a single thread, from the title to her hair
    th = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(th)
    (tx, ty), (hx, hy) = layout["thread"](W, H, x0, y0, scale)
    pts = []
    for t in np.linspace(0, 1, 200):
        x = tx + (hx - tx) * t
        y = ty + (hy - ty) * t - np.sin(t * np.pi) * 0.09 * H + np.sin(t * 9) * 0.006 * H
        pts.append((x, y))
    d.line(pts, fill=255, width=max(1, int(1.2 * S)))
    th = ndimage.gaussian_filter(np.asarray(th, np.float32) / 255, 0.5 * S)
    th *= smooth(0.0, 0.25, np.clip((xx - tx) / max(hx - tx, 1), 0, 1)) * 0.55
    img = multiply(img, ORANGE, th)

    for (px, py), r in layout["petals"](W, H, x0, y0, scale):
        r *= S
        ang = rng.random() * np.pi
        u = ((xx - px) * np.cos(ang) + (yy - py) * np.sin(ang)) / (r * 1.7)
        w = (-(xx - px) * np.sin(ang) + (yy - py) * np.cos(ang)) / r
        e = u ** 2 + (w / np.clip(1 - 0.55 * u, 0.2, None)) ** 2       # a petal: round at one end, pointed at the other
        pet = smooth(1.05, 0.85, e) * (0.75 + 0.25 * smooth(0.5, 1.0, e))   # watercolour: darker rim
        img = multiply(img, ORANGE, pet * 0.85)

    img = typeset(img, W, H, S, rng, layout)

    # vignette and grain
    r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    img *= (1 - 0.10 * smooth(0.35, 0.75, r))[..., None]
    img += 0.012 * noise(H, W, 0.7 * S, rng)[..., None]
    return np.clip(img, 0, 1)


def typeset(img, W, H, S, rng, layout):
    size = (W, H)
    # the title, set vertically
    tpx = layout["title_px"] * S
    tx, ty = layout["title_xy"](W, H)
    ft = font(SERIF, tpx, 600)
    title = text_mask(size, [((tx, ty), "芊", ft, {}), ((tx, ty + tpx * 1.08), "芊", ft, {})])
    img = multiply(img, INK, inky(title, rng, S, 1.2))
    # codepoint ruby beside it, like the captions in the film
    fr = font(MONO, 12 * S)
    ruby = text_mask(size, [((tx + tpx * 1.02, ty + tpx * (0.30 + 1.08 * i)), "U+828A", fr, {}) for i in range(2)])
    ruby_im = Image.fromarray((ruby * 255).astype(np.uint8)).rotate(0)
    img = multiply(img, INK * 3.5, np.asarray(ruby_im, np.float32) / 255 * 0.8)
    # a thin rule and the romanisation, running down beside the title
    rule = np.zeros((H, W), np.float32)
    rx = int(tx - 0.22 * tpx)
    rule[int(ty + 0.1 * tpx):int(ty + 2.05 * tpx), rx:rx + max(1, int(S))] = 0.55
    img = multiply(img, INK, rule)
    rom = Image.new("L", (int(2.2 * tpx), int(40 * S)), 0)
    ImageDraw.Draw(rom).text((0, 0), "Q I A N   Q I A N", font=font(SERIF, 15 * S, 400), fill=255)
    rom = rom.rotate(-90, expand=True)
    romm = np.zeros((H, W), np.float32)
    ox, oy = int(rx - 0.20 * tpx), int(ty + 0.12 * tpx)
    a = np.asarray(rom, np.float32) / 255
    romm[oy:oy + a.shape[0], ox:ox + a.shape[1]] = a[:H - oy, :W - ox]
    img = multiply(img, INK * 2.0, romm * 0.9)

    # the seal: 以碼作墨 (ink made of code)
    sx, sy = layout["seal_xy"](W, H, tx, ty, tpx)
    s = int(70 * S)
    seal = np.zeros((H, W), np.float32)
    box = np.zeros((s, s), np.float32)
    box[int(2 * S):-int(2 * S), int(2 * S):-int(2 * S)] = 1
    fs = font(SERIF, 24 * S, 900)
    cut = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(cut)
    for (cx, cy), ch in zip([(0.73, 0.28), (0.73, 0.72), (0.27, 0.28), (0.27, 0.72)], "以碼作墨"):
        d.text((cx * s, cy * s), ch, font=fs, fill=255, anchor="mm")
    cut = ndimage.gaussian_filter(np.asarray(cut, np.float32) / 255, 0.5 * S)
    box = np.clip(box - cut, 0, 1)
    wear = smooth(-0.55, -0.2, noise(s, s, 1.5 * S, rng, 2))
    box = ndimage.gaussian_filter(box, 0.7 * S) * wear
    seal[sy:sy + s, sx:sx + s] = box
    img = multiply(img, VERMILION / PAPER, seal * 0.92)

    # the credit, set in code
    fx, fy = layout["credit_xy"](W, H)
    f1 = font(MONO, 17 * S)
    f2 = font(MONO, 11.5 * S)
    line1 = "Claude Opus 5.5  \u00d7  Blender MCP"
    c1 = text_mask(size, [((fx, fy), line1, f1, {})])
    img = multiply(img, INK * 1.4, c1 * 0.92)
    w1 = ImageDraw.Draw(Image.new("L", (1, 1))).textlength(line1, font=f1)
    caret = np.zeros((H, W), np.float32)
    caret[int(fy + 3 * S):int(fy + 21 * S), int(fx + w1 + 6 * S):int(fx + w1 + 15 * S)] = 1
    img = multiply(img, ORANGE, caret * 0.9)
    c2 = text_mask(size, [((fx, fy - 22 * S), "// every frame = f(t)", f2, {}),
                          ((fx, fy + 30 * S), "ink \u00b7 light \u00b7 her own drawing", f2, {})])
    img = multiply(img, INK * 5.0, c2 * 0.85)
    rule = np.zeros((H, W), np.float32)
    rule[int(fy - 32 * S), int(fx):int(fx + w1 + 15 * S)] = 0.35
    img = multiply(img, INK, rule)
    return img


LAYOUTS = {
    "16x9": dict(size=(1920, 1080), fig_h=0.76, fig_x=0.665, water=0.835,
                 mt_from=0.05, mt_to=0.98, mt_h=0.17,
                 title_px=196, title_xy=lambda W, H: (0.235 * W, 0.15 * H),
                 seal_xy=lambda W, H, tx, ty, p: (int(tx + 0.05 * p), int(ty + 2.32 * p)),
                 credit_xy=lambda W, H: (0.052 * W, 0.865 * H),
                 petals=lambda W, H, x0, y0, s: [((x0 + 250 * s, y0 + 330 * s), 6.5), ((0.50 * W, 0.42 * H), 5.5),
                                                 ((0.44 * W, 0.30 * H), 4.5), ((0.385 * W, 0.50 * H), 4)],
                 thread=lambda W, H, x0, y0, s: ((0.235 * W + 230 * W / 1920, 0.31 * H),
                                                 (x0 + 330 * s, y0 + 300 * s))),
    "3x4": dict(size=(1080, 1440), fig_h=0.70, fig_x=0.64, water=0.86,
                mt_from=0.0, mt_to=1.0, mt_h=0.12,
                title_px=168, title_xy=lambda W, H: (0.10 * W, 0.10 * H),
                seal_xy=lambda W, H, tx, ty, p: (int(tx + 0.04 * p), int(ty + 2.30 * p)),
                credit_xy=lambda W, H: (0.075 * W, 0.915 * H),
                petals=lambda W, H, x0, y0, s: [((x0 + 250 * s, y0 + 330 * s), 4.5), ((0.36 * W, 0.42 * H), 3.5),
                                                ((0.30 * W, 0.50 * H), 3)],
                thread=lambda W, H, x0, y0, s: ((0.10 * W + 200 * W / 1080, 0.20 * H),
                                                (x0 + 330 * s, y0 + 300 * s))),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, lay in LAYOUTS.items():
        S = 2
        W, H = lay["size"][0] * S, lay["size"][1] * S
        img = render(W, H, S, lay)
        im = Image.fromarray((img * 255 + 0.5).astype(np.uint8))
        if name == "16x9":
            im.save(OUT / f"cover_{name}_4k.png")
        im.resize(lay["size"], Image.LANCZOS).save(OUT / f"cover_{name}.png")
        print("wrote", name)


if __name__ == "__main__":
    main()
