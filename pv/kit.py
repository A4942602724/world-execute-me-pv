"""Shared pieces of the renderer: canvas, song data, ink-on-paper layers, her silhouette, vertical type, debug overlay.

Every frame is a pure function of song time t. A scene layer is an ink density (0 = bare paper, 1 = full ink) plus an
ink colour; layers are multiplied onto the paper, so ink only ever darkens it, as on rice paper. The one colour in
this ink world is hers: the orange of her hair, laid on as a watercolour wash.
"""
import env

import bisect
import json
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

W, H, FPS = 1920, 1080, 24
WORK = env.ROOT / "work"
CACHE = WORK / "cache"
ASSETS = env.ROOT / "assets"
F_KAI = "C:/Windows/Fonts/kaiu.ttf"        # DFKai-SB: covers every traditional character in the lyrics
F_SONG = "C:/Windows/Fonts/STZHONGS.TTF"    # STZhongsong: a book serif for the lyrics typed in the input box
F_MONO = "C:/Windows/Fonts/consola.ttf"

PAPER = (241, 235, 221)
INK = (22, 19, 19)
INK_FAR = (96, 104, 114)
INK_MID = (58, 60, 66)
HAIR = (205, 98, 42)
TERRA = (217, 119, 87)                      # the cursor and the threads of attention
GREY = (146, 141, 132)

# what she is made of: code tokens, never lyrics
TOKENS = ("def", "self", "attn", "softmax(x)", "token", "<eos>", "return", "memory", "0x7f", "yield", "you", "her",
          "while", "True", "{", "}", "[]", "=", "+=", "->", "//", "0", "1", "01", "10", "async", "await", "embed",
          "kv_cache", "logits", "temp=0.7", "seed", "...", ";", "#", "if", "else", "for", "in", "remember()",
          "rain", "pond", "go", "breath", "dream", "listen", "stay", "again", "p=0.98", "</s>", "<s>", "\\n")

AFFINE = Image.Transform.AFFINE
BICUBIC = Image.Resampling.BICUBIC


# ---------------------------------------------------------------- easing
def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    return 1 - (1 - clamp01(x)) ** 3


def since(t, t0, dur):
    return clamp01((t - t0) / dur)


# ---------------------------------------------------------------- song data
@lru_cache(None)
def lines():
    """Lyric lines with per-character times ("s", "e"), from work/lyrics_timing.json, with late starts corrected.

    MMS_FA sometimes starts a character long after the previous one ends although the voice has already moved on:
    when that gap is over 0.35 s and a vocal onset falls within 0.15 s of the previous character's end, the
    character starts at that onset instead (the original start is kept as "s_mms").
    """
    data = json.loads((WORK / "lyrics_timing.json").read_text(encoding="utf-8"))["lines"]
    onsets = np.array(json.loads((WORK / "analysis.json").read_text(encoding="utf-8"))["onsets"]["vocals"])
    for ln in data:
        cs = ln["chars"]
        for k in range(1, len(cs)):
            prev_end, s = cs[k - 1]["e"], cs[k]["s"]
            near = onsets[(onsets >= prev_end - 0.05) & (onsets <= prev_end + 0.15)]
            if s - prev_end > 0.35 and len(near):
                cs[k]["s_mms"], cs[k]["s"] = s, round(float(near[0]), 3)
        ln["start"] = cs[0]["s"]
    return data


@lru_cache(None)
def analysis():
    return json.loads((WORK / "analysis.json").read_text(encoding="utf-8"))


@lru_cache(None)
def _envelopes():
    e = np.load(WORK / "envelopes.npz")
    return float(e["fps"]), {k: e[k] / (np.percentile(e[k], 99) + 1e-9) for k in e.files if k != "fps"}


def level(name, t):
    """An envelope (rms_vocals, onset_kick, ...) at time t, about 1.0 at its loud end."""
    fps, env_ = _envelopes()
    a, x = env_[name], t * fps
    i = int(x)
    if i < 0 or i + 1 >= len(a):
        return 0.0
    return float(a[i] + (a[i + 1] - a[i]) * (x - i))


@lru_cache(None)
def hits(name="kick", floor=0.25):
    """(time, strength) of the onsets of one stem, strength = its onset envelope there."""
    return tuple((tk, s) for tk in analysis()["onsets"][name] if (s := level(f"onset_{name}", tk)) >= floor)


def pulse(t, events, decay=0.12):
    """Sum of exponentially decaying spikes at the events before t."""
    times = [e[0] for e in events]
    i = bisect.bisect_right(times, t)
    out = 0.0
    for te, s in events[max(0, i - 8): i]:
        out += s * math.exp(-(t - te) / decay)
    return out


@lru_cache(None)
def note_ends():
    """(line number, char index) -> when that character's note really ends.

    MMS_FA spans stop soon after a syllable starts, so they miss held notes. Inside a phrase a note lasts until the
    next character starts; before a pause (a space in the lyric, or the end of the line) it lasts until the vocal
    stem decays below a third of its level just after the syllable starts.
    """
    fps, env_ = _envelopes()
    v = env_["rms_vocals"]
    out, L = {}, lines()
    for j, ln in enumerate(L):
        cs = ln["chars"]
        breaks, k = set(), 0
        for ch in ln["text"]:
            if ch.isspace():
                breaks.add(k - 1)
            else:
                k += 1
        for k, c in enumerate(cs):
            nxt = cs[k + 1]["s"] if k + 1 < len(cs) else (L[j + 1]["chars"][0]["s"] if j + 1 < len(L) else c["s"] + 3)
            if k + 1 < len(cs) and k not in breaks:
                out[(ln["i"], k)] = nxt
                continue
            i0 = int((c["s"] + 0.08) * fps)
            peak = float(v[i0:i0 + int(0.3 * fps)].max())
            stop = int((nxt - 0.05) * fps)
            below = np.flatnonzero(v[i0:stop] < peak / 3)
            end = (i0 + below[0]) / fps if len(below) else nxt - 0.05
            out[(ln["i"], k)] = max(c["e"], min(end, nxt - 0.05))
    return out


def line_at(t):
    """(line, character index) being sung at t, or the last ones sung; (None, None) before the first."""
    cur = None, None
    for ln in lines():
        for k, c in enumerate(ln["chars"]):
            if c["s"] <= t:
                cur = ln, k
    return cur


# ---------------------------------------------------------------- noise and the page
def fbm1d(n, rng, freq, octaves=5):
    out, amp = np.zeros(n), 1.0
    x = np.arange(n, dtype=np.float64)
    for _ in range(octaves):
        grid = rng.standard_normal(int(n * freq) + 3)
        xs = x * freq + rng.uniform(0, 1)
        i = xs.astype(int)
        f = xs - i
        f = f * f * (3 - 2 * f)
        out += amp * (grid[i] * (1 - f) + grid[i + 1] * f)
        amp, freq = amp * 0.5, freq * 2
    return (out - out.min()) / (np.ptp(out) + 1e-9)


def _cached(name, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npy"
    if path.exists():
        return np.load(path)
    arr = build()
    np.save(path, arr)
    return arr


@lru_cache(None)
def ink_grain():
    """Multiplier around 1.0 for every ink density: the fibres of the paper show through the wash."""
    def build():
        rng = np.random.default_rng(11)
        g = ndimage.gaussian_filter(rng.standard_normal((H, W)), 1.2)
        m = ndimage.gaussian_filter(rng.standard_normal((H, W)), 30)
        return (1 + 0.07 * g / g.std() + 0.06 * m / m.std()).astype(np.float32)
    return _cached("ink_grain", build)


@lru_cache(None)
def soft_grain():
    """Only the slow mottling of ink_grain: for her, where fine speckle would read as dirt."""
    def build():
        m = ndimage.gaussian_filter(np.random.default_rng(12).standard_normal((H, W)), 30)
        return (1 + 0.05 * m / m.std()).astype(np.float32)
    return _cached("soft_grain", build)


@lru_cache(None)
def paper():
    def build():
        rng = np.random.default_rng(7)
        mott = ndimage.gaussian_filter(rng.standard_normal((H, W)), 90)
        grain = ndimage.gaussian_filter(rng.standard_normal((H, W)), 0.8)
        fib = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(fib)
        for _ in range(2600):
            x, y, a, n = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0, math.pi), rng.uniform(8, 40)
            pts = [(x + math.cos(a) * s + rng.normal(0, 1.2), y + math.sin(a) * s + rng.normal(0, 1.2))
                   for s in np.linspace(-n / 2, n / 2, 6)]
            d.line(pts, fill=int(rng.uniform(40, 120)), width=1)
        fib = ndimage.gaussian_filter(np.asarray(fib, np.float32) / 255, 0.6)
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
        lum = (1 + 0.025 * mott / mott.std() + 0.012 * grain / grain.std() - 0.035 * fib) \
            * (1 - 0.12 * np.clip(r - 0.35, 0, None) ** 1.5)
        return np.clip(np.array(PAPER, np.float32)[None, None] * lum[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(_cached("paper", build))


def mountains(name, width, base_y, amp, fall, seed, freq=1 / 420, peak=1.6, edge=0.35, tex=0.3):
    """Ink density of a mountain range: dark along the ridge, fading downward into mist."""
    def build():
        rng = np.random.default_rng(seed)
        ridge = base_y - amp * fbm1d(width, rng, freq) ** peak
        below = np.arange(H, dtype=np.float32)[:, None] - ridge[None, :].astype(np.float32)
        dens = np.where(below > 0, np.exp(-np.maximum(below, 0) / fall), 0)
        dens += edge * np.exp(-np.abs(below) / 2.5) * (below > -2)
        soft = np.clip((below + 1.5) / 3.0, 0, 1)
        t = ndimage.gaussian_filter(rng.standard_normal((H, width)), (7, 1.5))
        return np.clip(dens * soft * (1 + tex * t / t.std()), 0, 1).astype(np.float32)
    return _cached(name, build)


RAMP = " .,:;-=+*#%@"


@lru_cache(None)
def mono_glyph(c, size):
    font = ImageFont.truetype(F_MONO, size)
    im = Image.new("L", (int(size * 0.75) + 2, int(size * 1.25) + 2), 0)
    ImageDraw.Draw(im).text((1, 1), c, font=font, fill=255)
    return im


def ascii_mist(name, density, lo=0.08, hi=0.45, cell=(8, 14), size=11):
    """Where an ink wash thins into mist it turns into characters: the foot of a mountain is printed, not painted.

    lo and hi are fractions of the layer's own 99th-percentile density: below hi the wash gives way to glyphs.
    """
    def build():
        top = float(np.percentile(density[density > 0.01], 99))
        a, b = lo * top, hi * top
        h, w = density.shape
        cw, ch = cell
        gh, gw = h // ch, w // cw
        d = density[: gh * ch, : gw * cw].reshape(gh, ch, gw, cw).mean((1, 3))
        out = Image.new("L", (w, h), 0)
        for gy, gx in zip(*np.nonzero((d > a) & (d < b * 1.3))):
            level = clamp01((d[gy, gx] - a) / (b - a))
            c = RAMP[1 + min(len(RAMP) - 2, int(level * (len(RAMP) - 1)))]
            g = mono_glyph(c, size)
            out.paste(Image.new("L", g.size, int(255 * (0.35 + 0.35 * level))), (int(gx * cw), int(gy * ch)), g)
        glyphs = np.asarray(out, np.float32) / 255
        wash = np.clip((density - 0.85 * b) / (0.5 * b), 0, 1)
        wash = wash * wash * (3 - 2 * wash)                           # 0 in the mist, 1 in the body of the wash
        return (density * wash + top * glyphs * (1 - wash)).astype(np.float32)
    return _cached(name, build)


def text_texture(name, w, h, size=13, line=16, seed=0):
    """Rows of code tokens as a density map: tiled vertically, it scrolls through her like a stream of thought."""
    def build():
        rng = np.random.default_rng(seed)
        font = ImageFont.truetype(F_MONO, size)
        im = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(im)
        for y in range(0, h, line):
            x = rng.uniform(-40, 0)
            while x < w:
                tok = TOKENS[rng.integers(len(TOKENS))]
                d.text((x, y), tok, font=font, fill=int(rng.uniform(150, 255)))
                x += font.getlength(tok + " ")
        return np.asarray(im, np.float32) / 255
    return _cached(name, build)


# ---------------------------------------------------------------- compositing
def to_ink(density, colour):
    """uint8 RGB image of an ink density in one colour, on white (ready to multiply)."""
    c = np.array(colour, np.float32) / 255
    rgb = 1 - np.clip(density, 0, 1)[..., None] * (1 - c)[None, None]
    return Image.fromarray((rgb * 255 + 0.5).astype(np.uint8))


def stack(*layers):
    """Multiply several (density, colour) layers into one RGB image on white."""
    out = np.ones((H, W, 3), np.float32)
    for density, colour in layers:
        c = np.array(colour, np.float32) / 255
        out *= 1 - np.clip(density, 0, 1)[..., None] * (1 - c)[None, None]
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8))


def camera(img, zoom=1.0, pan=(0.0, 0.0), origin=(0, 0), centre=(W / 2, H / 2)):
    """The layer as the camera sees it: zoom about centre, pan in world pixels; origin = layer's (0, 0) in the world."""
    cx, cy = centre
    a = 1 / zoom
    return img.transform((W, H), AFFINE, (a, 0, cx - cx * a - pan[0] - origin[0], 0, a, cy - cy * a - pan[1] - origin[1]),
                         resample=BICUBIC, fillcolor=(255, 255, 255))


def multiply(a, b):
    return ImageChops.multiply(a, b)


# ---------------------------------------------------------------- her
@lru_cache(None)
def figure(name, height):
    """Her silhouette from assets/silhouettes/<name>.npz scaled to `height` px: (ink density, hair density) arrays.

    Ink everywhere she is, except where the art is orange (hair, hair flower, skirt panels): that becomes the wash.
    """
    L = np.load(ASSETS / "silhouettes" / f"{name}.npz")
    h0, w0 = L["alpha"].shape
    size = (round(w0 * height / h0), height)
    alpha = np.asarray(Image.fromarray((L["alpha"].astype(np.float32) * 255).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS),
                       np.float32) / 255
    hsv = np.asarray(Image.fromarray(L["colour"]).resize(size, Image.Resampling.LANCZOS).convert("HSV"), np.float32)
    orange = (hsv[..., 0] > 8) & (hsv[..., 0] < 30) & (hsv[..., 1] > 100) & (hsv[..., 2] > 100)
    # hair = the orange regions joined to her head; orange skirt panels and ornaments stay ink
    rows = np.flatnonzero(alpha.max(1) > 0.5)
    head = rows[0] + 0.22 * (rows[-1] - rows[0])
    labels, _ = ndimage.label(ndimage.binary_dilation(orange, iterations=1), structure=np.ones((3, 3)))
    keep = np.unique(labels[: int(head)][labels[: int(head)] > 0])
    orange &= np.isin(labels, keep)
    hair = np.clip(ndimage.gaussian_filter(orange.astype(np.float32), 1.2) * 1.3, 0, 1) * alpha
    bleed = np.clip(ndimage.gaussian_filter(alpha, 1.1) * 1.15, 0, 1)
    ink = bleed * (1 - hair)
    edge = np.clip(hair - ndimage.gaussian_filter(hair, 2.0), 0, 1)       # watercolour pools at its edges
    return (0.92 * ink).astype(np.float32), np.clip(0.80 * hair + 0.5 * edge, 0, 1).astype(np.float32)


def place(arr, x, y):
    """A full-frame array with `arr` placed with its top-left corner at (x, y)."""
    out = np.zeros((H, W), np.float32)
    h, w = arr.shape
    x0, y0 = int(round(x)), int(round(y))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + w), min(H, y0 + h)
    if dx1 > dx0 and dy1 > dy0:
        out[dy0:dy1, dx0:dx1] = arr[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    return out


# ---------------------------------------------------------------- lyrics, set vertically
SIZE, STEP, COL_GAP = 64, 76, 100
ANCHOR_X, TOP = 0.83 * W, 0.15 * H


@lru_cache(None)
def columns():
    """Columns of characters in reading order: a line breaks into columns at its spaces, and long runs in two."""
    cols = []
    for ln in lines():
        segs, k = [], 0
        for part in ln["text"].split():
            segs.append(ln["chars"][k:k + len(part)])
            k += len(part)
        for seg in segs:
            if len(seg) > 9:
                half = (len(seg) + 1) // 2
                cols += [(ln["i"], seg[:half]), (ln["i"], seg[half:])]
            else:
                cols.append((ln["i"], seg))
    return cols


@lru_cache(None)
def glyph(c, size=SIZE):
    im = Image.new("L", (int(size * 1.5), int(size * 1.5)), 0)
    ImageDraw.Draw(im).text((size * 0.75, size * 0.75), c, font=ImageFont.truetype(F_KAI, size), fill=255, anchor="mm")
    return im


def lyrics_layer(t, ghost=0.0, keep=3):
    """Ink density of the lyric columns: right to left like a scroll; each character inks in as it is sung."""
    cols = columns()
    starts = [seg[0]["s"] for _, seg in cols]
    k = bisect.bisect_right(starts, t + 0.12) - 1
    if k < 0:
        return None
    scroll = k - 1 + ease_out(since(t, starts[k] - 0.12, 0.4)) if k > 0 else 0.0
    cur_line = cols[k][0]
    im = Image.new("L", (W, H), 0)
    for j, (li, seg) in enumerate(cols):
        x = ANCHOR_X + (scroll - j) * COL_GAP
        if x > W + COL_GAP or x < 0 or (j > k and li != cur_line):
            continue
        fade = clamp01(1 - (x - ANCHOR_X) / (COL_GAP * keep))
        rng = np.random.default_rng(j)
        for i, c in enumerate(seg):
            p = ease_out(since(t, c["s"] - 0.03, 0.22))
            a = max(ghost if li == cur_line else 0.0, p) * fade
            if a <= 0.004:
                continue
            g = glyph(c["c"])
            if 0 < p < 1:                                           # wet ink settling into the paper
                g = g.filter(ImageFilter.GaussianBlur((1 - p) * 2.6))
            jx, jy = rng.normal(0, 1.6), rng.normal(0, 1.2)
            gx, gy = x - g.width / 2 + jx, TOP + i * STEP - g.height / 2 + jy
            im.paste(Image.new("L", g.size, int(255 * a)), (int(gx), int(gy)), g)
    return np.asarray(im, np.float32) / 255 * 0.92


# ---------------------------------------------------------------- debug overlay
@lru_cache(None)
def _mono(size):
    return ImageFont.truetype(F_MONO, size)


def debug_overlay(img, t):
    """Timecode, the sung character and a +-2 s strip of beats, kicks and characters, to check sync by eye."""
    d = ImageDraw.Draw(img)
    a = analysis()
    ln, k = line_at(t)
    d.rectangle((0, 0, W, 36), fill=(0, 0, 0))
    where = f"line {ln['i']:2d} char {k + 1:2d}" if ln else "before line 1"
    d.text((12, 7), f"t {t:7.3f}s   frame {round(t * FPS):05d}   {where}   kick {pulse(t, hits('kick')):4.2f}   "
                    f"vocal {level('rms_vocals', t):4.2f}", font=_mono(20), fill=(255, 255, 255))
    y0 = H - 78
    d.rectangle((0, y0, W, H), fill=(0, 0, 0))
    px = lambda tt: W / 2 + (tt - t) * W / 4.4
    for b in a["beats"]:
        if abs(b - t) < 2.3:
            d.line((px(b), y0 + 6, px(b), y0 + 28), fill=(110, 110, 110), width=1)
    for b in a["downbeats"]:
        if abs(b - t) < 2.3:
            d.line((px(b), y0 + 2, px(b), y0 + 32), fill=(255, 255, 255), width=2)
    for tk, s in hits("kick"):
        if abs(tk - t) < 2.3:
            d.rectangle((px(tk) - 2, y0 + 36, px(tk) + 2, y0 + 36 + 14 * min(s, 1.5)), fill=(255, 140, 0))
    for ln_ in lines():
        for i, c in enumerate(ln_["chars"]):
            if abs(c["s"] - t) < 2.3:
                d.line((px(c["s"]), y0 + 56, px(c["e"]), y0 + 56), fill=(60, 140, 255), width=6)
                d.line((px(c["s"]), y0 + 50, px(c["s"]), y0 + 74), fill=(160, 210, 255), width=2)
                d.text((px(c["s"]) + 3, y0 + 60), f"{ln_['i']}.{i + 1}", font=_mono(13), fill=(160, 210, 255))
    d.line((W / 2, y0, W / 2, H), fill=(255, 40, 40), width=2)
    return img
