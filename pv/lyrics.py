"""Lyric typography, driven only by the voice (per-character times), never by the drums. Two styles:

  cinema  Each character is sampled, then arrives: a few faint code glyphs flicker in its slot, then two of the
          song's own characters (the model choosing the next token), then the serif character comes out of blur
          and drifts up into place, its Unicode codepoint typed under it in small monospace; a terminal caret
          follows the voice and blinks while a note is held. A hairline under it follows the held note. Just before the next line, the old one is pushed up and away in a blur, clearing
          the slot. The held last character of a line rises huge and pale behind the mountains.
  flow    The lines stand in a column like a poem in an editor, line numbers in the gutter: the current line in
          focus, the previous ones above it dimmer and softer, the next one below, barely there. Each character
          fills with ink from left to right across the time it is sung.

render(style, t) returns screen-space densities for the scene to multiply onto the frame: "ink" (the text),
"code" (grey: glyph flicker, line numbers), "accent" (terracotta: hairline, current line number), "ghost" (the huge
pale character, to go behind the mountains), and "voice": (time, x, y, line, char) of every character sung so far
that is on screen, for threads of attention.
"""
import env

import bisect
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import kit
from kit import H, W

SERIF = str(env.ROOT / "assets" / "fonts" / "notoseriftc" / "NotoSerifTC-VF.ttf")
WENKAI = str(env.ROOT / "assets" / "fonts" / "lxgwwenkaitc" / "LXGWWenKaiTC-Regular.ttf")   # handwritten kai, one weight
FLICK = "01{}[]<>/=+*#;:~^%&?"


def out_expo(x):
    x = kit.clamp01(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def out_quint(x):
    return 1 - (1 - kit.clamp01(x)) ** 5


def in_cubic(x):
    return kit.clamp01(x) ** 3


def in_out_cubic(x):
    x = kit.clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


@lru_cache(None)
def font(size, weight=500, path=SERIF):
    f = ImageFont.truetype(path, size)
    try:
        f.set_variation_by_axes([weight])
    except OSError:                                                  # not a variable font
        pass
    return f


@lru_cache(None)
def glyph(c, size, weight=500, path=SERIF):
    """(mask, origin = its left-baseline point inside the mask, advance)."""
    f = font(size, weight, path)
    l, t, r, b = f.getbbox(c, anchor="ls")
    pad = 6
    im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t), c, font=f, fill=255, anchor="ls")
    return im, (pad - l, pad - t), f.getlength(c)


def stamp(canvas, img, origin, x, y, alpha=1.0, blur=0.0, scale=1.0, sweep=None):
    """Max-composite a mask into canvas with its origin at (x, y), subpixel, scaled and blurred.

    sweep = (x_from, x_to, progress, base, feather): ink fills from left to right across [x_from, x_to].
    """
    if alpha <= 0.003:
        return
    ox, oy = origin
    m = int(math.ceil(3 * blur)) + 2
    X0, Y0 = int(math.floor(x - ox * scale)) - m, int(math.floor(y - oy * scale)) - m
    ow, oh = int(math.ceil(img.width * scale)) + 2 * m + 1, int(math.ceil(img.height * scale)) + 2 * m + 1
    a = 1 / scale
    out = img.transform((ow, oh), Image.Transform.AFFINE, (a, 0, (X0 - x) * a + ox, 0, a, (Y0 - y) * a + oy),
                        resample=Image.Resampling.BICUBIC)
    if blur > 0.3:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(out, np.float32) / 255 * alpha
    if sweep is not None:
        x_from, x_to, p, base, feather = sweep
        xs = (np.arange(X0, X0 + ow) - x_from) / max(1.0, x_to - x_from)
        arr = arr * (base + (1 - base) * np.clip((p * (1 + feather) - xs) / feather, 0, 1))[None, :]
    cx0, cy0, cx1, cy1 = max(0, X0), max(0, Y0), min(W, X0 + ow), min(H, Y0 + oh)
    if cx1 > cx0 and cy1 > cy0:
        region = canvas[cy0:cy1, cx0:cx1]
        np.maximum(region, arr[cy0 - Y0:cy1 - Y0, cx0 - X0:cx1 - X0], out=region)


def hairline(canvas, x0, x1, y, alpha, thick=2):
    if alpha <= 0.003 or x1 <= x0:
        return
    xa, xb = max(0, int(x0)), min(W, int(math.ceil(x1)))
    ya = int(round(y))
    if xb > xa and 0 <= ya < H - thick:
        region = canvas[ya:ya + thick, xa:xb]
        np.maximum(region, alpha, out=region)


@lru_cache(None)
def layout(size, track, weight=500, wrap_at=9, path=SERIF):
    """Per line number: [(char, x offset, row, advance)]. Long lines wrap at the space nearest their middle
    (or at the middle), and the second row is indented by one character."""
    out = {}
    for ln in kit.lines():
        text = ln["text"]
        n = len(text.replace(" ", ""))
        cut = None                                                   # index in text where row 2 starts
        if n > wrap_at:
            spaces = [i for i, ch in enumerate(text) if ch == " "]
            cut = min(spaces, key=lambda i: abs(i - len(text) / 2)) + 1 if spaces else (len(text) + 1) // 2
        items, x, row, k = [], 0.0, 0, 0
        for i, ch in enumerate(text):
            if cut is not None and i == cut:
                row, x = 1, glyph("口", size, weight, path)[2] + track * size
            if ch == " ":
                if cut is None or i != cut - 1:
                    x += 0.9 * size
                continue
            adv = glyph(ch, size, weight, path)[2]
            items.append((ln["chars"][k], x, row, adv))
            x += adv + track * size
            k += 1
        out[ln["i"]] = items
    return out


# ---------------------------------------------------------------- A: cinema
A_SIZE, A_TRACK, A_ROW = 58, 0.20, 92
A_X, A_Y = 0.555 * W, 0.205 * H


@lru_cache(None)
def candidates():
    """Every character the song has: the tokens the slot considers before it is sampled."""
    return tuple(sorted({c for ln in kit.lines() for c in ln["text"] if not c.isspace()}))


@lru_cache(None)
def ruby(text, size=13):
    """A small monospace mask (a codepoint under its character)."""
    f = ImageFont.truetype(kit.F_MONO, size)
    l, t_, r, b = f.getbbox(text)
    im = Image.new("L", (r - l + 4, b - t_ + 4), 0)
    ImageDraw.Draw(im).text((2 - l, 2 - t_), text, font=f, fill=255)
    return im


def caret(canvas, x, y, alpha):
    """A terminal caret: a thin bar the height of the type, standing on the baseline."""
    if alpha <= 0.003:
        return
    x0, y0, y1 = int(round(x)), int(round(y - 0.80 * A_SIZE)), int(round(y + 0.12 * A_SIZE))
    if 0 <= x0 < W - 3 and y0 >= 0 and y1 < H:
        region = canvas[y0:y1, x0:x0 + 3]
        np.maximum(region, alpha, out=region)


def cinema(t, path=SERIF):
    """... and the code in it: before each character lands, its slot runs through candidate tokens (a few code
    glyphs, then two of the song's own characters, faint); a caret follows the voice from character to character
    and blinks while a note is held; under each landed character its codepoint is typed in small monospace."""
    ink, code, accent, ghost = (np.zeros((H, W), np.float32) for _ in range(4))
    lines, L, voice = kit.lines(), layout(A_SIZE, A_TRACK, path=path), []
    current, exits = None, {}
    for i, ln in enumerate(lines):
        nxt = lines[i + 1]["chars"][0]["s"] if i + 1 < len(lines) else None
        if nxt is None:
            t_exit, dur = ln["end"] + 0.6, 0.6
        else:                                     # gone just as the next line arrives; never before its last word starts
            t_exit = min(ln["end"] + 0.6, max(nxt - 0.40, ln["chars"][-1]["s"] + 0.12))
            dur = min(0.45, max(0.25, nxt - t_exit + 0.10))
        exits[ln["i"]] = (t_exit, dur)
        n = len(ln["chars"])
        if ln["chars"][0]["s"] - 0.32 <= t:
            current = ln
        if t < ln["chars"][0]["s"] - 0.32 or t > t_exit + 0.015 * n + dur:
            continue
        for k, (c, dx, row, adv) in enumerate(L[ln["i"]]):
            x, y = A_X + dx, A_Y + row * A_ROW
            s, e = c["s"], max(kit.note_ends()[(ln["i"], k)], c["s"] + 0.12)
            if s - 0.30 <= t < s - 0.17:                              # the model sampling it: code glyphs ...
                g = kit.mono_glyph(FLICK[(int(t * 12) * 7 + k * 13 + ln["i"] * 5) % len(FLICK)], 22)
                ramp = math.sin(math.pi * (t - (s - 0.30)) / 0.13)
                stamp(code, g, (g.width / 2, g.height / 2), x + adv / 2, y - 0.38 * A_SIZE, 0.34 * ramp)
            elif s - 0.17 <= t < s - 0.04:                            # ... then candidate characters, in its slot
                pool = candidates()
                step = int((t - (s - 0.17)) * 12)
                cand = pool[(ln["i"] * 37 + k * 11 + step * 17) % len(pool)]
                if cand == c["c"]:
                    cand = pool[(pool.index(cand) + 5) % len(pool)]
                g, o, _ = glyph(cand, A_SIZE, 300, path)
                stamp(code, g, o, x, y, 0.20 * math.sin(math.pi * (t - (s - 0.17)) / 0.13), 1.0)
            p = (t - (s - 0.10)) / 0.55
            if p <= 0:
                continue
            ee = out_expo(p)                                          # arriving out of blur
            alpha, blur, ddy, ddx, sc = min(1.0, ee * 1.05), 9 * (1 - ee) ** 2, 22 * (1 - ee), 5 * (1 - ee), 1 + 0.06 * (1 - ee)
            u = in_out_cubic((t - (t_exit + 0.015 * k)) / dur)        # pushed up and away
            alpha, blur, ddy = alpha * (1 - u), blur + 10 * u, ddy - 48 * u
            g, o, _ = glyph(c["c"], A_SIZE, 500, path)
            stamp(ink, g, o, x + ddx, y + ddy, alpha, blur, sc)
            if t >= s + 0.04:                                         # its codepoint, typed under it
                hexs = f"{ord(c['c']):04X}"
                shown = hexs[:min(len(hexs), 1 + int((t - s - 0.04) / 0.035))]
                rm = ruby(shown)
                full = ruby(hexs)
                stamp(code, rm, (full.width / 2, 0), x + adv / 2, y + 0.30 * A_SIZE + 7 + ddy * 0.6, 0.62 * (1 - u))
            if s <= t < e + 0.35:                                     # the held note
                prog = kit.clamp01((t - s) / (e - s))
                hairline(accent, x, x + adv * prog, y + 0.30 * A_SIZE, 0.9 * (1 - kit.since(t, e, 0.35)) * (1 - u))
            if s <= t and u < 0.5:
                voice.append((s, x + adv / 2, y + 0.30 * A_SIZE + 1, ln["i"], k))
        c = ln["chars"][-1]                                           # the held last character, huge and pale
        held = kit.note_ends()[(ln["i"], n - 1)]
        if held - c["s"] >= 0.3:
            nxt_s = nxt if nxt is not None else held + 2
            a = 0.08 * kit.ease_out(kit.since(t, c["s"] - 0.05, 0.5)) * (1 - kit.smooth(kit.since(t, min(held, nxt_s - 0.3), 1.0)))
            if a > 0.002:
                sc = 1 + 0.05 * kit.since(t, c["s"], held - c["s"] + 1.0)
                g, o, adv = glyph(c["c"], 600, 300, path)
                stamp(ghost, g, o, 0.80 * W - adv * sc / 2, 0.43 * H + 0.36 * 600 * sc, a, 0, sc)
    if current is not None:                                           # line number, sliding like a counter
        label(code, accent, t, current, A_X, A_Y - 1.45 * A_SIZE)
        caret(accent, *caret_at(t, current, L[current["i"]], *exits[current["i"]]))
    return dict(ink=ink, code=code, accent=accent, ghost=ghost, voice=voice)


def caret_at(t, ln, items, t_exit, dur):
    """(x, y, alpha) of the caret on this line: after the last character sung, gliding to each new one; blinking
    (a steady second, never the drums) while nothing new comes; gone with the line."""
    landed = [k for k, it in enumerate(items) if it[0]["s"] <= t]
    if landed:
        k = landed[-1]
        c, dx, row, adv = items[k]
        x_to, y = A_X + dx + adv + 0.10 * A_SIZE, A_Y + row * A_ROW
        x_from = x_to
        if k > 0 and items[k - 1][2] == row:
            pc, pdx, _, padv = items[k - 1]
            x_from = A_X + pdx + padv + 0.10 * A_SIZE
        x = x_from + (x_to - x_from) * out_quint((t - c["s"]) / 0.09)
        last = c["s"]
    else:
        x, y = A_X - 0.16 * A_SIZE, A_Y
        last = ln["chars"][0]["s"] - 0.32
    idle = t - last
    on = 1.0 if idle < 0.4 or (idle - 0.4) % 1.0 < 0.55 else 0.0
    a = 0.85 * on * kit.clamp01((t - (ln["chars"][0]["s"] - 0.32)) / 0.12) * (1 - kit.clamp01((t - t_exit) / 0.12))
    return x, y, a


def label(code, accent, t, ln, x, y):
    lines = kit.lines()
    i = ln["i"]
    t_in = ln["chars"][0]["s"] - 0.32
    u = out_quint((t - t_in) / 0.4)
    hairline(code, x, x + 30, y - 7, 0.55)
    for num, dy, a in ((i, 10 * (1 - u), u), (i - 1, -10 * u, 1 - u)):
        if num < 1 or a <= 0.01:
            continue
        g = number_mask(f"{num:02d}")
        stamp(accent, g, (0, g.height / 2), x + 42, y - 7 + dy, 0.95 * a)
        g2 = number_mask(f"/ {len(lines)}")
        stamp(code, g2, (0, g2.height / 2), x + 42 + g.width + 6, y - 7 + dy, 0.75 * a)


@lru_cache(None)
def number_mask(text, size=16):
    f = ImageFont.truetype(kit.F_MONO, size)
    l, t, r, b = f.getbbox(text)
    im = Image.new("L", (r + 4, b + 4), 0)
    ImageDraw.Draw(im).text((2, 2), text, font=f, fill=255)
    return im


# ---------------------------------------------------------------- B: flow
B_SIZE, B_TRACK, B_ROW, B_GAP = 50, 0.12, 74, 34
B_X, B_Y = 0.56 * W, 0.215 * H
B_ALPHA = {-3: 0.0, -2: 0.0, -1: 0.30, 0: 1.0, 1: 0.12, 2: 0.0}
B_BLUR = {-3: 3.5, -2: 2.8, -1: 1.5, 0: 0.0, 1: 1.6, 2: 2.8}


def interp(table, d):
    d = max(-3.0, min(2.0, d))
    lo = math.floor(d)
    hi = min(2, lo + 1)
    f = d - lo
    return table[lo] * (1 - f) + table[hi] * f


def flow(t, path=SERIF):
    ink, code, accent, ghost = (np.zeros((H, W), np.float32) for _ in range(4))
    lines, L, voice = kit.lines(), layout(B_SIZE, B_TRACK, wrap_at=11, path=path), []
    tops, y = [], 0.0
    for ln in lines:
        tops.append(y)
        y += (max(r for _, _, r, _ in L[ln["i"]]) + 1) * B_ROW + B_GAP
    starts = [ln["chars"][0]["s"] - 0.18 for ln in lines]
    k = bisect.bisect_right(starts, t) - 1
    if k < 0:                                                         # before the first line: it waits one slot below
        scroll = tops[0] - (B_ROW + B_GAP)
    else:
        prev = tops[k - 1] if k > 0 else tops[0] - (B_ROW + B_GAP)
        scroll = prev + (tops[k] - prev) * out_quint((t - starts[k]) / 0.6)
    slot = B_ROW + B_GAP
    for j, ln in enumerate(lines):
        d = (tops[j] - scroll) / slot
        if d < -3 or d > 2:
            continue
        a_line, b_line = interp(B_ALPHA, d), interp(B_BLUR, d)
        if k < 0:
            a_line *= kit.smooth(kit.since(t, starts[0] - 1.6, 1.0))
        if a_line <= 0.004:
            continue
        y0 = B_Y + tops[j] - scroll
        for kk, (c, dx, row, adv) in enumerate(L[ln["i"]]):
            x, yb = B_X + dx, y0 + row * B_ROW
            s, e = c["s"], max(kit.note_ends()[(ln["i"], kk)], c["s"] + 0.15)
            p = kit.clamp01((t - (s - 0.03)) / (e - (s - 0.03)))
            lift = 3 * (out_quint((t - s) / 0.25) - out_quint((t - e) / 0.35))
            g, o, _ = glyph(c["c"], B_SIZE, 500, path)
            sweep = None if p >= 1 or p <= 0 else (x, x + adv, p, 0.26, 0.4)
            alpha = a_line * (1.0 if p >= 1 else 0.26 if p <= 0 else 1.0)
            stamp(ink, g, o, x, yb - lift, alpha, b_line, 1.0, sweep)
            if s <= t and abs(d) < 0.5:
                voice.append((s, x + adv / 2, yb + 0.30 * B_SIZE, ln["i"], kk))
        g = number_mask(f"{ln['i']:02d}")                            # the gutter
        is_cur = abs(d) < 0.5 and k >= 0
        stamp(accent if is_cur else code, g, (g.width, g.height / 2), B_X - 34, y0 - 0.36 * B_SIZE,
              (0.95 if is_cur else 0.7) * a_line)
    return dict(ink=ink, code=code, accent=accent, ghost=ghost, voice=voice)


def render(style, t):
    """style: cinema | flow, in Noto Serif TC; cinema_kai | flow_kai, the same in LXGW WenKai TC."""
    name, _, kai = style.partition("_")
    return {"cinema": cinema, "flow": flow}[name](t, WENKAI if kai else SERIF)
