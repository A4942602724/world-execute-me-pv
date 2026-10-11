"""Refine per-character lyric timing with CTC forced alignment (torchaudio MMS_FA) on the separated vocals.

    python pv/refine_lyrics.py      -> rewrites work/lyrics_timing.json: characters get MMS_FA times ("s", "e"),
                                       Whisper's are kept as "ws", "we"

Whisper (align_lyrics.py) finds the lines reliably but spreads the characters almost evenly inside them. Here each
line's window (Whisper's span plus a margin, never past the midpoint to a neighbouring line) is aligned again,
character by character, as toneless pinyin.
"""
import env

import json

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pypinyin import Style, lazy_pinyin
from torchaudio.pipelines import MMS_FA as BUNDLE

WORK = env.ROOT / "work"
TIMING = WORK / "lyrics_timing.json"
MARGIN = 0.6
LETTERS = set(BUNDLE.get_dict()) - {"-", "*", "'"}


def romanize(c):
    s = "".join(lazy_pinyin(c, style=Style.NORMAL, errors="ignore")).lower().replace("v", "u")   # pypinyin writes ü as v
    return "".join(ch for ch in s if ch in LETTERS) or "a"


def onset_distance(times, onsets):
    return np.array([np.min(np.abs(onsets - t)) for t in times])


def main():
    data = json.loads(TIMING.read_text(encoding="utf-8"))
    lines = data["lines"]
    wave, sr = sf.read(WORK / "stems" / "vocals.wav", dtype="float32", always_2d=True)
    wave = torchaudio.functional.resample(torch.from_numpy(wave.mean(1)), sr, BUNDLE.sample_rate)
    dur = wave.numel() / BUNDLE.sample_rate

    model = BUNDLE.get_model(with_star=False).cuda().eval()
    tokenizer, aligner = BUNDLE.get_tokenizer(), BUNDLE.get_aligner()

    for c in (c for ln in lines for c in ln["chars"]):
        c.setdefault("ws", c["s"])
        c.setdefault("we", c["e"])
    for k, ln in enumerate(lines):
        ws, we = ln["chars"][0]["ws"], ln["chars"][-1]["we"]
        lo = max(ws - MARGIN, (lines[k - 1]["chars"][-1]["we"] + ws) / 2) if k else max(0.0, ws - MARGIN)
        hi = min(we + MARGIN, (we + lines[k + 1]["chars"][0]["ws"]) / 2) if k + 1 < len(lines) else min(dur, we + MARGIN)
        seg = wave[int(lo * BUNDLE.sample_rate): int(hi * BUNDLE.sample_rate)]
        with torch.inference_mode():
            emission, _ = model(seg[None].cuda())
        spans = aligner(emission[0], tokenizer([romanize(c["c"]) for c in ln["chars"]]))
        sec = seg.numel() / BUNDLE.sample_rate / emission.size(1)
        for c, sp in zip(ln["chars"], spans):
            c["s"], c["e"] = round(lo + sp[0].start * sec, 3), round(lo + sp[-1].end * sec, 3)
            c["p"] = round(float(sum(t.score * (t.end - t.start) for t in sp) / sum(t.end - t.start for t in sp)), 3)
        ln["start"], ln["end"] = ln["chars"][0]["s"], ln["chars"][-1]["e"]

    data["method"] = "whisper lines + MMS_FA characters"
    TIMING.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    onsets = np.array(json.loads((WORK / "analysis.json").read_text(encoding="utf-8"))["onsets"]["vocals"])
    chars = [c for ln in lines for c in ln["chars"]]
    rng = np.random.default_rng(0)
    for name, key in (("whisper", "ws"), ("mms_fa", "s")):
        d = onset_distance([c[key] for c in chars], onsets)
        print(f"{name:8s} char start -> nearest vocal onset: median {np.median(d) * 1000:4.0f} ms, "
              f"p75 {np.percentile(d, 75) * 1000:4.0f} ms")
    jitter = onset_distance([c["s"] + rng.uniform(-0.4, 0.4) for c in chars], onsets)
    print(f"random   (same characters jittered +-0.4 s): median {np.median(jitter) * 1000:4.0f} ms")
    for ln in lines:   # the pause written as a space inside a line: where each method puts it
        if " " in ln["text"]:
            cut = len(ln["text"].split(" ")[0])
            a, b = ln["chars"][cut - 1], ln["chars"][cut]
            print(f"line {ln['i']:2d} mid-line pause: whisper {b['ws'] - a['we']:5.2f} s, mms_fa {b['s'] - a['e']:5.2f} s "
                  f"(at {a['e']:.2f})")


if __name__ == "__main__":
    main()
