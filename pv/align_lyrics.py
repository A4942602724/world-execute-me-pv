"""Per-character lyric timing: align the lyric text to the separated vocals with Whisper (stable-ts).

    python pv/align_lyrics.py [--model turbo]     -> work/lyrics_timing.json

The lyric text is read from input/ at run time and never copied into the code. Whisper returns token-level
"words" for Chinese; a multi-character token's span is split evenly across its characters, then the characters are
walked against the lyric lines in order, so the result does not depend on how Whisper groups segments.
"""
import env

import argparse
import json

import stable_whisper

LYRICS = env.ROOT / "input" / "geci.txt"
VOCALS = env.ROOT / "work" / "stems" / "vocals.wav"
OUT = env.ROOT / "work" / "lyrics_timing.json"


def char_times(result):
    chars = []
    for w in result.all_words():
        text = [c for c in w.word if not c.isspace()]
        if not text:
            continue
        step = (w.end - w.start) / len(text)
        prob = getattr(w, "probability", None)
        chars += [(c, w.start + i * step, w.start + (i + 1) * step, prob) for i, c in enumerate(text)]
    return chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="turbo")
    a = ap.parse_args()
    lines = [l.strip() for l in LYRICS.read_text(encoding="utf-8").splitlines() if l.strip()]

    model = stable_whisper.load_model(a.model, device="cuda")
    result = model.align(str(VOCALS), "\n".join(lines), language="zh", original_split=True)
    chars = iter(char_times(result))

    timed = []
    for i, line in enumerate(lines):
        cs = []
        for c in line:
            if c.isspace():
                continue
            got, s, e, p = next(chars)
            assert got == c, f"line {i + 1}: aligned text diverged from the lyric file"
            cs.append(dict(c=c, s=round(s, 3), e=round(e, 3), p=None if p is None else round(p, 3)))
        timed.append(dict(i=i + 1, start=cs[0]["s"], end=cs[-1]["e"], chars=cs, text=line))
    OUT.write_text(json.dumps(dict(model=a.model, lines=timed), ensure_ascii=False, indent=1), encoding="utf-8")

    for ln in timed:   # timings only: the text stays in the file
        probs = [c["p"] for c in ln["chars"] if c["p"] is not None]
        conf = f"{sum(probs) / len(probs):.2f}" if probs else "-"
        print(f"line {ln['i']:2d}  {ln['start']:6.2f} -> {ln['end']:6.2f}  ({len(ln['chars']):2d} chars, conf {conf})")


if __name__ == "__main__":
    main()
