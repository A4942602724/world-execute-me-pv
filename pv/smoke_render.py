"""Smoke test for the render chain: frame(t) drawn with PIL, a Playwright screenshot pasted in, raw RGB piped to ffmpeg.

    .venv/Scripts/python.exe pv/smoke_render.py      -> out/smoke.mp4 (2 s, 1280x720, 24 fps) and out/smoke_still.png
"""
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import env

W, H, FPS, DUR = 1280, 720, 24, 2.0
OUT = env.ROOT / "out"
MONO = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 16)
CJK = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
PANE = Image.open(OUT / "smoke_shot.png").convert("RGB")
BG, UI, ME = (4, 7, 15), (200, 214, 234), (77, 107, 254)


def frame(t: float) -> Image.Image:
    """Every frame is a pure function of t."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, W - 9, H - 9], outline=UI)
    d.text((20, 16), f"SMOKE.TEST  t={t:05.2f}s  frame={round(t * FPS):03d}", font=MONO, fill=UI)
    for i in range(40):                                   # a sine wave of characters, phase driven by t
        x = 420 + i * 20
        y = 360 + 120 * math.sin(i * 0.35 + t * math.tau)
        d.text((x, y), "01"[(i + round(t * FPS)) % 2], font=MONO, fill=ME)
    d.text((420, 640), "环境自检 · render chain ok", font=CJK, fill=UI)
    im.paste(PANE, (40, 120))
    return im


def main() -> None:
    frame(1.0).save(OUT / "smoke_still.png")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(OUT / "smoke.mp4")]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as p:
        for n in range(int(DUR * FPS)):
            p.stdin.write(frame(n / FPS).tobytes())
        p.stdin.close()
    print("ok", OUT / "smoke.mp4")


if __name__ == "__main__":
    main()
