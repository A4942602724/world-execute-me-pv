"""Render a scene: stills, or a video with the song (and a debug twin with timecode and the sync strip).

    python pv/render.py <scene> --stills 1.0,6.2,12.5        -> out/<scene>_<t>.png
    python pv/render.py <scene> --strip 5.2,6.3,2 --crop 1000,40,1800,330
                                                              -> out/<scene>_strip.png: every 2nd frame from 5.2 s to
                                                                 6.3 s, cropped and tiled, to read motion off a still
    python pv/render.py <scene> [--t0 T] [--t1 T] [--debug]  -> out/<scene>.mp4 (+ out/<scene>_debug.mp4)

Frames are rendered in parallel worker processes and piped, in order, to ffmpeg as raw RGB.
"""
import env

import argparse
import importlib
import multiprocessing
import os
import subprocess
import time
from multiprocessing import Pool

from PIL import Image, ImageDraw

import kit
from kit import FPS, H, W

OUT = env.ROOT / "out"
SCENE = None


def song():
    return next(p for p in (env.ROOT / "input").iterdir() if p.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a"})


GPU_WORKERS = 2                                      # more CUDA contexts than this would not fit in 6 GB


def _init(name):
    """Worker setup: the first GPU_WORKERS workers render light on the GPU, the rest on the CPU."""
    global SCENE
    ident = multiprocessing.current_process()._identity
    if ident and ident[0] > GPU_WORKERS:
        os.environ["PV_DEVICE"] = "cpu"
        import torch
        torch.set_num_threads(2)
    SCENE = importlib.import_module(name)
    SCENE.static()


def _frame(job):
    n, debug = job
    t = n / FPS
    im = SCENE.frame(t)
    out = [im.tobytes()]
    if debug:
        out.append(kit.debug_overlay(im.copy(), t).tobytes())
    return out


def encoder(path, t0, t1):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-i", str(song()), "-filter_complex", f"[1:a]atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS[a]",
         "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--stills")
    ap.add_argument("--strip", help="t0,t1,every_n_frames")
    ap.add_argument("--crop", help="x0,y0,x1,y1 for --strip")
    ap.add_argument("--scale", type=float, default=1.0, help="tile scale for --strip")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="each worker holds ~1.5 GB (torch + the scenes' data)")
    ap.add_argument("--no-warm", action="store_true", help="skip building caches here (they exist on disk already)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    scene = importlib.import_module(a.scene)
    if not a.no_warm or a.stills:
        scene.static()                               # build the disk caches once, before the workers start

    if a.stills:
        for s in a.stills.split(","):
            t = float(s)
            im = scene.frame(t)
            im.save(OUT / f"{a.scene}_{t:06.2f}.png")
            kit.debug_overlay(im, t).save(OUT / f"{a.scene}_{t:06.2f}_debug.png")
            print("still", t)
        return

    if a.strip:
        s0, s1, every = a.strip.split(",")
        box = tuple(int(v) for v in a.crop.split(",")) if a.crop else (0, 0, W, H)
        ns = range(round(float(s0) * FPS), round(float(s1) * FPS) + 1, int(every))
        with Pool(a.workers, initializer=_init, initargs=(a.scene,)) as pool:
            tiles = [Image.frombytes("RGB", (W, H), b[0]).crop(box) for b in pool.map(_frame, [(n, False) for n in ns])]
        if a.scale != 1.0:
            tiles = [im.resize((int(im.width * a.scale), int(im.height * a.scale)), Image.Resampling.LANCZOS) for im in tiles]
        cols = max(1, 1800 // tiles[0].width)
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tiles[0].width, rows * (tiles[0].height + 22)), "white")
        d = ImageDraw.Draw(sheet)
        for k, (n, tile) in enumerate(zip(ns, tiles)):
            x, y = (k % cols) * tile.width, (k // cols) * (tile.height + 22)
            sheet.paste(tile, (x, y + 22))
            d.text((x + 4, y + 3), f"{n / FPS:6.3f}s  f{n}", fill=(200, 40, 40), font=kit._mono(15))
        sheet.save(OUT / f"{a.scene}_strip.png")
        print("strip", len(tiles), "frames ->", OUT / f"{a.scene}_strip.png")
        return

    t0 = scene.T0 if a.t0 is None else a.t0
    t1 = scene.T1 if a.t1 is None else a.t1
    frames = range(round(t0 * FPS), round(t1 * FPS))
    outs = [encoder(OUT / f"{a.scene}.mp4", t0, t1)] + ([encoder(OUT / f"{a.scene}_debug.mp4", t0, t1)] if a.debug else [])
    start = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.scene,)) as pool:
        for k, bufs in enumerate(pool.imap(_frame, ((n, a.debug) for n in frames), chunksize=2)):
            for p, b in zip(outs, bufs):
                p.stdin.write(b)
            if k % 48 == 0:
                print(f"frame {k}/{len(frames)}  {time.time() - start:5.1f} s", flush=True)
    for p in outs:
        p.stdin.close()
        p.wait()
    print(f"done {len(frames)} frames in {time.time() - start:.1f} s ->", OUT / f"{a.scene}.mp4")


if __name__ == "__main__":
    main()
