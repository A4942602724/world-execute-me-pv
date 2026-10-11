"""Character poses from the reference art, through the image endpoint the user authorised (OpenAI-compatible).

    python pv/imagegen.py models                           models the key can use (free)
    python pv/imagegen.py pose <name> "<pose description>"  -> assets/poses/<name>.png and <name>.json
    python pv/imagegen.py scene <name> "<description>"      the same, for a free description (other figures,
                                                            her as a child): only the framing rules are added
    python pv/imagegen.py pose-hq <name> "<pose description>"  highest quality, largest size the endpoint takes,
                                                            on a flat chroma-key green for a clean matte
    python pv/imagegen.py sheet <name> "<pose description>"    a four-view turnaround of the pose (front, her
                                                            right side, back, her left side) on green, for 3D
    python pv/imagegen.py modelsheet <name> "<pose>"           the same as a 3D modeling reference sheet: A-pose
                                                            friendly, flat shadowless albedo, transparent ground

The endpoint, key and model are read from input/shouquanapi.txt at run time. The key is never printed, logged or
written anywhere else. Every pose is an edit of input/character_claude.jpg, so she stays the same character.
"""
import env

import base64
import io
import json
import re
import sys
import time

import requests
from PIL import Image

CFG = env.ROOT / "input" / "shouquanapi.txt"
REF = env.ROOT / "input" / "character_claude.jpg"
OUT = env.ROOT / "assets" / "poses"

CHARACTER = ("the same character as in the reference image, unchanged: same face, very long straight orange hair, "
             "orange sunburst-shaped flower with a black ribbon in her hair, white blouse with ruffled collar and black "
             "cravat, black corset belt with orange tassels, long layered white skirt with black and orange panels and "
             "sunburst flower ornaments, white lace-up heeled boots")
FRAMING = ("Full body, the whole figure inside the frame with a clear margin on every side. Same clean anime "
           "illustration style and line art as the reference. Plain pure white background, no ground, no shadow, "
           "no other people or objects unless described, no text.")
GREEN = ("Full body, the whole figure inside the frame with a clear margin on every side. Same clean anime "
         "illustration style and line art as the reference, highly detailed, finest quality. Background: one flat, "
         "solid, pure chroma-key green (#00FF00) filling the whole canvas, perfectly uniform, no gradient, no texture, "
         "no ground, no shadow, no light spill on the figure, no green anywhere on the character. No other people or "
         "objects unless described, no text.")


SHEET = ("Character turnaround reference sheet for 3D modeling. Four full-body views of her side by side in one row, "
         "left to right: (1) front view, facing the viewer; (2) side view, she has turned 90 degrees and faces the "
         "right edge of the image; (3) back view; (4) side view, she faces the left edge of the image. Exactly the "
         "same pose, outfit and proportions in every view, as if she stands on a turntable that rotates 90 degrees "
         "between views. Orthographic projection, no perspective distortion, camera at chest height; all four figures "
         "at exactly the same scale and height, feet on one shared baseline, evenly spaced with a clear gap between "
         "them, nothing overlapping. Even, flat, neutral studio lighting, no cast shadows, no rim light. Same clean "
         "anime illustration style and line art as the reference, highly detailed, finest quality, every costume "
         "detail drawn consistently from all sides. Background: one flat, solid, pure chroma-key green (#00FF00) "
         "filling the whole canvas, perfectly uniform, no gradient, no ground, no green anywhere on the character. "
         "No text, no labels, no frames, no divider lines, no other people or objects unless described.")


MODELSHEET = ("Professional 3D modeling reference sheet (character model sheet / turnaround for a 3D modeler), "
              "for image-to-3D reconstruction. Four full-body orthographic views of the same character side by side in "
              "one row, left to right: FRONT view, LEFT side view (she faces the left edge of the image), BACK view, "
              "RIGHT side view (she faces the right edge of the image). Strict orthographic projection, no perspective, "
              "camera at waist height; every view exactly the same height and scale, feet on one shared baseline, clear "
              "empty space between the views, nothing overlapping, nothing cropped. Identical proportions, costume, "
              "hair and accessories in every view, every detail placed consistently from all sides (as on a real model "
              "sheet), asymmetric details kept on the correct side. Flat, even, shadowless studio lighting with no cast "
              "shadows, no rim light, no ambient occlusion, no dramatic shading: clean albedo-like colours so the views "
              "can be used directly as textures. Crisp clean line art, highly detailed, finest quality, same anime "
              "illustration style as the reference. Neutral calm expression, mouth closed, eyes open looking straight "
              "ahead. Fully transparent background (or, if transparency is impossible, one flat uniform light grey), "
              "no ground, no floor shadow, no text, no labels, no arrows, no frames, no grid, no other objects.")


def config():
    text = CFG.read_text(encoding="utf-8")
    key = re.search(r"sk-[A-Za-z0-9_\-]+", text).group(0)
    host = re.search(r"https?://[^/\s]+", text).group(0)
    model = re.search(r"模型为\s*(\S+)", text).group(1)
    return f"{host}/v1", key, model


def models():
    base, key, model = config()
    r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=60)
    print("GET /models ->", r.status_code)
    r.raise_for_status()
    ids = sorted(m["id"] for m in r.json().get("data", []))
    print(f"{len(ids)} models; image-like:", [i for i in ids if "image" in i or "dall" in i])
    print(f"configured model {model!r}:", "available" if model in ids else "NOT listed")


def pose(name, description, free=False, hq=False, sheet=False):
    base, key, model = config()
    OUT.mkdir(parents=True, exist_ok=True)
    ref = io.BytesIO()
    Image.open(REF).convert("RGB").save(ref, "PNG")
    framing = MODELSHEET if sheet == "model" else SHEET if sheet else GREEN if hq else FRAMING
    prompt = f"{description} {framing}" if free else f"Draw {CHARACTER}. Pose: {description}. {framing}"
    t0 = time.time()
    tries = [("4096x2048", "high"), ("3840x2160", "high"), ("3072x2048", "high"), ("1536x1024", "high")] if sheet \
        else [("2048x3072", "high"), ("1024x1536", "high")] if hq else [("1024x1536", None)]
    for size, quality in tries:                          # the largest size first; fall back if the endpoint refuses it
        data = {"model": model, "prompt": prompt, "size": size, "n": "1"}
        if quality:
            data["quality"] = quality
        if sheet == "model":
            data["background"] = "transparent"
            data["output_format"] = "png"
        r = requests.post(f"{base}/images/edits", headers={"Authorization": f"Bearer {key}"},
                          files={"image": ("reference.png", ref.getvalue(), "image/png")}, data=data, timeout=900)
        print(f"POST /images/edits {size} quality={quality} -> {r.status_code} in {time.time() - t0:.0f} s")
        if r.status_code == 200:
            break
        print(r.text[:400])
    if r.status_code != 200:
        sys.exit(1)
    item = r.json()["data"][0]
    raw = base64.b64decode(item["b64_json"]) if item.get("b64_json") else requests.get(item["url"], timeout=120).content
    Image.open(io.BytesIO(raw)).save(OUT / f"{name}.png")
    meta = dict(name=name, model=model, description=description, prompt=prompt, seconds=round(time.time() - t0),
                usage=r.json().get("usage"))
    (OUT / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok", OUT / f"{name}.png")


if __name__ == "__main__":
    {"models": lambda: models(), "pose": lambda: pose(sys.argv[2], sys.argv[3]),
     "scene": lambda: pose(sys.argv[2], sys.argv[3], free=True),
     "pose-hq": lambda: pose(sys.argv[2], sys.argv[3], hq=True),
     "sheet": lambda: pose(sys.argv[2], sys.argv[3], sheet=True),
     "modelsheet": lambda: pose(sys.argv[2], sys.argv[3], sheet="model")}[sys.argv[1]]()
