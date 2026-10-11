"""Her 3D figures, from the AI four-view turnaround sheets (imagegen.py sheet) - never modelled by code.

    python pv/mv.py split <name>     assets/poses/sheet_<name>.png -> assets/mv/<name>/{front,left,back,right}.png
    python pv/mv.py shape <name>     the four views -> assets/mv/<name>/shape.glb (Hunyuan3D-2mv, local GPU)
    python pv/mv.py paint <name>     the four views projected back onto the shape -> assets/mv/<name>/her.ply
                                     (vertex colours, each from the views that really see that point)

A sheet shows, left to right: front; she faces the image's left edge (we see her left side, Hunyuan3D's "left",
the front turned clockwise by 90 degrees seen from above); back; she faces the right edge ("right").
"""
import env

import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

import silhouette

POSES = env.ROOT / "assets" / "poses"
OUT = env.ROOT / "assets" / "mv"
VIEWS = ("front", "left", "back", "right")


def split(name, sheet_name=None):
    """Key the green, find the four figures by the gaps between them, crop each with the sheet's full height."""
    sheet = np.asarray(Image.open(POSES / f"{sheet_name or 'sheet_' + name}.png").convert("RGBA"))
    if sheet[..., 3].min() < 250:                                   # the generator sometimes returns real transparency
        alpha, rgb = sheet[..., 3].astype(np.float32) / 255, sheet[..., :3]
    else:
        alpha, rgb = silhouette.chroma(sheet)
    solid = alpha > 0.5
    cols = ndimage.binary_closing(solid.any(0), iterations=6)
    lab, n = ndimage.label(cols)
    spans = sorted(ndimage.find_objects(lab), key=lambda s: s[0].stop - s[0].start, reverse=True)[:4]
    spans = sorted(s[0] for s in spans)
    if len(spans) != 4:                                             # touching figures (a spread skirt, pooled hair):
        cover = ndimage.gaussian_filter1d(solid.sum(0).astype(np.float32), 6)    # cut at the thinnest columns
        w = solid.shape[1]
        cuts = [int(q * w / 4 - 0.1 * w + np.argmin(cover[int(q * w / 4 - 0.1 * w): int(q * w / 4 + 0.1 * w)]))
                for q in (1, 2, 3)]
        edges = [0] + cuts + [w]
        spans = []
        for a_, b_ in zip(edges[:-1], edges[1:]):
            c = np.flatnonzero(solid[:, a_:b_].any(0))
            spans.append(slice(a_ + c[0], a_ + c[-1] + 1))
        print("touching figures: cut at columns", cuts, "coverage there", [int(cover[c]) for c in cuts])
    rows = np.flatnonzero(solid.any(1))
    top, bottom = rows[0], rows[-1] + 1
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for view, sp in zip(VIEWS, spans):
        x0, x1 = sp.start, sp.stop
        a = alpha[top:bottom, x0:x1]
        lab2, n2 = ndimage.label(a > 0.5)                          # only her: drop bits of a neighbour
        if n2 > 1:
            sizes = ndimage.sum(a > 0.5, lab2, range(1, n2 + 1))
            a = a * ndimage.binary_dilation(lab2 == 1 + int(np.argmax(sizes)), iterations=3)
        m = 24
        img = np.zeros((bottom - top + 2 * m, x1 - x0 + 2 * m, 4), np.uint8)
        img[m:-m, m:-m, :3] = rgb[top:bottom, x0:x1]
        img[m:-m, m:-m, 3] = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
        Image.fromarray(img).save(out / f"{view}.png")
        meta[view] = dict(x0=int(x0) - m, y0=int(top) - m, w=img.shape[1], h=img.shape[0])
        print(view, img.shape[1], "x", img.shape[0])
    (out / "views.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    strip = Image.new("RGB", (sum(Image.open(out / f"{v}.png").width for v in VIEWS), bottom - top + 48), (60, 60, 70))
    x = 0
    for v in VIEWS:
        im = Image.open(out / f"{v}.png")
        strip.paste(im, (x, 0), im)
        x += im.width
    strip.resize((strip.width // 3, strip.height // 3)).save(out / "views_preview.jpg", quality=90)


TRIPO = env.ROOT / "assets" / "tripo"


def tripo_pack(name, sheet=None):
    """The four views of a modeling sheet as Tripo multi-view inputs: transparent square PNGs, same scale for all four
    (the same canvas size and the same feet line), in assets/tripo/input/<name>/ as front / left / back / right."""
    split(f"t_{name}", sheet or f"modelsheet_{name}")          # its own folder: the Hunyuan views stay untouched
    src = OUT / f"t_{name}"
    dst = TRIPO / "input" / name
    dst.mkdir(parents=True, exist_ok=True)
    ims = {v: Image.open(src / f"{v}.png").convert("RGBA") for v in VIEWS}
    side = int(max(max(im.size) for im in ims.values()) * 1.08)
    for v, im in ims.items():
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(im, ((side - im.width) // 2, side - im.height - (side - max(i.height for i in ims.values())) // 2), im)
        canvas.save(dst / f"{v}.png")
    print("tripo inputs:", dst, side, "px")


MODEL = ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv")


def load_pipeline():
    """Hunyuan3D-2mv in fp16 on the CPU, built without ever holding the 4.9 GB file and an fp32 copy in RAM at once
    (16 GB machine): the parts are created in fp16 and filled tensor by tensor from the safetensors file."""
    import os
    os.environ.setdefault("HY3DGEN_MODELS", str(env.CACHE / "hy3dgen"))
    import torch
    import yaml
    from safetensors import safe_open
    from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.pipelines import instantiate_from_config
    from hy3dgen.shapegen.utils import smart_load_model
    config_path, ckpt_path = smart_load_model(MODEL[0], subfolder=MODEL[1], use_safetensors=True, variant="fp16")
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    import torch.nn.init as init
    names = ("normal_", "uniform_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_",
             "xavier_normal_", "zeros_", "ones_", "constant_")
    saved = {n: getattr(init, n) for n in names}
    for n in names:                                            # every weight is overwritten by the file: skip the
        setattr(init, n, lambda t, *a, **k: t)                 # random init of ~2 billion parameters on the CPU
    torch.set_default_dtype(torch.float16)
    try:
        parts = {k: instantiate_from_config(cfg[k]) for k in ("model", "vae", "conditioner")}
    finally:
        torch.set_default_dtype(torch.float32)
        for n, fn in saved.items():
            setattr(init, n, fn)
    with safe_open(ckpt_path, "pt", device="cpu") as f:
        keys = list(f.keys())
        for name, module in parts.items():
            sd = {k[len(name) + 1:]: f.get_tensor(k) for k in keys if k.split(".", 1)[0] == name}
            missing, unexpected = module.load_state_dict(sd, strict=False)
            print(f"{name}: {len(sd)} tensors, missing {len(missing)}, unexpected {len(unexpected)}")
            del sd
    pipe = Hunyuan3DDiTFlowMatchingPipeline(
        vae=parts["vae"].half(), model=parts["model"].half(), conditioner=parts["conditioner"].half(),
        scheduler=instantiate_from_config(cfg["scheduler"]), image_processor=instantiate_from_config(cfg["image_processor"]),
        device="cpu", dtype=torch.float16)
    return pipe


def shape(names, steps=40, guidance=5.0, octree=384, seed=1234, faces=400_000):
    """Several poses in one run: the 4.9 GB model loads once (slow from this disk)."""
    pipe = load_pipeline()
    for name in names:
        _shape(pipe, name, steps, guidance, octree, seed, faces)


def _shape(pipe, name, steps, guidance, octree, seed, faces):
    """The four views -> a mesh, one model part on the 6 GB GPU at a time: image encoder, then the diffusion
    transformer, then the shape decoder."""
    import torch
    import trimesh
    from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer
    from hy3dgen.shapegen.pipelines import retrieve_timesteps
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    d = OUT / name
    cuda = torch.device("cuda")
    pipe.device = cuda
    views = {v: Image.open(d / f"{v}.png").convert("RGBA") for v in VIEWS}
    with torch.inference_mode():
        cond_inputs = pipe.prepare_image(views)
        image = cond_inputs.pop("image").to(cuda, torch.float16)
        pipe.conditioner.to(cuda)
        cond = pipe.encode_cond(image=image, additional_cond_inputs=cond_inputs,
                                do_classifier_free_guidance=True, dual_guidance=False)
        pipe.conditioner.to("cpu")
        torch.cuda.empty_cache()
        pipe.model.to(cuda)
        sigmas = np.linspace(0, 1, steps)
        timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, cuda, sigmas=sigmas)
        gen = torch.Generator(device=cuda).manual_seed(seed)
        latents = pipe.prepare_latents(1, torch.float16, cuda, gen)
        for i, t in enumerate(timesteps):
            x = torch.cat([latents] * 2)
            ts = t.expand(x.shape[0]).to(latents.dtype) / pipe.scheduler.config.num_train_timesteps
            eps = pipe.model(x, ts, cond, guidance=None)
            c, u = eps.chunk(2)
            latents = pipe.scheduler.step(u + guidance * (c - u), t, latents).prev_sample
            if i % 10 == 0:
                print(f"step {i}/{steps}  vram {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
        pipe.model.to("cpu")
        torch.cuda.empty_cache()
        pipe.vae.to(cuda)
        lat = pipe.vae(1.0 / pipe.vae.scale_factor * latents)
        out = pipe.vae.latents2mesh(lat, bounds=1.01, mc_level=0.0, num_chunks=8000, octree_resolution=octree,
                                    mc_algo="mc", enable_pbar=False)
        mesh = export_to_trimesh(out)[0]
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
    print("raw", len(mesh.vertices), "verts", len(mesh.faces), "faces")
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    if len(mesh.faces) > faces:
        mesh = FaceReducer()(mesh, max_facenum=faces)
    print("clean", len(mesh.vertices), "verts", len(mesh.faces), "faces")
    mesh.export(d / "shape.glb")
    (d / "shape.json").write_text(json.dumps(dict(model=MODEL, steps=steps, guidance=guidance, octree=octree,
                                                  seed=seed, faces=len(mesh.faces))), encoding="utf-8")
    print("ok", d / "shape.glb")


HEIGHTS = {"stand": 1.65, "lookback": 1.65, "umbrella": 1.80, "kneel": 0.92}   # metres, top of the figure in Blender:
# she is 1.65 m; the umbrella pose includes the canopy (her body is drawn ~0.93x the stand sheet, by skirt width);
# kneeling in seiza is ~0.56 of standing


def _basis(f, u):
    """Camera direction (object -> camera) of each view, from her front axis f and up axis u."""
    f, u = np.asarray(f, np.float64), np.asarray(u, np.float64)
    return {"front": f, "left": np.cross(u, f), "back": -f, "right": np.cross(f, u)}


def _project(P, c, u):
    """Orthographic image coordinates (x right, y down) and depth towards the camera of points P seen along -c."""
    r = np.cross(-c, u)
    return P @ r, -(P @ u), P @ c


def _raster(x, y, depth, box, size):
    """Point-splat z-buffer (largest depth = nearest the camera) and coverage over box=(x0, y0, x1, y1) at size=(w, h)."""
    x0, y0, x1, y1 = box
    w, h = size
    ix = np.clip(((x - x0) / (x1 - x0) * w).astype(np.int64), 0, w - 1)
    iy = np.clip(((y - y0) / (y1 - y0) * h).astype(np.int64), 0, h - 1)
    zb = np.full(h * w, -np.inf)
    np.maximum.at(zb, iy * w + ix, depth)
    zb = zb.reshape(h, w)
    zb = ndimage.maximum_filter(zb, 3)                                  # close the gaps between splats
    return zb, ix, iy


def _alpha(view, d):
    a = np.asarray(Image.open(d / f"{view}.png"))[..., 3].astype(np.float32) / 255
    rows, cols = np.flatnonzero(a.max(1) > 0.5), np.flatnonzero(a.max(0) > 0.5)
    return a, (cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)


def paint(name, n_samples=3_000_000):
    """Orient the shape by matching its silhouettes to the four views, scale it to her height with her feet at 0 and
    her front towards -Y (Blender), and work out for every vertex how much each view should colour it: only views
    that really see the point (a z-buffer per view), and the more squarely the better. The colours themselves are
    sampled at full resolution by the Blender shader; here only weights, projections, and a fill colour for the few
    points no view sees."""
    import trimesh
    d = OUT / name
    mesh = trimesh.load(d / "shape.glb", force="mesh")
    V = np.asarray(mesh.vertices, np.float64)
    alphas = {v: _alpha(v, d) for v in VIEWS}

    # which axis is up and which way she faces: the candidate whose four silhouettes best match the four views
    pts, _ = trimesh.sample.sample_surface(mesh, 400_000, seed=3)
    axes = [np.eye(3)[i] * s for i in range(3) for s in (1, -1)]
    best = (-1, None)
    for u in axes:
        for f in axes:
            if abs(f @ u) > 0.5:
                continue
            ious = []
            for view, c in _basis(f, u).items():
                a, (bx0, by0, bx1, by1) = alphas[view]
                x, y, z = _project(pts, c, u)
                hh = 160
                ww = max(8, int(round(hh * (bx1 - bx0) / (by1 - by0))))
                cov = np.zeros((hh, ww), bool)
                ix = np.clip(((x - x.min()) / (np.ptp(x) + 1e-9) * ww).astype(int), 0, ww - 1)
                iy = np.clip(((y - y.min()) / (np.ptp(y) + 1e-9) * hh).astype(int), 0, hh - 1)
                cov[iy, ix] = True
                cov = ndimage.binary_closing(cov, iterations=2)
                ref = np.asarray(Image.fromarray((a[by0:by1, bx0:bx1] * 255).astype(np.uint8)).resize((ww, hh))) > 127
                ious.append((cov & ref).sum() / max(1, (cov | ref).sum()))
            score = float(np.mean(ious))
            if score > best[0]:
                best = (score, (f, u, ious))
    score, (f, u, ious) = best
    print(f"orientation: front {f}, up {u}, silhouette IoU {score:.3f}", [round(i, 3) for i in ious])

    # to Blender: up -> +Z, front -> -Y, height HEIGHT, feet at 0, centred on her footprint
    right = np.cross(f, u)
    R = np.stack([right * -1, -f, u])        # rows: Blender X (her left... see below), Y, Z expressed in mesh axes
    # her right must be -X in Blender, so Blender +X is her left: -right
    P = V @ R.T
    HEIGHT = HEIGHTS[name]
    s = HEIGHT / np.ptp(P[:, 2])
    P = P * s
    P[:, 2] -= P[:, 2].min()
    foot = P[P[:, 2] < 0.15 * HEIGHT]
    P[:, 0] -= np.median(foot[:, 0])
    P[:, 1] -= np.median(foot[:, 1])
    faces = np.asarray(mesh.faces, np.int64)
    m2 = trimesh.Trimesh(P, faces, process=False)
    N = np.asarray(m2.vertex_normals, np.float64)
    spts, sfi = trimesh.sample.sample_surface_even(m2, n_samples, seed=5) if False else trimesh.sample.sample_surface(m2, n_samples, seed=5)

    # each view in the Blender frame: f -> -Y, u -> +Z
    cams = _basis((0, -1, 0), (0, 0, 1))
    U = np.array([0.0, 0.0, 1.0])
    W = np.zeros((len(P), 4))
    proj = {}
    for k, (view, c) in enumerate(cams.items()):
        a, (bx0, by0, bx1, by1) = alphas[view]
        x, y, z = _project(spts, c, U)
        # the image holds her bounding box; map the projected box onto it (one scale, from her height)
        sc = (by1 - by0) / np.ptp(y)
        ox = (bx0 + bx1) / 2 - sc * (x.min() + x.max()) / 2
        oy = by0 - sc * y.min()
        # refine the horizontal fit: shift and widen a little to match the matte best
        h, w = a.shape
        best_fit = (-1, 0.0, 1.0)
        cx = (x.min() + x.max()) / 2
        for dx in np.linspace(-0.03, 0.03, 13) * (bx1 - bx0):
            for kx in (0.97, 0.985, 1.0, 1.015, 1.03):
                px = ox + dx + sc * (cx + (x - cx) * kx)
                py = oy + sc * y
                ix = np.clip(px.astype(int), 0, w - 1)
                iy = np.clip(py.astype(int), 0, h - 1)
                inside = a[iy, ix] > 0.5
                cov = np.zeros((h // 4 + 1, w // 4 + 1), bool)
                cov[iy // 4, ix // 4] = True
                ref = a[::4, ::4] > 0.5
                ref = ref[: cov.shape[0], : cov.shape[1]]
                iou = (cov[: ref.shape[0], : ref.shape[1]] & ref).sum() / max(1, (cov[: ref.shape[0], : ref.shape[1]] | ref).sum())
                if iou > best_fit[0]:
                    best_fit = (iou, dx, kx)
        iou, dx, kx = best_fit
        # image pixel of a Blender point p:  px = A . p + bx,  py = B . p + by
        r = np.cross(-c, U)
        A = sc * kx * r
        bx = ox + dx + sc * cx * (1 - kx)
        B = -sc * U
        by = oy
        proj[view] = dict(A=A.tolist(), bx=float(bx), B=B.tolist(), by=float(by), w=w, h=h, iou=float(iou))
        print(f"{view}: fit IoU {iou:.3f} (shift {dx:+.1f} px, width x{kx})")

        # visibility: a vertex is seen if nothing nearer the camera covers it
        xs, ys, zs = spts @ A + bx, spts @ B + by, spts @ c
        zb, _, _ = _raster(xs, ys, zs, (0, 0, w, h), (w // 2, h // 2))
        vx, vy, vz = P @ A + bx, P @ B + by, P @ c
        ix = np.clip((vx / 2).astype(int), 0, w // 2 - 1)
        iy = np.clip((vy / 2).astype(int), 0, h // 2 - 1)
        seen = vz >= zb[iy, ix] - 0.012
        inside = a[np.clip(vy.astype(int), 0, h - 1), np.clip(vx.astype(int), 0, w - 1)] > 0.35
        facing = np.clip(N @ c, 0, 1)
        # the most squarely facing view wins (blends only where two views see a surface about equally); the AI's
        # side views agree less with the front and back than those two agree with each other, so they count less
        bias = {"front": 1.0, "back": 1.0, "left": 0.5, "right": 0.5}[view]
        W[:, k] = seen * inside * bias * facing ** 6
    total = W.sum(1)
    covered = total > 1e-7
    print(f"vertices seen by at least one view: {covered.mean():.3f}")
    Wn = np.where(covered[:, None], W / np.maximum(total, 1e-9)[:, None], 0.0)

    # fill colour for the unseen: the mean colour of the seen, diffused over the mesh
    cols = np.zeros((len(P), 3))
    for k, view in enumerate(VIEWS):
        img = np.asarray(Image.open(d / f"{view}.png").convert("RGB"), np.float64) / 255
        p = proj[view]
        px = np.clip((P @ np.array(p["A"]) + p["bx"]).astype(int), 0, p["w"] - 1)
        py = np.clip((P @ np.array(p["B"]) + p["by"]).astype(int), 0, p["h"] - 1)
        cols += Wn[:, k:k + 1] * img[py, px]
    fill = cols.copy()
    known = covered.copy()
    edges = m2.edges_unique
    for _ in range(200):
        if known.all():
            break
        a_, b_ = edges[:, 0], edges[:, 1]
        acc = np.zeros_like(fill)
        cnt = np.zeros(len(P))
        for s_, t_ in ((a_, b_), (b_, a_)):
            ok = known[s_] & ~known[t_]
            np.add.at(acc, t_[ok], fill[s_[ok]])
            np.add.at(cnt, t_[ok], 1)
        new = (cnt > 0) & ~known
        fill[new] = acc[new] / cnt[new][:, None]
        known |= new
    np.savez_compressed(d / "her.npz", verts=P.astype(np.float32), faces=faces.astype(np.int32),
                        weights=Wn.astype(np.float32), fill=fill.astype(np.float32), colour=cols.astype(np.float32))

    # her as particles (materialising, dissolving): points on the surface, coloured the same way at full resolution
    n_pts = 260_000
    pts, fi = trimesh.sample.sample_surface(m2, n_pts, seed=9)
    bary = trimesh.triangles.points_to_barycentric(m2.triangles[fi], pts)
    tri = faces[fi]
    wp = np.einsum("nk,nkv->nv", bary, Wn[tri])
    cov = wp.sum(1, keepdims=True)
    pc = np.zeros((n_pts, 3))
    for k, view in enumerate(VIEWS):
        img = np.asarray(Image.open(d / f"{view}.png").convert("RGB"), np.float64) / 255
        p = proj[view]
        px = np.clip((pts @ np.array(p["A"]) + p["bx"]).astype(int), 0, p["w"] - 1)
        py = np.clip((pts @ np.array(p["B"]) + p["by"]).astype(int), 0, p["h"] - 1)
        pc += wp[:, k:k + 1] * img[py, px]
    pc += np.clip(1 - cov, 0, 1) * np.einsum("nk,nkc->nc", bary, fill[tri])
    pn = np.asarray(m2.face_normals[fi], np.float32)
    np.savez_compressed(d / "her_points.npz", xyz=pts.astype(np.float32), rgb=np.clip(pc, 0, 1).astype(np.float32),
                        normal=pn)
    meta = dict(height=HEIGHT, views=list(VIEWS), proj=proj, orientation_iou=score,
                images={v: str((d / f"{v}.png").resolve()) for v in VIEWS})
    (d / "her.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    preview(name)
    print("ok", d / "her.npz")


def preview(name):
    """Four turntable stills of the painted vertices, point-splatted, next to the views: a quick check."""
    d = OUT / name
    z = np.load(d / "her.npz")
    P, col = z["verts"].astype(np.float64), np.clip(z["colour"] + (z["weights"].sum(1, keepdims=True) == 0) * z["fill"], 0, 1)
    tiles = []
    for view, c in _basis((0, -1, 0), (0, 0, 1)).items():
        x, y, depth = _project(P, c, np.array([0.0, 0.0, 1.0]))
        h = 900
        sc = h * 0.95 / np.ptp(y)
        w = int(np.ptp(x) * sc) + 40
        ix = ((x - x.min()) * sc + 20).astype(int)
        iy = ((y - y.min()) * sc + h * 0.025).astype(int)
        order = np.argsort(depth)
        img = np.full((h, w, 3), 0.23)
        img[np.clip(iy[order], 0, h - 1), np.clip(ix[order], 0, w - 1)] = col[order]
        tiles.append((img * 255).astype(np.uint8))
    strip = np.concatenate([np.pad(t, ((0, 0), (0, 10), (0, 0)), constant_values=40) for t in tiles], 1)
    Image.fromarray(strip).save(d / "her_preview.jpg", quality=90)


if __name__ == "__main__":
    {"split": lambda: split(sys.argv[2]), "shape": lambda: shape(sys.argv[2:]),
     "paint": lambda: paint(sys.argv[2]), "preview": lambda: preview(sys.argv[2]),
     "tripo": lambda: [tripo_pack(n) for n in sys.argv[2:]]}[sys.argv[1]]()
