"""Her in the finale, from her original art (input/character_claude.jpg) - never the 3D model's surface.

The 3D world is where she is; the 2D art is how "I" see her. From 45 s on she appears in the world only as traces
(a silhouette, a shadow, lines, light) and in the end as the art itself, assembled in my eyes:

  standee   S5 - S6  the art's matte as a black standee on the mirror floor, only her hair glowing along the art's
                     own strands, a warm rim from the light behind her; revealed by a scan from the head down;
                     it always turns to face the camera (she always faces me).
  art       S8 - S11 the art cut into ~500 Voronoi shards, lifted into a shallow relief along my lines of sight.
                     S8: a pen of light draws the art's line work (many pens, growing along the strokes);
                     S9: light fills her from the heart, tile by tile like a bucket renderer (warm monochrome);
                     S10: the shards burst into depth - each slid along its own ray from my eye, so from that one
                     point they still make her - the camera wanders through them and comes back to my eye on the
                     syllable "eye" (L16): the fragments lock and the colour of the art returns from her eyes;
                     S11: she rises from the hem up as points of light.
  points    S11      those points.

Layers come from pv/art_prep.py (assets/art/): pre = the art's colours pre-compensated for the film's AgX grade
(as pure light they come out as the art's sRGB), aux = line art / pen times / hair / crack distance, warm = the warm
monochrome version and the rim. Object properties drive everything and are keyed in song time.
"""
import json
import math

import bpy
import numpy as np
from mathutils import Vector

import bl_common as C

ART = C.ROOT / "assets" / "art"
EYE = Vector((0.0, -4.0, 0.86))        # my eye: from here the shards make her whole (50 mm, looking along +Y)
EYE_LENS = 50.0
CYAN = (0.45, 0.85, 1.0)
GOLD = (1.0, 0.72, 0.36)


def meta():
    return json.loads((ART / "claude_art.json").read_text(encoding="utf-8"))


def plane_size():
    m = meta()
    w, h = m["size"]
    return w * m["px_m"], h * m["px_m"]


def image(name, colorspace):
    img = bpy.data.images.load(str(ART / name), check_existing=True)
    for cs in (colorspace, "Linear Rec.709", "Linear", "Non-Color"):
        try:
            img.colorspace_settings.name = cs
            break
        except TypeError:
            continue
    return img


def _tex(g, img, uv, interp="Cubic", ext="CLIP"):
    t = g.node("ShaderNodeTexImage", interpolation=interp, extension=ext)
    t.image = img
    g.link(uv, C.sock(t, "Vector"))
    return t


def _rgb_split(g, colour):
    n = g.node("ShaderNodeSeparateColor")
    g.link(colour, C.sock(n, "Color"))
    return n.outputs[0], n.outputs[1], n.outputs[2]


def _gauss(g, x, width):
    """exp(-(x / width)^2)"""
    q = g.math("DIVIDE", x, width)
    return g.math("EXPONENT", g.math("MULTIPLY", g.math("MULTIPLY", q, q), -1.0))


def _emission(g, colour, strength):
    em = g.node("ShaderNodeEmission")
    g._in(C.sock(em, "Color"), colour)
    g._in(C.sock(em, "Strength"), strength)
    return em.outputs[0]


def _add(g, a, b):
    n = g.node("ShaderNodeAddShader")
    g.link(a, n.inputs[0])
    g.link(b, n.inputs[1])
    return n.outputs[0]


def _cutout(g, shader, alpha):
    tr = g.node("ShaderNodeBsdfTransparent")
    mx = g.node("ShaderNodeMixShader")
    g.link(alpha, mx.inputs[0])
    g.link(tr.outputs[0], mx.inputs[1])
    g.link(shader, mx.inputs[2])
    return mx.outputs[0]


# ---------------------------------------------------------------- the standee (S5 - S6)
def standee_material():
    m, g = C.material("art_standee")
    m.use_transparent_shadow = True
    uv = C.sock(g.node("ShaderNodeTexCoord"), "UV", True)
    pre = _tex(g, image("claude_pre.exr", "Linear Rec.709"), uv)
    aux = _tex(g, image("claude_aux.exr", "Non-Color"), uv)
    warm = _tex(g, image("claude_warm.exr", "Linear Rec.709"), uv)
    alpha = C.sock(pre, "Alpha", True)
    _, _, hair = _rgb_split(g, C.sock(aux, "Color", True))
    rim = C.sock(warm, "Alpha", True)
    sep = g.node("ShaderNodeSeparateXYZ")
    g.link(uv, sep.inputs[0])
    v = sep.outputs[1]
    # revealed by a scan from the head down: shown above `scan`, a cyan seam on it
    scan = g.attr("scan")
    shown = g.smoothstep(g.math("SUBTRACT", v, scan), -0.002, 0.002)
    seam = g.math("MULTIPLY", _gauss(g, g.math("SUBTRACT", v, scan), 0.005), alpha)
    # light runs down her hair: a slow 4D noise along the strands
    nz = g.node("ShaderNodeTexNoise", noise_dimensions="4D")
    C.sock(nz, "Scale").default_value = 3.0
    C.sock(nz, "Detail").default_value = 3.0
    g.link(g.vmath("MULTIPLY", uv, (2.0, 0.6, 1.0)), C.sock(nz, "Vector"))
    g.link(g.attr("w"), C.sock(nz, "W"))
    flow = g.math("ADD", 0.35, g.math("MULTIPLY", C.sock(nz, "Fac", True), 1.3))
    hair_e = g.vmath("SCALE", C.sock(pre, "Color", True), scale=g.math("MULTIPLY", g.math("MULTIPLY", hair, flow),
                                                                         g.attr("hair")))
    rim_e = g.vmath("SCALE", (1.0, 0.62, 0.32), scale=g.math("MULTIPLY", rim, g.attr("rim")))
    seam_e = g.vmath("SCALE", CYAN, scale=g.math("MULTIPLY", seam, 14.0))
    col = g.vmath("ADD", g.vmath("ADD", hair_e, rim_e), seam_e)
    bsdf = g.node("ShaderNodeBsdfPrincipled")
    C.sock(bsdf, "Base Color").default_value = (0.004, 0.004, 0.005, 1)
    C.sock(bsdf, "Roughness").default_value = 0.55
    C.sock(bsdf, "Emission Color").default_value = (1, 1, 1, 1)
    g.link(col, C.sock(bsdf, "Emission Color"))
    C.sock(bsdf, "Emission Strength").default_value = 1.0
    a = g.math("MULTIPLY", g.math("MULTIPLY", alpha, shown), g.attr("alpha"))
    a = g.math("MAXIMUM", a, g.math("MULTIPLY", seam, g.attr("alpha")))
    g.link(a, C.sock(bsdf, "Alpha"))
    g.link(C.sock(bsdf, "BSDF", True), C.sock(g.node("ShaderNodeOutputMaterial"), "Surface"))
    return m


def standee(sc, name="standee"):
    """The art's matte standing on the floor at the origin, facing -Y, its foot on z = 0 (rotate it about z)."""
    W, H = plane_size()
    me = bpy.data.meshes.new(name)
    me.from_pydata([(-W / 2, 0, 0), (W / 2, 0, 0), (W / 2, 0, H), (-W / 2, 0, H)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", np.array([0, 0, 1, 0, 1, 1, 0, 1], np.float32))
    me.materials.append(standee_material())
    ob = bpy.data.objects.new(name, me)
    C.link(sc, ob)
    for k, v in dict(scan=1.05, hair=1.2, rim=1.0, alpha=1.0, w=0.0).items():
        ob[k] = v
    return ob


def face_camera(ob, pose, t0, t1):
    """Key the standee's turn about z so that it faces the camera on every frame of [t0, t1]."""
    ob.rotation_mode = "XYZ"
    for f in range(int(math.floor(C.frame(t0))), int(math.ceil(C.frame(t1))) + 1):
        pos = Vector(pose(f / C.FPS)[0])
        d = pos - ob.location
        ob.rotation_euler = (0.0, 0.0, math.atan2(d.x, -d.y))
        ob.keyframe_insert("rotation_euler", frame=f)


# ---------------------------------------------------------------- the art in shards (S8 - S11)
def _relief_xyz(px, py, depth, W, H, w, h):
    """Art pixels -> positions on the art plane, lifted toward my eye by the relief depth (along my line of sight,
    so from the eye nothing moves)."""
    u, v = px / w, 1.0 - py / h
    p = np.stack([(u - 0.5) * W, np.zeros_like(u), v * H], 1)
    d = depth[np.clip(py.astype(int), 0, h - 1), np.clip(px.astype(int), 0, w - 1)]
    e = np.array(EYE, np.float32)
    ray = e - p
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    return (p + d[:, None] * ray).astype(np.float32), np.stack([u, v], 1).astype(np.float32)


def art_shards(sc, times, seed=16):
    """The art as ~500 shards (one mesh): each a fan of triangles round its centre, UV = the art.
    times: dict(shatter, settle, flat) in song time. Returns (object, GN modifier)."""
    z = np.load(ART / "claude_art.npz")
    m = meta()
    w, h = m["size"]
    W, H = plane_size()
    depth = z["depth"].astype(np.float32)
    lens, xy = z["poly_len"], z["poly_xy"]
    rng = np.random.default_rng(seed)
    verts, uvs, faces, attrs = [], [], [], {k: [] for k in ("c", "S", "axis", "ang", "tb", "ts", "tf", "ph", "rnd")}
    hx, hy = m["heart"]
    o = 0
    base = 0
    far = math.hypot(w, h)
    for k in lens:
        P = xy[o:o + k]
        o += k
        ctr = P.mean(0)
        pts = np.vstack([ctr[None], P])
        xyz, uv = _relief_xyz(pts[:, 0], pts[:, 1], depth, W, H, w, h)
        n = len(pts)
        verts.append(xyz)
        uvs.append(uv)
        for j in range(k):                       # (centre, next, this): facing -Y, towards my eye
            faces.append((base, base + 1 + (j + 1) % k, base + 1 + j))
        base += n
        S = float(np.exp(rng.uniform(np.log(0.45), np.log(2.3))))
        axis = rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis)
        dist = math.hypot(ctr[0] - hx, ctr[1] - hy) / far
        vals = dict(c=xyz[0], S=S, axis=axis, ang=rng.choice([-1, 1]) * rng.uniform(0.6, 1.7),
                    tb=times["shatter"] + 0.30 * dist / 0.7 + rng.uniform(0, 0.04),
                    ts=times["settle"] + rng.uniform(0, 0.45), tf=times["flat"] + rng.uniform(0, 0.5),
                    ph=rng.uniform(0, 2 * math.pi), rnd=rng.uniform())
        for key_, val in vals.items():
            attrs[key_].append(np.repeat(np.asarray(val, np.float32)[None], n, 0))
    V = np.concatenate(verts)
    UV = np.concatenate(uvs)
    F = np.array(faces, np.int32)
    me = bpy.data.meshes.new("art_shards")
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3, dtype=np.int32))
    me.update(calc_edges=True)
    layer = me.uv_layers.new(name="UVMap")
    layer.data.foreach_set("uv", UV[F.ravel()].ravel())
    for key_, arr in attrs.items():
        a = np.concatenate(arr).astype(np.float32)
        if a.ndim == 1:
            me.attributes.new(key_, "FLOAT", "POINT").data.foreach_set("value", a)
        else:
            me.attributes.new(key_, "FLOAT_VECTOR", "POINT").data.foreach_set("vector", a.ravel())
    me.materials.append(art_material())
    ob, mod = C.gn_object(sc, "art_shards", shards_group(), mesh=me)
    for k, v in dict(lines=0.0, reveal=-0.05, fill=-0.2, colour=-0.2, edge=0.0, glint=0.0, cut=-0.05,
                     alpha=1.0, wgain=1.0, cgain=1.0, linegain=1.0).items():
        ob[k] = v
    return ob, mod


def shards_group():
    """Each shard: burst (kb), tumble (kr) and scatter in depth (kd) - always along its rays from my eye."""
    g = C.GN("art_shards_gn")
    tt = g.input("T", 0.0)
    eye = g.input("Eye", None, kind="NodeSocketVector")
    item = g.ng.interface.items_tree["Eye"]
    item.default_value = tuple(EYE)

    def clamp01(x):
        return g.math("MINIMUM", g.math("MAXIMUM", x, 0.0), 1.0)

    x = clamp01(g.math("DIVIDE", g.math("SUBTRACT", tt, g.named("tb")), 1.7))
    kb = g.math("SUBTRACT", 1.0, g.math("POWER", g.math("SUBTRACT", 1.0, x), 3.0))
    settle = g.smoothstep(g.math("SUBTRACT", tt, g.named("ts")), 0.0, 1.15)
    kr = g.math("MULTIPLY", kb, g.math("SUBTRACT", 1.0, settle))
    flat = g.smoothstep(g.math("SUBTRACT", tt, g.named("tf")), 0.0, 1.5)
    kd = g.math("MULTIPLY", kb, g.math("SUBTRACT", 1.0, flat))
    breathe = g.math("ADD", 1.0, g.math("MULTIPLY", g.math("MULTIPLY", g.math("SINE", g.math(
        "ADD", g.math("MULTIPLY", tt, 0.9), g.named("ph"))), 0.035), kd))
    S = g.math("MULTIPLY", g.math("ADD", 1.0, g.math("MULTIPLY", g.math("SUBTRACT", g.named("S"), 1.0), kd)), breathe)
    rot = g.node("ShaderNodeVectorRotate", rotation_type="AXIS_ANGLE")
    g.link(g.position(), C.sock(rot, "Vector"))
    g.link(g.named("c", "FLOAT_VECTOR"), C.sock(rot, "Center"))
    g.link(g.named("axis", "FLOAT_VECTOR"), C.sock(rot, "Axis"))
    g.link(g.math("MULTIPLY", g.named("ang"), kr), C.sock(rot, "Angle"))
    p = g.vmath("ADD", eye, g.vmath("SCALE", g.vmath("SUBTRACT", rot.outputs[0], eye), scale=S))
    sp = g.node("GeometryNodeSetPosition")
    g.link(g.geometry(), C.sock(sp, "Geometry"))
    g.link(p, C.sock(sp, "Position"))
    geo = C.sock(sp, "Geometry", True)
    for name, field in (("kd", kd), ("kr", kr)):              # for the shader: how scattered / tumbling
        st = g.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
        g.link(geo, C.sock(st, "Geometry"))
        C.sock(st, "Name").default_value = name
        g.link(field, next(s for s in st.inputs if s.name == "Value" and s.enabled))
        geo = C.sock(st, "Geometry", True)
    return g.finish(geo)


def art_material():
    """lines (S8) -> warm light by tiles from the heart (S9) -> the art's colour from the eyes (S10); cracks; rising
    cut (S11). Object props: lines, reveal, fill, colour, edge, glint, cut, alpha, wgain, cgain, linegain."""
    m, g = C.material("art_main")
    m.use_transparent_shadow = True
    uv = C.sock(g.node("ShaderNodeTexCoord"), "UV", True)
    pre = _tex(g, image("claude_pre.exr", "Linear Rec.709"), uv)
    aux = _tex(g, image("claude_aux.exr", "Non-Color"), uv, interp="Linear")
    warm = _tex(g, image("claude_warm.exr", "Linear Rec.709"), uv)
    md = meta()
    w, h = md["size"]
    aspect = w / h
    sep = g.node("ShaderNodeSeparateXYZ")
    g.link(uv, sep.inputs[0])
    u, v = sep.outputs[0], sep.outputs[1]
    line, pen, _ = _rgb_split(g, C.sock(aux, "Color", True))
    crack_px = C.sock(aux, "Alpha", True)
    alpha_art = C.sock(pre, "Alpha", True)

    def dist_from(px, py):                          # distance in art heights from an art pixel
        du = g.math("MULTIPLY", g.math("SUBTRACT", u, px / w), aspect)
        dv = g.math("SUBTRACT", v, 1.0 - py / h)
        return g.math("SQRT", g.math("ADD", g.math("MULTIPLY", du, du), g.math("MULTIPLY", dv, dv)))

    # S8 lines: drawn where the pen has been (pen time < reveal), the pens' heads bright
    has = g.math("LESS_THAN", pen, 1.5)
    dr = g.math("SUBTRACT", g.attr("reveal"), pen)
    drawn = g.math("MULTIPLY", g.smoothstep(dr, 0.0, 0.004), has)
    head = g.math("MULTIPLY", _gauss(g, dr, 0.010), g.math("MULTIPLY", has, g.math("GREATER_THAN", dr, -0.004)))
    line_a = g.math("MINIMUM", g.math("MULTIPLY", line, g.math("ADD", drawn, g.math("MULTIPLY", head, 1.5))), 1.0)
    line_e = g.math("MULTIPLY", line, g.math("ADD", g.math("MULTIPLY", drawn, 2.2), g.math("MULTIPLY", head, 28.0)))
    line_e = g.math("MULTIPLY", line_e, g.attr("linegain"))
    # S9 fill: 16 px tiles, each lit when the front from her heart reaches it (a little random), flashing as it lands
    tx = g.math("FLOOR", g.math("MULTIPLY", u, w / 16.0))
    ty = g.math("FLOOR", g.math("MULTIPLY", v, h / 16.0))
    wn = g.node("ShaderNodeTexWhiteNoise", noise_dimensions="2D")
    cx = g.node("ShaderNodeCombineXYZ")
    g.link(tx, cx.inputs[0])
    g.link(ty, cx.inputs[1])
    g.link(cx.outputs[0], C.sock(wn, "Vector"))
    rnd = C.sock(wn, "Value", True)
    hx, hy = md["heart"]
    tcu = g.math("DIVIDE", g.math("ADD", tx, 0.5), w / 16.0)
    tcv = g.math("DIVIDE", g.math("ADD", ty, 0.5), h / 16.0)
    du = g.math("MULTIPLY", g.math("SUBTRACT", tcu, hx / w), aspect)
    dv = g.math("SUBTRACT", tcv, 1.0 - hy / h)
    tdist = g.math("SQRT", g.math("ADD", g.math("MULTIPLY", du, du), g.math("MULTIPLY", dv, dv)))
    ttime = g.math("ADD", g.math("DIVIDE", tdist, 0.85), g.math("MULTIPLY", rnd, 0.10))
    age = g.math("SUBTRACT", g.attr("fill"), ttime)
    shown = g.math("GREATER_THAN", age, 0.0)
    flash = g.math("MULTIPLY", shown, g.math("EXPONENT", g.math("DIVIDE", age, -0.03)))
    warm_e = g.vmath("ADD", g.vmath("SCALE", C.sock(warm, "Color", True), scale=g.attr("wgain")),
                     g.vmath("SCALE", (1.0, 0.86, 0.66), scale=g.math("MULTIPLY", flash, 4.0)))
    # S10 colour: from her eyes outwards, a bright ring at the front
    ex, ey = md["eyes"]
    de = dist_from(ex, ey)
    cfront = g.math("SUBTRACT", g.attr("colour"), de)
    cmix = g.smoothstep(cfront, 0.0, 0.07)
    art_e = g.vmath("SCALE", C.sock(pre, "Color", True), scale=g.attr("cgain"))
    body_e = g.mix(warm_e, art_e, cmix)
    ring = g.math("MULTIPLY", _gauss(g, cfront, 0.018), g.math("GREATER_THAN", g.attr("colour"), 0.0))
    body_e = g.vmath("ADD", body_e, g.vmath("SCALE", GOLD, scale=g.math("MULTIPLY", ring, 3.0)))
    # the cracks between shards: gold light, stronger the more scattered
    crack = g.math("SUBTRACT", 1.0, g.smoothstep(crack_px, 0.6, 2.4))
    kd = g.attr("kd", "GEOMETRY")
    crack_e = g.math("MULTIPLY", g.math("MULTIPLY", crack, g.attr("edge")), g.math("ADD", 0.35, kd))
    # S11 cut: gone below `cut`, a seam of light on it
    cdv = g.math("SUBTRACT", v, g.attr("cut"))
    keep = g.smoothstep(cdv, -0.003, 0.003)
    seam = g.math("MULTIPLY", _gauss(g, cdv, 0.006), alpha_art)
    # together
    body_a = g.math("MULTIPLY", alpha_art, shown)
    lines_on = g.math("MULTIPLY", g.attr("lines"), g.math("SUBTRACT", 1.0, body_a))
    a = g.math("MAXIMUM", g.math("MULTIPLY", line_a, lines_on), body_a)
    a = g.math("MAXIMUM", a, g.math("MULTIPLY", g.math("MULTIPLY", crack, alpha_art),
                                    g.math("MINIMUM", g.attr("edge"), 1.0)))
    a = g.math("MULTIPLY", g.math("MULTIPLY", a, keep), g.attr("alpha"))
    a = g.math("MAXIMUM", a, g.math("MULTIPLY", seam, g.math("GREATER_THAN", g.attr("cut"), 0.0)))
    col = g.vmath("SCALE", CYAN, scale=g.math("MULTIPLY", line_e, lines_on))
    col = g.vmath("ADD", col, g.vmath("SCALE", body_e, scale=body_a))
    col = g.vmath("ADD", col, g.vmath("SCALE", GOLD, scale=g.math("MULTIPLY", crack_e, 4.0)))
    col = g.vmath("ADD", col, g.vmath("SCALE", GOLD, scale=g.math("MULTIPLY", seam, 10.0)))
    em = _emission(g, col, 1.0)
    # glints: tumbling glass catches the dusk (view-dependent, so it is gone before the shards lock)
    gl = g.node("ShaderNodeBsdfGlossy")
    C.sock(gl, "Roughness").default_value = 0.12
    C.sock(gl, "Color").default_value = (1.0, 0.85, 0.65, 1)
    lw = g.node("ShaderNodeLayerWeight")
    C.sock(lw, "Blend").default_value = 0.3
    gw = g.math("MULTIPLY", g.math("MULTIPLY", C.sock(lw, "Fresnel", True), g.attr("glint")),
                g.math("ADD", 0.2, g.attr("kr", "GEOMETRY")))
    glint = g.node("ShaderNodeMixShader")                   # (nothing, not transparency: that would add up)
    g.link(gw, glint.inputs[0])
    g.link(_emission(g, (0, 0, 0, 1), 0.0), glint.inputs[1])
    g.link(gl.outputs[0], glint.inputs[2])
    surf = _cutout(g, _add(g, em, glint.outputs[0]), a)
    g.link(surf, C.sock(g.node("ShaderNodeOutputMaterial"), "Surface"))
    return m


# ---------------------------------------------------------------- S11: she rises as points of light
def art_points(sc, t_release, dur=0.85, n=60000, seed=11):
    """Points of the art (its colours), released from the hem up as the cut passes them, rising and drifting."""
    img = image("claude_pre.exr", "Linear Rec.709")
    px = np.empty(img.size[0] * img.size[1] * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(img.size[1], img.size[0], 4)[::-1]                   # top row first, as the art's pixels
    m = meta()
    w, h = m["size"]
    W, H = plane_size()
    a = px[..., 3].astype(np.float64)
    rng = np.random.default_rng(seed)
    p = (a / a.sum()).ravel()
    pick = rng.choice(a.size, n, p=p)
    py, pxx = np.divmod(pick, w)
    pxx = pxx + rng.uniform(0, 1, n)
    py = py + rng.uniform(0, 1, n)
    depth = np.load(ART / "claude_art.npz")["depth"].astype(np.float32)
    xyz, uv = _relief_xyz(pxx, py, depth, W, H, w, h)
    col = px[py.astype(int), pxx.astype(int), :3].astype(np.float32)
    release = (t_release + uv[:, 1] * dur + rng.uniform(0, 0.05, n)).astype(np.float32)
    me = C.points_mesh("art_points", xyz, dict(release=release, drift=rng.normal(0, 1, (n, 3)).astype(np.float32),
                                                speed=rng.uniform(0.6, 1.8, n).astype(np.float32),
                                                col=np.concatenate([np.minimum(col, 6.0), np.ones((n, 1), np.float32)], 1)))
    g = C.GN("art_points_gn")
    tt = g.input("T", 0.0)
    age = g.math("MAXIMUM", g.math("SUBTRACT", tt, g.named("release")), 0.0)
    up = g.math("ADD", g.math("MULTIPLY", g.math("MULTIPLY", age, age), g.math("MULTIPLY", g.named("speed"), 0.45)),
                g.math("MULTIPLY", age, 0.3))
    spread = g.vmath("SCALE", g.named("drift", "FLOAT_VECTOR"), scale=g.math("MULTIPLY", age, 0.10))
    pos = g.vmath("ADD", g.vmath("ADD", g.position(), spread), g.xyz(0.0, 0.0, up))
    sp = g.node("GeometryNodeSetPosition")
    g.link(g.geometry(), C.sock(sp, "Geometry"))
    g.link(pos, C.sock(sp, "Position"))
    m2p = g.node("GeometryNodeMeshToPoints")
    g.link(C.sock(sp, "Geometry", True), C.sock(m2p, "Mesh"))
    g.link(g.math("GREATER_THAN", age, 0.0), C.sock(m2p, "Selection"))
    g.link(g.math("MULTIPLY", 0.0042, g.math("SUBTRACT", 1.0, g.smoothstep(age, 0.5, 1.5))), C.sock(m2p, "Radius"))
    mat, gm = C.material("art_points_mat")
    em = gm.node("ShaderNodeEmission")
    gm.link(gm.vmath("ADD", gm.attr("col", "GEOMETRY", "Color"), (0.3, 0.18, 0.06)), C.sock(em, "Color"))
    C.sock(em, "Strength").default_value = 1.6
    gm.link(em.outputs[0], C.sock(gm.node("ShaderNodeOutputMaterial"), "Surface"))
    sm = g.node("GeometryNodeSetMaterial")
    g.link(C.sock(m2p, "Points", True), C.sock(sm, "Geometry"))
    C.sock(sm, "Material").default_value = mat
    ob, mod = C.gn_object(sc, "art_points", g.finish(C.sock(sm, "Geometry", True)), mesh=me)
    return ob, mod
