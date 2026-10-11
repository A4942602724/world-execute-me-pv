"""S5 - S11, the climax (46.2 - 76.93 s, lyric lines 11 - 16): the runtime world, her as traces and as her own
art, and the new lyric type.

Her (v3.1): from 45 s on the 3D model is never seen. In the world she is a trace - the art's silhouette as a black
standee with only her hair alight (S5 - S6), the shadow of a body we never see on the go board (S7) - and in the end
the art itself (her_art.py): drawn by a pen of light (S8), filled with light from the heart (S9), burst into shards
that make her whole only from my eye, where the camera returns on the syllable "eye" and the colour comes back (S10),
then rising away as light (S11).

Spatial calligraphy, compiled from code: from the drop on, the lyrics are objects in her world - extruded Noto Serif
TC Black, lit by the scene, reflected in the floor - and each first appears as its own source: the glyph's skeleton
as a font editor shows it (points, handles, ink box, its codepoint), traced by the light pen, then lit, then burned
away into light. Layout per shot:
  L11  monoliths fall out of the dark sky onto the floor behind her, their skeletons compiling as they fall (S5)
  L12  a carousel of letters around her; ribbons of running code wind round her into a cocoon (S6)
  L13  go stones: each syllable drops a black or white stone with its letter into her shadow; its move in SGF (S7)
  L14  the pen draws her line art and writes the line beside her, skeletons in blueprint cyan (S8)
  L15  light fills her from the heart; the second column fills tile by tile, like a renderer (S9)
  L16  dusk; the letters rise like lanterns among the shards (S10, S11)
Everything is keyed in song time (frame = t * 24); nothing moves on the drums.
"""
import math

import bpy
import numpy as np
from mathutils import Vector

import bl_common as C
import her_art as A

T0, T1 = 46.22, 76.93


def times():
    L = {ln["i"]: ln for ln in C.timing()["lines"]}
    T = dict(L=L)
    for i in range(11, 17):
        T[f"l{i}"] = L[i]["start"]
        T[f"l{i}_end"] = L[i]["end"]
    T["drop"] = L[11]["chars"][2]["s"]                     # the third syllable of L11 lands on the drop (~47.2)
    T["l13b"] = L[13]["chars"][6]["s"]                    # second phrase of L13
    T["shatter"] = L[16]["start"]                         # her picture breaks into depth on the first syllable
    T["lock"] = L[16]["chars"][6]["s"]                    # ... and is whole again, in my eye, on "eye" (眸)
    T["settle"] = T["lock"] - 1.6                         # the shards stop turning before the camera arrives
    T["flat"] = T["lock"] + 1.1                           # then slide back onto her plane (unseen from my eye)
    T["end"] = T1
    # shot boundaries (cuts on line starts)
    T["s6"], T["s7"], T["s8"], T["s9"] = (L[12]["start"] - 0.25, L[13]["start"] - 0.35, L[14]["start"] - 0.4,
                                          L[15]["start"] - 0.2)
    T["s10"] = T["shatter"] - 0.02
    return T


def shown(ob, spans):
    """Keyframe visibility: shown only inside the [a, b) song-time spans."""
    ob.hide_render = True
    ob.keyframe_insert("hide_render", frame=C.frame(T0) - 1)
    for a, b in spans:
        ob.hide_render = False
        ob.keyframe_insert("hide_render", frame=round(C.frame(a)))
        ob.hide_render = True
        ob.keyframe_insert("hide_render", frame=round(C.frame(b)))
    for fc in C.fcurves(ob):
        if fc.data_path == "hide_render":
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


# ---------------------------------------------------------------- world, floor, lights
def world(sc, T):
    """void (black-blue) -> blueprint navy -> dusk gradient (gold horizon, violet sky): keyed mixes."""
    w = sc.world
    nt = w.node_tree
    nt.nodes.clear()
    g = C.Graph(nt)
    tc = g.node("ShaderNodeTexCoord")
    _, _, z = (lambda n: (n.outputs[0], n.outputs[1], n.outputs[2]))(g.node("ShaderNodeSeparateXYZ"))
    g.link(C.sock(tc, "Generated", True), z.node.inputs[0])
    up = g.smoothstep(z, 0.0, 0.11)                       # a narrow gold band on the horizon, deep blue above
    dusk = g.mix((0.62, 0.27, 0.085, 1), (0.010, 0.012, 0.036, 1), up)
    dusk = g.mix(dusk, (0.045, 0.030, 0.06, 1), g.smoothstep(z, 0.12, 0.45))
    void = (0.0015, 0.0018, 0.0035, 1)
    blue = (0.003, 0.010, 0.026, 1)
    k_blue = g.node("ShaderNodeValue", label="k_blue")
    k_dusk = g.node("ShaderNodeValue", label="k_dusk")
    col = g.mix(g.mix(void, blue, k_blue.outputs[0]), dusk, k_dusk.outputs[0])
    bg = g.node("ShaderNodeBackground")
    g.link(col, C.sock(bg, "Color"))
    C.sock(bg, "Strength").default_value = 1.0
    out = g.node("ShaderNodeOutputWorld")
    g.link(bg.outputs[0], C.sock(out, "Surface"))
    # the haze is a box, not a world volume: Eevee treats a world volume as infinite and it would swallow the sky
    me = bpy.data.meshes.new("finale_haze")
    hx, hy, hz = 45.0, 45.0, 25.0
    me.from_pydata([(sx * hx, sy * hy, sz * hz - 1.0) for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)], [],
                   [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    haze = bpy.data.objects.new("finale_haze", me)
    C.link(sc, haze)
    m, gm = C.material("finale_haze_mat")
    vol = gm.node("ShaderNodeVolumePrincipled")
    gm.link(gm.math("MULTIPLY", 0.0018, gm.attr("dens")), C.sock(vol, "Density"))
    C.sock(vol, "Anisotropy").default_value = 0.6
    gm.link(C.sock(vol, "Volume", True), C.sock(gm.node("ShaderNodeOutputMaterial"), "Volume"))
    me.materials.append(m)
    haze["dens"] = 1.0                                   # thinner at dusk: looking into the sun it would go milky
    C.key(haze, '["dens"]', T["s10"] - 0.01, 1.0, "CONSTANT")
    C.key(haze, '["dens"]', T["s10"], 0.3, "CONSTANT")
    for node, keys in ((k_blue, ((T0, 0.0), (T["s8"] - 0.01, 0.0), (T["s8"], 1.0), (T["s10"] - 0.01, 1.0), (T["s10"], 0.0))),
                       (k_dusk, ((T0, 0.0), (T["s10"] - 0.01, 0.0), (T["s10"], 1.0)))):
        for t, v in keys:
            node.outputs[0].default_value = v
            node.outputs[0].keyframe_insert("default_value", frame=C.frame(t))
    for fc in C.fcurves(nt):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"
    sc.eevee.volumetric_end = 60.0


def floor(sc, T):
    """A black mirror with a grid of light: faint in S5-S6, the go board in S7 (bright, 0.45 m cells), gone in the
    blueprint, warm and soft at dusk. 'grid', 'tint', 'rough' and 'litk' are keyed object props. With litk = 1 (S7)
    the board answers to light: the grid and a faint warm pool show only where the key light reaches, so a shadow
    on it reads as a hole in the grid."""
    me = bpy.data.meshes.new("finale_floor")
    s = 60.0
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("finale_floor", me)
    C.link(sc, ob)
    m, g = C.material("finale_floor_mat")
    tc = g.node("ShaderNodeTexCoord")
    p = C.sock(tc, "Object", True)
    x, y, _ = (lambda n: (n.outputs[0], n.outputs[1], n.outputs[2]))(g.node("ShaderNodeSeparateXYZ"))
    g.link(p, x.node.inputs[0])
    cell = 0.45

    def line(v):
        f = g.math("ABSOLUTE", g.math("SUBTRACT", g.math("FRACT", g.math("DIVIDE", v, cell)), 0.5))
        return g.smoothstep(f, 0.489, 0.4975)
    grid = g.math("MAXIMUM", line(x), line(y))
    dist = g.math("SQRT", g.math("ADD", g.math("MULTIPLY", x, x), g.math("MULTIPLY", y, y)))
    fade = g.math("DIVIDE", 1.0, g.math("ADD", 1.0, g.math("MULTIPLY", dist, 0.08)))
    # how much light reaches this point of the floor (with shadows): a white diffuse surface, read back as colour
    s2r = g.node("ShaderNodeShaderToRGB")
    g.link(g.node("ShaderNodeBsdfDiffuse").outputs[0], s2r.inputs[0])
    bw = g.node("ShaderNodeRGBToBW")
    g.link(s2r.outputs[0], bw.inputs[0])
    lit = g.smoothstep(bw.outputs[0], 0.015, 0.16)
    litk = g.attr("litk")
    gridmul = g.mix(1.0, lit, litk, kind="FLOAT")
    bsdf = g.node("ShaderNodeBsdfPrincipled")
    g.link(g.mix((0.008, 0.008, 0.010, 1), (0.035, 0.024, 0.018, 1), g.attr("tint")), C.sock(bsdf, "Base Color"))
    g.link(g.attr("rough"), C.sock(bsdf, "Roughness"))
    C.sock(bsdf, "Specular IOR Level").default_value = 0.8
    C.sock(bsdf, "Emission Color").default_value = (0.85, 0.92, 1.0, 1)
    strength = g.math("MULTIPLY", g.math("MULTIPLY", g.math("MULTIPLY", grid, fade), g.attr("grid")), gridmul)
    g.link(strength, C.sock(bsdf, "Emission Strength"))
    pool = g.node("ShaderNodeEmission")
    C.sock(pool, "Color").default_value = (1.0, 0.80, 0.58, 1)
    g.link(g.math("MULTIPLY", g.math("MULTIPLY", lit, litk), 0.012), C.sock(pool, "Strength"))
    add = g.node("ShaderNodeAddShader")
    g.link(C.sock(bsdf, "BSDF", True), add.inputs[0])
    g.link(pool.outputs[0], add.inputs[1])
    g.link(add.outputs[0], C.sock(g.node("ShaderNodeOutputMaterial"), "Surface"))
    me.materials.append(m)
    ob["grid"], ob["tint"], ob["rough"], ob["litk"] = 0.0, 0.0, 0.06, 0.0
    C.key(ob, '["grid"]', T0, 0.0)
    C.key(ob, '["grid"]', T["drop"], 0.0)
    C.key(ob, '["grid"]', T["drop"] + 0.6, 0.35)
    C.key(ob, '["grid"]', T["s7"], 0.35)
    C.key(ob, '["grid"]', T["s7"] + 0.8, 1.5)
    C.key(ob, '["grid"]', T["s8"] - 0.3, 1.5)
    C.key(ob, '["grid"]', T["s8"], 0.0)
    C.key(ob, '["litk"]', T["s7"] - 0.01, 0.0, "CONSTANT")
    C.key(ob, '["litk"]', T["s7"], 1.0, "CONSTANT")
    C.key(ob, '["litk"]', T["s8"], 0.0, "CONSTANT")
    C.key(ob, '["tint"]', T["s10"] - 0.01, 0.0, "CONSTANT")
    C.key(ob, '["tint"]', T["s10"], 1.0, "CONSTANT")
    C.key(ob, '["rough"]', T["s10"] - 0.01, 0.06, "CONSTANT")
    C.key(ob, '["rough"]', T["s10"], 0.15, "CONSTANT")
    return ob


def light(sc, name, kind, loc, aim, colour, energy_keys, size=1.0, spot=None, soft=0.2):
    ld = bpy.data.lights.new(name, kind)
    ld.color = colour
    if kind == "AREA":
        ld.size = size
    if kind == "SPOT":
        ld.spot_size = math.radians(spot or 40)
        ld.spot_blend = 0.6
    if kind in ("SPOT", "POINT", "AREA"):
        ld.shadow_soft_size = soft
    if kind == "AREA":
        ld.specular_factor = 0.0                                # no glowing rectangles mirrored in the floor
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (Vector(aim) - Vector(loc)).to_track_quat("-Z", "Y")
    sc.collection.objects.link(ob)
    for t, e in energy_keys:
        ld.energy = e
        ld.keyframe_insert("energy", frame=C.frame(t))
    return ob


def lights(sc, T):
    s5, s7, s8 = T["drop"], T["s7"], T["s8"]
    bg = light(sc, "back_glow", "AREA", (0, 7.5, 2.4), (0, 0, 1.2), (1.0, 0.55, 0.28),
               ((T0, 0), (s5 - 0.02, 0), (s5, 4200), (s5 + 0.6, 1300), (s7 - 0.2, 1300), (s7, 0)), size=9)
    bg.data.shape = "DISK"                                  # a round, softer glow in the haze, not a lit doorway
    bg.data.volume_factor = 0.45
    light(sc, "top_cool", "SPOT", (0.6, -1.2, 6.5), (0, 0, 1.0), (0.75, 0.85, 1.0),
          ((T0, 0), (s5, 0), (s5 + 0.4, 500), (s7, 500), (s7 + 0.6, 160), (s8, 0)), spot=50)
    light(sc, "board_fill", "AREA", (0, -2.5, 4.5), (0, 0, 0.4), (0.8, 0.88, 1.0),
          ((T0, 0), (s7, 0), (s7 + 0.6, 60), (s8, 0)), size=4)
    # S7: a low warm key from behind her, so the body we never see lays its shadow over the board
    key = light(sc, "shadow_key", "SPOT", (0.3, 3.0, 4.2), (0.0, -1.2, 0.0), (1.0, 0.84, 0.66),
                ((T0, 0), (s7, 0), (s7 + 0.8, 3400), (s8 - 0.3, 3400), (s8, 0)), spot=64, soft=0.05)
    key.data.spot_blend = 1.0
    light(sc, "pen_cyan", "POINT", (0.9, -1.4, 1.6), (0, 0, 1.0), (0.45, 0.85, 1.0),
          ((T0, 0), (s8, 0), (s8 + 0.3, 60), (T["s9"] + 3.0, 40), (T["s10"], 0)))
    sd = bpy.data.lights.new("dusk_sun", "SUN")
    sd.color = (1.0, 0.62, 0.32)
    sd.angle = math.radians(4)
    sun = bpy.data.objects.new("dusk_sun", sd)
    sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = (Vector((0.55, -1.0, -0.14))).to_track_quat("-Z", "Y")   # low, from behind her left
    sc.collection.objects.link(sun)
    for t, e in ((T0, 0), (T["s10"] - 0.01, 0), (T["s10"], 3.2), (T1, 3.2)):
        sd.energy = e
        sd.keyframe_insert("energy", frame=C.frame(t))
    light(sc, "dusk_fill", "AREA", (1.6, -3.0, 1.8), (0, 0, 1.3), (0.62, 0.66, 1.0),
          ((T0, 0), (T["s10"] - 0.01, 0), (T["s10"], 90), (T1, 90)), size=3)


# ---------------------------------------------------------------- her
def her(sc, T, pose):
    """The standee (S5 - S6), the unseen kneeling body (S7: its shadow only), her art (S8 - S11)."""
    st = A.standee(sc)
    shown(st, ((T["drop"] - 0.04, T["s7"] + 0.02),))
    C.key(st, '["scan"]', T["drop"] + 0.02, 1.05)                   # revealed by a scan from the head down
    C.key(st, '["scan"]', T["drop"] + 0.62, -0.05)
    C.key(st, '["hair"]', T["drop"], 2.4)
    C.key(st, '["hair"]', T["drop"] + 1.2, 1.25)
    C.key(st, '["rim"]', T["drop"], 2.6)
    C.key(st, '["rim"]', T["drop"] + 0.6, 1.0)
    C.key(st, '["w"]', T0, 0.0, "LINEAR")
    C.key(st, '["w"]', T["s7"], 2.4, "LINEAR")
    A.face_camera(st, pose, T["drop"] - 0.05, T["s7"] + 0.05)
    # S7: she stands at the head of the board, never seen; the light behind her lays her shadow (the art's own
    # silhouette) down the board, over the stones
    kn = bpy.data.objects.new("her_shadow", st.data)                 # the same standee (mesh and material)
    C.link(sc, kn)
    for k, v in dict(scan=-0.05, hair=1.0, rim=0.0, alpha=1.0, w=0.0).items():
        kn[k] = v
    kn.location = (0.0, 0.15, 0.0)
    kn.visible_camera = False
    kn.visible_glossy = False
    kn.visible_diffuse = False
    kn.visible_volume_scatter = False
    shown(kn, ((T["s7"], T["s8"]),))
    art, mod = A.art_shards(sc, dict(shatter=T["shatter"], settle=T["settle"], flat=T["flat"]))
    shown(art, ((T["s8"], T1 + 1),))
    C.gn_key(art, mod, "T", T["s8"], T["s8"])
    C.gn_key(art, mod, "T", T1 + 1, T1 + 1)
    C.linear(art, "modifiers")
    s8, s9 = T["s8"], T["s9"]
    C.key(art, '["lines"]', s8, 1.0, "CONSTANT")
    C.key(art, '["lines"]', T["shatter"], 0.0, "CONSTANT")
    C.key(art, '["reveal"]', s8 + 0.12, -0.02, "LINEAR")            # S8: the pens draw her, head to hem
    C.key(art, '["reveal"]', T["l14_end"] - 0.1, 1.03, "LINEAR")
    C.key(art, '["linegain"]', s9, 1.0)
    C.key(art, '["linegain"]', s9 + 1.5, 0.55)
    L15 = T["L"][15]["chars"]                                        # S9: light from the heart, a push per syllable
    n = len(L15)
    C.key(art, '["fill"]', s9 + 0.15, -0.12)
    for k, c in enumerate(L15):
        C.key(art, '["fill"]', c["s"] + 0.36, -0.05 + 1.35 * (k + 1) / n)
    C.key(art, '["colour"]', T["lock"] - 0.02, -0.15)              # S10: the colour comes back from her eyes
    C.key(art, '["colour"]', T["lock"] + 1.0, 1.35)
    C.key(art, '["edge"]', T["shatter"] - 0.02, 0.0)
    C.key(art, '["edge"]', T["shatter"] + 0.08, 1.6)
    C.key(art, '["edge"]', T["shatter"] + 1.3, 0.75)
    C.key(art, '["edge"]', T["lock"] - 0.12, 0.75)
    C.key(art, '["edge"]', T["lock"] + 0.04, 2.2)
    C.key(art, '["edge"]', T["lock"] + 1.25, 0.0)
    C.key(art, '["glint"]', T["shatter"], 0.0)
    C.key(art, '["glint"]', T["shatter"] + 0.4, 1.0)
    C.key(art, '["glint"]', T["settle"], 1.0)
    C.key(art, '["glint"]', T["lock"] - 0.15, 0.0)
    C.key(art, '["cut"]', T["l16_end"], -0.05, "LINEAR")             # S11: she rises from the hem up as light
    C.key(art, '["cut"]', T["l16_end"] + 0.85, 1.03, "LINEAR")
    pts, pmod = A.art_points(sc, T["l16_end"])
    C.gn_key(pts, pmod, "T", T["l16_end"] - 0.1, T["l16_end"] - 0.1)
    C.gn_key(pts, pmod, "T", T1 + 1, T1 + 1)
    C.linear(pts)
    shown(pts, ((T["l16_end"] - 0.05, T1 + 1),))
    return st, kn, art


def motes(sc, T):
    """Before the drop: the light she became in the rain is still rising past us into the dark."""
    rng = np.random.default_rng(46)
    n = 2600
    xyz = np.stack([rng.normal(0, 2.2, n), rng.uniform(-2, 14, n), rng.uniform(-1, 16, n)], 1).astype(np.float32)
    col = np.concatenate([np.clip(rng.normal([1.0, 0.82, 0.68], 0.06, (n, 3)), 0, 1), np.ones((n, 1))], 1)
    me = C.points_mesh("motes_pts", xyz, dict(speed=rng.uniform(1.5, 4.0, n).astype(np.float32),
                                              col=col.astype(np.float32)))
    g = C.GN("motes_gn")
    tt = g.input("T", 0.0)
    x, y, z = g.split(g.position())
    zz = g.math("WRAP", g.math("ADD", z, g.math("MULTIPLY", g.named("speed"), tt)), 16.0, -1.0)
    sp = g.node("GeometryNodeSetPosition")
    g.link(g.geometry(), C.sock(sp, "Geometry"))
    g.link(g.xyz(x, y, zz), C.sock(sp, "Position"))
    m2p = g.node("GeometryNodeMeshToPoints")
    g.link(C.sock(sp, "Geometry", True), C.sock(m2p, "Mesh"))
    C.sock(m2p, "Radius").default_value = 0.012
    m, gm = C.material("motes_mat")
    em = gm.node("ShaderNodeEmission")
    gm.link(gm.attr("col", "GEOMETRY", "Color"), C.sock(em, "Color"))
    C.sock(em, "Strength").default_value = 5.0
    gm.link(em.outputs[0], C.sock(gm.node("ShaderNodeOutputMaterial"), "Surface"))
    sm = g.node("GeometryNodeSetMaterial")
    g.link(C.sock(m2p, "Points", True), C.sock(sm, "Geometry"))
    C.sock(sm, "Material").default_value = m
    ob, mod = C.gn_object(sc, "motes", g.finish(C.sock(sm, "Geometry", True)), mesh=me)
    C.gn_key(ob, mod, "T", T0, 0.0)
    C.gn_key(ob, mod, "T", T["drop"] + 1, T["drop"] + 1 - T0)
    C.linear(ob)
    shown(ob, ((T0, T["drop"]),))
    return ob


# ---------------------------------------------------------------- L11 monoliths
def monoliths(sc, T):
    L = T["L"][11]
    n = len(L["chars"])
    parts = []
    for k, c in enumerate(L["chars"]):
        x = (k - (n - 1) / 2) * 3.0
        y = 6.0 + 0.10 * x * x
        part = C.glyph(sc, c["c"], f"mono_{k}", size=2.8, extrude=0.35, bevel=0.03, line_colour=C.ORANGE,
                       prefix="mono", radius=0.004, construct=True)
        root, fill, trace, mod = part
        root.rotation_euler = (0, 0, math.atan2(x, 9.0) * 0.6)
        land = c["s"]
        fall = 0.42
        for u in np.linspace(0, 1, 11):                   # gravity: slow at the top, fastest on impact
            root.location = (x, y, 1.45 + 26.0 * (1 - u) ** 2)
            root.keyframe_insert("location", frame=C.frame(land - fall + fall * u))
        root.location = (x, y, 1.45)
        root.keyframe_insert("location", frame=C.frame(land + 0.05))
        C.write_skeleton(root, land - fall, land, fade=0.7)  # its source compiles as it falls
        C.gn_key(trace, mod, "End", land - fall - 0.05, 1.0)
        C.key(trace, '["glow"]', land - fall, 3.0)
        C.key(trace, '["glow"]', land, 14.0)
        C.key(trace, '["glow"]', land + 0.5, 4.0)
        C.key(fill, '["alpha"]', land - fall, 1.0)
        C.key(fill, '["glow"]', land, 0.25)
        C.key(fill, '["glow"]', land + 0.4, 0.04)
        # L12: they rise again and burn away into the ribbons, one after another
        up = T["l12"] + 0.25 * k
        root.keyframe_insert("location", frame=C.frame(up))
        root.location = (x, y, 1.45 + 2.2)
        root.keyframe_insert("location", frame=C.frame(up + 1.6))
        C.key(fill, '["burn"]', up + 0.3, 0.0)
        C.key(fill, '["burn"]', up + 1.5, 1.0)
        C.key(trace, '["glow"]', up + 0.3, 4.0)
        C.key(trace, '["glow"]', up + 1.3, 0.0)
        C.retire(root, up + 1.6)
        shockwave(sc, Vector((x, y - 0.4, 0.004)), land, f"shock_{k}", C.ORANGE, 5.0)
        parts.append(part)
    return parts


def shockwave(sc, pos, t, name, colour, reach, gain=12.0):
    """A ring of light racing out over the floor from an impact."""
    mat = C.shared(f"shock_mat_{name[:5]}", lambda: C.emission_material(
        f"shock_mat_{name[:5]}", colour, 1.0, attr_strength=("OBJECT", "glow")))
    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.012, major_segments=160, minor_segments=6,
                                     location=pos)
    ring = bpy.context.active_object
    ring.name = name
    for col in list(ring.users_collection):
        col.objects.unlink(ring)
    sc.collection.objects.link(ring)
    ring.data.materials.append(mat)
    ring.visible_shadow = False
    ring["glow"] = 0.0
    C.key(ring, '["glow"]', t - 0.02, 0.0)
    for k in range(0, 25):
        a = k / 24 * 1.2
        r = 0.2 + reach * (1 - (1 - a / 1.2) ** 3)
        ring.scale = (r, r, 1 + 2 * a)
        ring.keyframe_insert("scale", frame=C.frame(t + a))
        C.key(ring, '["glow"]', t + a, gain * (1 - a / 1.2) ** 2)
    shown(ring, ((t - 0.04, t + 1.22),))                     # spent rings would stay as black hoops
    return ring


# ---------------------------------------------------------------- L12 carousel + cocoon of code
def carousel(sc, T, cam_angle):
    L = T["L"][12]
    ring = bpy.data.objects.new("carousel", None)
    C.link(sc, ring)
    ring.location = (0, 0, 1.18)
    omega = math.radians(-72)                                   # degrees per second, turning to the left
    t0, t1 = T["l12"] - 0.3, T["s7"] + 0.2
    for t in (t0, t1):
        ring.rotation_euler = (0, 0, omega * (t - t0))
        ring.keyframe_insert("rotation_euler", frame=C.frame(t))
    C.linear(ring, "rotation")
    parts = []
    for k, c in enumerate(L["chars"]):
        s = c["s"]
        # at its syllable the letter faces the camera from beside her (not across her): its angle on the ring
        # cancels the ring's turn
        world_ang = cam_angle(s) + (0.92 if k % 2 == 0 else -0.92)
        local = world_ang - omega * (s - t0)
        part = C.glyph(sc, c["c"], f"ring_{k}", size=0.36, extrude=0.03, bevel=0.006, line_colour=C.CREAM,
                       prefix="ring")
        root = part[0]
        root.parent = ring
        r = 1.6
        root.location = (math.sin(local) * r, -math.cos(local) * r, 0.0)
        root.rotation_euler = (0, 0, local)
        C.write_glyph(part, s, s + 0.28, glow_line=12.0, glow_fill=0.06, t_out=T["l12_end"] + 0.25 + 0.06 * k, out_dur=0.7)
        parts.append(part)
    return parts


def code_silk_group():
    """A ribbon swept along a curve (standing up, Z-up normals), growing to `End`; it carries u_len (metres along
    the untrimmed curve, so the code does not slide as it grows) and v_across (0..1 across) for the shader."""
    g = C.GN("code_silk_gn")
    end = g.input("End", 0.0)
    width = g.input("Width", 0.02)
    mat = g.input("Material", None, kind="NodeSocketMaterial")
    spar = g.node("GeometryNodeSplineParameter")
    st = g.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    g.link(g.geometry(), C.sock(st, "Geometry"))
    C.sock(st, "Name").default_value = "u_len"
    g.link(C.sock(spar, "Length", True), next(s for s in st.inputs if s.name == "Value" and s.enabled))
    nrm = g.node("GeometryNodeSetCurveNormal")
    g.link(C.sock(st, "Geometry", True), C.sock(nrm, "Curve"))
    C.sock(nrm, "Mode").default_value = "Z Up"
    trim = g.node("GeometryNodeTrimCurve", mode="FACTOR")
    g.link(C.sock(nrm, "Curve", True), C.sock(trim, "Curve"))
    g.link(end, next(s for s in trim.inputs if s.enabled and s.name == "End"))
    half = g.math("MULTIPLY", width, 0.5)
    prof = g.node("GeometryNodeCurvePrimitiveLine", mode="POINTS")
    g.link(g.xyz(g.math("MULTIPLY", half, -1.0), 0.0, 0.0), C.sock(prof, "Start"))
    g.link(g.xyz(half, 0.0, 0.0), C.sock(prof, "End"))
    spar2 = g.node("GeometryNodeSplineParameter")
    st2 = g.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    g.link(C.sock(prof, "Curve", True), C.sock(st2, "Geometry"))
    C.sock(st2, "Name").default_value = "v_across"
    g.link(C.sock(spar2, "Factor", True), next(s for s in st2.inputs if s.name == "Value" and s.enabled))
    c2m = g.node("GeometryNodeCurveToMesh")
    g.link(C.sock(trim, "Curve", True), C.sock(c2m, "Curve"))
    g.link(C.sock(st2, "Geometry", True), C.sock(c2m, "Profile Curve"))
    sm = g.node("GeometryNodeSetMaterial")
    g.link(C.sock(c2m, "Mesh", True), C.sock(sm, "Geometry"))
    g.link(mat, C.sock(sm, "Material"))
    return g.finish(C.sock(sm, "Geometry", True))


def code_silk_material():
    """Running code: the atlas of monospace rows (assets/art/code_atlas.png), one row per ribbon ('row'), scrolling
    along it ('scroll', characters), glowing ('glow'), with a faint body of light between the characters."""
    m, g = C.material("code_silk_mat", "BLENDED")
    img = bpy.data.images.load(str(A.ART / "code_atlas.png"), check_existing=True)
    img.colorspace_settings.name = "Non-Color"
    u_len = g.attr("u_len", "GEOMETRY")
    v = g.attr("v_across", "GEOMETRY")
    cols, rows = 160, 32
    U = g.math("DIVIDE", g.math("ADD", g.math("DIVIDE", u_len, 0.018), g.attr("scroll")), cols)
    V = g.math("SUBTRACT", 1.0, g.math("DIVIDE", g.math("SUBTRACT", g.math("ADD", g.attr("row"), 1.0), v), rows))
    uv = g.node("ShaderNodeCombineXYZ")
    g.link(U, uv.inputs[0])
    g.link(V, uv.inputs[1])
    tex = g.node("ShaderNodeTexImage", interpolation="Linear", extension="REPEAT")
    tex.image = img
    g.link(uv.outputs[0], C.sock(tex, "Vector"))
    ink = C.sock(tex, "Color", True)
    bw = g.node("ShaderNodeRGBToBW")
    g.link(ink, bw.inputs[0])
    edge = g.math("MULTIPLY", g.smoothstep(v, 0.0, 0.08), g.smoothstep(g.math("SUBTRACT", 1.0, v), 0.0, 0.08))
    amt = g.math("MULTIPLY", g.math("ADD", g.math("MULTIPLY", bw.outputs[0], 1.0), 0.10), edge)
    em = g.node("ShaderNodeEmission")
    C.sock(em, "Color").default_value = (1.0, 0.70, 0.44, 1)
    g.link(g.math("MULTIPLY", amt, g.math("MULTIPLY", g.attr("glow"), 2.2)), C.sock(em, "Strength"))
    tr = g.node("ShaderNodeBsdfTransparent")
    mx = g.node("ShaderNodeMixShader")
    g.link(g.math("MINIMUM", g.math("MULTIPLY", amt, 1.6), 1.0), mx.inputs[0])
    g.link(tr.outputs[0], mx.inputs[1])
    g.link(em.outputs[0], mx.inputs[2])
    g.link(mx.outputs[0], C.sock(g.node("ShaderNodeOutputMaterial"), "Surface"))
    return m


def cocoon(sc, T):
    """One ribbon of running code per L12 syllable winds round her and closes; at the end of the line the cocoon
    flashes."""
    L = T["L"][12]
    rng = np.random.default_rng(5)
    mat = code_silk_material()
    group = code_silk_group()
    obs = []
    for k, c in enumerate(L["chars"]):
        cv = bpy.data.curves.new(f"ribbon_{k}", "CURVE")
        cv.dimensions = "3D"
        sp = cv.splines.new("POLY")
        n = 260
        sp.points.add(n - 1)
        turns = rng.uniform(1.6, 2.6)
        z0, z1 = rng.uniform(0.0, 0.5), rng.uniform(1.5, 2.2)
        ph = rng.uniform(0, 2 * math.pi)
        tilt = rng.uniform(-0.25, 0.25)
        for j in range(n):
            u = j / (n - 1)
            a = ph + 2 * math.pi * turns * u * (1 if k % 2 == 0 else -1)
            r = 1.75 - 0.6 * u + 0.08 * math.sin(7 * u + k)
            z = z0 + (z1 - z0) * u + tilt * math.sin(a)
            sp.points[j].co = (r * math.cos(a), r * math.sin(a), z, 1.0)
        ob = bpy.data.objects.new(f"ribbon_{k}", cv)
        C.link(sc, ob)
        ob.data.materials.append(mat)
        mod = ob.modifiers.new("gn", "NODES")
        mod.node_group = group
        C.gn_set(mod, "Width", 0.036)
        C.gn_set(mod, "Material", mat)
        s = c["s"]
        C.gn_key(ob, mod, "End", s - 0.05, 0.0)
        C.gn_key(ob, mod, "End", s + 0.9, 1.0)
        ob["row"] = float(rng.integers(0, 32))
        ob["scroll"] = 0.0
        C.key(ob, '["scroll"]', s - 0.1, 0.0, "LINEAR")
        C.key(ob, '["scroll"]', T["s7"] + 0.2, -(T["s7"] + 0.3 - s) * 14.0, "LINEAR")   # code runs along the silk
        ob["glow"] = 0.0
        C.key(ob, '["glow"]', s - 0.06, 0.0)
        C.key(ob, '["glow"]', s, 3.0)
        C.key(ob, '["glow"]', s + 1.0, 1.2)
        C.key(ob, '["glow"]', T["l12_end"], 1.2)
        C.key(ob, '["glow"]', T["s7"] - 0.25, 5.0)
        C.key(ob, '["glow"]', T["s7"] + 0.1, 0.0)
        shown(ob, ((s - 0.1, T["s7"] + 0.15),))
        obs.append(ob)
    # the closing flash: a shell of light around her
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.6, location=(0, 0, 1.0))
    shell = bpy.context.active_object
    shell.name = "cocoon_shell"
    for col in list(shell.users_collection):
        col.objects.unlink(shell)
    sc.collection.objects.link(shell)
    m, g = C.material("shell_mat", "BLENDED")
    lw = g.node("ShaderNodeLayerWeight")
    C.sock(lw, "Blend").default_value = 0.4
    rim = g.math("POWER", C.sock(lw, "Facing", True), 2.0)
    em = g.node("ShaderNodeEmission")
    C.sock(em, "Color").default_value = (1.0, 0.80, 0.58, 1)
    g.link(g.math("MULTIPLY", g.attr("glow"), g.math("ADD", 0.08, rim)), C.sock(em, "Strength"))
    tr = g.node("ShaderNodeBsdfTransparent")
    mx = g.node("ShaderNodeMixShader")
    g.link(g.math("MULTIPLY", g.attr("glow_a"), g.math("ADD", 0.15, rim)), mx.inputs[0])
    g.link(tr.outputs[0], mx.inputs[1])
    g.link(em.outputs[0], mx.inputs[2])
    g.link(mx.outputs[0], C.sock(g.node("ShaderNodeOutputMaterial"), "Surface"))
    shell.data.materials.append(m)
    shell["glow"], shell["glow_a"] = 0.0, 0.0
    C.key(shell, '["glow"]', T["l12_end"] - 0.1, 0.0)
    C.key(shell, '["glow"]', T["s7"] - 0.05, 5.0)
    C.key(shell, '["glow"]', T["s7"] + 0.5, 0.0)
    C.key(shell, '["glow_a"]', T["l12_end"] - 0.1, 0.0)
    C.key(shell, '["glow_a"]', T["s7"] - 0.05, 0.85)
    C.key(shell, '["glow_a"]', T["s7"] + 0.5, 0.0)
    shell.scale = (1, 1, 1)
    shell.keyframe_insert("scale", frame=C.frame(T["s7"] - 0.05))
    shell.scale = (2.4, 2.4, 0.01)                              # it unfolds flat into the board
    shell.keyframe_insert("scale", frame=C.frame(T["s7"] + 0.5))
    shown(shell, ((T["l12_end"] - 0.2, T["s7"] + 0.55),))
    return obs


# ---------------------------------------------------------------- L13 go stones, in her shadow
SGF = "abcdefghijklmnopqrs"


def stones(sc, T):
    L = T["L"][13]
    black, gb = C.material("stone_black")
    b = gb.node("ShaderNodeBsdfPrincipled")
    C.sock(b, "Base Color").default_value = (0.01, 0.01, 0.012, 1)
    C.sock(b, "Roughness").default_value = 0.08
    C.sock(b, "Coat Weight").default_value = 1.0
    gb.link(C.sock(b, "BSDF", True), C.sock(gb.node("ShaderNodeOutputMaterial"), "Surface"))
    white, gw = C.material("stone_white")
    w = gw.node("ShaderNodeBsdfPrincipled")
    C.sock(w, "Base Color").default_value = (0.9, 0.88, 0.84, 1)
    C.sock(w, "Roughness").default_value = 0.18
    C.sock(w, "Coat Weight").default_value = 1.0
    C.sock(w, "Subsurface Weight").default_value = 0.4
    gw.link(C.sock(w, "BSDF", True), C.sock(gw.node("ShaderNodeOutputMaterial"), "Surface"))
    label_mat = C.label_material()
    first = [c for c in L["chars"] if c["s"] < T["l13b"] - 0.05]
    second = [c for c in L["chars"] if c["s"] >= T["l13b"] - 0.05]
    cell = 0.45
    out = []
    for row, chars in enumerate((first, second)):
        n = len(chars)
        for k, c in enumerate(chars):
            idx = L["chars"].index(c)
            x = (k - (n - 1) / 2) * cell * 1.0
            y = -1.25 - row * cell * 1.15
            bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.19, location=(x, y, 0))
            st = bpy.context.active_object
            st.name = f"stone_{idx}"
            for col in list(st.users_collection):
                col.objects.unlink(st)
            sc.collection.objects.link(st)
            st.scale = (1, 1, 0.36)
            is_black = idx % 2 == 0
            st.data.materials.append(black if is_black else white)
            for poly in st.data.polygons:
                poly.use_smooth = True
            s = c["s"]
            for t, z in ((s - 0.35, 1.6), (s - 0.17, 0.75), (s, 0.069), (s + 0.08, 0.12), (s + 0.16, 0.069)):
                st.location = (x, y, z)
                st.keyframe_insert("location", frame=C.frame(t))
            shown(st, ((s - 0.36, T["s8"]),))
            # its letter, inlaid on top, reading from above
            part = C.glyph(sc, c["c"], f"stone_glyph_{idx}", size=0.25, extrude=0.004, bevel=0.0015,
                           line_colour=(1.0, 0.8, 0.45) if is_black else (0.25, 0.18, 0.12), prefix="stoneg" + ("b" if is_black else "w"),
                           radius=0.0035)
            root = part[0]
            root.rotation_euler = (math.radians(-90), 0, 0)          # lying on the stone, read from above
            for t, z in ((s - 0.35, 1.6), (s - 0.17, 0.75), (s, 0.069), (s + 0.08, 0.12), (s + 0.16, 0.069)):
                root.location = (x, y, z + 0.19 * 0.36 + 0.004)
                root.keyframe_insert("location", frame=C.frame(t))
            C.write_glyph(part, s + 0.05, s + 0.35, glow_line=6.0 if is_black else 2.5, glow_fill=0.5 if is_black else 0.05)
            shown(root, ((s - 0.36, T["s8"]),))
            for child in root.children:
                shown(child, ((s - 0.36, T["s8"]),))
            # the move, as a game record writes it: number, colour and board point in SGF
            gx = SGF[min(18, max(0, int(round(x / cell)) + 9))]
            gy = SGF[min(18, max(0, int(round(-y / cell)) + 3))]
            tc = bpy.data.curves.new(f"sgf_{idx}", "FONT")
            tc.body = f"{idx + 1:02d} {'B' if is_black else 'W'}[{gx}{gy}]"
            tc.font = C.font(C.FONT_MONO)
            tc.size = 0.062
            tc.align_x, tc.align_y = "LEFT", "CENTER"
            lab = bpy.data.objects.new(f"sgf_{idx}", tc)
            C.link(sc, lab)
            lab.location = (x + 0.215, y - 0.02, 0.003)
            lab.data.materials.append(label_mat)
            lab["glow"] = 0.0
            C.key(lab, '["glow"]', s + 0.02, 0.0)
            C.key(lab, '["glow"]', s + 0.12, 1.6)
            C.key(lab, '["glow"]', s + 0.8, 0.8)
            C.key(lab, '["glow"]', T["s8"] - 0.35, 0.8)
            C.key(lab, '["glow"]', T["s8"] - 0.05, 0.0)
            shown(lab, ((s - 0.02, T["s8"]),))
            shockwave(sc, Vector((x, y, 0.003)), s, f"ripple_{idx}", (0.8, 0.9, 1.0), 1.8, gain=5.0)
            out.append(st)
    return out


# ---------------------------------------------------------------- L14/L15 written in the air, blueprint
def blueprint(sc, T):
    """The pen's construction lines round her: a vertical axis, circles at her feet, a height line with ticks."""
    mat = C.emission_material("blueprint_mat", (0.45, 0.85, 1.0), 1.0, attr_strength=("OBJECT", "glow"))
    curves = []

    def poly(name, pts, t0, t1, radius=0.0035):
        cv = bpy.data.curves.new(name, "CURVE")
        cv.dimensions = "3D"
        sp = cv.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for j, p in enumerate(pts):
            sp.points[j].co = (*p, 1.0)
        ob = bpy.data.objects.new(name, cv)
        C.link(sc, ob)
        ob.data.materials.append(mat)
        mod = ob.modifiers.new("gn", "NODES")
        mod.node_group = C.trace_group()
        C.gn_set(mod, "Radius", radius)
        C.gn_set(mod, "Material", mat)
        C.gn_key(ob, mod, "End", t0, 0.0)
        C.gn_key(ob, mod, "End", t1, 1.0)
        ob["glow"] = 0.0
        C.key(ob, '["glow"]', t0 - 0.02, 0.0)
        C.key(ob, '["glow"]', t0, 5.0)
        C.key(ob, '["glow"]', T["s9"] + 1.5, 2.0)
        C.key(ob, '["glow"]', T["l15_end"], 0.0)
        shown(ob, ((t0 - 0.05, T["s10"]),))
        curves.append(ob)
        return ob
    s8 = T["s8"]
    poly("bp_axis", [(0, 0, -0.02), (0, 0, 1.95)], s8 + 0.05, s8 + 0.9)
    for k, r in enumerate((0.55, 1.1)):
        poly(f"bp_circle_{k}", [(r * math.cos(a), r * math.sin(a), 0.004) for a in np.linspace(0, 2 * math.pi, 129)],
             s8 + 0.2 + 0.3 * k, s8 + 1.4 + 0.3 * k, 0.0025)
    poly("bp_height", [(-0.75, 0, 1.66), (0.75, 0, 1.66)], s8 + 0.6, s8 + 1.2, 0.0025)
    for k, z in enumerate(np.linspace(0, 1.66, 9)):
        poly(f"bp_tick_{k}", [(-0.62, 0, z), (-0.52 if k % 2 else -0.47, 0, z)], s8 + 0.7 + 0.05 * k, s8 + 0.9 + 0.05 * k, 0.002)
    return curves


def column(sc, T, i, x, top, size, colour, prefix, t_out, tiles=False):
    """A vertical column of letters written by the pen, one per syllable, top to bottom: each shows its skeleton
    first; with tiles, its body then lands tile by tile like a renderer's buckets."""
    L = T["L"][i]
    parts = []
    for k, c in enumerate(L["chars"]):
        part = C.glyph(sc, c["c"], f"{prefix}_{k}", size=size, extrude=0.012, bevel=0.003, line_colour=colour,
                       prefix=prefix, radius=0.0045, construct=True)
        root = part[0]
        root.location = (x, -0.25, top - k * size * 1.08)
        root.rotation_euler = (0, 0, math.radians(-12 if x > 0 else 12))
        C.write_glyph(part, c["s"], c["s"] + 0.42, glow_line=7.0, glow_fill=0.3, t_out=t_out + 0.05 * k, out_dur=0.8)
        if tiles:
            fill = part[1]
            C.key(fill, '["tiles"]', c["s"] + 0.12, 0.0, "LINEAR")
            C.key(fill, '["tiles"]', c["s"] + 0.62, 1.05, "LINEAR")
            C.key(fill, '["tiles"]', c["s"] + 0.64, 2.0, "CONSTANT")
        parts.append(part)
    return parts


# ---------------------------------------------------------------- L16: where the mountains end
def mountains(sc, T, r0=14.0, r1=46.0, nr=90, nt=900, seed=3):
    """Mountains all round the horizon at dusk, drawn the way code draws terrain: a height field (ridged noise rising
    with distance), dark slopes, and contour lines of light every 45 cm of height that come on from far to near
    (S10 - S11)."""
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    rr = r0 + (r1 - r0) * np.linspace(0, 1, nr) ** 1.3
    TH, R = np.meshgrid(th, rr)
    X, Y = R * np.sin(TH), R * np.cos(TH)
    h = np.zeros_like(X)
    amp, freq = 1.0, 0.035
    for _ in range(6):                                          # ridged fBm from random plane waves
        for _ in range(3):
            d = rng.normal(0, 1, 2)
            d /= np.linalg.norm(d)
            h += amp * (1 - np.abs(np.sin(2 * np.pi * freq * (d[0] * X + d[1] * Y) + rng.uniform(0, 6.3)))) ** 2
        amp *= 0.5
        freq *= 2.1
    h = (h - h.min()) / np.ptp(h)
    rise = np.clip((R - r0) / 16.0, 0, 1) ** 1.6
    H = rise * (0.6 + 4.2 * h)
    verts = np.stack([X, Y, H], -1).reshape(-1, 3)
    faces = []
    for j in range(nr - 1):
        for i in range(nt):
            a, b = j * nt + i, j * nt + (i + 1) % nt
            faces.append((a, b, b + nt, a + nt))
    me = bpy.data.meshes.new("mountains")
    me.from_pydata([tuple(v) for v in verts], [], faces)
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new("mountains", me)
    C.link(sc, ob)
    m, g = C.material("mountains_mat")
    tc = g.node("ShaderNodeTexCoord")
    sep = g.node("ShaderNodeSeparateXYZ")
    g.link(C.sock(tc, "Object", True), sep.inputs[0])
    x, y, z = sep.outputs[0], sep.outputs[1], sep.outputs[2]
    contour = g.smoothstep(g.math("ABSOLUTE", g.math("SUBTRACT", g.math("FRACT", g.math("DIVIDE", z, 0.8)), 0.5)), 0.455, 0.492)
    dist = g.math("SQRT", g.math("ADD", g.math("MULTIPLY", x, x), g.math("MULTIPLY", y, y)))
    reach = g.smoothstep(g.math("SUBTRACT", dist, g.attr("near")), -1.5, 0.0)      # lines on from far to near
    reach = g.math("MULTIPLY", reach, g.math("SUBTRACT", 1.0, g.smoothstep(dist, 26.0, 44.0)))   # the far ones fade
    lift = g.smoothstep(z, 0.15, 0.6)                                               # none on the flat foothills
    bsdf = g.node("ShaderNodeBsdfPrincipled")
    C.sock(bsdf, "Base Color").default_value = (0.010, 0.008, 0.012, 1)
    C.sock(bsdf, "Roughness").default_value = 0.85
    C.sock(bsdf, "Emission Color").default_value = (1.0, 0.60, 0.28, 1)
    g.link(g.math("MULTIPLY", g.math("MULTIPLY", g.math("MULTIPLY", contour, reach), lift), g.attr("glow")),
           C.sock(bsdf, "Emission Strength"))
    g.link(C.sock(bsdf, "BSDF", True), C.sock(g.node("ShaderNodeOutputMaterial"), "Surface"))
    me.materials.append(m)
    ob["glow"], ob["near"] = 0.0, 60.0
    C.key(ob, '["glow"]', T["s10"], 0.7)
    C.key(ob, '["near"]', T["s10"], 50.0)
    C.key(ob, '["near"]', T["s10"] + 2.6, r0)
    shown(ob, ((T["s10"], T1 + 1),))
    return ob


# ---------------------------------------------------------------- L16 lanterns, petals
def lanterns(sc, T):
    L = T["L"][16]
    parts = []
    rng = np.random.default_rng(16)
    for k, c in enumerate(L["chars"]):
        part = C.glyph(sc, c["c"], f"lantern_{k}", size=0.21, extrude=0.02, bevel=0.004, font_path=C.FONT_SEMI,
                       line_colour=(1.0, 0.45, 0.14), prefix="lantern", radius=0.004)
        root, fill, trace, mod = part
        s = c["s"]
        x0 = 1.0 + 0.06 * math.sin(k * 1.7)
        y0 = -0.35 + 0.05 * rng.standard_normal()
        for t in np.arange(s - 0.05, T1 + 0.05, 0.25):
            a = max(0.0, t - s)
            root.location = (x0 + 0.04 * math.sin(1.3 * a + k), y0, 0.95 + 0.20 * a + 0.4 * (1 - math.exp(-a * 1.4)))
            root.keyframe_insert("location", frame=C.frame(t))
        root.rotation_euler = (0, 0, math.radians(-18))
        C.write_glyph(part, s, s + 0.3, glow_line=5.0, glow_fill=1.4, t_out=T["l16_end"] + 0.4 + 0.07 * k, out_dur=1.0)
        parts.append(part)
    return parts


def petals(sc, T):
    """Petals of her sunburst flower in a slow vortex round her, rising (S10 - S11)."""
    rng = np.random.default_rng(21)
    n = 480
    ang = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
    rad = rng.uniform(0.5, 3.2, n).astype(np.float32)
    z0 = rng.uniform(-0.2, 2.6, n).astype(np.float32)
    spin = rng.uniform(0.25, 0.7, n).astype(np.float32)
    rise = rng.uniform(0.05, 0.25, n).astype(np.float32)
    me = C.points_mesh("petal_pts", np.zeros((n, 3), np.float32),
                       dict(ang=ang, rad=rad, z0=z0, spin=spin, rise=rise, seed=rng.uniform(0, 100, n).astype(np.float32)))
    g = C.GN("petals_gn")
    tt = g.input("T", 0.0)
    a = g.math("ADD", g.named("ang"), g.math("MULTIPLY", g.named("spin"), tt))
    r = g.named("rad")
    z = g.math("ADD", g.named("z0"), g.math("MULTIPLY", g.named("rise"), tt))
    z = g.math("WRAP", z, 3.0, -0.2)
    pos = g.xyz(g.math("MULTIPLY", r, g.math("COSINE", a)), g.math("MULTIPLY", r, g.math("SINE", a)), z)
    sp = g.node("GeometryNodeSetPosition")
    g.link(g.geometry(), C.sock(sp, "Geometry"))
    g.link(pos, C.sock(sp, "Position"))
    petal = g.node("GeometryNodeMeshGrid")
    C.sock(petal, "Size X").default_value = 0.022
    C.sock(petal, "Size Y").default_value = 0.009
    C.sock(petal, "Vertices X").default_value = 3
    C.sock(petal, "Vertices Y").default_value = 2
    inst = g.node("GeometryNodeInstanceOnPoints")
    g.link(C.sock(sp, "Geometry", True), C.sock(inst, "Points"))
    g.link(C.sock(petal, "Mesh", True), C.sock(inst, "Instance"))
    rot = g.random("FLOAT_VECTOR", (0, 0, 0), (6.28, 6.28, 6.28), 9, None)
    rot = g.vmath("ADD", rot, g.xyz(g.math("MULTIPLY", tt, 1.3), g.math("MULTIPLY", tt, 0.7), 0.0))
    e2r = g.node("FunctionNodeEulerToRotation")
    g.link(rot, e2r.inputs[0])
    g.link(e2r.outputs[0], C.sock(inst, "Rotation"))
    real = g.node("GeometryNodeRealizeInstances")
    g.link(C.sock(inst, "Instances", True), C.sock(real, "Geometry"))
    m, gm = C.material("petal_mat")
    bsdf = gm.node("ShaderNodeBsdfPrincipled")
    C.sock(bsdf, "Base Color").default_value = (1.0, 0.42, 0.10, 1)
    C.sock(bsdf, "Subsurface Weight").default_value = 0.6
    C.sock(bsdf, "Emission Color").default_value = (1.0, 0.5, 0.15, 1)
    C.sock(bsdf, "Emission Strength").default_value = 0.6
    gm.link(C.sock(bsdf, "BSDF", True), C.sock(gm.node("ShaderNodeOutputMaterial"), "Surface"))
    sm = g.node("GeometryNodeSetMaterial")
    g.link(C.sock(real, "Geometry", True), C.sock(sm, "Geometry"))
    C.sock(sm, "Material").default_value = m
    ob, mod = C.gn_object(sc, "petals", g.finish(C.sock(sm, "Geometry", True)), mesh=me)
    C.gn_key(ob, mod, "T", T["s10"], 0.0)
    C.gn_key(ob, mod, "T", T1 + 1, T1 + 1 - T["s10"])
    C.linear(ob)
    shown(ob, ((T["s10"], T1 + 1),))
    return ob


# ---------------------------------------------------------------- camera
def catmull(pts, u):
    """Uniform Catmull-Rom through pts (Vectors) at u in [0, 1] (end points doubled)."""
    n = len(pts) - 1
    x = min(max(u, 0.0), 1.0) * n
    i = min(int(x), n - 1)
    f = x - i
    p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, n)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)


def camera_pose(T):
    s5, s6, s7, s8, s9, s10 = T["drop"], T["s6"], T["s7"], T["s8"], T["s9"], T["s10"]
    lock = T["lock"]
    eye = Vector(A.EYE)
    # S10: from the end of S9, back and round to the side of the cloud of shards, and home to my eye on "eye"
    path = [Vector((-0.45, -2.3, 1.35)), Vector((-1.55, -1.35, 1.5)), Vector((-2.35, -3.15, 1.15)), eye]
    aims = [Vector((0.05, 0.0, 1.25)), Vector((0.1, 1.0, 1.0)), Vector((0.0, 0.8, 0.9)), Vector((0.0, 0.0, eye.z))]
    lenses = [44.0, 30.0, 32.0, A.EYE_LENS]

    def orbit(ang, r, h, tgt):
        return Vector((math.sin(ang) * r, -math.cos(ang) * r, h)), Vector(tgt)

    def pose(t):
        if t < s5:                                   # looking up into the dark: the first letters fall past us
            u = C.smooth((t - T0) / (s5 - T0))
            pos = Vector((0.4, -9.5, 1.2))
            tgt = Vector((0.0, 6.0, 9.0 - 3.0 * u))
            return pos, tgt, 15.0, None, 24.0
        if t < s6:                                   # the drop: tilt down and rush in between the monoliths
            u = C.in_out_cubic((t - s5) / 0.55)
            v = C.out_cubic((t - s5) / (s6 - s5))
            pos = Vector((0.4 - 0.3 * v, -9.5 + 5.6 * v, 1.2 - 0.1 * v))
            tgt = Vector((0.0, 6.0, 9.0 - 3.0)).lerp(Vector((0.0, 0.0, 1.15)), u)
            return pos, tgt, (Vector((0, 0, 1.1)) - pos).length, 2.8, 24.0 + 8.0 * v
        if t < s7:                                   # the cocoon: a slow orbit to the left round her
            u = C.in_out_cubic((t - s6) / (s7 - s6))
            pos, tgt = orbit(math.radians(-6 - 40 * u), 3.9 - 0.4 * u, 1.35 + 0.25 * u, (0, 0, 1.12))
            return pos, tgt, (tgt - pos).length, 2.8, 34.0
        if t < s8:                                   # the board: from straight above, then down to its side
            u = C.in_out_cubic((t - s7) / (T["l13b"] - s7))
            v = C.in_out_cubic((t - T["l13b"]) / (s8 - T["l13b"]))
            yaw = math.radians(20 * u + 25 * v)
            hi = Vector((0.0, -0.8, 5.2 - 0.8 * u))
            lo = Vector((math.sin(-0.7 + yaw) * 3.6, -math.cos(-0.7 + yaw) * 3.6, 2.4))
            pos = hi.lerp(lo, v)
            tgt = Vector((0.0, -0.75, 0.0)).lerp(Vector((0.0, -0.9, 0.3)), v)
            up_fix = Vector((0.0, 0.002, 0.0)) if v < 0.02 else Vector()
            return pos + up_fix, tgt, (tgt - pos).length, 4.0, 32.0
        if t < s9:                                   # the pen: a slow lateral truck in the dark
            u = C.in_out_cubic((t - s8) / (s9 - s8))
            pos = Vector((-1.9 + 1.2 * u, -3.4, 1.25))
            tgt = Vector((-0.2 + 0.25 * u, 0.0, 1.05))
            return pos, tgt, (tgt - pos).length, 2.8, 36.0
        if t < s10:                                  # the fill: push in to a medium shot
            u = C.in_out_cubic((t - s9) / (s10 - s9))
            pos = Vector((-0.7 + 0.25 * u, -3.4 + 1.1 * u, 1.25 + 0.1 * u))
            tgt = Vector((0.05, 0.0, 1.05 + 0.2 * u))
            return pos, tgt, (tgt - pos).length, 2.4, 36.0 + 8.0 * u
        if t < lock:                                 # through the shards and home to my eye
            u = C.in_out_cubic((t - s10) / (lock - s10))
            pos, tgt = catmull(path, u), catmull(aims, u)
            k = min(u * 3, 2.999)
            i, f = int(k), k - int(k)
            lens = lenses[i] + (lenses[i + 1] - lenses[i]) * C.smooth(f)
            fstop = 2.8 + 61.2 * C.smooth((u - 0.80) / 0.18)            # no depth of field once home
            return pos, tgt, (Vector((0.0, 0.5, 0.9)) - pos).length, fstop, lens
        # home: from my eye only - turning and zooming keep her whole; a slow zoom up to her face
        w = C.smooth((t - lock - 0.5) / (T1 - lock - 0.5))
        tgt = Vector((0.0, 0.0, eye.z + 0.36 * w))
        return eye.copy(), tgt, 4.0, 64.0, A.EYE_LENS + 28.0 * w

    def cam_angle(t):
        p, _, _, _, _ = pose(t)
        return math.atan2(p.x, -p.y)

    return pose, cam_angle


# ---------------------------------------------------------------- build
def build():
    T = times()
    C.CONSTRUCT.clear()
    sc = C.new_scene("finale", T0, T1)
    bpy.context.window.scene = sc
    world(sc, T)
    floor(sc, T)
    lights(sc, T)
    pose, cam_angle = camera_pose(T)
    her(sc, T, pose)
    motes(sc, T)
    monoliths(sc, T)
    carousel(sc, T, cam_angle)
    cocoon(sc, T)
    stones(sc, T)
    blueprint(sc, T)
    column(sc, T, 14, 0.85, 1.75, 0.22, (0.45, 0.85, 1.0), "pen14", T["l15_end"] - 0.3)
    column(sc, T, 15, -0.85, 1.75, 0.22, (1.0, 0.75, 0.45), "pen15", T["l15_end"], tiles=True)
    lanterns(sc, T)
    petals(sc, T)
    mountains(sc, T)
    cam = C.camera(sc, "cam_finale")
    C.bake_camera(cam, T0, T1, pose)
    sc.frame_current = int(C.frame(T["drop"] + 1.0))
    return sc
