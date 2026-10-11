"""S1 - S4, the rain (31.1 - 46.3 s, lyric lines 6 - 10): she steps out of the ink into a real 3D night.

S1 (hit - L7) the LED dots of the scanned ink page fly head-first into her 3D figure under a street lamp; a scan
   seam runs down her and the points become her. The camera, at first exactly the 2D frame, dollies in and arcs.
S2 (L8)  low angle at her front-left, looking up past the umbrella into the lamp; slow crane up.
S3 (L9)  bullet time: on the first word the rain stops in the air; the camera orbits through the frozen drops.
S4 (L10) the rain falls upwards, faster and faster; she comes apart from the feet up into light that rises with it;
         the lamp flickers out on the last word.
Cuts land on line starts, never on drum hits. Everything is keyed in song time (frame = t * 24).
"""
import math

import bpy
import numpy as np
from mathutils import Vector

import bl_common as C

HER = Vector((-1.15, 0.0, 0.0))
LAMP = Vector((-0.75, 1.05, 4.7))
T0, T1 = 31.08, 46.35


def pose():
    """Her pose here; while the umbrella figure is still being generated, the reference pose stands in."""
    return "umbrella" if (C.MV / "umbrella" / "her.npz").exists() else "stand"


def times():
    tm = C.timing()
    L = {ln["i"]: ln for ln in tm["lines"]}
    import numpy as _np
    d = _np.load(C.WORK / "dots.npz")
    hit = float(d["hit"])
    return dict(hit=hit, l7=L[7]["start"], l8=L[8]["start"], l9=L[9]["start"], l9b=L[9]["chars"][6]["s"],
                l10=L[10]["start"], l10_last=L[10]["chars"][-1]["s"], l10_end=L[10]["end"], l11=L[11]["start"],
                L=L)


# ---------------------------------------------------------------- the world
def world(sc):
    w = sc.world
    try:
        w.use_nodes = True
    except AttributeError:
        pass
    nt = w.node_tree
    nt.nodes.clear()
    g = C.Graph(nt)
    bg = g.node("ShaderNodeBackground")
    C.sock(bg, "Color").default_value = (0.0012, 0.0022, 0.0055, 1)
    C.sock(bg, "Strength").default_value = 1.0
    vol = g.node("ShaderNodeVolumePrincipled")
    C.sock(vol, "Density").default_value = 0.0065
    C.sock(vol, "Anisotropy").default_value = 0.65
    C.sock(vol, "Color").default_value = (0.9, 0.92, 1.0, 1)
    out = g.node("ShaderNodeOutputWorld")
    g.link(bg.outputs[0], C.sock(out, "Surface"))
    g.link(C.sock(vol, "Volume", True), C.sock(out, "Volume"))
    sc.eevee.volumetric_end = 40.0


def floor(sc):
    me = bpy.data.meshes.new("floor")
    s = 40.0
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("floor", me)
    C.link(sc, ob)
    m, g = C.material("wet_asphalt")
    tc = g.node("ShaderNodeTexCoord")
    p = C.sock(tc, "Object", True)
    puddle = g.node("ShaderNodeTexNoise")
    C.sock(puddle, "Scale").default_value = 0.35
    C.sock(puddle, "Detail").default_value = 3.0
    g.link(p, C.sock(puddle, "Vector"))
    wet = g.smoothstep(C.sock(puddle, "Fac", True), 0.47, 0.53)
    rough = g.mix(0.32, 0.035, wet, kind="FLOAT")
    grain = g.node("ShaderNodeTexNoise")
    C.sock(grain, "Scale").default_value = 90.0
    C.sock(grain, "Detail").default_value = 6.0
    g.link(p, C.sock(grain, "Vector"))
    bump = g.node("ShaderNodeBump")
    g.link(C.sock(grain, "Fac", True), C.sock(bump, "Height"))
    g.link(g.mix(0.25, 0.02, wet, kind="FLOAT"), C.sock(bump, "Strength"))
    bsdf = g.node("ShaderNodeBsdfPrincipled")
    C.sock(bsdf, "Base Color").default_value = (0.010, 0.011, 0.014, 1)
    g.link(rough, C.sock(bsdf, "Roughness"))
    C.sock(bsdf, "Specular IOR Level").default_value = 0.7
    g.link(C.sock(bump, "Normal", True), C.sock(bsdf, "Normal"))
    out = g.node("ShaderNodeOutputMaterial")
    g.link(C.sock(bsdf, "BSDF", True), C.sock(out, "Surface"))
    me.materials.append(m)
    return ob


def lamp(sc):
    """A plain street lamp behind her: pole, arm, head with a glowing panel, and its spot light."""
    metal, g = C.material("lamp_metal")
    b = g.node("ShaderNodeBsdfPrincipled")
    C.sock(b, "Base Color").default_value = (0.02, 0.02, 0.022, 1)
    C.sock(b, "Metallic").default_value = 0.8
    C.sock(b, "Roughness").default_value = 0.35
    g.link(C.sock(b, "BSDF", True), C.sock(g.node("ShaderNodeOutputMaterial"), "Surface"))
    pole_x, pole_y = LAMP.x + 1.25, LAMP.y + 0.35
    parts = []
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.06, depth=LAMP.z + 0.15,
                                        location=(pole_x, pole_y, (LAMP.z + 0.15) / 2))
    parts.append(bpy.context.active_object)
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.035, depth=1.35,
                                        location=((pole_x + LAMP.x) / 2 + 0.05, (pole_y + LAMP.y) / 2, LAMP.z + 0.12),
                                        rotation=(0, math.radians(90), math.atan2(pole_y - LAMP.y, pole_x - LAMP.x)))
    parts.append(bpy.context.active_object)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(LAMP.x, LAMP.y, LAMP.z + 0.06))
    head = bpy.context.active_object
    head.scale = (0.42, 0.22, 0.08)
    parts.append(head)
    for ob in parts:
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        sc.collection.objects.link(ob)
        ob.data.materials.append(metal)
    panel_mat = C.emission_material("lamp_panel", (1.0, 0.72, 0.42), 60.0, attr_strength=("OBJECT", "on"))
    bpy.ops.mesh.primitive_plane_add(size=1, location=(LAMP.x, LAMP.y, LAMP.z - 0.0))
    panel = bpy.context.active_object
    panel.scale = (0.38, 0.18, 1)
    panel.rotation_euler = (math.radians(180), 0, 0)
    for c in list(panel.users_collection):
        c.objects.unlink(panel)
    sc.collection.objects.link(panel)
    panel.data.materials.append(panel_mat)
    panel["on"] = 1.0
    ld = bpy.data.lights.new("lamp_spot", "SPOT")
    ld.energy = 3200.0
    ld.color = C.SODIUM
    ld.spot_size = math.radians(56)
    ld.spot_blend = 0.35
    ld.shadow_soft_size = 0.12
    spot = bpy.data.objects.new("lamp_spot", ld)
    spot.location = LAMP - Vector((0, 0, 0.05))
    sc.collection.objects.link(spot)
    return panel, spot


def fill_lights(sc):
    """A cold moon fill from the front-left, so the silhouette keeps a little form; a low warm kicker behind."""
    sd = bpy.data.lights.new("moon", "SUN")
    sd.energy = 0.55
    sd.color = (0.50, 0.62, 1.0)
    sd.angle = math.radians(3)
    sun = bpy.data.objects.new("moon", sd)
    sun.rotation_euler = (math.radians(62), 0, math.radians(-35))
    sc.collection.objects.link(sun)
    ad = bpy.data.lights.new("kicker", "AREA")
    ad.energy = 45.0
    ad.specular_factor = 0.0                                # no rectangle mirrored in the wet floor
    ad.color = (1.0, 0.55, 0.3)
    ad.size = 1.2
    kick = bpy.data.objects.new("kicker", ad)
    kick.location = HER + Vector((0.4, 1.6, 0.5))
    kick.rotation_euler = (math.radians(95), 0, math.radians(165))
    sc.collection.objects.link(kick)
    bd = bpy.data.lights.new("backlight", "SPOT")              # behind and above her: the rim that makes the silhouette
    bd.energy = 900.0
    bd.color = (1.0, 0.66, 0.38)
    bd.spot_size = math.radians(24)
    bd.spot_blend = 0.6
    bd.shadow_soft_size = 0.25
    bd.volume_factor = 0.15
    back = bpy.data.objects.new("backlight", bd)
    back.location = HER + Vector((0.25, 3.2, 2.6))
    back.rotation_mode = "QUATERNION"
    back.rotation_quaternion = (HER + Vector((0, 0, 1.2)) - back.location).to_track_quat("-Z", "Y")
    sc.collection.objects.link(back)
    return sun, kick


def city(sc):
    """Far lights behind her - windows, other lamps, a crossing - that the shallow focus turns into bokeh."""
    rng = np.random.default_rng(12)
    pts, cols = [], []
    for _ in range(140):
        x, y = rng.uniform(-30, 26), rng.uniform(18, 48)
        z = rng.choice([rng.uniform(0.4, 3.0), rng.uniform(3, 14)], p=[0.55, 0.45])
        pts.append((x, y, z))
        warm = rng.random() < 0.72
        cols.append((1.0, rng.uniform(0.45, 0.75), rng.uniform(0.18, 0.35), 1.0) if warm else
                    (rng.uniform(0.45, 0.65), rng.uniform(0.7, 0.85), 1.0, 1.0))
    me = C.points_mesh("city_pts", np.array(pts, np.float32), dict(col=np.array(cols, np.float32),
                                                                  gain=rng.uniform(0.4, 1.0, len(pts)).astype(np.float32)))
    g = C.GN("city_gn")
    m2p = g.node("GeometryNodeMeshToPoints")
    g.link(g.geometry(), C.sock(m2p, "Mesh"))
    C.sock(m2p, "Radius").default_value = 0.09
    sm = g.node("GeometryNodeSetMaterial")
    g.link(C.sock(m2p, "Points", True), C.sock(sm, "Geometry"))
    m, gg = C.material("city_mat")
    em = gg.node("ShaderNodeEmission")
    gg.link(gg.attr("col", "GEOMETRY", "Color"), C.sock(em, "Color"))
    gg.link(gg.math("MULTIPLY", gg.math("MULTIPLY", gg.attr("gain", "GEOMETRY"), 9.0), gg.attr("on", "OBJECT")),
            C.sock(em, "Strength"))
    gg.link(em.outputs[0], C.sock(gg.node("ShaderNodeOutputMaterial"), "Surface"))
    C.sock(sm, "Material").default_value = m
    ob, _ = C.gn_object(sc, "city", g.finish(C.sock(sm, "Geometry", True)), mesh=me)
    ob["on"] = 1.0
    return ob


# ---------------------------------------------------------------- rain
def rain_group():
    """Rain: drops in a box falling at their own speeds, wrapped in height; R is rain time (it can stop and run
    backwards), S the streak length (1 falling, ~0 frozen). No rain under her umbrella."""
    g = C.GN("rain_gn", geometry_in=False)
    R = g.input("R", 0.0)
    S = g.input("S", 1.0)
    n = 9000
    idx = g.index()
    p0 = g.random("FLOAT_VECTOR", (-9.0, -6.0, 0.0), (7.0, 12.0, 7.5), 1, idx)
    speed = g.random("FLOAT", 5.5, 7.5, 2, idx)
    x, y, z = g.split(p0)
    zf = g.math("WRAP", g.math("SUBTRACT", z, g.math("MULTIPLY", speed, R)), 7.5, 0.0)
    xf = g.math("SUBTRACT", x, g.math("MULTIPLY", 0.035, zf))
    pos = g.xyz(xf, y, zf)
    pts = g.node("GeometryNodePoints")
    C.sock(pts, "Count").default_value = n
    g.link(pos, C.sock(pts, "Position"))
    # keep the space under her umbrella dry
    dx = g.math("SUBTRACT", xf, HER.x)
    dy = g.math("SUBTRACT", y, HER.y)
    r_xy = g.math("SQRT", g.math("ADD", g.math("MULTIPLY", dx, dx), g.math("MULTIPLY", dy, dy)))
    under = g.math("MULTIPLY", g.math("LESS_THAN", r_xy, 0.62), g.math("LESS_THAN", zf, 2.05))
    dele = g.node("GeometryNodeDeleteGeometry")
    g.link(C.sock(pts, "Geometry", True), C.sock(dele, "Geometry"))
    g.link(under, C.sock(dele, "Selection"))
    drop = g.node("GeometryNodeMeshUVSphere")
    C.sock(drop, "Segments").default_value = 8
    C.sock(drop, "Rings").default_value = 6
    C.sock(drop, "Radius").default_value = 0.0013
    store = g.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    g.link(C.sock(dele, "Geometry", True), C.sock(store, "Geometry"))
    C.sock(store, "Name").default_value = "rnd"
    g.link(g.random("FLOAT", 0.25, 1.0, 3, idx), [s_ for s_ in store.inputs if s_.name == "Value" and s_.enabled][0])
    inst = g.node("GeometryNodeInstanceOnPoints")
    g.link(C.sock(store, "Geometry", True), C.sock(inst, "Points"))
    g.link(C.sock(drop, "Mesh", True), C.sock(inst, "Instance"))
    bead = g.math("ADD", 1.0, g.math("MULTIPLY", g.math("SUBTRACT", 1.0, S), 1.6))   # frozen: a round bead
    stretch = g.math("ADD", 1.0, g.math("MULTIPLY", S, 120.0))
    g.link(g.xyz(bead, bead, g.math("MULTIPLY", stretch, bead)), C.sock(inst, "Scale"))
    real = g.node("GeometryNodeRealizeInstances")
    g.link(C.sock(inst, "Instances", True), C.sock(real, "Geometry"))
    sm = g.node("GeometryNodeSetMaterial")
    g.link(C.sock(real, "Geometry", True), C.sock(sm, "Geometry"))
    C.sock(sm, "Material").default_value = rain_material()
    return g.finish(C.sock(sm, "Geometry", True))


def rain_material():
    """Rain is light: faint everywhere, bright where the lamp's cone catches it."""
    m, g = C.material("rain_mat")
    geo = g.node("ShaderNodeNewGeometry")
    P = C.sock(geo, "Position", True)
    v = g.vmath("NORMALIZE", g.vmath("SUBTRACT", P, tuple(LAMP)))
    cosang = g.vmath("DOT_PRODUCT", v, (0.0, 0.0, -1.0))
    cone = g.smoothstep(cosang, math.cos(math.radians(30)), math.cos(math.radians(14)))
    dist = g.vmath("DISTANCE", P, tuple(LAMP))
    fall = g.math("DIVIDE", 1.0, g.math("ADD", 1.0, g.math("MULTIPLY", 0.12, g.math("MULTIPLY", dist, dist))))
    lit = g.math("MULTIPLY", g.math("MULTIPLY", cone, fall), g.math("MULTIPLY", 16.0, g.attr("on", "OBJECT")))
    strength = g.math("MULTIPLY", g.math("ADD", 0.035, lit), g.attr("rnd", "GEOMETRY"))
    em = g.node("ShaderNodeEmission")
    C.sock(em, "Color").default_value = (*C.CREAM, 1)
    g.link(strength, C.sock(em, "Strength"))
    g.link(em.outputs[0], C.sock(g.node("ShaderNodeOutputMaterial"), "Surface"))
    return m


def rain_clock(T):
    """Rain time R(t) and streak S(t): falling, frozen on L9, drifting up, then rushing upwards through L10."""
    def rate(t):
        if t < T["l9"]:
            return 1.0
        if t < T["l9"] + 0.35:                                   # brakes hard into the freeze
            return 1.0 - C.smooth((t - T["l9"]) / 0.35)
        if t < T["l9b"]:
            return 0.0
        if t < T["l10"]:
            return -0.10 * C.smooth((t - T["l9b"]) / 1.2)        # the drops begin to float back up
        return -0.10 - 2.6 * C.smooth((t - T["l10"]) / 3.2) ** 1.6
    out, R = [], T0
    f0, f1 = int(C.frame(T0)), int(C.frame(T1)) + 1
    for f in range(f0, f1 + 1):
        t = f / C.FPS
        if f > f0:
            R += rate(t) / C.FPS
        out.append((t, R, min(1.0, abs(rate(t)))))
    return out


# ---------------------------------------------------------------- the syllable drops
def syllable_drops(sc, T):
    """L7 and L8: each sung syllable lets one bright drop fall and ring out on the wet floor."""
    rng = np.random.default_rng(31)
    ring_mat = C.emission_material("ring_mat", C.CREAM, 1.4, attr_strength=("OBJECT", "glow"))
    drop_mat = C.emission_material("drop_mat", C.CREAM, 30.0, attr_strength=("OBJECT", "glow"))
    made = []
    for i in (7, 8):
        for c in T["L"][i]["chars"]:
            s = c["s"]
            pos = HER + Vector((rng.uniform(-1.6, 2.4), rng.uniform(-1.8, 1.6), 0.0))
            for lag, gain in ((0.0, 1.0),):
                bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.0025, major_segments=128, minor_segments=6,
                                                 location=pos + Vector((0, 0, 0.002)))
                ring = bpy.context.active_object
                for col in list(ring.users_collection):
                    col.objects.unlink(ring)
                sc.collection.objects.link(ring)
                ring.data.materials.append(ring_mat)
                ring["glow"] = 0.0
                for k in range(0, 31):
                    a = k / 24
                    t = s + lag + a
                    r = 0.05 + 0.75 * math.sqrt(a)
                    ring.scale = (r, r, 1.0)
                    ring.keyframe_insert("scale", frame=C.frame(t))
                    C.key(ring, '["glow"]', t, gain * (1 - a / 1.25) ** 2 if a > 0 else 0.0)
                C.key(ring, '["glow"]', s + lag - 0.05, 0.0)
                made.append(ring)
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.006, segments=8, ring_count=6, location=pos)
            drop = bpy.context.active_object
            for col in list(drop.users_collection):
                col.objects.unlink(drop)
            sc.collection.objects.link(drop)
            drop.data.materials.append(drop_mat)
            drop.scale = (1, 1, 9)
            drop["glow"] = 0.0
            for k in range(0, 9):
                u = k / 8
                t = s - 0.3 + 0.3 * u
                drop.location = pos + Vector((0, 0, 2.6 * (1 - u * u) + 0.03))
                drop.keyframe_insert("location", frame=C.frame(t))
                C.key(drop, '["glow"]', t, 1.0)
            C.key(drop, '["glow"]', s - 0.34, 0.0)
            C.key(drop, '["glow"]', s + 0.02, 0.0)
            made.append(drop)
    return made


# ---------------------------------------------------------------- her
def her(sc, T):
    ob = C.her_mesh(pose())
    ob.location = HER
    sc.collection.objects.link(ob)
    ob["rim"] = 0.0
    ob["ink"] = 0.55                     # her real colours: the darkness comes from having no front light
    # S1: invisible until the points have arrived, then a seam runs down her from the umbrella to the boots
    hit = T["hit"]
    top = 1.85
    C.key(ob, '["cut"]', T0, top + 0.2)
    C.key(ob, '["cut"]', hit + 1.15, top + 0.2)
    C.key(ob, '["cut"]', hit + 2.25, -0.08)
    C.key(ob, '["cut_glow"]', hit + 1.1, 1.0)
    C.key(ob, '["cut_glow"]', hit + 2.3, 1.0)
    C.key(ob, '["cut_glow"]', hit + 2.6, 0.0)
    C.key(ob, '["cut_soft"]', T0, 0.035)
    # S4: she exists above the cut, so a rising cut takes her apart from the boots up, into the rising rain
    c0, c1 = rise_times(T)
    C.key(ob, '["cut"]', c0, -0.08)
    C.key(ob, '["cut"]', c1, top + 0.2, "LINEAR")
    C.key(ob, '["cut_glow"]', c0 - 0.2, 0.0)
    C.key(ob, '["cut_glow"]', c0 + 0.1, 0.7)
    C.key(ob, '["cut_glow"]', c1, 0.7)
    return ob


def rise_times(T):
    """When the rising cut leaves her feet and when it passes the top of the umbrella."""
    return T["l10"] + 0.25, T["l10_last"] + 0.25


def her_points(sc, T):
    """The dot matrix of the ink page flying head-first into her (S1) as points of light."""
    d = np.load(C.WORK / "dots.npz")
    hp = C.her_points(pose())
    rng = np.random.default_rng(4)
    n = 70000
    pick = rng.choice(len(hp["xyz"]), n, replace=False)
    end = hp["xyz"][pick] + np.array(HER, np.float32)
    col = hp["rgb"][pick]
    dots = d["xyz"]
    # head first: rank the targets by height, the dots by their height on the page
    order_e = np.argsort(-end[:, 2] + 0.15 * end[:, 0])
    order_s = np.argsort(-dots[:, 2] + 0.15 * dots[:, 0])
    src = np.empty(n, np.int64)
    src[order_e] = order_s[(np.arange(n) * len(dots)) // n]
    start = dots[src] + rng.normal(0, 0.004, (n, 3)).astype(np.float32)
    height = (end[:, 2].max() - end[:, 2]) / np.ptp(end[:, 2])
    delay = (0.05 + 0.75 * height + rng.uniform(0, 0.12, n)).astype(np.float32)
    swirl = rng.normal(0, 1, (n, 3)).astype(np.float32)
    me = C.points_mesh("her_points", end, dict(start=start, delay=delay, swirl=swirl, endz=end[:, 2],
                                               col=np.concatenate([col, np.ones((n, 1), np.float32)], 1)))
    g = C.GN("her_points_gn")
    tt = g.input("T", 0.0)            # seconds since the hit
    cut = g.input("Cut", 3.0)         # points below the cut still show; above it she is solid
    pos = g.position()
    start_ = g.named("start", "FLOAT_VECTOR")
    delay_ = g.named("delay")
    sw = g.named("swirl", "FLOAT_VECTOR")
    u = g.smoothstep(g.math("DIVIDE", g.math("SUBTRACT", tt, delay_), 1.15), 0.0, 1.0)
    mid = g.math("SINE", g.math("MULTIPLY", u, math.pi))
    p = g.vmath("ADD", g.mix(start_, pos, u, kind="VECTOR"), g.vmath("SCALE", sw, scale=g.math("MULTIPLY", mid, 0.22)))
    sp = g.node("GeometryNodeSetPosition")
    g.link(g.geometry(), C.sock(sp, "Geometry"))
    g.link(p, C.sock(sp, "Position"))
    keep = g.math("LESS_THAN", g.named("endz"), g.math("ADD", cut, 0.02))
    m2p = g.node("GeometryNodeMeshToPoints")
    g.link(C.sock(sp, "Geometry", True), C.sock(m2p, "Mesh"))
    g.link(keep, C.sock(m2p, "Selection"))
    C.sock(m2p, "Radius").default_value = 0.0045
    sm = g.node("GeometryNodeSetMaterial")
    g.link(C.sock(m2p, "Points", True), C.sock(sm, "Geometry"))
    C.sock(sm, "Material").default_value = point_material()
    ob, mod = C.gn_object(sc, "her_points", g.finish(C.sock(sm, "Geometry", True)), mesh=me)
    hit = T["hit"]
    C.gn_key(ob, mod, "T", T0, T0 - hit)
    C.gn_key(ob, mod, "T", T0 + 20, T0 + 20 - hit)
    for kp in _fcurves(ob):
        for k in kp.keyframe_points:
            k.interpolation = "LINEAR"
    C.gn_key(ob, mod, "Cut", hit + 1.15, 2.05)
    C.gn_key(ob, mod, "Cut", hit + 2.25, -0.08)
    C.gn_key(ob, mod, "Cut", T0, 3.0)
    return ob


def point_material():
    """Shared by the points that become her and the points she becomes: made once (re-making it would take it away
    from the first user)."""
    m = bpy.data.materials.get("her_point_mat")
    if m is not None and m.get("built") == 3:          # bump when the definition below changes
        return m
    m, g = C.material("her_point_mat")
    m["built"] = 3
    col = g.attr("col", "GEOMETRY", "Color")
    em = g.node("ShaderNodeEmission")
    g.link(g.vmath("ADD", g.vmath("SCALE", col, scale=1.0), (0.25, 0.18, 0.12)), C.sock(em, "Color"))
    C.sock(em, "Strength").default_value = 1.5
    g.link(em.outputs[0], C.sock(g.node("ShaderNodeOutputMaterial"), "Surface"))
    return m


def _fcurves(ob):
    return C.fcurves(ob)


def her_rise(sc, T):
    """S4: where the rising cut passes, her points let go and rise with the rain, drifting apart as they go."""
    hp = C.her_points(pose())
    rng = np.random.default_rng(6)
    n = 90000
    pick = rng.choice(len(hp["xyz"]), n, replace=False)
    xyz = hp["xyz"][pick] + np.array(HER, np.float32)
    col = hp["rgb"][pick]
    c0, c1 = rise_times(T)
    top = 2.05
    release = (c0 + (xyz[:, 2] + 0.08) / (top + 0.08) * (c1 - c0) + rng.uniform(0, 0.08, n)).astype(np.float32)
    drift = rng.normal(0, 1, (n, 3)).astype(np.float32)
    speed = rng.uniform(0.6, 1.8, n).astype(np.float32)
    me = C.points_mesh("her_rise", xyz, dict(release=release, drift=drift, speed=speed,
                                             col=np.concatenate([col, np.ones((n, 1), np.float32)], 1)))
    g = C.GN("her_rise_gn")
    tt = g.input("T", 0.0)                                     # song time
    age = g.math("MAXIMUM", g.math("SUBTRACT", tt, g.named("release")), 0.0)
    up = g.math("MULTIPLY", g.math("MULTIPLY", age, age), g.math("MULTIPLY", g.named("speed"), 0.55))
    spread = g.vmath("SCALE", g.named("drift", "FLOAT_VECTOR"), scale=g.math("MULTIPLY", age, 0.12))
    p = g.vmath("ADD", g.vmath("ADD", g.position(), spread), g.xyz(0.0, 0.0, g.math("ADD", up, g.math("MULTIPLY", age, 0.35))))
    sp = g.node("GeometryNodeSetPosition")
    g.link(g.geometry(), C.sock(sp, "Geometry"))
    g.link(p, C.sock(sp, "Position"))
    live = g.math("GREATER_THAN", age, 0.0)
    m2p = g.node("GeometryNodeMeshToPoints")
    g.link(C.sock(sp, "Geometry", True), C.sock(m2p, "Mesh"))
    g.link(live, C.sock(m2p, "Selection"))
    g.link(g.math("MULTIPLY", 0.0045, g.math("SUBTRACT", 1.0, g.smoothstep(age, 1.2, 3.4))), C.sock(m2p, "Radius"))
    sm = g.node("GeometryNodeSetMaterial")
    g.link(C.sock(m2p, "Points", True), C.sock(sm, "Geometry"))
    C.sock(sm, "Material").default_value = point_material()
    ob, mod = C.gn_object(sc, "her_rise", g.finish(C.sock(sm, "Geometry", True)), mesh=me)
    C.gn_key(ob, mod, "T", T0, T0)
    C.gn_key(ob, mod, "T", T1 + 1, T1 + 1)
    for fc in _fcurves(ob):
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"
    return ob


# ---------------------------------------------------------------- camera
def camera_pose(T):
    hit = T["hit"]
    d = np.load(C.WORK / "dots.npz")
    cam0 = Vector(d["cam"].tolist())
    lens0 = 12.0 / math.tan(math.radians(float(d["fov"])) / 2)        # 38 deg vertical on a 24 mm sensor
    focus_pt = HER + Vector((0.0, 0.0, 1.2))

    def pose(t):
        if t < T["l8"]:                                         # S1: from the 2D frame into depth
            e = C.in_out_cubic((t - hit) / 2.6)
            s = max(0.0, t - hit)
            pivot = Vector((0.0, 0.0, 1.05)).lerp(HER + Vector((0.35, 0.0, 1.05)), e)
            yaw = math.radians(18) * e + math.radians(5) * math.sin(2 * math.pi * s / 9) * e
            r = 5.2 + 0.9 * e                                    # pull back while the lens lengthens: depth compresses
            pos = pivot + Vector((math.sin(yaw) * r, -math.cos(yaw) * r, 0.12 * e))
            lens = lens0 + (52.0 - lens0) * e
            if t <= hit:
                pos, pivot, lens = cam0, Vector((0.0, 0.0, 1.05)), lens0
            return pos, pivot, (focus_pt - pos).length, (2.0 if t > hit else None), lens
        if t < T["l9"]:                                         # S2: low and wide, up past her into the lamp
            u = C.in_out_cubic((t - T["l8"]) / (T["l9"] - T["l8"]))
            ang = math.radians(-32 + 12 * u)
            r = 2.2 - 0.2 * u
            pos = HER + Vector((math.sin(ang) * r, -math.cos(ang) * r, 0.28 + 0.5 * u))
            tgt = HER + Vector((0.25, 0.6, 1.75 + 0.2 * u))
            return pos, tgt, (HER + Vector((0, 0, 1.4)) - pos).length, 2.0, 24.0
        if t < T["l10"]:                                        # S3: bullet time, a long lens orbiting the frozen rain
            u = C.in_out_cubic((t - T["l9"]) / (T["l10"] - T["l9"]))
            ang = math.radians(-65 + 120 * u)
            r = 5.4
            pos = HER + Vector((math.sin(ang) * r, -math.cos(ang) * r, 1.45 + 0.15 * math.sin(math.pi * u)))
            tgt = HER + Vector((0.0, 0.0, 1.2))
            return pos, tgt, (tgt - pos).length, 1.4, 85.0
        u = C.in_out_cubic((t - T["l10"]) / (T1 - T["l10"]))       # S4: she rises with the rain; tilt up after her
        pos = HER + Vector((0.55, -3.6 + 0.5 * u, 0.85 + 0.6 * u))
        tgt = HER + Vector((0.0, 0.0, 1.0 + 5.5 * u ** 1.4))
        return pos, tgt, (HER + Vector((0, 0, 1.0 + 2.0 * u)) - pos).length, 2.2, 40.0
    return pose


def world_fade_in(sc, T, glowing, spot):
    """After the hit the world comes up out of the dark over ~1.4 s: the lamp first, then rain and the far lights;
    the frame Blender takes over from the 2D scan is the dot matrix in near darkness."""
    hit = T["hit"]
    for ob, (a, b) in zip(glowing, ((0.15, 0.9), (0.35, 1.5), (0.6, 1.8))):
        C.key(ob, '["on"]', hit, 0.0)
        C.key(ob, '["on"]', hit + a, 0.0)
        C.key(ob, '["on"]', hit + b, 1.0)
    lights = [o for o in sc.objects if o.type == "LIGHT"]
    for L in lights:
        E = L.data.energy if L is not spot else spot.data.energy
        if L is spot:
            fc = [f for f in C.fcurves(L.data) if f.data_path == "energy"]
            E = max((k.co[1] for f in fc for k in f.keyframe_points), default=E)
        for t, v in ((hit, 0.0), (hit + 0.15, 0.0), (hit + 1.2, 1.0)):
            L.data.energy = E * v
            L.data.keyframe_insert("energy", frame=C.frame(t))
        L.data.energy = E


# ---------------------------------------------------------------- build
def build():
    T = times()
    sc = C.new_scene("rain", T0, T1)
    bpy.context.window.scene = sc
    world(sc)
    floor(sc)
    panel, spot = lamp(sc)
    fill_lights(sc)
    city_ob = city(sc)
    ob_rain, mod = C.gn_object(sc, "rain", rain_group())
    ob_rain["on"] = 1.0
    for t, R, S in rain_clock(T):
        C.gn_key(ob_rain, mod, "R", t, R)
        C.gn_key(ob_rain, mod, "S", t, S)
    syllable_drops(sc, T)
    h = her(sc, T)
    her_points(sc, T)
    her_rise(sc, T)
    cam = C.camera(sc, "cam_rain")
    C.bake_camera(cam, T0, T1, camera_pose(T))
    # the lamp gutters out on the last word of L10
    for ob in (panel, ob_rain):
        C.key(ob, '["on"]', T["l10_last"], 1.0)
        for k, (dt, v) in enumerate(((0.08, 0.15), (0.16, 0.9), (0.24, 0.05), (0.34, 0.7), (0.5, 0.0))):
            C.key(ob, '["on"]', T["l10_last"] + dt, v, "CONSTANT")
    E = spot.data.energy
    spot.data.keyframe_insert("energy", frame=C.frame(T["l10_last"]))
    for dt, v in ((0.08, 0.15), (0.16, 0.9), (0.24, 0.05), (0.34, 0.7), (0.5, 0.0)):
        spot.data.energy = E * v
        spot.data.keyframe_insert("energy", frame=C.frame(T["l10_last"] + dt))
    world_fade_in(sc, T, (panel, ob_rain, city_ob), spot)
    sc.frame_current = int(C.frame(T["hit"] + 3.0))
    return sc
