"""Shared helpers for the Blender scenes (run inside Blender 5.2, through the Blender MCP or `blender -b`).

Every scene is built from data: the song timing (work/blender/timing.json, from pv/blender_data.py) and her figures
(assets/mv/<pose>/her.npz + her.json, from pv/mv.py: geometry from Hunyuan3D-2mv, colour projected from the AI
four-view sheets). Frame f of every scene is song time f / 24 s, so each frame is still a function of song time.
"""
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work" / "blender"
MV = ROOT / "assets" / "mv"
FONTS = ROOT / "assets" / "fonts"
FPS = 24

ORANGE = (1.0, 0.42, 0.12)
CREAM = (1.0, 0.86, 0.70)
TERRA = (0.85, 0.47, 0.34)
SODIUM = (1.0, 0.62, 0.28)


def frame(t):
    return t * FPS


def timing():
    return json.loads((WORK / "timing.json").read_text(encoding="utf-8"))


def line(i):
    return next(ln for ln in timing()["lines"] if ln["i"] == i)


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def in_out_cubic(x):
    x = min(1.0, max(0.0, x))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def out_cubic(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


# ---------------------------------------------------------------- scenes
def new_scene(name, t0, t1):
    """A fresh scene covering song time [t0, t1] (frames at song time), Eevee, 1080p, linear EXR output."""
    old = bpy.data.scenes.get(name)
    if old is not None:
        for ob in list(old.collection.all_objects):
            if ob.users <= 1:
                bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.scenes.remove(old)
    sc = bpy.data.scenes.new(name)
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = int(math.floor(frame(t0))), int(math.ceil(frame(t1)))
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1920, 1080, 100
    try:
        sc.render.engine = "BLENDER_EEVEE"
    except TypeError:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    ee = sc.eevee
    ee.taa_render_samples = 48
    ee.use_raytracing = True
    ee.use_shadows = True
    ee.volumetric_tile_size = "2"
    ee.volumetric_samples = 96
    ee.use_volumetric_shadows = True
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5
    sc.render.film_transparent = False
    try:
        sc.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    im = sc.render.image_settings
    im.file_format = "OPEN_EXR"
    im.color_depth = "16"
    try:
        im.exr_codec = "DWAA"
    except TypeError:
        im.exr_codec = "ZIP"
    sc.render.filepath = str(WORK / "frames" / name / "f_")
    sc.render.use_file_extension = True
    world = bpy.data.worlds.new(f"{name}_world")
    sc.world = world
    return sc


def collection(sc, name):
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if col.name not in sc.collection.children:
        sc.collection.children.link(col)
    return col


def link(sc, ob, col=None):
    (col or sc.collection).objects.link(ob)
    return ob


# ---------------------------------------------------------------- nodes
def sock(node, name, out=False):
    """A node socket by identifier or name (names can be localised; identifiers cannot)."""
    socks = node.outputs if out else node.inputs
    for s in socks:
        if s.identifier == name:
            return s
    for s in socks:
        if s.name == name:
            return s
    raise KeyError(f"{node.bl_idname} has no {'output' if out else 'input'} {name!r}: "
                   f"{[s.identifier for s in socks]}")


class Graph:
    """Small builder for shader node trees: g.node(type, **props), g.link(a, b), g.math(op, a, b)."""

    def __init__(self, tree):
        self.t = tree
        self.x = 0

    def node(self, kind, **props):
        n = self.t.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = (self.x, 0)
        self.x += 40
        return n

    def link(self, a, b):
        self.t.links.new(a, b)

    def out(self, node, name=None):
        return node.outputs[0] if name is None else sock(node, name, True)

    def val(self, v):
        n = self.node("ShaderNodeValue")
        n.outputs[0].default_value = v
        return n.outputs[0]

    def _in(self, s, x):
        if isinstance(x, (int, float)):
            s.default_value = x
        elif isinstance(x, (tuple, list)):
            s.default_value = x
        else:
            self.link(x, s)

    def math(self, op, a, b=0.0, c=None, *, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=bool(clamp))
        self._in(n.inputs[0], a)
        self._in(n.inputs[1], b)
        if c is not None:
            self._in(n.inputs[2], c)
        return n.outputs[0]

    def vmath(self, op, a, b=(0, 0, 0), scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self._in(n.inputs[0], a)
        self._in(n.inputs[1], b)
        if scale is not None:
            self._in(sock(n, "Scale"), scale)
        return n.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs["Vector"]

    def mix(self, a, b, fac, kind="RGBA", blend="MIX"):
        n = self.node("ShaderNodeMix", data_type=kind, blend_type=blend)
        ins = {"FLOAT": ("A_Float", "B_Float"), "VECTOR": ("A_Vector", "B_Vector"), "RGBA": ("A_Color", "B_Color")}[kind]
        self._in(sock(n, "Factor_Float"), fac)
        self._in(sock(n, ins[0]), a)
        self._in(sock(n, ins[1]), b)
        outs = {"FLOAT": "Result_Float", "VECTOR": "Result_Vector", "RGBA": "Result_Color"}[kind]
        return sock(n, outs, True)

    def attr(self, name, kind="OBJECT", output="Fac"):
        n = self.node("ShaderNodeAttribute", attribute_type=kind, attribute_name=name)
        return sock(n, output, True)

    def smoothstep(self, x, e0, e1):
        n = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", clamp=True)
        self._in(sock(n, "Value"), x)
        self._in(sock(n, "From Min"), e0)
        self._in(sock(n, "From Max"), e1)
        return sock(n, "Result", True)


def material(name, render="DITHERED"):
    """A fresh, empty node material. Re-making a material that something still uses takes it away from that user:
    callers that share one must create it once (get-or-create)."""
    m = bpy.data.materials.get(name)
    if m is not None:
        bpy.data.materials.remove(m)
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except AttributeError:
        pass
    m.node_tree.nodes.clear()
    try:
        m.surface_render_method = render
    except (AttributeError, TypeError):
        pass
    return m, Graph(m.node_tree)


def emission_material(name, color, strength=1.0, attr_strength=None, attr_alpha=None):
    """Pure light; strength (and alpha) can come from an object property or a geometry attribute."""
    m, g = material(name, "BLENDED" if attr_alpha else "DITHERED")
    em = g.node("ShaderNodeEmission")
    sock(em, "Color").default_value = (*color, 1.0)
    s = g.val(strength)
    if attr_strength:
        kind, an = attr_strength
        s = g.math("MULTIPLY", s, g.attr(an, kind))
    g.link(s, sock(em, "Strength"))
    out = g.node("ShaderNodeOutputMaterial")
    if attr_alpha:
        kind, an = attr_alpha
        tr = g.node("ShaderNodeBsdfTransparent")
        mx = g.node("ShaderNodeMixShader")
        g.link(g.attr(an, kind), mx.inputs[0])
        g.link(tr.outputs[0], mx.inputs[1])
        g.link(em.outputs[0], mx.inputs[2])
        g.link(mx.outputs[0], sock(out, "Surface"))
    else:
        g.link(em.outputs[0], sock(out, "Surface"))
    return m


class GN(Graph):
    """Builder for a geometry-node modifier group: g = GN(name); x = g.input("T"); ...; g.finish(geometry)."""

    def __init__(self, name, geometry_in=True):
        old = bpy.data.node_groups.get(name)
        if old is not None:
            bpy.data.node_groups.remove(old)
        ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
        super().__init__(ng)
        self.ng = ng
        if geometry_in:
            ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        self.gi = ng.nodes.new("NodeGroupInput")
        self.go = ng.nodes.new("NodeGroupOutput")

    def input(self, name, default=0.0, kind="NodeSocketFloat"):
        item = self.ng.interface.new_socket(name=name, in_out="INPUT", socket_type=kind)
        if default is not None and hasattr(item, "default_value"):
            item.default_value = default
        return sock(self.gi, item.identifier, True)

    def geometry(self):
        return sock(self.gi, "Socket_0", True) if "Socket_0" in [s.identifier for s in self.gi.outputs] else self.gi.outputs[0]

    def named(self, name, kind="FLOAT"):
        n = self.node("GeometryNodeInputNamedAttribute", data_type=kind)
        sock(n, "Name").default_value = name
        return sock(n, "Attribute", True)

    def position(self):
        return self.node("GeometryNodeInputPosition").outputs[0]

    def index(self):
        return self.node("GeometryNodeInputIndex").outputs[0]

    def random(self, kind, lo, hi, seed, ident=None):
        n = self.node("FunctionNodeRandomValue", data_type=kind)
        ins = [s for s in n.inputs if s.enabled]
        mn = next(s for s in ins if s.name == "Min")
        mx = next(s for s in ins if s.name == "Max")
        mn.default_value, mx.default_value = lo, hi
        next(s for s in ins if s.name == "Seed").default_value = seed
        if ident is not None:
            self.link(ident, next(s for s in ins if s.name == "ID"))
        return next(s for s in n.outputs if s.enabled)

    def xyz(self, x, y, z):
        n = self.node("ShaderNodeCombineXYZ")
        for k, v in enumerate((x, y, z)):
            self._in(n.inputs[k], v)
        return n.outputs[0]

    def split(self, v):
        n = self.node("ShaderNodeSeparateXYZ")
        self._in(n.inputs[0], v)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def finish(self, geo):
        self.link(geo, sock(self.go, "Socket_0") if "Socket_0" in [s.identifier for s in self.go.inputs]
                  else self.go.inputs[0])
        return self.ng


def gn_object(sc, name, group, mesh=None, col=None):
    """An object carrying a geometry-nodes modifier with this group; returns (object, modifier)."""
    ob = bpy.data.objects.new(name, mesh or bpy.data.meshes.new(name + "_mesh"))
    link(sc, ob, col)
    mod = ob.modifiers.new("gn", "NODES")
    mod.node_group = group
    return ob, mod


def _gn_input(mod, socket_name):
    """(owner, attribute, data path) of a geometry-nodes modifier input: Blender 5 keeps the values in
    mod.properties.inputs.<identifier>.value; older versions in ID properties mod[identifier]."""
    ident = mod.node_group.interface.items_tree[socket_name].identifier
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        return getattr(props.inputs, ident), "value", f'modifiers["{mod.name}"].properties.inputs.{ident}.value'
    return mod, f'["{ident}"]', f'modifiers["{mod.name}"]["{ident}"]'


def gn_set(mod, socket_name, value):
    owner, attr, _ = _gn_input(mod, socket_name)
    if attr.startswith("["):
        owner[attr[2:-2]] = value
    else:
        setattr(owner, attr, value)


def gn_key(ob, mod, socket_name, t, value):
    """Keyframe a geometry-nodes modifier input (by its interface name) at song time t."""
    gn_set(mod, socket_name, value)
    _, _, path = _gn_input(mod, socket_name)
    ob.keyframe_insert(data_path=path, frame=frame(t))


def points_mesh(name, xyz, attrs=None):
    """A mesh of loose vertices carrying per-point attributes (float, vector or colour arrays)."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(xyz))
    me.vertices.foreach_set("co", np.asarray(xyz, np.float32).ravel())
    for key_, arr in (attrs or {}).items():
        arr = np.asarray(arr, np.float32)
        if arr.ndim == 1:
            a = me.attributes.new(key_, "FLOAT", "POINT")
            a.data.foreach_set("value", arr)
        elif arr.shape[1] == 3:
            a = me.attributes.new(key_, "FLOAT_VECTOR", "POINT")
            a.data.foreach_set("vector", arr.ravel())
        else:
            a = me.attributes.new(key_, "FLOAT_COLOR", "POINT")
            a.data.foreach_set("color", arr.ravel())
    me.update()
    return me


# ---------------------------------------------------------------- spatial calligraphy
FONT_BLACK = FONTS / "notoseriftc" / "NotoSerifTC-Black-lyrics.ttf"
FONT_SEMI = FONTS / "notoseriftc" / "NotoSerifTC-SemiBold-lyrics.ttf"


def font(path):
    for f in bpy.data.fonts:
        if Path(bpy.path.abspath(f.filepath)) == Path(path):
            return f
    return bpy.data.fonts.load(str(path), check_existing=False)


def trace_group():
    """Light-pen tracing of curves: trim every spline to [0, End] of its length, sweep a round profile."""
    ng = bpy.data.node_groups.get("trace_gn")
    if ng is not None and "Material" in ng.interface.items_tree:
        return ng
    g = GN("trace_gn")
    end = g.input("End", 0.0)
    radius = g.input("Radius", 0.006)
    mat = g.input("Material", None, kind="NodeSocketMaterial")
    trim = g.node("GeometryNodeTrimCurve", mode="FACTOR")
    g.link(g.geometry(), sock(trim, "Curve"))
    g.link(end, next(s for s in trim.inputs if s.enabled and s.name == "End"))
    circ = g.node("GeometryNodeCurvePrimitiveCircle", mode="RADIUS")
    sock(circ, "Resolution").default_value = 8
    g.link(radius, sock(circ, "Radius"))
    c2m = g.node("GeometryNodeCurveToMesh")
    g.link(sock(trim, "Curve", True), sock(c2m, "Curve"))
    g.link(sock(circ, "Curve", True), sock(c2m, "Profile Curve"))
    sm = g.node("GeometryNodeSetMaterial")                  # a curve's own slots do not reach the swept mesh
    g.link(sock(c2m, "Mesh", True), sock(sm, "Geometry"))
    g.link(mat, sock(sm, "Material"))
    return g.finish(sock(sm, "Geometry", True))


_MADE = set()                                               # shared materials made in this build (reloaded per build)


def shared(name, make):
    """A material shared by many objects: made fresh once per build (so code changes reach it), then reused."""
    if name not in _MADE or bpy.data.materials.get(name) is None:
        make()
        _MADE.add(name)
    return bpy.data.materials[name]


def glyph_materials(line_colour=CREAM, prefix="glyph"):
    """Fill: polished obsidian whose letter glows from inside ('glow', 'alpha', 'burn' object props; 'tiles' < 1
    lets it appear in square tiles, each flashing as it lands, like a bucket renderer); the trace line: pure light
    ('glow')."""
    fill = shared(f"{prefix}_fill", lambda: _glyph_fill(f"{prefix}_fill", line_colour))
    line = shared(f"{prefix}_line", lambda: emission_material(f"{prefix}_line", line_colour, 1.0,
                                                              attr_strength=("OBJECT", "glow")))
    return fill, line


def _glyph_fill(name, line_colour):
    """The letter's body: polished obsidian glowing from inside, appearing in tiles, burning away."""
    fill, g = material(name)
    bsdf = g.node("ShaderNodeBsdfPrincipled")
    sock(bsdf, "Base Color").default_value = (0.012, 0.012, 0.016, 1)
    sock(bsdf, "Roughness").default_value = 0.12
    sock(bsdf, "Specular IOR Level").default_value = 0.8
    sock(bsdf, "Emission Color").default_value = (*line_colour, 1)
    tc = g.node("ShaderNodeTexCoord")
    # tiles: 'tiles' sweeps 0 -> 1 over the tiles' random order
    sep = g.node("ShaderNodeSeparateXYZ")
    g.link(sock(tc, "Object", True), sep.inputs[0])
    cxy = g.node("ShaderNodeCombineXYZ")
    g.link(g.math("FLOOR", g.math("DIVIDE", sep.outputs[0], 0.016)), cxy.inputs[0])
    g.link(g.math("FLOOR", g.math("DIVIDE", sep.outputs[1], 0.016)), cxy.inputs[1])
    wn = g.node("ShaderNodeTexWhiteNoise", noise_dimensions="2D")
    g.link(cxy.outputs[0], sock(wn, "Vector"))
    tile_age = g.math("SUBTRACT", g.attr("tiles"), sock(wn, "Value", True))
    tile_on = g.math("GREATER_THAN", tile_age, 0.0)
    tile_flash = g.math("MULTIPLY", tile_on, g.math("EXPONENT", g.math("DIVIDE", tile_age, -0.05)))
    g.link(g.math("ADD", g.attr("glow"), g.math("MULTIPLY", tile_flash, 6.0)), sock(bsdf, "Emission Strength"))
    # burn away: a noise front eats the letter, its edge glowing
    nz = g.node("ShaderNodeTexNoise")
    sock(nz, "Scale").default_value = 6.0
    sock(nz, "Detail").default_value = 4.0
    g.link(sock(tc, "Object", True), sock(nz, "Vector"))
    burn = g.attr("burn")
    keep = g.smoothstep(g.math("SUBTRACT", sock(nz, "Fac", True), g.math("SUBTRACT", g.math("MULTIPLY", burn, 1.2), 0.1)),
                        0.0, 0.02)
    edge = g.math("SUBTRACT", g.smoothstep(g.math("SUBTRACT", sock(nz, "Fac", True),
                                                  g.math("SUBTRACT", g.math("MULTIPLY", burn, 1.2), 0.1)), -0.06, 0.0),
                  keep)
    em2 = g.node("ShaderNodeEmission")
    sock(em2, "Color").default_value = (*ORANGE, 1)
    g.link(g.math("MULTIPLY", g.math("ABSOLUTE", edge), 30.0), sock(em2, "Strength"))
    add = g.node("ShaderNodeAddShader")
    g.link(sock(bsdf, "BSDF", True), add.inputs[0])
    g.link(em2.outputs[0], add.inputs[1])
    a = g.math("MULTIPLY", g.attr("alpha"), g.math("MAXIMUM", keep, g.math("ABSOLUTE", edge)))
    g.link(g.math("MULTIPLY", a, tile_on), sock(bsdf, "Alpha"))
    g.link(add.outputs[0], sock(g.node("ShaderNodeOutputMaterial"), "Surface"))
    return fill


CONSTRUCT = {}                                              # root name -> (skeleton object, label object)
UI_CYAN = (0.55, 0.88, 1.0)
FONT_MONO = Path("C:/Windows/Fonts/consola.ttf")


def construct_group():
    """Edges of a mesh drawn as thin tubes of light (the skeleton of a glyph): Mesh to Curve, swept, material."""
    ng = bpy.data.node_groups.get("construct_gn")
    if ng is not None:
        return ng
    g = GN("construct_gn")
    radius = g.input("Radius", 0.002)
    mat = g.input("Material", None, kind="NodeSocketMaterial")
    m2c = g.node("GeometryNodeMeshToCurve")
    g.link(g.geometry(), sock(m2c, "Mesh"))
    circ = g.node("GeometryNodeCurvePrimitiveCircle", mode="RADIUS")
    sock(circ, "Resolution").default_value = 5
    g.link(radius, sock(circ, "Radius"))
    c2m = g.node("GeometryNodeCurveToMesh")
    g.link(sock(m2c, "Curve", True), sock(c2m, "Curve"))
    g.link(sock(circ, "Curve", True), sock(c2m, "Profile Curve"))
    sm = g.node("GeometryNodeSetMaterial")
    g.link(sock(c2m, "Mesh", True), sock(sm, "Geometry"))
    g.link(mat, sock(sm, "Material"))
    return g.finish(sock(sm, "Geometry", True))


def construct_material():
    """The skeleton glows dimly once shown ('glow'); each point flashes as the pen passes it ('reveal' vs the
    vertex attribute 'order')."""
    return shared("construct_mat", _construct_material)


def _construct_material():
    m, g = material("construct_mat", "BLENDED")
    em = g.node("ShaderNodeEmission")
    sock(em, "Color").default_value = (*UI_CYAN, 1)
    d = g.math("SUBTRACT", g.attr("reveal"), g.attr("order", "GEOMETRY"))
    hit = g.math("EXPONENT", g.math("MULTIPLY", g.math("MULTIPLY", g.math("DIVIDE", d, 0.06), g.math("DIVIDE", d, 0.06)), -1.0))
    passed = g.smoothstep(d, -0.02, 0.0)
    s = g.math("MULTIPLY", g.attr("glow"), g.math("ADD", g.math("ADD", 0.35, g.math("MULTIPLY", passed, 0.5)),
                                                  g.math("MULTIPLY", hit, 6.0)))
    g.link(s, sock(em, "Strength"))
    tr = g.node("ShaderNodeBsdfTransparent")
    mx = g.node("ShaderNodeMixShader")
    g.link(g.math("MINIMUM", g.attr("glow"), 1.0), mx.inputs[0])
    g.link(tr.outputs[0], mx.inputs[1])
    g.link(em.outputs[0], mx.inputs[2])
    g.link(mx.outputs[0], sock(g.node("ShaderNodeOutputMaterial"), "Surface"))
    return m


def label_material():
    """Monospace labels (codepoints, game records): pale cyan light, 'glow'."""
    return shared("label_mat", lambda: emission_material("label_mat", UI_CYAN, 2.2, attr_strength=("OBJECT", "glow")))


def construction(sc, name, cv, size, col=None, codepoint=None):
    """A font editor's view of the glyph whose outline is the curve cv (local XY): on-curve points as small squares,
    off-curve handles as small circles on hairlines (straight segments have none), the ink box with corner ticks and
    a centre cross, and the codepoint as a monospace label. Vertex attribute 'order' = how far along its contour the
    pen reaches the point (0..1; the box is -0.2: there before the pen). Returns (skeleton object, label object)."""
    verts, edges, order = [], [], []

    def poly(pts, closed, o):
        base = len(verts)
        for p in pts:
            verts.append((p[0], p[1], 0.0))
            order.append(o)
        n = len(pts)
        for i in range(n if closed else n - 1):
            edges.append((base + i, base + (i + 1) % n))

    a, r = 0.011 * size, 0.008 * size
    xs, ys = [], []
    for s in cv.splines:
        bp = s.bezier_points
        n = len(bp)
        seg = [(bp[(j + 1) % n].co - bp[j].co).length for j in range(n)]
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = max(cum[-1], 1e-6)
        for j, p in enumerate(bp):
            o = cum[j] / total
            x, y = p.co.x, p.co.y
            xs.append(x)
            ys.append(y)
            poly([(x - a, y - a), (x + a, y - a), (x + a, y + a), (x - a, y + a)], True, o)
            for hnd, nb in ((p.handle_left, bp[j - 1].co), (p.handle_right, bp[(j + 1) % n].co)):
                d = hnd - p.co
                if d.length < 0.03 * size:
                    continue
                sg = nb - p.co
                if sg.length > 1e-6 and abs(sg.x * d.y - sg.y * d.x) / (sg.length * d.length) < 0.03:
                    continue                                   # a straight segment: its handles are not drawn
                poly([(x, y), (hnd.x, hnd.y)], False, o)
                poly([(hnd.x + r * math.cos(k * math.pi / 4), hnd.y + r * math.sin(k * math.pi / 4)) for k in range(8)],
                     True, o)
    m = 0.05 * size
    x0, x1, y0, y1 = min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m
    poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], True, -0.2)
    t = 0.06 * size
    for cx, cy, sx, sy in ((x0, y0, -1, -1), (x1, y0, 1, -1), (x1, y1, 1, 1), (x0, y1, -1, 1)):
        poly([(cx + sx * t, cy), (cx, cy)], False, -0.2)
        poly([(cx, cy + sy * t), (cx, cy)], False, -0.2)
    mx_, my_ = (x0 + x1) / 2, (y0 + y1) / 2
    for k in range(9):                                         # a dashed centre cross
        u0, u1 = k / 9, (k + 0.45) / 9
        poly([(x0 + (x1 - x0) * u0, my_), (x0 + (x1 - x0) * u1, my_)], False, -0.2)
        poly([(mx_, y0 + (y1 - y0) * u0), (mx_, y0 + (y1 - y0) * u1)], False, -0.2)
    me = bpy.data.meshes.new(name + "_skel")
    me.from_pydata(verts, edges, [])
    at = me.attributes.new("order", "FLOAT", "POINT")
    at.data.foreach_set("value", np.array(order, np.float32))
    mat = construct_material()
    ob = bpy.data.objects.new(name + "_skel", me)
    link(sc, ob, col)
    mod = ob.modifiers.new("gn", "NODES")
    mod.node_group = construct_group()
    gn_set(mod, "Radius", max(0.0011, 0.0016 * size))
    gn_set(mod, "Material", mat)
    ob["glow"], ob["reveal"] = 0.0, 0.0
    lab = None
    if codepoint:
        tc = bpy.data.curves.new(name + "_label", "FONT")
        tc.body = codepoint
        tc.font = font(FONT_MONO)
        tc.size = max(0.022, 0.075 * size)
        tc.align_x, tc.align_y = "LEFT", "BOTTOM"
        lab = bpy.data.objects.new(name + "_label", tc)
        link(sc, lab, col)
        lab.location = (x0, y1 + 0.02 * size + 0.004, 0.0)
        lab.data.materials.append(label_material())
        lab["glow"] = 0.0
    return ob, lab


def glyph(sc, ch, name, size=1.0, extrude=0.08, bevel=0.012, font_path=FONT_BLACK, col=None, prefix="glyph",
          line_colour=CREAM, radius=0.006, construct=False):
    """One character as a 3D object in her world: root empty (animate this) > fill (extruded letter) + trace (its
    outlines as curves, drawn by the light pen) [+ its skeleton and codepoint label, when construct]. Upright, facing
    -Y, centred on the root. Returns (root, fill, trace, trace modifier); object props on fill: glow, alpha, burn,
    tiles; on trace: glow; modifier input End, Radius. The skeleton is in CONSTRUCT[root.name]."""
    fill_mat, line_mat = glyph_materials(line_colour, prefix)
    root = bpy.data.objects.new(name, None)
    root.empty_display_size = 0.2
    link(sc, root, col)
    cu = bpy.data.curves.new(name + "_fill", "FONT")
    cu.body = ch
    cu.font = font(font_path)
    cu.size = size
    cu.align_x, cu.align_y = "CENTER", "CENTER"
    cu.extrude, cu.bevel_depth, cu.bevel_resolution = extrude, bevel, 2
    fill = bpy.data.objects.new(name + "_fill", cu)
    link(sc, fill, col)
    fill.parent = root
    fill.rotation_euler = (math.radians(90), 0, 0)
    fill.data.materials.append(fill_mat)
    for k, v in dict(glow=0.0, alpha=0.0, burn=0.0, tiles=2.0).items():
        fill[k] = v
    # the outline as real curves, so a trim can draw it
    dg = bpy.context.evaluated_depsgraph_get()
    ev = fill.evaluated_get(dg)
    src = ev.to_curve(dg, apply_modifiers=False)
    cv = bpy.data.curves.new(name + "_trace", "CURVE")
    cv.dimensions = "3D"
    for s in src.splines:
        if s.type != "BEZIER":
            continue
        ns = cv.splines.new("BEZIER")
        ns.bezier_points.add(len(s.bezier_points) - 1)
        for a, b in zip(ns.bezier_points, s.bezier_points):
            a.handle_left_type = a.handle_right_type = "FREE"
            a.co, a.handle_left, a.handle_right = b.co, b.handle_left, b.handle_right
        ns.use_cyclic_u = s.use_cyclic_u
        ns.resolution_u = 10
    ev.to_curve_clear()
    trace = bpy.data.objects.new(name + "_trace", cv)
    link(sc, trace, col)
    trace.parent = root
    trace.rotation_euler = (math.radians(90), 0, 0)
    trace.location = (0, -(extrude + bevel) - 0.002, 0)          # on the letter's front face
    trace.data.materials.append(line_mat)
    trace["glow"] = 0.0
    mod = trace.modifiers.new("gn", "NODES")
    mod.node_group = trace_group()
    gn_set(mod, "End", 0.0)
    gn_set(mod, "Radius", radius * size)
    gn_set(mod, "Material", line_mat)
    if construct:
        skel, lab = construction(sc, name, cv, size, col, codepoint=f"U+{ord(ch):04X}")
        for o in (skel, lab):
            if o is not None:
                o.parent = root
                o.rotation_euler = (math.radians(90), 0, 0)
                o.location = (0, -(extrude + bevel) - 0.006, 0) if o is skel else \
                    (o.location.x, -(extrude + bevel) - 0.006, o.location.y)
        CONSTRUCT[root.name] = (skel, lab)
    return root, fill, trace, mod


def write_skeleton(root, t0, t1, fade=0.55):
    """The glyph's skeleton: shown just before the pen (its 'source'), each point flashing as the pen passes, then
    gone once the letter has lit."""
    skel, lab = CONSTRUCT.get(root.name, (None, None))
    if skel is None:
        return
    for o in (skel, lab):
        if o is None:
            continue
        key(o, '["glow"]', t0 - 0.36, 0.0)
        key(o, '["glow"]', t0 - 0.14, 1.0)
        key(o, '["glow"]', t1 + 0.18, 1.0)
        key(o, '["glow"]', t1 + 0.18 + fade, 0.0)
    key(skel, '["reveal"]', t0 - 0.01, 0.0, "LINEAR")
    key(skel, '["reveal"]', t1, 1.0, "LINEAR")


def retire(root, t):
    """Hide everything under a glyph's root from song time t on (a spent trace is black, not gone: against a bright
    sky it would stay as a dark outline)."""
    for ob in root.children_recursive:
        ob.hide_render = False
        ob.keyframe_insert("hide_render", frame=frame(t) - 1)
        ob.hide_render = True
        ob.keyframe_insert("hide_render", frame=frame(t))
        for fc in fcurves(ob):
            if fc.data_path == "hide_render":
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


def write_glyph(parts, t0, t1, glow_line=8.0, glow_fill=2.5, hold=None, t_out=None, out_dur=0.6):
    """The pen traces the outline over [t0, t1], the letter ignites behind it, holds, and (at t_out) burns away."""
    root, fill, trace, mod = parts
    write_skeleton(root, t0, t1)
    if t_out is not None:
        retire(root, t_out + out_dur + 0.1)
    gn_key(trace, mod, "End", t0 - 0.01, 0.0)
    gn_key(trace, mod, "End", t1, 1.0)
    key(trace, '["glow"]', t0 - 0.05, 0.0)
    key(trace, '["glow"]', t0, glow_line)
    key(trace, '["glow"]', t1 + 0.4, glow_line * 0.35)
    key(fill, '["alpha"]', t0 + 0.4 * (t1 - t0), 0.0)
    key(fill, '["alpha"]', t1 + 0.1, 1.0)
    key(fill, '["glow"]', t1, glow_fill * 2.0)
    key(fill, '["glow"]', t1 + 0.5, glow_fill)
    if t_out is not None:
        key(fill, '["burn"]', t_out, 0.0)
        key(fill, '["burn"]', t_out + out_dur, 1.0)
        key(trace, '["glow"]', t_out, glow_line * 0.35)
        key(trace, '["glow"]', t_out + out_dur * 0.6, 0.0)


# ---------------------------------------------------------------- her
TRIPO = ROOT / "assets" / "tripo"
HEIGHTS = {"stand": 1.65, "lookback": 1.65, "umbrella": 1.80, "kneel": 0.92}


def tripo_mesh(pose):
    """Her figure from Tripo (assets/tripo/<pose>.glb): imported once, joined into one mesh, standing on z = 0,
    facing -Y, scaled to her height, with the GLB's base-colour texture feeding her control material."""
    name = f"her_{pose}"
    me = bpy.data.meshes.get(name)
    if me is not None and me.get("source") == "tripo":
        return me
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(TRIPO / f"{pose}.glb"))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    # the texture: the image feeding the first material's base colour
    image = None
    for o in meshes:
        for slot in o.material_slots:
            if slot.material and slot.material.node_tree:
                for n in slot.material.node_tree.nodes:
                    if n.type == "TEX_IMAGE" and n.image is not None:
                        bsdf = next((l.to_node for l in n.outputs["Color"].links), None)
                        if image is None or (bsdf is not None and bsdf.type == "BSDF_PRINCIPLED"):
                            image = n.image
    dg = bpy.context.evaluated_depsgraph_get()
    verts, faces, uvs, off = [], [], [], 0
    for o in meshes:
        ev = o.evaluated_get(dg)
        m = ev.to_mesh()
        m.calc_loop_triangles()
        mw = o.matrix_world
        co = np.empty(len(m.vertices) * 3, np.float32)
        m.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3) @ np.array(mw.to_3x3()).T + np.array(mw.translation)
        tri = np.empty(len(m.loop_triangles) * 3, np.int32)
        m.loop_triangles.foreach_get("vertices", tri)
        tl = np.empty(len(m.loop_triangles) * 3, np.int32)
        m.loop_triangles.foreach_get("loops", tl)
        uv = np.zeros((len(m.loops), 2), np.float32)
        if m.uv_layers.active is not None:
            m.uv_layers.active.data.foreach_get("uv", uv.ravel())
        verts.append(co)
        faces.append(tri.reshape(-1, 3) + off)
        uvs.append(uv[tl].reshape(-1, 3, 2))
        off += len(co)
        ev.to_mesh_clear()
    V, F, UV = np.concatenate(verts), np.concatenate(faces), np.concatenate(uvs)
    for o in new:
        bpy.data.objects.remove(o, do_unlink=True)
    # stand her up: glTF import is already Z-up and faces -Y; scale to her height, feet on the floor, centred
    s = HEIGHTS[pose] / float(np.ptp(V[:, 2]))
    V = V * s
    V[:, 2] -= V[:, 2].min()
    foot = V[V[:, 2] < 0.15 * HEIGHTS[pose]]
    V[:, 0] -= np.median(foot[:, 0])
    V[:, 1] -= np.median(foot[:, 1])
    if me is not None:
        bpy.data.meshes.remove(me)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3, dtype=np.int32))
    me.update(calc_edges=True)
    layer = me.uv_layers.new(name="UVMap")
    layer.data.foreach_set("uv", UV.ravel())
    me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
    me["source"] = "tripo"
    if image is not None:
        image.pack()
        image.name = f"tripo_{pose}_albedo"
        me["albedo"] = image.name
    me.materials.append(her_material(pose, image=image))
    return me


def her_points(pose, n=260_000, seed=9):
    """dict(xyz, rgb) of points on her surface in this pose (her local frame), coloured like her (scene-linear):
    sampled from the Tripo mesh and its texture when there is one, else the points mv.py made."""
    if not (TRIPO / f"{pose}.glb").exists():
        z = np.load(MV / pose / "her_points.npz")
        return dict(xyz=z["xyz"], rgb=np.clip(z["rgb"], 0, 1) ** 2.2)
    me = tripo_mesh(pose)
    V = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    F = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", F)
    F = F.reshape(-1, 3)
    UV = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers[0].data.foreach_get("uv", UV)
    UV = UV.reshape(-1, 3, 2)
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(F), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    b = np.stack([1 - s, s * (1 - r2), s * r2], 1).astype(np.float32)
    xyz = np.einsum("nk,nkd->nd", b, tri[fi])
    uv = np.einsum("nk,nkd->nd", b, UV[fi])
    img = bpy.data.images.get(me.get("albedo", ""))
    if img is None:
        return dict(xyz=xyz, rgb=np.full((n, 3), 0.6, np.float32))
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    iu = np.clip((uv[:, 0] % 1.0) * w, 0, w - 1).astype(int)
    iv = np.clip((uv[:, 1] % 1.0) * h, 0, h - 1).astype(int)
    rgb = px[iv, iu, :3]
    return dict(xyz=xyz.astype(np.float32), rgb=(np.clip(rgb, 0, 1) ** 2.2).astype(np.float32))


def her_mesh(pose):
    """Her figure in this pose as a mesh object (cached as data; each call returns a new object using it).
    A Tripo model (assets/tripo/<pose>.glb) is used when there is one; else the Hunyuan3D-2mv figure."""
    if (TRIPO / f"{pose}.glb").exists():
        me = tripo_mesh(pose)
        ob = bpy.data.objects.new(f"her_{pose}", me)
        for k, v in dict(look=0.0, rim=1.0, glow=0.0, cut=-1.0, cut_soft=0.04, cut_glow=0.0, alpha=1.0,
                         heart=10.0, flat=0.0, flat_gain=1.0, ink=0.1, lines=0.0, line_reveal=-1.0).items():
            ob[k] = v
        return ob
    me = bpy.data.meshes.get(f"her_{pose}")
    if me is None:
        z = np.load(MV / pose / "her.npz")
        V, F = z["verts"], z["faces"]
        me = bpy.data.meshes.new(f"her_{pose}")
        me.vertices.add(len(V))
        me.vertices.foreach_set("co", V.ravel())
        me.loops.add(F.size)
        me.loops.foreach_set("vertex_index", F.ravel())
        me.polygons.add(len(F))
        me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3, dtype=np.int32))
        me.update(calc_edges=True)
        me.validate(clean_customdata=False)
        w = z["weights"]
        a = me.color_attributes.new("vw", "FLOAT_COLOR", "POINT")
        a.data.foreach_set("color", w.ravel())
        fill = np.concatenate([z["fill"], np.ones((len(V), 1), np.float32)], 1)
        b = me.color_attributes.new("fill", "FLOAT_COLOR", "POINT")
        b.data.foreach_set("color", fill.ravel())
        me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
    if not me.materials or me.materials[0] is None:
        me.materials.clear()
        me.materials.append(bpy.data.materials.get(f"her_{pose}") or her_material(pose))
    ob = bpy.data.objects.new(f"her_{pose}", me)
    for k, v in dict(look=0.0, rim=1.0, glow=0.0, cut=-1.0, cut_soft=0.04, cut_glow=0.0, alpha=1.0,
                     heart=10.0, flat=0.0, flat_gain=1.0, ink=0.1, lines=0.0, line_reveal=-1.0).items():
        ob[k] = v
    return ob


def her_material(pose, image=None):
    """Colour = a Tripo model's own texture (image, through its UVs), or else the four sheet views projected back
    onto her (weights per vertex from mv.py: only views that really see the point). Controls are object
    properties, all keyable:
      look   0 = backlit ink silhouette with a hint of her colours, 1 = full colour
      rim    rim light (her hair orange, along the edges facing away)
      glow   self light
      cut    materialise: above z = cut she exists, below she is not yet (a glowing seam at the cut)
      heart  radius of light filling her from the chest out (S9); 10 = all
      alpha  overall presence"""
    m, g = material(f"her_{pose}")
    tc = g.node("ShaderNodeTexCoord")
    p = sock(tc, "Object", True)
    if image is not None:
        tex = g.node("ShaderNodeTexImage", interpolation="Cubic")
        tex.image = image
        g.link(sock(tc, "UV", True), sock(tex, "Vector"))
        albedo = sock(tex, "Color", True)
        return _her_shading(m, g, p, albedo)
    meta = json.loads((MV / pose / "her.json").read_text(encoding="utf-8"))
    vw = g.node("ShaderNodeVertexColor", layer_name="vw")
    sep = g.node("ShaderNodeSeparateColor")
    g.link(sock(vw, "Color", True), sock(sep, "Color"))
    weights = [sep.outputs[0], sep.outputs[1], sep.outputs[2], sock(vw, "Alpha", True)]
    acc = None
    cover = None
    for k, view in enumerate(meta["views"]):
        pr = meta["proj"][view]
        u = g.math("DIVIDE", g.math("ADD", g.vmath("DOT_PRODUCT", p, tuple(pr["A"])), pr["bx"]), pr["w"])
        v = g.math("SUBTRACT", 1.0, g.math("DIVIDE", g.math("ADD", g.vmath("DOT_PRODUCT", p, tuple(pr["B"])),
                                                              pr["by"]), pr["h"]))
        uv = g.node("ShaderNodeCombineXYZ")
        g.link(u, uv.inputs[0])
        g.link(v, uv.inputs[1])
        img = bpy.data.images.load(meta["images"][view], check_existing=True)
        img.colorspace_settings.name = "sRGB"
        tex = g.node("ShaderNodeTexImage", interpolation="Cubic", extension="CLIP")
        tex.image = img
        g.link(uv.outputs[0], sock(tex, "Vector"))
        part = g.vmath("SCALE", sock(tex, "Color", True), scale=weights[k])
        acc = part if acc is None else g.vmath("ADD", acc, part)
        cover = weights[k] if cover is None else g.math("ADD", cover, weights[k])
    fill = g.node("ShaderNodeVertexColor", layer_name="fill")
    albedo = g.vmath("ADD", acc, g.vmath("SCALE", sock(fill, "Color", True), scale=g.math("SUBTRACT", 1.0, cover, clamp=True)))
    return _her_shading(m, g, p, albedo)


def _her_shading(m, g, p, albedo):
    """Everything after the colour: silhouette / full colour, rim, materialising seam, heart fill, flat, lines."""
    look = g.attr("look")
    ink = g.vmath("SCALE", albedo, scale=g.attr("ink"))
    base = g.mix(ink, albedo, look)
    bsdf = g.node("ShaderNodeBsdfPrincipled")
    g.link(base, sock(bsdf, "Base Color"))
    sock(bsdf, "Roughness").default_value = 0.6
    # rim: edges turned away from the camera glow in her hair's orange, warmed by her own colour
    lw = g.node("ShaderNodeLayerWeight")
    sock(lw, "Blend").default_value = 0.35
    fres = g.math("POWER", sock(lw, "Facing", True), 2.2)
    rim_col = g.mix(ORANGE + (1.0,), albedo, 0.35)
    rim = g.math("MULTIPLY", fres, g.attr("rim"))
    rim = g.math("MULTIPLY", rim, g.math("SUBTRACT", 1.0, g.math("MULTIPLY", look, 0.7)))
    # the materialising seam and the light filling her from the heart
    z = sock(g.node("ShaderNodeSeparateXYZ"), "Z", True)
    sepz = z.node
    g.link(p, sepz.inputs[0])
    cut = g.attr("cut")
    soft = g.attr("cut_soft")
    above = g.smoothstep(g.math("SUBTRACT", z, cut), g.math("MULTIPLY", soft, -1.0), soft)
    dist = g.math("ABSOLUTE", g.math("DIVIDE", g.math("SUBTRACT", z, cut), g.math("MULTIPLY", soft, 3.0)))
    seam = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, dist, clamp=True), g.attr("cut_glow"))
    chest = g.vmath("DISTANCE", p, (0.0, 0.0, 1.22))
    heart = g.attr("heart")
    filled = g.smoothstep(g.math("SUBTRACT", heart, chest), -0.05, 0.05)
    em_strength = g.math("ADD", g.math("ADD", g.math("MULTIPLY", rim, 3.0), g.attr("glow")), g.math("MULTIPLY", seam, 8.0))
    g.link(rim_col, sock(bsdf, "Emission Color"))
    g.link(em_strength, sock(bsdf, "Emission Strength"))
    alpha = g.math("MULTIPLY", g.math("MULTIPLY", above, filled), g.attr("alpha"))
    g.link(alpha, sock(bsdf, "Alpha"))
    # flat: her colours as the illustration has them (anime look, no lighting), mixed over the lit surface
    flat_em = g.node("ShaderNodeEmission")
    g.link(albedo, sock(flat_em, "Color"))
    g.link(g.attr("flat_gain"), sock(flat_em, "Strength"))
    tr = g.node("ShaderNodeBsdfTransparent")
    flat_mix = g.node("ShaderNodeMixShader")
    g.link(g.attr("flat"), flat_mix.inputs[0])
    g.link(sock(bsdf, "BSDF", True), flat_mix.inputs[1])
    flat_a = g.node("ShaderNodeMixShader")                   # the flat colour keeps the same cut / fill / alpha
    g.link(alpha, flat_a.inputs[0])
    g.link(tr.outputs[0], flat_a.inputs[1])
    g.link(flat_em.outputs[0], flat_a.inputs[2])
    g.link(flat_a.outputs[0], flat_mix.inputs[2])
    # lines: her as a blueprint drawing - the outline (where the surface turns away) and contour lines every
    # 4.5 cm of height - drawn by the pen from the head down to line_reveal, with a bright seam where it draws
    facing = sock(lw, "Facing", True)
    outline = g.smoothstep(facing, 0.62, 0.82)
    stripe = g.smoothstep(g.math("ABSOLUTE", g.math("SUBTRACT", g.math("FRACT", g.math("MULTIPLY", z, 22.0)), 0.5)),
                          0.44, 0.48)
    drawn = g.smoothstep(g.math("SUBTRACT", z, g.attr("line_reveal")), -0.008, 0.008)
    line_v = g.math("MULTIPLY", g.math("MAXIMUM", outline, g.math("MULTIPLY", stripe, 0.55)), drawn)
    pen_d = g.math("DIVIDE", g.math("SUBTRACT", z, g.attr("line_reveal")), 0.02)
    pen = g.math("MULTIPLY", g.math("EXPONENT", g.math("MULTIPLY", g.math("MULTIPLY", pen_d, pen_d), -1.0)), outline)
    line_em = g.node("ShaderNodeEmission")
    sock(line_em, "Color").default_value = (0.45, 0.85, 1.0, 1)
    g.link(g.math("ADD", g.math("MULTIPLY", line_v, 5.0), g.math("MULTIPLY", pen, 30.0)), sock(line_em, "Strength"))
    line_tr = g.node("ShaderNodeBsdfTransparent")
    line_mix = g.node("ShaderNodeMixShader")
    g.link(g.math("MINIMUM", g.math("ADD", line_v, pen), 1.0), line_mix.inputs[0])
    g.link(line_tr.outputs[0], line_mix.inputs[1])
    g.link(line_em.outputs[0], line_mix.inputs[2])
    # in S9 the light filling her (heart) replaces the drawing where it has arrived
    lines = g.math("MULTIPLY", g.attr("lines"), g.math("SUBTRACT", 1.0, filled))
    final = g.node("ShaderNodeMixShader")
    g.link(lines, final.inputs[0])
    g.link(flat_mix.outputs[0], final.inputs[1])
    g.link(line_mix.outputs[0], final.inputs[2])
    out = g.node("ShaderNodeOutputMaterial")
    g.link(final.outputs[0], sock(out, "Surface"))
    return m


# ---------------------------------------------------------------- animation
def key(owner, path, t, value, interp="BEZIER"):
    """Keyframe owner.path (or owner["prop"]) at song time t."""
    if path.startswith("["):
        owner[path[2:-2]] = value
    else:
        setattr(owner, path, value)
    owner.keyframe_insert(data_path=path, frame=frame(t))
    fc = _fcurve(owner, path)
    if fc is not None:
        for kp in fc.keyframe_points:
            if abs(kp.co[0] - frame(t)) < 1e-3:
                kp.interpolation = interp


def _fcurve(owner, path):
    ad = getattr(owner, "animation_data", None)
    if ad is None or ad.action is None:
        return None
    try:
        for fc in ad.action.fcurves:
            if fc.data_path == path:
                return fc
    except AttributeError:                                  # Blender 5: layered actions
        for layer in ad.action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for fc in bag.fcurves:
                        if fc.data_path == path:
                            return fc
    return None


def fcurves(owner):
    """All F-curves animating owner (Blender 5 layered actions or the older flat list)."""
    ad = getattr(owner, "animation_data", None)
    if ad is None or ad.action is None:
        return []
    act = ad.action
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                out += list(bag.fcurves)
    return out


def linear(owner, contains=""):
    """Make the keys of owner's F-curves (whose data path contains `contains`) linear."""
    for fc in fcurves(owner):
        if contains in fc.data_path:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


def camera(sc, name="cam", fov_y=38.0):
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = 24.0
    cd.angle = math.radians(fov_y)
    cd.clip_start, cd.clip_end = 0.05, 400
    ob = bpy.data.objects.new(name, cd)
    sc.collection.objects.link(ob)
    sc.camera = ob
    return ob


def bake_camera(cam, t0, t1, pose, step=1):
    """Keyframe the camera on every frame from pose(t) -> (position, target, focus distance, f-stop or None)."""
    f0, f1 = int(math.floor(frame(t0))), int(math.ceil(frame(t1)))
    for f in range(f0, f1 + 1, step):
        t = f / FPS
        got = pose(t)
        pos, tgt, focus, fstop = got[:4]
        lens = got[4] if len(got) > 4 else None
        pos, tgt = Vector(pos), Vector(tgt)
        cam.location = pos
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = (tgt - pos).to_track_quat("-Z", "Y")
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        d = cam.data
        if lens:
            d.lens = lens
            d.keyframe_insert("lens", frame=f)
        if fstop:
            d.dof.use_dof = True
            d.dof.focus_distance = focus
            d.dof.aperture_fstop = fstop
            d.dof.keyframe_insert("focus_distance", frame=f)
            d.dof.keyframe_insert("aperture_fstop", frame=f)
    return cam
