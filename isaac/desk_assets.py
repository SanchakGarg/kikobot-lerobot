"""Desk sorting station assets: Nucleus / SimReady props where they exist,
procedural ones where they don't, and the USD that puts them on the desk.

Find the props your Isaac Sim asset server has (run once per machine, with
the Isaac Sim Python):

    python isaac/desk_assets.py --discover              # -> models/desk_assets.json
    python isaac/desk_assets.py --discover --folder https://.../simready_content

The server then uses those for the item kinds they cover and builds the rest
(and always the pen cup and tray) from simple meshes. Either way every item
gets a primitive collider fitted to its size (Newton needs convex shapes;
asset colliders are authored for PhysX) and its own material, which the
server re-colours and re-textures each episode.

Nothing here imports Isaac Sim or pxr at module level.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MODELS = Path(os.environ.get("KIKOBOT_MODELS_DIR", REPO / "models"))
DEFAULT_ASSETS_JSON = MODELS / "desk_assets.json"
DEFAULT_TEXTURES = MODELS / "textures" / "desk"

# Words that name each kind in an asset's file name (case and CamelCase split).
KEYWORDS: dict[str, tuple[str, ...]] = {
    "pen": ("pen", "ballpoint", "ballpen", "biro"),
    "pencil": ("pencil",),
    "marker": ("marker", "sharpie"),
    "highlighter": ("highlighter",),
    "glue_stick": ("glue", "gluestick"),
    "tape_roll": ("tape",),
    "scissors": ("scissors", "scissor"),
    "eraser": ("eraser",),
}
# Where to look, relative to the Isaac assets root.
DEFAULT_FOLDERS = ("/Isaac/Props", "/Isaac/Environments/Office", "/Isaac/SimReady", "/NVIDIA/Assets/simready_content")
USD_EXT = (".usd", ".usda", ".usdc", ".usdz")


# ============================================================== discovery


def tokens(name: str) -> set[str]:
    stem = re.sub(r"\.(usd[acz]?)$", "", name, flags=re.I)
    stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    return {t for t in re.split(r"[^A-Za-z0-9]+", stem.lower()) if t}


def match_kind(name: str) -> str | None:
    t = tokens(name)
    for kind, words in KEYWORDS.items():
        if t & set(words):
            return kind
    return None


def rank(url: str) -> tuple[int, int, str]:
    """Prefer plain visual assets over ones with physics baked in, then short paths."""
    low = url.lower()
    return ("physics" in low or "collision" in low, len(url), url)


def discover(roots: list[str], max_depth: int = 7, max_entries: int = 50_000) -> dict[str, list[str]]:
    import omni.client

    found: dict[str, list[str]] = {k: [] for k in KEYWORDS}
    queue = [(r.rstrip("/"), 0) for r in roots]
    seen = 0
    while queue and seen < max_entries:
        url, depth = queue.pop(0)
        result, entries = omni.client.list(url)
        if result != omni.client.Result.OK:
            continue
        for e in entries:
            seen += 1
            child = f"{url}/{e.relative_path}"
            if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN and not e.relative_path.lower().endswith(USD_EXT):
                if depth + 1 < max_depth and not e.relative_path.startswith("."):
                    queue.append((child, depth + 1))
            elif e.relative_path.lower().endswith(USD_EXT):
                kind = match_kind(e.relative_path)
                if kind:
                    found[kind].append(child)
    return {k: sorted(v, key=rank) for k, v in found.items()}


def load_catalog(path: Path) -> dict[str, list[str]]:
    if not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text())
    return {k: list(v) for k, v in data.get("kinds", {}).items() if v}


# ========================================================= mesh generators


class MeshBuilder:
    """Unshared-vertex mesh with per-vertex normals and UVs. `face` fixes the
    winding itself from the outward direction it is given."""

    def __init__(self):
        self.points: list[np.ndarray] = []
        self.normals: list[np.ndarray] = []
        self.st: list[np.ndarray] = []
        self.counts: list[int] = []

    def face(self, pts, nrm, st, outward) -> None:
        pts, st = np.asarray(pts, float), np.asarray(st, float)
        nrm = np.asarray(nrm, float)
        if nrm.ndim == 1:
            nrm = np.repeat(nrm[None], len(pts), 0)
        # Newell's method: polygon normal from its winding.
        n = np.zeros(3)
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
        if n @ np.asarray(outward, float) < 0:
            pts, nrm, st = pts[::-1], nrm[::-1], st[::-1]
        self.points.extend(pts)
        self.normals.extend(nrm)
        self.st.extend(st)
        self.counts.append(len(pts))

    def extend(self, other: "MeshBuilder", transform=None) -> "MeshBuilder":
        R, t = transform if transform is not None else (np.eye(3), np.zeros(3))
        self.points.extend(np.asarray(other.points) @ R.T + t)
        self.normals.extend(np.asarray(other.normals) @ R.T)
        self.st.extend(other.st)
        self.counts.extend(other.counts)
        return self

    def arrays(self):
        return (np.asarray(self.points), np.asarray(self.normals), np.asarray(self.st), list(self.counts))


def _norm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def revolve_x(r0: float, r1: float, x0: float, x1: float, n: int = 24, cap0: bool = True, cap1: bool = True,
              flat: bool = False, offset: float = 0.0, inward: bool = False, v_range=(0.0, 1.0)) -> MeshBuilder:
    """Frustum along X from (x0, r0) to (x1, r1). u around, v along."""
    m = MeshBuilder()
    a = offset + 2 * math.pi * np.arange(n + 1) / n
    dx = x1 - x0
    slope = -(r1 - r0) / dx if dx else 0.0
    sgn = -1.0 if inward else 1.0
    for k in range(n):
        ak, ak1 = a[k], a[k + 1]
        p = [(x0, r0 * math.cos(ak), r0 * math.sin(ak)), (x0, r0 * math.cos(ak1), r0 * math.sin(ak1)),
             (x1, r1 * math.cos(ak1), r1 * math.sin(ak1)), (x1, r1 * math.cos(ak), r1 * math.sin(ak))]
        if flat:
            am = 0.5 * (ak + ak1)
            nr = [sgn * _norm((slope, math.cos(am), math.sin(am)))] * 4
        else:
            nr = [sgn * _norm((slope, math.cos(t), math.sin(t))) for t in (ak, ak1, ak1, ak)]
        u0, u1 = k / n, (k + 1) / n
        st = [(u0, v_range[0]), (u1, v_range[0]), (u1, v_range[1]), (u0, v_range[1])]
        am = 0.5 * (ak + ak1)
        m.face(p, nr, st, sgn * np.array([0.0, math.cos(am), math.sin(am)]))
    for cap, x, r, d in ((cap0, x0, r0, -1.0), (cap1, x1, r1, 1.0)):
        if not cap or r <= 0:
            continue
        ring = [(x, r * math.cos(t), r * math.sin(t)) for t in a[:-1]]
        st = [(0.5 + 0.5 * math.cos(t), 0.5 + 0.5 * math.sin(t)) for t in a[:-1]]
        m.face(ring, (d, 0, 0), st, (d, 0, 0))
    return m


def tube_x(r_in: float, r_out: float, x0: float, x1: float, n: int = 32) -> MeshBuilder:
    """Hollow cylinder along X (tape, cup, scissor loops)."""
    m = revolve_x(r_out, r_out, x0, x1, n, cap0=False, cap1=False)
    m.extend(revolve_x(r_in, r_in, x0, x1, n, cap0=False, cap1=False, inward=True))
    a = 2 * math.pi * np.arange(n + 1) / n
    for x, d in ((x0, -1.0), (x1, 1.0)):
        for k in range(n):
            t0, t1 = a[k], a[k + 1]
            p = [(x, r_in * math.cos(t0), r_in * math.sin(t0)), (x, r_out * math.cos(t0), r_out * math.sin(t0)),
                 (x, r_out * math.cos(t1), r_out * math.sin(t1)), (x, r_in * math.cos(t1), r_in * math.sin(t1))]
            st = [(k / n, 0), (k / n, 1), ((k + 1) / n, 1), ((k + 1) / n, 0)]
            m.face(p, (d, 0, 0), st, (d, 0, 0))
    return m


def box_mesh(half, center=(0, 0, 0)) -> MeshBuilder:
    m = MeshBuilder()
    h, c = np.asarray(half, float), np.asarray(center, float)
    for ax in range(3):
        t1, t2 = (ax + 1) % 3, (ax + 2) % 3
        for s in (-1.0, 1.0):
            pts = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[ax], p[t1], p[t2] = s * h[ax], a * h[t1], b * h[t2]
                pts.append(c + p)
            nrm = np.zeros(3)
            nrm[ax] = s
            m.face(pts, nrm, [(0, 0), (1, 0), (1, 1), (0, 1)], nrm)
    return m


# X axis -> Z axis (cyclic permutation, a proper rotation).
X_TO_Z = (np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float), np.zeros(3))


def _xz(m: MeshBuilder, shift=(0, 0, 0)) -> MeshBuilder:
    R, _ = X_TO_Z
    return MeshBuilder().extend(m, (R, np.asarray(shift, float)))


# ============================================================ procedural items

# Each item is a list of (part, MeshBuilder), part = body | accent | metal | wood | graphite | card.
# Item frame as in desk_layout: long axis X, resting face down, centred.


def procedural_item(kind: str) -> list[tuple[str, MeshBuilder]]:
    if kind == "pen":
        r = 0.005
        return [
            ("metal", revolve_x(0.0008, 0.0035, -0.070, -0.057, 20, cap0=True)),
            ("body", revolve_x(r, r, -0.057, 0.040, 24)),
            ("accent", revolve_x(0.0056, 0.0056, 0.024, 0.068, 24)),
            ("accent", revolve_x(0.0056, 0.0045, 0.068, 0.070, 24, cap0=False)),
            ("accent", box_mesh((0.018, 0.0011, 0.0007), (0.050, 0.0, 0.0062))),
        ]
    if kind == "pencil":
        rc = 0.0075 / 2 / math.sin(math.radians(60))  # corner radius for 7.5 mm across the flats
        return [
            ("graphite", revolve_x(0.0, 0.0012, -0.0875, -0.0855, 6, flat=True)),
            ("wood", revolve_x(0.0012, rc, -0.0855, -0.070, 6, cap0=False, cap1=False, flat=True)),
            ("body", revolve_x(rc, rc, -0.070, 0.0735, 6, flat=True)),
            ("metal", revolve_x(0.0040, 0.0040, 0.0735, 0.0815, 20)),
            ("accent", revolve_x(0.0037, 0.0037, 0.0815, 0.0875, 20)),
        ]
    if kind == "marker":
        return [
            ("accent", revolve_x(0.0014, 0.0040, -0.065, -0.058, 20)),
            ("body", revolve_x(0.0085, 0.0085, -0.058, 0.030, 28)),
            ("accent", revolve_x(0.0090, 0.0090, 0.018, 0.063, 28)),
            ("accent", revolve_x(0.0090, 0.0060, 0.063, 0.065, 28, cap0=False)),
        ]
    if kind == "highlighter":
        return [
            ("accent", box_mesh((0.003, 0.004, 0.002), (-0.0605, 0.0, -0.002))),
            ("body", box_mesh((0.046, 0.011, 0.007), (-0.0115, 0.0, 0.0))),
            ("accent", box_mesh((0.0175, 0.0114, 0.0074), (0.0425, 0.0, 0.0))),
        ]
    if kind == "glue_stick":
        return [
            ("body", revolve_x(0.011, 0.011, -0.0425, 0.018, 32)),
            ("accent", revolve_x(0.0114, 0.0114, 0.012, 0.0425, 32)),
        ]
    if kind == "tape_roll":
        return [
            ("body", _xz(tube_x(0.0125, 0.022, -0.0065, 0.0065, 40))),
            ("card", _xz(tube_x(0.0105, 0.0125, -0.0066, 0.0066, 40))),
        ]
    if kind == "scissors":
        loops = MeshBuilder()
        for y in (-0.011, 0.011):
            loops.extend(_xz(tube_x(0.0065, 0.0105, -0.004, 0.004, 28), (-0.052, y, 0.0)))
        grips = box_mesh((0.012, 0.014, 0.0035), (-0.030, 0.0, 0.0))
        return [
            ("metal", box_mesh((0.046, 0.0065, 0.0012), (0.024, 0.0025, -0.0028))),
            ("metal", box_mesh((0.046, 0.0065, 0.0012), (0.024, -0.0025, -0.0016))),
            ("metal", _xz(revolve_x(0.0025, 0.0025, -0.0005, 0.0012, 16), (-0.012, 0.0, 0.0))),
            ("body", loops),
            ("body", grips),
        ]
    if kind == "eraser":
        return [
            ("body", box_mesh((0.022, 0.011, 0.006))),
            ("accent", box_mesh((0.010, 0.0113, 0.0062), (0.008, 0.0, 0.0))),
        ]
    raise KeyError(kind)


# ======================================================================= USD


FIXED_LOOKS = {
    # name: (rgb, roughness, metallic)
    "metal": ((0.62, 0.63, 0.66), 0.25, 1.0),
    "wood": ((0.80, 0.62, 0.42), 0.8, 0.0),
    "graphite": ((0.08, 0.08, 0.09), 0.45, 0.3),
    "card": ((0.55, 0.42, 0.28), 0.9, 0.0),
    "cup": ((0.18, 0.19, 0.21), 0.35, 0.8),
    "tray": ((0.88, 0.88, 0.86), 0.55, 0.0),
    "tray_dark": ((0.30, 0.32, 0.35), 0.55, 0.0),
}
PHYSICS = {"plastic": 0.8, "rubber": 1.2, "metal": 0.5, "desk": 1.5}


@dataclass
class DeskScene:
    """What the backend keeps after building: paths and the shaders to recolour."""

    pool: list  # desk_layout.PoolItem, with real sizes
    paths: dict[str, str] = field(default_factory=dict)  # item name -> rigid body prim
    looks: dict[str, dict] = field(default_factory=dict)  # item name -> {surface, tex, accent}
    textures: dict[str, Path] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)  # item name -> url | "procedural"


class _Usd:
    """Small helpers over pxr, bound to one stage."""

    def __init__(self, stage):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        self.stage, self.Gf, self.Sdf, self.Usd = stage, Gf, Sdf, Usd
        self.UsdGeom, self.UsdPhysics, self.UsdShade = UsdGeom, UsdPhysics, UsdShade
        self._fixed: dict[str, object] = {}
        self._phys: dict[str, object] = {}

    # Materials ---------------------------------------------------------

    def surface(self, path: str, rgb, rough: float, metal: float):
        S, Sdf, Gf = self.UsdShade, self.Sdf, self.Gf
        mat = S.Material.Define(self.stage, path)
        sh = S.Shader.Define(self.stage, f"{path}/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat, sh

    def textured(self, path: str, tex_file: Path):
        """UsdPreviewSurface whose colour is a grayscale texture tinted by
        scale/bias: rgb = pattern * (body - print) + print."""
        S, Sdf, Gf = self.UsdShade, self.Sdf, self.Gf
        mat, sh = self.surface(path, (1, 1, 1), 0.5, 0.0)
        reader = S.Shader.Define(self.stage, f"{path}/StReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        tex = S.Shader.Define(self.stage, f"{path}/Pattern")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(tex_file)))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, 1, 1, 1))
        tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0, 0, 0, 0))
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
        return mat, sh, tex

    def fixed(self, name: str):
        if name not in self._fixed:
            rgb, rough, metal = FIXED_LOOKS[name]
            self._fixed[name] = self.surface(f"/World/Looks/Desk/Fixed_{name}", rgb, rough, metal)[0]
        return self._fixed[name]

    def physics_material(self, name: str):
        if name not in self._phys:
            mat = self.UsdShade.Material.Define(self.stage, f"/World/Looks/Desk/Physics_{name}")
            pm = self.UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            mu = PHYSICS.get(name, 0.8)
            pm.CreateStaticFrictionAttr(mu)
            pm.CreateDynamicFrictionAttr(mu)
            pm.CreateRestitutionAttr(0.0)
            self._phys[name] = mat
        return self._phys[name]

    def bind(self, prim, mat, stronger: bool = False) -> None:
        S = self.UsdShade
        api = S.MaterialBindingAPI.Apply(prim)
        strength = S.Tokens.strongerThanDescendants if stronger else S.Tokens.weakerThanDescendants
        api.Bind(mat, bindingStrength=strength, materialPurpose=S.Tokens.allPurpose)

    # Geometry ------------------------------------------------------------

    def mesh(self, path: str, m: MeshBuilder):
        G, Sdf, Gf = self.UsdGeom, self.Sdf, self.Gf
        pts, nrm, st, counts = m.arrays()
        g = G.Mesh.Define(self.stage, path)
        g.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
        g.CreateFaceVertexCountsAttr(counts)
        g.CreateFaceVertexIndicesAttr(list(range(len(pts))))
        g.CreateNormalsAttr([Gf.Vec3f(*n) for n in nrm])
        g.SetNormalsInterpolation(G.Tokens.vertex)
        g.CreateSubdivisionSchemeAttr(G.Tokens.none)
        pv = G.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, G.Tokens.vertex)
        pv.Set([Gf.Vec2f(*s) for s in st])
        lo, hi = pts.min(0), pts.max(0)
        g.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
        return g

    def cube(self, path: str, center, half, collide: bool = True, visible: bool = True):
        G, Gf = self.UsdGeom, self.Gf
        g = G.Cube.Define(self.stage, path)
        g.CreateSizeAttr(1.0)
        x = G.Xformable(g)
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        x.AddScaleOp().Set(Gf.Vec3f(*(2.0 * float(h) for h in half)))
        if collide:
            self.UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        if not visible:
            g.CreatePurposeAttr(G.Tokens.guide)
        return g

    def collider(self, path: str, shape: str, half, center=(0, 0, 0), axis: str = "X"):
        """A primitive collider, hidden from the cameras (purpose guide, as
        Isaac's importers do for collision geometry)."""
        G, Gf = self.UsdGeom, self.Gf
        if shape == "box":
            return self.cube(path, center, half, visible=False)
        if shape == "capsule":
            g = G.Capsule.Define(self.stage, path)
            r = float(half[1])
            g.CreateRadiusAttr(r)
            g.CreateHeightAttr(max(2.0 * float(half[0]) - 2.0 * r, 1e-4))
        else:
            g = G.Cylinder.Define(self.stage, path)
            if axis == "Z":
                g.CreateRadiusAttr(float(half[0]))
                g.CreateHeightAttr(2.0 * float(half[2]))
            else:
                g.CreateRadiusAttr(float(half[1]))
                g.CreateHeightAttr(2.0 * float(half[0]))
        g.CreateAxisAttr(axis)
        G.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        g.CreatePurposeAttr(G.Tokens.guide)
        self.UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        return g


def _add_colliders(u: _Usd, path: str, kind: str, half, phys) -> None:
    hx, hy, hz = half
    if kind == "scissors":
        # Blades (thin, sitting low) + the handle block, scaled to the item.
        sx, sy, sz = hx / 0.070, hy / 0.022, hz / 0.004
        prims = [
            u.collider(f"{path}/collider_blades", "box", (0.045 * sx, 0.007 * sy, 0.002 * sz), (0.025 * sx, 0, -0.002 * sz)),
            u.collider(f"{path}/collider_handles", "box", (0.025 * sx, hy, hz), (-0.045 * sx, 0, 0)),
        ]
    else:
        from desk_layout import KINDS  # noqa: PLC0415 - same folder, loaded by the server

        shape = KINDS[kind].collider
        if shape == "cylinder_x":
            prims = [u.collider(f"{path}/collider", "cylinder", half, axis="X")]
        elif shape == "cylinder_z":
            prims = [u.collider(f"{path}/collider", "cylinder", half, axis="Z")]
        else:
            prims = [u.collider(f"{path}/collider", shape, half)]
    for g in prims:
        u.bind(g.GetPrim(), phys)


def _physics_name(kind: str) -> str:
    return {"eraser": "rubber", "glue_stick": "plastic", "scissors": "plastic"}.get(kind, "plastic")


def _fit_asset(u: _Usd, xform_path: str, url: str, target_long: float):
    """Reference `url` under xform_path, rotated so its longest axis is X and its
    shortest is Z, scaled to `target_long` and centred. Returns the scaled
    half extents (X, Y, Z), or None if it did not load."""
    G, Gf, Usd, P = u.UsdGeom, u.Gf, u.Usd, u.UsdPhysics
    holder = G.Xform.Define(u.stage, xform_path)
    src = u.stage.DefinePrim(f"{xform_path}/src", "Xform")
    src.GetReferences().AddReference(url)
    src.SetInstanceable(False)
    bound = G.BBoxCache(Usd.TimeCode.Default(), [G.Tokens.default_, G.Tokens.render]).ComputeUntransformedBound(src)
    rng = bound.ComputeAlignedRange()
    if rng.IsEmpty():
        u.stage.RemovePrim(xform_path)
        return None
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    ext, center = hi - lo, 0.5 * (lo + hi)
    order = list(np.argsort(-ext))  # long -> X, mid -> Y, short -> Z (lying on the desk)
    M = np.zeros((3, 3))
    for new, old in enumerate(order):
        M[new, old] = 1.0
    if np.linalg.det(M) < 0:
        M[1] *= -1.0
    s = target_long / float(ext[order[0]])
    A = s * M
    t = -A @ center
    mat = np.eye(4)
    mat[:3, :3] = A.T  # USD: row vectors, p' = p * M
    mat[3, :3] = t
    G.Xformable(holder).AddTransformOp().Set(Gf.Matrix4d(mat.tolist()))
    # The asset is visual only: switch off any physics it brings.
    for prim in Usd.PrimRange(src):
        if prim.HasAPI(P.RigidBodyAPI):
            P.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.HasAPI(P.CollisionAPI):
            P.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    scaled = s * ext[order]
    return tuple(float(v) / 2.0 for v in scaled)


def build_desk(stage, pool, textures_dir: Path = DEFAULT_TEXTURES, catalog: dict[str, list[str]] | None = None,
               recolor_assets: bool = True) -> DeskScene:
    """Pen cup, tray and every pool item, on a stage that already has the table.

    Items start parked under the table; the backend moves them per episode.
    """
    import desk_layout as L  # noqa: PLC0415
    import desk_textures as T  # noqa: PLC0415

    u = _Usd(stage)
    G, Gf, P = u.UsdGeom, u.Gf, u.UsdPhysics
    catalog = catalog or {}
    scene = DeskScene(pool=[], textures=T.ensure_textures(Path(textures_dir)))
    tex = scene.textures
    G.Scope.Define(stage, "/World/Looks/Desk")

    # Wooden desk top over the table's (now hidden) collider.
    top = stage.GetPrimAtPath("/World/Table/Top")
    if top:
        G.Imageable(top).CreatePurposeAttr(G.Tokens.guide)
        wood, wsh, wtex = u.textured("/World/Looks/Desk/Wood", tex["wood"])
        dark, light = np.array([0.30, 0.17, 0.08]), np.array([0.62, 0.42, 0.24])
        wtex.GetInput("scale").Set(Gf.Vec4f(*(light - dark), 1))
        wtex.GetInput("bias").Set(Gf.Vec4f(*dark, 0))
        wsh.GetInput("roughness").Set(0.55)
        g = u.mesh("/World/Table/TopVisual", box_mesh((0.5, 0.5, 0.01), (0, 0, -0.01)))
        u.bind(g.GetPrim(), wood)

    _build_pen_cup(u, L.PEN_CUP, L.CUP_HEIGHT)
    _build_tray(u, L.TRAY, L.TRAY_WALL_HEIGHT)

    G.Xform.Define(stage, "/World/Desk/Items")
    for i, item in enumerate(pool):
        path = f"/World/Desk/Items/{item.name}"
        root = G.Xform.Define(stage, path)
        x = G.Xformable(root)
        park = L.park_pose(i, item)
        x.AddTranslateOp(G.XformOp.PrecisionDouble).Set(Gf.Vec3d(*park))
        x.AddOrientOp(G.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))

        body_mat, body_sh, body_tex = u.textured(f"/World/Looks/Desk/{item.name}_body", tex["solid"])
        accent_mat, accent_sh = u.surface(f"/World/Looks/Desk/{item.name}_accent", (0.1, 0.1, 0.8), 0.4, 0.0)

        half = item.half
        urls = catalog.get(item.kind, [])
        source = "procedural"
        if urls:
            url = urls[sum(1 for p in pool[:i] if p.kind == item.kind) % len(urls)]
            fitted = _fit_asset(u, f"{path}/asset", url, 2 * item.half[0])
            if fitted is not None:
                half, source = fitted, url
                if recolor_assets:
                    u.bind(stage.GetPrimAtPath(f"{path}/asset"), body_mat, stronger=True)
        if source == "procedural":
            parts = procedural_item(item.kind)
            for j, (part, m) in enumerate(parts):
                g = u.mesh(f"{path}/visual_{j}_{part}", m)
                mat = {"body": body_mat, "accent": accent_mat}.get(part) or u.fixed(part)
                u.bind(g.GetPrim(), mat)

        _add_colliders(u, path, item.kind, half, u.physics_material(_physics_name(item.kind)))
        P.RigidBodyAPI.Apply(root.GetPrim())
        P.MassAPI.Apply(root.GetPrim()).CreateMassAttr(item.spec.mass)

        scene.pool.append(L.PoolItem(item.name, item.kind, tuple(half)))
        scene.paths[item.name] = path
        scene.looks[item.name] = {"surface": body_sh, "tex": body_tex, "accent": accent_sh}
        scene.sources[item.name] = source
    return scene


def _build_pen_cup(u: _Usd, st, height: float, n_walls: int = 16) -> None:
    """Static hollow cup: a tube mesh for the eye, box walls for physics."""
    G, Gf = u.UsdGeom, u.Gf
    base = "/World/Desk/PenCup"
    root = G.Xform.Define(u.stage, base)
    G.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(st.xy[0], st.xy[1], 0.0))
    r_out = st.half[0]
    r_in = r_out - 0.004
    vis = _xz(tube_x(r_in, r_out, 0.0, height, 48))
    vis.extend(_xz(revolve_x(r_in, r_in, 0.0, 0.004, 48)))  # the floor of the cup
    g = u.mesh(f"{base}/visual", vis)
    u.bind(g.GetPrim(), u.fixed("cup"))
    phys = u.physics_material("desk")
    r_mid, thick = 0.5 * (r_in + r_out), 0.003
    seg = 2 * r_mid * math.tan(math.pi / n_walls) + 0.001
    for k in range(n_walls):
        a = 2 * math.pi * k / n_walls
        w = G.Cube.Define(u.stage, f"{base}/wall_{k}")
        w.CreateSizeAttr(1.0)
        x = G.Xformable(w)
        x.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(a), r_mid * math.sin(a), height / 2))
        x.AddRotateZOp().Set(math.degrees(a))
        x.AddScaleOp().Set(Gf.Vec3f(thick, seg, height))
        w.CreatePurposeAttr(G.Tokens.guide)
        u.UsdPhysics.CollisionAPI.Apply(w.GetPrim())
        u.bind(w.GetPrim(), phys)
    floor = u.cube(f"{base}/floor", (0, 0, 0.002), (r_in, r_in, 0.002), visible=False)
    u.bind(floor.GetPrim(), phys)


def _build_tray(u: _Usd, st, wall_h: float) -> None:
    """Static tray with three compartments along its length."""
    G, Gf = u.UsdGeom, u.Gf
    base = "/World/Desk/Tray"
    root = G.Xform.Define(u.stage, base)
    x = G.Xformable(root)
    x.AddTranslateOp().Set(Gf.Vec3d(st.xy[0], st.xy[1], 0.0))
    x.AddRotateZOp().Set(math.degrees(st.yaw))
    hx, hy = st.half
    t = 0.002  # half wall thickness
    phys = u.physics_material("desk")
    parts = [
        ("floor", (0, 0, 0.002), (hx, hy, 0.002), "tray_dark"),
        ("wall_px", (hx - t, 0, wall_h / 2), (t, hy, wall_h / 2), "tray"),
        ("wall_nx", (-hx + t, 0, wall_h / 2), (t, hy, wall_h / 2), "tray"),
        ("wall_py", (0, hy - t, wall_h / 2), (hx, t, wall_h / 2), "tray"),
        ("wall_ny", (0, -hy + t, wall_h / 2), (hx, t, wall_h / 2), "tray"),
        ("divider_a", (-hx / 3, 0, wall_h * 0.4), (t, hy, wall_h * 0.4), "tray"),
        ("divider_b", (hx / 3, 0, wall_h * 0.4), (t, hy, wall_h * 0.4), "tray"),
    ]
    for name, c, h, look in parts:
        g = u.cube(f"{base}/{name}", c, h, visible=False)
        u.bind(g.GetPrim(), phys)
        vis = u.mesh(f"{base}/{name}_visual", box_mesh(h, c))
        u.bind(vis.GetPrim(), u.fixed(look))


def apply_looks(scene: DeskScene, placements) -> None:
    """Set each item's texture, colours and finish for this episode."""
    from pxr import Gf, Sdf

    for p in placements:
        look = scene.looks.get(p.name)
        if look is None:
            continue
        body, prnt = np.asarray(p.body_rgb), np.asarray(p.print_rgb)
        tex = scene.textures.get(p.texture, scene.textures["solid"])
        look["tex"].GetInput("file").Set(Sdf.AssetPath(str(tex)))
        look["tex"].GetInput("scale").Set(Gf.Vec4f(*(body - prnt), 1.0))
        look["tex"].GetInput("bias").Set(Gf.Vec4f(*prnt, 0.0))
        look["surface"].GetInput("roughness").Set(float(p.roughness))
        look["surface"].GetInput("metallic").Set(float(p.metallic))
        look["accent"].GetInput("diffuseColor").Set(Gf.Vec3f(*p.accent_rgb))


# ======================================================================= CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--discover", action="store_true", help="Search the asset server and write --out.")
    p.add_argument("--assets-root", default=os.environ.get("KIKOBOT_ASSETS_ROOT"),
                   help="Isaac assets root (default: Isaac Sim's configured one).")
    p.add_argument("--folder", action="append", default=[],
                   help="Extra folder or URL to search (repeatable). Relative ones are under the assets root.")
    p.add_argument("--out", default=str(DEFAULT_ASSETS_JSON))
    p.add_argument("--max-depth", type=int, default=7)
    args = p.parse_args()
    if not args.discover:
        p.print_help()
        return

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        root = args.assets_root
        if not root:
            from isaacsim.storage.native import get_assets_root_path

            root = get_assets_root_path()
        if not root:
            raise SystemExit("No Isaac assets root found. Pass --assets-root or set KIKOBOT_ASSETS_ROOT.")
        roots = [f if "://" in f else f"{root.rstrip('/')}{f}" for f in (*DEFAULT_FOLDERS, *args.folder)]
        print(f"Searching {len(roots)} folders under {root} ...")
        found = discover(roots, args.max_depth)
        for kind, urls in found.items():
            print(f"  {kind:12s} {len(urls):3d}  {urls[0] if urls else '- (procedural)'}")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"assets_root": root, "kinds": found}, indent=2))
        print(f"Wrote {out}. Edit it to drop or reorder assets; empty kinds are built procedurally.")
    finally:
        app.close()


if __name__ == "__main__":
    main()
