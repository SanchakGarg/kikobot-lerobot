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

# The item and station meshes and their fixed looks, shared with the MuJoCo
# scene (lerobot_robot_kikobot/desk/meshes.py), loaded by the server first.
from desk_meshes import (  # noqa: E402
    FIXED_LOOKS, MeshBuilder, box_mesh, procedural_item, revolve_x, station_tube, x_to_z,
)

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
    "sticky_notes": ("sticky", "postit", "notepad"),
    "stapler": ("stapler",),
    "ruler": ("ruler",),
    "paper_clip_box": ("paperclip", "clips"),
    "sharpener": ("sharpener",),
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


# ======================================================================= USD


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

    def cube(self, path: str, center, half, collide: bool = True, visible: bool = True, yaw: float = 0.0):
        G, Gf = self.UsdGeom, self.Gf
        g = G.Cube.Define(self.stage, path)
        g.CreateSizeAttr(1.0)
        x = G.Xformable(g)
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        if yaw:
            x.AddRotateZOp().Set(math.degrees(yaw))
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
        from desk_layout import KINDS  # noqa: PLC0415 - loaded by the server

        shape = KINDS[kind].collider
        if shape == "ring":
            # A hollow tape roll: a ring of boxes, so it can drop over the tape peg.
            spec, n = KINDS[kind], 12
            r_mid, t = 0.5 * (spec.ring_inner + hx), 0.5 * (hx - spec.ring_inner)
            seg = r_mid * math.tan(math.pi / n) + 0.0005
            prims = []
            for k in range(n):
                a = 2 * math.pi * k / n
                g = u.cube(f"{path}/collider_{k}", (r_mid * math.cos(a), r_mid * math.sin(a), 0.0), (t, seg, hz),
                           visible=False, yaw=a)
                prims.append(g)
        elif shape == "cylinder_x":
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
    """The stations and every pool item, on a stage that already has the table.

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

    for st in L.STATIONS:
        _build_station(u, st)

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


def _build_station(u: _Usd, st) -> None:
    """A static station from its parts (desk_layout.STATIONS), as the MuJoCo scene builds it."""
    G, Gf = u.UsdGeom, u.Gf
    base = f"/World/Desk/Stations/{st.name}"
    root = G.Xform.Define(u.stage, base)
    x = G.Xformable(root)
    x.AddTranslateOp().Set(Gf.Vec3d(st.xy[0], st.xy[1], 0.0))
    x.AddRotateZOp().Set(math.degrees(st.yaw))
    phys = u.physics_material("desk")
    for p in st.parts:
        c, h = np.asarray(p.center, float), np.asarray(p.half, float)
        if p.visual:
            if p.shape == "tube":
                vis = station_tube(h[0], h[1], h[2], c)
            elif p.shape == "cylinder":
                vis = x_to_z(revolve_x(h[0], h[0], -h[2], h[2], 32), c)
            else:
                cz, sz = math.cos(p.yaw), math.sin(p.yaw)
                rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
                vis = MeshBuilder().extend(box_mesh(h), (rz, c))
            g = u.mesh(f"{base}/{p.name}_visual", vis)
            u.bind(g.GetPrim(), u.fixed(p.role))
        if p.collide and p.shape != "tube":
            if p.shape == "cylinder":
                col = u.collider(f"{base}/{p.name}", "cylinder", (h[0], h[0], h[2]), c, axis="Z")
            else:
                col = u.cube(f"{base}/{p.name}", c, h, visible=False, yaw=p.yaw)
            u.bind(col.GetPrim(), phys)


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
