"""The lab of `lerobot_robot_kikobot/env/room.py`, built in USD for Isaac Sim.

The same surfaces, colours and lights as the MuJoCo scene (env/mujoco_scene.py):
textured surfaces are UV-mapped box meshes with a hidden box collider, the
rest plain cubes and cylinders; the ceiling panels are RectLights, the
daylight a DistantLight, plus a faint DomeLight for the light that bounces
around a real room. `RoomLightsUsd.apply` sets one episode's lighting.

Loaded by the Isaac Sim server by path; the room description comes in as the
module `kikobot_room`, the texture patterns as `desk_textures` (both loaded
by the server first). Nothing here imports pxr at module level, so the file
also loads (and is tested, with `pip install usd-core`) without Isaac Sim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Relative light intensity (room.py) to Isaac Sim's units (nits for area
# lights, lux-like for the distant light). `--light-scale` scales them all.
PANEL_INTENSITY = 9000.0
SUN_INTENSITY = 2500.0
DOME_INTENSITY = 120.0
ROOT = "/World/Room"
# Material binding purpose PhysX and Newton read friction from.
PHYSICS_PURPOSE = "physics"


def _quat_from_minus_z(d) -> tuple[float, float, float, float]:
    """(w, x, y, z) turning a light's -Z (where USD lights shine) onto `d`."""
    a = np.array([0.0, 0.0, -1.0])
    b = np.asarray(d, dtype=float)
    b = b / np.linalg.norm(b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)  # Half a turn about X.
    axis = np.cross(a, b)
    w = 1.0 + c
    q = np.array([w, *axis])
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def box_mesh_uv(half, repeat: float):
    """24 vertices, 6 quads; UVs in metres times `repeat`, so a texture keeps
    its scale whatever the box's size (MuJoCo's texuniform does the same)."""
    hx, hy, hz = (float(h) for h in half)
    faces = [  # (normal, u axis, v axis)
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
    ]
    h = np.array([hx, hy, hz])
    pts, nrm, st = [], [], []
    for n, u, v in faces:
        n, u, v = (np.array(x, dtype=float) for x in (n, u, v))
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = (n + su * u + sv * v) * h
            pts.append(p)
            nrm.append(n)
            st.append((float(np.dot(p, u)) * repeat, float(np.dot(p, v)) * repeat))
    return np.array(pts), np.array(nrm), np.array(st), [4] * 6


def write_room_textures(room, folder: Path) -> dict[tuple[str, float], Path]:
    """One PNG per (pattern, contrast) the room uses, written once."""
    import desk_textures as T  # noqa: PLC0415 - loaded by the server
    import kikobot_room as R  # noqa: PLC0415

    out = {}
    for m in room.materials.values():
        if not m.pattern or (m.pattern, m.contrast) in out:
            continue
        p = Path(folder) / f"{m.pattern}_{round(m.contrast * 100)}.png"
        if not p.is_file():
            size = 512 if m.pattern == "wood" else 256
            T.write_png(p, R.texture_values(T.PATTERNS[m.pattern](size), m.contrast))
        out[(m.pattern, m.contrast)] = p
    return out


class RoomLightsUsd:
    """The room's light prims; `apply` sets one episode's `LightState`s."""

    def __init__(self, stage, room, scale: float = 1.0):
        self.stage = stage
        self.scale = scale
        self.kinds = {l.name: l.kind for l in room.lights}
        self.paths = {l.name: f"{ROOT}/Lights/{l.name}" for l in room.lights}

    def apply(self, states) -> None:
        from pxr import Gf, UsdGeom, UsdLux  # noqa: PLC0415

        for s in states:
            prim = self.stage.GetPrimAtPath(self.paths.get(s.name, ""))
            if not prim:
                continue
            light = UsdLux.LightAPI(prim)
            base = SUN_INTENSITY if self.kinds[s.name] == "sun" else PANEL_INTENSITY
            light.GetIntensityAttr().Set(float(base * s.intensity * self.scale) if s.on else 0.0)
            light.GetColorTemperatureAttr().Set(float(s.temp_k))
            if self.kinds[s.name] == "sun":
                w, x, y, z = _quat_from_minus_z(s.dir)
                orient = [op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                          if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
                if orient:
                    orient[0].Set(Gf.Quatf(w, x, y, z))


def build_room(stage, room, textures_dir: Path, light_scale: float = 1.0) -> RoomLightsUsd:
    """Every part, light and material of `room` under /World/Room."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: PLC0415

    UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.Scope.Define(stage, f"{ROOT}/Looks")
    textures = write_room_textures(room, textures_dir)

    # Materials ------------------------------------------------------------
    looks = {}
    for m in room.materials.values():
        path = f"{ROOT}/Looks/{m.name}"
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, f"{path}/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(m.roughness))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(m.metallic))
        diffuse = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        if m.pattern:
            reader = UsdShade.Shader.Define(stage, f"{path}/StReader")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
            tex = UsdShade.Shader.Define(stage, f"{path}/Pattern")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(textures[(m.pattern, m.contrast)])))
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            # The pattern tints the colour, as MuJoCo's rgba x texture does.
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*m.rgb, 1.0))
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            diffuse.ConnectToSource(tex.ConnectableAPI(), "rgb")
        else:
            diffuse.Set(Gf.Vec3f(*m.rgb))
        if m.emission > 0:
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(c * m.emission for c in m.rgb)))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        looks[m.name] = mat
    friction = UsdShade.Material.Define(stage, f"{ROOT}/Looks/Physics")
    pm = UsdPhysics.MaterialAPI.Apply(friction.GetPrim())
    pm.CreateStaticFrictionAttr(1.2)
    pm.CreateDynamicFrictionAttr(1.2)
    pm.CreateRestitutionAttr(0.0)

    def bind(prim, mat, purpose=UsdShade.Tokens.allPurpose) -> None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose=purpose)

    def collider(path, center, half):
        g = UsdGeom.Cube.Define(stage, path)
        g.CreateSizeAttr(1.0)
        x = UsdGeom.Xformable(g)
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        x.AddScaleOp().Set(Gf.Vec3f(*(2.0 * float(h) for h in half)))
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        bind(g.GetPrim(), friction, PHYSICS_PURPOSE)
        return g

    # Parts ----------------------------------------------------------------
    UsdGeom.Xform.Define(stage, f"{ROOT}/Parts")
    for p in room.parts:
        path = f"{ROOT}/Parts/{p.name}"
        m = room.materials[p.material]
        if p.shape == "cylinder":
            g = UsdGeom.Cylinder.Define(stage, path)
            g.CreateRadiusAttr(float(p.half[0]))
            g.CreateHeightAttr(2.0 * float(p.half[2]))
            g.CreateAxisAttr("Z")
            UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(*map(float, p.center)))
            if p.collide:
                UsdPhysics.CollisionAPI.Apply(g.GetPrim())
                bind(g.GetPrim(), friction, PHYSICS_PURPOSE)
        elif m.pattern:
            # Textured: a UV-mapped mesh to look at, a hidden cube to collide with.
            pts, nrm, st, counts = box_mesh_uv(p.half, m.repeat)
            g = UsdGeom.Mesh.Define(stage, path)
            g.CreatePointsAttr([Gf.Vec3f(*q) for q in pts])
            g.CreateFaceVertexCountsAttr(counts)
            g.CreateFaceVertexIndicesAttr(list(range(len(pts))))
            g.CreateNormalsAttr([Gf.Vec3f(*n) for n in nrm])
            g.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            g.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            pv = UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set([Gf.Vec2f(*s) for s in st])
            g.CreateExtentAttr([Gf.Vec3f(*pts.min(0)), Gf.Vec3f(*pts.max(0))])
            UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(*map(float, p.center)))
            if p.collide:
                c = collider(f"{ROOT}/Parts/{p.name}_collider", p.center, p.half)
                c.CreatePurposeAttr(UsdGeom.Tokens.guide)
        else:
            g = UsdGeom.Cube.Define(stage, path)
            g.CreateSizeAttr(1.0)
            x = UsdGeom.Xformable(g)
            x.AddTranslateOp().Set(Gf.Vec3d(*map(float, p.center)))
            x.AddScaleOp().Set(Gf.Vec3f(*(2.0 * float(h) for h in p.half)))
            if p.collide:
                UsdPhysics.CollisionAPI.Apply(g.GetPrim())
                bind(g.GetPrim(), friction, PHYSICS_PURPOSE)
        bind(g.GetPrim(), looks[p.material])

    # Lights ---------------------------------------------------------------
    UsdGeom.Xform.Define(stage, f"{ROOT}/Lights")
    for l in room.lights:
        path = f"{ROOT}/Lights/{l.name}"
        if l.kind == "sun":
            light = UsdLux.DistantLight.Define(stage, path)
            light.CreateAngleAttr(0.53)
            base = SUN_INTENSITY
            x = UsdGeom.Xformable(light)
            x.AddOrientOp().Set(Gf.Quatf(*_quat_from_minus_z(l.dir)))
        else:
            light = UsdLux.RectLight.Define(stage, path)
            light.CreateWidthAttr(2.0 * l.size[0])
            light.CreateHeightAttr(2.0 * l.size[1])
            base = PANEL_INTENSITY
            # A RectLight shines along its -Z: straight down from the ceiling.
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*map(float, l.pos)))
        light.CreateIntensityAttr(float(base * l.intensity * light_scale))
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(float(l.temp_k))
        UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(True)
    dome = UsdLux.DomeLight.Define(stage, f"{ROOT}/Lights/ambient")
    dome.CreateIntensityAttr(float(DOME_INTENSITY * light_scale))
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.93, 0.95))
    return RoomLightsUsd(stage, room, light_scale)

