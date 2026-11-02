"""The desk sorting station in MuJoCo: the stations, the item pool, and a new desk every reset.

    add_desk(spec)                       # while building the scene
    desk = DeskRuntime(model)            # once compiled
    desk.reset(model, data, rng)         # each episode: new items, poses, colours, textures

The same stations (layout.STATIONS) and items (layout.KINDS, meshes.py) as
the Isaac Sim scene. Every pool item exists all the time: each reset puts a
random subset on the desk and parks the rest on the floor under the table.
Looks change by swapping materials' colours and textures, so the model is
never recompiled.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import layout as L
from . import meshes as M
from . import textures as T

TEXTURE_SIZE = 256
# Item textures tint between the body colour (pattern 1) and a dark print
# (pattern 0, here 25 % of the body colour: MuJoCo can only multiply).
PRINT_LEVEL = 0.25
Z_TO_X = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]  # Turns a geom's Z axis onto X.
RING_SEGMENTS = 12
HIDDEN_GROUP = 3  # Colliders: simulated, not drawn.


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _add_mesh(spec: mujoco.MjSpec, name: str, mesh: M.MeshBuilder) -> None:
    pts, nrm, st, counts = mesh.arrays()
    faces, k = [], 0
    for n in counts:  # Fan-triangulate each (convex) polygon.
        faces += [(k, k + i, k + i + 1) for i in range(1, n - 1)]
        k += n
    m = spec.add_mesh(name=name)
    m.uservert = pts.astype(float).ravel().tolist()
    m.usernormal = nrm.astype(float).ravel().tolist()
    m.usertexcoord = st.astype(float).ravel().tolist()
    m.userface = np.asarray(faces, dtype=int).ravel().tolist()
    m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL


def _phong(roughness: float, metallic: float) -> dict[str, float]:
    return dict(
        specular=float(0.5 * (1.0 - roughness) + 0.3 * metallic),
        shininess=float(1.0 - roughness),
        reflectance=float(0.15 * metallic),
    )


def _add_looks(spec: mujoco.MjSpec) -> None:
    for name in T.PATTERNS:
        gray = PRINT_LEVEL + (1.0 - PRINT_LEVEL) * np.clip(T.PATTERNS[name](TEXTURE_SIZE), 0.0, 1.0)
        img = (gray * 255.0 + 0.5).astype(np.uint8)
        h, w = img.shape
        tex = spec.add_texture(name=f"desk_tex_{name}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                               width=w, height=h, nchannel=3)
        tex.data = np.repeat(img[..., None], 3, axis=2).tobytes()
    for role, (rgb, rough, metal) in M.FIXED_LOOKS.items():
        spec.add_material(name=f"desk_fixed_{role}", rgba=[*rgb, 1.0], **_phong(rough, metal))


def _station_geoms(spec: mujoco.MjSpec, st: L.Station) -> None:
    c, s = math.cos(st.yaw), math.sin(st.yaw)
    for p in st.parts:
        x, y, z = p.center
        pos = [st.xy[0] + c * x - s * y, st.xy[1] + s * x + c * y, z]
        name = f"desk_{st.name}_{p.name}"
        common = dict(pos=pos, quat=_yaw_quat(st.yaw + p.yaw), material=f"desk_fixed_{p.role}",
                      friction=[1.5, 0.02, 0.002])
        if p.shape == "tube":
            _add_mesh(spec, f"{name}_mesh", M.station_tube(p.half[0], p.half[1], p.half[2]))
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_mesh",
                                    contype=0, conaffinity=0, **common)
            continue
        if p.shape == "box":
            kind, size = mujoco.mjtGeom.mjGEOM_BOX, list(p.half)
        else:
            kind, size = mujoco.mjtGeom.mjGEOM_CYLINDER, [p.half[0], p.half[2], 0.0]
        spec.worldbody.add_geom(
            name=name, type=kind, size=size,
            contype=1 if p.collide else 0, conaffinity=1 if p.collide else 0,
            group=0 if p.visual else HIDDEN_GROUP, rgba=[1, 1, 1, 1] if p.visual else [0, 0, 0, 0],
            **common,
        )


def _colliders(body, item: L.PoolItem) -> None:
    """Primitive colliders fitted to the item, the same shapes as the Isaac scene's."""
    hx, hy, hz = item.half
    spec = item.spec
    common = dict(group=HIDDEN_GROUP, rgba=[0, 0, 0, 0], friction=[spec.friction, 0.01, 0.001], condim=4)
    shape = spec.collider
    if item.kind == "scissors":
        sx, sy, sz = hx / 0.070, hy / 0.022, hz / 0.004
        body.add_geom(name=f"{body.name}_blades", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.045 * sx, 0.007 * sy, 0.002 * sz], pos=[0.025 * sx, 0, -0.002 * sz], **common)
        body.add_geom(name=f"{body.name}_handles", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.025 * sx, hy, hz], pos=[-0.045 * sx, 0, 0], **common)
    elif shape == "capsule":
        body.add_geom(name=f"{body.name}_col", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      size=[hy, max(hx - hy, 1e-4), 0.0], quat=Z_TO_X, **common)
    elif shape == "cylinder_x":
        body.add_geom(name=f"{body.name}_col", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[hy, hx, 0.0], quat=Z_TO_X, **common)
    elif shape == "cylinder_z":
        body.add_geom(name=f"{body.name}_col", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[hx, hz, 0.0], **common)
    elif shape == "ring":
        r_in, r_out = spec.ring_inner, hx
        r_mid, t = 0.5 * (r_in + r_out), 0.5 * (r_out - r_in)
        seg = r_mid * math.tan(math.pi / RING_SEGMENTS) + 0.0005
        for k in range(RING_SEGMENTS):
            a = 2 * math.pi * k / RING_SEGMENTS
            body.add_geom(name=f"{body.name}_col{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[t, seg, hz],
                          pos=[r_mid * math.cos(a), r_mid * math.sin(a), 0.0], quat=_yaw_quat(a), **common)
    else:
        body.add_geom(name=f"{body.name}_col", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz], **common)


def add_desk(spec: mujoco.MjSpec, pool: list[L.PoolItem] | None = None) -> list[L.PoolItem]:
    """The stations and every pool item (parked under the table)."""
    pool = pool if pool is not None else L.default_pool()
    _add_looks(spec)
    for st in L.STATIONS:
        _station_geoms(spec, st)
    for i, item in enumerate(pool):
        body = spec.worldbody.add_body(name=f"desk_{item.name}", pos=list(L.park_pose(i, item)))
        body.add_freejoint(name=f"desk_{item.name}_free")
        hx, hy, hz = item.half
        m = item.spec.mass
        body.explicitinertial = True
        body.mass = m
        body.inertia = [m / 3 * (hy**2 + hz**2), m / 3 * (hx**2 + hz**2), m / 3 * (hx**2 + hy**2)]
        body_mat = spec.add_material(name=f"desk_{item.name}_body", rgba=[0.8, 0.8, 0.8, 1.0])
        body_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "desk_tex_solid"
        spec.add_material(name=f"desk_{item.name}_accent", rgba=[0.1, 0.1, 0.8, 1.0])
        for j, (role, mesh) in enumerate(M.procedural_item(item.kind)):
            mesh_name = f"desk_{item.name}_mesh{j}"
            _add_mesh(spec, mesh_name, mesh)
            material = {"body": f"desk_{item.name}_body", "accent": f"desk_{item.name}_accent"}.get(role, f"desk_fixed_{role}")
            body.add_geom(name=f"desk_{item.name}_vis{j}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name,
                          material=material, contype=0, conaffinity=0, density=0.0)
        _colliders(body, item)
    return pool


class DeskRuntime:
    """Puts a new desk on the table: which items, where, and how they look."""

    def __init__(self, model: mujoco.MjModel, pool: list[L.PoolItem] | None = None):
        self.pool = pool if pool is not None else L.default_pool()

        def id_of(kind, name):
            i = mujoco.mj_name2id(model, kind, name)
            if i < 0:
                raise RuntimeError(f"The scene has no {name!r}: it was not built with add_desk.")
            return i

        joints = [id_of(mujoco.mjtObj.mjOBJ_JOINT, f"desk_{it.name}_free") for it in self.pool]
        self.qpos_adr = [int(model.jnt_qposadr[j]) for j in joints]
        self.dof_adr = [int(model.jnt_dofadr[j]) for j in joints]
        self.body_mat = [id_of(mujoco.mjtObj.mjOBJ_MATERIAL, f"desk_{it.name}_body") for it in self.pool]
        self.accent_mat = [id_of(mujoco.mjtObj.mjOBJ_MATERIAL, f"desk_{it.name}_accent") for it in self.pool]
        self.tex = {name: id_of(mujoco.mjtObj.mjOBJ_TEXTURE, f"desk_tex_{name}") for name in T.PATTERNS}
        self.placements: list[L.Placement] = []

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> list[L.Placement]:
        self.placements = L.sample_desk(rng, self.pool)
        rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
        for k, p in enumerate(self.placements):
            a, d = self.qpos_adr[k], self.dof_adr[k]
            data.qpos[a : a + 3] = p.pos
            data.qpos[a + 3 : a + 7] = _yaw_quat(p.yaw)
            data.qvel[d : d + 6] = 0.0
            mb, ma = self.body_mat[k], self.accent_mat[k]
            model.mat_rgba[mb] = [*p.body_rgb, 1.0]
            model.mat_texid[mb, rgb_role] = self.tex.get(p.texture, self.tex["solid"])
            look = _phong(p.roughness, p.metallic)
            model.mat_specular[mb] = look["specular"]
            model.mat_shininess[mb] = look["shininess"]
            model.mat_reflectance[mb] = look["reflectance"]
            model.mat_rgba[ma] = [*p.accent_rgb, 1.0]
        return self.placements
