"""Builds the MuJoCo scenes: the arm, in the lab of `room.py`, with a task on its desk.

    spec = build_scene("pick_place")   # empty | pick_place | cylinder | desk_sort
    model = spec.compile()
    lights = RoomLights(model, lab_room())
    lights.apply(sample_lighting(rng, lab_room()))   # once per reset

Built in code with `mujoco.MjSpec`, from the same description as the Isaac
Sim scene, so the two look alike. Textures are generated in memory, so
nothing is written to disk.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ..desk import textures as desk_textures
from ..paths import MODELS_DIR
from .room import LightState, Part, RoomSpec, lab_room, look_at_axes, texture_values

SCENE_NAMES = ("empty", "pick_place", "cylinder", "desk_sort")
TEXTURE_SIZE = 256

# Relative light intensity (room.py) to MuJoCo's light colours, per kind of
# light. MuJoCo adds lights linearly (no fall-off with distance) and clips, so
# eight downlights plus the tube stay well under 1 on the desk. The tube is
# drawn as TUBE_SPOTS spot lights along its length.
LIGHT_TERMS = {
    # kind: (diffuse, specular, ambient: each light also lifts the shadows a
    # little, as light bouncing off the walls would)
    "downlight": (0.20, 0.04, 0.035),
    "tube": (0.50, 0.10, 0.06),
}
DOWNLIGHT_CUTOFF_DEG = 75.0
# A shadow map spans the light's whole cone: a shadow-casting downlight gets a
# narrower one (it still covers the desk), or its shadows come out jagged.
SHADOW_DOWNLIGHT_CUTOFF_DEG = 40.0
TUBE_SPOTS = 3
TUBE_CUTOFF_DEG = 70.0
# Soft shadow edges: the lamp is this big, not a point.
SHADOW_BULB_RADIUS_M = {"downlight": 0.06, "tube": 0.15}


def _light_name(name: str) -> str:
    return f"room_{name}"


def _texture_name(m) -> str:
    return f"room_tex_{m.pattern}_{round(m.contrast * 100)}"


def _add_materials(spec: mujoco.MjSpec, room: RoomSpec, textures: bool = True) -> None:
    made = set()
    for m in room.materials.values():
        if not textures or not m.pattern or _texture_name(m) in made:
            continue
        made.add(_texture_name(m))
        size = TEXTURE_SIZE * 2 if m.pattern in ("wood", "laminate") else TEXTURE_SIZE
        gray = texture_values(desk_textures.PATTERNS[m.pattern](size), m.contrast)
        img = (gray * 255.0 + 0.5).astype(np.uint8)
        h, w = img.shape
        tex = spec.add_texture(name=_texture_name(m), type=mujoco.mjtTexture.mjTEXTURE_2D,
                               width=w, height=h, nchannel=3)
        tex.data = np.repeat(img[..., None], 3, axis=2).tobytes()
    for m in room.materials.values():
        mat = spec.add_material(
            name=f"room_{m.name}",
            rgba=[*m.rgb, 1.0],
            # The classic renderer's Phong terms from the PBR description.
            specular=float(0.5 * (1.0 - m.roughness) + 0.3 * m.metallic),
            shininess=float(1.0 - m.roughness),
            reflectance=float(0.12 * m.metallic + (0.06 if m.roughness < 0.3 else 0.0)),
            emission=float(m.emission),
        )
        if textures and m.pattern:
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _texture_name(m)
            mat.texuniform = True
            mat.texrepeat = [m.repeat, m.repeat]


def prism_vertices(p: Part) -> np.ndarray:
    """The corners of a prism part, about its centre."""
    if p.extrude == "x":
        hx = p.half[0]
        return np.array([(s * hx, y, z) for s in (-1.0, 1.0) for y, z in p.profile], dtype=float)
    hy = p.half[1]
    return np.array([(x, s * hy, z) for s in (-1.0, 1.0) for x, z in p.profile], dtype=float)


def _add_part(spec: mujoco.MjSpec, p: Part) -> None:
    extra = {}
    if p.shape == "box":
        kind, size = mujoco.mjtGeom.mjGEOM_BOX, list(p.half)
    elif p.shape == "cylinder":
        kind, size = mujoco.mjtGeom.mjGEOM_CYLINDER, [p.half[0], p.half[2], 0.0]
    elif p.shape == "prism":
        # A convex mesh: MuJoCo builds its hull from the corners.
        mesh = spec.add_mesh(name=f"room_{p.name}_mesh")
        mesh.uservert = prism_vertices(p).ravel().tolist()
        kind, size = mujoco.mjtGeom.mjGEOM_MESH, [0.0, 0.0, 0.0]
        extra["meshname"] = f"room_{p.name}_mesh"
    else:
        raise ValueError(f"Unknown part shape {p.shape!r}.")
    spec.worldbody.add_geom(
        name=f"room_{p.name}", type=kind, size=size, pos=list(p.center), material=f"room_{p.material}",
        contype=1 if p.collide else 0, conaffinity=1 if p.collide else 0,
        friction=[1.2, 0.02, 0.002], **extra,
    )


def _spots(l) -> list[tuple[str, list[float], bool]]:
    """MuJoCo lights for one room light: (name, position, casts a shadow).
    A tube is a row of spots along X; only the middle one casts a shadow."""
    if l.kind != "tube":
        return [(_light_name(l.name), list(l.pos), l.castshadow)]
    out = []
    for k in range(TUBE_SPOTS):
        x = l.pos[0] + l.length * ((k + 0.5) / TUBE_SPOTS - 0.5)
        out.append((f"{_light_name(l.name)}_{k}", [x, l.pos[1], l.pos[2]], l.castshadow and k == TUBE_SPOTS // 2))
    return out


def _add_lights(spec: mujoco.MjSpec, room: RoomSpec) -> None:
    for l in room.lights:
        diffuse, specular, ambient = LIGHT_TERMS[l.kind]
        spots = _spots(l)
        share = l.intensity / len(spots)
        for name, pos, shadow in spots:
            light = spec.worldbody.add_light(name=name, pos=pos, dir=list(l.dir), castshadow=shadow)
            light.type = mujoco.mjtLightType.mjLIGHT_SPOT
            if l.kind == "tube":
                light.cutoff = TUBE_CUTOFF_DEG
            else:
                light.cutoff = SHADOW_DOWNLIGHT_CUTOFF_DEG if shadow else DOWNLIGHT_CUTOFF_DEG
            if shadow:
                light.bulbradius = SHADOW_BULB_RADIUS_M[l.kind]
            light.exponent = 1.0
            light.diffuse = [diffuse * share] * 3
            light.specular = [specular * share] * 3
            light.ambient = [ambient * share] * 3


def _add_cameras(spec: mujoco.MjSpec, room: RoomSpec) -> None:
    for c in room.cameras:
        right, up = look_at_axes(c.eye, c.target, c.up)
        cam = spec.worldbody.add_camera(name=c.name, pos=list(c.eye), fovy=c.fovy, resolution=[1280, 720])
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = [*right, *up]


def _visual(spec: mujoco.MjSpec) -> None:
    v = spec.visual
    # The room lights the scene; the headlight only lifts the deepest shadows.
    v.headlight.diffuse = [0.06, 0.06, 0.06]
    v.headlight.ambient = [0.10, 0.10, 0.10]
    v.headlight.specular = [0.0, 0.0, 0.0]
    v.quality.shadowsize = 4096
    v.quality.offsamples = 8
    v.global_.offwidth, v.global_.offheight = 1280, 720
    v.global_.azimuth, v.global_.elevation = 160.0, -20.0
    v.map.znear = 0.005
    v.map.zfar = 12.0
    v.rgba.haze = [0.85, 0.86, 0.88, 1.0]


def _add_pick_place(spec: mujoco.MjSpec, cylinder: bool) -> None:
    w = spec.worldbody
    if cylinder:
        # 30 mm across and 80 mm long, lying on its side. condim 6 adds rolling
        # friction, so a finger brushing it does not send it rolling off.
        body = w.add_body(name="target_cylinder", pos=[-0.125, 0.217, 0.015], quat=[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
        body.add_freejoint(name="target_cylinder_free")
        body.add_geom(name="target_cylinder", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.015, 0.04, 0.0],
                      rgba=[0.95, 0.55, 0.1, 1.0], mass=0.03, friction=[1.5, 0.02, 0.002],
                      solref=[0.004, 1.0], condim=6)
    else:
        # A 32 mm cube: ~11 mm of approach clearance per side in the open jaws.
        body = w.add_body(name="target_cube", pos=[-0.125, 0.217, 0.016])
        body.add_freejoint(name="target_cube_free")
        body.add_geom(name="target_cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.016, 0.016, 0.016],
                      rgba=[0.85, 0.12, 0.12, 1.0], mass=0.03, friction=[1.5, 0.02, 0.002],
                      solref=[0.004, 1.0], condim=4)
    # The place target: a mocap body is moved by writing data.mocap_pos and still collides.
    pad = w.add_body(name="place_pad", mocap=True, pos=[-0.225, 0.13, 0.008])
    pad.add_geom(name="place_pad", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.055, 0.008, 0.0],
                 rgba=[0.13, 0.72, 0.25, 1.0], friction=[1.5, 0.02, 0.002], condim=4)


def build_scene(name: str, room: RoomSpec | None = None, textures: bool = True) -> mujoco.MjSpec:
    """The arm in the lab, with the named task on its desk.

    `textures=False` leaves out the (in-memory) textures, so the spec can be
    written out as MJCF (`to_xml`) for an engine that imports MJCF, such as
    Newton: same bodies, joints and actuators, plain colours.
    """
    if name not in SCENE_NAMES:
        raise ValueError(f"Unknown scene {name!r}; use one of {SCENE_NAMES}.")
    room = room or lab_room()
    spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "robot.xml"))
    spec.modelname = f"kikobot_{name}"
    # Absolute, so the MJCF written from this spec finds the arm's meshes anywhere.
    spec.meshdir = str(MODELS_DIR / "assets")
    _visual(spec)
    _add_materials(spec, room, textures)
    for p in room.parts:
        _add_part(spec, p)
    _add_lights(spec, room)
    _add_cameras(spec, room)
    if name in ("pick_place", "cylinder"):
        _add_pick_place(spec, cylinder=name == "cylinder")
    elif name == "desk_sort":
        from ..desk.mujoco_desk import add_desk  # noqa: PLC0415

        add_desk(spec, textures=textures)
    return spec


def scene_mjcf(name: str, room: RoomSpec | None = None) -> str:
    """The scene as MJCF text, without textures (see `build_scene`).

    Position actuators carry their damping as `dampratio`, which MuJoCo turns
    into a velocity gain when it compiles; MJCF importers such as Newton's
    skip it, and the servos then ring. So the compiled gain is written out
    explicitly.
    """
    spec = build_scene(name, room, textures=False)
    return explicit_damping_mjcf(spec)


def explicit_damping_mjcf(spec: mujoco.MjSpec) -> str:
    model = spec.compile()
    for i, act in enumerate(spec.actuators):
        bias = list(act.biasprm)
        bias[2] = float(model.actuator_biasprm[i][2])  # -kv, once compiled
        act.biasprm = bias
    return spec.to_xml()


class RoomLights:
    """Writes one episode's `LightState`s into a compiled model."""

    def __init__(self, model: mujoco.MjModel, room: RoomSpec):
        self.model = model
        self.kinds = {l.name: l.kind for l in room.lights}
        self.ids: dict[str, list[int]] = {}
        for l in room.lights:
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name) for name, _, _ in _spots(l)]
            ids = [i for i in ids if i >= 0]
            if ids:
                self.ids[l.name] = ids

    def apply(self, states: list[LightState]) -> None:
        m = self.model
        for s in states:
            ids = self.ids.get(s.name)
            if not ids:
                continue
            diffuse, specular, ambient = LIGHT_TERMS[self.kinds[s.name]]
            rgb = np.asarray(s.rgb) * s.intensity / len(ids)
            for i in ids:
                m.light_active[i] = s.on
                m.light_diffuse[i] = rgb * diffuse
                m.light_specular[i] = rgb * specular
                m.light_ambient[i] = rgb * ambient
