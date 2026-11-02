"""Builds the MuJoCo scenes: the arm, in the lab of `room.py`, with a task on its table.

    spec = build_scene("pick_place")   # empty | pick_place | cylinder
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

SCENE_NAMES = ("empty", "pick_place", "cylinder")
TEXTURE_SIZE = 256

# Relative light intensity (room.py) to MuJoCo's light colours. MuJoCo adds
# lights linearly and clips, so six panels plus the sun stay well under 1.
PANEL_DIFFUSE = 0.2
PANEL_SPECULAR = 0.05
SUN_DIFFUSE = 0.45
SUN_SPECULAR = 0.15
PANEL_CUTOFF_DEG = 80.0


def _light_name(name: str) -> str:
    return f"room_{name}"


def _texture_name(m) -> str:
    return f"room_tex_{m.pattern}_{round(m.contrast * 100)}"


def _add_materials(spec: mujoco.MjSpec, room: RoomSpec) -> None:
    made = set()
    for m in room.materials.values():
        if not m.pattern or _texture_name(m) in made:
            continue
        made.add(_texture_name(m))
        size = TEXTURE_SIZE * 2 if m.pattern == "wood" else TEXTURE_SIZE
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
        if m.pattern:
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _texture_name(m)
            mat.texuniform = True
            mat.texrepeat = [m.repeat, m.repeat]


def _add_part(spec: mujoco.MjSpec, p: Part) -> None:
    if p.shape == "box":
        kind, size = mujoco.mjtGeom.mjGEOM_BOX, list(p.half)
    elif p.shape == "cylinder":
        kind, size = mujoco.mjtGeom.mjGEOM_CYLINDER, [p.half[0], p.half[2], 0.0]
    else:
        raise ValueError(f"Unknown part shape {p.shape!r}.")
    spec.worldbody.add_geom(
        name=f"room_{p.name}", type=kind, size=size, pos=list(p.center), material=f"room_{p.material}",
        contype=1 if p.collide else 0, conaffinity=1 if p.collide else 0,
        friction=[1.2, 0.02, 0.002],
    )


def _add_lights(spec: mujoco.MjSpec, room: RoomSpec) -> None:
    for l in room.lights:
        light = spec.worldbody.add_light(name=_light_name(l.name), pos=list(l.pos), dir=list(l.dir),
                                         castshadow=l.castshadow)
        if l.kind == "sun":
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            light.diffuse = [SUN_DIFFUSE * l.intensity] * 3
            light.specular = [SUN_SPECULAR * l.intensity] * 3
        else:
            light.type = mujoco.mjtLightType.mjLIGHT_SPOT
            light.cutoff = PANEL_CUTOFF_DEG
            light.exponent = 2.0
            light.diffuse = [PANEL_DIFFUSE * l.intensity] * 3
            light.specular = [PANEL_SPECULAR * l.intensity] * 3


def _add_cameras(spec: mujoco.MjSpec, room: RoomSpec) -> None:
    for c in room.cameras:
        right, up = look_at_axes(c.eye, c.target, c.up)
        cam = spec.worldbody.add_camera(name=c.name, pos=list(c.eye), fovy=c.fovy, resolution=[1280, 720])
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = [*right, *up]


def _visual(spec: mujoco.MjSpec) -> None:
    v = spec.visual
    # The room lights the scene; the headlight only lifts the deepest shadows.
    v.headlight.diffuse = [0.12, 0.12, 0.12]
    v.headlight.ambient = [0.15, 0.15, 0.15]
    v.headlight.specular = [0.0, 0.0, 0.0]
    v.quality.shadowsize = 4096
    v.quality.offsamples = 8
    v.global_.offwidth, v.global_.offheight = 1280, 720
    v.global_.azimuth, v.global_.elevation = 160.0, -20.0
    v.map.znear = 0.005
    v.map.zfar = 12.0
    v.map.shadowclip = 3.0
    v.rgba.haze = [0.85, 0.86, 0.88, 1.0]


def _add_pick_place(spec: mujoco.MjSpec, cylinder: bool) -> None:
    w = spec.worldbody
    if cylinder:
        # 30 mm across and 80 mm long, lying on its side. condim 6 adds rolling
        # friction, so a finger brushing it does not send it rolling off.
        body = w.add_body(name="target_cylinder", pos=[0.25, 0.0, 0.015], quat=[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
        body.add_freejoint(name="target_cylinder_free")
        body.add_geom(name="target_cylinder", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.015, 0.04, 0.0],
                      rgba=[0.95, 0.55, 0.1, 1.0], mass=0.03, friction=[1.5, 0.02, 0.002],
                      solref=[0.004, 1.0], condim=6)
    else:
        # A 32 mm cube: ~11 mm of approach clearance per side in the open jaws.
        body = w.add_body(name="target_cube", pos=[0.25, 0.0, 0.016])
        body.add_freejoint(name="target_cube_free")
        body.add_geom(name="target_cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.016, 0.016, 0.016],
                      rgba=[0.85, 0.12, 0.12, 1.0], mass=0.03, friction=[1.5, 0.02, 0.002],
                      solref=[0.004, 1.0], condim=4)
    # The place target: a mocap body is moved by writing data.mocap_pos and still collides.
    pad = w.add_body(name="place_pad", mocap=True, pos=[0.26, 0.22, 0.008])
    pad.add_geom(name="place_pad", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.055, 0.008, 0.0],
                 rgba=[0.13, 0.72, 0.25, 1.0], friction=[1.5, 0.02, 0.002], condim=4)


def build_scene(name: str, room: RoomSpec | None = None) -> mujoco.MjSpec:
    """The arm in the lab, with the named task on its table."""
    if name not in SCENE_NAMES:
        raise ValueError(f"Unknown scene {name!r}; use one of {SCENE_NAMES}.")
    room = room or lab_room()
    spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "robot.xml"))
    spec.modelname = f"kikobot_{name}"
    _visual(spec)
    _add_materials(spec, room)
    for p in room.parts:
        _add_part(spec, p)
    _add_lights(spec, room)
    _add_cameras(spec, room)
    if name in ("pick_place", "cylinder"):
        _add_pick_place(spec, cylinder=name == "cylinder")
    return spec


class RoomLights:
    """Writes one episode's `LightState`s into a compiled model."""

    def __init__(self, model: mujoco.MjModel, room: RoomSpec):
        self.model = model
        self.kinds = {l.name: l.kind for l in room.lights}
        self.ids = {}
        for l in room.lights:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, _light_name(l.name))
            if i >= 0:
                self.ids[l.name] = i

    def apply(self, states: list[LightState]) -> None:
        m = self.model
        for s in states:
            i = self.ids.get(s.name)
            if i is None:
                continue
            rgb = np.asarray(s.rgb) * s.intensity
            sun = self.kinds[s.name] == "sun"
            m.light_active[i] = s.on
            m.light_diffuse[i] = rgb * (SUN_DIFFUSE if sun else PANEL_DIFFUSE)
            m.light_specular[i] = rgb * (SUN_SPECULAR if sun else PANEL_SPECULAR)
            if sun:
                d = np.asarray(s.dir, dtype=float)
                m.light_dir[i] = d / np.linalg.norm(d)
