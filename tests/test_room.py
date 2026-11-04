"""The lab room: one description, built the same in MuJoCo and in USD (Isaac Sim)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from lerobot_robot_kikobot.env import room as R
from lerobot_robot_kikobot.env.mujoco_scene import SCENE_NAMES, RoomLights, build_scene

LAB = R.lab_room()
# The arm's reach above its desk: a cylinder about the base. Only the desk
# and its hutch may enter it.
REACH_M, REACH_TOP_M = 0.45, 0.65


def _bounds(p):
    c, h = np.array(p.center), np.array(p.half)
    return c - h, c + h


def test_desk_top_is_flush_with_the_arm_base_in_its_front_left_corner():
    top = next(p for p in LAB.parts if p.name == "desk_top")
    lo, hi = _bounds(top)
    assert hi[2] == pytest.approx(0.0)
    assert np.allclose([lo[0], hi[0]], R.DESK_X) and np.allclose([lo[1], hi[1]], R.DESK_Y)
    # The base is near the front (+X) and left (-Y) edges, and the desk
    # reaches beyond the arm to the back and the right.
    assert 0.2 <= R.DESK_X[1] <= 0.35 and 0.2 <= -R.DESK_Y[0] <= 0.35
    assert -R.DESK_X[0] > REACH_M and R.DESK_Y[1] > REACH_M


def test_hutch_shelf_is_shallower_than_the_desk_and_above_the_arm():
    shelf = next(p for p in LAB.parts if p.name == "hutch_shelf")
    lo, hi = _bounds(shelf)
    assert hi[0] - lo[0] < 0.5 * (R.DESK_X[1] - R.DESK_X[0])
    assert lo[2] >= REACH_TOP_M
    for side in ("hutch_side_left", "hutch_side_right"):
        p = next(q for q in LAB.parts if q.name == side)
        assert p.shape == "prism" and len(p.profile) == 4
        top = [x for x, z in p.profile if z > 0]
        bottom = [x for x, z in p.profile if z < 0]
        assert max(top) - min(top) < max(bottom) - min(bottom)  # narrower at the top


def test_nothing_else_is_in_the_arms_way():
    for p in LAB.parts:
        if p.name.startswith(("desk_", "hutch_", "tube_")):
            continue
        lo, hi = _bounds(p)
        # Nearest point of the part's box to the base axis, below the reach top.
        near = np.clip([0.0, 0.0], lo[:2], hi[:2])
        assert np.hypot(*near) > REACH_M or lo[2] > REACH_TOP_M or hi[2] < 0.0, p.name


def test_the_hutch_keeps_clear_of_the_base():
    for p in LAB.parts:
        if not p.name.startswith(("hutch_", "tube_")):
            continue
        lo, hi = _bounds(p)
        near = np.clip([0.0, 0.0], lo[:2], hi[:2])
        assert np.hypot(*near) > 0.25, p.name


def test_everything_is_inside_the_room_and_has_a_material():
    lo = np.array([R.ROOM_X[0] - R.WALL_T - 1e-6, -R.ROOM_HALF_Y - R.WALL_T - 1e-6, R.FLOOR_Z - 0.05])
    hi = np.array([R.ROOM_X[1] + R.WALL_T + 1e-6, R.ROOM_HALF_Y + R.WALL_T + 1e-6, R.CEILING_Z + 0.05])
    for p in LAB.parts:
        a, b = _bounds(p)
        assert np.all(a >= lo) and np.all(b <= hi), p.name
        assert p.material in LAB.materials, p.name
    for l in LAB.lights:
        assert np.all(np.array(l.pos) >= lo) and np.all(np.array(l.pos) <= hi), l.name


def test_the_room_has_eight_downlights_and_a_tube_light():
    kinds = [l.kind for l in LAB.lights]
    assert kinds.count("downlight") == 8 and kinds.count("tube") == 1
    tube = next(l for l in LAB.lights if l.kind == "tube")
    # Under the hutch's shelf, over the desk.
    assert tube.pos[2] < R.HUTCH_SHELF_Z and R.DESK_X[0] < tube.pos[0] < R.DESK_X[1]
    assert tube.length > 1.0


def test_lighting_varies_but_stays_lit():
    rng = np.random.default_rng(0)
    runs = [R.sample_lighting(rng, LAB) for _ in range(50)]
    downs_on = [sum(s.on for s in run if s.name.startswith("down_")) for run in runs]
    assert min(downs_on) >= 2
    temps = {round(run[0].temp_k) for run in runs}
    assert len(temps) > 40
    assert any(s.on for run in runs for s in run if s.name == "tube")
    assert any(not s.on for run in runs for s in run if s.name == "tube")
    for run in runs:
        for s in run:
            assert max(s.rgb) == pytest.approx(1.0) and min(s.rgb) > 0.4


def test_kelvin_runs_warm_to_cool():
    warm, cool = R.kelvin_to_rgb(3000), R.kelvin_to_rgb(8000)
    assert warm[0] > warm[2] and cool[2] > cool[0]


@pytest.mark.parametrize("name", [n for n in SCENE_NAMES if n != "desk_sort"])
def test_mujoco_scene_compiles_and_holds_still(name):
    model = build_scene(name).compile()
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=500)
    assert np.all(np.isfinite(data.qpos)) and np.abs(data.qvel).max() < 5.0
    assert model.ncam == 1 and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "record") == 0


def test_record_camera_frames_the_arm_and_the_work_area():
    """The arm, every station and the whole clutter fan fall inside the image."""
    import math

    from lerobot_robot_kikobot.desk import layout as L

    (cam,) = LAB.cameras
    pts = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.40)]
    for st in L.STATIONS:
        r = math.hypot(*st.half)
        pts += [(st.xy[0] + r * math.cos(a), st.xy[1] + r * math.sin(a), 0.0) for a in np.linspace(0, 2 * math.pi, 8)]
    for rr in L.CLUTTER_RADIUS_M:
        for aa in np.linspace(-L.CLUTTER_HALF_ANGLE_DEG, L.CLUTTER_HALF_ANGLE_DEG, 7) + L.WORK_YAW_DEG:
            pts.append((rr * math.cos(math.radians(aa)), rr * math.sin(math.radians(aa)), 0.0))
    eye = np.array(cam.eye)
    right, up = R.look_at_axes(cam.eye, cam.target, cam.up)
    fwd = np.cross(up, right)
    t = math.tan(math.radians(cam.fovy) / 2)
    aspect = 640 / 480
    for p in pts:
        d = np.asarray(p) - eye
        z = d @ fwd
        assert z > 0
        assert abs(d @ right / (z * t * aspect)) < 0.95 and abs(d @ up / (z * t)) < 0.95, p


def test_mujoco_lighting_changes_the_image():
    model = build_scene("empty").compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lights = RoomLights(model, LAB)
    renderer = mujoco.Renderer(model, 120, 160)

    def shot(states):
        lights.apply(states)
        renderer.update_scene(data, camera="record")
        return renderer.render().astype(float)

    bright = [s.__class__(**{**s.__dict__, "intensity": 1.3}) for s in R.fixed_lighting(LAB)]
    dark = [s.__class__(**{**s.__dict__, "on": s.name == "down_00"}) for s in R.fixed_lighting(LAB)]
    assert shot(bright).mean() > shot(dark).mean() + 5
    renderer.close()


# ----------------------------------------------------------------- USD


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_usd_room_matches_the_description(tmp_path):
    pytest.importorskip("pxr")
    import types

    from pxr import Gf, Usd, UsdGeom, UsdLux

    pxr = types.SimpleNamespace(Gf=Gf, Usd=Usd, UsdGeom=UsdGeom, UsdLux=UsdLux)
    repo = Path(__file__).resolve().parent.parent
    _load("kikobot_room", repo / "lerobot_robot_kikobot" / "env" / "room.py")
    _load("desk_textures", repo / "lerobot_robot_kikobot" / "desk" / "textures.py")
    room_usd = _load("room_usd", repo / "isaac" / "room_usd.py")
    import kikobot_room

    lab = kikobot_room.lab_room()
    stage = pxr.Usd.Stage.CreateInMemory()
    pxr.UsdGeom.SetStageUpAxis(stage, pxr.UsdGeom.Tokens.z)
    lights = room_usd.build_room(stage, lab, tmp_path)
    for p in lab.parts:
        prim = stage.GetPrimAtPath(f"/World/Room/Parts/{p.name}")
        assert prim, p.name
        box = pxr.UsdGeom.Imageable(prim).ComputeWorldBound(0, "default").ComputeAlignedRange()
        assert np.allclose(box.GetMin(), np.array(p.center) - p.half, atol=1e-5), p.name
        assert np.allclose(box.GetMax(), np.array(p.center) + p.half, atol=1e-5), p.name
    assert any(tmp_path.glob("speckle_*.png"))
    states = kikobot_room.fixed_lighting(lab)
    states[0] = states[0].__class__(**{**states[0].__dict__, "on": False})
    lights.apply(states)
    off = pxr.UsdLux.LightAPI(stage.GetPrimAtPath(f"/World/Room/Lights/{states[0].name}"))
    on = pxr.UsdLux.LightAPI(stage.GetPrimAtPath(f"/World/Room/Lights/{states[1].name}"))
    assert off.GetIntensityAttr().Get() == 0.0 and on.GetIntensityAttr().Get() > 0.0
    # The tube light is a cylinder light along Y, the downlights disks.
    assert stage.GetPrimAtPath("/World/Room/Lights/tube").IsA(pxr.UsdLux.CylinderLight)
    assert stage.GetPrimAtPath("/World/Room/Lights/down_00").IsA(pxr.UsdLux.DiskLight)
