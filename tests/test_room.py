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
# The arm's reach above its table, which only the table may enter.
WORK = (np.array([-0.5, -0.5, 0.0]), np.array([0.5, 0.5, 0.65]))


def _bounds(p):
    c, h = np.array(p.center), np.array(p.half)
    return c - h, c + h


def test_table_top_is_flush_with_the_arm_base():
    top = next(p for p in LAB.parts if p.name == "table_top")
    lo, hi = _bounds(top)
    assert hi[2] == pytest.approx(0.0)
    assert np.allclose(hi[:2], R.TABLE_HALF) and np.allclose(lo[:2], -R.TABLE_HALF)


def test_nothing_else_is_in_the_arms_way():
    for p in LAB.parts:
        if p.name.startswith("table_"):
            continue
        lo, hi = _bounds(p)
        overlap = np.all(lo < WORK[1]) and np.all(hi > WORK[0])
        assert not overlap, p.name


def test_everything_is_inside_the_room_and_has_a_material():
    lo = np.array([R.ROOM_X[0] - R.WALL_T - 1e-6, -R.ROOM_HALF_Y - R.WALL_T - 1e-6, R.FLOOR_Z - 0.05])
    hi = np.array([R.ROOM_X[1] + R.WALL_T + 1e-6, R.ROOM_HALF_Y + R.WALL_T + 1e-6, R.CEILING_Z + 0.05])
    for p in LAB.parts:
        a, b = _bounds(p)
        assert np.all(a >= lo) and np.all(b <= hi), p.name
        assert p.material in LAB.materials, p.name
    for l in LAB.lights:
        assert np.all(np.array(l.pos) >= lo) and np.all(np.array(l.pos) <= hi), l.name


def test_lighting_varies_but_stays_lit():
    rng = np.random.default_rng(0)
    runs = [R.sample_lighting(rng, LAB) for _ in range(50)]
    panels_on = [sum(s.on for s in run if s.name.startswith("panel")) for run in runs]
    assert min(panels_on) >= 2
    temps = {round(run[0].temp_k) for run in runs}
    assert len(temps) > 40
    assert any(s.on for run in runs for s in run if s.name == "sun")
    assert any(not s.on for run in runs for s in run if s.name == "sun")
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
    for cam in ("record", "overhead"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0


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
    dark = [s.__class__(**{**s.__dict__, "on": s.name == "panel_00"}) for s in R.fixed_lighting(LAB)]
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
    # The sun shines along its direction: its -Z axis, turned by the orient op.
    sun = stage.GetPrimAtPath("/World/Room/Lights/sun")
    m = pxr.UsdGeom.Xformable(sun).ComputeLocalToWorldTransform(0)
    d = m.TransformDir(pxr.Gf.Vec3d(0, 0, -1))
    want = next(s for s in states if s.name == "sun").dir
    assert np.allclose(np.array(d) / np.linalg.norm(d), want, atol=1e-5)
