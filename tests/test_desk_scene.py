"""The desk sorting station in MuJoCo (and its USD twin): builds, settles, re-randomizes."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from lerobot_robot_kikobot.desk import layout as L
from lerobot_robot_kikobot.sim_server import SimWorld

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def world():
    return SimWorld("desk_sort", seed=1)


def _item_positions(w):
    return {p.name: w.data.qpos[a : a + 3].copy() for p, a in zip(w._desk.placements, w._desk.qpos_adr)}


def test_desk_settles_on_the_desk(world):
    w = world
    w.step(500)  # 1 s more after the reset's own settling.
    speed = max(np.linalg.norm(w.data.qvel[a : a + 6]) for a in w._desk.dof_adr)
    assert speed < 0.1, f"the desk is not at rest: an item still moves at {speed:.2f}"
    for p, pos in zip(w._desk.placements, _item_positions(w).values()):
        if p.active:
            assert -0.01 < pos[2] < 0.12, (p.name, pos)  # on the desk, or in a cup
            assert math.hypot(*pos[:2]) < 0.5, (p.name, pos)
        else:
            assert pos[2] < L.FLOOR_Z + 0.1, (p.name, pos)  # still parked on the floor


@pytest.mark.parametrize("seed", [2, 7, 11])
def test_every_desk_comes_to_rest(seed):
    w = SimWorld("desk_sort", seed=seed)
    w.step(500)
    speed = max(np.linalg.norm(w.data.qvel[a : a + 6]) for a in w._desk.dof_adr)
    assert speed < 0.1, f"seed {seed}: an item still moves at {speed:.2f}"


def test_items_carry_their_mass_at_their_centre(world):
    for it in world._desk.pool:
        b = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, f"desk_{it.name}")
        assert np.allclose(world.model.body_ipos[b], 0.0), it.name


def test_each_reset_is_a_new_desk(world):
    w = world
    looks_a = w.model.mat_rgba.copy()
    active_a = [p.active for p in w._desk.placements]
    w.reset()
    assert [p.active for p in w._desk.placements] != active_a
    assert not np.allclose(w.model.mat_rgba, looks_a)


def test_tape_roll_drops_over_the_peg():
    w = SimWorld("desk_sort", seed=0)
    k = next(i for i, it in enumerate(w._desk.pool) if it.kind == "tape_roll")
    a, d = w._desk.qpos_adr[k], w._desk.dof_adr[k]
    x, y = L.TAPE_PEG.xy
    w.data.qpos[a : a + 7] = [x, y, 0.09, 1, 0, 0, 0]
    w.data.qvel[d : d + 6] = 0.0
    w.step(600)
    pos = w.data.qpos[a : a + 3]
    assert math.hypot(pos[0] - x, pos[1] - y) < 0.006
    assert pos[2] < 0.025  # down on the peg's base, around the post


def test_camera_sees_the_desk(world):
    img = world.render(["record"], 160, 120)["record"]
    assert img.std() > 10


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_usd_desk_has_every_station_and_item(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    pkg = REPO / "lerobot_robot_kikobot"
    layout = _load("desk_layout", pkg / "desk" / "layout.py")
    _load("desk_textures", pkg / "desk" / "textures.py")
    _load("desk_meshes", pkg / "desk" / "meshes.py")
    assets = _load("desk_assets", REPO / "isaac" / "desk_assets.py")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    scene = assets.build_desk(stage, layout.default_pool(), tmp_path)
    for st in layout.STATIONS:
        for p in st.parts:
            name = f"{p.name}_visual" if p.visual else p.name
            assert stage.GetPrimAtPath(f"/World/Desk/Stations/{st.name}/{name}"), (st.name, name)
    assert len(scene.paths) == len(layout.default_pool())
    tape = next(n for n in scene.paths if n.startswith("tape_roll"))
    assert stage.GetPrimAtPath(f"{scene.paths[tape]}/collider_11")  # the ring
