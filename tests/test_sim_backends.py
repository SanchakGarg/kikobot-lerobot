"""Engines and renderers of the sim server. Newton / MuJoCo Warp tests run only where installed."""

from __future__ import annotations

import re

import numpy as np
import pytest

from lerobot_robot_kikobot.env.mujoco_scene import scene_mjcf
from lerobot_robot_kikobot.sim_backends import ENGINES, RENDERERS, resolve_engine, resolve_renderer
from lerobot_robot_kikobot.sim_server import SimServer, SimWorld


def test_engine_and_renderer_names():
    assert resolve_engine("mujoco") == "mujoco"
    assert resolve_engine("auto") in ENGINES
    assert resolve_renderer("opengl", "mujoco") == "opengl"
    assert resolve_renderer("auto", "mujoco") == "opengl"  # MJWarp only with Newton, on a GPU
    with pytest.raises(ValueError):
        resolve_engine("bullet")
    with pytest.raises(ValueError):
        resolve_renderer("vulkan", "mujoco")
    assert "mjwarp" in RENDERERS


def test_mjcf_for_newton_spells_out_the_servo_damping():
    """Importers skip `dampratio`; without the damping it implies, the servos ring."""
    xml = scene_mjcf("empty")
    j1 = re.search(r'<(?:general|position)[^>]*name="J1"[^>]*>', xml).group(0)
    kv = float(re.search(r'biasprm="([^"]+)"', j1).group(1).split()[2])
    assert kv < -1.0


def test_newton_window_needs_the_newton_engine():
    with pytest.raises(ValueError, match="engine newton"):
        SimServer(SimWorld("empty"), viewer="newton")


def _step_response(engine: str) -> list[float]:
    w = SimWorld("empty", seed=0, engine=engine)
    q0 = np.array([w.state()[f"J{i}.pos"] for i in range(1, 7)])
    w.command(np.deg2rad(q0 + [20, 0, 0, 0, 0, 0]), 100.0)
    out = []
    for _ in range(20):
        w.step(25)
        out.append(w.state()["J1.pos"] - q0[0])
    return out


def test_newton_moves_like_mujoco():
    pytest.importorskip("newton")
    a, b = np.array(_step_response("mujoco")), np.array(_step_response("newton"))
    assert np.abs(a - b).max() < 1.0, (a.round(1), b.round(1))
    assert abs(b[-1] - 20.0) < 0.5


def test_newton_teleports_and_respawns():
    pytest.importorskip("newton")
    w = SimWorld("pick_place", seed=3, engine="newton")
    w.pose(np.deg2rad([10, -20, 60, 30, -45, 5]), 100.0)
    w.step(10)
    assert np.allclose([w.state()[f"J{i}.pos"] for i in range(1, 7)], [10, -20, 60, 30, -45, 5], atol=0.5)
    w.reset()
    w.step(100)
    cube = w.data.qpos[w._object_adr : w._object_adr + 3]
    assert 0.15 < np.linalg.norm(cube[:2]) < 0.4 and abs(cube[2] - 0.016) < 0.003  # resting where it respawned
    mocap = w._engine.mocap
    assert mocap and np.allclose(w._engine.solver.mj_data.mocap_pos[mocap[0][1]], w.data.mocap_pos[mocap[0][0]])


def test_newton_runs_the_desk():
    pytest.importorskip("newton")
    w = SimWorld("desk_sort", seed=5, engine="newton")
    w.step(250)
    for p, a in zip(w._desk.placements, w._desk.qpos_adr):
        if p.active:
            assert -0.01 < w.data.qpos[a + 2] < 0.12, p.name


def test_mjwarp_renders_where_things_are():
    pytest.importorskip("mujoco_warp")
    w = SimWorld("pick_place", seed=0, renderer="mjwarp")
    a = w.render(["record"], 80, 60)["record"]
    w.pose(np.deg2rad([0, 0, 0, 0, 0, 0]), 100.0)  # The arm straight up.
    b = w.render(["record"], 80, 60)["record"]
    assert a.shape == (60, 80, 3) and a.std() > 5
    assert np.abs(a.astype(int) - b).mean() > 2
