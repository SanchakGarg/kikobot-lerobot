"""`kikobot_isaac` client against the real server loop, with a fake simulator.

Isaac Sim itself is not needed: the server's protocol handling, shaping and
units are tested here with `FakeBackend` in place of `IsaacNewtonBackend`.
"""

from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zmq")

from lerobot_robot_kikobot import events
from lerobot_robot_kikobot.config import ARM_JOINTS, KikobotIsaacConfig, KikobotSimConfig
from lerobot_robot_kikobot.isaac import KikobotIsaac
from lerobot_robot_kikobot.isaac_protocol import ProtocolError

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / "isaac" / "kikobot_isaac_server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("kikobot_isaac_server", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


srv = _load_server()


class FakeBackend(srv.Backend):
    """Joints that follow their targets with a first-order lag; a gradient image."""

    engine = "fake"

    def __init__(self):
        self.q = np.zeros(6)
        self.finger = 0.0
        self.t = 0.0
        self.targets = (np.zeros(6), 0.0)
        self.respawns = []
        self.configured = []

    def configure(self, fps, width, height, cameras, scene):
        self.configured.append(scene)
        self.dt = 1.0 / (fps * srv.SUBSTEPS)
        self.shape = (height, width)
        self.cameras = cameras

    def reset_arm(self, q, gripper_open):
        self.q, self.finger = q.copy(), gripper_open

    def respawn(self, obj_xy, pad_xy, yaw):
        self.respawns.append((obj_xy, pad_xy, yaw))

    def set_targets(self, q, finger):
        self.targets = (q.copy(), finger)

    def physics_step(self):
        q, f = self.targets
        self.q += 0.5 * (q - self.q)
        self.finger += 0.5 * (f - self.finger)
        self.t += self.dt

    def render(self):
        pass

    def joint_positions(self):
        return self.q.copy(), self.finger

    def images(self):
        h, w = self.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        img[..., 2] = 200
        return {c: img for c in self.cameras}

    def sim_time(self):
        return self.t


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    backend = FakeBackend()
    server = srv.KikobotServer(backend, srv.mjcf_limits())
    port = _free_port()
    stop = threading.Event()
    th = threading.Thread(
        target=srv.serve,
        args=(server, f"tcp://127.0.0.1:{port}"),
        kwargs={"keep_running": lambda: not stop.is_set()},
        daemon=True,
    )
    th.start()
    yield backend, port
    stop.set()
    th.join(timeout=2)


def _robot(port: int, **kw) -> KikobotIsaac:
    kw = {"width": 64, "height": 48, "seed": 0, "timeout_s": 5, "connect_timeout_s": 5, **kw}
    cfg = KikobotIsaacConfig(port=port, **kw)
    return KikobotIsaac(cfg)


def test_features_match_mujoco_sim():
    isaac = KikobotIsaac(KikobotIsaacConfig())
    from lerobot_robot_kikobot.sim import KikobotSim

    sim = KikobotSim(KikobotSimConfig())
    assert isaac.action_features == sim.action_features
    assert isaac.observation_features == sim.observation_features


@pytest.mark.parametrize("encoding", ["raw", "jpeg"])
def test_connect_step_observe(server, encoding):
    backend, port = server
    robot = _robot(port, image_encoding=encoding)
    robot.connect()
    try:
        assert backend.configured == ["pick_place"]
        assert len(backend.respawns) == 1  # reset_on_connect
        obs = robot.get_observation()
        assert set(obs) == set(robot.observation_features)
        assert obs["record"].shape == (48, 64, 3) and obs["record"].dtype == np.uint8
        if encoding == "raw":
            assert obs["record"][0, -1, 0] == 255
        assert obs["gripper.pos"] == pytest.approx(100.0)

        # Hold a target: the shaped joint gets there, in degrees.
        goal = {f"{j}.pos": obs[f"{j}.pos"] for j in ARM_JOINTS} | {"gripper.pos": 0.0}
        goal["J1.pos"] = 10.0
        for _ in range(90):  # 3 s at 30 fps
            sent = robot.send_action(goal)
        assert sent == pytest.approx(goal)
        obs = robot.get_observation()
        assert obs["J1.pos"] == pytest.approx(10.0, abs=0.1)
        assert obs["gripper.pos"] == pytest.approx(0.0, abs=0.5)
    finally:
        robot.disconnect()


def test_partial_action_and_limits(server):
    _, port = server
    robot = _robot(port)
    robot.connect()
    try:
        sent = robot.send_action({"J2.pos": 1e6})
        assert set(sent) == set(robot.action_features)
        assert sent["J2.pos"] == pytest.approx(np.rad2deg(2.1816615649929156))
    finally:
        robot.disconnect()


def test_gamepad_back_button_resets_scene(server):
    backend, port = server
    robot = _robot(port)
    robot.connect()
    try:
        events.request_scene_reset()
        robot.send_action({})
        assert len(backend.respawns) == 2
        robot.send_action({})
        assert len(backend.respawns) == 2  # one reset per press
    finally:
        robot.disconnect()


def test_same_seed_same_spawns(server):
    backend, port = server
    for _ in range(2):
        robot = _robot(port)
        robot.connect()
        robot.disconnect()
    (o1, p1, y1), (o2, p2, y2) = backend.respawns
    assert np.allclose(o1, o2) and np.allclose(p1, p2) and y1 == y2
    assert np.linalg.norm(o1 - p1) >= srv.MIN_OBJECT_PAD_GAP_M
    assert backend.configured == ["pick_place"]  # same geometry: not rebuilt


def test_server_errors_reach_the_client(server):
    _, port = server
    robot = _robot(port, camera_names=("wrist",))
    with pytest.raises(ProtocolError, match="Unknown cameras"):
        robot.connect()


def test_no_server_times_out():
    robot = _robot(_free_port(), connect_timeout_s=0.3)
    with pytest.raises(TimeoutError, match="kikobot_isaac_server"):
        robot.connect()


def test_server_loads_without_lerobot_or_isaac():
    """The server runs under Isaac Sim's Python, which has no LeRobot."""
    code = (
        "import sys, builtins\n"
        "real = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name.split('.')[0] in ('lerobot', 'lerobot_robot_kikobot', 'isaacsim', 'omni', 'mujoco'):\n"
        "        raise ImportError('blocked ' + name)\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        "import runpy\n"
        f"m = runpy.run_path({str(SERVER)!r}, run_name='not_main')\n"
        "print(sorted(m['mjcf_limits']()))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/")
    assert out.returncode == 0, out.stderr
    assert "J1" in out.stdout
