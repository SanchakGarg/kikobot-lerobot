"""`kikobot_isaac` client against the real server loop, with a fake simulator.

Isaac Sim itself is not needed: the server's protocol handling, real-time loop and
units are tested here with `FakeBackend` in place of `IsaacNewtonBackend`.
"""

from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zmq")

from lerobot_robot_kikobot import events
from lerobot_robot_kikobot.config import ARM_JOINTS, KikobotIsaacConfig, KikobotSimConfig
from lerobot_robot_kikobot.isaac import KikobotIsaac
from lerobot_robot_kikobot.protocol import ProtocolError

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
        self.targets = (np.zeros(6), 0.0)
        self.respawns = []
        self.desks = []
        self.steps = 0
        self.configured = []

    def configure(self, physics_dt, width, height, cameras, scene):
        self.configured.append(scene)
        self.shape = (height, width)
        self.cameras = cameras

    def reset_arm(self, q, gripper_open):
        self.q, self.finger = q.copy(), gripper_open
        self.targets = (q.copy(), gripper_open)

    def respawn(self, obj_xy, pad_xy, yaw):
        self.respawns.append((obj_xy, pad_xy, yaw))

    desk_pool = srv.desk_layout.default_pool()

    def randomize_desk(self, placements):
        self.desks.append(placements)

    def set_targets(self, q, finger):
        self.targets = (q.copy(), finger)

    def physics_step(self):
        self.steps += 1
        q, f = self.targets
        self.q += 0.05 * (q - self.q)
        self.finger += 0.05 * (f - self.finger)

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


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def start_server():
    """start_server(scene=..., cameras=...) -> (backend, port), stopped after the test."""
    stops = []

    def start(**kw):
        kw = {"width": 64, "height": 48, "seed": 0, **kw}
        backend = FakeBackend()
        server = srv.KikobotServer(backend, srv.mjcf_limits(), **kw)
        port = _free_port()
        stop = threading.Event()
        th = threading.Thread(
            target=srv.serve,
            args=(server, f"tcp://127.0.0.1:{port}"),
            kwargs={"keep_running": lambda: not stop.is_set()},
            daemon=True,
        )
        th.start()
        stops.append((stop, th))
        return backend, port

    yield start
    for stop, th in stops:
        stop.set()
        th.join(timeout=2)


def _robot(port: int, **kw) -> KikobotIsaac:
    kw = {"width": 64, "height": 48, "timeout_s": 5, "connect_timeout_s": 5, **kw}
    return KikobotIsaac(KikobotIsaacConfig(port=port, **kw))


def test_features_match_mujoco_sim():
    from lerobot_robot_kikobot.sim import KikobotSim

    isaac = KikobotIsaac(KikobotIsaacConfig())
    sim = KikobotSim(KikobotSimConfig())
    assert isaac.action_features == sim.action_features
    assert isaac.observation_features == sim.observation_features


@pytest.mark.parametrize("encoding", ["raw", "jpeg"])
def test_connect_command_observe(start_server, encoding):
    backend, port = start_server()
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

        # Hold a target: physics, running in real time, takes the joint there.
        goal = {f"{j}.pos": obs[f"{j}.pos"] for j in ARM_JOINTS} | {"gripper.pos": 0.0}
        goal["J1.pos"] = 10.0
        for _ in range(60):
            sent = robot.send_action(goal)
            time.sleep(1 / 60)
        assert sent == pytest.approx(goal)
        time.sleep(0.3)
        obs = robot.get_observation()
        assert obs["J1.pos"] == pytest.approx(10.0, abs=0.1)
        assert obs["gripper.pos"] == pytest.approx(0.0, abs=0.5)
        assert backend.steps > 0
    finally:
        robot.disconnect()


def test_targets_are_clipped_to_the_joint_limits(start_server):
    _, port = start_server()
    robot = _robot(port)
    robot.connect()
    try:
        sent = robot.send_action({"J2.pos": 1e6})
        assert set(sent) == set(robot.action_features)
        assert robot._target[1] == pytest.approx(np.rad2deg(2.1816615649929156))
    finally:
        robot.disconnect()


def test_gamepad_back_button_resets_scene(start_server):
    backend, port = start_server()
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


def test_same_seed_same_spawns(start_server):
    spawns = []
    for _ in range(2):
        backend, port = start_server(seed=7)
        robot = _robot(port)
        robot.connect()
        robot.disconnect()
        spawns.append(backend.respawns[0])
    (o1, p1, y1), (o2, p2, y2) = spawns
    assert np.allclose(o1, o2) and np.allclose(p1, p2) and y1 == y2
    assert np.linalg.norm(o1 - p1) >= srv.MIN_OBJECT_PAD_GAP_M


def test_desk_sort_scene(start_server):
    backend, port = start_server(scene="desk_sort", cameras=("record", "overhead"))
    robot = _robot(port, camera_names=("record", "overhead"))
    robot.connect()
    obs = robot.get_observation()
    assert obs["overhead"].shape == (48, 64, 3)
    robot.disconnect()
    assert backend.respawns == []
    (desk,) = backend.desks
    assert len(desk) == len(backend.desk_pool)
    assert sum(p.active for p in desk) >= 4  # at least 3 writing tools + 1 supply
    # The settle phase ran physics before the episode started.
    assert backend.steps >= round(srv.SETTLE_S / srv.PHYSICS_DT_S)


def test_server_errors_reach_the_client(start_server):
    _, port = start_server()
    robot = _robot(port, camera_names=("overhead",))  # The server renders only `record`.
    robot.connect()
    try:
        with pytest.raises(ProtocolError, match="Unknown cameras"):
            robot.get_observation()
    finally:
        robot.disconnect()


def test_image_size_must_match_the_server(start_server):
    _, port = start_server()
    robot = _robot(port, width=640, height=480)
    robot.connect()
    try:
        with pytest.raises(ProtocolError, match="64x48"):
            robot.get_observation()
    finally:
        robot.disconnect()


def test_unknown_scene_is_refused():
    with pytest.raises(ValueError, match="Unknown scene"):
        srv.KikobotServer(FakeBackend(), srv.mjcf_limits(), scene="moon")


def test_no_server_times_out():
    robot = _robot(_free_port(), connect_timeout_s=0.3)
    with pytest.raises(TimeoutError, match="sim server"):
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
