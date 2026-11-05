"""The real arm, on a fake bus: each goal must reach the servos on the tick it is sent."""

import threading
import time

import numpy as np
import pytest

from lerobot_robot_kikobot import follower as follower_mod
from lerobot_robot_kikobot.config import KikobotFollowerConfig
from lerobot_robot_kikobot.follower import KikobotFollower
from lerobot_robot_kikobot.paths import ARM_SCENE_PATH
from lerobot_robot_kikobot.sim_server import SimServer, SimWorld

from test_sim_server import free_port


class FakeBus:
    """Ideal servos: each goal is where the servo is by the next read."""

    def __init__(self, port, baudrate, ids):
        self.ids = list(ids)
        self.positions = {i: 2048 for i in self.ids}
        self.writes: list[dict[int, int]] = []
        self.torque = False

    def ping_all(self): pass
    def close(self): pass
    def set_eeprom_byte(self, *a): pass
    def write_byte(self, *a): pass
    def write_word(self, *a): pass
    def set_torque(self, on): self.torque = on
    def sync_read_positions(self): return dict(self.positions)

    def sync_write_goals(self, goals):
        self.writes.append(dict(goals))
        self.positions.update(goals)


@pytest.fixture
def arm(monkeypatch, tmp_path):
    monkeypatch.setattr(follower_mod, "FeetechBus", FakeBus)

    def make(**kw):
        cfg = KikobotFollowerConfig(port="fake", id="test", calibration_dir=tmp_path, sag_joints=(), **kw)
        robot = KikobotFollower(cfg)
        robot.connect()
        return robot

    return make


def angles(robot):
    obs = robot.get_observation()
    return np.array([obs[f"J{i}.pos"] for i in range(1, 7)])


def test_goal_is_written_on_the_same_tick(arm):
    robot = arm(sim_backend="none")
    q0 = angles(robot)
    robot.send_action({f"J{i}.pos": float(q) for i, q in enumerate(q0, 1)})  # Ends the soft start.
    target = q0 + np.array([1.0, -1.0, 0.5, 0, 0, 0])
    time.sleep(1 / 60)  # One teleop tick: 3 deg at the 180 deg/s cap, more than this step.
    n_writes = len(robot.bus.writes)
    sent = robot.send_action({f"J{i}.pos": float(q) for i, q in enumerate(target, 1)})
    assert len(robot.bus.writes) == n_writes + 1
    assert np.allclose([sent[f"J{i}.pos"] for i in range(1, 7)], target)
    assert np.allclose(angles(robot), target, atol=0.1)  # Servo steps are 0.088 deg.
    robot.disconnect()


def test_first_move_is_slow(arm):
    robot = arm(sim_backend="none", soft_start_deg_s=30.0)
    q0 = angles(robot)
    time.sleep(0.05)
    sent = robot.send_action({"J1.pos": float(q0[0]) + 40.0})
    assert 0.0 < sent["J1.pos"] - q0[0] <= 30.0 * 0.1 + 1e-6
    robot.disconnect()


def test_commands_are_mirrored_to_the_sim(arm):
    port = free_port()
    # The arm alone: in the room it starts upright, and turned that way its
    # shoulder meets the hutch's side panel beside its base.
    server = SimServer(SimWorld(ARM_SCENE_PATH), port=port, viewer=False)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        robot = arm(sim_backend="mujoco", sim_port=port, sim_timeout_s=5.0)
        assert robot._sim is not None
        q0 = angles(robot)
        target = q0.copy()
        target[0] += 15.0
        robot.send_action({f"J{i}.pos": float(q) for i, q in enumerate(q0, 1)})
        for _ in range(60):
            robot.send_action({f"J{i}.pos": float(q) for i, q in enumerate(target, 1)})
            time.sleep(1 / 60)
        time.sleep(0.4)
        sim_j1 = server.world.state()["J1.pos"]
        assert abs(sim_j1 - target[0]) < 2.0
        robot.disconnect()
    finally:
        server.stop_event.set()
        thread.join(timeout=5)


def test_no_sim_server_is_not_fatal(arm):
    robot = arm(sim_backend="mujoco", sim_port=free_port(), sim_timeout_s=0.2)
    assert robot._sim is None
    robot.send_action({"J1.pos": 0.0})
    robot.disconnect()


def test_a_dead_sim_server_never_holds_up_the_real_arm(arm):
    port = free_port()
    server = SimServer(SimWorld("empty"), port=port, viewer=False)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    robot = arm(sim_backend="mujoco", sim_port=port, sim_timeout_s=0.5)
    assert robot._sim is not None
    server.stop_event.set()  # The sim server goes away mid-run.
    thread.join(timeout=5)
    q0 = angles(robot)
    start = time.perf_counter()
    for k in range(30):
        robot.send_action({"J1.pos": float(q0[0]) + 0.01 * k})
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1, f"30 actions took {elapsed:.3f} s"
    time.sleep(0.7)
    assert robot._sim.error is not None  # Noticed, in the background.
    robot.disconnect()


def test_default_port_finds_pi_gpio_uart(monkeypatch):
    """A Driver HAT on a Pi's GPIO UART is used when no USB adapter is plugged in."""
    from lerobot_robot_kikobot import config

    monkeypatch.setattr(config.os, "name", "posix")
    monkeypatch.setattr(config.os.path, "exists", lambda p: p == "/dev/serial0")
    assert config.default_port() == "/dev/serial0"
    monkeypatch.setattr(config.os.path, "exists", lambda p: p in ("/dev/ttyACM0", "/dev/serial0"))
    assert config.default_port() == "/dev/ttyACM0"
    monkeypatch.setattr(config.os.path, "exists", lambda p: False)
    assert config.default_port() == "/dev/ttyACM0"
