"""The MuJoCo sim server, driven over its socket like LeRobot drives it."""

import socket
import threading
import time

import numpy as np
import pytest

from lerobot_robot_kikobot.protocol import unpack_observation
from lerobot_robot_kikobot.sim_client import SimClient
from lerobot_robot_kikobot.sim_server import SimServer, SimWorld


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def client():
    port = free_port()
    server = SimServer(SimWorld("pick_place", seed=0), port=port, viewer=False)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    c = SimClient("127.0.0.1", port, timeout_s=10.0)
    yield c
    c.close()
    server.stop_event.set()
    thread.join(timeout=5)


def joints(state):
    return np.array([state[f"J{i}.pos"] for i in range(1, 7)])


def test_hello(client):
    info = client.hello()
    assert info["joints"] == ["J1", "J2", "J3", "J4", "J5", "J6"]
    assert len(info["lower_deg"]) == 6 and len(info["ready_deg"]) == 6


def test_command_moves_the_arm_in_real_time(client):
    reply, _ = client.call({"cmd": "reset", "ready_deg": None})
    start = joints(reply["state"])
    goal = start + np.array([20.0, 0, 0, 0, 0, 0])
    client.call({"cmd": "command", "q_deg": goal.tolist(), "gripper_pct": 100.0})
    time.sleep(0.6)
    reply, _ = client.call({"cmd": "observe", "cameras": []})
    assert abs(joints(reply["state"])[0] - goal[0]) < 2.0


def test_pose_teleports(client):
    q = [10.0, -20.0, 60.0, 30.0, -45.0, 5.0]
    reply, _ = client.call({"cmd": "pose", "q_deg": q, "gripper_pct": 50.0})
    assert np.allclose(joints(reply["state"]), q, atol=1e-6)


def test_observe_renders_the_camera(client):
    reply, frames = client.call({"cmd": "observe", "cameras": ["record"], "width": 320, "height": 240})
    obs = unpack_observation(reply, frames)
    assert obs["record"].shape == (240, 320, 3)
    assert obs["record"].std() > 1.0  # Not a blank frame.


def test_bad_request_reports_an_error(client):
    from lerobot_robot_kikobot.protocol import ProtocolError

    with pytest.raises(ProtocolError):
        client.call({"cmd": "command", "q_deg": [1.0, 2.0]})
