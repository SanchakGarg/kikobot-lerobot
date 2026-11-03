"""The arm server and the servo-limits web node, on a fake register bus (no hardware, no LeRobot)."""

import json
import socket
import threading
import urllib.error
import urllib.request

import pytest

from lerobot_robot_kikobot.arm_server import ArmServer
from lerobot_robot_kikobot.protocol import ARM_JOINTS
from lerobot_robot_kikobot.servo_limits import BLOCKS, BY_KEY, REGISTERS, ServoLimits
from lerobot_robot_kikobot.servo_web import ArmServerLimits, make_server
from lerobot_robot_kikobot.sim_client import SimClient

IDS = {**{j: i for i, j in enumerate(ARM_JOINTS, 1)}, "gripper": 7}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeRegisterBus:
    """Each servo is 80 bytes of control table; `refuse` registers ignore writes."""

    def __init__(self, ids):
        self.mem = {i: bytearray(80) for i in ids}
        self.refuse: set[int] = set()
        self.dead: set[int] = set()
        for i in ids:
            self.poke(i, "max_torque", 1000)
            self.poke(i, "torque_limit", 1000)
            self.poke(i, "protection_current", 500)
            self.poke(i, "voltage", 121)
            self.mem[i][60:62] = (1 << 10 | 250).to_bytes(2, "little")  # Load -25.0 %.

    def poke(self, servo_id, key, value):
        reg = BY_KEY[key]
        self.mem[servo_id][reg.address:reg.address + reg.size] = value.to_bytes(reg.size, "little")

    def read_block(self, servo_id, address, length):
        if servo_id in self.dead:
            raise ConnectionError(f"read servo {servo_id}: There is no status packet!")
        return list(self.mem[servo_id][address:address + length])

    def write_register(self, servo_id, address, size, value, eeprom):
        if address not in self.refuse:
            self.mem[servo_id][address:address + size] = value.to_bytes(size, "little")


class FakeArm:
    """The surface of `KikobotFollower` the arm server uses."""

    def __init__(self):
        self.motor_ids = dict(IDS)
        self.bus = FakeRegisterBus(IDS.values())
        self.cameras = {}
        self.actions = []
        self.q = {f"{j}.pos": 0.0 for j in ARM_JOINTS} | {"gripper.pos": 100.0}

    def send_action(self, action):
        self.actions.append(action)
        self.q.update(action)
        return action

    def read_state(self):
        return dict(self.q)

    def joint_limits_deg(self):
        return [-180.0] * 6, [180.0] * 6


def test_two_reads_cover_every_register():
    for reg in REGISTERS:
        assert any(s <= reg.address and reg.address + reg.size <= s + n for s, n in BLOCKS), reg.key


def test_limits_decode_units_and_signs():
    bus = FakeRegisterBus(IDS.values())
    values = ServoLimits(bus, IDS).read()["servos"][0]["values"]
    assert values["max_torque"] == 1000 and values["protection_current"] == 500
    assert values["load"] == -250  # Sign in bit 10.
    assert values["voltage"] * BY_KEY["voltage"].scale == pytest.approx(12.1)


def test_limit_writes_are_checked_and_read_back():
    bus = FakeRegisterBus(IDS.values())
    limits = ServoLimits(bus, IDS)
    assert limits.write("J2", "torque_limit", 600) == {"id": 2, "key": "torque_limit", "value": 600}
    assert limits.write(3, "protection_current", 300)["value"] == 300
    with pytest.raises(ValueError, match="0..1000"):
        limits.write("J2", "torque_limit", 1001)
    with pytest.raises(ValueError, match="not a settable"):
        limits.write("J2", "min_angle", 10)
    with pytest.raises(ValueError, match="No servo"):
        limits.write("J9", "torque_limit", 10)
    bus.refuse.add(BY_KEY["max_torque"].address)
    with pytest.raises(RuntimeError, match="refused"):
        limits.write("J1", "max_torque", 500)


def test_a_dead_servo_does_not_hide_the_others():
    bus = FakeRegisterBus(IDS.values())
    bus.dead.add(4)
    servos = ServoLimits(bus, IDS).read()["servos"]
    assert "status packet" in servos[3]["error"]
    assert all("values" in s for i, s in enumerate(servos) if i != 3)


@pytest.fixture
def arm_server():
    arm = FakeArm()
    server = ArmServer(arm, "127.0.0.1", free_port())
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()
    assert server.ready.wait(5)
    yield arm, server
    server.stop_event.set()
    t.join(5)


def test_teleop_drives_the_arm_through_the_server(arm_server):
    arm, server = arm_server
    client = SimClient(server.host, server.port, 2.0)
    info = client.hello()
    assert info["joints"] == list(ARM_JOINTS) and info["engine"] == "arm"
    reply, _ = client.call({"cmd": "command", "q_deg": [10, -20, 30, 0, 0, 0], "gripper_pct": 40})
    assert arm.actions[-1]["J2.pos"] == -20 and arm.actions[-1]["gripper.pos"] == 40
    assert reply["state"]["J3.pos"] == 30
    reply, _ = client.call({"cmd": "observe", "cameras": []})
    assert reply["state"]["J1.pos"] == 10
    n = len(arm.actions)
    client.call({"cmd": "reset", "ready_deg": [0] * 6})
    assert len(arm.actions) == n  # The real arm does not jump to ready on a scene reset.
    with pytest.raises(Exception, match="No camera"):
        client.call({"cmd": "observe", "cameras": ["record"]})
    client.close()


def test_web_page_reads_and_sets_limits_through_the_arm_server(arm_server):
    arm, server = arm_server
    backend = ArmServerLimits(server.host, server.port)
    web = make_server(backend, "127.0.0.1", 0)
    threading.Thread(target=web.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{web.server_address[1]}"
    try:
        assert b"servo limits" in urllib.request.urlopen(base + "/").read()
        body = json.load(urllib.request.urlopen(base + "/api/limits"))
        assert [s["name"] for s in body["servos"]] == list(IDS)
        assert body["values"]["servos"][1]["values"]["torque_limit"] == 1000

        def post(payload):
            req = urllib.request.Request(base + "/api/limits", json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req))

        assert post({"servo": "J2", "key": "torque_limit", "value": 700})["value"] == 700
        assert ServoLimits(arm.bus, IDS).read_servo(2)["torque_limit"] == 700
        with pytest.raises(urllib.error.HTTPError) as err:
            post({"servo": "J2", "key": "torque_limit", "value": 5000})
        assert err.value.code == 400 and "0..1000" in json.load(err.value)["error"]
    finally:
        web.shutdown()
        web.server_close()
        backend.close()
