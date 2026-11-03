"""`kikobot-arm-server`: the physical arm behind a socket, like `kikobot-sim-server`.

It opens the servo bus, holds the arm where it is, and serves the sim
protocol (`protocol.py`) on port 5558. A teleop or policy then drives the real
arm as `--robot.type=kikobot_arm`, the same way it drives a sim, and other
nodes can connect at the same time: `kikobot-servo-web` reads and sets the
servos' current and torque limits through it while the arm is being driven.

    kikobot-arm-server --robot.port=/dev/ttyACM0 --robot.id=kikobot
    lerobot-teleoperate --robot.type=kikobot_arm --teleop.type=kikobot_gamepad --fps=60
    kikobot-servo-web                     # http://127.0.0.1:8080

Every `--robot.<name>=` option of `kikobot_follower` works here (port, id,
servo IDs, speed cap, sag correction, sim mirror...). `--host 0.0.0.0` lets
nodes on other machines connect.

One thread owns the bus and answers one request at a time, so a limits
request never interleaves with a goal on the wire. Between requests the
servos hold the last goal: a teleop that stops or drops leaves the arm where
it was.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from .protocol import (
    ARM_DEFAULT_PORT,
    ARM_JOINTS,
    IMAGE_ENCODINGS,
    PROTOCOL_VERSION,
    pack,
    pack_observation,
    unpack,
)
from .servo_limits import ServoLimits

logger = logging.getLogger(__name__)

POLL_MS = 100


class ArmServer:
    """Serves a connected `KikobotFollower` (or anything with its surface)."""

    def __init__(self, robot: Any, host: str = "127.0.0.1", port: int = ARM_DEFAULT_PORT):
        self.robot = robot
        self.host = host
        self.port = port
        self.limits = ServoLimits(robot.bus, robot.motor_ids)
        self.stop_event = threading.Event()
        self.ready = threading.Event()

    # -------------------------------------------------------------- handlers

    def _move(self, header: dict[str, Any]) -> list[bytes]:
        q = np.asarray(header["q_deg"], dtype=float)
        if q.shape != (len(ARM_JOINTS),):
            raise ValueError(f"q_deg needs {len(ARM_JOINTS)} angles.")
        action = {f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, q)}
        action["gripper.pos"] = float(header.get("gripper_pct", 100.0))
        self.robot.send_action(action)
        return pack({"state": self.robot.read_state()})

    def _observe(self, header: dict[str, Any]) -> list[bytes]:
        encoding = header.get("encoding", "raw")
        if encoding not in IMAGE_ENCODINGS:
            raise ValueError(f"encoding must be one of {IMAGE_ENCODINGS}.")
        state = self.robot.read_state()
        images = {}
        for name in header.get("cameras", []):
            if name not in self.robot.cameras:
                raise ValueError(f"No camera {name!r} on the arm; it has {sorted(self.robot.cameras)}.")
            img = self.robot.read_camera(name)
            w, h = header.get("width"), header.get("height")
            if w and h and img.shape[:2] != (h, w):
                import cv2

                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            images[name] = img
        return pack_observation(state, images, encoding)

    def handle(self, header: dict[str, Any]) -> list[bytes]:
        cmd = header.get("cmd")
        if cmd == "hello":
            lower, upper = self.robot.joint_limits_deg()
            return pack({
                "protocol": PROTOCOL_VERSION,
                "joints": list(ARM_JOINTS),
                "lower_deg": lower,
                "upper_deg": upper,
                "scene": "real arm",
                "engine": "arm",
                "cameras": sorted(self.robot.cameras),
                "servos": self.robot.motor_ids,
            })
        if cmd in ("command", "pose"):
            # The real arm cannot teleport: a pose is a goal, under the speed cap.
            return self._move(header)
        if cmd == "observe":
            return self._observe(header)
        if cmd == "reset":
            # No scene to respawn; the arm stays where it is.
            return pack({"state": self.robot.read_state()})
        if cmd == "limits":
            return pack({**self.limits.spec(), "values": self.limits.read()})
        if cmd == "limits_write":
            result = self.limits.write(header["servo"], header["key"], header["value"])
            logger.info("Set %s of servo %d to %d (raw).", result["key"], result["id"], result["value"])
            return pack(result)
        if cmd == "close":
            return pack({})
        raise ValueError(f"Unknown command {cmd!r}.")

    # ------------------------------------------------------------------ loop

    def serve(self) -> None:
        import zmq

        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.bind(f"tcp://{self.host}:{self.port}")
            logger.info("Arm server on tcp://%s:%d.", self.host, self.port)
            self.ready.set()
            while not self.stop_event.is_set():
                if not sock.poll(POLL_MS):
                    continue
                try:
                    header, _ = unpack(sock.recv_multipart())
                    reply = self.handle(header)
                except Exception as exc:  # Report it to the client; keep serving.
                    logger.warning("Request failed: %s", exc)
                    reply = pack({"error": str(exc)})
                sock.send_multipart(reply)
        finally:
            sock.close(linger=0)
            ctx.term()


def main(argv: list[str] | None = None) -> None:
    import draccus

    from .config import ArmServerConfig
    from .follower import KikobotFollower

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = draccus.parse(ArmServerConfig, args=argv)
    robot = KikobotFollower(cfg.robot)
    robot.connect()
    try:
        ArmServer(robot, cfg.host, cfg.port).serve()
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
