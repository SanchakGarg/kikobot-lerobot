"""The Kiko-Bot in NVIDIA Isaac Sim (Newton physics) behind LeRobot's `Robot` interface.

Isaac Sim runs in its own process: `isaac/kikobot_isaac_server.py`, started
with Isaac Sim's Python. This class is the LeRobot side. It sends joint
targets and reads back joint angles and camera frames over ZMQ (see
`isaac_protocol.py`), so the server can sit on the same PC or on another one.

Same keys and units as `kikobot_sim` and the real arm, so datasets and
policies move between all three. The server runs the same `MotionShaper` per
physics step, with the `--robot.motion.*` settings sent from here.
"""

from __future__ import annotations

import dataclasses
import logging
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from . import events
from .config import ARM_JOINTS, KikobotIsaacConfig
from .isaac_protocol import (
    IMAGE_ENCODINGS,
    PROTOCOL_VERSION,
    ProtocolError,
    pack,
    unpack,
    unpack_observation,
)
from .kinematics import ArmKinematics

logger = logging.getLogger(__name__)


class KikobotIsaac(Robot):
    config_class = KikobotIsaacConfig
    name = "kikobot_isaac"

    def __init__(self, config: KikobotIsaacConfig):
        super().__init__(config)
        self.config = config
        if config.image_encoding not in IMAGE_ENCODINGS:
            raise ValueError(f"--robot.image_encoding must be one of {IMAGE_ENCODINGS}")
        self._ctx = None
        self._sock = None
        self._state: dict[str, float] = {}
        # Last commanded target, so a partial action keeps the other joints.
        self._target = {**{f"{j}.pos": 0.0 for j in ARM_JOINTS}, "gripper.pos": 100.0}
        self.server_info: dict[str, Any] = {}

    # ---------------------------------------------------------------- features

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {**{f"{j}.pos": float for j in ARM_JOINTS}, "gripper.pos": float}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        cams = {c: (self.config.height, self.config.width, 3) for c in self.config.camera_names}
        return {**self._motors_ft, **cams}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # -------------------------------------------------------------- transport

    @property
    def address(self) -> str:
        return f"tcp://{self.config.host}:{self.config.port}"

    def _open_socket(self, timeout_s: float) -> None:
        import zmq

        if self._ctx is None:
            self._ctx = zmq.Context()
        if self._sock is not None:
            self._sock.close(linger=0)
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
        sock.connect(self.address)
        self._sock = sock

    def _call(self, header: dict[str, Any]) -> tuple[dict[str, Any], list[bytes]]:
        import zmq

        try:
            self._sock.send_multipart(pack(header))
            reply, frames = unpack(self._sock.recv_multipart())
        except zmq.Again as e:
            # A REQ socket that missed a reply is stuck; reopen it so the next
            # call can go through once the server is back.
            self._open_socket(self.config.timeout_s)
            raise TimeoutError(
                f"No reply from the Isaac Sim server at {self.address} to {header['cmd']!r}. "
                "Is kikobot_isaac_server.py running (and the SSH tunnel up, if remote)?"
            ) from e
        if "error" in reply:
            raise ProtocolError(f"Isaac Sim server: {reply['error']}")
        return reply, frames

    # -------------------------------------------------------------- connection

    @property
    def is_connected(self) -> bool:
        return self._sock is not None and bool(self.server_info)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        cfg = self.config
        logger.info("Connecting to the Isaac Sim server at %s ...", self.address)
        self._open_socket(cfg.connect_timeout_s)
        info, _ = self._call(
            {
                "cmd": "hello",
                "protocol": PROTOCOL_VERSION,
                "fps": cfg.fps,
                "width": cfg.width,
                "height": cfg.height,
                "cameras": list(cfg.camera_names),
                "scene": cfg.scene,
                "seed": cfg.seed,
                "image_encoding": cfg.image_encoding,
                "motion": dataclasses.asdict(cfg.motion),
            }
        )
        if info.get("protocol") != PROTOCOL_VERSION:
            raise ProtocolError(
                f"Server speaks protocol {info.get('protocol')}, this plugin {PROTOCOL_VERSION}. "
                "Update both sides from the same commit (git pull)."
            )
        self.server_info = info
        self._lower = np.asarray(info["lower_deg"], dtype=float)
        self._upper = np.asarray(info["upper_deg"], dtype=float)
        logger.info("Isaac Sim server ready (%s).", info.get("engine", "?"))
        # The first reset compiles shaders on a fresh server; keep the long timeout.
        if cfg.reset_on_connect:
            self.reset_scene()
        self._open_socket(cfg.timeout_s)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self._call({"cmd": "close"})
        except (TimeoutError, ProtocolError) as e:
            logger.warning("Isaac Sim server did not acknowledge close: %s", e)
        self._sock.close(linger=0)
        self._sock = None
        self._ctx.term()
        self._ctx = None
        self.server_info = {}

    # ----------------------------------------------------------------- episode

    def reset_scene(self) -> None:
        """Arm to the ready pose, jaws open, cube and pad respawned. The same as
        `KikobotSim.reset_scene`; the gamepad's Back button triggers it too."""
        ready = np.rad2deg(ArmKinematics().ready_q())
        reply, _ = self._call({"cmd": "reset", "ready_deg": ready.tolist()})
        self._state = reply["state"]
        self._target = {**{f"{j}.pos": float(q) for j, q in zip(ARM_JOINTS, ready)}, "gripper.pos": 100.0}

    # --------------------------------------------------------------------- i/o

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        reply, frames = self._call({"cmd": "observe"})
        obs = unpack_observation(reply, frames)
        self._state = reply["state"]
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if events.take_scene_reset():
            self.reset_scene()
        for i, j in enumerate(ARM_JOINTS):
            key = f"{j}.pos"
            if key in action:
                self._target[key] = float(np.clip(action[key], self._lower[i], self._upper[i]))
        if "gripper.pos" in action:
            self._target["gripper.pos"] = float(np.clip(action["gripper.pos"], 0.0, 100.0))
        reply, _ = self._call({"cmd": "step", "action": self._target})
        self._state = reply["state"]
        return dict(self._target)
