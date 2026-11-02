"""The simulated Kiko-Bot behind LeRobot's `Robot` interface.

The physics runs in `kikobot-sim-server`, in real time, in its own process;
this class sends it joint targets and reads back joint angles and camera
frames. Same keys and units as the real arm (`J1.pos`..`J6.pos` in degrees,
`gripper.pos` in % open), so a teleop, dataset or policy works on either.
"""

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from . import events
from .config import ARM_JOINTS, KikobotSimConfig
from .kinematics import ArmKinematics
from .motion import RateLimiter
from .protocol import IMAGE_ENCODINGS, unpack_observation
from .sim_client import SimClient

logger = logging.getLogger(__name__)


class KikobotSim(Robot):
    config_class = KikobotSimConfig
    name = "kikobot_sim"

    def __init__(self, config: KikobotSimConfig):
        super().__init__(config)
        self.config = config
        if config.image_encoding not in IMAGE_ENCODINGS:
            raise ValueError(f"--robot.image_encoding must be one of {IMAGE_ENCODINGS}")
        self.client: SimClient | None = None
        self.server_info: dict[str, Any] = {}
        # In degrees here: the limiter does not care about units.
        self._limiter = RateLimiter(config.max_velocity_deg_s)
        self._target = np.zeros(len(ARM_JOINTS))
        self._gripper_pct = 100.0
        self._ready_deg = np.rad2deg(ArmKinematics().ready_q())

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

    # -------------------------------------------------------------- connection

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        cfg = self.config
        client = SimClient(cfg.host, cfg.port, cfg.connect_timeout_s)
        try:
            self.server_info = client.hello()
            # A fresh server's first reset can be slow (shader compile): keep the
            # long timeout for it.
            self.client = client
            if cfg.reset_on_connect:
                self.reset_scene()
        except Exception:
            self.client = None
            client.close()
            raise
        client.set_timeout(cfg.timeout_s)
        self._lower = np.asarray(self.server_info["lower_deg"], dtype=float)
        self._upper = np.asarray(self.server_info["upper_deg"], dtype=float)
        logger.info(
            "Connected to the %s server at %s (scene %s).",
            self.server_info.get("engine", "sim"), client.address, self.server_info["scene"],
        )
        if not cfg.reset_on_connect:
            state = self._observe_state()
            self._hold(np.array([state[f"{j}.pos"] for j in ARM_JOINTS]), state["gripper.pos"])

    def _hold(self, q_deg: np.ndarray, gripper_pct: float) -> None:
        self._target = np.asarray(q_deg, dtype=float).copy()
        self._gripper_pct = float(gripper_pct)
        self._limiter.reset(self._target, time.perf_counter())

    def _observe_state(self) -> dict[str, float]:
        reply, _ = self.client.call({"cmd": "observe", "cameras": []})
        return reply["state"]

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        self.client.close()
        self.client = None
        self.server_info = {}

    # ----------------------------------------------------------------- episode

    def reset_scene(self) -> None:
        """Arm to the ready pose, jaws open, objects respawned. LeRobot has no
        episode hook; the gamepad's Back button triggers it."""
        # Sent from here: the Isaac server has no MuJoCo to solve it.
        reply, _ = self.client.call({"cmd": "reset", "ready_deg": self._ready_deg.tolist()})
        state = reply["state"]
        self._hold(np.array([state[f"{j}.pos"] for j in ARM_JOINTS]), 100.0)

    # --------------------------------------------------------------------- i/o

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        cfg = self.config
        reply, frames = self.client.call({
            "cmd": "observe",
            "cameras": list(cfg.camera_names),
            "width": cfg.width,
            "height": cfg.height,
            "encoding": cfg.image_encoding,
        })
        return unpack_observation(reply, frames)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if events.take_scene_reset():
            self.reset_scene()
        for i, j in enumerate(ARM_JOINTS):
            if f"{j}.pos" in action:
                self._target[i] = float(np.clip(action[f"{j}.pos"], self._lower[i], self._upper[i]))
        if "gripper.pos" in action:
            self._gripper_pct = float(np.clip(action["gripper.pos"], 0.0, 100.0))
        q = self._limiter.step(self._target, time.perf_counter())
        self.client.call({"cmd": "command", "q_deg": q.tolist(), "gripper_pct": self._gripper_pct})
        sent = {f"{j}.pos": float(q[i]) for i, j in enumerate(ARM_JOINTS)}
        sent["gripper.pos"] = self._gripper_pct
        return sent
