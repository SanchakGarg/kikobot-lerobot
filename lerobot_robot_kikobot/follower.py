"""The physical Kiko-Bot arm behind LeRobot's `Robot` interface.

Same action and observation keys as the sim (`J1.pos`..`J6.pos` in degrees,
`gripper.pos` in % open), so a teleop, dataset or policy works on either.
`send_action` only hands the target to `ServoStreamer`, which streams shaped
goals to the servos at `stream_hz` from its own thread.
"""

from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import MotorCalibration
from lerobot.robots import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config import ARM_JOINTS, KikobotFollowerConfig
from .feetech import (
    ADDR_ACCELERATION,
    ADDR_D_COEFFICIENT,
    ADDR_GOAL_SPEED,
    ADDR_I_COEFFICIENT,
    ADDR_P_COEFFICIENT,
    ADDR_RETURN_DELAY,
    RESOLUTION,
    FeetechBus,
)
from .kinematics import ArmKinematics
from .motion import MotionShaper
from .streamer import ServoStreamer

logger = logging.getLogger(__name__)

RAD_PER_STEP = 2.0 * math.pi / RESOLUTION
# Uncalibrated default: the servo's mid-scale reading is the joint's zero,
# which is how the arm was assembled.
DEFAULT_HOMING = RESOLUTION // 2


class KikobotFollower(Robot):
    config_class = KikobotFollowerConfig
    name = "kikobot_follower"

    def __init__(self, config: KikobotFollowerConfig):
        super().__init__(config)
        self.config = config
        if len(config.servo_ids) != len(ARM_JOINTS):
            raise ValueError(f"servo_ids needs one ID per joint {ARM_JOINTS}.")
        self.motor_ids = dict(zip(ARM_JOINTS, config.servo_ids))
        if config.gripper_id is not None:
            self.motor_ids["gripper"] = config.gripper_id
        self.kin = ArmKinematics()
        self.lower = self.kin.lower.copy()
        self.upper = self.kin.upper.copy()
        if config.gripper_id is not None:
            ends = np.deg2rad([config.gripper_open_deg, config.gripper_closed_deg])
            self.lower = np.append(self.lower, ends.min())
            self.upper = np.append(self.upper, ends.max())
        self.bus: FeetechBus | None = None
        self.streamer: ServoStreamer | None = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._target: np.ndarray | None = None
        self._last_gripper_pct = 100.0
        self._viewer = None

    # ---------------------------------------------------------------- features

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {**{f"{j}.pos": float for j in ARM_JOINTS}, "gripper.pos": float}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {k: (c.height, c.width, 3) for k, c in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ------------------------------------------------------------- calibration

    def _calib(self, motor: str) -> MotorCalibration:
        return self.calibration.get(motor) or MotorCalibration(
            id=self.motor_ids[motor], drive_mode=0, homing_offset=DEFAULT_HOMING,
            range_min=0, range_max=RESOLUTION - 1,
        )

    def _to_raw(self, q: np.ndarray) -> dict[int, int]:
        out = {}
        for motor, angle in zip(self.motor_ids, q):
            c = self._calib(motor)
            sign = -1 if c.drive_mode else 1
            raw = c.homing_offset + sign * angle / RAD_PER_STEP
            out[self.motor_ids[motor]] = int(round(min(max(raw, c.range_min), c.range_max)))
        return out

    def _from_raw(self, raw: dict[int, int]) -> np.ndarray:
        q = []
        for motor, servo_id in self.motor_ids.items():
            c = self._calib(motor)
            sign = -1 if c.drive_mode else 1
            q.append(sign * (raw[servo_id] - c.homing_offset) * RAD_PER_STEP)
        return np.array(q)

    @property
    def is_calibrated(self) -> bool:
        # Uncalibrated still runs, on the mid-scale default above.
        return True

    def calibrate(self) -> None:
        """Record each servo's reading at the model's zero pose."""
        if self.calibration:
            answer = input(
                f"Press ENTER to keep the calibration in {self.calibration_fpath}, "
                "or type 'c' and ENTER to recalibrate: "
            )
            if answer.strip().lower() != "c":
                return
        own_bus = self.bus is None
        bus = self.bus or FeetechBus(self.config.port, self.config.baudrate, list(self.motor_ids.values()))
        try:
            bus.set_torque(False)
            print(
                "\nTorque is off. Move the arm by hand to its zero pose: every joint at\n"
                "mid-travel, as in the MuJoCo model with all joints at 0 (open\n"
                "models/scene.xml in the MuJoCo viewer to see it). Jaws fully open.\n"
            )
            input("Press ENTER when the arm is in the zero pose...")
            raw = bus.sync_read_positions()
        finally:
            if own_bus:
                bus.close()
        self.calibration = {
            motor: MotorCalibration(
                id=servo_id,
                # Keep a direction flip set by hand in the file.
                drive_mode=self.calibration[motor].drive_mode if motor in self.calibration else 0,
                homing_offset=int(raw[servo_id]),
                range_min=0,
                range_max=RESOLUTION - 1,
            )
            for motor, servo_id in self.motor_ids.items()
        }
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")
        print("If a joint turns the wrong way, set its drive_mode to 1 in that file.")

    def configure(self) -> None:
        bus = self.bus
        cfg = self.config
        bus.set_torque(False)
        for servo_id in self.motor_ids.values():
            bus.set_eeprom_byte(servo_id, ADDR_RETURN_DELAY, 0)
            bus.set_eeprom_byte(servo_id, ADDR_P_COEFFICIENT, cfg.p_coefficient)
            bus.set_eeprom_byte(servo_id, ADDR_I_COEFFICIENT, cfg.i_coefficient)
            bus.set_eeprom_byte(servo_id, ADDR_D_COEFFICIENT, cfg.d_coefficient)
            # Let the stream set the pace: the servo's own profile at full
            # acceleration and speed, so it adds no lag of its own.
            bus.write_byte(servo_id, ADDR_ACCELERATION, 254)
            bus.write_word(servo_id, ADDR_GOAL_SPEED, 0)

    # -------------------------------------------------------------- connection

    @property
    def is_connected(self) -> bool:
        return self.streamer is not None and all(c.is_connected for c in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        cfg = self.config
        self.bus = FeetechBus(cfg.port, cfg.baudrate, list(self.motor_ids.values()))
        try:
            self.bus.ping_all()
            if not self.calibration:
                if calibrate:
                    logger.warning(
                        "No calibration at %s; using mid-scale as each joint's zero. "
                        "Run lerobot-calibrate --robot.type=kikobot_follower to set it.",
                        self.calibration_fpath,
                    )
            self.configure()
            # Goal = where the arm is before torque comes on, so nothing jumps.
            present_raw = self.bus.sync_read_positions()
            self.bus.sync_write_goals(present_raw)
            self.bus.set_torque(True)
            present = self._from_raw(present_raw)
            self.streamer = ServoStreamer(
                self.bus,
                MotionShaper(len(self.motor_ids), cfg.motion),
                self._to_raw,
                self._from_raw,
                cfg.stream_hz,
                cfg.read_every,
            )
            self.streamer.start(present)
            self._target = present.copy()
        except Exception:
            self.bus.close()
            self.bus = None
            raise

        for cam in self.cameras.values():
            cam.connect()
        if cfg.viewer:
            self._open_viewer()
        logger.info(
            "%s connected on %s; streaming at %.0f Hz with a %s shaper at %.1f Hz.",
            self, cfg.port, cfg.stream_hz, cfg.motion.shaper_type, cfg.motion.shaper_freq_hz,
        )

    def _open_viewer(self) -> None:
        self._viewer = mujoco.viewer.launch_passive(self.kin.model, self.kin.data)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.streamer.stop()
        if self.streamer.late_ticks:
            logger.info("Servo stream ran late on %d ticks.", self.streamer.late_ticks)
        self.streamer = None
        try:
            if self.config.disable_torque_on_disconnect:
                self.bus.set_torque(False)
        finally:
            self.bus.close()
            self.bus = None
        for cam in self.cameras.values():
            cam.disconnect()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        logger.info("%s disconnected.", self)

    # --------------------------------------------------------------------- i/o

    def _gripper_pct(self, angle: float) -> float:
        open_, closed = np.deg2rad([self.config.gripper_open_deg, self.config.gripper_closed_deg])
        return float(100.0 * (angle - closed) / (open_ - closed))

    def _gripper_angle(self, pct: float) -> float:
        open_, closed = np.deg2rad([self.config.gripper_open_deg, self.config.gripper_closed_deg])
        return float(closed + np.clip(pct, 0.0, 100.0) / 100.0 * (open_ - closed))

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        q = self.streamer.latest_present()
        obs: dict[str, Any] = {f"{j}.pos": float(np.rad2deg(q[i])) for i, j in enumerate(ARM_JOINTS)}
        # No gripper servo: report the commanded opening so the key stays valid.
        obs["gripper.pos"] = (
            self._gripper_pct(q[6]) if self.config.gripper_id is not None else self._last_gripper_pct
        )
        for key, cam in self.cameras.items():
            obs[key] = cam.read_latest() if hasattr(cam, "read_latest") else cam.async_read()
        if self._viewer is not None and self._viewer.is_running():
            self.kin.data.qpos[self.kin.qpos_adr] = q[:6]
            mujoco.mj_forward(self.kin.model, self.kin.data)
            self._viewer.sync()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        target = self._target.copy()
        for i, j in enumerate(ARM_JOINTS):
            if f"{j}.pos" in action:
                target[i] = math.radians(action[f"{j}.pos"])
        if "gripper.pos" in action:
            self._last_gripper_pct = float(np.clip(action["gripper.pos"], 0.0, 100.0))
            if self.config.gripper_id is not None:
                target[6] = self._gripper_angle(self._last_gripper_pct)
        target = np.clip(target, self.lower, self.upper)
        self._target = target
        self.streamer.set_target(target)

        sent = {f"{j}.pos": float(np.rad2deg(target[i])) for i, j in enumerate(ARM_JOINTS)}
        sent["gripper.pos"] = self._last_gripper_pct
        return sent
