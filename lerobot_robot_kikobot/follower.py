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

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import MotorCalibration
from lerobot.robots import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .calibration import fit_range, is_range_calibrated
from .config import ARM_JOINTS, KikobotFollowerConfig
from .display import DISPLAY_MODES, SimDisplay
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
        if config.display not in DISPLAY_MODES:
            raise ValueError(f"display must be one of {DISPLAY_MODES}, not {config.display!r}.")
        unknown = set(config.inverted_joints) - set(self.motor_ids)
        if unknown:
            raise ValueError(f"inverted_joints {sorted(unknown)} not in {list(self.motor_ids)}.")
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
        self._display: SimDisplay | None = None

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

    def _steps_per_rad(self, i: int, motor: str, c: MotorCalibration) -> float:
        """Signed servo steps per joint radian."""
        sign = -1 if c.drive_mode else 1
        if is_range_calibrated(c.range_min, c.range_max):
            # The measured travel spans the model's range exactly; its direction
            # was measured too, so `inverted_joints` does not apply.
            return sign * (c.range_max - c.range_min) / (self.upper[i] - self.lower[i])
        if motor in self.config.inverted_joints:
            sign = -sign
        return sign / RAD_PER_STEP

    def _to_raw(self, q: np.ndarray) -> dict[int, int]:
        out = {}
        for i, (motor, angle) in enumerate(zip(self.motor_ids, q)):
            c = self._calib(motor)
            # Not clamped to the calibrated ends: an arm resting past one would
            # jump there at full speed. Targets are clipped in send_action, so
            # the shaper eases it back in instead.
            raw = c.homing_offset + angle * self._steps_per_rad(i, motor, c)
            out[self.motor_ids[motor]] = int(round(raw))
        return out

    def _from_raw(self, raw: dict[int, int]) -> np.ndarray:
        q = []
        for i, (motor, servo_id) in enumerate(self.motor_ids.items()):
            c = self._calib(motor)
            q.append((raw[servo_id] - c.homing_offset) / self._steps_per_rad(i, motor, c))
        return np.array(q)

    @property
    def is_calibrated(self) -> bool:
        # Uncalibrated still runs, on the mid-scale default above.
        return True

    def _end_names(self, i: int, motor: str) -> tuple[str, str]:
        if motor == "gripper":
            closed_is_min = self.config.gripper_closed_deg < self.config.gripper_open_deg
            return ("jaws closed", "jaws open") if closed_is_min else ("jaws open", "jaws closed")
        return (f"min, {np.rad2deg(self.lower[i]):+.0f} deg", f"max, {np.rad2deg(self.upper[i]):+.0f} deg")

    def _motors_to_calibrate(self) -> list[str]:
        motors = list(self.motor_ids)
        if not self.calibration:
            return motors
        missing = [m for m in motors if m not in self.calibration]
        print(f"Calibration file: {self.calibration_fpath}")
        if missing:
            print(f"Not calibrated yet: {' '.join(missing)}")
        while True:
            answer = input(
                "Press ENTER to keep it, type 'all' to redo every joint, or name the joints\n"
                f"to redo, e.g. 'J3' or 'J2 gripper' ({' '.join(motors)}): "
            ).replace(",", " ").split()
            if not answer:
                return []
            if [a.lower() for a in answer] in (["all"], ["c"]):
                return motors
            names = {m.lower(): m for m in motors}
            unknown = [a for a in answer if a.lower() not in names]
            if not unknown:
                return [names[a.lower()] for a in answer]
            print(f"Unknown joint(s) {unknown}.")

    def calibrate(self) -> None:
        """Map each joint's travel onto the model's range: min, middle, max by hand.

        Torque stays off the whole time, and afterwards: the servos never move.
        """
        motors = self._motors_to_calibrate()
        if not motors:
            return
        if self.streamer is not None:
            self.streamer.stop()
            self.streamer = None
        own_bus = self.bus is None
        bus = self.bus or FeetechBus(self.config.port, self.config.baudrate, list(self.motor_ids.values()))
        fits = {}
        try:
            bus.set_torque(False)
            # Readings are taken as actual positions; the servos keep their old
            # offset until the end, so an aborted run leaves the old calibration valid.
            old_offset = {m: bus.homing_offset(self.motor_ids[m]) for m in motors}
            print(
                "\nTorque is off and stays off: support the arm. For each joint, move it\n"
                "by hand to its min end, then to any point in between, then to its max\n"
                "end, pressing ENTER at each. Min and max are the MuJoCo model's joint\n"
                "limits. The middle point only tells which way the joint turned.\n"
            )
            for motor in motors:
                i = list(self.motor_ids).index(motor)
                servo_id = self.motor_ids[motor]
                lo_name, hi_name = self._end_names(i, motor)
                while True:
                    readings = []
                    for step in (lo_name, "middle, anywhere", hi_name):
                        input(f"  {motor}: move to {step}, then ENTER ")
                        readings.append(bus.read_positions()[servo_id] + old_offset[motor])
                    try:
                        fit = fit_range(*readings, self.lower[i], self.upper[i])
                    except ValueError as exc:
                        print(f"  {motor}: {exc}. Again.")
                        continue
                    ratio = fit.span_steps * RAD_PER_STEP / (self.upper[i] - self.lower[i])
                    print(f"  {motor}: {fit.span_steps} steps, {ratio:.2f}x the servo's nominal scale")
                    if abs(ratio - 1.0) > 0.15:
                        answer = input(
                            f"  warning: {motor}'s travel differs from the model's by "
                            f"{abs(ratio - 1) * 100:.0f}%. Type 'r' and ENTER to redo it, or ENTER to keep: "
                        )
                        if answer.strip().lower() == "r":
                            continue
                    fits[motor] = fit
                    break
            for motor, fit in fits.items():
                bus.set_homing_offset(self.motor_ids[motor], fit.servo_offset)
        finally:
            if own_bus:
                bus.close()
        # Joints not redone keep their entry.
        self.calibration = {
            **self.calibration,
            **{
                motor: MotorCalibration(
                    id=self.motor_ids[motor],
                    drive_mode=fit.drive_mode,
                    homing_offset=fit.zero,
                    range_min=fit.range_min,
                    range_max=fit.range_max,
                )
                for motor, fit in fits.items()
            },
        }
        self._save_calibration()
        print(f"Calibration of {' '.join(fits)} saved to {self.calibration_fpath}")
        print("Torque is still off. lerobot-teleoperate uses this calibration from now on.")

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

    def _start_stream(self) -> None:
        cfg = self.config
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

    @property
    def is_connected(self) -> bool:
        return self.bus is not None and all(c.is_connected for c in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        cfg = self.config
        self.bus = FeetechBus(cfg.port, cfg.baudrate, list(self.motor_ids.values()))
        try:
            self.bus.ping_all()
            if not calibrate:
                # lerobot-calibrate connects with calibrate=False and then calls
                # calibrate(): no torque, no stream, so the arm never moves.
                self.bus.set_torque(False)
                logger.info("%s connected on %s for calibration; torque off.", self, cfg.port)
                return
            if not self.calibration:
                logger.warning(
                    "No calibration at %s; using mid-scale as each joint's zero. Run "
                    "lerobot-calibrate --robot.type=kikobot_follower to match the arm to the sim.",
                    self.calibration_fpath,
                )
            else:
                missing = [m for m in self.motor_ids if m not in self.calibration]
                logger.info("Using calibration %s.", self.calibration_fpath)
                if missing:
                    logger.warning("Not calibrated, using mid-scale: %s", ", ".join(missing))
            self.configure()
            self._start_stream()
        except Exception:
            self.bus.close()
            self.bus = None
            raise

        for cam in self.cameras.values():
            cam.connect()
        if cfg.display != "none":
            self._display = SimDisplay(cfg.display, cfg.display_scene)
        logger.info(
            "%s connected on %s; streaming at %.0f Hz with a %s shaper at %.1f Hz.",
            self, cfg.port, cfg.stream_hz, cfg.motion.shaper_type, cfg.motion.shaper_freq_hz,
        )

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.streamer is not None:
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
        if self._display is not None:
            self._display.close()
            self._display = None
        logger.info("%s disconnected.", self)

    # --------------------------------------------------------------------- i/o

    def _gripper_pct(self, angle: float) -> float:
        open_, closed = np.deg2rad([self.config.gripper_open_deg, self.config.gripper_closed_deg])
        return float(100.0 * (angle - closed) / (open_ - closed))

    def _gripper_angle(self, pct: float) -> float:
        open_, closed = np.deg2rad([self.config.gripper_open_deg, self.config.gripper_closed_deg])
        return float(closed + np.clip(pct, 0.0, 100.0) / 100.0 * (open_ - closed))

    def _require_stream(self) -> None:
        if self.streamer is None:
            raise RuntimeError(f"{self} is connected for calibration only (torque off).")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        self._require_stream()
        q = self.streamer.latest_present()
        obs: dict[str, Any] = {f"{j}.pos": float(np.rad2deg(q[i])) for i, j in enumerate(ARM_JOINTS)}
        # No gripper servo: report the commanded opening so the key stays valid.
        obs["gripper.pos"] = (
            self._gripper_pct(q[6]) if self.config.gripper_id is not None else self._last_gripper_pct
        )
        for key, cam in self.cameras.items():
            obs[key] = cam.read_latest() if hasattr(cam, "read_latest") else cam.async_read()
        if self._display is not None:
            self._display.update(self.streamer.commanded, self._last_gripper_pct, q)
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        self._require_stream()
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
