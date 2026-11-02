"""The physical Kiko-Bot arm behind LeRobot's `Robot` interface.

Same action and observation keys as the sim (`J1.pos`..`J6.pos` in degrees,
`gripper.pos` in % open), so a teleop, dataset or policy works on either.
`send_action` only hands the target to `ServoStreamer`, which streams shaped
goals to the servos at `stream_hz` from its own thread.
"""

from __future__ import annotations

import json
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
from .paths import home_pose_path
from .motion import MotionShaper
from .streamer import ServoStreamer

logger = logging.getLogger(__name__)

RAD_PER_STEP = 2.0 * math.pi / RESOLUTION
# Uncalibrated default: the servo's mid-scale reading is the joint's zero,
# which is how the arm was assembled.
DEFAULT_HOMING = RESOLUTION // 2

# Which physical way each joint's min and max lie, read off the MuJoCo model.
# "Front" is the way the arm reaches in the ready pose; stand behind the base
# facing it. Getting these the wrong way round flips the joint, and the IK's
# moves then come out in the wrong directions on the real arm.
JOINT_ENDS = {
    "J1": ("base turned fully counter-clockwise, seen from above", "base turned fully clockwise, seen from above"),
    "J2": ("upper arm tilted fully forward, toward the front", "upper arm tilted fully back"),
    "J3": ("forearm bent fully up and back, the opposite way to the ready pose",
           "forearm bent fully forward and down, the way the ready pose bends it"),
    "J4": ("wrist tilted fully so the jaws swing back, toward the base",
           "wrist tilted fully so the jaws swing forward and up"),
    "J5": ("wrist swung fully to the left", "wrist swung fully to the right"),
    "J6": ("jaws turned fully clockwise, seen from above with the jaws pointing down",
           "jaws turned fully counter-clockwise, seen from above with the jaws pointing down"),
}


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
        # The bus during calibrate(), reopened there if it drops.
        self._cal_bus: FeetechBus | None = None

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
        lo, hi = JOINT_ENDS[motor]
        return (f"min ({np.rad2deg(self.lower[i]):+.0f} deg): {lo}", f"max ({np.rad2deg(self.upper[i]):+.0f} deg): {hi}")

    def _calibration_plan(self) -> tuple[list[str], str]:
        """Joints to calibrate, or "home" / "check" for those steps alone."""
        motors = list(self.motor_ids)
        if not self.calibration:
            return motors, ""
        missing = [m for m in motors if m not in self.calibration]
        print(f"Calibration file: {self.calibration_fpath}")
        if missing:
            print(f"Not calibrated yet: {' '.join(missing)}")
        while True:
            answer = input(
                "Press ENTER to keep it, or type:\n"
                "  all            redo every joint\n"
                "  J3, J2 gripper redo only those joints\n"
                "  home           only set the home pose\n"
                "  check          print the arm's angles as you pose it, to compare with the sim\n"
                "  flip J3 J4     reverse those joints' direction, without moving them\n"
                "> "
            ).replace(",", " ").split()
            words = [a.lower() for a in answer]
            if not words:
                return [], ""
            if words[0] == "flip":
                if self._flip([w for w in answer[1:]]):
                    return [], ""
                continue
            if words in (["all"], ["c"]):
                return motors, ""
            if words in (["home"], ["check"]):
                return [], words[0]
            names = {m.lower(): m for m in motors}
            unknown = [a for a, w in zip(answer, words) if w not in names]
            if not unknown:
                return [names[w] for w in words], ""
            print(f"Unknown joint(s) {unknown}.")

    def _flip(self, joints: list[str]) -> bool:
        """Swap which measured end is min for calibrated joints; saves the file."""
        names = {m.lower(): m for m in self.calibration}
        bad = [j for j in joints if j.lower() not in names]
        if not joints or bad:
            print(f"Name calibrated joints to flip, e.g. 'flip J3'. Not calibrated: {bad}")
            return False
        for j in joints:
            motor = names[j.lower()]
            c = self.calibration[motor]
            c.drive_mode = 1 - c.drive_mode
            if is_range_calibrated(c.range_min, c.range_max):
                # Mirror the zero inside the measured range, so the ends trade places.
                c.homing_offset = c.range_min + c.range_max - c.homing_offset
            print(f"  {motor} flipped.")
        self._save_calibration()
        print(f"Saved to {self.calibration_fpath}")
        return True

    def _bus_retry(self, what: str, call):
        """Run `call(bus)`; if the bus drops, wait for the user, reopen it and retry.

        Moving a joint by hand to its end can pull a servo cable loose, and a
        back-driven servo can brown out the supply; neither should lose the run.
        """
        while True:
            try:
                return call(self._cal_bus)
            except Exception as exc:
                answer = input(
                    f"  Lost the servos while {what}: {exc}\n"
                    "  Check that servo's cable (and the ones before it) and the 12 V supply,\n"
                    "  then ENTER to retry, or 'q' and ENTER to stop: "
                )
                if answer.strip().lower() == "q":
                    raise
            try:
                self._cal_bus.close()
            except Exception:
                pass
            try:
                cfg = self.config
                self._cal_bus = FeetechBus(cfg.port, cfg.baudrate, list(self.motor_ids.values()))
                # A servo that browned out may come back up with torque on.
                self._cal_bus.set_torque(False)
            except Exception as exc:
                print(f"  Could not reopen {self.config.port}: {exc}")

    def _check(self) -> None:
        """Print the arm's joint angles, as the sim would show them, until 'q'."""
        ready = np.rad2deg(self.kin.ready_q())
        print(
            "\nTorque is off, support the arm. Bend it by hand into the sim's ready pose\n"
            "(the pose teleop starts in) and compare. A joint with the wrong sign was\n"
            "calibrated the wrong way round: redo just that joint.\n"
            f"  ready  {'  '.join(f'{j} {a:+6.0f}' for j, a in zip(ARM_JOINTS, ready))}"
        )
        while input("ENTER reads the arm, 'q' and ENTER quits: ").strip().lower() != "q":
            q = np.rad2deg(self._from_raw(self._bus_retry("reading the arm", lambda b: b.read_positions())))
            print(f"  arm    {'  '.join(f'{j} {a:+6.0f}' for j, a in zip(ARM_JOINTS, q))}")

    def _set_home(self) -> None:
        """Save the pose the arm is held in as the teleop's start and Y pose."""
        path = home_pose_path(self.id, self.calibration_dir)
        answer = input(
            "\nHome pose (torque is off, support the arm): move it to where teleop should\n"
            "start and where Y returns it, jaws included, then ENTER. 's' and ENTER skips: "
        )
        if answer.strip().lower() == "s":
            return
        q = self._from_raw(self._bus_retry("reading the home pose", lambda b: b.read_positions()))
        home = {j: round(float(np.rad2deg(q[i])), 2) for i, j in enumerate(ARM_JOINTS)}
        if self.config.gripper_id is not None:
            home["gripper"] = round(float(np.clip(self._gripper_pct(q[6]), 0.0, 100.0)), 1)
        outside = [j for i, j in enumerate(ARM_JOINTS) if not self.lower[i] <= q[i] <= self.upper[i]]
        if outside:
            print(f"  warning: {' '.join(outside)} past the model's limits; teleop will clip them.")
        path.write_text(json.dumps(home, indent=4) + "\n")
        print(f"Home pose {home} saved to {path}")

    def calibrate(self) -> None:
        """Map each joint's travel onto the model's range: min, middle, max by hand.

        Torque stays off the whole time, and afterwards: the servos never move.
        """
        motors, only = self._calibration_plan()
        if not motors and not only:
            return
        if self.streamer is not None:
            self.streamer.stop()
            self.streamer = None
        own_bus = self.bus is None
        cfg = self.config
        self._cal_bus = self.bus or FeetechBus(cfg.port, cfg.baudrate, list(self.motor_ids.values()))
        try:
            self._bus_retry("turning torque off", lambda b: b.set_torque(False))
            if motors:
                print(
                    "\nTorque is off and stays off: support the arm. For each joint, move it\n"
                    "by hand to its min end, then to any point in between, then to its max\n"
                    "end, pressing ENTER at each. Min and max are the MuJoCo model's joint\n"
                    "limits. The middle point only tells which way the joint turned.\n"
                    "Move the joints slowly: a servo turned fast by hand feeds power back\n"
                    "into the bus. Each joint is saved as soon as it is done.\n"
                )
            for motor in motors:
                self._calibrate_joint(motor)
            # Read with the new calibration, so it is in the model's angles.
            if only == "check":
                self._check()
            else:
                self._set_home()
        finally:
            if own_bus:
                self._cal_bus.close()
            else:
                # It may have been reopened.
                self.bus = self._cal_bus
            self._cal_bus = None
        print("Torque is still off. lerobot-teleoperate uses this from now on.")

    def _calibrate_joint(self, motor: str) -> None:
        i = list(self.motor_ids).index(motor)
        servo_id = self.motor_ids[motor]
        lo_name, hi_name = self._end_names(i, motor)
        # Readings are taken as actual positions. The servo keeps its old
        # offset until the joint is done, so stopping half-way leaves the old
        # calibration valid.
        old_offset = self._bus_retry(f"reading {motor}'s offset", lambda b: b.homing_offset(servo_id))
        while True:
            readings = []
            for step in (lo_name, "middle, anywhere", hi_name):
                input(f"  {motor}: move to {step}, then ENTER ")
                # Only this servo: a loose cable elsewhere does not matter here.
                raw = self._bus_retry(f"reading {motor}", lambda b: b.read_position(servo_id))
                readings.append(raw + old_offset)
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
            break
        self._bus_retry(f"writing {motor}'s offset", lambda b: b.set_homing_offset(servo_id, fit.servo_offset))
        self.calibration[motor] = MotorCalibration(
            id=servo_id,
            drive_mode=fit.drive_mode,
            homing_offset=fit.zero,
            range_min=fit.range_min,
            range_max=fit.range_max,
        )
        self._save_calibration()
        print(f"  {motor} saved to {self.calibration_fpath}")

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
