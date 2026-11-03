"""The physical Kiko-Bot arm behind LeRobot's `Robot` interface.

Same action and observation keys as the sim (`J1.pos`..`J6.pos` in degrees,
`gripper.pos` in % open), so a teleop, dataset or policy works on either.

It drives the servos the way LeRobot's SO-101 follower does: `send_action`
writes the goal straight to every servo in one sync-write packet (no reply,
well under a millisecond), and `get_observation` reads them back in one sync
read. Nothing filters or delays the goal; only a speed cap (`RateLimiter`)
stops a target that jumps from being reached at the servo's full speed.

The same command goes to a sim server (`--robot.sim_backend`) on the same
tick, so the sim arm and the real arm move together. It is sent from a
thread of its own (`SimMirror`): the real arm never waits for the sim.
"""

from __future__ import annotations

import json
import logging
import math
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import MotorCalibration
from lerobot.robots import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .calibration import fit_range, is_range_calibrated
from .config import ARM_JOINTS, SIM_BACKENDS, KikobotFollowerConfig
from .feetech import (
    ADDR_ACCELERATION,
    ADDR_D_COEFFICIENT,
    ADDR_GOAL_POSITION,
    ADDR_GOAL_SPEED,
    ADDR_I_COEFFICIENT,
    ADDR_P_COEFFICIENT,
    ADDR_RETURN_DELAY,
    ADDR_TORQUE_ENABLE,
    RESOLUTION,
    FeetechBus,
)
from .kinematics import ArmKinematics, GravityModel
from .paths import home_pose_path
from .motion import RateLimiter, SagCorrector
from .sim_client import SimClient, SimMirror

logger = logging.getLogger(__name__)

RAD_PER_STEP = 2.0 * math.pi / RESOLUTION
# Uncalibrated default: the servo's mid-scale reading is the joint's zero,
# which is how the arm was assembled.
DEFAULT_HOMING = RESOLUTION // 2

# Speed of the move to centre after a joint is calibrated, the only motion
# calibration makes.
CENTER_SPEED_DEG_S = 25.0
CENTER_RATE_HZ = 50.0
# Position reads in a row that may fail before the arm is declared lost.
MAX_READ_FAILURES = 25

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
        if config.sim_backend not in SIM_BACKENDS:
            raise ValueError(f"sim_backend must be one of {SIM_BACKENDS}, not {config.sim_backend!r}.")
        for field_name in ("inverted_joints", "sag_joints"):
            unknown = set(getattr(config, field_name)) - set(self.motor_ids)
            if unknown:
                raise ValueError(f"{field_name} {sorted(unknown)} not in {list(self.motor_ids)}.")
        self.kin = ArmKinematics()
        self.lower = self.kin.lower.copy()
        self.upper = self.kin.upper.copy()
        if config.gripper_id is not None:
            ends = np.deg2rad([config.gripper_open_deg, config.gripper_closed_deg])
            self.lower = np.append(self.lower, ends.min())
            self.upper = np.append(self.upper, ends.max())
        self.bus: FeetechBus | None = None
        # Last goal written to the servos (radians, J1..J6 then the gripper),
        # without the sag correction. None until torque is on: calibration only.
        self._command: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._present: np.ndarray | None = None
        self._limiter = RateLimiter(math.radians(config.max_velocity_deg_s), math.radians(config.soft_start_deg_s))
        self._sag: SagCorrector | None = None
        self._read_failures = 0
        self._last_read_t: float | None = None
        # The sim server mirror, and when it last failed (warned at most every 5 s).
        self._sim: SimMirror | None = None
        self._sim_warned_t = -math.inf
        self._sag_warned_t = -math.inf
        self.cameras = make_cameras_from_configs(config.cameras)
        self._last_gripper_pct = 100.0
        # The bus during calibrate(), reopened there if it drops.
        self._cal_bus: FeetechBus | None = None
        # Joints calibrate() has moved to their centre and holds there.
        self._held: list[str] = []
        # The sim server, posed the way calibrate() is asking for.
        self._cal_view: SimClient | None = None

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
            # the speed cap eases it back in instead.
            raw = c.homing_offset + angle * self._steps_per_rad(i, motor, c)
            out[self.motor_ids[motor]] = int(round(raw))
        return out

    def _from_raw(self, raw: dict[int, int]) -> np.ndarray:
        q = []
        for i, (motor, servo_id) in enumerate(self.motor_ids.items()):
            c = self._calib(motor)
            q.append((raw[servo_id] - c.homing_offset) / self._steps_per_rad(i, motor, c))
        return np.array(q)

    def _reachable(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint limits: the model's, narrowed to the measured ends of calibrated joints.

        After a joint is re-zeroed its measured ends no longer sit exactly on
        the model's limits; this keeps it off its hard stop.
        """
        lower, upper = self.lower.copy(), self.upper.copy()
        for i, motor in enumerate(self.motor_ids):
            c = self.calibration.get(motor)
            if c is None or not is_range_calibrated(c.range_min, c.range_max):
                continue
            ends = sorted((r - c.homing_offset) / self._steps_per_rad(i, motor, c) for r in (c.range_min, c.range_max))
            lower[i], upper[i] = max(lower[i], ends[0]), min(upper[i], ends[1])
        return lower, upper

    @property
    def is_calibrated(self) -> bool:
        # Uncalibrated still runs, on the mid-scale default above.
        return True

    def _open_cal_view(self) -> None:
        if not self.config.calibration_view or self.config.sim_backend == "none":
            return
        cfg = self.config
        client = SimClient(*cfg.sim_address, cfg.sim_timeout_s)
        try:
            client.hello()
            self._cal_view = client
            print(f"The sim server's window shows each pose to match ({client.address}).")
        except Exception as exc:
            client.close()
            logger.warning(
                "No calibration view (%s). Start the sim server first to see each pose; "
                "the prompts alone work too.", exc,
            )

    def _show(self, motor: str | None = None, angle: float = 0.0, q: np.ndarray | None = None) -> None:
        """Pose the sim arm: every joint at its centre except `motor` at `angle`, or `q`."""
        if self._cal_view is None:
            return
        if q is None:
            q = (self.lower + self.upper) / 2
            if motor is not None:
                q[list(self.motor_ids).index(motor)] = angle
        pct = self._gripper_pct(q[6]) if len(q) > len(ARM_JOINTS) else 100.0
        try:
            self._cal_view.call({
                "cmd": "pose",
                "q_deg": np.rad2deg(q[: len(ARM_JOINTS)]).tolist(),
                "gripper_pct": float(np.clip(pct, 0.0, 100.0)),
            })
        except Exception as exc:
            logger.warning("Calibration view lost (%s); prompts only from here.", exc)
            self._cal_view.close()
            self._cal_view = None

    def _end_names(self, i: int, motor: str) -> tuple[str, str]:
        if motor == "gripper":
            closed_is_min = self.config.gripper_closed_deg < self.config.gripper_open_deg
            return ("jaws closed", "jaws open") if closed_is_min else ("jaws open", "jaws closed")
        lo, hi = JOINT_ENDS[motor]
        return (f"{np.rad2deg(self.lower[i]):+.0f} deg, {lo}", f"{np.rad2deg(self.upper[i]):+.0f} deg, {hi}")

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
                "  zero J5        re-zero a joint that sits off the sim by a fixed amount\n"
                "> "
            ).replace(",", " ").split()
            words = [a.lower() for a in answer]
            if not words:
                return [], ""
            if words[0] == "flip":
                if self._flip([w for w in answer[1:]]):
                    return [], ""
                continue
            if words[0] == "zero":
                names = {m.lower(): m for m in self.calibration if m in ARM_JOINTS}
                chosen = [names.get(w) for w in words[1:]]
                if chosen and None not in chosen:
                    return chosen, "zero"
                print(f"Name calibrated arm joints to re-zero, e.g. 'zero J5' ({' '.join(names.values())}).")
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
        self._mirror_home([names[j.lower()] for j in joints])
        return True

    def _mirror_home(self, motors: list[str]) -> None:
        """Keep the saved home the same physical pose after `motors` were flipped."""
        path = home_pose_path(self.id, self.calibration_dir)
        if not path.is_file():
            return
        home = json.loads(path.read_text())
        for motor in motors:
            if motor not in home:
                continue
            if motor == "gripper":
                # % open runs the other way once the ends trade places.
                home[motor] = round(100.0 - home[motor], 1)
            else:
                # A flipped joint reads lower + upper - angle at the same place.
                i = list(self.motor_ids).index(motor)
                home[motor] = round(float(np.rad2deg(self.lower[i] + self.upper[i])) - home[motor], 2)
        path.write_text(json.dumps(home, indent=4) + "\n")
        print(f"Home pose updated to the same physical pose: {home}")

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
            except Exception as exc:
                print(f"  Could not reopen {self.config.port}: {exc}")

    def _zero_joint(self, motor: str) -> None:
        """Make where the joint is now read 0 deg, keeping its measured scale."""
        i = list(self.motor_ids).index(motor)
        servo_id = self.motor_ids[motor]
        c = self.calibration[motor]
        self._show()
        input(
            f"\n{motor} (limp): set it exactly as in the MuJoCo view, its zero (with every\n"
            "joint at 0 the arm stands straight up). ENTER: "
        )
        raw = self._bus_retry(f"reading {motor}", lambda b: b.read_position(servo_id))
        shift = (raw - c.homing_offset) / self._steps_per_rad(i, motor, c)
        c.homing_offset = int(raw)
        self._save_calibration()
        print(f"  {motor} re-zeroed: it read {np.rad2deg(shift):+.1f} deg there before. Saved.")
        self._shift_home(motor, shift)

    def _shift_home(self, motor: str, shift: float) -> None:
        """Keep the saved home the same physical pose after `motor`'s zero moved by `shift`."""
        path = home_pose_path(self.id, self.calibration_dir)
        if not path.is_file():
            return
        home = json.loads(path.read_text())
        if motor in home:
            home[motor] = round(home[motor] - float(np.rad2deg(shift)), 2)
            path.write_text(json.dumps(home, indent=4) + "\n")
            print(f"  Home pose kept the same physical pose: {motor} {home[motor]:+.1f} deg.")

    def _check(self) -> None:
        """Print the arm's joint angles, as the sim would show them, until 'q'."""
        ready = np.rad2deg(self.kin.ready_q())
        self._show(q=np.append(self.kin.ready_q(), self.upper[6:]))
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
        input(
            "\nHome pose (every joint is limp, support the arm): move it to where teleop\n"
            "should start and where Y returns it, jaws included, then ENTER: "
        )
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

        Every servo is limp. The only motion is a slow move of the joint just
        calibrated to its centre, one joint at a time, after the user confirms.
        """
        motors, only = self._calibration_plan()
        if not motors and not only:
            return
        to_zero = motors if only == "zero" else []
        if to_zero:
            motors = []
        # Stop driving the servos; calibration only reads them.
        self._command = None
        own_bus = self.bus is None
        cfg = self.config
        self._cal_bus = self.bus or FeetechBus(cfg.port, cfg.baudrate, list(self.motor_ids.values()))
        self._open_cal_view()
        try:
            self._bus_retry("turning torque off", lambda b: b.set_torque(False))
            if motors:
                print(
                    "\nEvery servo is limp: support the arm. One joint at a time, move it by\n"
                    "hand to its min end, then to any point in between, then to its max end,\n"
                    "pressing ENTER at each. Min and max are the MuJoCo model's joint limits;\n"
                    "each joint's are described before you start it. The middle point only\n"
                    "tells which way the joint turned. Move slowly: a servo turned fast by\n"
                    "hand feeds power back into the bus. Each joint is saved as soon as it\n"
                    "is done; then, if you say so, only that joint turns slowly to its centre\n"
                    "and holds until you release it.\n"
                )
            for motor in motors:
                self._calibrate_joint(motor)
            # Read with the new calibration, so it is in the model's angles.
            if to_zero:
                for motor in to_zero:
                    self._zero_joint(motor)
            elif only == "check":
                self._check()
            elif only == "home" or input("\nSet the home pose now? ENTER for yes, 's' and ENTER to skip: ").strip().lower() != "s":
                self._release_held("so you can pose it")
                self._set_home()
        finally:
            if own_bus:
                self._cal_bus.close()
            else:
                # It may have been reopened.
                self.bus = self._cal_bus
            self._cal_bus = None
            if self._cal_view is not None:
                self._cal_view.close()
                self._cal_view = None
        if self._held:
            print(f"{' '.join(self._held)} stay powered at their centres; the rest are limp.")
        else:
            print("Every joint is limp.")
        self._held = []
        print("lerobot-teleoperate uses this calibration from now on.")

    def _calibrate_joint(self, motor: str) -> None:
        i = list(self.motor_ids).index(motor)
        servo_id = self.motor_ids[motor]
        lo_name, hi_name = self._end_names(i, motor)
        # Readings are taken as actual positions. The servo keeps its old
        # offset until the joint is done, so stopping half-way leaves the old
        # calibration valid.
        old_offset = self._bus_retry(f"reading {motor}'s offset", lambda b: b.homing_offset(servo_id))
        print(
            f"\n--- {motor} --- (every servo is limp; support the arm)\n"
            f"  min = {lo_name}\n"
            f"  max = {hi_name}\n"
            "  Turn it to min, then sweep it through the middle to max."
        )
        while True:
            readings = []
            mid = (self.lower[i] + self.upper[i]) / 2
            for step, angle in (
                ("min", self.lower[i]), ("middle (anywhere in between)", mid), ("max", self.upper[i])
            ):
                self._show(motor, angle)
                input(f"  {motor}: at {step}, as in the MuJoCo view? ENTER ")
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
        self._center_joint(motor)

    def _center_joint(self, motor: str) -> None:
        """Move just this joint, slowly, to the middle of its range, and hold it there.

        The only motion calibration makes, and only after the user says go.
        Joints not calibrated yet stay limp.
        """
        servo_id = self.motor_ids[motor]
        c = self.calibration[motor]
        target = (c.range_min + c.range_max) // 2
        answer = input(
            f"  Only {motor} will now power on, turn slowly to the centre of its range\n"
            f"  ({CENTER_SPEED_DEG_S:.0f} deg/s) and hold there. Joints not calibrated yet stay limp.\n"
            "  Keep clear, then ENTER to move it, or 's' and ENTER to leave it limp: "
        )
        if answer.strip().lower() == "s":
            return

        self._show()

        def move(bus: FeetechBus) -> None:
            start = bus.read_position(servo_id)
            # Goal = where it is before torque comes on, so nothing jumps.
            bus.write_word(servo_id, ADDR_GOAL_POSITION, start)
            bus.write_byte(servo_id, ADDR_TORQUE_ENABLE, 1)
            steps_per_tick = CENTER_SPEED_DEG_S / 360.0 * RESOLUTION / CENTER_RATE_HZ
            n = max(1, math.ceil(abs(target - start) / steps_per_tick))
            for k in range(1, n + 1):
                bus.write_word(servo_id, ADDR_GOAL_POSITION, round(start + (target - start) * k / n))
                time.sleep(1.0 / CENTER_RATE_HZ)

        self._bus_retry(f"moving {motor} to centre", move)
        self._held.append(motor)
        print(f"  {motor} is holding at its centre.")

    def _release_held(self, why: str) -> None:
        """Let the joints held at their centres go limp, once the user is ready."""
        if not self._held:
            return
        input(f"\n{' '.join(self._held)} are powered. Support the arm, then ENTER to release them {why}: ")
        self._bus_retry("releasing the joints", lambda b: b.set_torque(False))
        self._held = []

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

    def _hold_present(self) -> None:
        cfg = self.config
        # Goal = where the arm is before torque comes on, so nothing jumps.
        present_raw = self.bus.sync_read_positions()
        self.bus.sync_write_goals(present_raw)
        self.bus.set_torque(True)
        present = self._from_raw(present_raw)
        now = time.perf_counter()
        self._present = present.copy()
        self._command = present.copy()
        self._target = present.copy()
        self._limiter.reset(present, now)
        self._read_failures = 0
        self._last_read_t = now
        self._sag = (
            SagCorrector(
                [m in cfg.sag_joints for m in self.motor_ids],
                cfg.sag_time_constant_s,
                math.radians(cfg.sag_max_deg),
                GravityModel(),
            )
            if cfg.sag_joints
            else None
        )

    def _connect_sim(self) -> None:
        cfg = self.config
        client = SimClient(*cfg.sim_address, cfg.sim_timeout_s)
        try:
            info = client.hello()
        except Exception as exc:
            client.close()
            logger.warning(
                "No sim server at %s (%s): the real arm runs without the sim mirror. "
                "Start the sim server first, or pass --robot.sim_backend=none.", client.address, exc,
            )
            return
        logger.info("Mirroring the arm to the sim server at %s (scene %s).", client.address, info["scene"])
        try:
            # Start the sim where the real arm is, before anything moves.
            client.call(self._mirror_header("pose", self._command))
        except Exception as exc:
            logger.warning("Sim mirror: %s", exc)
        self._sim = SimMirror(client)

    def _mirror_header(self, cmd: str, q: np.ndarray) -> dict[str, Any]:
        return {
            "cmd": cmd,
            "q_deg": np.rad2deg(q[: len(ARM_JOINTS)]).tolist(),
            "gripper_pct": self._last_gripper_pct,
        }

    def _mirror(self, q: np.ndarray) -> None:
        """Hand the command the servos got to the mirror thread. Never waits."""
        if self._sim is None:
            return
        self._sim.send(self._mirror_header("command", q))
        if self._sim.error is not None:
            now = time.perf_counter()
            if now - self._sim_warned_t > 5.0:
                self._sim_warned_t = now
                logger.warning("Sim mirror: %s (the real arm carries on)", self._sim.error)

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
            self._hold_present()
        except Exception:
            self.bus.close()
            self.bus = None
            raise

        for cam in self.cameras.values():
            cam.connect()
        if cfg.sim_backend != "none":
            self._connect_sim()
        logger.info("%s connected on %s.", self, cfg.port)

    @check_if_not_connected
    def disconnect(self) -> None:
        self._command = None
        try:
            if self.config.disable_torque_on_disconnect:
                self.bus.set_torque(False)
        finally:
            self.bus.close()
            self.bus = None
        for cam in self.cameras.values():
            cam.disconnect()
        if self._sim is not None:
            self._sim.close()
            self._sim = None
        logger.info("%s disconnected.", self)

    # --------------------------------------------------------------------- i/o

    def _gripper_pct(self, angle: float) -> float:
        open_, closed = np.deg2rad([self.config.gripper_open_deg, self.config.gripper_closed_deg])
        return float(100.0 * (angle - closed) / (open_ - closed))

    def _gripper_angle(self, pct: float) -> float:
        open_, closed = np.deg2rad([self.config.gripper_open_deg, self.config.gripper_closed_deg])
        return float(closed + np.clip(pct, 0.0, 100.0) / 100.0 * (open_ - closed))

    def _require_torque(self) -> None:
        if self._command is None:
            raise RuntimeError(f"{self} is connected for calibration only (torque off).")

    def _read_present(self) -> np.ndarray:
        """Measured joints. A corrupted reply now and then is normal on a busy
        Feetech bus: it reuses the last reading, and only a run of them raises."""
        try:
            present = self._from_raw(self.bus.sync_read_positions())
        except ConnectionError as exc:
            self._read_failures += 1
            if self._read_failures >= MAX_READ_FAILURES:
                raise ConnectionError(f"{MAX_READ_FAILURES} position reads in a row failed: {exc}") from exc
            logger.debug("Skipped a position read: %s", exc)
            return self._present
        self._read_failures = 0
        now = time.perf_counter()
        if self._sag is not None:
            self._sag.learn(self._command, present, now - self._last_read_t)
        self._last_read_t = now
        self._present = present
        return present

    @check_if_not_connected
    def read_state(self) -> dict[str, float]:
        """Measured joints (`J1.pos`.. in degrees, `gripper.pos` in % open), no cameras."""
        self._require_torque()
        q = self._read_present()
        state = {f"{j}.pos": float(np.rad2deg(q[i])) for i, j in enumerate(ARM_JOINTS)}
        # No gripper servo: report the commanded opening so the key stays valid.
        state["gripper.pos"] = (
            self._gripper_pct(q[6]) if self.config.gripper_id is not None else self._last_gripper_pct
        )
        return state

    def joint_limits_deg(self) -> tuple[list[float], list[float]]:
        """J1..J6 limits the arm accepts goals within, in degrees."""
        lower, upper = self._reachable()
        n = len(ARM_JOINTS)
        return np.rad2deg(lower[:n]).tolist(), np.rad2deg(upper[:n]).tolist()

    def read_camera(self, key: str) -> np.ndarray:
        cam = self.cameras[key]
        return cam.read_latest() if hasattr(cam, "read_latest") else cam.async_read()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict[str, Any] = self.read_state()
        for key in self.cameras:
            obs[key] = self.read_camera(key)
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        self._require_torque()
        target = self._target.copy()
        for i, j in enumerate(ARM_JOINTS):
            if f"{j}.pos" in action:
                target[i] = math.radians(action[f"{j}.pos"])
        if "gripper.pos" in action:
            self._last_gripper_pct = float(np.clip(action["gripper.pos"], 0.0, 100.0))
            if self.config.gripper_id is not None:
                target[6] = self._gripper_angle(self._last_gripper_pct)
        self._target = np.clip(target, *self._reachable())

        command = self._limiter.step(self._target, time.perf_counter())
        goal = command
        if self._sag is not None:
            # The goal carries the droop correction; the command stays the
            # motion itself, for the sim and the observation.
            goal = command + self._sag.correction(command)
            if self._sag.capped.any() and time.perf_counter() - self._sag_warned_t > 30.0:
                self._sag_warned_t = time.perf_counter()
                logger.warning(
                    "Sag correction at its cap on joint(s) %s: the servo cannot hold the "
                    "load there. Raise --robot.sag_max_deg, or see the README on sag.",
                    [m for m, c in zip(self.motor_ids, self._sag.capped) if c],
                )
        self.bus.sync_write_goals(self._to_raw(goal))
        self._command = command
        self._mirror(command)

        sent = {f"{j}.pos": float(np.rad2deg(command[i])) for i, j in enumerate(ARM_JOINTS)}
        sent["gripper.pos"] = self._last_gripper_pct
        return sent
