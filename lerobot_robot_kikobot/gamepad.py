"""6-DOF Cartesian gamepad teleop for Kiko-Bot.

The sticks drive the gripper's full pose (XYZ and roll/pitch/yaw); IK over
all six joints turns that into joint targets. The teleop owns the
commanded pose and solves on its own copy of the model, so it works under
plain `lerobot-teleoperate` / `lerobot-record` with no link to the robot.
It starts at the same ready pose the sim resets to; the real arm's motion
shaper eases into it from wherever the arm is.

Xbox layout (PlayStation: cross = A, circle = B, square = X, triangle = Y):

    Left stick      move in X / Y (forward-back, left-right)
    LT / RT         move down / up
    Right stick X   roll  (twist about the jaw axis)
    Right stick Y   pitch (tilt the jaws forward / back)
    LB / RB         yaw   (tilt the jaws left / right)
    A (hold)        close the jaws
    B (hold)        open the jaws
    X               toggle precision mode (slower)
    Y               back to the ready pose
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.teleoperators import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config import ARM_JOINTS, KikobotGamepadConfig
from .kinematics import ArmKinematics, axis_angle_quat, quat_to_mat

logger = logging.getLogger(__name__)

# IK iterations per control tick, warm-started from the last solution.
IK_ITERATIONS = 4
# A step is accepted only if IK reaches it this closely. Otherwise a smaller
# fraction of it is tried, and failing that the arm holds: at the edge of its
# reach it stops instead of trading one axis for another and drifting.
POS_TOLERANCE_M = 0.0015
ROT_TOLERANCE_RAD = math.radians(1.0)
STEP_FRACTIONS = (1.0, 0.5, 0.25)


class KikobotGamepad(Teleoperator):
    config_class = KikobotGamepadConfig
    name = "kikobot_gamepad"

    def __init__(self, config: KikobotGamepadConfig):
        super().__init__(config)
        self.config = config
        self.kin = ArmKinematics()
        self._glfw = None
        self._q = np.zeros(len(ARM_JOINTS))
        self._pos = np.zeros(3)
        self._rot = np.eye(3)
        self._gripper_pct = 100.0
        self._twist = np.zeros(6)  # Low-passed stick command.
        self._precision = False
        self._last_buttons: dict[str, bool] = {}
        self._last_t: float | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {**{f"{j}.pos": float for j in ARM_JOINTS}, "gripper.pos": float}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._glfw is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import glfw

        if not glfw.init():
            raise RuntimeError(
                "glfw.init() failed. GLFW needs a desktop session (on Ubuntu, run "
                "from the desktop, not a bare SSH shell)."
            )
        self._glfw = glfw
        self._joystick = glfw.JOYSTICK_1 + self.config.joystick
        if glfw.joystick_is_gamepad(self._joystick):
            logger.info("Gamepad: %s", glfw.get_gamepad_name(self._joystick))
        else:
            # GLFW only scans for controllers at init, so it must be in first.
            logger.warning("No gamepad on slot %d. Plug it in, then restart.", self.config.joystick)
        self._go_ready()

    def _go_ready(self) -> None:
        self._q = self.kin.ready_q()
        self._pos, self._rot = self.kin.fk(self._q)
        self._twist[:] = 0.0

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------ input

    def _read(self) -> tuple[np.ndarray, float, dict[str, bool]]:
        """Raw 6-axis command in [-1, 1], jaw direction, and button states."""
        glfw = self._glfw
        state = glfw.get_gamepad_state(self._joystick) if glfw.joystick_is_gamepad(self._joystick) else None
        if state is None:
            return np.zeros(6), 0.0, {}
        ax, bt = state.axes, state.buttons

        def stick(axis: int) -> float:
            v = ax[axis]
            dz = self.config.deadzone
            # Rescale past the deadzone so speed starts from 0, not from a jump.
            return 0.0 if abs(v) < dz else math.copysign((abs(v) - dz) / (1.0 - dz), v)

        def pressed(button: int) -> bool:
            return bt[button] == glfw.PRESS

        # Triggers rest at -1 and read +1 when fully pressed.
        lt = (ax[glfw.GAMEPAD_AXIS_LEFT_TRIGGER] + 1.0) / 2.0
        rt = (ax[glfw.GAMEPAD_AXIS_RIGHT_TRIGGER] + 1.0) / 2.0
        raw = np.array(
            [
                -stick(glfw.GAMEPAD_AXIS_LEFT_Y),  # Stick up = forward (+X).
                -stick(glfw.GAMEPAD_AXIS_LEFT_X),  # Stick left = +Y.
                rt - lt,
                stick(glfw.GAMEPAD_AXIS_RIGHT_X),  # Roll.
                -stick(glfw.GAMEPAD_AXIS_RIGHT_Y),  # Pitch.
                float(pressed(glfw.GAMEPAD_BUTTON_RIGHT_BUMPER))
                - float(pressed(glfw.GAMEPAD_BUTTON_LEFT_BUMPER)),  # Yaw.
            ]
        )
        jaw = float(pressed(glfw.GAMEPAD_BUTTON_B)) - float(pressed(glfw.GAMEPAD_BUTTON_A))
        buttons = {"x": pressed(glfw.GAMEPAD_BUTTON_X), "y": pressed(glfw.GAMEPAD_BUTTON_Y)}
        return raw, jaw, buttons

    def _rising(self, buttons: dict[str, bool], name: str) -> bool:
        was = self._last_buttons.get(name, False)
        return buttons.get(name, False) and not was

    # ----------------------------------------------------------------- action

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        cfg = self.config
        now = time.perf_counter()
        # Real elapsed time keeps speeds in m/s; clamped so a stall cannot jump.
        dt = 1.0 / cfg.fps if self._last_t is None else min(now - self._last_t, 0.1)
        self._last_t = now

        raw, jaw, buttons = self._read()
        if self._rising(buttons, "x"):
            self._precision = not self._precision
            logger.info("Precision mode %s.", "on" if self._precision else "off")
        if self._rising(buttons, "y"):
            self._go_ready()
        self._last_buttons = buttons

        # First-order low-pass: a flicked stick ramps the speed, never steps it.
        alpha = 1.0 if cfg.input_smoothing_s <= 0 else 1.0 - math.exp(-dt / cfg.input_smoothing_s)
        self._twist += alpha * (raw - self._twist)
        scale = cfg.precision_scale if self._precision else 1.0
        lin = self._twist[:3] * cfg.linear_speed_m_s * scale * dt
        ang = self._twist[3:] * math.radians(cfg.angular_speed_deg_s) * scale * dt

        # Rotations are about the gripper's own axes: X pitch, Y yaw, Z roll.
        rotvec = np.array([ang[1], ang[2], ang[0]])
        self._gripper_pct = float(np.clip(self._gripper_pct + jaw * cfg.gripper_speed_pct_s * dt, 0, 100))

        if np.any(lin) or np.any(rotvec):
            self._step_pose(lin, rotvec)

        action = {f"{j}.pos": float(np.rad2deg(self._q[i])) for i, j in enumerate(ARM_JOINTS)}
        action["gripper.pos"] = self._gripper_pct
        return action

    def _clip_workspace(self, pos: np.ndarray) -> np.ndarray:
        cfg = self.config
        pos = pos.copy()
        pos[2] = np.clip(pos[2], cfg.workspace_min_z_m, cfg.workspace_max_z_m)
        r = np.linalg.norm(pos[:2])
        if r > cfg.workspace_radius_m:
            pos[:2] *= cfg.workspace_radius_m / r
        return pos

    def _step_pose(self, lin: np.ndarray, rotvec: np.ndarray) -> None:
        """Move the commanded pose by as much of the step as IK can follow."""
        for fraction in STEP_FRACTIONS:
            pos = self._clip_workspace(self._pos + fraction * lin)
            rot = self._rot @ quat_to_mat(axis_angle_quat(fraction * rotvec))
            q = self.kin.solve(self._q, pos, rot, IK_ITERATIONS)
            e_pos, e_rot = self.kin.pose_error(q, pos, rot)
            if np.linalg.norm(e_pos) <= POS_TOLERANCE_M and np.linalg.norm(e_rot) <= ROT_TOLERANCE_RAD:
                self._q, self._pos, self._rot = q, pos, rot
                return
        # Out of reach or at a joint limit: hold, so reversing frees it at once.

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        # No glfw.terminate(): a MuJoCo viewer in this process shares GLFW,
        # and terminating it would close that window too.
        self._glfw = None
