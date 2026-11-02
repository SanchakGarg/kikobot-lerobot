"""What sits between a joint target and the servos: a speed cap and the droop fix.

Nothing here filters or delays the command. Like LeRobot's SO-101, the arm
gets each target as soon as it arrives; the gamepad already produces smooth
targets (low-passed sticks, IK, a glide back home). `RateLimiter` only caps
joint speed, so a target that jumps (a policy's first action, the pose the arm
was in when it powered up) is reached at a safe speed. Under the cap it
passes targets through unchanged, with no lag.

Pure numpy. `SagCorrector` works in radians; `RateLimiter` in whatever unit it is given.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

# Longest step the limiter treats as one tick; a stalled loop must not jump.
MAX_DT_S = 0.1


class RateLimiter:
    """Caps each joint's speed. A target within reach this tick is passed as is.

    Until the first target is reached after `reset`, the cap is
    `soft_start_vel` instead, so the first move from wherever the arm was
    when it connected is slow.
    """

    def __init__(self, max_vel: float, soft_start_vel: float | None = None):
        self.max_vel = float(max_vel)
        self.soft_start = float(soft_start_vel) if soft_start_vel else self.max_vel
        self.pos: np.ndarray | None = None
        self._t: float | None = None
        self._starting = True

    def reset(self, pos: np.ndarray, t: float) -> None:
        self.pos = np.array(pos, dtype=float)
        self._t = t
        self._starting = True

    def step(self, target: np.ndarray, t: float) -> np.ndarray:
        target = np.asarray(target, dtype=float)
        if self.pos is None or self._t is None:
            self.reset(target, t)
            return self.pos.copy()
        dt = min(max(t - self._t, 0.0), MAX_DT_S)
        self._t = t
        vel = self.soft_start if self._starting else self.max_vel
        limit = vel * dt
        self.pos = self.pos + np.clip(target - self.pos, -limit, limit)
        if self._starting and np.allclose(self.pos, target, atol=1e-6):
            self._starting = False
        return self.pos.copy()


class SagCorrector:
    """Cancels joints' droop under the arm's weight, in every pose.

    A position servo pushes back in proportion to how far it is off its goal,
    so a loaded joint (J2) settles short of it, by an amount that grows with
    the gravity torque on it: nothing with the arm upright, most stretched
    out. The goal is raised by `compliance * gravity torque`, the torque taken
    from the model at the commanded pose, so the correction follows the arm
    as it moves. The compliance (rad per N·m) is learned while the arm holds
    still, from the gap between the commanded and measured angle. It is not
    learned during moves, where that gap is tracking lag, not droop.
    """

    # Commanded speed below which the arm counts as holding still.
    REST_SPEED_RAD_S = math.radians(5.0)
    # Below this gravity torque (N·m) the droop is too small to learn from.
    MIN_TORQUE_NM = 0.05

    def __init__(
        self,
        mask: np.ndarray,
        time_constant_s: float,
        max_rad: float,
        gravity: Callable[[np.ndarray], np.ndarray],
    ):
        self.mask = np.asarray(mask, dtype=float)
        self.time_constant_s = time_constant_s
        self.max_rad = max_rad
        self.gravity = gravity
        self.compliance = np.zeros(len(self.mask))
        self.capped = np.zeros(len(self.mask), dtype=bool)
        self._last_command: np.ndarray | None = None

    def correction(self, command: np.ndarray) -> np.ndarray:
        """Goal offset (radians) that cancels the droop at `command`."""
        raw = self.mask * self.compliance * self.gravity(command)
        self.capped = np.abs(raw) > self.max_rad
        return np.clip(raw, -self.max_rad, self.max_rad)

    def learn(self, command: np.ndarray, present: np.ndarray, dt: float) -> None:
        """Update the compliance from one measurement, taken with the current correction on."""
        if self._last_command is not None and dt > 0 and self.time_constant_s > 0:
            at_rest = np.abs(command - self._last_command) / dt < self.REST_SPEED_RAD_S
            tau = self.gravity(command)
            learn = at_rest & (self.mask > 0) & (np.abs(tau) > self.MIN_TORQUE_NM)
            # What is still missing, per N·m, closed a fraction per measurement.
            gain = min(dt / self.time_constant_s, 1.0)
            step = np.divide(command - present, tau, out=np.zeros_like(tau), where=learn)
            self.compliance += gain * step
            # Gravity only ever pulls a joint down, so the compliance is not negative.
            np.maximum(self.compliance, 0.0, out=self.compliance)
        self._last_command = command.copy()
