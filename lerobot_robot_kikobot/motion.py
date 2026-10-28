"""Turns a coarse stream of joint targets into smooth servo motion.

LeRobot hands the robot a new joint target once per control tick (30 Hz). Fed
straight to a position servo, each target is a step: the servo sprints to it
at full speed, stops, and waits for the next one. At low update rates that is
the "move, pause, move" stutter.

The fix is the same two stages a 3D printer's motion planner uses:

1. `TrajectoryFollower` runs at the stream rate (100 Hz) and chases the
   latest target with bounded velocity and acceleration, so the servo always
   gets a goal a small, continuous distance ahead of where it is. With
   feedforward it extrapolates the target's own velocity between updates,
   so a steadily moving stick produces a steady ramp, not a staircase.
2. `InputShaper` convolves that trajectory with a few delayed impulses
   (Klipper's ZV / ZVD / MZV / EI shapers). Each impulse excites the arm's
   ringing mode, timed so the ringing of one cancels the next. It removes
   the wobble an acceleration change sets off, at the cost of a delay of
   about half a ringing period.

Everything here is in radians and is pure numpy, so the sim runs the same
code per physics step and moves exactly like the hardware.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .config import MotionConfig

SHAPER_TYPES = ("none", "zv", "zvd", "mzv", "ei")
# Three missed 30 Hz updates.
FEEDFORWARD_TIMEOUT_S = 0.1


def shaper_impulses(kind: str, freq_hz: float, damping: float) -> tuple[np.ndarray, np.ndarray]:
    """Amplitudes and delays (s) of an input shaper.

    Same definitions as Klipper's `shaper_defs.py`. Amplitudes sum to 1, so
    the shaped signal settles at the same place as the input.
    """
    if kind == "none":
        return np.array([1.0]), np.array([0.0])
    if freq_hz <= 0:
        raise ValueError("shaper_freq_hz must be positive.")
    if not 0 <= damping < 1:
        raise ValueError("shaper_damping must be in [0, 1).")

    df = math.sqrt(1.0 - damping**2)
    period = 1.0 / (freq_hz * df)  # Damped ringing period.
    k = math.exp(-damping * math.pi / df)

    if kind == "zv":
        a, t = [1.0, k], [0.0, 0.5 * period]
    elif kind == "zvd":
        a, t = [1.0, 2.0 * k, k * k], [0.0, 0.5 * period, period]
    elif kind == "mzv":
        k = math.exp(-0.75 * damping * math.pi / df)
        a1 = 1.0 - 1.0 / math.sqrt(2.0)
        a, t = [a1, (math.sqrt(2.0) - 1.0) * k, a1 * k * k], [0.0, 0.375 * period, 0.75 * period]
    elif kind == "ei":
        v_tol = 0.05  # Tolerated residual vibration.
        a1 = 0.25 * (1.0 + v_tol)
        a, t = [a1, 0.5 * (1.0 - v_tol) * k, a1 * k * k], [0.0, 0.5 * period, period]
    else:
        raise ValueError(f"Unknown shaper {kind!r}; expected one of {SHAPER_TYPES}.")

    amps = np.asarray(a)
    return amps / amps.sum(), np.asarray(t)


class InputShaper:
    """Applies `shaper_impulses` to a vector signal sampled at irregular times."""

    def __init__(self, kind: str, freq_hz: float, damping: float):
        self.amps, self.delays = shaper_impulses(kind, freq_hz, damping)
        self._history: deque[tuple[float, np.ndarray]] = deque()

    @property
    def delay_s(self) -> float:
        return float(self.delays[-1])

    def reset(self, t: float, value: np.ndarray) -> None:
        self._history.clear()
        self._history.append((t, np.array(value, dtype=float)))

    def __call__(self, t: float, value: np.ndarray) -> np.ndarray:
        self._history.append((t, np.array(value, dtype=float)))
        # Keep one sample older than the longest delay to interpolate against.
        horizon = t - self.delay_s
        while len(self._history) > 2 and self._history[1][0] <= horizon:
            self._history.popleft()
        return sum(a * self._sample(t - d) for a, d in zip(self.amps, self.delays))

    def _sample(self, t: float) -> np.ndarray:
        """Input at time `t`, linearly interpolated; held at the ends."""
        hist = self._history
        if t <= hist[0][0]:
            return hist[0][1]
        for (t0, v0), (t1, v1) in zip(reversed(list(hist)[:-1]), reversed(hist)):
            if t0 <= t:
                w = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
                return v0 + w * (v1 - v0)
        return hist[-1][1]


class TrajectoryFollower:
    """Chases a target with bounded velocity and acceleration.

    Far from the target it brakes along `v = sqrt(a * |e|)`, which always
    leaves room to stop within the acceleration limit; close in it switches
    to a linear `v = gain * e`, which settles without chatter.
    """

    def __init__(self, n: int, max_vel: np.ndarray | float, max_acc: np.ndarray | float,
                 gain: float, feedforward: bool = True):
        self.max_vel = np.broadcast_to(np.asarray(max_vel, dtype=float), (n,)).copy()
        self.max_acc = np.broadcast_to(np.asarray(max_acc, dtype=float), (n,)).copy()
        self.gain = gain
        self.feedforward = feedforward
        self.pos = np.zeros(n)
        self.vel = np.zeros(n)
        self.target = np.zeros(n)
        self.target_vel = np.zeros(n)
        self._target_t: float | None = None

    def reset(self, pos: np.ndarray) -> None:
        self.pos = np.array(pos, dtype=float)
        self.vel[:] = 0.0
        self.target = self.pos.copy()
        self.target_vel[:] = 0.0
        self._target_t = None

    def set_target(self, target: np.ndarray, t: float) -> None:
        target = np.asarray(target, dtype=float)
        if self.feedforward and self._target_t is not None:
            dt = t - self._target_t
            # A gap longer than a few ticks means the stream paused; do not
            # extrapolate a stale velocity across it.
            if 1e-3 < dt < 0.25:
                measured = (target - self.target) / dt
                self.target_vel = 0.5 * self.target_vel + 0.5 * measured
            else:
                self.target_vel[:] = 0.0
        self.target = target.copy()
        self._target_t = t

    def step(self, dt: float, t: float, speed_scale: float = 1.0) -> np.ndarray:
        if dt <= 0:
            return self.pos
        max_vel = self.max_vel * speed_scale
        ref, ff = self.target, np.zeros_like(self.target)
        age = None if self._target_t is None else t - self._target_t
        # Once targets stop arriving the motion has ended: drop the
        # feedforward, or the follower would coast past the last target.
        if self.feedforward and age is not None and age < FEEDFORWARD_TIMEOUT_S:
            # Extrapolate at most one nominal update ahead.
            ff = np.clip(self.target_vel, -max_vel, max_vel)
            ref = self.target + ff * min(age, 0.05)

        err = ref - self.pos
        mag = np.abs(err)
        v_fb = np.sign(err) * np.minimum(np.sqrt(self.max_acc * mag), self.gain * mag)
        v_des = np.clip(ff + v_fb, -max_vel, max_vel)
        # Never step past the reference in one tick.
        v_des = np.clip(v_des, -mag / dt - np.abs(ff), mag / dt + np.abs(ff))

        dv = np.clip(v_des - self.vel, -self.max_acc * dt, self.max_acc * dt)
        self.vel += dv
        self.pos = self.pos + self.vel * dt
        return self.pos


class MotionShaper:
    """Follower then input shaper, with a soft start. One per arm."""

    def __init__(self, n: int, cfg: MotionConfig):
        self.cfg = cfg
        self.follower = TrajectoryFollower(
            n,
            math.radians(cfg.max_velocity_deg_s),
            math.radians(cfg.max_acceleration_deg_s2),
            cfg.tracking_gain,
            cfg.feedforward,
        )
        self.shaper = InputShaper(cfg.shaper_type, cfg.shaper_freq_hz, cfg.shaper_damping)
        self._start_t = 0.0
        self._last_t: float | None = None

    def reset(self, pos: np.ndarray, t: float) -> None:
        self.follower.reset(pos)
        self.shaper.reset(t, pos)
        self._start_t = t
        self._last_t = t

    def set_target(self, target: np.ndarray, t: float) -> None:
        self.follower.set_target(target, t)

    def step(self, t: float) -> np.ndarray:
        """Advance to time `t` and return the shaped joint command."""
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        cfg = self.cfg
        ramp = 1.0 if cfg.soft_start_s <= 0 else min(1.0, (t - self._start_t) / cfg.soft_start_s)
        scale = cfg.soft_start_fraction + (1.0 - cfg.soft_start_fraction) * ramp
        return self.shaper(t, self.follower.step(dt, t, scale))
