"""The motion shaper must keep a stair-step target smooth and settle exactly."""

import math

import numpy as np
import pytest

from lerobot_robot_kikobot.config import MotionConfig
from lerobot_robot_kikobot.motion import MotionShaper, TrajectoryFollower, shaper_impulses

STREAM_DT = 0.01  # 100 Hz, as on the real arm.
CONTROL_DT = 1 / 30


@pytest.mark.parametrize("kind", ["none", "zv", "zvd", "mzv", "ei"])
def test_shaper_amplitudes_sum_to_one(kind):
    amps, delays = shaper_impulses(kind, 6.0, 0.1)
    assert amps.sum() == pytest.approx(1.0)
    assert delays[0] == 0.0 and np.all(np.diff(delays) > 0)


def _run(cfg, targets_per_tick, seconds):
    """Feed 30 Hz targets, step at 100 Hz; return commands at each stream tick."""
    shaper = MotionShaper(1, cfg)
    shaper.reset(np.zeros(1), 0.0)
    out, t, next_target = [], 0.0, 0.0
    while t < seconds:
        if t >= next_target - 1e-9:
            shaper.set_target(np.array([targets_per_tick(next_target)]), t)
            next_target += CONTROL_DT
        out.append(shaper.step(t)[0])
        t += STREAM_DT
    return np.array(out)


def test_step_target_respects_limits_and_settles():
    cfg = MotionConfig(soft_start_s=0.0, shaper_type="none")
    cmd = _run(cfg, lambda t: math.radians(30.0), 2.0)
    vel = np.diff(cmd) / STREAM_DT
    acc = np.diff(vel) / STREAM_DT
    assert np.max(np.abs(vel)) <= math.radians(cfg.max_velocity_deg_s) + 1e-6
    assert np.max(np.abs(acc)) <= math.radians(cfg.max_acceleration_deg_s2) + 1e-6
    assert cmd[-1] == pytest.approx(math.radians(30.0), abs=1e-4)
    # No overshoot past the target.
    assert np.max(cmd) <= math.radians(30.0) + math.radians(0.2)


def test_ramp_target_has_no_staircase():
    """A steadily moving 30 Hz target must come out as steady motion.

    This is the 'move, pause, move' bug: without the follower, the command
    holds still between updates and then jumps.
    """
    cfg = MotionConfig(soft_start_s=0.0, shaper_type="zv")
    speed = math.radians(40.0)
    cmd = _run(cfg, lambda t: speed * t, 3.0)
    vel = np.diff(cmd) / STREAM_DT
    steady = vel[150:]  # After the start-up transient.
    assert np.min(steady) > 0.8 * speed, "command paused between target updates"
    assert np.max(steady) < 1.2 * speed


def test_stops_on_last_target_when_stream_ends():
    """Targets stop arriving mid-ramp: settle on the last one, not past it."""
    cfg = MotionConfig(soft_start_s=0.0, shaper_type="zv")
    speed = math.radians(40.0)
    shaper = MotionShaper(1, cfg)
    shaper.reset(np.zeros(1), 0.0)
    t = 0.0
    for k in range(30):  # 1 s of 30 Hz updates, then silence.
        shaper.set_target(np.array([speed * k / 30]), t)
        for _ in range(3):
            shaper.step(t)
            t += STREAM_DT
    last = speed * 29 / 30
    for _ in range(200):
        out = shaper.step(t)[0]
        t += STREAM_DT
    assert out == pytest.approx(last, abs=math.radians(0.05))


def test_soft_start_slows_the_first_move():
    slow = _run(MotionConfig(shaper_type="none"), lambda t: 1.0, 0.5)
    fast = _run(MotionConfig(shaper_type="none", soft_start_s=0.0), lambda t: 1.0, 0.5)
    assert slow[-1] < fast[-1]


def test_zv_shaper_cancels_ringing():
    """Drive a lightly damped 6 Hz oscillator (an arm on a springy servo)
    with and without the matching ZV shaper; shaped residual ringing is far
    smaller."""
    freq, zeta = 6.0, 0.05
    wn = 2 * math.pi * freq

    def residual(kind):
        cfg = MotionConfig(soft_start_s=0.0, shaper_type=kind, shaper_freq_hz=freq, shaper_damping=zeta)
        cmd = _run(cfg, lambda t: 0.5 if t > 0.1 else 0.0, 3.0)
        x = v = 0.0
        xs = []
        for u in cmd:  # Semi-implicit Euler at 10 substeps per tick.
            for _ in range(10):
                a = wn * wn * (u - x) - 2 * zeta * wn * v
                v += a * STREAM_DT / 10
                x += v * STREAM_DT / 10
            xs.append(x)
        return np.max(np.abs(np.array(xs[200:]) - 0.5))

    assert residual("zv") < 0.25 * residual("none")


def test_follower_holds_position_without_target_changes():
    f = TrajectoryFollower(2, 1.0, 5.0, gain=12.0)
    f.reset(np.array([0.3, -0.2]))
    for i in range(100):
        f.step(STREAM_DT, i * STREAM_DT)
    np.testing.assert_allclose(f.pos, [0.3, -0.2])
