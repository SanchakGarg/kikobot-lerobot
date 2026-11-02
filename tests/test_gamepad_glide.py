"""Y must glide home smoothly (starting and ending at rest), not jump there."""

import numpy as np

from lerobot_robot_kikobot.config import KikobotGamepadConfig
from lerobot_robot_kikobot.gamepad import GLIDE_MIN_S, KikobotGamepad


def test_glide_home_is_smooth_and_ends_home():
    pad = KikobotGamepad(KikobotGamepadConfig(robot_id=None))
    pad._load_home()
    pad._go_ready()
    home = pad._home_q.copy()
    pad._q = home + np.deg2rad([40, -20, 10, 0, 0, 30])
    pad._start_glide(0.0)
    duration = pad._glide[3]
    assert duration >= GLIDE_MIN_S

    ts = np.linspace(0.0, duration, 121)
    qs = []
    for t in ts:
        pad._step_glide(t) if pad._glide is not None else None
        qs.append(pad._q.copy())
    qs = np.array(qs)
    vel = np.diff(qs, axis=0) / np.diff(ts)[:, None]
    assert np.allclose(qs[-1], home)
    assert pad._glide is None  # Done: sticks drive again.
    # At rest at both ends, and never faster than the glide speed allows (min-jerk peaks at 1.875x mean).
    assert np.abs(vel[0]).max() < 0.05 and np.abs(vel[-1]).max() < 0.05
    mean_speed = np.abs(qs[-1] - qs[0]).max() / duration
    assert np.abs(vel).max() <= 1.9 * mean_speed
