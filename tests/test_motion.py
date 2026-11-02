"""The speed cap must pass smooth targets through untouched (no lag) and only cap jumps."""

import numpy as np

from lerobot_robot_kikobot.motion import RateLimiter

DT = 1.0 / 60.0


def run(limiter, targets, t0=0.0):
    return [limiter.step(np.atleast_1d(q), t0 + (k + 1) * DT) for k, q in enumerate(targets)]


def test_ramp_under_the_cap_has_no_lag():
    lim = RateLimiter(max_vel=3.0)
    lim.reset(np.zeros(1), 0.0)
    lim._starting = False
    # 1 rad/s ramp, a third of the cap: every output equals its input on the same tick.
    ramp = [k * DT for k in range(1, 61)]
    out = run(lim, ramp)
    assert np.allclose(np.concatenate(out), ramp)


def test_step_is_capped_then_reached():
    lim = RateLimiter(max_vel=2.0)
    lim.reset(np.zeros(1), 0.0)
    lim._starting = False
    out = np.concatenate(run(lim, [1.0] * 40))
    assert np.allclose(np.diff(np.concatenate([[0.0], out]))[:20], 2.0 * DT)
    assert out[-1] == 1.0


def test_soft_start_until_first_target_is_reached():
    lim = RateLimiter(max_vel=3.0, soft_start_vel=0.5)
    lim.reset(np.zeros(1), 0.0)
    out = np.concatenate(run(lim, [0.25] * 60))
    assert np.isclose(out[0], 0.5 * DT)  # Slow on the way to the first target...
    assert out[-1] == 0.25
    nxt = lim.step(np.array([1.0]), 61 * DT)  # One tick after the last.
    assert np.isclose(nxt[0] - 0.25, 3.0 * DT)  # ...then full speed.


def test_stall_does_not_jump():
    lim = RateLimiter(max_vel=1.0)
    lim.reset(np.zeros(1), 0.0)
    lim._starting = False
    out = lim.step(np.array([10.0]), 5.0)  # A 5 s stall counts as 0.1 s at most.
    assert np.isclose(out[0], 0.1)
