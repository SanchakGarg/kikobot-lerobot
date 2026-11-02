"""Range calibration must map min/max readings onto the model's limits, across the wrap."""

import math

import pytest

from lerobot_robot_kikobot.calibration import fit_range, is_range_calibrated
from lerobot_robot_kikobot.feetech import RESOLUTION

LOWER, UPPER = -math.radians(140), math.radians(140)


def _angle(fit, raw_after_offset):
    sign = -1 if fit.drive_mode else 1
    steps_per_rad = sign * (fit.range_max - fit.range_min) / (UPPER - LOWER)
    return (raw_after_offset - fit.zero) / steps_per_rad


def _shifted(fit, actual):
    return (actual - fit.servo_offset) % RESOLUTION


@pytest.mark.parametrize(
    "at_min, at_mid, at_max, rising",
    [
        (500, 2000, 3700, True),  # Plain, reading rises.
        (3700, 2000, 500, False),  # Plain, reading falls.
        (3500, 100, 2600, True),  # Rises through 4095 -> 0.
        (600, 3900, 1500, False),  # Falls through 0 -> 4095.
    ],
)
def test_ends_map_to_limits(at_min, at_mid, at_max, rising):
    fit = fit_range(at_min, at_mid, at_max, LOWER, UPPER)
    assert fit.drive_mode == (0 if rising else 1)
    assert 0 <= fit.range_min < fit.range_max <= RESOLUTION - 1
    assert abs(fit.servo_offset) < RESOLUTION // 2
    assert _angle(fit, _shifted(fit, at_min)) == pytest.approx(LOWER, abs=1e-3)
    assert _angle(fit, _shifted(fit, at_max)) == pytest.approx(UPPER, abs=1e-3)
    assert LOWER < _angle(fit, _shifted(fit, at_mid)) < UPPER
    assert is_range_calibrated(fit.range_min, fit.range_max)


def test_middle_picks_the_direction():
    # Same ends, the middle decides which arc the joint travelled.
    short = fit_range(1000, 1500, 2000, LOWER, UPPER)
    long = fit_range(1000, 500, 2000, LOWER, UPPER)
    assert short.span_steps == 1000 and short.drive_mode == 0
    assert long.span_steps == RESOLUTION - 1000 and long.drive_mode == 1


def test_unmoved_joint_is_rejected():
    with pytest.raises(ValueError):
        fit_range(2000, 2010, 2020, LOWER, UPPER)
    with pytest.raises(ValueError):
        fit_range(1000, 1000, 3000, LOWER, UPPER)


def test_full_range_means_not_range_calibrated():
    assert not is_range_calibrated(0, RESOLUTION - 1)
