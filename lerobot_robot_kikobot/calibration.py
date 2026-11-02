"""Range calibration: map each servo's travel onto the MuJoCo joint's range.

Each joint is moved by hand to the model's lower limit, to any point in
between, and to its upper limit. The middle reading only tells which way the
servo turned on the way, so a travel that crosses the encoder's 4095 -> 0
wrap is still read correctly.

The servo's Homing_Offset register is then set so that the middle of the
travel reads 2048: the range never wraps, and the servo never takes the long
way round to a goal. Positions map linearly, lower limit <-> one end of the
range and upper limit <-> the other, so the real joint reaches exactly the
model's limits.
"""

from __future__ import annotations

from dataclasses import dataclass

from .feetech import RESOLUTION

HALF_TURN = RESOLUTION // 2
# A joint that moved less than this between min and max was not moved.
MIN_SPAN_STEPS = 100


@dataclass
class JointRange:
    # Value for the servo's Homing_Offset register (Present = actual - offset).
    servo_offset: int
    # 0: the reading rises from the lower to the upper limit; 1: it falls.
    drive_mode: int
    # Readings at the two ends, after the offset, so range_min < range_max.
    range_min: int
    range_max: int
    # Reading at the model's zero, after the offset.
    zero: int
    span_steps: int


def _forward(a: int, b: int) -> int:
    """Steps from a to b with the reading rising, through the wrap if need be."""
    return (b - a) % RESOLUTION


def fit_range(at_min: int, at_mid: int, at_max: int, lower: float, upper: float) -> JointRange:
    """Fit a joint from raw readings at its min, middle and max (radians for limits)."""
    at_min, at_mid, at_max = (r % RESOLUTION for r in (at_min, at_mid, at_max))
    rising = _forward(at_min, at_mid) < _forward(at_min, at_max)
    span = _forward(at_min, at_max) if rising else _forward(at_max, at_min)
    if span < MIN_SPAN_STEPS:
        raise ValueError(f"the joint only moved {span} steps between min and max")
    if at_mid in (at_min, at_max):
        raise ValueError("the middle reading equals an end; move it somewhere in between")

    sign = 1 if rising else -1
    centre = at_min + sign * span / 2
    # Into the register's sign-magnitude range, -2047..2047.
    servo_offset = max(round(centre) % RESOLUTION - HALF_TURN, -(HALF_TURN - 1))
    lo = (at_min - servo_offset) % RESOLUTION
    hi = lo + sign * span
    zero = lo + sign * span * (0.0 - lower) / (upper - lower)
    return JointRange(
        servo_offset=servo_offset,
        drive_mode=0 if rising else 1,
        range_min=min(lo, hi),
        range_max=max(lo, hi),
        zero=round(zero),
        span_steps=span,
    )


def is_range_calibrated(range_min: int, range_max: int) -> bool:
    """A zero-pose (or no) calibration keeps the full encoder range."""
    return range_max - range_min < RESOLUTION - 1
