"""`MotionConfig`, kept free of LeRobot imports.

The Isaac Sim server (`isaac/kikobot_isaac_server.py`) runs under Isaac Sim's
own Python, where LeRobot is not installed, and loads `motion.py` and this
file by path. So neither may import anything LeRobot-specific.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotionConfig:
    """How a stream of joint targets is turned into servo motion.

    Shared by the sim and the real arm, so the sim moves the way the hardware
    will. See `motion.py` for what each stage does.
    """

    # Trajectory follower: the commanded joints never exceed these.
    max_velocity_deg_s: float = 120.0
    max_acceleration_deg_s2: float = 600.0
    # Gain of the final approach (1/s). Higher tracks tighter, lower is softer.
    tracking_gain: float = 12.0
    # Extrapolate the incoming target between updates so the follower does
    # not lag a steadily moving stick by a whole control period.
    feedforward: bool = True
    # Input shaper, as in Klipper: none | zv | zvd | mzv | ei.
    shaper_type: str = "zv"
    # The arm's dominant ringing frequency. Measure it with
    # `python scripts/measure_resonance.py`; the default is a typical value
    # for a servo arm of this size.
    shaper_freq_hz: float = 6.0
    shaper_damping: float = 0.1
    # Velocity limit ramps from this fraction to 1 over `soft_start_s`, so the
    # first move after connecting (wherever the arm was) is gentle.
    soft_start_fraction: float = 0.25
    soft_start_s: float = 3.0
