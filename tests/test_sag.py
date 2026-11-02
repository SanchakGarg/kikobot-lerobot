"""The droop correction must pull a sagging joint onto its target at rest, and not learn during moves."""

import math

import numpy as np

from lerobot_robot_kikobot.motion import SagCorrector

READ_DT = 0.03  # Positions are read at ~33 Hz on the real arm.
SAG = math.radians(6.0)  # J2 settles this far short under load.


def _servo(goal):
    """A loaded position servo: settles a fixed droop below its goal (joint 0 only)."""
    return goal - np.array([SAG, 0.0])


def test_holding_still_removes_the_sag():
    sag = SagCorrector([True, False], time_constant_s=0.5, max_rad=math.radians(15))
    command = np.array([0.3, 0.2])
    for _ in range(int(3.0 / READ_DT)):
        present = _servo(command + sag.bias)
        sag.update(command, present, READ_DT)
    present = _servo(command + sag.bias)
    assert abs(present[0] - command[0]) < math.radians(0.2)
    assert sag.bias[1] == 0.0  # Not in the mask.


def test_does_not_learn_while_moving():
    sag = SagCorrector([True, False], time_constant_s=0.5, max_rad=math.radians(15))
    command = np.array([0.0, 0.0])
    for _ in range(50):
        command = command + np.array([math.radians(30.0) * READ_DT, 0.0])  # 30 deg/s.
        sag.update(command, _servo(command + sag.bias), READ_DT)
    assert np.all(sag.bias == 0.0)


def test_correction_is_capped():
    sag = SagCorrector([True], time_constant_s=0.1, max_rad=math.radians(2))
    command = np.array([0.0])
    for _ in range(200):
        sag.update(command, command - math.radians(10), READ_DT)
    assert sag.bias[0] == math.radians(2)
