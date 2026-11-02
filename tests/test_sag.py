"""The droop correction must pull a sagging joint onto its target, in every pose."""

import math

import numpy as np

from lerobot_robot_kikobot.motion import SagCorrector

READ_DT = 0.03  # Positions are read at ~33 Hz on the real arm.
GIVE = math.radians(4.0)  # The real joint's droop per N·m.


def gravity(q):
    """Joint 0 carries the arm: most torque stretched out (q=pi/2), none upright."""
    return np.array([2.0 * math.sin(q[0]), 0.0])


def servo(goal):
    """A loaded position servo: settles short of its goal by GIVE per N·m of the
    load at the pose it settles in."""
    q = goal.copy()
    for _ in range(50):
        q = goal - GIVE * gravity(q)
    return q


def corrector(**kw):
    args = dict(time_constant_s=0.5, max_rad=math.radians(15), gravity=gravity)
    args.update(kw)
    return SagCorrector([True, False], **args)


def hold(sag, command, seconds):
    for _ in range(int(seconds / READ_DT)):
        sag.learn(command, servo(command + sag.correction(command)), READ_DT)


def test_holding_still_removes_the_sag():
    sag = corrector()
    command = np.array([1.0, 0.2])
    assert abs(servo(command)[0] - command[0]) > math.radians(5)  # Sags without it.
    hold(sag, command, 3.0)
    present = servo(command + sag.correction(command))
    assert abs(present[0] - command[0]) < math.radians(0.3)
    assert sag.compliance[1] == 0.0  # Not in the mask.


def test_correction_follows_the_pose():
    # Learned at a light pose, right at once at a heavy one: no relearning needed.
    sag = corrector()
    hold(sag, np.array([0.4, 0.0]), 3.0)
    heavy = np.array([1.5, 0.0])
    present = servo(heavy + sag.correction(heavy))
    assert abs(present[0] - heavy[0]) < math.radians(0.6)


def test_does_not_learn_while_moving():
    sag = corrector()
    command = np.array([0.5, 0.0])
    for _ in range(50):
        command = command + np.array([math.radians(30.0) * READ_DT, 0.0])  # 30 deg/s.
        sag.learn(command, servo(command), READ_DT)
    assert np.all(sag.compliance == 0.0)


def test_correction_is_capped():
    sag = corrector(time_constant_s=0.1, max_rad=math.radians(2))
    command = np.array([1.2, 0.0])
    hold(sag, command, 3.0)
    assert abs(sag.correction(command)[0]) == math.radians(2)
    assert sag.capped[0]
