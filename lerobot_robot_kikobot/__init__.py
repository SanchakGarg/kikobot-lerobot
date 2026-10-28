"""Kiko-Bot for LeRobot: the MuJoCo sim, the real arm and a 6-DOF gamepad teleop.

LeRobot imports this package because its distribution name starts with
`lerobot_robot_`; importing it registers all three device types:

    --robot.type=kikobot_sim         MuJoCo simulation
    --robot.type=kikobot_follower    the physical arm (Feetech STS servos)
    --teleop.type=kikobot_gamepad    6-DOF Cartesian gamepad control
"""

from .config import KikobotFollowerConfig, KikobotGamepadConfig, KikobotSimConfig, MotionConfig
from .follower import KikobotFollower
from .gamepad import KikobotGamepad
from .sim import KikobotSim

__all__ = [
    "KikobotFollower",
    "KikobotFollowerConfig",
    "KikobotGamepad",
    "KikobotGamepadConfig",
    "KikobotSim",
    "KikobotSimConfig",
    "MotionConfig",
]
