"""Kiko-Bot for LeRobot: MuJoCo and Isaac Sim servers, the real arm and a 6-DOF gamepad teleop.

LeRobot imports this package because its distribution name starts with
`lerobot_robot_`; importing it registers all five device types:

    --robot.type=kikobot_sim         the arm in `kikobot-sim-server` (MuJoCo, optionally Newton)
    --robot.type=kikobot_isaac       the arm in NVIDIA Isaac Sim, via isaac/kikobot_isaac_server.py
    --robot.type=kikobot_follower    the physical arm (Feetech STS servos), mirrored to either sim
    --robot.type=kikobot_arm         the physical arm through `kikobot-arm-server`
    --teleop.type=kikobot_gamepad    6-DOF Cartesian gamepad control
"""

import sys
from pathlib import Path

# Tools that never use LeRobot skip importing it (and with it torch, OpenCV and
# the camera stack), which can take a minute on a slow machine before they
# print anything.
_NO_LEROBOT = {"kikobot-servo-web", "servo_web.py", "lerobot_robot_kikobot.servo_web"}


def _skip_lerobot() -> bool:
    # `python -m pkg.mod` imports the package while sys.argv[0] is still "-m";
    # sys.orig_argv keeps the module name.
    args = [*sys.argv[:1], *getattr(sys, "orig_argv", [])]
    return any(Path(a).name in _NO_LEROBOT or a in _NO_LEROBOT for a in args)


try:
    if _skip_lerobot():
        raise ImportError
    import lerobot  # noqa: F401
except ImportError:
    # The sim server alone (`kikobot-sim-server`, e.g. in a Newton environment)
    # needs no LeRobot; the LeRobot devices are only registered with it.
    __all__: list[str] = []
else:
    from .arm import KikobotArm
    from .config import (
        KikobotArmConfig,
        KikobotFollowerConfig,
        KikobotGamepadConfig,
        KikobotIsaacConfig,
        KikobotSimConfig,
    )
    from .follower import KikobotFollower
    from .gamepad import KikobotGamepad
    from .isaac import KikobotIsaac
    from .sim import KikobotSim

    __all__ = [
        "KikobotArm",
        "KikobotArmConfig",
        "KikobotFollower",
        "KikobotFollowerConfig",
        "KikobotGamepad",
        "KikobotGamepadConfig",
        "KikobotIsaac",
        "KikobotIsaacConfig",
        "KikobotSim",
        "KikobotSimConfig",
    ]
