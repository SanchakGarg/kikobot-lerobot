"""Kiko-Bot for LeRobot: MuJoCo and Isaac Sim servers, the real arm and a 6-DOF gamepad teleop.

LeRobot imports this package because its distribution name starts with
`lerobot_robot_`; importing it registers all four device types:

    --robot.type=kikobot_sim         the arm in `kikobot-sim-server` (MuJoCo, optionally Newton)
    --robot.type=kikobot_isaac       the arm in NVIDIA Isaac Sim, via isaac/kikobot_isaac_server.py
    --robot.type=kikobot_follower    the physical arm (Feetech STS servos), mirrored to either sim
    --teleop.type=kikobot_gamepad    6-DOF Cartesian gamepad control
"""

try:
    import lerobot  # noqa: F401
except ImportError:
    # The sim server alone (`kikobot-sim-server`, e.g. in a Newton environment)
    # needs no LeRobot; the LeRobot devices are only registered with it.
    __all__: list[str] = []
else:
    from .config import KikobotFollowerConfig, KikobotGamepadConfig, KikobotIsaacConfig, KikobotSimConfig
    from .follower import KikobotFollower
    from .gamepad import KikobotGamepad
    from .isaac import KikobotIsaac
    from .sim import KikobotSim

    __all__ = [
        "KikobotFollower",
        "KikobotFollowerConfig",
        "KikobotGamepad",
        "KikobotGamepadConfig",
        "KikobotIsaac",
        "KikobotIsaacConfig",
        "KikobotSim",
        "KikobotSimConfig",
    ]
