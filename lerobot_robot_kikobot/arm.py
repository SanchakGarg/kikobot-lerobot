"""The physical Kiko-Bot through `kikobot-arm-server`, behind LeRobot's `Robot` interface.

The server owns the servo bus and speaks the sim protocol (`protocol.py`), so
this robot is `kikobot_sim` pointed at the arm server's port: the gamepad,
`lerobot-record` or a policy drive the real arm exactly as they drive a sim,
from this machine or another one. Same keys and units as the sims.
"""

from __future__ import annotations

from .config import KikobotArmConfig
from .sim import KikobotSim


class KikobotArm(KikobotSim):
    config_class = KikobotArmConfig
    name = "kikobot_arm"

    def reset_scene(self) -> None:
        """No scene to respawn on the real desk; as with `kikobot_follower`, the
        gamepad still sends the arm home, under the speed cap."""
