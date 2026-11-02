"""The Kiko-Bot in NVIDIA Isaac Sim behind LeRobot's `Robot` interface.

Isaac Sim runs in its own process: `isaac/kikobot_isaac_server.py`, started
with Isaac Sim's Python, on this PC or another one. It speaks the same
protocol as `kikobot-sim-server` (`protocol.py`) and also runs in real time,
so this robot is `kikobot_sim` with Isaac's port and timeouts. Same keys and
units as `kikobot_sim` and the real arm, so datasets and policies move
between all three.
"""

from __future__ import annotations

from .config import KikobotIsaacConfig
from .sim import KikobotSim


class KikobotIsaac(KikobotSim):
    config_class = KikobotIsaacConfig
    name = "kikobot_isaac"
