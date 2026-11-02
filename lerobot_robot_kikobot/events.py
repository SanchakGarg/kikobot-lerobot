"""A scene-reset request from the gamepad to a simulated robot.

LeRobot's teleoperate/record loops have no hook to reset a sim between
episodes, and the teleop never sees the robot. Both run in the same process,
though, so the gamepad raises this flag (Back / View / Share button) and the
sim robots (`kikobot_sim`, `kikobot_isaac`) take it on their next
`send_action` and ask their sim server to respawn the scene. The real arm ignores it.
"""

from __future__ import annotations

import threading

_reset = threading.Event()


def request_scene_reset() -> None:
    _reset.set()


def take_scene_reset() -> bool:
    """True once per request: reading it clears it."""
    if _reset.is_set():
        _reset.clear()
        return True
    return False
