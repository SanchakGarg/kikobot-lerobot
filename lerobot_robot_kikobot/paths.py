"""Where the MuJoCo models and the saved home pose live.

The package is installed editable (`pip install -e .`), so the repo's top-level
`models/` folder sits next to this package. `KIKOBOT_MODELS_DIR` overrides it
for a non-editable install or a model kept elsewhere.
"""

import json
import os
from pathlib import Path

MODELS_DIR = Path(
    os.environ.get("KIKOBOT_MODELS_DIR", Path(__file__).resolve().parent.parent / "models")
)

# The bare arm on a floor, for kinematics. str, not Path: MjModel.from_xml_path
# wants a string. The task scenes are built in code: env/mujoco_scene.py.
ARM_SCENE_PATH = str(MODELS_DIR / "scene.xml")


def home_pose_path(robot_id: str, calibration_dir: Path | None = None) -> Path:
    """Where `lerobot-calibrate` saves the real arm's home pose, next to its calibration."""
    if calibration_dir is None:
        from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

        calibration_dir = HF_LEROBOT_CALIBRATION / ROBOTS / "kikobot_follower"
    return Path(calibration_dir) / f"{robot_id}_home.json"


def load_home_pose(robot_id: str) -> dict[str, float] | None:
    """Joint angles in degrees and `gripper` in % open, or None if none is saved."""
    path = home_pose_path(robot_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text())
