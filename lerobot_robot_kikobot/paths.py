"""Where the MuJoCo models live.

The package is installed editable (`pip install -e .`), so the repo's top-level
`models/` folder sits next to this package. `KIKOBOT_MODELS_DIR` overrides it
for a non-editable install or a model kept elsewhere.
"""

import os
from pathlib import Path

MODELS_DIR = Path(
    os.environ.get("KIKOBOT_MODELS_DIR", Path(__file__).resolve().parent.parent / "models")
)

# Short names accepted by `--robot.scene`; anything else is treated as a path.
SCENES = {
    "empty": "scene.xml",
    "pick_place": "pick_place_scene.xml",
    "cylinder": "cylinder_scene.xml",
}

# The bare arm, for kinematics. str, not Path: MjModel.from_xml_path wants a string.
ARM_SCENE_PATH = str(MODELS_DIR / SCENES["empty"])


def resolve_scene(scene: str) -> str:
    """Map a short scene name or a path to an existing MJCF path."""
    path = MODELS_DIR / SCENES[scene] if scene in SCENES else Path(scene)
    if not path.is_file():
        raise FileNotFoundError(
            f"Scene {scene!r} not found at {path}. Use one of {sorted(SCENES)} or a path "
            "to an MJCF file, or set KIKOBOT_MODELS_DIR."
        )
    return str(path)
