"""The Kiko-Bot desk datasets in OpenVLA's dataset registry.

OpenVLA (github.com/openvla/openvla) finds an RLDS dataset by name in three
module-level dicts. This adds `kikobot_desk_<task>` for every task, and the
mixture `kikobot_desk_all_tasks` (each task's dataset, equally weighted).
`kikobot_desk_all` is also a single dataset of every task.

Either import this before OpenVLA builds its dataset (finetune_kikobot.py
does), or paste the same entries into prismatic/vla/datasets/rlds/oxe/
configs.py, transforms.py and mixtures.py.

The datasets follow the layout of OpenVLA's `bridge_orig` entry:
  image (primary; there is one camera), 224 x 224
  state: EEF_state (x y z roll pitch yaw) + pad + gripper_state  -> POS_EULER
  action: delta x y z, delta roll pitch yaw, gripper 1 = open / 0 = close -> EEF_POS
so the standardization transform has nothing left to do but pass it through.
"""

from __future__ import annotations

TASKS = (
    "pen_to_pen_cup",
    "pencil_to_pen_cup",
    "marker_to_pen_cup",
    "highlighter_to_pen_cup",
    "glue_stick_to_supply_tray",
    "tape_roll_to_tape_peg",
    "scissors_to_supply_tray",
    "eraser_to_supply_tray",
    "stapler_to_supply_tray",
    "ruler_to_supply_tray",
    "paper_clip_box_to_supply_tray",
    "sharpener_to_supply_tray",
)
DATASETS = tuple(f"kikobot_desk_{t}" for t in TASKS) + ("kikobot_desk_all",)
MIXTURE = "kikobot_desk_all_tasks"


def kikobot_desk_dataset_transform(trajectory: dict) -> dict:
    """Already standard: 7-D EEF delta actions with an absolute 1-open/0-close gripper."""
    return trajectory


def dataset_config(StateEncoding, ActionEncoding) -> dict:  # noqa: N803 - OpenVLA's enum classes
    return {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    }


def register() -> None:
    """Add the datasets and the mixture to OpenVLA's registries (idempotent)."""
    from prismatic.vla.datasets.rlds.oxe import configs, mixtures, transforms

    cfg = dataset_config(configs.StateEncoding, configs.ActionEncoding)
    for name in DATASETS:
        configs.OXE_DATASET_CONFIGS[name] = dict(cfg)
        transforms.OXE_STANDARDIZATION_TRANSFORMS[name] = kikobot_desk_dataset_transform
    mixtures.OXE_NAMED_MIXTURES[MIXTURE] = [(f"kikobot_desk_{t}", 1.0) for t in TASKS]
