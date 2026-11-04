"""The Kiko-Bot desk RLDS datasets: `kikobot_desk_<task>` for each task, and `kikobot_desk_all`.

One TFDS dataset per task (flat names, as OpenVLA's dataset configs expect),
built from the raw episodes of `kikobot-datagen record` (the TFDS manual
dir, `<out>/raw/<task>/episode_*.npz`) into `<out>/tfds/kikobot_desk_<task>/2.0.0`.
The layout follows OpenVLA's Bridge entry, so OpenVLA's RLDS loader reads it
with the config in openvla_snippets/kikobot_oxe.py:

    steps/observation/image           224x224x3 jpeg   the `record` camera (primary)
    steps/observation/EEF_state       float32[6]  TCP x y z (m, robot base frame), roll pitch yaw (rad)
    steps/observation/gripper_state   float32[1]  0 closed .. 1 open
    steps/observation/joint_state     float32[7]  J1..J6 (rad), gripper 0..1
    steps/action                      float32[7]  dx dy dz, d roll d pitch d yaw (to the TCP target
                                                  0.1 s on, base frame), gripper 1 open / 0 close
    steps/action_joint                float32[7]  joint targets, gripper 0..1
    steps/language_instruction        e.g. "put the pen in the pen cup"
    steps/reward, discount, is_first, is_last, is_terminal

Episodes are split 95 / 5 into `train` and `val`, by episode, deterministically.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

from ..record import decode_jpeg, load_episode
from ..tasks import TASK_NAMES, TASKS

IMAGE_SIZE = 224
VAL_FRACTION = 0.05


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((size, size), Image.Resampling.LANCZOS))


def _is_val(path: Path) -> bool:
    h = int(hashlib.sha1(f"{path.parent.name}/{path.name}".encode()).hexdigest()[:8], 16)
    return (h % 1000) < VAL_FRACTION * 1000


class KikobotDeskBase(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    """Scripted, verified Kiko-Bot desk-tidying demonstrations (MuJoCo / Newton sim)."""

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {
        "1.0.0": "First release.",
        "2.0.0": "One camera (`record`, over the arm in the desk's corner); no `overhead_image`.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Record episodes with `kikobot-datagen record`; the manual dir is <out>/raw."
    TASKS: tuple[str, ...] = ()

    def _info(self) -> tfds.core.DatasetInfo:
        img = tfds.features.Image(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8, encoding_format="jpeg")

        def vec(n, doc):
            return tfds.features.Tensor(shape=(n,), dtype=np.float32, doc=doc)

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": img,
                        "EEF_state": vec(6, "TCP x y z (m), roll pitch yaw (rad), robot base frame."),
                        "gripper_state": vec(1, "Jaw opening, 0 closed .. 1 open."),
                        "joint_state": vec(7, "J1..J6 (rad), gripper 0..1."),
                    }),
                    "action": vec(7, "TCP delta x y z, roll pitch yaw to the target 0.1 s on; gripper 1 open, 0 close."),
                    "action_joint": vec(7, "Joint targets J1..J6 (rad), gripper 0..1."),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    "task": tfds.features.Text(),
                    "seed": tfds.features.Scalar(dtype=np.int64),
                }),
            }),
            homepage="https://github.com/",
            description=__doc__.split("\n\n")[0],
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        raw = Path(dl_manager.manual_dir)
        files = []
        for task in self.TASKS:
            found = sorted((raw / task).glob("episode_*.npz"))
            if not found and len(self.TASKS) == 1:
                raise FileNotFoundError(f"No episodes in {raw / task}: run `kikobot-datagen record` first.")
            files += found
        train = [f for f in files if not _is_val(f)]
        val = [f for f in files if _is_val(f)]
        out = {"train": self._generate_examples(train)}
        if val:
            out["val"] = self._generate_examples(val)
        return out

    def _generate_examples(self, paths: list[Path]):
        for path in paths:
            yield f"{path.parent.name}/{path.stem}", episode_example(path)


def episode_example(path: Path) -> dict:
    """One raw episode as an RLDS episode dict."""
    ep = load_episode(path)
    n = len(ep["joint_state"])
    instruction = str(ep["language_instruction"])
    eef = ep["eef_state"].astype(np.float32)
    steps = []
    for i in range(n):
        steps.append({
            "observation": {
                "image": _resize(decode_jpeg(ep["image"][i]), IMAGE_SIZE),
                "EEF_state": eef[i, :6],
                "gripper_state": eef[i, 6:7],
                "joint_state": ep["joint_state"][i].astype(np.float32),
            },
            "action": ep["action_delta"][i].astype(np.float32),
            "action_joint": ep["action_joint"][i].astype(np.float32),
            "discount": 1.0,
            "reward": float(i == n - 1),
            "is_first": i == 0,
            "is_last": i == n - 1,
            "is_terminal": i == n - 1,
            "language_instruction": instruction,
        })
    return {
        "steps": steps,
        "episode_metadata": {"file_path": str(path), "task": str(ep["task"]), "seed": int(ep["seed"])},
    }


def _camel(name: str) -> str:
    return "".join(w.capitalize() for w in name.split("_"))


def dataset_name(task: str) -> str:
    """The TFDS (and OpenVLA) name of a task's dataset; `all` for every task."""
    return f"kikobot_desk_{task}"


# One registered builder per task, and one for all of them: kikobot_desk_pen_to_pen_cup, ...
BUILDERS: dict[str, type[KikobotDeskBase]] = {}
for _task in (*TASK_NAMES, "all"):
    _cls = type(f"KikobotDesk{_camel(_task)}", (KikobotDeskBase,), {
        "TASKS": TASK_NAMES if _task == "all" else (_task,),
        "__doc__": f"Kiko-Bot desk: {TASKS[_task].instruction if _task in TASKS else 'every task'}.",
        "__module__": __name__,
    })
    assert _cls.name == dataset_name(_task), (_cls.name, _task)
    BUILDERS[_task] = _cls


def build(out_dir: Path, task: str) -> Path:
    """Build one task's dataset (or `all`) into <out_dir>/tfds. Returns its directory."""
    if task not in BUILDERS:
        raise KeyError(f"Unknown task {task!r}.")
    out_dir = Path(out_dir)
    builder = BUILDERS[task](data_dir=str(out_dir / "tfds"))
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(manual_dir=str(out_dir / "raw")),
    )
    return Path(builder.data_dir)
