"""OpenVLA's vla-scripts/finetune.py, with the Kiko-Bot desk datasets registered first.

    OPENVLA_DIR=../openvla torchrun --standalone --nnodes 1 --nproc-per-node 1 \\
        scripts/openvla/finetune_kikobot.py \\
        --vla_path openvla/openvla-7b \\
        --data_root_dir /raid/home/$USER/datasets/kikobot_desk/tfds \\
        --dataset_name kikobot_desk_pen_to_pen_cup \\
        --run_root_dir runs --batch_size 8 --use_lora True

Every argument goes to OpenVLA's finetune.py unchanged. --dataset_name is a
task's dataset, kikobot_desk_all (one dataset of every task) or the mixture
kikobot_desk_all_tasks.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    openvla = Path(os.environ.get("OPENVLA_DIR", HERE.parents[2] / "openvla")).resolve()
    script = openvla / "vla-scripts" / "finetune.py"
    if not script.is_file():
        raise SystemExit(f"No OpenVLA at {openvla} (set OPENVLA_DIR to your clone of github.com/openvla/openvla).")
    sys.path[:0] = [str(openvla), str(HERE)]
    import kikobot_oxe

    kikobot_oxe.register()
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
