"""kikobot-datagen: record scripted, verified desk demonstrations and export them for OpenVLA.

    kikobot-datagen tasks
    kikobot-datagen record  --task pen_to_pen_cup --episodes 200 --out datasets/kikobot_desk
    kikobot-datagen convert --task pen_to_pen_cup --out datasets/kikobot_desk
    kikobot-datagen preview --task pen_to_pen_cup --out datasets/kikobot_desk --episode 0

Recording defaults to the Newton engine and MuJoCo Warp's ray-traced renderer
(a CUDA GPU, no OpenGL). Neither falls back silently: without them the
command fails, unless --engine / --renderer ask for MuJoCo / OpenGL.
See docs/datagen-openvla.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from .tasks import TASK_NAMES, TASKS, get_task

logger = logging.getLogger("kikobot-datagen")


def _tasks_arg(value: str) -> list[str]:
    if value == "all":
        return list(TASK_NAMES)
    return [get_task(v.strip()).name for v in value.split(",") if v.strip()]


def _check_backends(engine: str, renderer: str, render: bool) -> None:
    """Fail early and clearly if the GPU path was asked for and is not there."""
    from ..sim_backends import cuda_available

    if engine == "newton":
        try:
            import newton  # noqa: F401
        except ImportError as exc:
            raise SystemExit(f"--engine newton: Newton is not installed ({exc}). pip install -e '.[newton]'") from exc
        if not cuda_available():
            raise SystemExit("--engine newton: no CUDA GPU visible to Warp. On the cluster, run inside a GPU job; "
                             "locally, pass --engine mujoco.")
    if render and renderer == "mjwarp":
        try:
            import mujoco_warp  # noqa: F401
        except ImportError as exc:
            raise SystemExit(f"--renderer mjwarp: MuJoCo Warp is not installed ({exc}).") from exc
        if not cuda_available():
            raise SystemExit("--renderer mjwarp needs a CUDA GPU; locally, pass --renderer opengl.")


def cmd_tasks(args) -> None:
    for name in TASK_NAMES:
        t = TASKS[name]
        print(f"{name:32s} {t.strategy:11s} {t.instruction}")


def cmd_record(args) -> None:
    from ..sim_server import SimWorld
    from .record import record_task

    render = not args.no_render
    _check_backends(args.engine, args.renderer, render)
    world = SimWorld("desk_sort", seed=args.seed_base, engine=args.engine, renderer=args.renderer,
                     randomize_lighting=not args.fixed_lighting)
    logger.info("Engine %s, renderer %s.", world.engine_name, world.renderer_name)
    out = Path(args.out)
    summary = {}
    for i, name in enumerate(_tasks_arg(args.task)):
        # Each task its own seed range, so the tasks' desks differ.
        seed_base = args.seed_base + 1_000_000 * TASK_NAMES.index(name)
        stats = record_task(world, get_task(name), out, args.episodes, seed_base=seed_base,
                            image_size=args.image_size, render=render, max_attempts=args.max_attempts)
        summary[name] = stats
        logger.info("%s: %d kept of %d attempts (%.0f %%).", name, stats["kept"], stats["attempts"],
                    100 * stats["success_rate"])
    print(json.dumps(summary, indent=2))
    world.close()


def cmd_convert(args) -> None:
    import os

    # TensorFlow only encodes images here: keep it off the GPU (and its memory).
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from .rlds.kikobot_desk import build, dataset_name

    out = Path(args.out)
    if args.task == "all":
        configs = ["all"]
    elif args.task == "tasks":
        configs = list(TASK_NAMES)
    else:
        configs = _tasks_arg(args.task)
    for config in configs:
        target = out / "tfds" / dataset_name(config)
        if target.exists() and not args.keep_existing:
            shutil.rmtree(target)
        t0 = time.time()
        path = build(out, config)
        logger.info("Built %s in %.0fs: %s", dataset_name(config), time.time() - t0, path)
        if args.delete_raw and config != "all":
            for f in (out / "raw" / config).glob("episode_*.npz"):
                f.unlink()
            logger.info("Deleted the raw episodes of %s (stats.json kept).", config)


def cmd_preview(args) -> None:
    """Write one episode's cameras side by side as an mp4 (needs imageio[ffmpeg]) or a PNG strip."""
    import numpy as np

    from .record import decode_jpeg, load_episode

    path = Path(args.out) / "raw" / get_task(args.task).name / f"episode_{args.episode:05d}.npz"
    ep = load_episode(path)
    frames = [np.concatenate([decode_jpeg(a), decode_jpeg(b)], axis=1)
              for a, b in zip(ep["image"], ep["overhead_image"])]
    dst = Path(args.dst or path.with_suffix(".mp4"))
    try:
        import imageio.v3 as iio

        iio.imwrite(dst, np.stack(frames), fps=int(ep["fps"]))
    except Exception:  # noqa: BLE001 - no ffmpeg: a strip of stills instead
        from PIL import Image

        dst = dst.with_suffix(".png")
        pick = np.linspace(0, len(frames) - 1, 8).astype(int)
        Image.fromarray(np.concatenate([frames[i] for i in pick], axis=0)).save(dst)
    print(f"{ep['language_instruction']}: {len(frames)} frames -> {dst}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="kikobot-datagen", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tasks", help="list the tasks").set_defaults(fn=cmd_tasks)

    r = sub.add_parser("record", help="record perfect episodes of one or more tasks")
    r.add_argument("--task", required=True, help="a task name or item kind, a comma list, or 'all'")
    r.add_argument("--episodes", type=int, default=200, help="perfect episodes to keep per task")
    r.add_argument("--out", default="datasets/kikobot_desk")
    r.add_argument("--seed-base", type=int, default=0)
    r.add_argument("--image-size", type=int, default=224)
    r.add_argument("--engine", choices=("newton", "mujoco"), default="newton")
    r.add_argument("--renderer", choices=("mjwarp", "opengl"), default="mjwarp")
    r.add_argument("--no-render", action="store_true", help="no cameras (planning and checks only)")
    r.add_argument("--fixed-lighting", action="store_true")
    r.add_argument("--max-attempts", type=int, default=None, help="give up after this many desks per task")
    r.set_defaults(fn=cmd_record)

    c = sub.add_parser("convert", help="build the RLDS (TFDS) dataset for OpenVLA")
    c.add_argument("--task", default="all", help="a task, a comma list, 'tasks' (each task), or 'all' (one dataset of every task)")
    c.add_argument("--out", default="datasets/kikobot_desk")
    c.add_argument("--keep-existing", action="store_true", help="do not rebuild a config already built")
    c.add_argument("--delete-raw", action="store_true", help="delete a task's raw npz once it is built")
    c.set_defaults(fn=cmd_convert)

    v = sub.add_parser("preview", help="one recorded episode as a video")
    v.add_argument("--task", required=True)
    v.add_argument("--out", default="datasets/kikobot_desk")
    v.add_argument("--episode", type=int, default=0)
    v.add_argument("--dst", default=None)
    v.set_defaults(fn=cmd_preview)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
                        stream=sys.stdout)
    args.fn(args)


if __name__ == "__main__":
    main()
