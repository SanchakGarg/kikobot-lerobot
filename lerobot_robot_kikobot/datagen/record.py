"""Records perfect episodes of one task: plan, run, check, keep or throw away.

    kikobot-datagen record --task pen_to_pen_cup --episodes 200 --out datasets/kikobot_desk

Each kept episode is `<out>/raw/<task>/episode_XXXXX.npz`. Images are JPEG
bytes, the rest float32 arrays at RECORD_HZ, one row per frame:

    image, overhead_image    the `record` and `overhead` cameras
    joint_state   [T, 7]     J1..J6 (rad), gripper (0 closed .. 1 open)
    eef_state     [T, 7]     TCP x y z (m, base frame), roll pitch yaw (rad), gripper
    action_joint  [T, 7]     joint targets at the end of the frame, gripper target
    action_eef    [T, 7]     TCP target at the end of the frame (x y z r p y), gripper 1 open / 0 closed
    action_delta  [T, 7]     that target minus the measured TCP (dx dy dz, d roll pitch yaw), gripper 1/0

Thrown-away episodes are only counted, with why, in `<out>/raw/<task>/stats.json`.
Runs resume: episodes already on disk count toward --episodes.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import checks as C
from ..kinematics import GravityModel
from .expert import CONTROL_DT, Expert, Plan, PlanError
from .geometry import mat_to_rpy
from .tasks import TaskSpec

logger = logging.getLogger(__name__)

RECORD_HZ = 10.0
SETTLE_AFTER_S = 1.0
CAMERAS = ("record", "overhead")
CARRY_PHASES = ("carry_up", "carry", "align", "place")
GRIP_CLOSED_BELOW = 5.0  # %: a target below this is "close the jaws"


class EpisodeFailed(RuntimeError):
    pass


@dataclass
class Episode:
    seed: int
    task: TaskSpec
    frames: dict[str, list] = field(default_factory=lambda: {
        "image": [], "overhead_image": [], "joint_state": [], "eef_state": [], "tick": [],
    })
    # Every executed control tick: what was commanded.
    cmd_q: list[np.ndarray] = field(default_factory=list)
    cmd_grip: list[float] = field(default_factory=list)
    cmd_pos: list[np.ndarray] = field(default_factory=list)
    cmd_rot: list[np.ndarray] = field(default_factory=list)
    cmd_phase: list[str] = field(default_factory=list)


def encode_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def decode_jpeg(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


class Runner:
    """Runs episodes of one task in one SimWorld."""

    def __init__(self, world, task: TaskSpec, image_size: int = 224, cameras=CAMERAS, render: bool = True):
        self.world = world
        self.task = task
        self.size = int(image_size)
        self.cameras = list(cameras) if render else []
        self.expert = Expert(world, task)
        m = world.model
        self.ticks_per_control = max(1, int(round(CONTROL_DT / m.opt.timestep)))
        # Gravity feed-forward for the position servos (the expert's low-level
        # controller): without it the arm hangs 2-3 mm below its targets, more than
        # the tape peg's clearance. Recorded actions stay the desired joint angles.
        self._gravity = GravityModel()
        self._kp = m.actuator_gainprm[world._arm_act, 0].copy()
        self.record_every = max(1, int(round((1.0 / RECORD_HZ) / CONTROL_DT)))

    # ---------------------------------------------------------- one tick

    def _capture(self, ep: Episode) -> None:
        w = self.world
        imgs = w.render(self.cameras, self.size, self.size) if self.cameras else {}
        f = ep.frames
        f["image"].append(imgs.get("record"))
        f["overhead_image"].append(imgs.get("overhead"))
        q = w.data.qpos[w._qpos_adr].copy()
        grip = w.state()["gripper.pos"] / 100.0
        pos, rot = self.expert.tcp_pose(q)
        f["joint_state"].append(np.concatenate([q, [grip]]))
        f["eef_state"].append(np.concatenate([pos, mat_to_rpy(rot), [grip]]))
        f["tick"].append(len(ep.cmd_q))

    def _command(self, q: np.ndarray, grip_pct: float) -> None:
        tau = self._gravity(np.asarray(q, dtype=float))
        self.world.command(np.asarray(q) + tau / self._kp, grip_pct)

    def _execute(self, ep: Episode, plan: Plan, rules: C.ContactRules, target: int,
                 hold_ref=None, record: bool = True) -> None:
        w = self.world
        for i in range(len(plan)):
            if record and len(ep.cmd_q) % self.record_every == 0:
                self._capture(ep)
            self._command(plan.q[i], plan.grip[i])
            w.step(self.ticks_per_control)
            ep.cmd_q.append(plan.q[i])
            ep.cmd_grip.append(plan.grip[i])
            ep.cmd_pos.append(plan.tcp_pos[i])
            ep.cmd_rot.append(plan.tcp_rot[i])
            ep.cmd_phase.append(plan.phase[i])
            C.refresh_contacts(w.model, w.data)
            bad = rules.violation(w.data, plan.phase[i])
            if bad:
                raise EpisodeFailed(bad)
            if hold_ref is not None and plan.phase[i] in CARRY_PHASES:
                p_rel, _ = self.expert.measured_hold(target)
                if np.linalg.norm(p_rel - hold_ref[0]) > C.SLIP_M:
                    raise EpisodeFailed(f"slipped@{plan.phase[i]}")

    # ------------------------------------------------------- an episode

    def run(self, seed: int) -> Episode:
        """One attempt on a fresh desk. Raises PlanError or EpisodeFailed."""
        w, ex, task = self.world, self.expert, self.task
        w.rng = np.random.default_rng(seed)
        w.reset(require=(task.kind,))
        ep = Episode(seed=seed, task=task)
        q0 = w.data.qpos[w._qpos_adr].copy()
        start = {k: ex.item_pose(k)[0] for k, p in enumerate(ex.desk.placements) if p.active}

        pick, target, q_lift = ex.plan_pick(q0)
        rules = C.ContactRules(ex.index, target.index)
        z0 = ex.item_pose(target.index)[0][2]
        self._execute(ep, pick, rules, target.index)

        # Did it come up with the jaws?
        if ex.item_pose(target.index)[0][2] < z0 + 0.04:
            raise EpisodeFailed("grasp_missed")
        held = ex.measured_hold(target.index)
        # Plan the place from where the item really is in the jaws, from where the arm really is.
        q_now = w.data.qpos[w._qpos_adr].copy()
        place, opt = ex.plan_place(q_now if np.max(np.abs(q_now - q_lift)) < 0.05 else q_lift, target, held)
        # Carry it over the station, then aim the last stretch again from where the
        # item has settled in the jaws on the way.
        n_carry = sum(1 for ph in place.phase if ph in ("carry_up", "carry"))
        carry = Plan()
        for i in range(n_carry):
            carry.add(place.q[i], place.grip[i], place.phase[i], place.tcp_pos[i], place.tcp_rot[i])
        self._execute(ep, carry, rules, target.index, hold_ref=held)
        held_now = ex.measured_hold(target.index)
        try:
            final = ex.replan_final(np.asarray(carry.q[-1]), target, opt, held_now)
        except PlanError as exc:
            raise EpisodeFailed(f"replan_{exc}") from exc
        self._execute(ep, final, rules, target.index, hold_ref=held)

        # Settle, unrecorded, with the arm held at ready, then judge.
        n = int(round(SETTLE_AFTER_S / CONTROL_DT))
        before = None
        for i in range(n):
            self._command(final.q[-1], final.grip[-1])
            w.step(self.ticks_per_control)
            if i == n - 6:
                before = ex.item_pose(target.index)[0]
        C.refresh_contacts(w.model, w.data)
        bad = rules.violation(w.data, "settle")
        if bad:
            raise EpisodeFailed(bad)
        pos, rot = ex.item_pose(target.index)
        speed = np.linalg.norm(pos - before) / (5 * CONTROL_DT)
        if speed > C.REST_SPEED_M_S:
            raise EpisodeFailed("target_still_moving")
        bad = C.station_check(task, pos, rot, target.item.half)
        if bad:
            raise EpisodeFailed(bad)
        for k, p0 in start.items():
            if k == target.index:
                continue
            if np.linalg.norm(ex.item_pose(k)[0] - p0) > C.DISTURB_M:
                raise EpisodeFailed("disturbed_other_item")
        return ep

    # ------------------------------------------------------------ saving

    def arrays(self, ep: Episode) -> dict[str, np.ndarray]:
        f = ep.frames
        n_cmd = len(ep.cmd_q)
        js = np.asarray(f["joint_state"], dtype=np.float32)
        es = np.asarray(f["eef_state"], dtype=np.float32)
        a_joint, a_eef, a_delta = [], [], []
        for k, tick in enumerate(f["tick"]):
            j = min(tick + self.record_every - 1, n_cmd - 1)
            grip_frac = ep.cmd_grip[j] / 100.0
            grip_bin = 0.0 if ep.cmd_grip[j] < GRIP_CLOSED_BELOW else 1.0
            a_joint.append(np.concatenate([ep.cmd_q[j], [grip_frac]]))
            rpy_t = mat_to_rpy(ep.cmd_rot[j])
            a_eef.append(np.concatenate([ep.cmd_pos[j], rpy_t, [grip_bin]]))
            q = js[k, :6].astype(float)
            pos, rot = self.expert.tcp_pose(q)
            d_rot = ep.cmd_rot[j] @ rot.T  # Base-frame rotation from now to the target.
            a_delta.append(np.concatenate([ep.cmd_pos[j] - pos, mat_to_rpy(d_rot), [grip_bin]]))
        out = {
            "joint_state": js,
            "eef_state": es,
            "action_joint": np.asarray(a_joint, dtype=np.float32),
            "action_eef": np.asarray(a_eef, dtype=np.float32),
            "action_delta": np.asarray(a_delta, dtype=np.float32),
            "phase": np.asarray([ep.cmd_phase[min(t, n_cmd - 1)] for t in f["tick"]]),
            "language_instruction": np.asarray(ep.task.instruction),
            "task": np.asarray(ep.task.name),
            "seed": np.asarray(ep.seed),
            "fps": np.asarray(RECORD_HZ),
        }
        if self.cameras:
            out["image"] = np.asarray([encode_jpeg(i) for i in f["image"]], dtype=object)
            out["overhead_image"] = np.asarray([encode_jpeg(i) for i in f["overhead_image"]], dtype=object)
        return out


def save_episode(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_episode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def record_task(
    world, task: TaskSpec, out_dir: Path, episodes: int, seed_base: int = 0, image_size: int = 224,
    render: bool = True, max_attempts: int | None = None,
) -> dict:
    """Keep running fresh desks until `episodes` perfect ones are on disk."""
    raw = Path(out_dir) / "raw" / task.name
    raw.mkdir(parents=True, exist_ok=True)
    stats_path = raw / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    reasons = Counter(stats.get("failures", {}))
    attempts = int(stats.get("attempts", 0))
    have = sorted(raw.glob("episode_*.npz"))
    kept = len(have)
    runner = Runner(world, task, image_size=image_size, render=render)
    max_attempts = max_attempts or 40 * episodes + 100
    t_start = time.time()
    logger.info("%s: %d/%d episodes on disk, %d attempts so far.", task.name, kept, episodes, attempts)
    while kept < episodes and attempts < max_attempts:
        seed = seed_base + attempts
        attempts += 1
        t0 = time.time()
        try:
            ep = runner.run(seed)
        except (PlanError, EpisodeFailed) as exc:
            key = str(exc).split(" (")[0]
            reasons[key] += 1
            logger.info("%s seed %d: discarded (%s) in %.1fs.", task.name, seed, exc, time.time() - t0)
        else:
            arrays = runner.arrays(ep)
            save_episode(raw / f"episode_{kept:05d}.npz", arrays)
            kept += 1
            logger.info("%s seed %d: kept episode %d/%d, %d frames, %.1fs.",
                        task.name, seed, kept, episodes, len(arrays["joint_state"]), time.time() - t0)
        # Every attempt: a job killed at its time limit resumes at the next seed.
        _write_stats(stats_path, task, kept, attempts, reasons, world, image_size, time.time() - t_start)
    _write_stats(stats_path, task, kept, attempts, reasons, world, image_size, time.time() - t_start)
    if kept < episodes:
        raise RuntimeError(f"{task.name}: only {kept}/{episodes} perfect episodes in {attempts} attempts; "
                           f"see {stats_path}")
    return json.loads(stats_path.read_text())


def _write_stats(path: Path, task, kept, attempts, reasons, world, image_size, seconds) -> None:
    path.write_text(json.dumps({
        "task": task.name,
        "instruction": task.instruction,
        "kept": kept,
        "attempts": attempts,
        "success_rate": round(kept / max(1, attempts), 4),
        "failures": dict(reasons.most_common()),
        "engine": world.engine_name,
        "renderer": world.renderer_name,
        "image_size": image_size,
        "record_hz": RECORD_HZ,
        "last_run_seconds": round(seconds, 1),
    }, indent=2))

