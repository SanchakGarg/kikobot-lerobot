"""The scripted expert, its perfection checks, the recorder and the RLDS export.

The expert tests run on MuJoCo with no cameras (any machine); the Newton
ones only where Newton is installed, the RLDS one only with TensorFlow Datasets.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lerobot_robot_kikobot.datagen import checks as C
from lerobot_robot_kikobot.datagen.expert import PlanError
from lerobot_robot_kikobot.datagen.geometry import mat_to_rpy, rot_z, rpy_to_mat
from lerobot_robot_kikobot.datagen.record import (
    EpisodeFailed, Runner, encode_jpeg, load_episode, save_episode,
)
from lerobot_robot_kikobot.datagen.tasks import EXCLUDED_KINDS, TASK_NAMES, TASKS, get_task
from lerobot_robot_kikobot.desk import layout as L
from lerobot_robot_kikobot.sim_server import SimWorld


def test_every_kind_but_the_note_pad_has_a_task():
    kinds = {t.kind for t in TASKS.values()}
    assert kinds == set(L.KINDS) - set(EXCLUDED_KINDS)
    assert len(TASK_NAMES) == 12
    assert get_task("pen").instruction == "put the pen in the pen cup"
    assert get_task("tape_roll").station == "tape_peg"
    for t in TASKS.values():
        assert t.kind in t.station_spec.accepts


@pytest.mark.parametrize("kind", ["pen", "eraser", "tape_roll", "ruler"])
def test_a_required_item_is_on_the_desk_and_free(kind):
    pool = L.default_pool()
    for seed in range(20):
        desk = L.sample_desk(np.random.default_rng(seed), pool, require=(kind,))
        mine = [p for p in desk if p.kind == kind and p.active]
        # At least one of them lies on the desk itself with nothing stacked on it.
        free = [m for m in mine if m.layer == 0 and not any(m.name in p.on_top_of for p in desk)]
        assert free, seed


def test_desks_without_require_are_unchanged():
    pool = L.default_pool()
    a = L.sample_desk(np.random.default_rng(4), pool)
    b = L.sample_desk(np.random.default_rng(4), pool, require=None)
    assert [(p.name, p.active, p.pos) for p in a] == [(p.name, p.active, p.pos) for p in b]


def test_rpy_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        rpy = rng.uniform([-3, -1.5, -3], [3, 1.5, 3])
        assert np.allclose(mat_to_rpy(rpy_to_mat(rpy)), rpy, atol=1e-9)


# ------------------------------------------------------------- the checks


def _upright(axis_tilt_deg: float = 0.0) -> np.ndarray:
    """An item frame with its long axis (X) pointing down, leaning by `axis_tilt_deg`."""
    t = math.radians(90.0 - axis_tilt_deg)
    x = np.array([math.cos(t), 0.0, -math.sin(t)])
    z = np.array([math.sin(t), 0.0, math.cos(t)])
    return np.column_stack([x, np.cross(z, x), z])


def test_cup_check():
    task = get_task("pen")
    cx, cy = L.PEN_CUP.xy
    half = L.KINDS["pen"].half
    assert C.station_check(task, np.array([cx, cy, 0.075]), _upright(10), half) is None
    assert C.station_check(task, np.array([cx + 0.1, cy, 0.075]), _upright(10), half) == "not_in_cup"
    flat = np.eye(3)
    assert C.station_check(task, np.array([cx, cy, 0.075]), flat, half) is not None


def test_peg_check():
    task = get_task("tape_roll")
    x, y = L.TAPE_PEG.xy
    half = L.KINDS["tape_roll"].half
    assert C.station_check(task, np.array([x, y, 0.0145]), np.eye(3), half) is None
    assert C.station_check(task, np.array([x + 0.01, y, 0.0145]), np.eye(3), half) == "not_on_peg"
    assert C.station_check(task, np.array([x, y, 0.07]), np.eye(3), half) == "not_down_on_peg"


def test_tray_and_compartment_checks():
    task = get_task("eraser")  # supply tray, compartment 1
    st = task.station_spec
    half = L.KINDS["eraser"].half
    along = rot_z(st.yaw + math.pi / 2)  # across the tray: along a compartment
    good = C.compartment_center(st, 1) + np.array([0, 0, 0.004 + half[2]])
    assert C.station_check(task, good, along, half) is None
    wrong = C.compartment_center(st, 0) + np.array([0, 0, 0.004 + half[2]])
    assert C.station_check(task, wrong, along, half) == "wrong_compartment"
    assert C.station_check(task, good + [0.3, 0, 0], along, half) == "not_in_tray"
    assert C.station_check(task, good + [0, 0, 0.05], along, half) == "not_down_in_tray"


def test_arm_in_a_station_is_a_violation():
    w = SimWorld("desk_sort", seed=0, engine="mujoco")
    index = C.SceneIndex(w.model, w._desk.pool)
    rules = C.ContactRules(index, target=0)
    C.refresh_contacts(w.model, w.data)
    assert rules.violation(w.data, "carry") is None  # At ready: touching nothing.
    from lerobot_robot_kikobot.kinematics import ArmKinematics
    from lerobot_robot_kikobot.datagen.expert import down_rot

    kin = ArmKinematics()
    # The jaws straight down onto the supply tray's +X wall, fingertips below its top.
    st = L.TRAY
    wall = np.array(st.xy) + (st.half[0] - 0.002) * np.array([math.cos(st.yaw), math.sin(st.yaw)])
    target = np.array([wall[0], wall[1], 0.015 + 0.0472])
    q = kin.solve(kin.ready_q(), target, down_rot(np.array([-math.sin(st.yaw), math.cos(st.yaw), 0.0])), 400)
    assert np.linalg.norm(kin.fk(q)[0] - target) < 0.002
    w.pose(q, 100.0)
    C.refresh_contacts(w.model, w.data)
    assert rules.violation(w.data, "carry").startswith("arm_hit_supply_tray")


# --------------------------------------------------------- the expert, MuJoCo


@pytest.fixture(scope="module")
def world():
    return SimWorld("desk_sort", seed=0, engine="mujoco")


@pytest.mark.parametrize("task", ["pen_to_pen_cup", "tape_roll_to_tape_peg", "eraser_to_supply_tray",
                                  "ruler_to_long_tray"])
def test_expert_does_the_task(world, task):
    runner = Runner(world, get_task(task), render=False)
    reasons = []
    for seed in range(12):
        try:
            ep = runner.run(seed)
        except (PlanError, EpisodeFailed) as exc:
            reasons.append(str(exc))
            continue
        arrays = runner.arrays(ep)
        n = len(arrays["joint_state"])
        assert 60 < n < 400
        for key in ("eef_state", "action_joint", "action_eef", "action_delta"):
            assert arrays[key].shape == (n, 7), key
        # The gripper closes once and opens once.
        g = arrays["action_delta"][:, 6]
        assert g[0] == 1.0 and g[-1] == 1.0 and (g == 0.0).any()
        assert np.count_nonzero(np.diff(g)) == 2
        # The deltas take the measured TCP to the next one, give or take the servo lag.
        step = arrays["eef_state"][1:, :3] - arrays["eef_state"][:-1, :3]
        assert np.abs(step - arrays["action_delta"][:-1, :3]).max() < 0.01
        return
    pytest.fail(f"{task}: no perfect episode in 12 desks: {reasons}")


def test_episode_round_trip(tmp_path):
    img = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
    arrays = {
        "joint_state": np.zeros((3, 7), np.float32),
        "image": np.asarray([encode_jpeg(img)] * 3, dtype=object),
        "language_instruction": np.asarray("put the pen in the pen cup"),
    }
    save_episode(tmp_path / "episode_00000.npz", arrays)
    back = load_episode(tmp_path / "episode_00000.npz")
    assert str(back["language_instruction"]) == "put the pen in the pen cup"
    assert len(back["image"]) == 3 and back["joint_state"].shape == (3, 7)


# ------------------------------------------------------------------ Newton


def test_expert_does_the_task_on_newton():
    pytest.importorskip("newton")
    w = SimWorld("desk_sort", seed=0, engine="newton")
    runner = Runner(w, get_task("eraser_to_supply_tray"), render=False)
    ok = 0
    for seed in range(300, 308):
        try:
            runner.run(seed)
            ok += 1
        except (PlanError, EpisodeFailed):
            pass
    assert ok >= 3


# -------------------------------------------------------------------- RLDS


def _fake_episode(path, task: str, n: int = 6, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    img = lambda: encode_jpeg((rng.random((224, 224, 3)) * 255).astype(np.uint8))  # noqa: E731
    act = np.zeros((n, 7), np.float32)
    act[:, 6] = 1.0
    save_episode(path, {
        "joint_state": rng.random((n, 7)).astype(np.float32),
        "eef_state": rng.random((n, 7)).astype(np.float32),
        "action_joint": rng.random((n, 7)).astype(np.float32),
        "action_eef": act,
        "action_delta": act,
        "image": np.asarray([img() for _ in range(n)], dtype=object),
        "overhead_image": np.asarray([img() for _ in range(n)], dtype=object),
        "language_instruction": np.asarray(TASKS[task].instruction),
        "task": np.asarray(task),
        "seed": np.asarray(seed),
        "fps": np.asarray(10.0),
    })


def test_rlds_export(tmp_path):
    tfds = pytest.importorskip("tensorflow_datasets")
    pytest.importorskip("importlib_resources")
    from lerobot_robot_kikobot.datagen.rlds.kikobot_desk import build, dataset_name

    task = "pen_to_pen_cup"
    raw = tmp_path / "raw" / task
    raw.mkdir(parents=True)
    for k in range(3):
        _fake_episode(raw / f"episode_{k:05d}.npz", task, seed=k)
    build(tmp_path, task)
    # Read back the way OpenVLA does: by name and data_dir, without our builder code.
    b = tfds.builder_from_directory(str(tmp_path / "tfds" / dataset_name(task) / "1.0.0"))
    total = sum(s.num_examples for s in b.info.splits.values())
    assert total == 3
    split = next(iter(b.info.splits))
    ep = next(iter(b.as_dataset(split=split)))
    steps = list(ep["steps"].as_numpy_iterator())
    assert len(steps) == 6
    s = steps[0]
    assert s["observation"]["image"].shape == (224, 224, 3)
    assert s["observation"]["EEF_state"].shape == (6,) and s["observation"]["gripper_state"].shape == (1,)
    assert s["action"].shape == (7,) and s["is_first"] and steps[-1]["is_last"]
    assert s["language_instruction"] == b"put the pen in the pen cup"


def test_openvla_registry_lists_every_task():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "openvla" / "kikobot_oxe.py"
    spec = importlib.util.spec_from_file_location("kikobot_oxe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TASKS == TASK_NAMES
    assert "kikobot_desk_all" in mod.DATASETS
