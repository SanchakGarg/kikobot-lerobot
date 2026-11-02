# Scripted desk demonstrations for OpenVLA

`kikobot-datagen` records demonstrations of the desk-tidying tasks without
a human. A scripted expert picks one kind of item off a random desk and
puts it where it belongs, for example *"put the pen in the pen cup"*. It keeps
an episode only if every check below passes, and throws the rest away. The kept
episodes become one RLDS dataset per task, in the layout OpenVLA's `finetune.py`
reads.

On the B200 cluster it runs as a Slurm job. Physics is **NVIDIA Newton** (the
MuJoCo Warp solver on the GPU), and the cameras are **MuJoCo Warp's ray tracer**.
There is no OpenGL and no display.

- [1. The tasks](#1-the-tasks)
- [2. What "perfect" means](#2-what-perfect-means)
- [3. How the expert works](#3-how-the-expert-works)
- [4. What is recorded](#4-what-is-recorded)
- [5. On the cluster](#5-on-the-cluster)
- [6. Locally](#6-locally)
- [7. Fine-tune OpenVLA](#7-fine-tune-openvla)
- [8. Troubleshooting](#8-troubleshooting)

## 1. The tasks

There are 12 tasks. Each is its own dataset, `kikobot_desk_<task>`, and
`kikobot_desk_all` holds them all. `kikobot-datagen tasks` lists them.

| Task | Instruction | How |
|---|---|---|
| `pen_to_pen_cup` | put the pen in the pen cup | grasped near one end, turned upright over the cup, lowered tip first |
| `pencil_to_pen_cup` | put the pencil in the pen cup | 〃 |
| `marker_to_marker_cup` | put the marker in the marker cup | 〃 |
| `highlighter_to_marker_cup` | put the highlighter in the marker cup | 〃 |
| `tape_roll_to_tape_peg` | hang the tape roll on the tape holder | carried level and dropped over the post |
| `glue_stick_to_supply_tray` | put the glue stick in the supply tray | turned to lie across its compartment (the first) |
| `eraser_to_supply_tray` | put the eraser in the supply tray | 〃 (second compartment) |
| `sharpener_to_supply_tray` | put the pencil sharpener in the supply tray | 〃 (third) |
| `paper_clip_box_to_supply_tray` | put the box of paper clips in the supply tray | 〃 (third) |
| `scissors_to_long_tray` | put the scissors in the long tray | turned to lie along the tray |
| `stapler_to_long_tray` | put the stapler in the long tray | 〃 |
| `ruler_to_long_tray` | put the ruler in the long tray | 〃 |

There is no task for the sticky-note pad. At 76 mm across, it is wider than
the open jaws (53 mm).

Every episode starts from a fresh desk with its own seed. A desk has
random items, poses, stacking, colours and textures, and random room
lighting. The desk always holds at least one item of the task's kind, with
nothing stacked on it.

## 2. What "perfect" means

An episode is saved only if all of these hold. Anything else is discarded,
and the reason is counted in `stats.json`.

- **The plan holds**, before any physics runs:
  - every tick of every straight move is within 1.5 mm and 1.5° of its target
  - no joint jumps more than 3 rad/s between ticks, and none comes within 2° of a limit
  - the whole motion, with the carried item, is checked in a kinematic copy of the scene: no arm against a station, the table or another item, and no carried item against anything before it is let go
- **In the physics:**
  - the arm never touches a station, another item or the table; only the fingertips may brush the desk while grasping
  - the item comes up with the jaws
  - it never slips more than 15 mm in them between the lift and the release
- **At the end**, after the arm has gone home and 1 s of settling:
  - **Cup:** the item is upright (more than 45° from flat), with its low end down in the cup and nothing poking through the wall.
  - **Peg:** the item is round the post, within 6 mm of its centre and down on the base.
  - **Tray:** the whole item is inside the tray, in its own compartment, and lying on the floor.
  - The target is at rest.
  - No other item on the desk moved more than 15 mm.

## 3. How the expert works

The expert, `lerobot_robot_kikobot/datagen/expert.py`, has privileged state: item poses
and station geometry. It uses the arm's own IK (`kinematics.py`).

1. **Pick.** It chooses the nearest free item of the kind and tries grasps from
   above: both closing directions, pitched up to ±24° along the item, and for
   pen-like items held toward one end. It takes the first grasp that it can
   carry through to a placement as well.
2. **Lift**, then **measure** where the item really sits in the jaws.
3. **Carry** it up and over the station. Long moves are minimum-jerk in joint
   space; moves near the desk are straight lines for the TCP.
4. **Re-aim.** Over the station it measures the grip again and replans the last
   stretch from it. A pen that slid a few millimetres in the jaws on the way
   still goes in the middle of the cup.
5. **Place**, **open**, **back out** (up along the item for cups), and go **home** to ready.

The expert's low-level controller adds gravity feed-forward to the joint
targets, so the arm tracks within a fraction of a millimetre. The recorded
actions are the desired joint angles and TCP poses, without the feed-forward.

## 4. What is recorded

Frames are at **10 Hz**, with the actuators commanded at 50 Hz and physics at 500 Hz.
An episode is about 15 s, or 140–180 frames. The **raw** episodes, kept for
re-export, are `<out>/raw/<task>/episode_XXXXX.npz`, with `stats.json` beside them.

| Key | Shape | |
|---|---|---|
| `image`, `overhead_image` | T × JPEG | `record` and `overhead` cameras, 224 × 224 |
| `joint_state` | T × 7 | J1..J6 (rad), gripper 0 closed .. 1 open |
| `eef_state` | T × 7 | TCP x y z (m, base frame), roll pitch yaw (rad), gripper |
| `action_delta` | T × 7 | TCP target 0.1 s on minus the TCP now (dx dy dz, d roll pitch yaw), gripper 1 open / 0 close |
| `action_eef` | T × 7 | that TCP target, absolute |
| `action_joint` | T × 7 | joint targets, gripper 0..1 |

The **RLDS** datasets are `<out>/tfds/kikobot_desk_<task>/1.0.0/` (and `kikobot_desk_all`).
Each step holds:

- `observation/image` (primary), `observation/overhead_image` (secondary)
- `observation/EEF_state` (6), `observation/gripper_state` (1), `observation/joint_state` (7)
- `action`, which is `action_delta`, plus `action_joint`
- `language_instruction`, `reward` (1 on the last step), `discount`, `is_first`, `is_last` and `is_terminal`

Episodes are split 95 / 5 into `train` and `val`. The layout and conventions
match OpenVLA's `bridge_orig` entry, including the gripper convention (+1 = open, 0 = close).

**Size.** One episode is about 4 MB raw and about 5.5 MB as RLDS. With 200 episodes ×
12 tasks, the raw episodes, the per-task datasets and `kikobot_desk_all` come to
roughly 36 GB. That fits the student quota (100 GB). `KEEP_RAW=0` drops
the raw episodes once a task is built, about 10 GB, but then `kikobot_desk_all`
cannot be built.

## 5. On the cluster

The cluster runs Slurm Guard, so the login shell runs no Python. Everything
below is `sbatch`. The jobs use account `student` and partition `gpu-small` (a
1g.23gb MIG slice). They set no `--qos` or `--gres`, because the MIG policy fills those in. To
change the account or partition, edit the `#SBATCH` lines at the top of each script.

**0. Put the repo in your home**, for example by cloning or `scp`-ing it from your machine:

```bash
# from your laptop
scp -r kikobot-lerobot <username>@10.2.100.14:projects/
```

The `logs/` folder must exist in it (it is in git): Slurm writes the job logs there.

**1. Set up once:** a venv in `~/projects/kikobot-venv`, then a smoke test on the GPU.

```bash
cd ~/projects/kikobot-lerobot
sbatch scripts/slurm/datagen_setup.sbatch
squeue                                   # wait for it to finish
cat logs/kikobot-dg-setup-<jobid>.out    # ends with "setup OK"
```

The setup job installs this repo without its LeRobot/PyTorch dependencies, which data
generation does not need. It adds Newton, Warp, MuJoCo Warp, TensorFlow (CPU) and TFDS.
It then runs the tests (the Newton and MuJoCo Warp ones run here, on the GPU), records two
eraser episodes with Newton and `mjwarp`, builds their RLDS dataset, and writes
a preview of an episode to `~/scratch/kikobot_datagen_smoke/`. The node needs
internet access for PyPI. If the node's `python3` is older than 3.12, the job fetches 3.12 with `uv`.

**2. A quick try:** five episodes of one task.

```bash
EPISODES=5 sbatch --array=0 scripts/slurm/datagen.sbatch
cat logs/kikobot-dg-<jobid>_0.out        # per-desk log, then stats.json
```

**3. Everything:** 12 array jobs, one per task, 200 perfect episodes each.

```bash
sbatch scripts/slurm/datagen.sbatch
squeue
sacct --today
```

**4. The all-tasks dataset**, once the array is done:

```bash
sbatch --dependency=afterok:<array job id> scripts/slurm/datagen_convert.sbatch
```

Settings are environment variables in front of `sbatch`:

| Variable | Default | |
|---|---|---|
| `EPISODES` | 200 | perfect episodes per task |
| `OUT` | `~/datasets/kikobot_desk` | raw episodes and RLDS datasets |
| `VENV` | `~/projects/kikobot-venv` | from step 1 |
| `TASKS` | (the array index) | a task, a comma list, or `all` for every task in one job, in sequence, plus `kikobot_desk_all`. Use it if the guard rejects arrays: `TASKS=all sbatch --array=0 ...` |
| `SEED_BASE` | 0 | another value gives other desks |
| `IMAGE_SIZE` | 224 | camera resolution |
| `KEEP_RAW` | 1 | 0 deletes a task's raw episodes after its RLDS build |
| `ENGINE`, `RENDERER` | `newton`, `mjwarp` | keep these on the cluster |

A job killed at its time limit (12 h) can be **resubmitted unchanged**. It keeps
the episodes on disk and continues from the next seed. Each task's desk seeds are its own
(`SEED_BASE + 1,000,000 × task index + attempt`), so reruns are reproducible.

## 6. Locally

Use the same commands, with MuJoCo on the CPU and OpenGL cameras for a quick look:

```bash
pip install -e ".[openvla]"                    # into your LeRobot venv, or any 3.12 venv
kikobot-datagen record --task pen --episodes 3 --out datasets/kikobot_desk --engine mujoco --renderer opengl
kikobot-datagen preview --task pen --out datasets/kikobot_desk --episode 0    # mp4 (or a PNG strip)
kikobot-datagen convert --task pen_to_pen_cup --out datasets/kikobot_desk
pytest tests/test_datagen.py
```

`record` refuses `--engine newton` or `--renderer mjwarp` without a CUDA GPU. It never
falls back on its own, so a cluster run cannot quietly record with the wrong engine.

## 7. Fine-tune OpenVLA

`scripts/openvla/kikobot_oxe.py` adds the datasets to OpenVLA's registries:

- `kikobot_desk_<task>` and `kikobot_desk_all`: the `bridge_orig` layout, `POS_EULER` state, `EEF_POS` actions
- the mixture `kikobot_desk_all_tasks`

`scripts/openvla/finetune_kikobot.py` registers them, then runs OpenVLA's own
`vla-scripts/finetune.py` with your arguments:

```bash
git clone https://github.com/openvla/openvla ../openvla     # and install it as its README says
OPENVLA_DIR=../openvla torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    scripts/openvla/finetune_kikobot.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir ~/datasets/kikobot_desk/tfds \
    --dataset_name kikobot_desk_pen_to_pen_cup \
    --run_root_dir runs --batch_size 8 --use_lora True
```

You can also paste the same entries into OpenVLA's `configs.py`, `transforms.py` and `mixtures.py`.
The `secondary` camera is there for a policy that loads it. OpenVLA itself loads
only `primary` by default.

At deployment the policy gives a 7-D EEF delta. Add it to the TCP pose, solve IK
(`ArmKinematics.solve`) and send the joints. The gripper is 1 = open, 0 = close.

## 8. Troubleshooting

| What you see | Why, and what to do |
|---|---|
| `--engine newton: no CUDA GPU visible to Warp` | Not in a GPU job, or the MIG slice is not visible. Check `nvidia-smi -L` in the job log. |
| `Command 'python3' is not permitted` | Run Python only inside `sbatch` or `srun`, never at the login shell. |
| `only k/200 perfect episodes in N attempts` | The task gave up after 40 attempts per wanted episode. Read the failure counts in `stats.json`. The same counts help with tuning: `ik_*` means out of reach, `plan_arm_touched_other_item` means a crowded desk, and `slipped` or `grasp_missed` mean grip trouble. |
| Fewer kept episodes on one task than on others | This is expected. Success rates per desk run from about 40 % (the marker cup, at the edge of the arm's reach) to 90 % (the stapler). Only kept episodes are saved. |
| Setup cannot reach PyPI | Build the venv where there is internet, with the same Python, and copy it to `~/projects/kikobot-venv`. |
| `OOM` | Raise `--mem` in the script. Recording needs a few GB of host RAM, and conversion more for `kikobot_desk_all`. |

**Newton and this scene.** Four things in how Newton imports an MJCF scene
had to be corrected for the desk to simulate as MuJoCo does. All four are in
`NewtonEngine` (`sim_backends.py`) and are checked by `tests/test_sim_backends.py`:

- free-body poses are synced in full (only x had been)
- the contact gap is small (Newton's default of 10 cm flooded the contact buffer)
- Euler integration and multi-point CCD are kept (implicitfast with single-point jaw contacts let held items slide out)
- the midphase is off (it went stale when Newton re-based the bodies)
