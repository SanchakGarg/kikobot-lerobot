#!/bin/bash
# Record perfect scripted episodes of the desk tasks and export them as RLDS for
# OpenVLA. One array task per desk task (12), each on its own MIG slice:
# Newton physics, MuJoCo Warp ray-traced cameras (GPU, no OpenGL).
#
#   sbatch scripts/slurm/datagen.sbatch                       # all 12 tasks, 200 episodes each
#   sbatch --array=0 scripts/slurm/datagen.sbatch             # just pen_to_pen_cup
#   EPISODES=5 sbatch --array=0 scripts/slurm/datagen.sbatch  # a quick try
#   TASKS=all sbatch --array=0 scripts/slurm/datagen.sbatch   # every task in one job, one after another
#
# Then, for the all-tasks dataset (after the array has finished):
#   sbatch --dependency=afterok:<array job id> scripts/slurm/datagen_convert.sbatch
#
# A job that hits its time limit can be resubmitted as is: it keeps the
# episodes on disk and carries on from the next seed. Run datagen_setup.sbatch
# once first. See docs/datagen-openvla.md.
#
#SBATCH --job-name=kikobot-dg
#SBATCH --account=student
#SBATCH --partition=gpu-small
#SBATCH --array=0-11
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
# No --qos / --gres: the cluster's MIG policy sets them from the partition.

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
VENV=${VENV:-$HOME/projects/kikobot-venv}
OUT=${OUT:-$HOME/datasets/kikobot_desk}
EPISODES=${EPISODES:-200}
SEED_BASE=${SEED_BASE:-0}
IMAGE_SIZE=${IMAGE_SIZE:-224}
KEEP_RAW=${KEEP_RAW:-1}            # 0: delete a task's raw .npz once its RLDS dataset is built
ENGINE=${ENGINE:-newton}           # physics: Newton (MuJoCo Warp solver on the GPU)
RENDERER=${RENDERER:-mjwarp}       # cameras: MuJoCo Warp's ray tracer (GPU, no OpenGL)
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/scratch/warp-cache}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

ALL_TASKS=(
    pen_to_pen_cup
    pencil_to_pen_cup
    marker_to_marker_cup
    highlighter_to_marker_cup
    glue_stick_to_supply_tray
    tape_roll_to_tape_peg
    scissors_to_long_tray
    eraser_to_supply_tray
    stapler_to_long_tray
    ruler_to_long_tray
    paper_clip_box_to_supply_tray
    sharpener_to_supply_tray
)
if [[ "${TASKS:-}" == "all" ]]; then
    RUN=("${ALL_TASKS[@]}")
elif [[ -n "${TASKS:-}" ]]; then
    IFS=, read -r -a RUN <<< "$TASKS"
else
    RUN=("${ALL_TASKS[${SLURM_ARRAY_TASK_ID:-0}]}")
fi

cd "$REPO"
source "$VENV/bin/activate"
echo "=== job ${SLURM_JOB_ID:-local} (array ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}) on $(hostname)"
echo "=== tasks: ${RUN[*]} | $EPISODES episodes each | out $OUT"
nvidia-smi -L || true

for TASK in "${RUN[@]}"; do
    echo "=== record $TASK"
    kikobot-datagen record --task "$TASK" --episodes "$EPISODES" --out "$OUT" \
        --seed-base "$SEED_BASE" --image-size "$IMAGE_SIZE" --engine "$ENGINE" --renderer "$RENDERER"
    echo "=== convert $TASK -> $OUT/tfds/kikobot_desk_$TASK"
    if [[ "$KEEP_RAW" == "0" ]]; then
        kikobot-datagen convert --task "$TASK" --out "$OUT" --delete-raw
    else
        kikobot-datagen convert --task "$TASK" --out "$OUT"
    fi
    cat "$OUT/raw/$TASK/stats.json"
done

if [[ "${TASKS:-}" == "all" && "$KEEP_RAW" != "0" ]]; then
    echo "=== convert all tasks -> $OUT/tfds/kikobot_desk_all"
    kikobot-datagen convert --task all --out "$OUT"
fi
du -sh "$OUT"/raw "$OUT"/tfds 2>/dev/null || true
echo "=== done"
