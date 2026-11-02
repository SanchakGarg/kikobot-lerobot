#!/bin/bash
# Build RLDS datasets from episodes already recorded by datagen.sbatch. By
# default the all-tasks dataset, kikobot_desk_all (it needs the raw episodes, so
# run datagen.sbatch with the default KEEP_RAW=1).
#
#   sbatch --dependency=afterok:<datagen array job id> scripts/slurm/datagen_convert.sbatch
#   TASK=tasks sbatch scripts/slurm/datagen_convert.sbatch          # rebuild every per-task dataset
#   TASK=pen_to_pen_cup sbatch scripts/slurm/datagen_convert.sbatch
#
# TensorFlow runs on the CPU here; the job still goes to the GPU partition,
# the only one there is.
#
#SBATCH --job-name=kikobot-dg-convert
#SBATCH --account=student
#SBATCH --partition=gpu-small
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
VENV=${VENV:-$HOME/projects/kikobot-venv}
OUT=${OUT:-$HOME/datasets/kikobot_desk}
TASK=${TASK:-all}

cd "$REPO"
source "$VENV/bin/activate"
echo "=== job ${SLURM_JOB_ID:-local} on $(hostname): convert $TASK in $OUT"
kikobot-datagen convert --task "$TASK" --out "$OUT"
for s in "$OUT"/raw/*/stats.json; do
    python -c "import json,sys; s=json.load(open(sys.argv[1])); print(f\"{s['task']:32s} {s['kept']:4d} kept / {s['attempts']:5d} desks ({100*s['success_rate']:.0f} %)\")" "$s"
done
du -sh "$OUT"/tfds/* 2>/dev/null || true
