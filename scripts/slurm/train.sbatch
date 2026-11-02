#!/bin/bash
# Train a LeRobot policy on a Kiko-Bot dataset on one SLURM GPU (e.g. a B200).
#
#   sbatch scripts/slurm/train.sbatch
#   DATASET=kikobot_isaac POLICY=diffusion STEPS=200000 sbatch scripts/slurm/train.sbatch
#   RESUME=outputs/train/act_kikobot_isaac sbatch scripts/slurm/train.sbatch
#
# Edit the #SBATCH lines for your cluster (partition, account, time limit).
# See docs/slurm-training.md.
#
#SBATCH --job-name=kikobot-train
#SBATCH --partition=gpu            # <- your GPU partition
##SBATCH --account=YOUR_ACCOUNT    # <- uncomment if your cluster needs one
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16         # video decoding for the data loader
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# ---------------------------------------------------------------- settings
LEROBOT_DIR=${LEROBOT_DIR:-$HOME/lerobot}       # LeRobot checkout with its .venv
DATASET=${DATASET:-kikobot_isaac}               # folder under datasets/
DATASET_ROOT=${DATASET_ROOT:-$PWD/datasets/$DATASET}
REPO_ID=${REPO_ID:-local/$DATASET}               # or your-hf-user/kikobot_isaac to pull from the Hub
POLICY=${POLICY:-act}                            # act | diffusion | smolvla | ...
STEPS=${STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-32}
SAVE_FREQ=${SAVE_FREQ:-10000}
JOB_NAME=${JOB_NAME:-${POLICY}_${DATASET}}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/outputs/train/$JOB_NAME}
WANDB=${WANDB:-false}
RESUME=${RESUME:-}                               # an earlier OUTPUT_DIR to continue

# Uncomment what your cluster needs, e.g.:
# module load cuda/12.8 ffmpeg

# ------------------------------------------------------------------- run
mkdir -p logs
source "$LEROBOT_DIR/.venv/bin/activate"
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "Host $(hostname), job ${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available: is this a GPU node, and torch a CUDA build?"
cap = torch.cuda.get_device_capability()
arch = f"sm_{cap[0]}{cap[1]}"
print("torch", torch.__version__, "on", torch.cuda.get_device_name(), arch)
if arch not in torch.cuda.get_arch_list():
    raise SystemExit(f"This torch build has no kernels for {arch}; install a cu128 (or newer) build.")
PY

WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))

if [[ -n "$RESUME" ]]; then
    srun lerobot-train \
        --config_path="$RESUME/checkpoints/last/pretrained_model/train_config.json" \
        --resume=true
else
    srun lerobot-train \
        --dataset.repo_id="$REPO_ID" \
        --dataset.root="$DATASET_ROOT" \
        --policy.type="$POLICY" \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --output_dir="$OUTPUT_DIR" \
        --job_name="$JOB_NAME" \
        --steps="$STEPS" \
        --batch_size="$BATCH_SIZE" \
        --save_freq="$SAVE_FREQ" \
        --num_workers="$WORKERS" \
        --wandb.enable="$WANDB"
fi
