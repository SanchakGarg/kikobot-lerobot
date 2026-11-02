#!/bin/bash
# One-time setup for kikobot-datagen on the B200 cluster (Slurm Guard): a Python
# venv with this repo, Newton + MuJoCo Warp (GPU physics and ray-traced
# cameras, no OpenGL) and TensorFlow Datasets (CPU, for the RLDS export), then
# a smoke test on the GPU: tests, one recorded episode, one RLDS build.
#
#   cd /raid/home/$USER/projects/kikobot-lerobot
#   sbatch scripts/slurm/datagen_setup.sbatch
#   squeue                      # then: cat logs/kikobot-dg-setup-<jobid>.out
#
# Python only runs inside a job on this cluster, so the venv is built here,
# not at the login shell. The compute node needs internet (PyPI) for this job.
# See docs/datagen-openvla.md.
#
#SBATCH --job-name=kikobot-dg-setup
#SBATCH --account=student
#SBATCH --partition=gpu-small
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# No --qos / --gres: the cluster's MIG policy sets them from the partition.

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
VENV=${VENV:-$HOME/projects/kikobot-venv}
SMOKE=${SMOKE:-$HOME/scratch/kikobot_datagen_smoke}
PYTHON=${PYTHON:-python3}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$HOME/scratch/pip-cache}
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/scratch/warp-cache}

cd "$REPO"
echo "=== job ${SLURM_JOB_ID:-local} on $(hostname), partition ${SLURM_JOB_PARTITION:-?}, account ${SLURM_JOB_ACCOUNT:-?}"
nvidia-smi -L || true

# ------------------------------------------------------------ the venv
# The package needs Python >= 3.12. If the node's python3 is older, get one with uv.
if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
    echo "=== $("$PYTHON" --version 2>&1) is older than 3.12: using uv for a 3.12 venv"
    "$PYTHON" -m pip install --user --quiet uv
    UV="$("$PYTHON" -m site --user-base)/bin/uv"
    [[ -d "$VENV" ]] || "$UV" venv --python 3.12 "$VENV"
    "$UV" pip install --python "$VENV/bin/python" --quiet pip
elif [[ ! -d "$VENV" ]]; then
    "$PYTHON" -m venv "$VENV"
fi
source "$VENV/bin/activate"
python --version
pip install --quiet --upgrade pip

# --no-deps: the package itself depends on LeRobot (and so PyTorch) for the real
# arm and teleop, which data generation does not use. Its sim needs only these.
pip install --quiet --no-deps -e .
pip install --quiet "mujoco>=3.15" "numpy>=2" pyzmq trimesh pytest \
    "newton[sim,importers]>=1.7" "warp-lang>=1.17" \
    "tensorflow-cpu>=2.18" "tensorflow-datasets>=4.9" importlib_resources pillow imageio imageio-ffmpeg

python - <<'PY'
import mujoco, numpy, newton, warp, mujoco_warp, tensorflow_datasets as tfds
print("mujoco", mujoco.__version__, "| numpy", numpy.__version__, "| newton", newton.__version__,
      "| warp", warp.__version__, "| tfds", tfds.__version__)
warp.init()
assert warp.is_cuda_available(), "Warp sees no CUDA device: is this a GPU job?"
print("CUDA devices:", [str(d) for d in warp.get_cuda_devices()])
PY

# ------------------------------------------------------------ smoke test
echo "=== tests (the Newton and MuJoCo Warp ones run here, on the GPU)"
python -m pytest -q tests/test_sim_backends.py tests/test_datagen.py tests/test_desk_scene.py

echo "=== one episode: Newton physics, MuJoCo Warp cameras"
rm -rf "$SMOKE"
kikobot-datagen record --task eraser_to_supply_tray --episodes 2 --out "$SMOKE" \
    --engine newton --renderer mjwarp --max-attempts 40
kikobot-datagen convert --task eraser_to_supply_tray --out "$SMOKE"
kikobot-datagen preview --task eraser_to_supply_tray --out "$SMOKE" --episode 0
cat "$SMOKE/raw/eraser_to_supply_tray/stats.json"
ls -la "$SMOKE/tfds"
echo "=== setup OK: venv $VENV. Next: sbatch scripts/slurm/datagen.sbatch"
