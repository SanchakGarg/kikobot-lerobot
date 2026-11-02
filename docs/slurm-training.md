# Training on a SLURM cluster (B200)

The cluster only trains. Isaac Sim is not installed there and doesn't need
to be: `lerobot-train` reads the recorded dataset (joint values and MP4
video), not the simulator. B200s also cannot render Isaac Sim cameras (no RT
cores), so recording always happens on the RTX laptop.

```
laptop: record in Isaac ──dataset──> cluster: sbatch train ──checkpoint──> laptop: rollout in Isaac / real arm
```

## 1. One-time setup on the cluster

Log in to a login node. LeRobot needs Python 3.12, and the B200
(compute capability 10.0, `sm_100`) needs a PyTorch build for CUDA 12.8 or
newer.

```bash
# Python 3.12: use the cluster's module, or uv if there is none.
module avail python 2>&1 | grep -i 3.12        # e.g. module load python/3.12
# or: curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12

# Run in the folder that should hold LeRobot and this repo side by side.
git clone https://github.com/huggingface/lerobot.git
cd lerobot
python3.12 -m venv .venv          # uv: uv venv --python 3.12 .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -e ".[training]"

# The training scripts and this repo's sbatch file.
git clone https://github.com/SanchakGarg/kikobot-lerobot.git ../kikobot-lerobot
```

The plugin itself is not needed to train, so there is no `pip install` of
`kikobot-lerobot` here.

**Video decoding** needs FFmpeg's shared libraries (for `torchcodec`). If
training fails with an FFmpeg or `libavutil` error, `module load ffmpeg`, or
install it into the venv's prefix with conda
(`conda install -c conda-forge "ffmpeg<8"`).

**Check the GPU stack on a GPU node,** with a short interactive job, from
inside `lerobot`:

```bash
srun --gres=gpu:1 --partition=gpu --time=00:10:00 --pty \
  .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_arch_list())"
```

`sm_100` must be in the list. If it is not, reinstall torch from the
`cu128` (or newer) index.

## 2. Copy the dataset to the cluster

**Option A, straight copy.** Use PowerShell on the laptop; Windows 10/11 has
`scp` built in:

```powershell
# inside kikobot-lerobot on the laptop
scp -r datasets\kikobot_isaac user@cluster:CLUSTER_REPO/datasets/
```

`CLUSTER_REPO` is the path of `kikobot-lerobot` on the cluster (run `pwd`
inside it there). For later updates, `rsync -av` from WSL or Git Bash only sends what changed.

**Option B, via the Hugging Face Hub.** This is handy across several
machines. The dataset stays private.

```powershell
$W = Split-Path $PWD              # inside kikobot-lerobot
& "$W\lerobot\.venv\Scripts\hf.exe" auth login
& "$W\lerobot\.venv\Scripts\hf.exe" upload YOUR_HF_USER/kikobot_isaac `
  datasets\kikobot_isaac --repo-type dataset --private
```

Then on the cluster, set `REPO_ID=YOUR_HF_USER/kikobot_isaac` (below) and
log in once with `../lerobot/.venv/bin/hf auth login` from inside the repo. The data downloads when
the job starts.

## 3. Submit the training job

`scripts/slurm/train.sbatch` is ready to use. First edit its `#SBATCH`
lines for your cluster: the partition name, an account if your cluster
needs one, and the time limit. Then:

```bash
cd CLUSTER_REPO                    # kikobot-lerobot on the cluster
sbatch scripts/slurm/train.sbatch
```

Every setting has a default and can be overridden from the command line:

```bash
# Diffusion policy, longer run:
POLICY=diffusion STEPS=200000 sbatch scripts/slurm/train.sbatch

# A different dataset, from the Hub, bigger batches, with W&B logging:
DATASET=kikobot_isaac_cylinder REPO_ID=you/kikobot_isaac_cylinder \
  BATCH_SIZE=64 WANDB=true sbatch scripts/slurm/train.sbatch
```

| Variable | Default | Meaning |
|---|---|---|
| `DATASET` | `kikobot_isaac` | Folder under `datasets/` |
| `REPO_ID` | `local/$DATASET` | Dataset id. Use `user/name` to pull from the Hub |
| `POLICY` | `act` | `act`, `diffusion`, `smolvla`, ... |
| `STEPS` | 100000 | Training steps |
| `BATCH_SIZE` | 32 | A B200 (192 GB) fits far more. Raise it, with a higher learning rate if you do |
| `SAVE_FREQ` | 10000 | Checkpoint every N steps |
| `OUTPUT_DIR` | `outputs/train/<policy>_<dataset>` | Where checkpoints go |
| `WANDB` | false | Log to Weights & Biases (run `wandb login` once on the login node) |
| `RESUME` | empty | An earlier `OUTPUT_DIR` to continue from |
| `LEROBOT_DIR` | `../lerobot`, next to this repo | LeRobot checkout with `.venv` |

**Watch the job:**

```bash
squeue --me                          # queued / running
tail -f logs/kikobot-train-<jobid>.out
scancel <jobid>                      # stop it
```

The log first prints the GPU and torch's architecture check, then a loss
line every 200 steps. ACT's loss should fall steadily for the first 10–20k
steps.

**Hit the time limit?** Resume from the last checkpoint:

```bash
RESUME=outputs/train/act_kikobot_isaac sbatch scripts/slurm/train.sbatch
```

## 4. Bring the policy back

Only the final `pretrained_model` folder is needed, about 200 MB for ACT. In
PowerShell on the laptop:

```powershell
# inside kikobot-lerobot on the laptop
mkdir outputs\train\act_kikobot_isaac -Force
scp -r user@cluster:CLUSTER_REPO/outputs/train/act_kikobot_isaac/checkpoints/last/pretrained_model `
  outputs\train\act_kikobot_isaac\
```

Then run it in Isaac Sim, or on the real arm, with the commands in
[Record, train, evaluate §4–5](isaac-sim-record-and-train.md#4-run-the-policy-in-isaac-sim),
using `--policy.path=outputs/train/act_kikobot_isaac/pretrained_model`.

**Via the Hub instead:** upload the folder from the cluster with
`hf upload YOUR_HF_USER/act_kikobot_isaac outputs/train/act_kikobot_isaac/checkpoints/last/pretrained_model --private`,
and pass `--policy.path=YOUR_HF_USER/act_kikobot_isaac` on the laptop.
