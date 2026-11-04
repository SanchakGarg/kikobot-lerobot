# Record, train and evaluate with Isaac Sim

The loop is: record gamepad demonstrations in Isaac Sim, train a policy on
them (on the laptop or [on the cluster](slurm-training.md)), then run the
policy back in Isaac Sim and, if it works, on the real arm.

Open each terminal inside `kikobot-lerobot` and set the folders as in
[teleop](isaac-sim-teleop.md):

```powershell
$W = Split-Path $PWD; $I = "$W\isaacsim-env\Scripts"; $V = "$W\lerobot\.venv\Scripts"
```

Start the server first, and leave it running for all of the below:

```powershell
cd $W\kikobot-lerobot
& "$I\python.exe" isaac\kikobot_isaac_server.py
```

## 1. Record demonstrations

Run in a second terminal, from the repo:

```powershell
cd $W\kikobot-lerobot
& "$V\lerobot-record.exe" `
  --robot.type=kikobot_isaac `
  --teleop.type=kikobot_gamepad `
  --dataset.repo_id=local/kikobot_isaac `
  --dataset.root=datasets/kikobot_isaac `
  --dataset.single_task="Pick up the red cube and place it on the green pad" `
  --dataset.num_episodes=50 `
  --dataset.episode_time_s=40 `
  --dataset.reset_time_s=10 `
  --dataset.push_to_hub=false `
  --display_data=true
```

For each episode:

1. **Do the task with the gamepad.** Recording runs for `episode_time_s`. To
   end the episode early once the cube is on the pad, press **→** (right
   arrow) on the keyboard.
2. **Reset the scene.** During the reset period (`reset_time_s`), press
   **Back / View** on the gamepad. The arm goes home, and the cube and pad
   respawn somewhere new for the next episode. Press → to skip the rest of
   the reset period.
3. **If an episode went wrong,** press **←** to throw it away and record it
   again. **Esc** stops the session; everything recorded so far is saved.

Tips for a good dataset:

- **Do the task the same way each time,** for example approach from above,
  close, lift, move, lower, open. Consistent demos train far better than
  varied ones.
- **Each reset gives a new cube and pad position.** That spread is what
  makes the policy generalize.
- **Aim for 50 or more episodes.** ACT usually starts working on this task
  at 30–50 and gets reliable with more.
- **Add to an existing dataset** by running the same command with
  `--resume=true`.
- **A repeatable set of starting positions:** start the server with `--seed 0`.
- **Other tasks:** restart the server with `--scene cylinder` (stand the
  cylinder up on the pad) and use its own `--dataset.repo_id` and
  `--dataset.single_task`.

Recording on a different machine from the server (see
[setup](isaac-sim-setup.md#remote-server-optional))? Add
`--robot.image_encoding=jpeg`. JPEG frames are re-encoded into the
dataset's video, so nothing else changes.

## 2. Look at what you recorded

```powershell
& "$V\lerobot-dataset-viz.exe" --repo-id local/kikobot_isaac `
  --root datasets/kikobot_isaac --episode-index 0
```

Replay an episode's actions in the sim. This is a good check that the
actions alone reproduce the demo:

```powershell
& "$V\lerobot-replay.exe" --robot.type=kikobot_isaac `
  --dataset.repo_id=local/kikobot_isaac --dataset.root=datasets/kikobot_isaac `
  --dataset.episode=0
```

The arm moves the same way. The cube may land somewhere else, because a
replay spawns a new scene. Start the server with the recording's `--seed` to match it.

## 3. Train

**On the cluster (recommended for a B200):** see
[Training on SLURM](slurm-training.md).

**On the laptop:** an RTX 4060 with 8 GB VRAM trains ACT at batch size 8.
That takes a few hours for 100k steps. Stop the Isaac Sim server first, so
the training has the whole GPU.

```powershell
& "$V\lerobot-train.exe" `
  --dataset.repo_id=local/kikobot_isaac `
  --dataset.root=datasets/kikobot_isaac `
  --policy.type=act `
  --policy.device=cuda `
  --policy.push_to_hub=false `
  --output_dir=outputs/train/act_kikobot_isaac `
  --job_name=act_kikobot_isaac `
  --batch_size=8 `
  --steps=100000 `
  --save_freq=10000 `
  --wandb.enable=false
```

- **Other policies:** `--policy.type=diffusion` (slower to train, smoother
  motion) or `--policy.type=smolvla` (needs a large GPU, use the cluster).
- **Continue a stopped run:**
  `--config_path=outputs/train/act_kikobot_isaac/checkpoints/last/pretrained_model/train_config.json --resume=true`.
- **Log to Weights & Biases:** run `wandb login` once, then pass
  `--wandb.enable=true`.

Checkpoints land in `outputs/train/<job>/checkpoints/<step>/pretrained_model`;
`last` points at the newest one.

## 4. Run the policy in Isaac Sim

With the server running:

```powershell
& "$V\lerobot-rollout.exe" `
  --strategy.type=base `
  --policy.path=outputs/train/act_kikobot_isaac/checkpoints/last/pretrained_model `
  --robot.type=kikobot_isaac `
  --task="Pick up the red cube and place it on the green pad" `
  --duration=60 `
  --display_data=true
```

Each run starts from a fresh scene. To score the policy over many trials and
keep the rollouts as a dataset, use the episodic strategy with the gamepad.
During each reset period, press **Back** to respawn the scene. You can also
take over with the sticks:

```powershell
& "$V\lerobot-rollout.exe" `
  --strategy.type=episodic `
  --policy.path=outputs/train/act_kikobot_isaac/checkpoints/last/pretrained_model `
  --robot.type=kikobot_isaac `
  --teleop.type=kikobot_gamepad `
  --dataset.repo_id=local/eval_act_kikobot_isaac `
  --dataset.root=datasets/eval_act_kikobot_isaac `
  --dataset.single_task="Pick up the red cube and place it on the green pad" `
  --dataset.num_episodes=20 --dataset.episode_time_s=40 --dataset.reset_time_s=10 `
  --dataset.push_to_hub=false
```

Count the successes while it runs, or afterwards with `lerobot-dataset-viz`.
With the server started with a fixed `--seed` (e.g. `--seed 1000`), two
policies are scored on the same cube and pad positions.

## 5. Run the policy on the real arm

The real arm takes the same actions. Its observations must match the
training data: the same joint keys, and a camera with the same name
(`record`), resolution and roughly the same view. In the sim, that view is
0.78 m in front of the base, 0.34 m up, looking at the base.

```powershell
& "$V\lerobot-rollout.exe" `
  --strategy.type=base `
  --policy.path=outputs/train/act_kikobot_isaac/checkpoints/last/pretrained_model `
  --robot.type=kikobot_follower --robot.port=COM3 --robot.id=kikobot `
  --robot.cameras="{record: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
  --task="Pick up the red cube and place it on the green pad" `
  --duration=60
```

Keep a hand near the power switch the first time. A policy trained only in
sim has never seen a real camera image. Expect it to reach toward the cube
but grasp unreliably, until you add some real demonstrations
(`lerobot-record` with `kikobot_follower`, [real arm §6](real-arm.md#6-recording-on-the-real-arm)) to the training set.

## Mixing data from MuJoCo, Isaac and the real arm

All three record the same features, so their datasets can be trained on
together.
- **Merging:** `lerobot-edit-dataset` can merge datasets into one.
- **Training on Isaac data only:** keep the MuJoCo and Isaac sets separate
  and pass only the Isaac one.
- **Why sim data helps:** Isaac's RTX images are closer to a real camera than
  MuJoCo's, so for sim-to-real, Isaac demos plus a few real ones usually beat
  MuJoCo demos.
