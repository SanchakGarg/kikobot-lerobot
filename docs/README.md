# Kiko-Bot docs

The main [README](../README.md) covers the MuJoCo sim and the real arm. The
guides here add **NVIDIA Isaac Sim with the Newton physics engine** and
**training on a SLURM GPU cluster**.

| Guide | What it covers |
|---|---|
| [Isaac Sim setup](isaac-sim-setup.md) | Install Isaac Sim 6 and LeRobot side by side, convert the arm to USD, check it runs |
| [Teleop in Isaac Sim](isaac-sim-teleop.md) | Start the sim server, drive the arm with the gamepad, all options |
| [Record, train, evaluate](isaac-sim-record-and-train.md) | `lerobot-record` in Isaac, inspect and replay datasets, train, run the policy in Isaac and on the real arm |
| [Training on SLURM](slurm-training.md) | Move a dataset to the cluster, train on a B200 with `sbatch`, bring the policy back |
| [Troubleshooting](isaac-sim-troubleshooting.md) | Errors and fixes for the Isaac Sim side |

## The typical setup

```
 Windows laptop (RTX GPU)                               SLURM cluster (B200)
 ┌────────────────────────────────────────────┐          ┌─────────────────────┐
 │ Isaac Sim venv          LeRobot venv        │ dataset  │ LeRobot venv        │
 │ ┌───────────────────┐   ┌────────────────┐  │ ───────> │ lerobot-train       │
 │ │kikobot_isaac_     │ZMQ│lerobot-        │  │          │ (no Isaac Sim, no   │
 │ │server.py          │<─>│teleoperate/    │  │ policy   │  display)           │
 │ │Newton physics +   │   │record/rollout  │  │ <─────── │                     │
 │ │RTX camera         │   │+ gamepad       │  │          └─────────────────────┘
 │ └───────────────────┘   └────────────────┘  │
 └────────────────────────────────────────────┘
```

- **Isaac Sim runs as its own process** (`isaac/kikobot_isaac_server.py`) in
  its own Python environment. LeRobot connects to it as the robot type
  `kikobot_isaac`, over a local socket. The two never share a venv, so their
  dependencies cannot conflict, and the server can also run on another
  machine.
- **Record and evaluate on an RTX GPU, train anywhere.** Isaac Sim's renderer
  needs RTX (ray-tracing) cores. Datacenter GPUs such as the B200, H100 and
  A100 have none, so camera images cannot be rendered there. The cluster
  only trains, and training needs only the recorded dataset.
- **One action space everywhere.** `kikobot_sim` (MuJoCo), `kikobot_isaac`
  and `kikobot_follower` (the real arm) all use `J1.pos`..`J6.pos` in degrees
  and `gripper.pos` in % open, plus a `record` camera in the sims. A policy
  trained on Isaac data loads on all three.
