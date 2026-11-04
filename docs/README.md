# Kiko-Bot docs

The main [README](../README.md) says what the project is and gets you running.
The guides here have the details.

| Guide | What it covers |
|---|---|
| [The MuJoCo sim](sim.md) | The sim server and its options, the room, the desk and its stations, the camera, Newton physics |
| [The real arm](real-arm.md) | Ports, calibration, teleop options, sag and joint direction, the arm server and servo limits page, the Raspberry Pi HAT |
| [Isaac Sim setup](isaac-sim-setup.md) | Install Isaac Sim 6 and LeRobot side by side, convert the arm to USD, check it runs |
| [Teleop in Isaac Sim](isaac-sim-teleop.md) | Start the Isaac server, drive the arm with the gamepad, all options |
| [Record, train, evaluate](isaac-sim-record-and-train.md) | `lerobot-record` in Isaac, inspect and replay datasets, train, run the policy in Isaac and on the real arm |
| [Scripted demos for OpenVLA](datagen-openvla.md) | `kikobot-datagen`: a scripted expert records perfect desk episodes (Newton + MuJoCo Warp, on SLURM), exported as RLDS for OpenVLA |
| [Training on SLURM](slurm-training.md) | Move a dataset to the cluster, train on a B200 with `sbatch`, bring the policy back |
| [Troubleshooting](troubleshooting.md) | Errors and fixes, for the sim, the real arm and Isaac Sim |

## Isaac Sim and the cluster

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
  and `gripper.pos` in % open, plus the one `record` camera in the sims. A policy
  trained on Isaac data loads on all three.
