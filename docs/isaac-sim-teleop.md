# Teleop in Isaac Sim

Do [setup](isaac-sim-setup.md) first. Everything lives in the repo's parent
folder, `$W`, as set up there. Open each terminal inside `kikobot-lerobot`
and set:

```powershell
$W = Split-Path $PWD                    # run inside kikobot-lerobot: its parent
$I = "$W\isaacsim-env\Scripts"          # Isaac Sim's Python
$V = "$W\lerobot\.venv\Scripts"         # LeRobot's tools
```

```bash
W=$(dirname "$PWD"); I=$W/isaacsim-env/bin; V=$W/lerobot/.venv/bin   # Ubuntu
```

Commands are shown for Windows. On Ubuntu, change `` ` `` line endings to
`\`, and `"$I\python.exe"` / `"$V\lerobot-*.exe"` to `$I/python` / `$V/lerobot-*`.

## 1. Start the sim server

Run from the repo:

```powershell
cd $W\kikobot-lerobot
& "$I\python.exe" isaac\kikobot_isaac_server.py
```

Leave it running. It serves one LeRobot client at a time, and a new client
picks up where the last one left off. The scene is only rebuilt if the
camera size, fps or scene name changes. Stop it with Ctrl+C or by closing the
Isaac Sim window.

Server options (`--name value`):

| Option | Default | What it does |
|---|---|---|
| `--headless` | off | No window. Saves about 1 GB of VRAM. Camera images are still rendered. |
| `--livestream` | off | With `--headless`: watch in the Isaac Sim WebRTC Streaming Client |
| `--bind` | `tcp://127.0.0.1:5556` | Address to listen on |
| `--usd` | `models/usd/kikobot.usd` | The arm, from `isaac/convert_mjcf.py` |
| `--integrator` | `implicitfast` | Newton's MuJoCo-Warp integrator: `euler`, `rk4`, `implicit`, `implicitfast` |
| `--kp` / `--kd` | 200 / 3 | Arm joint stiffness (N·m/rad) and damping (N·m·s/rad) |
| `--max-torque` | 2.94 | Arm joint torque limit, N·m (the STS servo's stall torque) |
| `--finger-kp` / `--finger-kd` / `--finger-max-force` | 800 / 8 / 20 | Gripper drive, N/m, N·s/m, N |
| `--physics-variant` | `newton` | Which physics variant of the imported USD to load (the log lists them) |

## 2. Drive it with the gamepad

Plug the controller in first. In a second terminal:

```powershell
& "$V\lerobot-teleoperate.exe" `
  --robot.type=kikobot_isaac `
  --teleop.type=kikobot_gamepad
```

Add `--display_data=true` to watch the `record` camera and the joint values
in rerun.

The controls are the same as in MuJoCo and on the real arm:

| Control | Action |
|---|---|
| Left stick | Move forward / back / left / right |
| LT / RT | Move down / up |
| Right stick ← → / ↑ ↓ | Roll / pitch the jaws |
| LB / RB | Yaw the jaws |
| A / B (hold) | Close / open the jaws |
| X | Precision mode (30 % speed) |
| Y | Arm back to home |
| **Back / View / Share** | **Reset the scene**: arm home, cube and pad respawned. Sims only. |

The full list of teleop options (`--teleop.linear_speed_m_s`, ...) is in
the [main README](../README.md#4-gamepad-controls).

## 3. Robot options

`--robot.<name>=value`:

| Option | Default | What it does |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `5556` | Where the server is |
| `scene` | `pick_place` | `pick_place` (red cube), `cylinder` (orange cylinder lying down), `empty` (just the arm) |
| `fps` | 30 | Control rate. Each tick advances physics by exactly 1/fps |
| `width` / `height` | 640 / 480 | `record` camera resolution |
| `seed` | random | Fixes where the cube and pad spawn: the same seed gives the same sequence |
| `image_encoding` | `raw` | `raw`: exact pixels, for one machine. `jpeg`: about 20× smaller, for a network |
| `reset_on_connect` | true | Start each run from a fresh scene |
| `timeout_s` / `connect_timeout_s` | 10 / 120 | How long to wait for the server. The first connect compiles shaders |
| `motion.*` | as on the real arm | Input shaping and speed limits, same as `kikobot_sim` ([README §7](../README.md#7-tune-the-motion-input-shaping)) |

For example, the cylinder task with a fixed seed:

```powershell
& "$V\lerobot-teleoperate.exe" --robot.type=kikobot_isaac `
  --robot.scene=cylinder --robot.seed=7 --teleop.type=kikobot_gamepad
```

## How it runs

- **Lockstep.** The sim only moves when LeRobot sends an action: one
  control tick is 17 physics steps of 1/(30×17) s. If the GPU is slow, the
  episode slows down, but the motion does not change.
- **Same motion as the other backends.** The same `MotionShaper` as the
  MuJoCo sim and the real arm runs before every physics step, so the arm
  accelerates and settles the same way. Your `--robot.motion.*` settings are
  sent to the server when you connect.
- **Newton physics.** Newton runs MuJoCo-Warp on the GPU with the
  `<option>` settings from `robot.xml` (elliptic friction cones, Newton
  solver). Contacts, friction and the servo model match the MuJoCo scene
  closely, but not bit for bit: Newton and MuJoCo are different builds of
  the solver, and the meshes go through USD.
- **Cameras.** The `record` camera is rendered by Isaac Sim's RTX renderer.
  It sits where it does in `pick_place_scene.xml`: 0.78 m in front, 0.34 m
  up, aimed at the base, with a 48° vertical field of view. A policy trained
  on Isaac images sees the same framing as the MuJoCo camera, but more
  realistic lighting.
