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
& "$I\python.exe" isaac\kikobot_isaac_server.py --scene pick_place
```

Leave it running. The scene, cameras and image size are chosen here, when
the server starts; restart it to change them. Physics runs in real time
from the start, whether or not a client is connected. It serves one LeRobot
client at a time, and a new client picks up where the last one left off.
Stop it with Ctrl+C or by closing the Isaac Sim window.

Server options (`--name value`):

| Option | Default | What it does |
|---|---|---|
| `--scene` | `pick_place` | `pick_place` (red cube), `cylinder` (orange cylinder lying down), `desk_sort` ([desk sorting station](#desk-sorting-station)), `empty` (just the arm) |
| `--cameras` | `record` | Cameras to render: `record` (front) and/or `overhead` (straight down on the desk), e.g. `--cameras record overhead` |
| `--width` / `--height` | 640 / 480 | Camera resolution. The robot's `--robot.width/height` must match |
| `--seed` | random | Fixes where things spawn: the same seed gives the same sequence of scenes |
| `--physics-dt` | 0.002 | Physics step, seconds |
| `--render-hz` | 30 | How often the window is redrawn |
| `--fixed-lighting` | off | The same room light every reset (by default it varies, see [the lab](#the-lab)) |
| `--light-scale` | 1.0 | Scale every room light, if the images come out too dark or too bright |
| `--headless` | off | No window. Saves about 1 GB of VRAM. Camera images are still rendered. |
| `--livestream` | off | With `--headless`: watch in the Isaac Sim WebRTC Streaming Client |
| `--bind` | `tcp://127.0.0.1:5557` | Address to listen on (the MuJoCo server has 5556, so both can run) |
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
  --teleop.type=kikobot_gamepad --fps=60
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
| `host` / `port` | `127.0.0.1` / `5557` | Where the server is |
| `camera_names` | `[record]` | Which of the server's `--cameras` to observe, e.g. `'[record,overhead]'` |
| `width` / `height` | 640 / 480 | Must match the server's `--width` / `--height` |
| `image_encoding` | `raw` | `raw`: exact pixels, for one machine. `jpeg`: about 20× smaller, for a network |
| `reset_on_connect` | true | Start each run from a fresh scene |
| `max_velocity_deg_s` | 180 | Joint speed cap, the same as the real arm's |
| `timeout_s` / `connect_timeout_s` | 10 / 120 | How long to wait for the server. The first connect compiles shaders |

The scene and seed are server options. For example, the cylinder task with
a fixed seed:

```powershell
& "$I\python.exe" isaac\kikobot_isaac_server.py --scene cylinder --seed 7
& "$V\lerobot-teleoperate.exe" --robot.type=kikobot_isaac --teleop.type=kikobot_gamepad --fps=60
```

## The lab

Every scene puts the arm on a grey lab table in a makerspace, the same room
as the MuJoCo server's, built from the same description
(`lerobot_robot_kikobot/env/room.py`) by `isaac/room_usd.py`: grey epoxy
floor, off-white walls, a strip window with daylight, six LED ceiling panels
(RectLights), a workbench with a pegboard of tools and a 3D printer behind
the arm, steel shelving with storage bins, a whiteboard and a stool. A faint
dome light stands in for the light bouncing around a real room.

On every reset the light changes: the panels' dimmer level and colour
temperature (3800–6500 K, all panels alike), some panels switched off, the
sun on or off, and its angle through the window. `--fixed-lighting` turns
that off. RTX exposure differs between machines; if the images look too
dark or washed out, start the server with `--light-scale 0.5` or `2`.

## Mirror the real arm into Isaac Sim

The real arm can show its every move in Isaac Sim instead of MuJoCo: start
the Isaac server, then drive the arm with `--robot.sim_backend=isaac`:

```powershell
& "$V\lerobot-teleoperate.exe" --robot.type=kikobot_follower `
  --robot.port=COM3 --robot.id=kikobot --robot.sim_backend=isaac `
  --teleop.type=kikobot_gamepad --fps=60
```

Each goal written to the servos is sent to the server on the same tick, so
the Isaac arm and the real one move together. If the server is not
running, the real arm still runs and the log says so.

## Desk sorting station

`--robot.scene=desk_sort` puts a desk organiser in front of the arm:

- a **pen cup** on the arm's right, for pens, pencils, markers and highlighters;
- a **three-compartment tray** on its left, for the glue stick, tape roll,
  scissors and eraser;
- a random clutter between them: 3 to 8 writing tools and 1 to 4 other
  supplies, each with a random colour, texture (stripes, bands, grip dots,
  printed label, polka dots, ...) and finish, on a wood-grain desk top.

Every reset (and the gamepad's Back button) draws a new desk. **Pens can
land on top of each other**: about a third of the writing tools are dropped
across ones already on the desk (up to three layers), then the server runs
0.6 s of physics with the arm held still so the piles settle before the
episode starts (`--settle-s` on the server). The same `--seed` gives
the same sequence of desks.

```powershell
& "$I\python.exe" isaac\kikobot_isaac_server.py --scene desk_sort --cameras record overhead
& "$V\lerobot-teleoperate.exe" --robot.type=kikobot_isaac `
  --robot.camera_names='[record,overhead]' `
  --teleop.type=kikobot_gamepad --display_data=true
```

**Assets.** Items come from your Isaac Sim asset server (Nucleus / SimReady)
where it has them, and are built from simple meshes otherwise, so the scene
always loads, even offline. Find what your server has once per machine:

```powershell
& "$I\python.exe" isaac\desk_assets.py --discover
# a SimReady pack somewhere else as well:
& "$I\python.exe" isaac\desk_assets.py --discover --folder https://.../simready_content
```

This writes `models/desk_assets.json` (git-ignored) and prints the matches
for each kind. Edit the file to drop or reorder assets: an empty list means
that kind is built procedurally. Server options:

| Server option | Default | What it does |
|---|---|---|
| `--desk-assets` | `auto` | `auto`: use the JSON where it has a match. `nucleus`: require the JSON. `procedural`: ignore it |
| `--desk-assets-json` | `models/desk_assets.json` | Another catalog |
| `--keep-asset-materials` | off | Keep an asset's own look instead of the random colours and textures |
| `--settle-s` | 0.6 | Physics time after each reset for dropped items to come to rest |

Assets are only used for their looks. Each one is turned so its longest
side lies along the desk, scaled to the real size of that kind, and given a
simple hidden collider (capsule, cylinder or box) fitted to it. Newton
needs convex shapes, and asset colliders are made for PhysX. The pen cup and
tray are always procedural, with box-wall colliders, so items can drop inside them.

## How it runs

- **Real time.** Physics steps on wall-clock time (2 ms steps by default),
  whatever the client does, and the server answers requests between steps.
  The log prints the real-time factor every 10 s. If the GPU cannot keep up,
  the sim drops the time rather than racing to catch up.
- **No filter on the motion.** A `command` sets the joint drive targets and
  physics takes the arm there, as the real servos do with the same goal.
  The robot only caps joint speed (`--robot.max_velocity_deg_s`), the same
  way as the real arm, so the sim, Isaac and the real arm get identical
  commands.
- **Newton physics.** Newton runs MuJoCo-Warp on the GPU with the
  `<option>` settings from `robot.xml` (elliptic friction cones, Newton
  solver). Contacts, friction and the servo model match the MuJoCo scene
  closely, but not bit for bit: Newton and MuJoCo are different builds of
  the solver, and the meshes go through USD.
- **Cameras.** The `record` camera is rendered by Isaac Sim's RTX renderer.
  It sits where the MuJoCo scene's does: 0.78 m in front, 0.34 m up, aimed
  at the base, with a 48° vertical field of view. A policy trained on Isaac
  images sees the same framing as the MuJoCo camera, but more realistic
  lighting.
- **Protocol.** The same as `kikobot-sim-server`'s
  (`lerobot_robot_kikobot/protocol.py`), so `kikobot_isaac`, `kikobot_sim`
  and the real arm's mirror share one client.
