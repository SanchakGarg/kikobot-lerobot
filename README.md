# kikobot-lerobot

Kiko-Bot, a 6-DOF arm, as a [LeRobot](https://github.com/huggingface/lerobot)
plugin. The same commands and the same gamepad drive the MuJoCo simulation,
NVIDIA Isaac Sim and the real arm. Each simulation runs as a server in its
own process, in real time. The real arm sends every command it receives to
one of them, so the simulated arm moves with it:

| LeRobot type | What it is |
|---|---|
| `--robot.type=kikobot_sim` | The arm in `kikobot-sim-server` (MuJoCo, optionally Newton physics) |
| `--robot.type=kikobot_isaac` | The arm in the Isaac Sim server, Newton physics and RTX cameras ([docs/](docs/README.md)) |
| `--robot.type=kikobot_follower` | The physical arm on Feetech STS servos, mirrored into a sim |
| `--robot.type=kikobot_arm` | The physical arm through `kikobot-arm-server`, the real arm's own server ([6a](#6a-the-arm-server)) |
| `--teleop.type=kikobot_gamepad` | 6-DOF gamepad control: full IK over all six joints |

**Isaac Sim, and training on a SLURM cluster:** see [docs/](docs/README.md).

**Scripted demonstrations for OpenVLA:** `kikobot-datagen` records verified,
perfect episodes of the desk tasks (each kind of item into its station) on
Newton + MuJoCo Warp, and exports them as RLDS. See
[docs/datagen-openvla.md](docs/datagen-openvla.md).

The sims and the real arm share one action space (`J1.pos`..`J6.pos` in
degrees, `gripper.pos` in % open), so a dataset recorded on one, or a policy
trained on it, runs on the other.

## Contents

0. [Quick start: the sim](#0-quick-start-the-sim)
1. [Quick start: real arm + gamepad](#1-quick-start-real-arm--gamepad)
2. [Repository layout](#2-repository-layout)
3. [Install](#3-install)
4. [Gamepad controls](#4-gamepad-controls)
5. [The sim server](#5-the-sim-server)
6. [Drive the real arm](#6-drive-the-real-arm)
   - 6a. [The arm server, and the servo limits page](#6a-the-arm-server)
7. [Speed and smoothness](#7-speed-and-smoothness)
8. [Record a dataset](#8-record-a-dataset)
9. [Platform notes](#9-platform-notes)
10. [Troubleshooting](#10-troubleshooting)
11. [Commit messages](#11-commit-messages)

---

## 0. Quick start: the sim

The desk sorting scene in MuJoCo, driven with a gamepad. No hardware is
needed. Install LeRobot and this plugin first ([3](#3-install)), and plug the
controller in before starting teleop.

Terminal 1, the sim server (leave it running):

```bash
# Ubuntu
~/lerobot/.venv/bin/kikobot-sim-server --scene desk_sort
# macOS: MuJoCo's window needs mjpython
~/lerobot/.venv/bin/mjpython -m lerobot_robot_kikobot.sim_server --scene desk_sort
```

Terminal 2, the gamepad:

```bash
~/lerobot/.venv/bin/lerobot-teleoperate --robot.type=kikobot_sim \
  --teleop.type=kikobot_gamepad --fps=60
```

On Windows the same commands are
`& "$HOME\lerobot\.venv\Scripts\kikobot-sim-server.exe" --scene desk_sort` and
`& "$HOME\lerobot\.venv\Scripts\lerobot-teleoperate.exe" --robot.type=kikobot_sim --teleop.type=kikobot_gamepad --fps=60`.

- **Back / View / Share** on the gamepad lays out a new desk; **Y** returns
  the arm to its home pose. All controls: [4](#4-gamepad-controls).
- Add `--display_data=true` to the teleop command to see the `record`
  camera in rerun, and `--robot.camera_names='[record,overhead]'` for the
  top-down camera as well.
- Other scenes: `--scene pick_place` (a cube and a place pad),
  `--scene cylinder` or `--scene empty` on the server ([5](#5-the-sim-server)).

---

## 1. Quick start: real arm + gamepad

The shortest path from a fresh machine to driving the physical arm with a
gamepad. Sections 3–6 explain each step in detail.

**Before you start**

- Python 3.12+ and git installed.
- The servo bus powered (12 V) and its USB adapter plugged into the computer.
- An Xbox or PlayStation controller plugged in (USB) or paired (Bluetooth)
  **before** you launch teleop. It is only detected at startup.
- The arm clear of obstacles: on connect it eases to the ready pose.

### Ubuntu (22.04 / 24.04)

Run from a terminal on the desktop, not over bare SSH: the gamepad is read
through GLFW, which needs a display.

```bash
# 1. System packages and serial port access (log out and back in after usermod).
sudo apt install -y git python3.12-venv libgl1 libglfw3
sudo usermod -aG dialout $USER

# 2. LeRobot, from source, in its own venv.
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
python3.12 -m venv .venv
.venv/bin/pip install -e ".[core_scripts,training,feetech]"

# 3. This plugin, into the same venv.
git clone https://github.com/SanchakGarg/kikobot-lerobot.git ~/kikobot-lerobot
cd ~/kikobot-lerobot
~/lerobot/.venv/bin/pip install -e .

# 4. Optional but recommended: stable /dev/kikobot name, 1 ms FTDI latency.
sudo cp udev/99-kikobot.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# Unplug and replug the servo adapter.

# 5. Find the port (skip if you installed the udev rule: use /dev/kikobot).
~/lerobot/.venv/bin/lerobot-find-port

# 6. Start the sim server, in its own terminal; leave it running.
~/lerobot/.venv/bin/kikobot-sim-server

# 7. Try it in the sim first (no hardware moves), in a second terminal.
~/lerobot/.venv/bin/lerobot-teleoperate --robot.type=kikobot_sim \
  --teleop.type=kikobot_gamepad --fps=60

# 8. Calibrate the real arm once, so it moves like the sim (see below).
~/lerobot/.venv/bin/lerobot-calibrate --robot.type=kikobot_follower \
  --robot.port=/dev/ttyACM0 --robot.id=kikobot

# 9. Drive the real arm. The sim arm in the server's window moves with it.
~/lerobot/.venv/bin/lerobot-teleoperate --robot.type=kikobot_follower \
  --robot.port=/dev/ttyACM0 --robot.id=kikobot \
  --teleop.type=kikobot_gamepad --fps=60
```

Replace `/dev/ttyACM0` with what step 5 printed (or `/dev/kikobot`).
Step 8 walks you through every joint: min, any middle point, max
([calibration](#calibration)).
To check the controller is seen by Linux: `ls /dev/input/by-id/ | grep -i joystick`.

**Ubuntu: the real arm through the arm server.** Instead of step 9, run the
arm as its own server, so the gamepad and the servo limits page can both
reach it at once ([6a](#6a-the-arm-server)). One terminal each, after
calibrating (step 8):

```bash
# Terminal 1 (optional): the sim server, so the sim arm mirrors the real one.
~/lerobot/.venv/bin/kikobot-sim-server

# Terminal 2: the arm server. Opens the servo bus and holds the arm; leave it running.
~/lerobot/.venv/bin/kikobot-arm-server --robot.port=/dev/ttyACM0 --robot.id=kikobot

# Terminal 3: drive it with the gamepad.
~/lerobot/.venv/bin/lerobot-teleoperate --robot.type=kikobot_arm \
  --teleop.type=kikobot_gamepad --teleop.robot_id=kikobot --fps=60

# Terminal 4 (optional): the servo limits page, at http://127.0.0.1:8080
~/lerobot/.venv/bin/kikobot-servo-web
```

On a Raspberry Pi with the Driver HAT, use `--robot.port=/dev/serial0`
([Pi notes](#pi-hat)). To drive it from a laptop, add `--host=0.0.0.0` to the
arm server and `--robot.host=<Pi's IP>` to the teleop.

### Windows 10 / 11 (PowerShell)

Install Python 3.12 from python.org (tick "Add python.exe to PATH") and
Git for Windows first. If PowerShell refuses to run scripts, you do not need
to activate the venv: the commands below call its executables directly.

```powershell
# 1. LeRobot, from source, in its own venv.
git clone https://github.com/huggingface/lerobot.git $HOME\lerobot
cd $HOME\lerobot
py -3.12 -m venv .venv
# NVIDIA GPU only: install CUDA torch first, or pip picks the CPU build.
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install -e ".[core_scripts,training,feetech]"

# 2. This plugin, into the same venv.
git clone https://github.com/SanchakGarg/kikobot-lerobot.git $HOME\kikobot-lerobot
cd $HOME\kikobot-lerobot
& "$HOME\lerobot\.venv\Scripts\pip.exe" install -e .

# 3. Find the port (unplug the adapter when asked). Usually COM3, COM4, ...
& "$HOME\lerobot\.venv\Scripts\lerobot-find-port.exe"

# 4. Start the sim server, in its own window; leave it running.
& "$HOME\lerobot\.venv\Scripts\kikobot-sim-server.exe"

# 5. Try it in the sim first (no hardware moves), in a second window.
& "$HOME\lerobot\.venv\Scripts\lerobot-teleoperate.exe" `
  --robot.type=kikobot_sim --teleop.type=kikobot_gamepad --fps=60

# 6. Calibrate the real arm once, so it moves like the sim (see below).
& "$HOME\lerobot\.venv\Scripts\lerobot-calibrate.exe" `
  --robot.type=kikobot_follower --robot.port=COM3 --robot.id=kikobot

# 7. Drive the real arm. The sim arm in the server's window moves with it.
& "$HOME\lerobot\.venv\Scripts\lerobot-teleoperate.exe" `
  --robot.type=kikobot_follower --robot.port=COM3 --robot.id=kikobot `
  --teleop.type=kikobot_gamepad --fps=60
```

Replace `COM3` with the port from step 3 (also listed in Device Manager →
Ports (COM & LPT)). Step 6 walks you through every joint: min, any middle
point, max ([calibration](#calibration)). Keep the same `--robot.id` in both
commands: it names the calibration file. The servo adapter needs its
USB-serial driver: CH340/CH343 adapters (Feetech URT-1, Waveshare) from the WCH website, FTDI from ftdichip.com.
For FTDI adapters, set Device Manager → the port → Port Settings → Advanced →
Latency Timer to 1 ms. To check the controller, open `joy.cpl` (Win + R).

### Once it is running

- The arm first holds where it is, then moves slowly (30°/s) to the home pose.
- The sim server's window shows the simulated arm receiving the same
  commands as the real one, on the same tick, so the two move together.
  `--display_data=true` plots each joint's commanded (action) and measured
  (observation) angle in rerun. Without a sim server the real arm still
  runs, and the log reports that the mirror is off.
- After calibration every joint's direction and range come from what you
  measured. Without a calibration file, J5 is inverted by default because
  its servo turns opposite to the model ([joint direction](#joint-direction)).
- Left stick and LT/RT move the gripper; right stick and LB/RB rotate it;
  hold A to close, B to open; X toggles precision mode; Y returns to the home pose.
  Full table in [section 4](#4-gamepad-controls).
- **Ctrl+C** stops teleop. Torque stays on so the arm does not drop; power
  the servos off, or add `--robot.disable_torque_on_disconnect=true`, to let
  it go limp.
- If a joint turns the wrong way or does not line up with the sim,
  calibrate again ([calibration](#calibration)).
- To check or change the servos' current and torque limits while you
  drive, run the arm as a server instead and open the limits page:
  [6a](#6a-the-arm-server).

---

## 2. Repository layout

| Path | What's in it |
|---|---|
| `models/robot.xml` | The arm, generated by onshape-to-robot. Plain radians. |
| `models/extras.xml` | Hand additions spliced into `robot.xml` on each re-export (`config.json` points at it). |
| `models/scene.xml` | The arm alone on a floor (also the kinematics model). |
| `lerobot_robot_kikobot/env/` | The lab room (`room.py`, shared with Isaac Sim) and the MuJoCo scene builder: empty, pick-and-place, cylinder, desk sorting. |
| `lerobot_robot_kikobot/kinematics.py` | 6-DOF damped-least-squares IK on the MuJoCo model. |
| `lerobot_robot_kikobot/sim_server.py` | `kikobot-sim-server`: the MuJoCo scene in real time, behind a socket. |
| `lerobot_robot_kikobot/protocol.py`, `sim_client.py` | The ZMQ messages both sim servers speak, and the client for them. |
| `lerobot_robot_kikobot/motion.py` | The speed cap and the droop (sag) correction. |
| `lerobot_robot_kikobot/feetech.py` | Sync read/write over the Feetech bus. |
| `lerobot_robot_kikobot/sim.py`, `isaac.py`, `follower.py`, `arm.py`, `gamepad.py` | The five LeRobot devices. |
| `lerobot_robot_kikobot/arm_server.py` | `kikobot-arm-server`: the real arm behind the sim servers' protocol. |
| `lerobot_robot_kikobot/servo_limits.py`, `servo_web.py`, `web/` | The servos' current, torque and safety limits, and `kikobot-servo-web`, the page that shows and sets them. |
| `lerobot_robot_kikobot/desk/` | The desk sorting station, shared by the sims: stations, item pool and sampler, meshes, textures, and its MuJoCo builder. |
| `lerobot_robot_kikobot/datagen/` | `kikobot-datagen`: the desk tasks, the scripted expert, its perfection checks, the recorder and the RLDS export for OpenVLA. |
| `isaac/` | Run with Isaac Sim's Python: the Isaac sim server (Newton) and the MJCF → USD converter. |
| `scripts/slurm/train.sbatch` | Train a policy on a SLURM GPU node. |
| `scripts/slurm/datagen*.sbatch` | Record scripted desk episodes on the cluster (Newton + MuJoCo Warp) and build their RLDS datasets. |
| `scripts/openvla/` | The desk datasets in OpenVLA's registry, and a `finetune.py` wrapper. |
| `docs/` | Isaac Sim setup, teleop, record/train/evaluate and SLURM guides. |
| `lerobot_robot_kikobot/calibration.py` | Fits each joint's min/middle/max readings onto the model's range. |
| `udev/99-kikobot.rules` | Ubuntu: serial port permissions and a stable `/dev/kikobot` name. |
| `tests/` | Speed cap, droop correction, calibration, sim server, real arm (fake bus) and Isaac client tests (no hardware or Isaac Sim needed). |

---

## 3. Install

You need Python 3.12+ and LeRobot installed from source. LeRobot, MuJoCo and
this plugin must be in **one** environment.

> **Coming from the old `KikoBot` repo?** Its `lerobot_integration` package
> uses the same name, `lerobot_robot_kikobot`. Remove it first:
> `pip uninstall lerobot_robot_kikobot`.

**Ubuntu** (24.04 ships Python 3.12; on 22.04 get it from the deadsnakes PPA)

```bash
sudo apt install python3.12-venv libgl1 libglfw3
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
python3.12 -m venv .venv
.venv/bin/pip install -e ".[core_scripts,training,feetech]"

cd ~/kikobot-lerobot
~/lerobot/.venv/bin/pip install -e .

# Serial port access without sudo (log out and back in afterwards).
sudo usermod -aG dialout $USER
# Optional: a stable /dev/kikobot name, and FTDI latency fixed.
sudo cp udev/99-kikobot.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/huggingface/lerobot.git $HOME\lerobot
cd $HOME\lerobot
py -3.12 -m venv .venv
# NVIDIA GPU only: install CUDA torch first, or pip picks the CPU build.
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install -e ".[core_scripts,training,feetech]"

cd C:\path\to\kikobot-lerobot
$HOME\lerobot\.venv\Scripts\pip install -e .
```

**macOS**: same as Ubuntu without the `apt`, `dialout` and udev steps.

The commands below use `$V` for the venv's scripts folder:
`V=~/lerobot/.venv/bin` on Ubuntu/macOS, `$V="$HOME\lerobot\.venv\Scripts"` on
Windows (then run tools as `& "$V\lerobot-teleoperate.exe"`, and continue
lines with a backtick instead of `\`).

**Check the plugin loaded.** If it fails to import, LeRobot only logs the
error and the types are silently missing:

```bash
$V/python -c "
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.robots import RobotConfig
register_third_party_plugins()
print(sorted(c for c in RobotConfig.get_known_choices() if 'kikobot' in c))"
```

It must print `['kikobot_arm', 'kikobot_follower', 'kikobot_isaac', 'kikobot_sim']`.

**Run the tests** (optional, no hardware): `$V/pip install pytest && $V/python -m pytest`.

---

## 4. Gamepad controls

Xbox layout. On PlayStation: cross = A, circle = B, square = X, triangle = Y.

| Control | Action |
|---|---|
| Left stick | Move forward / back / left / right |
| LT / RT | Move down / up |
| Right stick ← → | **Roll**: twist the jaws about their own axis |
| Right stick ↑ ↓ | **Pitch**: tilt the jaws forward / back |
| LB / RB | **Yaw**: tilt the jaws left / right |
| A (hold) | Close the jaws |
| B (hold) | Open the jaws |
| X | Precision mode on/off (30 % speed) |
| Y | Glide back to the home pose ([set in calibration](#home-pose); the ready pose if none) |
| Back / View / Share | Sims only (`kikobot_sim`, `kikobot_isaac`): reset the scene (arm home, object and pad respawned) |

Moves are in the base frame, rotations about the gripper's own axes, and
every axis is independent: moving in X does not tilt the jaws. At the edge
of the arm's reach, or at a joint limit, the arm stops; move back the other
way to continue.

**Plug the controller in before starting.** GLFW only looks for controllers
when it starts.

Teleop options (`--teleop.<name>=`): `linear_speed_m_s` (0.12),
`angular_speed_deg_s` (90), `gripper_speed_pct_s` (120), `deadzone` (0.12),
`input_smoothing_s` (0.12), `stop_rate_per_s` (4: how fast the speed falls
when you ease off the stick, in full speed per second; higher stops sooner,
too high overshoots), `precision_scale` (0.3), `joystick` (0),
`robot_id` (`kikobot`: whose saved [home pose](#home-pose) to use).

---

## 5. The sim server

The simulation is its own process. Start it first and leave it running:

```bash
$V/kikobot-sim-server --scene pick_place
```

On macOS, MuJoCo's window needs `mjpython`:
`$V/mjpython -m lerobot_robot_kikobot.sim_server --scene pick_place`.
Without a display, add `--no-viewer`.

Physics runs in real time from the moment the server starts. A client's
`command` sets the joint targets and physics moves the arm there, as the
servos do with the same goal. The log prints the real-time factor every 10 s.

**The lab.** Every scene puts the arm on a grey lab table in a makerspace:
grey epoxy floor, a window with daylight, six LED ceiling panels and a
task light over the table, a
workbench with a pegboard of tools and a 3D printer behind the arm,
shelving with storage bins, a whiteboard and a stool. The room is described
once, in `lerobot_robot_kikobot/env/room.py`, and the Isaac Sim server builds
the same room from that file. **The lighting changes on every reset**:
dimmer level, colour temperature (4300–6500 K), panels switched off, and
the sun on or off at a random angle, so that recorded datasets cover a range
of lighting. `--fixed-lighting` disables this.

| Scene | On the table |
|---|---|
| `pick_place` | A red 32 mm cube and a green place pad, respawned each reset |
| `cylinder` | An orange cylinder lying on its side, and the pad |
| `desk_sort` | A stationery sorting station: a pen cup, a marker cup, a tape peg, trays and a note mat around the arm, and 32 pens, pencils, markers, highlighters, tapes, glue sticks, erasers, sticky notes, scissors, a stapler, a ruler, ... of which a random 5–11 (8 on average) lie in front of it, each with a random colour and texture, pens sometimes piled up ([details](docs/isaac-sim-teleop.md#desk-sorting-station)) |
| `empty` | Nothing |

Drive it from a second terminal:

```bash
$V/lerobot-teleoperate --robot.type=kikobot_sim --teleop.type=kikobot_gamepad --fps=60
```

`--display_data=true` shows the `record` camera in rerun.

Server options: `--scene` (`pick_place`, `cylinder`, `desk_sort`, `empty`, or a path to
an MJCF file, used as is, without the lab), `--host` (`127.0.0.1`; `0.0.0.0`
to accept other machines), `--port` (5556), `--window` (`mujoco`, `newton`,
`none`; `--no-viewer` = `none`), `--seed`, `--timestep` (0.002 s),
`--fixed-lighting`, `--engine` and `--renderer` (below).

<a id="newton"></a>**Newton physics and ray-traced cameras (NVIDIA GPU).** The
server can step the physics with NVIDIA [Newton](https://github.com/newton-physics/newton)
(its MuJoCo solver, run on the GPU by MuJoCo Warp) and render the cameras
with the MuJoCo Warp ray tracer, which adds ray-traced shadows and
anti-aliasing. Install the extra into the same venv on the GPU machine:

```bash
$V/pip install -e ".[newton]"      # Newton, Warp, MuJoCo Warp; pins MuJoCo 3.15
$V/kikobot-sim-server --scene desk_sort --engine newton --renderer mjwarp
```

| Option | Values | Default |
|---|---|---|
| `--engine` | `mujoco` (CPU), `newton` | `auto`: `newton` when it is installed and a CUDA GPU is there |
| `--renderer` | `opengl`, `mjwarp` (GPU ray tracing) | `auto`: `mjwarp` with the Newton engine on a GPU |
| `--window` | `mujoco` (the full scene, with either engine), `newton` (Newton's ViewerGL, plain colours) | `mujoco` |

The scene is identical with either engine. The server keeps its MuJoCo
model (room, desk, materials) as the scene of record, Newton receives it as
MJCF, and joint positions are copied back by name after every step. The
servo model, including the actuators' damping, is the same, so a commanded
joint moves the same way on both engines. Without CUDA, `--engine newton`
falls back to MuJoCo's C solver and `--renderer mjwarp` runs on Warp's CPU
backend at several seconds per frame, which is too slow for teleop.

Sim robot options (`--robot.<name>=`): `host` / `port` (where the server
is), `camera_names` (`[record]`; `overhead` too), `width` / `height`
(640 / 480), `image_encoding` (`raw`, or `jpeg` over a network),
`reset_on_connect` (true), `max_velocity_deg_s` (180), `timeout_s` (5).

Isaac Sim has its own server, with the same protocol:
[docs/isaac-sim-teleop.md](docs/isaac-sim-teleop.md).

---

## 6. Drive the real arm

**1. Find the port.** `$V/lerobot-find-port`. Usually `COM3` on Windows,
`/dev/ttyACM0` (or `/dev/ttyUSB0`) on Ubuntu, or `/dev/kikobot` with the udev
rule installed. On Linux, leaving out `--robot.port` uses the first of
`/dev/kikobot`, `/dev/ttyACM0`, `/dev/ttyUSB0` and `/dev/serial0` that exists.
A Waveshare Bus Servo Driver HAT on a Raspberry Pi has its own steps:
[Raspberry Pi + Driver HAT](#pi-hat).

<a id="calibration"></a>**2. Calibrate (once).** This matches the real
arm to the sim: each joint's two ends are mapped onto the MuJoCo model's
joint limits, so a command of, say, J2 = -125° puts the real J2 at the same
place as the model's.

Ubuntu:

```bash
~/lerobot/.venv/bin/lerobot-calibrate --robot.type=kikobot_follower \
  --robot.port=/dev/ttyACM0 --robot.id=kikobot
```

Windows (PowerShell):

```powershell
& "$HOME\lerobot\.venv\Scripts\lerobot-calibrate.exe" `
  --robot.type=kikobot_follower --robot.port=COM3 --robot.id=kikobot
```

Every servo goes limp as soon as it connects, so **support the arm**. The
only motion is the one you confirm after each joint: that joint alone turns
slowly (25°/s) to the centre of its range and holds there. Joints not yet
calibrated stay limp the whole time, so the one you are moving by hand is
always free, and the ones already done hold the arm up.

1. If a calibration file exists, it asks what to do:
   - **Enter**: keep it as it is;
   - **`all`**: redo every joint;
   - **joint names**: redo only those, e.g. `J3`, `gripper` or `J2 gripper`.
     The other joints keep their saved calibration;
   - **`home`**: only set the [home pose](#home-pose);
   - **`check`**: only [check the directions](#check-directions);
   - **`flip J3`** (or `flip J3 J4`): reverse those joints' direction
     without moving them ([flip a joint](#flip-joint));
   - **`zero J5`**: re-zero a joint that sits off the sim by a fixed
     amount ([re-zero a joint](#zero-joint)).
   A joint missing from the file (e.g. the gripper, if you calibrated before
   it was added) is listed as "not calibrated yet".
2. For each joint, J1 to J6 and the gripper (bus ID 7,
   `--robot.gripper_id`), move it by hand and press Enter at each of:
   - **min**: the joint's lower limit;
   - **middle**: anywhere in between. It is only used to tell which way
     the joint turned from min to max, so a joint whose travel passes the
     servo's 4095 → 0 wrap is still read right;
   - **max**: the joint's upper limit.

   **Which end is min matters.** It sets the joint's direction; the wrong
   way round, the real joint turns opposite to the sim, and gamepad moves
   come out in the wrong directions (e.g. a forward push lifts the arm).
   "Front" is the way the arm reaches in the ready pose: stand behind the
   base facing it. The prompt prints these too:

   | Joint | min | max |
   |---|---|---|
   | J1 (−165° / +165°) | base turned counter-clockwise, seen from above | clockwise |
   | J2 (−125° / +125°) | upper arm tilted forward, toward the front | tilted back |
   | J3 (−140° / +140°) | forearm bent up and back (opposite to the ready pose) | bent forward and down (as in the ready pose) |
   | J4 (−140° / +140°) | wrist tilted so the jaws swing back, toward the base | jaws swing forward and up |
   | J5 (−140° / +140°) | wrist swung to the left | to the right |
   | J6 (−175° / +175°) | jaws turned clockwise, seen from above with the jaws pointing down | counter-clockwise |
   | gripper | jaws closed | jaws open |
   Before each joint it prints which way its min and max lie (the table
   below), then asks for min, middle and max in turn.

   With the **sim server running**, its window shows the pose each step asks
   for: the joint being calibrated at min, middle or max, every other joint
   at its centre (where the calibrated ones are holding). Match the real arm
   to it. `check` shows the ready pose there. Without the server you get the
   prompts only; `--robot.calibration_view=false` turns it off.
3. Each joint prints its travel in servo steps and how it compares with
   the servo's nominal 4096 steps per turn. A warning means the measured
   travel is more than 15 % off the model's: type `r` to redo that joint
   with it pushed fully to its ends, or Enter to keep it (e.g. a geared joint).
4. The joint is saved, and it says that only this joint will power on and
   turn to its centre. Keep clear and press Enter, and it moves there
   slowly and holds; `s` leaves it limp instead. Then the next joint.

What is stored: each servo's Homing_Offset register is set so that the
middle of its travel reads 2048 (the range never wraps), and the file
`~/.cache/huggingface/lerobot/calibration/robots/kikobot_follower/kikobot.json`
(`%USERPROFILE%\.cache\...` on Windows) gets, per joint, the readings at
its two ends (`range_min` / `range_max`), its direction (`drive_mode`) and
its reading at the model's zero (`homing_offset`). Positions in between are
linear. Each joint is saved as soon as it is done, so Ctrl+C (or a crash)
keeps the joints finished so far and leaves the rest as they were.

If the servos stop answering mid-calibration (a cable pulled loose at a
joint's end, or the supply dipping), it says which servo and waits: fix it
and press Enter to retry; the port is reopened if the adapter dropped. Move
joints slowly by hand: a servo turned fast acts as a generator and feeds
voltage spikes back into the bus.

<a id="home-pose"></a>**Home pose.** After the joints it asks whether to
set the home pose (`s` skips; `home` at the first prompt does only this).
It first releases the joints holding at their centres, after you press
Enter, so support the arm. Then hold it in its home pose and press Enter.
If you skip it, the calibrated joints stay powered at their centres when
calibration ends. That is where teleop starts and where **Y** returns the arm,
jaws included. It is saved as `kikobot_home.json` next to the calibration
file. The gamepad reads it for `--teleop.robot_id` (default `kikobot`; use
the same name as `--robot.id`); with none saved it uses the built-in ready
pose. Pick a pose with the jaws above the table and inside the teleop
workspace, or the first move pulls them in.

<a id="check-directions"></a>**Check the directions.** `check` prints
the arm's joint angles as the sim would show them, each time you press
Enter, next to the sim's ready pose. Bend the real arm by hand into the
ready pose you see in the sim and compare: every joint should have the
same sign and roughly the same value. A joint whose sign is opposite was
calibrated the wrong way round: [flip it](#flip-joint).

<a id="zero-joint"></a>**Re-zero a joint.** Calibration puts a joint's 0° exactly
halfway between the two ends you recorded. If its physical stops are not
symmetric about the straight position the model has at 0°, the joint sits
off by a fixed amount everywhere, and redoing its range does not help. Run
calibrate and type `zero J5` (any arm joint, several at once too). The
MuJoCo view shows every joint at 0, the arm standing straight up: set that
joint (limp) exactly like it and press Enter. That spot becomes 0°; the
scale from its range is kept, and it prints how far off it was. A saved
home pose is shifted with it so it stays the same physical pose. Commands
are also clipped to the joint's measured ends, so after re-zeroing it can
never be driven into its hard stop, at the cost of a few degrees of range
at one end.

<a id="flip-joint"></a>**Flip a joint.** Run calibrate and type `flip J3`
(several at once: `flip J3 J4`; the gripper too). The saved min and max
trade places, so the joint turns the other way; nothing moves and nothing
else is asked. Then `check` again. A saved [home pose](#home-pose) is
updated with it, so it stays the same physical pose of the arm.

`lerobot-teleoperate` and `lerobot-record` load that file automatically when
given the same `--robot.id` (here `kikobot`); the log says
`Using calibration ...` on connect, and names any joint that has none. The servo offsets live in
the servos themselves, so the file and the arm belong together: after
swapping a servo, recalibrate that joint.

Without a calibration file the arm still runs: each servo's mid-scale
reading (2048) is taken as the joint's zero, which is how the arm was
assembled, and `--robot.inverted_joints` sets the directions.

**3. Teleoperate.**

```bash
$V/lerobot-teleoperate --robot.type=kikobot_follower --robot.port=/dev/ttyACM0 \
  --robot.id=kikobot --teleop.type=kikobot_gamepad
```

Start the sim server first (section 5) to see the sim arm move with it.
On connect the arm holds where it is (the goal is set to the measured
position before torque comes on), then moves to the [home pose](#home-pose)
(or the ready pose, if none is saved) at `soft_start_deg_s` (30°/s).

<a id="live-view"></a>**Sim mirror.** Every goal written to the servos is
sent to a sim server on the same tick, so the simulated arm moves with the
real one. `--robot.sim_backend=` picks which:

| Value | Mirrors into |
|---|---|
| `mujoco` (default) | `kikobot-sim-server`, port 5556 |
| `isaac` | the Isaac Sim server, port 5557 ([docs](docs/isaac-sim-teleop.md#mirror-the-real-arm-into-isaac-sim)) |
| `none` | nothing |

If the server is not running, or stops answering, the real arm carries on
and the log says so at most every 5 s. Where the real arm and the sim
differ, `--display_data=true` (LeRobot's own rerun view) plots each joint's
action (commanded) against its observation (measured): a joint that lags,
overshoots or moves the opposite way shows up there (then fix its direction
below, or its zero in step 2).

<a id="sag"></a>**Sag under load.** A position servo only pushes back in
proportion to how far it is off its goal, so a heavily loaded joint (J2,
which lifts the whole arm) settles low, and more so the further the arm
reaches out. For each joint in `--robot.sag_joints` (default `[J2]`), the
goal is raised by the joint's *give* times the gravity torque on it, which
comes from the MuJoCo model at the commanded pose. So the correction follows
the arm as it moves: none upright, most stretched out. The give (degrees
per N·m) is learned while the arm holds still, over about
`sag_time_constant_s` (0.5 s); hold the arm still out to the side for a
couple of seconds after connecting, and it is right in every pose from
then on. It is not learned during moves, so it adds no overshoot.

With `--display_data=true`, the corrected joint's action (commanded) and
observation (measured) lines should meet once the arm holds still for a
moment; the goal actually sent to the servo sits above them by the correction.

- **Measured still below commanded, though the log says the correction is at its cap:** the servo
  cannot hold the load. The log says `Sag correction at its cap` once the
  correction reaches `sag_max_deg` (15°). Raise it a little, but the real fix
  is torque: a stronger servo (STS3250 instead of STS3215), a counterbalance
  spring, a lighter forearm and gripper, or a stiffer 12 V supply.
- **Measured meets commanded, but the arm still looks lower than the sim:**
  the servo is where it was told; the droop is elsewhere. Either the arm
  flexes or the horn slips (play between the servo and the link, which the
  servo's encoder cannot see: tighten the horn screw, check for printed-part
  flex), or J2's zero is off: run calibrate, `check`, and compare J2 in the
  ready pose ([check the directions](#check-directions)); redo `J2` if needed.

Add J3 if it droops too: `--robot.sag_joints='[J2,J3]'`.

<a id="joint-direction"></a>**Joint direction.** Some servos are mounted so
that they turn opposite to the model. `--robot.inverted_joints` lists those
joints and defaults to `[J5]`, which is how the Kiko-Bot is built. To change
it, pass the full list: `--robot.inverted_joints='[J5,J3]'`, or
`--robot.inverted_joints='[]'` for none. It only applies without a
calibration: [calibration](#calibration) measures each joint's direction,
so a calibrated joint ignores this list. If a calibrated joint turns the
wrong way, its min and max were swapped: calibrate and type `flip <joint>`
([flip a joint](#flip-joint)).

Follower options (`--robot.<name>=`):

| Option | Default | Meaning |
|---|---|---|
| `port` | `COM3` / `/dev/ttyACM0` | Serial port |
| `baudrate` | `1000000` | Bus speed |
| `servo_ids` | `[1,2,3,4,5,6]` | Bus IDs of J1..J6 |
| `gripper_id` | `7` | Bus ID of the gripper servo; `null` if none is fitted |
| `gripper_open_deg` / `gripper_closed_deg` | `0` / `-60` | Gripper servo angle at 100 % / 0 % |
| `inverted_joints` | `[J5]` | Joints whose servo turns opposite to the model (uncalibrated only) |
| `cameras` | none | LeRobot cameras, e.g. `"{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"` |
| `max_velocity_deg_s` | `180` | Joint speed cap; targets under it pass through untouched |
| `soft_start_deg_s` | `30` | Speed of the first move after connecting |
| `p_coefficient` / `d_coefficient` / `i_coefficient` | `16` / `32` / `0` | Servo position-loop gains |
| `sag_joints` | `[J2]` | Joints whose droop under the arm's weight is corrected ([sag](#sag)); `'[]'` for none |
| `sag_time_constant_s` / `sag_max_deg` | `0.5` / `15` | How fast the correction builds up, and its cap |
| `disable_torque_on_disconnect` | `false` | Off by default so the arm does not drop |
| `sim_backend` | `mujoco` | Sim to mirror into: `mujoco`, `isaac`, `none` ([sim mirror](#live-view)) |
| `sim_host` / `sim_port` | `127.0.0.1` / per backend | Where that sim server is (5556 mujoco, 5557 isaac) |
| `sim_timeout_s` | `0.5` | Longest wait for a sim server reply, in the mirror's own thread: the real arm never waits for the sim |
| `calibration_view` | `true` | Pose the sim server's arm during calibration to show the pose to match |

<a id="6a-the-arm-server"></a>

### 6a. The arm server, and the servo limits page

`kikobot_follower` opens the servo bus inside `lerobot-teleoperate`, so
nothing else can reach the servos while it runs. `kikobot-arm-server` runs the
real arm as its own node instead, the way `kikobot-sim-server` runs the sim:
it opens the bus, holds the arm where it is, and serves the sim protocol on
port **5558**. Then any number of nodes connect to it:

- **the gamepad teleop** (or `lerobot-record`, or a policy) with
  `--robot.type=kikobot_arm`, exactly as with `kikobot_sim`;
- **`kikobot-servo-web`**, a web page that shows each servo's live load,
  current, voltage, temperature and protection flags, shows every current,
  torque and safety limit with the range it accepts, and sets them, on one
  servo or all at once, while the arm is being driven.

The server puts one request at a time on the bus, so a limits read never
collides with a goal. Between requests the servos hold the last goal: if the
teleop stops or drops, the arm stays where it is.

Terminal 1, the arm server (leave it running). It takes every
`kikobot_follower` option from the table above as `--robot.<name>=`:

```bash
$V/kikobot-arm-server --robot.port=/dev/ttyACM0 --robot.id=kikobot
```

Terminal 2, the gamepad:

```bash
$V/lerobot-teleoperate --robot.type=kikobot_arm \
  --teleop.type=kikobot_gamepad --fps=60
```

Terminal 3, the limits page, then open <http://127.0.0.1:8080>:

```bash
$V/kikobot-servo-web
```

On Windows: `& "$V\kikobot-arm-server.exe" --robot.port=COM3 --robot.id=kikobot`,
`& "$V\lerobot-teleoperate.exe" --robot.type=kikobot_arm --teleop.type=kikobot_gamepad --fps=60`
and `& "$V\kikobot-servo-web.exe"`.

The sim mirror still works: start `kikobot-sim-server` first and the arm
server mirrors every goal into it, as `kikobot_follower` does.

**Nodes on other machines.** Both servers listen on `127.0.0.1` only, by
default. To drive the arm or open the page from another machine (for
example the arm on a Raspberry Pi, the gamepad on a laptop):

```bash
# On the arm's machine:
$V/kikobot-arm-server --robot.port=/dev/serial0 --robot.id=kikobot --host=0.0.0.0
$V/kikobot-servo-web --host 0.0.0.0          # page at http://<arm machine's IP>:8080
# On the laptop:
$V/lerobot-teleoperate --robot.type=kikobot_arm --robot.host=<arm machine's IP> \
  --teleop.type=kikobot_gamepad --teleop.robot_id=kikobot --fps=60
# Or the page on the laptop, talking to the arm server over the network:
$V/kikobot-servo-web --arm-host <arm machine's IP>
```

Anyone who can reach these ports can move the arm and change its limits:
open them only on a network you trust. The gamepad's home pose (Y) is read
from the teleop machine's calibration folder; copy `kikobot_home.json` from
the arm's machine
(`~/.cache/huggingface/lerobot/calibration/robots/kikobot_follower/`) to
use the same home on the laptop.

**Without the arm server.** With nothing else holding the bus, the page can
open it itself: `$V/kikobot-servo-web --bus /dev/ttyACM0` (`--ids 1,2,3,4,5,6,7`
for other bus IDs). Do not do this while `kikobot_follower` or the arm server
runs: two programs on one serial port garble each other's packets.

**The limits.** Values are shown in units and sent raw; the page enforces
each one's range.

| Limit | Range | Stored | What it does |
|---|---|---|---|
| Torque limit (live) | 0–100 % | RAM | The torque cap in force now. Applies at once; back to Max torque at power-up |
| Max torque | 0–100 % | EEPROM | Torque cap loaded at power-up |
| Protection current | 0–3322 mA (6.5 mA steps) | EEPROM | Current above which over-current protection starts timing |
| Over-current time | 0–2540 ms | EEPROM | How long the current may stay above it before torque is cut |
| Overload torque | 0–100 % | EEPROM | Load above which overload protection starts timing |
| Overload time | 0–2540 ms | EEPROM | How long the load may stay above it before protection trips |
| Protective torque | 0–100 % | EEPROM | Torque left once overload protection has tripped |
| Max temperature | 0–100 °C | EEPROM | Over-temperature threshold |
| Max / Min voltage | 0–25.4 V | EEPROM | Over- and under-voltage thresholds |

The servo's own angle limits and its protection mask are shown but read-only:
calibration relies on the angle limits. Every write is read back, and the
page says so if a servo kept its old value. To try a lower torque safely,
change **Torque limit (live)** first: a power cycle undoes it. Set **Max
torque** once you are happy with it. A joint with too little torque sags or
stalls under the arm's weight, J2 first.

Arm server options: `--host` (`127.0.0.1`), `--port` (`5558`), and every
`--robot.<name>=` above. `kikobot_arm` options: those of `kikobot_sim`
([5](#5-the-sim-server)), with `port` `5558`, `camera_names` `[]` (keys of
the server's `--robot.cameras`) and `reset_on_connect` `false`. As with
`kikobot_follower`, the gamepad's Back button respawns nothing on the real
arm; it only sends it home under the speed cap. `kikobot-servo-web` options:
`--host` / `--port` (`127.0.0.1` / `8080`) for the page, `--arm-host` /
`--arm-port` (`127.0.0.1` / `5558`) for the arm server, or `--bus` / `--ids`
/ `--baudrate` to open the bus itself.

---

## 7. Speed and smoothness

Each `send_action` writes its goal straight to all servos in one
sync-write packet, as LeRobot's SO-101 does, and sends the same command to
the sim server; only a joint speed cap sits in between. The smoothness of
the motion comes from the gamepad (low-passed sticks, a minimum-jerk glide
home on Y) and from the loop rate:

| To change | Option |
|---|---|
| Teleop loop rate | `--fps=60` (more goals per second, smaller steps; the servo bus easily takes 100) |
| How quickly the arm starts and stops with the stick | `--teleop.input_smoothing_s` (0.12) and `--teleop.stop_rate_per_s` (4) |
| Top speed | `--teleop.linear_speed_m_s` (0.12), `--teleop.angular_speed_deg_s` (90) |
| Safety cap on any joint | `--robot.max_velocity_deg_s` (180) |
| First move after connecting | `--robot.soft_start_deg_s` (30) |

If the arm wobbles after a quick stop, lower the teleop speeds or
`--teleop.stop_rate_per_s` (a gentler stop); if it buzzes holding still, lower
`--robot.p_coefficient` ([troubleshooting](#10-troubleshooting)).

---

## 8. Record a dataset

On the real arm, LeRobot's own recorder works as-is:

```bash
$V/lerobot-record --robot.type=kikobot_follower --robot.port=/dev/ttyACM0 --robot.id=kikobot \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=kikobot_gamepad \
  --dataset.repo_id=local/kikobot_real --dataset.root=datasets/kikobot_real \
  --dataset.single_task="Pick up the red cube and place it on the green pad" \
  --dataset.num_episodes=20 --dataset.push_to_hub=false
```

The same works with `--robot.type=kikobot_sim` and `kikobot_isaac`, with
their sim server running. LeRobot
has no hook to reset a sim scene between episodes, so press **Back / View**
on the gamepad during the reset period: the arm goes home and the object
and pad respawn. Training and running a policy are in
[docs/isaac-sim-record-and-train.md](docs/isaac-sim-record-and-train.md);
the commands are the same for `kikobot_sim`.

Without a human: `kikobot-datagen record` has a scripted expert do the desk
tasks and keeps only the episodes that pass every check, then `kikobot-datagen
convert` makes RLDS datasets for OpenVLA. On the cluster it is three `sbatch`
jobs: [docs/datagen-openvla.md](docs/datagen-openvla.md).

---

## 9. Platform notes

| | Ubuntu | Windows | macOS |
|---|---|---|---|
| Serial port | `/dev/ttyACM0`, `/dev/ttyUSB0` or `/dev/kikobot`; `/dev/serial0` on a Pi with the Driver HAT ([below](#pi-hat)) | `COM3` etc. | `/dev/tty.usbmodem*` |
| Port permission | `dialout` group or the udev rule | none | none |
| Sim server window | `kikobot-sim-server` | `kikobot-sim-server.exe` | `mjpython -m lerobot_robot_kikobot.sim_server` |
| Gamepad | needs a desktop session (GLFW) | works | works |
| USB latency | FTDI: the udev rule sets 1 ms | FTDI: Device Manager → Port Settings → Advanced → Latency Timer = 1 | n/a |

The sim server paces physics with ZMQ polls and `time.perf_counter`, accurate
to about 1 ms on all three.

<a id="pi-hat"></a>

### Raspberry Pi + Waveshare Bus Servo Driver HAT

The Waveshare **Bus Servo Driver HAT (A)** works in place of the Bus Servo
Adapter / Feetech URT-1: it drives the same STS servos with the same protocol
at the same 1 Mbps. (The HAT (B) is for SC-series servos, not the STS3215.)
Its UART jumper picks how it is driven:

- **Jumper on A (USB):** plug the HAT's USB port into any PC. It has the same
  CH343 chip as the adapter, so it shows up as `/dev/ttyACM0` / `COM3` and
  everything above applies unchanged, udev rule included.
- **Jumper on B (Raspberry Pi GPIO):** the Pi drives the bus through its GPIO
  UART, `/dev/serial0`. Set the Pi up once:

```bash
# Serial port on, login shell over serial off: raspi-config → Interface
# Options → Serial Port → "login shell" No, "serial port hardware" Yes.
sudo raspi-config nonint do_serial_hw 0
sudo raspi-config nonint do_serial_cons 1
# Pi 3 / 4 / Zero 2 W only: give the GPIO pins the full UART instead of the
# mini UART, whose baud rate drifts with the CPU clock. Bluetooth goes off.
# (Pi 5 needs nothing here.)
echo dtoverlay=disable-bt | sudo tee -a /boot/firmware/config.txt
sudo systemctl disable --now hciuart
sudo usermod -aG dialout $USER
sudo reboot
```

Then run the follower with `--robot.port=/dev/serial0` (or leave
`--robot.port` out: with no USB adapter plugged in, `/dev/serial0` is the
default). The rest of the setup is the Ubuntu one, on Raspberry Pi OS
64-bit: run the gamepad teleop from the Pi's desktop, not over bare SSH.
The MuJoCo sim mirror is heavy for a Pi: run `kikobot-sim-server` on another
machine and point the arm at it with `--robot.sim_host=<that machine's IP>`,
or turn the mirror off with `--robot.sim_backend=none`.

Power the servos from the HAT's DC input with the supply they need (12 V
for the 12 V STS3215). The HAT also powers the Pi from it, so do not plug a
second supply into the Pi. A sagging supply shows up as servos that stop
answering.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `kikobot_*` is not a valid type | The plugin failed to import, or the old KikoBot plugin with the same name is installed. Run the check in [3](#3-install). |
| `Could not open /dev/ttyACM0` | Wrong port (`lerobot-find-port`), or not in `dialout` (log out and in after `usermod`). `/dev/serial0` on a Pi: the serial port is off or the serial console holds it ([Driver HAT](#pi-hat)). |
| `There is no status packet!` | A servo stopped replying after answering the ping. Transactions are retried and sync read falls back to single reads automatically; if it persists, the full error names the servo: check its cable and the supply (a sagging 12 V drops replies). Waveshare adapter, or Driver HAT on USB: jumper on **A**; Driver HAT on the Pi's GPIO: jumper on **B**, and on a Pi 3/4 the `disable-bt` overlay ([Driver HAT](#pi-hat)). |
| `No sim server at tcp://127.0.0.1:5556` | Start `kikobot-sim-server` first (or the Isaac server with `--robot.sim_backend=isaac`). The real arm runs without it; `--robot.sim_backend=none` silences the warning. |
| `No reply from the sim server` (kikobot_sim / kikobot_isaac) | The server is not running, or is on another port: check `--robot.port`. |
| `No reply from the sim server at tcp://…:5558` (kikobot_arm) | Start `kikobot-arm-server` first. On another machine: `--host=0.0.0.0` on the server, `--robot.host=<its IP>` on the teleop. |
| Limits page says `no arm` | `kikobot-arm-server` is not running, or is elsewhere: `kikobot-servo-web --arm-host <IP>`. |
| A limit does not change ("kept … at …") | That servo's firmware refused the write. Some only take EEPROM writes with torque off: stop the arm server and use `kikobot-servo-web --bus <port>`. |
| No rerun window opens with `--display_data=true` | `rerun-sdk` comes with LeRobot's `core_scripts` extra; otherwise `pip install rerun-sdk`. |
| The connection drops during calibration | Calibration is the only time every joint reaches its ends, which pulls the daisy-chained servo cables tight: a half-seated connector drops that servo and every one after it. Reseat the cables, leave slack at each joint (J6 turns ±175°), move joints slowly, and press Enter to retry at the prompt. If it is the USB adapter that drops (the port vanishes), on Ubuntu turn off USB autosuspend: `echo -1 \| sudo tee /sys/module/usbcore/parameters/autosuspend`. |
| `Servos [...] did not answer` | The bus has no 12 V, a cable is loose, or `--robot.servo_ids` is wrong. `[7]` alone: no gripper servo on the bus; pass `--robot.gripper_id=null`. |
| `glfw.init() failed` | No desktop session (e.g. bare SSH on Ubuntu). Run from the desktop. |
| `No gamepad found` | Plug the controller in, then restart (it is only detected at startup). The log lists every joystick GLFW sees; on Ubuntu, if none appear, run `sudo usermod -aG input $USER` and log in again. Xbox pads unknown to GLFW get the xpad layout automatically. |
| Wrong device used as the gamepad | Pick its slot from the `Joystick slot N` log lines: `--teleop.joystick=N`. |
| `Settings schema 'org.gnome.settings-daemon.plugins.xsettings' does not contain a key named 'antialiasing'` (Ubuntu) | Launched from a Snap app's terminal (e.g. Snap VS Code). Run from a normal terminal, or install VS Code from the `.deb`. |
| Gamepad moves come out in the wrong directions (e.g. X/Y moves the arm up and down) | One or more joints are calibrated the wrong way round. Run calibrate, type `check`, and compare with the sim's ready pose ([check the directions](#check-directions)); `flip` each joint with the wrong sign. |
| A joint moves the wrong way (opposite to the sim) | Compare the real arm with the sim window, or with `--display_data=true` (a joint's measured line moves opposite to its commanded line). Calibrated: calibrate and type `flip <joint>` ([flip a joint](#flip-joint)). Uncalibrated: list it in `--robot.inverted_joints` (default `[J5]`), e.g. `'[J5,J3]'` ([6](#joint-direction)). |
| Real arm stops short of, or overshoots, the sim's pose | Redo [calibration](#calibration) with each joint pushed fully to the ends the model shows. |
| Arm wobbles after stopping | Lower the teleop speeds or `--teleop.stop_rate_per_s` ([7](#7-speed-and-smoothness)). |
| Real arm lags behind the sim | Run teleop at `--fps=60`; check the servos got P = 16 (the log at connect); a loaded joint lags most, see [sag](#sag). |
| J2 (or another joint) hangs lower than the sim | Droop under load: corrected for `--robot.sag_joints` (default `[J2]`) while the arm holds still ([sag](#sag)). Add others, e.g. `'[J2,J3]'`. |
| A joint sits off the sim by the same amount everywhere, even after calibration | Its stops are not symmetric about its zero: calibrate and type `zero <joint>` ([re-zero a joint](#zero-joint)). |
| A tiny stick nudge makes the arm jump, then creep the rest of the way | Stick-slip in the servo: with a soft position gain, a slowly creeping goal does not move the servo until the gear friction gives, then it jumps, and the soft gain makes the last bit slow. Raise the gain: `--robot.p_coefficient=24` (try 32; lower it again if a joint buzzes). Releasing the stick stops the motion within `--teleop.stop_rate_per_s`. |
| Servo buzzes while holding | Lower `--robot.p_coefficient` (try 12). |
| `25 position reads in a row failed` | The bus failed repeatedly: check power and cable. The log has the cause. |

---

## 11. Commit messages

One line: a lowercase type, a colon, then a short summary of what changed.

| Type | For |
|---|---|
| `feat:` | New behaviour |
| `fix:` | A bug fix |
| `add:` | New files or assets (models, scenes, rules) |
| `update:` | Changes to existing assets or settings |
| `docs:` | README and comments only |
| `refactor:` | Restructuring with no change in behaviour |
| `test:` | Tests only |
| `chore:` | Tooling, ignores, housekeeping |

Example: `feat: write servo goals straight to the bus like the SO-101`
