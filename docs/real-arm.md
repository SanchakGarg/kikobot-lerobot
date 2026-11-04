# The real arm

Everything about driving the physical Kiko-Bot: finding the port,
calibration, teleop, the arm server and the servo limits page, the
Raspberry Pi HAT, and tuning. The short version is in the
[main README](../README.md#run-the-real-arm).

The commands use `$V` for the LeRobot venv's scripts folder:
`V=~/lerobot/.venv/bin` on Ubuntu/macOS, `$V="$HOME\lerobot\.venv\Scripts"` on
Windows (then run tools as `& "$V\lerobot-teleoperate.exe"`, and continue
lines with a backtick instead of `\`).

## 1. Find the port

`$V/lerobot-find-port`. Usually `COM3` on Windows,
`/dev/ttyACM0` (or `/dev/ttyUSB0`) on Ubuntu, or `/dev/kikobot` with the udev
rule installed. On Linux, leaving out `--robot.port` uses the first of
`/dev/kikobot`, `/dev/ttyACM0`, `/dev/ttyUSB0` and `/dev/serial0` that exists.
A Waveshare Bus Servo Driver HAT on a Raspberry Pi has its own steps:
[Raspberry Pi + Driver HAT](#pi-hat).

<a id="calibration"></a>

## 2. Calibrate (once)

This matches the real
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

## 3. Teleoperate

```bash
$V/lerobot-teleoperate --robot.type=kikobot_follower --robot.port=/dev/ttyACM0 \
  --robot.id=kikobot --teleop.type=kikobot_gamepad
```

Start the sim server first ([sim](sim.md)) to see the sim arm move with it.
On connect the arm holds where it is (the goal is set to the measured
position before torque comes on), then moves to the [home pose](#home-pose)
(or the ready pose, if none is saved) at `soft_start_deg_s` (30°/s).

<a id="live-view"></a>**Sim mirror.** Every goal written to the servos is
sent to a sim server on the same tick, so the simulated arm moves with the
real one. `--robot.sim_backend=` picks which:

| Value | Mirrors into |
|---|---|
| `mujoco` (default) | `kikobot-sim-server`, port 5556 |
| `isaac` | the Isaac Sim server, port 5557 ([docs](isaac-sim-teleop.md#mirror-the-real-arm-into-isaac-sim)) |
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

### Follower options

Set as `--robot.<name>=`:

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

## 4. The arm server and the servo limits page

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
([sim](sim.md)), with `port` `5558`, `camera_names` `[]` (keys of
the server's `--robot.cameras`) and `reset_on_connect` `false`. As with
`kikobot_follower`, the gamepad's Back button respawns nothing on the real
arm; it only sends it home under the speed cap. `kikobot-servo-web` options:
`--host` / `--port` (`127.0.0.1` / `8080`) for the page, `--arm-host` /
`--arm-port` (`127.0.0.1` / `5558`) for the arm server, or `--bus` / `--ids`
/ `--baudrate` to open the bus itself.

## 5. Speed and smoothness

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
`--robot.p_coefficient` ([troubleshooting](troubleshooting.md)).

## 6. Recording on the real arm

LeRobot's own recorder works as-is. Add a real camera with `--robot.cameras`:

```bash
$V/lerobot-record --robot.type=kikobot_follower --robot.port=/dev/ttyACM0 --robot.id=kikobot \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=kikobot_gamepad \
  --dataset.repo_id=local/kikobot_real --dataset.root=datasets/kikobot_real \
  --dataset.single_task="Pick up the red cube and place it on the green pad" \
  --dataset.num_episodes=20 --dataset.push_to_hub=false
```

For sims and scripted demos see the [main README](../README.md#record-datasets).

## 7. Platforms

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

