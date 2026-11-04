# Troubleshooting

## Sim and real arm

| Problem | Fix |
|---|---|
| `kikobot_*` is not a valid type | The plugin failed to import, or the old KikoBot plugin with the same name is installed. Run the check in the [README](../README.md#install). |
| `Could not open /dev/ttyACM0` | Wrong port (`lerobot-find-port`), or not in `dialout` (log out and in after `usermod`). `/dev/serial0` on a Pi: the serial port is off or the serial console holds it ([Driver HAT](real-arm.md#pi-hat)). |
| `There is no status packet!` | A servo stopped replying after answering the ping. Transactions are retried and sync read falls back to single reads automatically; if it persists, the full error names the servo: check its cable and the supply (a sagging 12 V drops replies). Waveshare adapter, or Driver HAT on USB: jumper on **A**; Driver HAT on the Pi's GPIO: jumper on **B**, and on a Pi 3/4 the `disable-bt` overlay ([Driver HAT](real-arm.md#pi-hat)). |
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
| Gamepad moves come out in the wrong directions (e.g. X/Y moves the arm up and down) | One or more joints are calibrated the wrong way round. Run calibrate, type `check`, and compare with the sim's ready pose ([check the directions](real-arm.md#check-directions)); `flip` each joint with the wrong sign. |
| A joint moves the wrong way (opposite to the sim) | Compare the real arm with the sim window, or with `--display_data=true` (a joint's measured line moves opposite to its commanded line). Calibrated: calibrate and type `flip <joint>` ([flip a joint](real-arm.md#flip-joint)). Uncalibrated: list it in `--robot.inverted_joints` (default `[J5]`), e.g. `'[J5,J3]'` ([joint direction](real-arm.md#joint-direction)). |
| Real arm stops short of, or overshoots, the sim's pose | Redo [calibration](real-arm.md#calibration) with each joint pushed fully to the ends the model shows. |
| Arm wobbles after stopping | Lower the teleop speeds or `--teleop.stop_rate_per_s` ([speed](real-arm.md#5-speed-and-smoothness)). |
| Real arm lags behind the sim | Run teleop at `--fps=60`; check the servos got P = 16 (the log at connect); a loaded joint lags most, see [sag](real-arm.md#sag). |
| J2 (or another joint) hangs lower than the sim | Droop under load: corrected for `--robot.sag_joints` (default `[J2]`) while the arm holds still ([sag](real-arm.md#sag)). Add others, e.g. `'[J2,J3]'`. |
| A joint sits off the sim by the same amount everywhere, even after calibration | Its stops are not symmetric about its zero: calibrate and type `zero <joint>` ([re-zero a joint](real-arm.md#zero-joint)). |
| A tiny stick nudge makes the arm jump, then creep the rest of the way | Stick-slip in the servo: with a soft position gain, a slowly creeping goal does not move the servo until the gear friction gives, then it jumps, and the soft gain makes the last bit slow. Raise the gain: `--robot.p_coefficient=24` (try 32; lower it again if a joint buzzes). Releasing the stick stops the motion within `--teleop.stop_rate_per_s`. |
| Servo buzzes while holding | Lower `--robot.p_coefficient` (try 12). |
| `25 position reads in a row failed` | The bus failed repeatedly: check power and cable. The log has the cause. |

## Isaac Sim

| Problem | Fix |
|---|---|
| `No reply from the sim server at tcp://127.0.0.1:5557` | **Is the server running?** It must say `Serving Kiko-Bot on ...` before you start LeRobot. **First connect after a fresh install?** Shader compilation can take minutes; raise `--robot.connect_timeout_s=600`. **Remote?** Check the SSH tunnel is up. |
| `models/usd/kikobot.usd not found` | Run `isaac\convert_mjcf.py` with Isaac Sim's Python ([setup §3](isaac-sim-setup.md#3-convert-the-arm-to-usd)). |
| `ModuleNotFoundError: zmq` in the server | Install it in the Isaac Sim env: `& "$I\pip.exe" install pyzmq`. |
| `ModuleNotFoundError: isaacsim` | You started the server with LeRobot's Python. Use `"$W\isaacsim-env\Scripts\python.exe"` (`$I` in the guides). |
| `invalid choice: 'kikobot_isaac'` in a LeRobot command | The plugin in LeRobot's venv is older than this guide. `git pull`, then `pip install -e .` into LeRobot's venv. |
| `Server speaks protocol N, this plugin M` | The two sides are from different commits. `git pull` on both machines. |
| Isaac Sim install fails with a path error on Windows | Enable long paths ([setup §2](isaac-sim-setup.md#2-isaac-sim-6-with-newton)), or move `lerobot`, `kikobot-lerobot` and `isaacsim-env` into a short folder such as `C:\kb` (recreating the venvs there). |
| `Isaac Sim could not switch to the Newton engine` | The Newton extensions did not load: reinstall with `isaacsim[all,extscache]`. The log lists the engines that are available. |
| `Joints [...] are not in the articulation` | The USD is stale or failed to import. Run `convert_mjcf.py` again and read its log. The server log lists the dofs it found. |
| The arm falls limp or flies apart at start | **USD variant:** the imported USD's physics variant does not suit Newton. The log shows `Physics variants [...]`; try another with `--physics-variant` (e.g. `physics`, `mujoco`). **Reversed joints:** Newton also refuses those ("Reversed joints are not supported"). Re-convert after updating Isaac Sim. |
| The arm sags, oscillates or is sluggish | **Oscillates:** raise `--kd` (try 5). **Sags under gravity:** raise `--kp` (try 300); the real servo is stiffer than its torque limit suggests. **Moves sluggishly:** check `--max-torque` (2.94 N·m is the STS3215's stall torque). These are server options. Restart the server to apply them. |
| The cube slips out of the jaws | Raise `--finger-max-force` (try 30) and `--finger-kp`. Approach slower in precision mode (X). |
| Black camera images for the first frame or two | Normal while the renderer warms up (the server logs it once). If they stay black: the laptop is rendering on the integrated GPU. Set python.exe to High performance ([setup §2](isaac-sim-setup.md#2-isaac-sim-6-with-newton)). |
| Out of GPU memory / Isaac Sim crashes on the 4060 | Run the server with `--headless`. Close browsers and games. Keep one 640×480 camera. Don't train on the laptop while the server runs. |
| Teleop feels slow (under 30 fps) | **GPU rendering:** `--headless` helps, or a lower resolution (`--robot.width=320 --robot.height=240`; record the dataset at that size too). **Image transfer, remote server:** `--robot.image_encoding=jpeg`. Motion is unaffected either way; the sim waits for each tick. |
| The scene does not reset between recorded episodes | Press **Back / View** on the gamepad during the reset period. LeRobot has no reset hook of its own. |
| Training on the cluster: `no kernel image is available` / `sm_100` missing | Torch is not a CUDA 12.8+ build. Reinstall from the `cu128` index ([SLURM §1](slurm-training.md#1-one-time-setup-on-the-cluster)). |
| A policy works in MuJoCo but not in Isaac, or the other way | The two renderers and solvers differ. Train on data from the backend you run on, or on both together. |
