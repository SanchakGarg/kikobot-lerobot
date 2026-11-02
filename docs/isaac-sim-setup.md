# Isaac Sim setup

You will end up with two Python environments on the GPU machine:

| Environment | Holds | Runs |
|---|---|---|
| `isaacsim-env` | Isaac Sim 6 + pyzmq | `isaac/kikobot_isaac_server.py`, `isaac/convert_mjcf.py` |
| LeRobot's `.venv` | LeRobot + this plugin | `lerobot-teleoperate`, `lerobot-record`, `lerobot-rollout`, ... |

Both use **Python 3.12**. Keep them separate: Isaac Sim pins its own
versions of many packages.

## Requirements

- **GPU:** an NVIDIA RTX card (RTX 20xx or newer, or RTX Ada/Blackwell
  workstation cards). An RTX 4060 Laptop with 8 GB runs this scene; it is at
  the bottom of Isaac Sim's range, so close other GPU-heavy apps.
  Datacenter GPUs with no RT cores (B200, H100, A100) cannot render cameras;
  use them for [training](slurm-training.md) only.
- **Driver:** a recent NVIDIA driver. The compatibility checker below tells
  you if yours is too old.
- **OS:** Windows 10/11 or Ubuntu 22.04/24.04 (GLIBC 2.35+). Isaac Sim does
  not run on macOS. A Mac can still drive a remote Isaac Sim, see
  [Remote server](#remote-server-optional).
- **Disk:** about 30 GB for Isaac Sim and its extension cache.
- **Gamepad:** Xbox or PlayStation, plugged in before you start teleop.

## 1. LeRobot and this plugin

If you followed the main README's quick start, this is done. Otherwise, on
Windows (PowerShell):

```powershell
git clone https://github.com/huggingface/lerobot.git $HOME\lerobot
cd $HOME\lerobot
py -3.12 -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -e ".[core_scripts,training,feetech]"

git clone https://github.com/SanchakGarg/kikobot-lerobot.git $HOME\kikobot-lerobot
cd $HOME\kikobot-lerobot
& "$HOME\lerobot\.venv\Scripts\pip.exe" install -e .
```

Already installed? Update and reinstall the plugin so it picks up `pyzmq`:

```powershell
cd $HOME\kikobot-lerobot
git pull
& "$HOME\lerobot\.venv\Scripts\pip.exe" install -e .
```

On Ubuntu, the same with `python3.12 -m venv`, `.venv/bin/pip` and `~/` paths.

## 2. Isaac Sim 6 (with Newton)

Isaac Sim installs from pip. Newton ships with it; the server switches to it
at startup, so there is no separate install.

**Windows (PowerShell):**

```powershell
# Once, as administrator: long paths, or the install fails partway.
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

py -3.12 -m venv $HOME\isaacsim-env
$I = "$HOME\isaacsim-env\Scripts"
& "$I\python.exe" -m pip install --upgrade pip
& "$I\pip.exe" install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
& "$I\pip.exe" install "isaacsim[all,extscache]==6.0.0.1" --extra-index-url https://pypi.nvidia.com
& "$I\pip.exe" install pyzmq opencv-python-headless
```

**Ubuntu:**

```bash
python3.12 -m venv ~/isaacsim-env
I=~/isaacsim-env/bin
$I/pip install --upgrade pip
$I/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
$I/pip install "isaacsim[all,extscache]==6.0.0.1" --extra-index-url https://pypi.nvidia.com
$I/pip install pyzmq opencv-python-headless
```

The `[all,extscache]` extras matter: plain `pip install isaacsim` installs an
empty shell that imports fine and then fails at runtime. A newer 6.x release
should work too. The Isaac-specific code is all in one class,
`IsaacNewtonBackend` in `isaac/kikobot_isaac_server.py`.

**Accept the licence once** (the server also sets this for you):

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"      # Ubuntu: export OMNI_KIT_ACCEPT_EULA=YES
```

**Check the install:**

```powershell
& "$I\isaacsim.exe" isaacsim.exp.compatibility_check
```

It lists your GPU, driver and VRAM, and flags anything below the minimum.
The first launch of anything Isaac takes several minutes while it builds its
shader cache; later starts take well under a minute.

**Laptops with two GPUs:** make Windows run Isaac Sim on the RTX card.
Settings → System → Display → Graphics → add
`%USERPROFILE%\isaacsim-env\Scripts\python.exe` → Options → High performance.

## 3. Convert the arm to USD

Isaac Sim loads USD, not MJCF. Convert `models/robot.xml` once, and again
whenever it changes:

```powershell
cd $HOME\kikobot-lerobot
& "$I\python.exe" isaac\convert_mjcf.py
```

This writes `models/usd/kikobot.usd` (and the files next to it). It is
git-ignored, so convert again on every machine.

The importer's joint gains are not used. The server sets each joint's
drive from the MJCF servo model (stiffness 200 N·m/rad, torque limit
2.94 N·m, gripper 800 N/m and 20 N) every time it loads the arm. Change them
with `--kp`, `--kd`, `--max-torque` and the `--finger-*` options.

## 4. Smoke test

Terminal 1, the sim:

```powershell
cd $HOME\kikobot-lerobot
& "$I\python.exe" isaac\kikobot_isaac_server.py
```

Wait for `Serving Kiko-Bot on tcp://127.0.0.1:5556`. The Isaac Sim window
stays empty until a client connects.

Terminal 2, LeRobot, with the gamepad plugged in:

```powershell
& "$HOME\lerobot\.venv\Scripts\lerobot-teleoperate.exe" `
  --robot.type=kikobot_isaac --teleop.type=kikobot_gamepad
```

The server log shows `Scene 'pick_place' ready`, and the arm, desk, red cube
and green pad appear. The left stick moves the gripper. Check, in order:

1. **Server log:** it shows `Physics variants [...]` and the dof list, and the
   dofs include J1..J6 and `finger_l`.
2. **Starting pose:** the arm starts in the ready pose (bent, jaws pointing
   down), not stretched straight up.
3. **Gripper:** holding A closes the jaws, and holding B opens them.
4. **No drift:** the arm holds still when you let go. If it sags or
   oscillates, see [Troubleshooting](isaac-sim-troubleshooting.md) ("The arm sags, oscillates or is sluggish").
5. **Camera:** run once with `--display_data=true` on the teleop. Rerun then
   shows the `record` camera looking at the arm from in front of the desk.

## Remote server (optional)

Any LeRobot machine can drive an Isaac Sim server on another machine, for
example a Mac (where Isaac Sim cannot run) driving a Linux RTX workstation.

On the Isaac machine:

```bash
~/isaacsim-env/bin/python isaac/kikobot_isaac_server.py --headless --livestream
```

On the LeRobot machine, forward the port over SSH and connect through it:

```bash
ssh -N -L 5556:127.0.0.1:5556 user@isaac-machine &
lerobot-teleoperate --robot.type=kikobot_isaac --robot.image_encoding=jpeg \
  --teleop.type=kikobot_gamepad
```

- **Bandwidth:** `jpeg` makes each frame about 20 times smaller. Use it on
  any link slower than gigabit Ethernet.
- **Watching the sim:** `--livestream` lets you see the headless sim in
  NVIDIA's *Isaac Sim WebRTC Streaming Client*, pointed at the Isaac
  machine's address.
- **Security:** the server listens only on 127.0.0.1 by default, and the
  protocol has no authentication. `--bind tcp://*:5556` would accept anyone
  on the network, so keep to the SSH tunnel.
