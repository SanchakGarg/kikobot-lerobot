# Isaac Sim troubleshooting

| Problem | Fix |
|---|---|
| `No reply from the Isaac Sim server at tcp://127.0.0.1:5556` | **Is the server running?** It must say `Serving Kiko-Bot on ...` before you start LeRobot. **First connect after a fresh install?** Shader compilation can take minutes; raise `--robot.connect_timeout_s=600`. **Remote?** Check the SSH tunnel is up. |
| `models/usd/kikobot.usd not found` | Run `isaac\convert_mjcf.py` with Isaac Sim's Python ([setup §3](isaac-sim-setup.md#3-convert-the-arm-to-usd)). |
| `ModuleNotFoundError: zmq` in the server | Install it in the Isaac Sim env: `& "$I\pip.exe" install pyzmq`. |
| `ModuleNotFoundError: isaacsim` | You started the server with LeRobot's Python. Use `isaacsim-env`'s `python.exe`. |
| `invalid choice: 'kikobot_isaac'` in a LeRobot command | The plugin in LeRobot's venv is older than this guide. `git pull`, then `pip install -e .` into LeRobot's venv. |
| `Server speaks protocol N, this plugin M` | The two sides are from different commits. `git pull` on both machines. |
| Isaac Sim install fails with a path error on Windows | Enable long paths ([setup §2](isaac-sim-setup.md#2-isaac-sim-6-with-newton)), or create the venv at a short path such as `C:\isim`. |
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
