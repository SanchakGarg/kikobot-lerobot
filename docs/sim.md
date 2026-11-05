# The MuJoCo sim

`kikobot-sim-server` runs the arm in MuJoCo, in real time, behind a socket.
`--robot.type=kikobot_sim` connects LeRobot to it. The Isaac Sim server
speaks the same protocol and builds the same room ([Isaac Sim teleop](isaac-sim-teleop.md)).

The commands use `$V` for the LeRobot venv's scripts folder:
`V=~/lerobot/.venv/bin` on Ubuntu/macOS, `$V="$HOME\lerobot\.venv\Scripts"` on
Windows (run tools as `& "$V\kikobot-sim-server.exe"` there).

## 1. Start the server

Start it first and leave it running:

```bash
$V/kikobot-sim-server --scene desk_sort
```

- **macOS:** MuJoCo's window needs `mjpython`:
  `$V/mjpython -m lerobot_robot_kikobot.sim_server --scene desk_sort`.
- **No display:** add `--no-viewer`.

Physics runs in real time from the moment the server starts. A client's
`command` sets the joint targets, and physics moves the arm there, the same
way the servos do with the same goal. The log prints the real-time factor
every 10 s.

Drive it from a second terminal:

```bash
$V/lerobot-teleoperate --robot.type=kikobot_sim --teleop.type=kikobot_gamepad --fps=60
```

`--display_data=true` shows the `record` camera and the joints in rerun.

| Server option | Default | What it does |
|---|---|---|
| `--scene` | `pick_place` | `desk_sort`, `pick_place`, `cylinder`, `empty`, or a path to an MJCF file (used as is, without the room) |
| `--host` / `--port` | `127.0.0.1` / `5556` | Where to listen. Use `0.0.0.0` to accept other machines |
| `--window` | `mujoco` | `mujoco`, `newton` (Newton's ViewerGL) or `none`. `--no-viewer` is the same as `none` |
| `--seed` | random | The same seed gives the same sequence of desks and spawns |
| `--timestep` | 0.002 | Physics step, seconds |
| `--fixed-lighting` | off | The same light on every reset |
| `--engine` / `--renderer` | `auto` | See [Newton](#4-newton-physics-and-ray-traced-cameras) |

| Robot option (`--robot.<name>=`) | Default | What it does |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `5556` | Where the server is |
| `camera_names` | `[record]` | The scene's one camera |
| `width` / `height` | 640 / 480 | Image size |
| `image_encoding` | `raw` | `jpeg` is about 20× smaller, for use over a network |
| `reset_on_connect` | `true` | Start from a fresh scene |
| `max_velocity_deg_s` | 180 | Joint speed cap, the same as the real arm's |
| `timeout_s` | 5 | How long to wait for the server |

## 2. The room and the desk

The room is described once, in `lerobot_robot_kikobot/env/room.py`. The
MuJoCo server builds it with `env/mujoco_scene.py`, and the Isaac server
with `isaac/room_usd.py`, so both have the same geometry, colours and
lights.

**The desk** is a study desk with a hutch:
- A walnut-laminate top, 1.5 m wide and 0.95 m deep, 75 cm high.
- Two side panels. Each runs the full depth of the desk at the bottom and
  narrows to the shelf's depth at the top.
- A top shelf 30 cm deep, 75 cm above the desktop.
- A tube light under the front edge of the shelf.

The arm stands **flush in the desk's front-left corner**: one edge of its
20 cm base plate lies on the desk's front edge, and the other against the
left side panel. In its rest pose it faces into the desk, and its blind spot
(where J1's ±165° range runs out) points off the desk, at the person. Its
reach of about 40 cm leaves the back and the right of the desk out of reach.

Around the desk is a makerspace: a workbench with a pegboard and a
3D printer, steel shelving with bins, a whiteboard, a window and a stool.

**The lights** follow a real room:
- 8 small downlights in two rows of 4, cool white.
- The tube light under the hutch (6500 K), which casts the shadows on the
  desk.

On every reset the light changes slightly:
- The downlights' dimmer level and colour temperature (5000–6500 K, all
  alike).
- Now and then a downlight is off; at least two stay on.
- The tube is on in most episodes, a little brighter or dimmer each time.

`--fixed-lighting` turns the variation off.

**The camera.** There is one camera, `record` (`RECORD_CAMERA` in
`room.py`). It sits low over the desk, on the diagonal through the arm's
corner, and looks back at the arm across the items and stations. To match
it on the real desk, measured from the centre of the arm's base, put the
camera:
- 58 cm to the right and 58 cm back (toward the hutch);
- 34 cm above the desktop;
- aimed at a point 7 cm right of, 7 cm behind and 10 cm above the base
  centre;
- with a 58° vertical field of view.

## 3. The scenes

| Scene | On the desk |
|---|---|
| `desk_sort` | The stationery sorting station (below) |
| `pick_place` | A red 32 mm cube and a green place pad, respawned on each reset |
| `cylinder` | An orange cylinder lying on its side, and the pad |
| `empty` | Nothing |

The gamepad's **Back / View / Share** button resets the scene: the arm goes
home and the objects respawn (a new desk in `desk_sort`).

### The desk sorting station

The stations and items are described once, in `lerobot_robot_kikobot/desk/`,
and both simulators build them from that description. With the arm flush in
the corner, its working patch of desk is small, so there are three stations,
at its edges. Directions below are as seen by the person at the desk.

| Station | Where, from the arm | For |
|---|---|---|
| **Pen cup** | just to its right | pens, pencils, markers, highlighters |
| **Tape peg** | further right, on the front edge | tape rolls: they drop over the post |
| **Supply tray** | at the far edge of its reach, back-right | glue sticks, erasers, sharpeners, paper clip boxes, scissors, the ruler, the stapler |

The clutter lies between the stations and the left side panel, within reach
and kept clear of the panel. Each desk shows 2 to 4 writing tools and 1 or 2
other supplies, drawn from a pool of 39 items.

The scripted expert (`kikobot-datagen`) succeeds on about 70–100 % of desks
for each task (pencils are the hardest).

**Every item comes in real sizes.** Each kind exists in several real
products' sizes, and the pool cycles through them, so every desk shows a
spread:

| Item | Sizes |
|---|---|
| Pens | Ballpoint, slim stick and chunky gel pens |
| Pencils | New, part used, and a stub |
| Markers | Felt tip, fineliner, whiteboard marker |
| Highlighters | Pocket, standard, large |
| Glue sticks | 8 g, 15 g, 21 g |
| Tape rolls | 12 mm mini, 13 mm, 19 mm and 25 mm, all on a 25 mm core |
| Scissors | 13 and 14 cm |
| Other items | Erasers, sticky-note pads (76 mm and 51 × 38 mm), standard and mini staplers, 15 cm rulers in two widths, paper clip boxes, single- and double-hole sharpeners |

Every size fits the open jaws and its station. Mass scales with size.

**The items are modelled in detail:**
- **Pens:** a metal tip, a ribbed grip, a clip and a click button.
- **Pencils:** sharpened to the wood and graphite, with a crimped ferrule
  and an eraser.
- **Markers and highlighters:** caps with clips. Highlighters have a chisel
  nib.
- **Glue sticks:** ribbed caps and twist rings.
- **Tape:** wound on a card core.
- **Scissors:** pointed blades, a pivot screw and a small and a large loop.
- **Rulers:** a bevelled edge with printed ticks.
- **Staplers:** a curved arm and a rubber foot.
- **Sharpeners:** a sloped blade plate.

Colours, printed patterns and finishes change every episode.

Every reset draws a new desk. **Pens can land on top of each other:** about
a fifth of the writing tools are dropped across others, up to three layers
deep. The server then runs 1 s of physics with the arm held still, so the
piles settle before the episode starts. The same `--seed` gives the same
sequence of desks.

## 4. Newton physics and ray-traced cameras

On an NVIDIA GPU, the server can step the physics with
[Newton](https://github.com/newton-physics/newton). That is its MuJoCo
solver, run on the GPU by MuJoCo Warp. The server can also render the camera
with the MuJoCo Warp ray tracer, for ray-traced shadows and anti-aliasing.

```bash
$V/pip install -e ".[newton]"      # Newton, Warp, MuJoCo Warp; pins MuJoCo 3.15
$V/kikobot-sim-server --scene desk_sort --engine newton --renderer mjwarp
```

| Option | Values | Default |
|---|---|---|
| `--engine` | `mujoco` (CPU), `newton` | `auto`: `newton` when it is installed and a CUDA GPU is there |
| `--renderer` | `opengl`, `mjwarp` (GPU ray tracing) | `auto`: `mjwarp` with the Newton engine on a GPU |

The scene is identical with either engine:
- The MuJoCo model stays the scene of record. Newton receives it as MJCF,
  and joint positions are copied back by name after every step.
- The servo model is the same, so a commanded joint moves the same way on
  both engines.

Without CUDA, `--engine newton` falls back to MuJoCo's C solver, and
`--renderer mjwarp` runs on the CPU at several seconds per frame.

## 5. Record a dataset in the sim

```bash
$V/lerobot-record --robot.type=kikobot_sim --teleop.type=kikobot_gamepad \
  --dataset.repo_id=local/kikobot_sim --dataset.root=datasets/kikobot_sim \
  --dataset.single_task="Put the pen in the pen cup" \
  --dataset.num_episodes=20 --dataset.push_to_hub=false
```

LeRobot has no hook to reset a sim scene between episodes. Press
**Back / View** on the gamepad during the reset period instead.

For episodes with no human, see [scripted demos for OpenVLA](datagen-openvla.md):
`kikobot-datagen` records perfect desk episodes with a scripted expert.
