"""`kikobot-sim-server`: the Kiko-Bot MuJoCo scene, simulated in real time, behind a socket.

Start it first, then point LeRobot at it:

    kikobot-sim-server --scene pick_place          # macOS: mjpython -m lerobot_robot_kikobot.sim_server
    lerobot-teleoperate --robot.type=kikobot_sim --teleop.type=kikobot_gamepad --fps=60
    lerobot-teleoperate --robot.type=kikobot_follower --robot.port=/dev/ttyACM0 \\
        --robot.id=kikobot --teleop.type=kikobot_gamepad --fps=60

Physics runs on wall-clock time, whatever the clients do. A `command` only
sets the actuator targets, and the arm moves there on the following physics
steps, as the real servos do with the same goal. No shaping or filtering sits
in between, so the sim shows exactly what the servos are told. The protocol
is in `protocol.py`.

The physics can run on MuJoCo or on NVIDIA Newton (`--engine`), and the
camera images come from MuJoCo's OpenGL renderer or MuJoCo Warp's ray tracer
(`--renderer`); see `sim_backends.py`. Either way the scene is the same.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import threading
import time
from typing import Any

import mujoco
import numpy as np

from .protocol import ARM_JOINTS
from .kinematics import ArmKinematics
from .env.mujoco_scene import SCENE_NAMES, RoomLights, build_scene, scene_mjcf
from .env.room import fixed_lighting, lab_room, sample_lighting
from .protocol import DEFAULT_PORT, IMAGE_ENCODINGS, PROTOCOL_VERSION, pack, pack_observation, unpack
from .sim_backends import ENGINES, RENDERERS, MjWarpRenderer, NewtonEngine, OpenGLRenderer, resolve_engine, resolve_renderer

logger = logging.getLogger(__name__)

PHYSICS_DT_S = 0.002
VIEWER_HZ = 60.0
WINDOWS = ("mujoco", "newton", "none")
# Further behind real time than this (a slow render, a stalled machine), the
# sim drops the time instead of racing to catch up.
MAX_LAG_S = 0.05
STATS_EVERY_S = 10.0

# desk_sort: physics with the arm held at ready after each new desk, so
# stacked pens drop and come to rest before the episode starts.
SETTLE_S = 0.6
# Objects respawn in a fan in front of the arm, within comfortable reach.
SPAWN_RADIUS_M = (0.22, 0.32)
SPAWN_HALF_ANGLE_RAD = math.radians(40.0)
MIN_OBJECT_PAD_GAP_M = 0.12


class SimWorld:
    """The MuJoCo model and data, and everything the protocol can do to them."""

    def __init__(
        self,
        scene: str = "pick_place",
        seed: int | None = None,
        timestep: float = PHYSICS_DT_S,
        randomize_lighting: bool = True,
        engine: str = "mujoco",
        renderer: str = "opengl",
    ):
        self.scene = scene
        self.engine_name = resolve_engine(engine)
        self.renderer_name = resolve_renderer(renderer, self.engine_name)
        if scene in SCENE_NAMES:
            # The arm in the lab (env/room.py), with the task on its table.
            self.room = lab_room()
            m = self.model = build_scene(scene, self.room).compile()
        else:
            self.room = None
            m = self.model = mujoco.MjModel.from_xml_path(scene)
        m.opt.timestep = timestep
        self.data = mujoco.MjData(m)
        self.rng = np.random.default_rng(seed)

        def id_of(kind, name):
            return mujoco.mj_name2id(m, kind, name)

        joints = [id_of(mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        if min(joints) < 0:
            raise RuntimeError(f"Scene {scene!r} has no arm joints {ARM_JOINTS}.")
        self._qpos_adr = m.jnt_qposadr[joints]
        self._dof_adr = m.jnt_dofadr[joints]
        self._arm_act = np.array([id_of(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ARM_JOINTS])
        self._grip_act = id_of(mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_l")
        fingers = [id_of(mujoco.mjtObj.mjOBJ_JOINT, f) for f in ("finger_l", "finger_r")]
        self._finger_qpos = m.jnt_qposadr[fingers]
        self._finger_dof = m.jnt_dofadr[fingers]
        self._grip_closed, self._grip_open = m.actuator_ctrlrange[self._grip_act]
        self.lower = m.jnt_range[joints, 0].copy()
        self.upper = m.jnt_range[joints, 1].copy()
        self._desk = None
        self._object_adr = None
        if scene == "desk_sort":
            from .desk.mujoco_desk import DeskRuntime  # noqa: PLC0415

            self._desk = DeskRuntime(m)
        else:
            # The first free joint is the object to pick; the pad is a mocap body.
            free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
            self._object_adr = int(m.jnt_qposadr[free[0]]) if free else None
        pad = id_of(mujoco.mjtObj.mjOBJ_BODY, "place_pad")
        self._pad_mocap = int(m.body_mocapid[pad]) if pad >= 0 else -1

        self.randomize_lighting = randomize_lighting
        self._lights = RoomLights(m, self.room) if self.room is not None else None
        self.ready = ArmKinematics().ready_q()
        # Who steps the physics (the MuJoCo model stays the scene of record),
        # and who draws the cameras. See sim_backends.py.
        self._engine = None
        if self.engine_name == "newton":
            mjcf = scene_mjcf(scene, self.room) if scene in SCENE_NAMES else scene
            self._engine = NewtonEngine(mjcf, m)
        self._renderer = MjWarpRenderer(m) if self.renderer_name == "mjwarp" else OpenGLRenderer(m)
        self.reset()

    # ----------------------------------------------------------------- state

    def _grip_ctrl(self, pct: float) -> float:
        pct = float(np.clip(pct, 0.0, 100.0))
        return self._grip_closed + pct / 100.0 * (self._grip_open - self._grip_closed)

    def state(self) -> dict[str, float]:
        d = self.data
        out = {f"{j}.pos": float(np.rad2deg(d.qpos[a])) for j, a in zip(ARM_JOINTS, self._qpos_adr)}
        span = self._grip_open - self._grip_closed
        out["gripper.pos"] = float(100.0 * (d.qpos[self._finger_qpos[0]] - self._grip_closed) / span)
        return out

    # --------------------------------------------------------------- actions

    def command(self, q: np.ndarray, gripper_pct: float) -> None:
        """New actuator targets (radians); physics moves the arm there."""
        self.data.ctrl[self._arm_act] = np.clip(q, self.lower, self.upper)
        self.data.ctrl[self._grip_act] = self._grip_ctrl(gripper_pct)
        if self._engine is not None:
            self._engine.set_ctrl(self.data.ctrl)

    def pose(self, q: np.ndarray, gripper_pct: float) -> None:
        """Put the arm at `q` at once, at rest, and hold it there."""
        m, d = self.model, self.data
        q = np.clip(q, self.lower, self.upper)
        d.qpos[self._qpos_adr] = q
        d.qvel[self._dof_adr] = 0.0
        grip = self._grip_ctrl(gripper_pct)
        # finger_r mirrors finger_l through an equality constraint.
        d.qpos[self._finger_qpos] = grip
        d.qvel[self._finger_dof] = 0.0
        self.command(q, gripper_pct)
        mujoco.mj_forward(m, d)
        if self._engine is not None:
            # Everything a reset or a teleport set in the MuJoCo data, into Newton.
            self._engine.push(d)

    def step(self, n: int) -> None:
        if self._engine is None:
            mujoco.mj_step(self.model, self.data, nstep=n)
        else:
            self._engine.step(n)
            self._engine.pull(self.model, self.data)

    def reset(self, ready: np.ndarray | None = None) -> None:
        """Arm to the ready pose, jaws open, object and pad respawned (or a new
        desk), and new room lighting."""
        mujoco.mj_resetData(self.model, self.data)
        if self._lights is not None:
            self._lights.apply(
                sample_lighting(self.rng, self.room) if self.randomize_lighting else fixed_lighting(self.room)
            )
        ready = self.ready if ready is None else ready
        if self._desk is not None:
            # A new desk: which items, where, and how they look.
            self._desk.reset(self.model, self.data, self.rng)
            self.pose(ready, 100.0)
            self.step(int(round(SETTLE_S / self.model.opt.timestep)))
        else:
            self._respawn_objects()
        self.pose(ready, 100.0)
        self._renderer.looks_changed()

    def _sample_xy(self) -> np.ndarray:
        r = self.rng.uniform(*SPAWN_RADIUS_M)
        a = self.rng.uniform(-SPAWN_HALF_ANGLE_RAD, SPAWN_HALF_ANGLE_RAD)
        return np.array([r * math.cos(a), r * math.sin(a)])

    def _respawn_objects(self) -> None:
        m, d = self.model, self.data
        obj_xy = self._sample_xy()
        if self._object_adr is not None:
            adr = self._object_adr
            # Keep the XML's height and tilt (a lying cylinder stays lying),
            # then spin about world Z.
            base = m.qpos0[adr + 3 : adr + 7]
            yaw = np.zeros(4)
            mujoco.mju_axisAngle2Quat(yaw, np.array([0.0, 0.0, 1.0]), self.rng.uniform(-math.pi, math.pi))
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, yaw, base)
            d.qpos[adr : adr + 3] = (*obj_xy, m.qpos0[adr + 2])
            d.qpos[adr + 3 : adr + 7] = quat
        if self._pad_mocap >= 0:
            pad_xy = self._sample_xy()
            for _ in range(100):
                if np.linalg.norm(pad_xy - obj_xy) >= MIN_OBJECT_PAD_GAP_M:
                    break
                pad_xy = self._sample_xy()
            d.mocap_pos[self._pad_mocap][:2] = pad_xy

    # ------------------------------------------------------------- rendering

    def render(self, cameras: list[str], width: int, height: int) -> dict[str, np.ndarray]:
        if not cameras:
            return {}
        return self._renderer.render(self.data, cameras, width, height)

    def close(self) -> None:
        self._renderer.close()


class SimServer:
    """Steps `world` in real time and answers protocol requests between steps."""

    def __init__(self, world: SimWorld, host: str = "127.0.0.1", port: int = DEFAULT_PORT, viewer: bool | str = True):
        self.world = world
        self.host, self.port = host, port
        # mujoco: MuJoCo's window on the full scene (both engines). newton:
        # Newton's ViewerGL on its own copy (plain colours). none / False: no window.
        self.window = {True: "mujoco", False: "none"}.get(viewer, viewer)
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, not {viewer!r}.")
        if self.window == "newton" and world._engine is None:
            raise ValueError("--window newton needs --engine newton.")
        self._viewer = None
        self._gl = None
        self.stop_event = threading.Event()

    # -------------------------------------------------------------- handlers

    def handle(self, header: dict[str, Any]) -> list[bytes]:
        w = self.world
        cmd = header.get("cmd")
        if cmd == "hello":
            return pack({
                "protocol": PROTOCOL_VERSION,
                "joints": list(ARM_JOINTS),
                "lower_deg": np.rad2deg(w.lower).tolist(),
                "upper_deg": np.rad2deg(w.upper).tolist(),
                "ready_deg": np.rad2deg(w.ready).tolist(),
                "scene": w.scene,
                "engine": "mujoco",
            })
        if cmd in ("command", "pose"):
            q = np.deg2rad(np.asarray(header["q_deg"], dtype=float))
            if q.shape != (len(ARM_JOINTS),):
                raise ValueError(f"q_deg needs {len(ARM_JOINTS)} angles.")
            getattr(w, cmd)(q, float(header.get("gripper_pct", 100.0)))
            return pack({"state": w.state()})
        if cmd == "observe":
            encoding = header.get("encoding", "raw")
            if encoding not in IMAGE_ENCODINGS:
                raise ValueError(f"encoding must be one of {IMAGE_ENCODINGS}.")
            images = w.render(list(header.get("cameras", [])), header.get("width", 640), header.get("height", 480))
            return pack_observation(w.state(), images, encoding)
        if cmd == "reset":
            ready = header.get("ready_deg")
            w.reset(None if ready is None else np.deg2rad(np.asarray(ready, dtype=float)))
            return pack({"state": w.state()})
        if cmd == "close":
            return pack({})
        raise ValueError(f"Unknown command {cmd!r}.")

    def _lock(self):
        return self._viewer.lock() if self._viewer is not None else contextlib.nullcontext()

    # ------------------------------------------------------------------ loop

    def serve(self) -> None:
        import zmq

        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://{self.host}:{self.port}")
        w = self.world
        if self.window == "newton":
            from newton.viewer import ViewerGL

            self._gl = ViewerGL()
            self._gl.set_model(w._engine.nm)
        elif self.window == "mujoco":
            import mujoco.viewer

            try:
                self._viewer = mujoco.viewer.launch_passive(w.model, w.data)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{exc}\nOn macOS the MuJoCo window needs mjpython: "
                    "mjpython -m lerobot_robot_kikobot.sim_server ..."
                ) from exc
        logger.info("Sim server: scene %r on tcp://%s:%d.", w.scene, self.host, self.port)

        dt = w.model.opt.timestep
        start = time.perf_counter()
        sim_t = 0.0  # Simulated seconds, against `start`; independent of resets.
        dropped = 0.0
        last_sync = last_stats = start
        steps_since_stats = 0
        try:
            while not self.stop_event.is_set():
                if self._viewer is not None and not self._viewer.is_running():
                    break
                if self._gl is not None and not self._gl.is_running():
                    break
                now = time.perf_counter()
                behind = (now - start - dropped) - sim_t
                if behind > MAX_LAG_S:
                    dropped += behind - dt
                    behind = dt
                n = int(behind / dt)
                if n:
                    with self._lock():
                        w.step(n)
                    sim_t += n * dt
                    steps_since_stats += n
                if now - last_sync >= 1.0 / VIEWER_HZ:
                    if self._viewer is not None:
                        self._viewer.sync()
                    elif self._gl is not None:
                        self._gl.begin_frame(sim_t)
                        self._gl.log_state(w._engine.s0)
                        self._gl.end_frame()
                    last_sync = now
                if now - last_stats >= STATS_EVERY_S:
                    logger.info("Real-time factor %.2f.", steps_since_stats * dt / (now - last_stats))
                    last_stats, steps_since_stats = now, 0

                # Wait for a request at most until the next physics step is due.
                wait_ms = int(max(0.0, dt - (behind - n * dt)) * 1000)
                if sock.poll(wait_ms):
                    while True:
                        try:
                            parts = sock.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        try:
                            header, _ = unpack(parts)
                            with self._lock():
                                reply = self.handle(header)
                        except Exception as exc:  # Report it to the client; keep serving.
                            logger.warning("Request failed: %s", exc)
                            reply = pack({"error": str(exc)})
                        sock.send_multipart(reply)
        finally:
            sock.close(linger=0)
            ctx.term()
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            if self._gl is not None:
                self._gl.close()
                self._gl = None
            w.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="kikobot-sim-server", description=__doc__.split("\n\n")[0])
    p.add_argument("--scene", default="pick_place",
                   help=f"{' | '.join(SCENE_NAMES)} (the arm in the lab), or a path to an MJCF file")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0 for other machines)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--window", choices=WINDOWS, default="mujoco",
                   help="mujoco: MuJoCo's window on the full scene (with either engine); newton: "
                        "Newton's ViewerGL (needs --engine newton; plain colours); none")
    p.add_argument("--no-viewer", dest="window", action="store_const", const="none", help="same as --window none")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timestep", type=float, default=PHYSICS_DT_S, help="physics step, seconds")
    p.add_argument("--fixed-lighting", action="store_true", help="the same room light every reset")
    p.add_argument("--engine", choices=ENGINES, default="auto",
                   help="physics: mujoco (CPU), or newton (MuJoCo Warp on a CUDA GPU); auto picks "
                        "newton when it is installed and a GPU is there")
    p.add_argument("--renderer", choices=RENDERERS, default="auto",
                   help="camera images: opengl, or mjwarp (ray-traced, GPU); auto picks mjwarp with newton")
    args = p.parse_args(argv)
    # force: importing the package imports LeRobot, which may have set up logging already.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    world = SimWorld(args.scene, args.seed, args.timestep, randomize_lighting=not args.fixed_lighting,
                     engine=args.engine, renderer=args.renderer)
    logger.info("Engine %s, renderer %s.", world.engine_name, world.renderer_name)
    server = SimServer(world, args.host, args.port, args.window)
    try:
        server.serve()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
