"""Kiko-Bot in NVIDIA Isaac Sim with the Newton physics engine, served to LeRobot.

Run it with the Python that has Isaac Sim installed (not LeRobot's):

    python isaac/kikobot_isaac_server.py --scene pick_place           # with the Isaac Sim window
    python isaac/kikobot_isaac_server.py --scene desk_sort --headless # no window (servers, SSH)

then, from LeRobot's Python, `lerobot-teleoperate --robot.type=kikobot_isaac ...`,
or mirror the real arm into it with `--robot.sim_backend=isaac`.
See docs/isaac-sim-teleop.md.

It speaks the same protocol as the MuJoCo server (`kikobot-sim-server`) and,
like it, runs physics on wall-clock time: a `command` sets the joint drive
targets, and the arm moves there on the following physics steps, exactly as
the real servos move to the same goal. No shaping or filtering sits in
between. The scene, the cameras and the image size are fixed when the server
starts (`--scene`, `--cameras`, `--width`, `--height`).

Nothing in this file imports LeRobot. `protocol.py` and the desk layout are
loaded from the plugin's folder by path, and the arm comes from `models/usd/`,
made once by `isaac/convert_mjcf.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PKG = REPO / "lerobot_robot_kikobot"
MODELS = Path(os.environ.get("KIKOBOT_MODELS_DIR", REPO / "models"))
DEFAULT_USD = MODELS / "usd" / "kikobot.usd"

ARM_JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")
FINGERS = ("finger_l", "finger_r")
PHYSICS_DT_S = 0.002
DEFAULT_PORT = 5557  # The MuJoCo server has 5556, so both can run at once.
# Further behind real time than this, the sim drops the time instead of racing.
MAX_LAG_S = 0.05
STATS_EVERY_S = 10.0
# Respawn fan, as in sim_server.py.
SPAWN_RADIUS_M = (0.22, 0.32)
SPAWN_HALF_ANGLE_RAD = math.radians(40.0)
MIN_OBJECT_PAD_GAP_M = 0.12
SCENES = ("empty", "pick_place", "cylinder", "desk_sort")
# desk_sort: physics run with the arm held at ready after each respawn, so
# stacked pens drop and come to rest before the episode starts.
SETTLE_S = 1.0

log = logging.getLogger("kikobot_isaac")


def _load(name: str, path: Path):
    """Import one module of the plugin by path, without the package (and LeRobot)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


proto = _load("kikobot_protocol", PKG / "protocol.py")
# The lab, shared with the MuJoCo scene, and its USD builder.
room = _load("kikobot_room", PKG / "env" / "room.py")
# The desk sorting station (no Isaac Sim imports at module level either).
desk_layout = _load("desk_layout", PKG / "desk" / "layout.py")
_load("desk_textures", PKG / "desk" / "textures.py")
_load("desk_meshes", PKG / "desk" / "meshes.py")
desk_assets = _load("desk_assets", HERE / "desk_assets.py")
room_usd = _load("room_usd", HERE / "room_usd.py")

LAB = room.lab_room()
# name: (eye, target, up, vertical fov deg), the same as the MuJoCo scene's.
CAMERA_POSES = {c.name: (c.eye, c.target, c.up, c.fovy) for c in LAB.cameras}
CAMERAS = tuple(CAMERA_POSES)
DEFAULT_ROOM_TEXTURES = MODELS / "textures" / "room"


# ------------------------------------------------------------------ the model


def mjcf_limits(robot_xml: Path = MODELS / "robot.xml") -> dict[str, tuple[float, float]]:
    """Joint ranges straight from the MJCF, so Isaac clips exactly like MuJoCo."""
    out = {}
    for j in ET.parse(robot_xml).getroot().iter("joint"):
        name, rng = j.get("name"), j.get("range")
        if name in (*ARM_JOINTS, *FINGERS) and rng:
            lo, hi = (float(v) for v in rng.split())
            out[name] = (lo, hi)
    missing = (set(ARM_JOINTS) | {"finger_l"}) - set(out)
    if missing:
        raise RuntimeError(f"Joints {sorted(missing)} have no range in {robot_xml}.")
    return out


def sample_xy(rng: np.random.Generator) -> np.ndarray:
    r = rng.uniform(*SPAWN_RADIUS_M)
    a = rng.uniform(-SPAWN_HALF_ANGLE_RAD, SPAWN_HALF_ANGLE_RAD)
    return np.array([r * math.cos(a), r * math.sin(a)])


def spawn_positions(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """Object xy, pad xy (at least MIN_OBJECT_PAD_GAP_M apart) and object yaw."""
    obj = sample_xy(rng)
    pad = sample_xy(rng)
    for _ in range(100):
        if np.linalg.norm(pad - obj) >= MIN_OBJECT_PAD_GAP_M:
            break
        pad = sample_xy(rng)
    return obj, pad, float(rng.uniform(-math.pi, math.pi))


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# ---------------------------------------------------------------- the backend


class Backend:
    """What the protocol loop needs from a simulator. Angles in radians here."""

    engine: str = "?"

    def configure(self, physics_dt: float, width: int, height: int, cameras: list[str], scene: str) -> None: ...
    def reset_arm(self, q: np.ndarray, gripper_open: float) -> None: ...  # teleport, at rest, and hold
    def respawn(self, obj_xy: np.ndarray, pad_xy: np.ndarray, yaw: float) -> None: ...
    # desk_sort: the items that exist (with real sizes), and one episode's layout.
    desk_pool: list = []
    def randomize_desk(self, placements: list) -> None: ...
    def set_lighting(self, states: list) -> None: ...  # room.LightState, one per light
    def set_targets(self, q: np.ndarray, finger: float) -> None: ...
    def physics_step(self) -> None: ...
    def render(self) -> None: ...
    def joint_positions(self) -> tuple[np.ndarray, float]: ...
    def images(self) -> dict[str, np.ndarray]: ...
    def close(self) -> None: ...


class KikobotServer:
    """The protocol on top of a Backend: units, limits, episode resets."""

    def __init__(
        self,
        backend: Backend,
        limits: dict[str, tuple[float, float]],
        scene: str = "pick_place",
        cameras: tuple[str, ...] = ("record",),
        width: int = 640,
        height: int = 480,
        physics_dt: float = PHYSICS_DT_S,
        seed: int | None = None,
        settle_s: float = SETTLE_S,
        randomize_lighting: bool = True,
    ):
        unknown = [c for c in cameras if c not in CAMERAS]
        if unknown:
            raise ValueError(f"Unknown cameras {unknown}; this server has {list(CAMERAS)}.")
        if scene not in SCENES:
            raise ValueError(f"Unknown scene {scene!r}; use one of {list(SCENES)}.")
        self.b = backend
        self.scene, self.cameras = scene, tuple(cameras)
        self.width, self.height = width, height
        self.physics_dt = physics_dt
        self.settle_s = settle_s
        self.randomize_lighting = randomize_lighting
        self.placements: list = []
        self.lower = np.array([limits[j][0] for j in ARM_JOINTS])
        self.upper = np.array([limits[j][1] for j in ARM_JOINTS])
        self.grip_closed, self.grip_open = limits["finger_l"]
        self.rng = np.random.default_rng(seed)
        self.gripper_pct = 100.0
        self.b.configure(physics_dt, width, height, list(cameras), scene)
        log.info("Scene %r, cameras %s at %dx%d.", scene, list(cameras), width, height)

    # Units --------------------------------------------------------------

    def _finger(self, pct: float) -> float:
        return self.grip_closed + pct / 100.0 * (self.grip_open - self.grip_closed)

    def _q(self, req: dict[str, Any], key: str = "q_deg") -> np.ndarray:
        q = np.deg2rad(np.asarray(req[key], dtype=float))
        if q.shape != (len(ARM_JOINTS),):
            raise ValueError(f"{key} needs {len(ARM_JOINTS)} angles.")
        return np.clip(q, self.lower, self.upper)

    def state(self) -> dict[str, float]:
        q, finger = self.b.joint_positions()
        out = {f"{j}.pos": float(np.rad2deg(v)) for j, v in zip(ARM_JOINTS, q)}
        span = self.grip_open - self.grip_closed
        out["gripper.pos"] = float(100.0 * (finger - self.grip_closed) / span)
        return out

    # Commands -----------------------------------------------------------

    def hello(self, req: dict[str, Any]) -> list[bytes]:
        return proto.pack(
            {
                "protocol": proto.PROTOCOL_VERSION,
                "joints": list(ARM_JOINTS),
                "lower_deg": np.rad2deg(self.lower).tolist(),
                "upper_deg": np.rad2deg(self.upper).tolist(),
                "scene": self.scene,
                "cameras": list(self.cameras),
                "width": self.width,
                "height": self.height,
                "engine": self.b.engine,
            }
        )

    def command(self, req: dict[str, Any]) -> list[bytes]:
        self.gripper_pct = float(np.clip(req.get("gripper_pct", self.gripper_pct), 0.0, 100.0))
        self.b.set_targets(self._q(req), self._finger(self.gripper_pct))
        return proto.pack({"state": self.state()})

    def pose(self, req: dict[str, Any]) -> list[bytes]:
        self.gripper_pct = float(np.clip(req.get("gripper_pct", self.gripper_pct), 0.0, 100.0))
        self.b.reset_arm(self._q(req), self._finger(self.gripper_pct))
        return proto.pack({"state": self.state()})

    def reset(self, req: dict[str, Any]) -> list[bytes]:
        if req.get("ready_deg") is None:
            raise ValueError("reset needs ready_deg here: Isaac Sim's Python has no MuJoCo to solve it.")
        ready = self._q(req, "ready_deg")
        self.gripper_pct = 100.0
        self.b.reset_arm(ready, self._finger(100.0))
        self.b.set_lighting(room.sample_lighting(self.rng, LAB) if self.randomize_lighting else room.fixed_lighting(LAB))
        if self.scene == "desk_sort":
            self.placements = desk_layout.sample_desk(self.rng, self.b.desk_pool)
            self.b.randomize_desk(self.placements)
            self._settle(ready)
        elif self.scene != "empty":
            self.b.respawn(*spawn_positions(self.rng))
        self.b.render()
        return proto.pack({"state": self.state()})

    def _settle(self, ready: np.ndarray) -> None:
        """Let dropped items come to rest with the arm held at ready, then put
        the arm back exactly at ready."""
        finger = self._finger(100.0)
        self.b.set_targets(ready, finger)
        for _ in range(int(round(self.settle_s / self.physics_dt))):
            self.b.physics_step()
        self.b.reset_arm(ready, finger)

    def observe(self, req: dict[str, Any]) -> list[bytes]:
        wanted = list(req.get("cameras", []))
        unknown = [c for c in wanted if c not in self.cameras]
        if unknown:
            raise ValueError(
                f"Unknown cameras {unknown}; this server renders {list(self.cameras)} "
                "(start it with --cameras)."
            )
        if wanted and (req.get("width", self.width), req.get("height", self.height)) != (self.width, self.height):
            raise ValueError(
                f"This server renders {self.width}x{self.height}, not {req.get('width')}x{req.get('height')}: "
                "start it with matching --width/--height, or set --robot.width/height."
            )
        encoding = req.get("encoding", "raw")
        if encoding not in proto.IMAGE_ENCODINGS:
            raise ValueError(f"encoding must be one of {proto.IMAGE_ENCODINGS}.")
        images = {}
        if wanted:
            self.b.render()
            images = {c: img for c, img in self.b.images().items() if c in wanted}
        return proto.pack_observation(self.state(), images, encoding)

    def handle(self, parts: list[bytes]) -> list[bytes]:
        try:
            req, _ = proto.unpack(parts)
            cmd = req.get("cmd")
            if cmd == "close":
                log.info("Client disconnected.")
                return proto.pack({})
            handler = {
                "hello": self.hello, "command": self.command, "pose": self.pose,
                "reset": self.reset, "observe": self.observe,
            }.get(cmd)
            if handler is None:
                raise ValueError(f"Unknown command {cmd!r}.")
            return handler(req)
        except Exception as e:  # noqa: BLE001 - every error goes back to the client
            log.error("Request failed: %s\n%s", e, traceback.format_exc())
            return proto.pack({"error": f"{type(e).__name__}: {e}"})


def serve(server: KikobotServer, bind: str, keep_running=lambda: True, idle=lambda: None,
          render_hz: float = 30.0) -> None:
    """Step physics on wall-clock time and answer requests between steps until
    keep_running() is False. `idle` redraws the window, at `render_hz`."""
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(bind)
    log.info("Serving Kiko-Bot on %s", bind)
    dt = server.physics_dt
    start = time.perf_counter()
    sim_t = dropped = 0.0
    last_render = last_stats = start
    steps_since_stats = 0
    try:
        while keep_running():
            now = time.perf_counter()
            behind = (now - start - dropped) - sim_t
            if behind > MAX_LAG_S:
                dropped += behind - dt
                behind = dt
            n = int(behind / dt)
            for _ in range(n):
                server.b.physics_step()
            sim_t += n * dt
            steps_since_stats += n
            if now - last_render >= 1.0 / render_hz:
                idle()
                last_render = now
            if now - last_stats >= STATS_EVERY_S:
                log.info("Real-time factor %.2f.", steps_since_stats * dt / (now - last_stats))
                last_stats, steps_since_stats = now, 0
            wait_ms = int(max(0.0, dt - (behind - n * dt)) * 1000)
            if sock.poll(wait_ms):
                while True:
                    try:
                        parts = sock.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    sock.send_multipart(server.handle(parts))
    finally:
        sock.close(linger=0)


# ---------------------------------------------------------- Isaac Sim (Newton)


class IsaacNewtonBackend(Backend):
    """The arm, the desk and the record camera in Isaac Sim, simulated by Newton
    with its MuJoCo-Warp solver.

    Every Isaac Sim call lives in this class. Isaac Sim's Python API moves
    between releases; this targets 6.0 (`isaacsim.core.experimental`,
    `isaacsim.core.simulation_manager`). If an import fails on another
    release, this is the only class to adapt.
    """

    engine = "isaac-sim/newton"

    def __init__(self, app, args: argparse.Namespace):
        self.app = app
        self.args = args
        self.usd = Path(args.usd)
        if not self.usd.is_file():
            raise FileNotFoundError(
                f"{self.usd} not found. Convert the arm first: python isaac/convert_mjcf.py"
            )
        for ext in ("isaacsim.physics.newton", "isaacsim.physics.newton.tensors", "omni.replicator.core"):
            self._enable(ext)
        self.cfg: dict[str, Any] | None = None
        self._annotators: dict[str, Any] = {}
        self._warned_blank = False

    @staticmethod
    def _enable(ext: str) -> None:
        import omni.kit.app

        mgr = omni.kit.app.get_app().get_extension_manager()
        if not mgr.set_extension_enabled_immediate(ext, True):
            log.warning("Could not enable extension %s.", ext)

    # Scene -----------------------------------------------------------------

    def configure(self, physics_dt: float, width: int, height: int, cameras: list[str], scene: str) -> None:
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.simulation_manager.impl.mjc_scene import NewtonMjcScene

        self.cfg = dict(width=width, height=height, cameras=cameras, scene=scene)
        self.dt = physics_dt
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        omni.usd.get_context().new_stage()
        if not SimulationManager.switch_physics_engine("newton"):
            raise RuntimeError(
                "Isaac Sim could not switch to the Newton engine. Available: "
                f"{SimulationManager.get_available_physics_engines()}"
            )
        stage = omni.usd.get_context().get_stage()
        self.stage = stage

        # MuJoCo-Warp solver, set up like models/robot.xml's <option>.
        mjc = NewtonMjcScene("/World/PhysicsScene")
        mjc.set_gravity((0.0, 0.0, -9.81))
        mjc.set_dt(self.dt)
        mjc.set_integrator(self.args.integrator)
        mjc.set_solver("newton")
        mjc.set_cone("elliptic")
        mjc.set_iterations(100)

        self._build_world(stage, scene)
        self._add_arm(stage)
        self._add_cameras(stage, cameras, width, height)

        # Fixed-step, unthrottled main loop: physics is advanced only by the
        # server's real-time loop, through physics_step().
        import carb

        settings = carb.settings.get_settings()
        settings.set("/app/runLoops/main/rateLimitEnabled", False)
        settings.set("/app/player/useFixedTimeStepping", True)

        timeline.play()
        self.timeline = timeline
        self.app.update()  # Create the physics views.
        from isaacsim.core.experimental.prims import Articulation, RigidPrim

        self.art = Articulation(self.art_path)
        names = list(self.art.dof_names)
        missing = [j for j in (*ARM_JOINTS, "finger_l") if j not in names]
        if missing:
            raise RuntimeError(f"Joints {missing} are not in the articulation (it has {names}).")
        self.arm_idx = [names.index(j) for j in ARM_JOINTS]
        self.finger_idx = [names.index(f) for f in FINGERS if f in names]
        self.objects = {
            name: RigidPrim(path) for name, path in self.object_paths.items()
        }
        self.desk_view = None
        if scene == "desk_sort":
            self.desk_view = RigidPrim([self.desk.paths[it.name] for it in self.desk_pool])
        timeline.pause()  # physics_step() advances physics explicitly from here on.
        log.info("Scene %r ready: dofs %s, physics dt %.5f s.", scene, names, self.dt)

    def _build_world(self, stage, scene: str) -> None:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        # The lab, its lights and the arm's grey table (env/room.py), the same as MuJoCo's.
        self.room_lights = room_usd.build_room(stage, LAB, Path(self.args.room_textures), self.args.light_scale)

        def material(path: str, rgb, friction: float | None = None):
            mat = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            if friction is not None:
                pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                pm.CreateStaticFrictionAttr(friction)
                pm.CreateDynamicFrictionAttr(friction)
                pm.CreateRestitutionAttr(0.0)
            return mat

        def bind(prim, mat) -> None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose=UsdShade.Tokens.allPurpose)

        def box(path, center, half, mat, collide=True):
            g = UsdGeom.Cube.Define(stage, path)
            g.CreateSizeAttr(1.0)
            x = UsdGeom.Xformable(g)
            x.AddTranslateOp().Set(Gf.Vec3d(*center))
            x.AddScaleOp().Set(Gf.Vec3f(*(2 * h for h in half)))
            if collide:
                UsdPhysics.CollisionAPI.Apply(g.GetPrim())
            bind(g.GetPrim(), mat)
            return g

        self.object_paths: dict[str, str] = {}
        self.desk_pool = []
        if scene == "empty":
            return

        if scene == "desk_sort":
            self._build_desk(stage)
            return

        # Object to pick: free rigid body. Pose is set per episode by respawn().
        if scene == "pick_place":
            red = material("/World/Looks/Cube", (0.85, 0.12, 0.12), friction=1.5)
            obj = box("/World/Object", (0.25, 0, 0.016), (0.016, 0.016, 0.016), red)
            self.object_base_quat = np.array([1.0, 0, 0, 0])
            self.object_z = 0.016
        else:  # cylinder, lying on its side
            orange = material("/World/Looks/Cylinder", (0.95, 0.55, 0.1), friction=1.5)
            obj = UsdGeom.Cylinder.Define(stage, "/World/Object")
            obj.CreateRadiusAttr(0.015)
            obj.CreateHeightAttr(0.08)
            obj.CreateAxisAttr("Z")
            UsdGeom.Xformable(obj).AddTranslateOp().Set(Gf.Vec3d(0.25, 0, 0.015))
            UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
            bind(obj.GetPrim(), orange)
            s = math.sqrt(0.5)
            self.object_base_quat = np.array([s, s, 0.0, 0.0])  # euler 90 0 0
            self.object_z = 0.015
        UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
        UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(0.03)
        self.object_paths["object"] = "/World/Object"

        # Place pad: kinematic, so it never moves unless respawn() puts it somewhere.
        green = material("/World/Looks/Pad", (0.13, 0.72, 0.25), friction=1.5)
        pad = UsdGeom.Cylinder.Define(stage, "/World/Pad")
        pad.CreateRadiusAttr(0.055)
        pad.CreateHeightAttr(0.016)
        UsdGeom.Xformable(pad).AddTranslateOp().Set(Gf.Vec3d(0.26, 0.22, 0.008))
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
        rb = UsdPhysics.RigidBodyAPI.Apply(pad.GetPrim())
        rb.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(pad.GetPrim()).CreateMassAttr(1.0)
        bind(pad.GetPrim(), green)
        self.object_paths["pad"] = "/World/Pad"

    def _build_desk(self, stage) -> None:
        mode = self.args.desk_assets
        catalog = {} if mode == "procedural" else desk_assets.load_catalog(Path(self.args.desk_assets_json))
        if mode == "nucleus" and not catalog:
            raise FileNotFoundError(
                f"--desk-assets nucleus needs {self.args.desk_assets_json}. "
                "Make it with: python isaac/desk_assets.py --discover"
            )
        self.desk = desk_assets.build_desk(
            stage, desk_layout.default_pool(), desk_assets.DEFAULT_TEXTURES, catalog,
            recolor_assets=not self.args.keep_asset_materials,
        )
        self.desk_pool = self.desk.pool
        n_assets = sum(1 for s in self.desk.sources.values() if s != "procedural")
        log.info("Desk: %d items (%d from assets, %d procedural).", len(self.desk_pool), n_assets,
                 len(self.desk_pool) - n_assets)
        for name, src in self.desk.sources.items():
            log.debug("  %s: %s", name, src)

    def _add_arm(self, stage) -> None:
        from pxr import UsdPhysics

        root = stage.DefinePrim("/World/kikobot", "Xform")
        root.GetReferences().AddReference(str(self.usd))
        # The importer writes one variant per physics engine; pick the one Newton reads.
        vsets = root.GetVariantSets()
        if vsets.HasVariantSet("Physics"):
            vs = vsets.GetVariantSet("Physics")
            names = vs.GetVariantNames()
            want = self.args.physics_variant
            choice = want if want in names else next(
                (v for v in ("newton", "mujoco", "physics", "physx") if v in names), None
            )
            if choice:
                vs.SetVariantSelection(choice)
            log.info("Physics variants %s, using %r.", names, choice)

        # Joint drives: authored here, not taken from the importer, which in
        # Isaac Sim 6.x drops kp-only <position> actuators and leaves hinge
        # gains in rad units (isaac-sim/IsaacSim#857). USD angular drives are
        # per degree; Newton converts them back.
        a = self.args
        self.art_path = None
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith("/World/kikobot"):
                continue
            if self.art_path is None and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                self.art_path = path
            name = prim.GetName()
            if name in ARM_JOINTS and prim.IsA(UsdPhysics.RevoluteJoint):
                self._drive(prim, "angular", a.kp * math.pi / 180, a.kd * math.pi / 180, a.max_torque)
            elif name in FINGERS and prim.IsA(UsdPhysics.PrismaticJoint):
                self._drive(prim, "linear", a.finger_kp, a.finger_kd, a.finger_max_force)
        if self.art_path is None:
            self.art_path = "/World/kikobot"
            UsdPhysics.ArticulationRootAPI.Apply(root)
            log.warning("No articulation root in %s; applied one at /World/kikobot.", self.usd)

    @staticmethod
    def _drive(prim, kind: str, stiffness: float, damping: float, max_force: float) -> None:
        from pxr import UsdPhysics

        d = UsdPhysics.DriveAPI.Apply(prim, kind)
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr().Set(stiffness)
        d.CreateDampingAttr().Set(damping)
        d.CreateMaxForceAttr().Set(max_force)

    def _add_cameras(self, stage, cameras: list[str], width: int, height: int) -> None:
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        for ann in self._annotators.values():
            ann.detach()
        self._annotators = {}
        for name in cameras:
            eye, target, up, vfov = CAMERA_POSES[name]
            path = f"/World/Cameras/{name}"
            cam = UsdGeom.Camera.Define(stage, path)
            focal = 18.0
            v_ap = 2 * focal * math.tan(math.radians(vfov) / 2)
            cam.CreateFocalLengthAttr(focal)
            cam.CreateVerticalApertureAttr(v_ap)
            cam.CreateHorizontalApertureAttr(v_ap * width / height)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
            UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())
            rp = rep.create.render_product(path, (width, height))
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([rp])
            self._annotators[name] = ann

    # Simulation ------------------------------------------------------------

    def reset_arm(self, q: np.ndarray, gripper_open: float) -> None:
        n = len(self.art.dof_names)
        pos = np.zeros((1, n), dtype=np.float32)
        pos[0, self.arm_idx] = q
        pos[0, self.finger_idx] = gripper_open
        self.art.set_dof_positions(pos)
        self.art.set_dof_velocities(np.zeros((1, n), dtype=np.float32))
        self.art.set_dof_position_targets(pos)

    def respawn(self, obj_xy: np.ndarray, pad_xy: np.ndarray, yaw: float) -> None:
        if "object" in self.objects:
            quat = quat_mul(yaw_quat(yaw), self.object_base_quat)
            self.objects["object"].set_world_poses(
                positions=[[obj_xy[0], obj_xy[1], self.object_z]], orientations=[quat]
            )
            self.objects["object"].set_velocities(linear_velocities=[[0, 0, 0]], angular_velocities=[[0, 0, 0]])
        if "pad" in self.objects:
            self.objects["pad"].set_world_poses(positions=[[pad_xy[0], pad_xy[1], 0.008]], orientations=[[1, 0, 0, 0]])

    def randomize_desk(self, placements: list) -> None:
        n = len(placements)
        pos = np.array([p.pos for p in placements], dtype=np.float32)
        quat = np.array([yaw_quat(p.yaw) for p in placements], dtype=np.float32)
        self.desk_view.set_world_poses(positions=pos, orientations=quat)
        zeros = np.zeros((n, 3), dtype=np.float32)
        self.desk_view.set_velocities(linear_velocities=zeros, angular_velocities=zeros)
        desk_assets.apply_looks(self.desk, placements)

    def set_lighting(self, states: list) -> None:
        self.room_lights.apply(states)

    def set_targets(self, q: np.ndarray, finger: float) -> None:
        idx = [*self.arm_idx, *self.finger_idx]
        vals = np.array([[*q, *([finger] * len(self.finger_idx))]], dtype=np.float32)
        self.art.set_dof_position_targets(vals, dof_indices=idx)

    def physics_step(self) -> None:
        from isaacsim.core.simulation_manager import SimulationManager

        SimulationManager.step(steps=1)

    def render(self) -> None:
        # Paused timeline: draws the frame and fills the annotators, no physics.
        self.app.update()

    def joint_positions(self) -> tuple[np.ndarray, float]:
        q = _numpy(self.art.get_dof_positions())[0]
        return q[self.arm_idx].astype(float), float(q[self.finger_idx[0]])

    def images(self) -> dict[str, np.ndarray]:
        out = {}
        h, w = self.cfg["height"], self.cfg["width"]
        for name, ann in self._annotators.items():
            data = np.asarray(ann.get_data())
            if data.size == 0:
                if not self._warned_blank:
                    log.warning("Camera %r has no frame yet (renderer warming up); sending black.", name)
                    self._warned_blank = True
                data = np.zeros((h, w, 3), dtype=np.uint8)
            out[name] = data[..., :3]
        return out

    def close(self) -> None:
        self.timeline.stop()


def _numpy(x) -> np.ndarray:
    """Core Experimental returns Warp arrays; tests pass numpy."""
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


# ------------------------------------------------------------------------ main


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bind", default=f"tcp://127.0.0.1:{DEFAULT_PORT}",
                   help=f"ZMQ address to listen on. Use tcp://*:{DEFAULT_PORT} to accept other machines "
                        "(prefer an SSH tunnel; the protocol has no authentication).")
    p.add_argument("--scene", choices=SCENES, default="pick_place")
    p.add_argument("--cameras", nargs="*", choices=CAMERAS, default=["record"],
                   help="Cameras to render (the robot's --robot.camera_names picks from these).")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--physics-dt", type=float, default=PHYSICS_DT_S, help="Physics step, seconds.")
    p.add_argument("--render-hz", type=float, default=30.0, help="Window redraw rate.")
    p.add_argument("--fixed-lighting", action="store_true", help="The same room light every reset.")
    p.add_argument("--light-scale", type=float, default=1.0,
                   help="Scale every room light (RTX exposure differs between machines).")
    p.add_argument("--room-textures", default=str(DEFAULT_ROOM_TEXTURES),
                   help="Where the room's texture PNGs are written (once).")
    p.add_argument("--headless", action="store_true", help="No Isaac Sim window.")
    p.add_argument("--livestream", action="store_true",
                   help="With --headless: stream the viewport (Isaac Sim WebRTC Streaming Client).")
    p.add_argument("--usd", default=str(DEFAULT_USD), help="The arm, from isaac/convert_mjcf.py.")
    p.add_argument("--physics-variant", default="newton",
                   help="Variant of the imported USD's 'Physics' set to use if present "
                        "(falls back to newton, mujoco, physics, physx).")
    p.add_argument("--integrator", default="implicitfast", help="MuJoCo-Warp integrator.")
    # desk_sort scene.
    p.add_argument("--desk-assets", choices=("auto", "nucleus", "procedural"), default="auto",
                   help="desk_sort items: auto uses --desk-assets-json where it has a match and "
                        "builds the rest; nucleus requires the JSON; procedural ignores it.")
    p.add_argument("--desk-assets-json", default=str(desk_assets.DEFAULT_ASSETS_JSON),
                   help="From `python isaac/desk_assets.py --discover`.")
    p.add_argument("--keep-asset-materials", action="store_true",
                   help="Keep Nucleus assets' own look instead of the random colours and textures.")
    p.add_argument("--settle-s", type=float, default=SETTLE_S,
                   help="desk_sort: seconds of physics after each respawn for items to come to rest.")
    # Servo model: models/robot.xml's position actuators.
    p.add_argument("--kp", type=float, default=200.0, help="Arm joint stiffness, N*m/rad.")
    p.add_argument("--kd", type=float, default=3.0, help="Arm joint damping, N*m*s/rad.")
    p.add_argument("--max-torque", type=float, default=2.94, help="Arm joint torque limit, N*m.")
    p.add_argument("--finger-kp", type=float, default=800.0, help="Finger stiffness, N/m.")
    p.add_argument("--finger-kd", type=float, default=8.0, help="Finger damping, N*s/m.")
    p.add_argument("--finger-max-force", type=float, default=20.0, help="Finger force limit, N.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp  # Must come before any other omni/isaacsim import.

    app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})
    if args.livestream:
        IsaacNewtonBackend._enable("omni.kit.livestream.webrtc")
    try:
        backend = IsaacNewtonBackend(app, args)
        server = KikobotServer(
            backend, mjcf_limits(), args.scene, tuple(args.cameras), args.width, args.height,
            args.physics_dt, args.seed, settle_s=args.settle_s,
            randomize_lighting=not args.fixed_lighting,
        )
        serve(server, args.bind, keep_running=app.is_running, idle=backend.render, render_hz=args.render_hz)
    except KeyboardInterrupt:
        pass
    finally:
        app.close()


if __name__ == "__main__":
    main()
