"""Kiko-Bot in NVIDIA Isaac Sim with the Newton physics engine, served to LeRobot.

Run it with the Python that has Isaac Sim installed (not LeRobot's):

    python isaac/kikobot_isaac_server.py                    # with the Isaac Sim window
    python isaac/kikobot_isaac_server.py --headless         # no window (servers, SSH)

then, from LeRobot's Python, `lerobot-teleoperate --robot.type=kikobot_isaac ...`.
See docs/isaac-sim-teleop.md.

The server is lockstep: physics only advances when the robot sends a `step`,
by exactly one control period (1/fps), split into SUBSTEPS physics steps with
the same `MotionShaper` as the MuJoCo sim and the real arm running before
each one. A slow GPU frame therefore slows the episode down instead of
changing the motion, and a dataset recorded here plays back the same.

Nothing in this file imports LeRobot. `motion.py`, `motion_config.py` and
`isaac_protocol.py` are loaded from the plugin's folder by path, and the arm
comes from `models/usd/`, made once by `isaac/convert_mjcf.py`.
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

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "lerobot_robot_kikobot"
MODELS = Path(os.environ.get("KIKOBOT_MODELS_DIR", REPO / "models"))
DEFAULT_USD = MODELS / "usd" / "kikobot.usd"

ARM_JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")
FINGERS = ("finger_l", "finger_r")
# Physics steps per control tick, as in sim.py.
SUBSTEPS = 17
# Respawn fan, as in sim.py.
SPAWN_RADIUS_M = (0.22, 0.32)
SPAWN_HALF_ANGLE_RAD = math.radians(40.0)
MIN_OBJECT_PAD_GAP_M = 0.12
SCENES = ("empty", "pick_place", "cylinder")
CAMERAS = ("record",)

log = logging.getLogger("kikobot_isaac")


def _load(name: str, path: Path):
    """Import one module of the plugin by path, without the package (and LeRobot)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# motion.py imports `motion_config` by bare name when it has no package.
_load("motion_config", PKG / "motion_config.py")
motion = _load("kikobot_motion", PKG / "motion.py")
proto = _load("kikobot_isaac_protocol", PKG / "isaac_protocol.py")


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

    lower: np.ndarray
    upper: np.ndarray
    engine: str = "?"

    def configure(self, fps: int, width: int, height: int, cameras: list[str], scene: str) -> None: ...
    def reset_arm(self, q: np.ndarray, gripper_open: float) -> None: ...
    def respawn(self, obj_xy: np.ndarray, pad_xy: np.ndarray, yaw: float) -> None: ...
    def set_targets(self, q: np.ndarray, finger: float) -> None: ...
    def physics_step(self) -> None: ...
    def render(self) -> None: ...
    def joint_positions(self) -> tuple[np.ndarray, float]: ...
    def images(self) -> dict[str, np.ndarray]: ...
    def sim_time(self) -> float: ...
    def close(self) -> None: ...


class KikobotServer:
    """The protocol on top of a Backend: shaping, units, episode resets."""

    def __init__(self, backend: Backend, limits: dict[str, tuple[float, float]]):
        self.b = backend
        self.limits = limits
        self.lower = np.array([limits[j][0] for j in ARM_JOINTS])
        self.upper = np.array([limits[j][1] for j in ARM_JOINTS])
        self.grip_closed, self.grip_open = limits["finger_l"]
        self.cfg: dict[str, Any] | None = None
        self.shaper = None
        self.target = np.zeros(len(ARM_JOINTS))
        self.gripper_pct = 100.0
        self.rng = np.random.default_rng()

    # Units --------------------------------------------------------------

    def _finger(self, pct: float) -> float:
        return self.grip_closed + pct / 100.0 * (self.grip_open - self.grip_closed)

    def state(self) -> dict[str, float]:
        q, finger = self.b.joint_positions()
        out = {f"{j}.pos": float(np.rad2deg(v)) for j, v in zip(ARM_JOINTS, q)}
        span = self.grip_open - self.grip_closed
        out["gripper.pos"] = float(100.0 * (finger - self.grip_closed) / span)
        return out

    # Commands -----------------------------------------------------------

    def hello(self, req: dict[str, Any]) -> list[bytes]:
        if req.get("protocol") != proto.PROTOCOL_VERSION:
            raise ValueError(
                f"Client speaks protocol {req.get('protocol')}, server {proto.PROTOCOL_VERSION}. "
                "Update both from the same commit."
            )
        unknown = [c for c in req["cameras"] if c not in CAMERAS]
        if unknown:
            raise ValueError(f"Unknown cameras {unknown}; this scene has {list(CAMERAS)}.")
        if req["scene"] not in SCENES:
            raise ValueError(f"Unknown scene {req['scene']!r}; use one of {list(SCENES)}.")
        if req["image_encoding"] not in proto.IMAGE_ENCODINGS:
            raise ValueError(f"image_encoding must be one of {proto.IMAGE_ENCODINGS}.")
        geometry = {k: req[k] for k in ("fps", "width", "height", "cameras", "scene")}
        if self.cfg is None or {k: self.cfg[k] for k in geometry} != geometry:
            self.b.configure(**geometry)
        self.cfg = dict(req)
        self.rng = np.random.default_rng(req.get("seed"))
        self.shaper = motion.MotionShaper(len(ARM_JOINTS), motion.MotionConfig(**req["motion"]))
        log.info("Client: %s", {k: req[k] for k in (*geometry, "seed", "image_encoding")})
        return proto.pack(
            {
                "protocol": proto.PROTOCOL_VERSION,
                "joints": list(ARM_JOINTS),
                "lower_deg": np.rad2deg(self.lower).tolist(),
                "upper_deg": np.rad2deg(self.upper).tolist(),
                "engine": self.b.engine,
            }
        )

    def reset(self, req: dict[str, Any]) -> list[bytes]:
        self._need_hello()
        ready = np.clip(np.deg2rad(np.asarray(req["ready_deg"], dtype=float)), self.lower, self.upper)
        self.target = ready.copy()
        self.gripper_pct = 100.0
        self.b.reset_arm(ready, self._finger(100.0))
        if self.cfg["scene"] != "empty":
            self.b.respawn(*spawn_positions(self.rng))
        self.shaper.reset(ready, self.b.sim_time())
        self.b.render()
        return proto.pack({"state": self.state()})

    def step(self, req: dict[str, Any]) -> list[bytes]:
        self._need_hello()
        action = req["action"]
        for i, j in enumerate(ARM_JOINTS):
            if f"{j}.pos" in action:
                self.target[i] = math.radians(float(action[f"{j}.pos"]))
        self.target = np.clip(self.target, self.lower, self.upper)
        if "gripper.pos" in action:
            self.gripper_pct = float(np.clip(action["gripper.pos"], 0.0, 100.0))
        self.shaper.set_target(self.target, self.b.sim_time())
        finger = self._finger(self.gripper_pct)
        for _ in range(SUBSTEPS):
            self.b.set_targets(self.shaper.step(self.b.sim_time()), finger)
            self.b.physics_step()
        self.b.render()
        return proto.pack({"state": self.state()})

    def observe(self, req: dict[str, Any]) -> list[bytes]:
        self._need_hello()
        imgs = self.b.images()
        return proto.pack_observation(self.state(), imgs, self.cfg["image_encoding"])

    def _need_hello(self) -> None:
        if self.cfg is None or self.shaper is None:
            raise RuntimeError("Send `hello` first.")

    def handle(self, parts: list[bytes]) -> list[bytes]:
        try:
            req, _ = proto.unpack(parts)
            cmd = req.get("cmd")
            if cmd == "close":
                log.info("Client disconnected.")
                return proto.pack({})
            handler = {"hello": self.hello, "reset": self.reset, "step": self.step, "observe": self.observe}.get(cmd)
            if handler is None:
                raise ValueError(f"Unknown command {cmd!r}.")
            return handler(req)
        except Exception as e:  # noqa: BLE001 - every error goes back to the client
            log.error("Request failed: %s\n%s", e, traceback.format_exc())
            return proto.pack({"error": f"{type(e).__name__}: {e}"})


def serve(server: KikobotServer, bind: str, keep_running=lambda: True, idle=lambda: None) -> None:
    """Answer requests until keep_running() is False. `idle` runs while waiting
    (Isaac Sim uses it to keep its window responsive between requests)."""
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(bind)
    log.info("Serving Kiko-Bot on %s", bind)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    try:
        while keep_running():
            if not poller.poll(timeout=20):
                idle()
                continue
            sock.send_multipart(server.handle(sock.recv_multipart()))
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
        self._t = 0.0
        self._warned_blank = False

    @staticmethod
    def _enable(ext: str) -> None:
        import omni.kit.app

        mgr = omni.kit.app.get_app().get_extension_manager()
        if not mgr.set_extension_enabled_immediate(ext, True):
            log.warning("Could not enable extension %s.", ext)

    # Scene -----------------------------------------------------------------

    def configure(self, fps: int, width: int, height: int, cameras: list[str], scene: str) -> None:
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.simulation_manager.impl.mjc_scene import NewtonMjcScene

        self.cfg = dict(fps=fps, width=width, height=height, cameras=cameras, scene=scene)
        self.dt = 1.0 / (fps * SUBSTEPS)
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

        # Fixed-step, unthrottled main loop: physics is advanced only by step().
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
        timeline.pause()  # step() advances physics explicitly from here on.
        self._t = 0.0
        log.info("Scene %r ready: dofs %s, physics dt %.5f s.", scene, names, self.dt)

    def _build_world(self, stage, scene: str) -> None:
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(600)
        sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
        sun.CreateIntensityAttr(2500)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35, 20, 0))

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

        floor_z = 0.0 if scene == "empty" else -0.4
        floor = material("/World/Looks/Floor", (0.15, 0.22, 0.3), friction=1.0)
        box("/World/Floor", (0, 0, floor_z - 0.05), (5, 5, 0.05), floor)
        self.object_paths: dict[str, str] = {}
        if scene == "empty":
            return

        # Table: top face at z = 0, flush with the arm's base plate.
        top = material("/World/Looks/Tabletop", (0.72, 0.58, 0.43), friction=1.5)
        frame = material("/World/Looks/TableFrame", (0.28, 0.29, 0.32))
        box("/World/Table/Top", (0, 0, -0.01), (0.5, 0.5, 0.01), top)
        for i, (x, y) in enumerate(((0.45, 0.45), (-0.45, 0.45), (0.45, -0.45), (-0.45, -0.45))):
            leg = UsdGeom.Cylinder.Define(stage, f"/World/Table/Leg{i}")
            leg.CreateRadiusAttr(0.022)
            leg.CreateHeightAttr(0.38)
            UsdGeom.Xformable(leg).AddTranslateOp().Set(Gf.Vec3d(x, y, -0.21))
            bind(leg.GetPrim(), frame)

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
            # "record": in front of the table on +X, aimed at the base, 48 deg
            # vertical field of view, as in pick_place_scene.xml.
            path = f"/World/Cameras/{name}"
            cam = UsdGeom.Camera.Define(stage, path)
            focal = 18.0
            v_ap = 2 * focal * math.tan(math.radians(48.0) / 2)
            cam.CreateFocalLengthAttr(focal)
            cam.CreateVerticalApertureAttr(v_ap)
            cam.CreateHorizontalApertureAttr(v_ap * width / height)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
            eye, target = Gf.Vec3d(0.78, 0.0, 0.34), Gf.Vec3d(0.0, 0.0, 0.0)
            view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
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

    def set_targets(self, q: np.ndarray, finger: float) -> None:
        idx = [*self.arm_idx, *self.finger_idx]
        vals = np.array([[*q, *([finger] * len(self.finger_idx))]], dtype=np.float32)
        self.art.set_dof_position_targets(vals, dof_indices=idx)

    def physics_step(self) -> None:
        from isaacsim.core.simulation_manager import SimulationManager

        SimulationManager.step(steps=1)
        self._t += self.dt

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

    def sim_time(self) -> float:
        return self._t

    def close(self) -> None:
        self.timeline.stop()


def _numpy(x) -> np.ndarray:
    """Core Experimental returns Warp arrays; tests pass numpy."""
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


# ------------------------------------------------------------------------ main


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bind", default=f"tcp://127.0.0.1:{proto.DEFAULT_PORT}",
                   help="ZMQ address to listen on. Use tcp://*:5556 to accept other machines "
                        "(prefer an SSH tunnel; the protocol has no authentication).")
    p.add_argument("--headless", action="store_true", help="No Isaac Sim window.")
    p.add_argument("--livestream", action="store_true",
                   help="With --headless: stream the viewport (Isaac Sim WebRTC Streaming Client).")
    p.add_argument("--usd", default=str(DEFAULT_USD), help="The arm, from isaac/convert_mjcf.py.")
    p.add_argument("--physics-variant", default="newton",
                   help="Variant of the imported USD's 'Physics' set to use if present "
                        "(falls back to newton, mujoco, physics, physx).")
    p.add_argument("--integrator", default="implicitfast", help="MuJoCo-Warp integrator.")
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
        server = KikobotServer(backend, mjcf_limits())
        serve(server, args.bind, keep_running=app.is_running, idle=app.update)
    except KeyboardInterrupt:
        pass
    finally:
        app.close()


if __name__ == "__main__":
    main()
