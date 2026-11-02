"""Physics engines and camera renderers for the sim server.

The server's MuJoCo model (`SimWorld.model`, with the lab's textures, lights
and the desk's randomized looks) is always the scene of record. What changes
is who steps its physics and who draws it:

    --engine mujoco    MuJoCo itself, on the CPU (any machine).
    --engine newton    NVIDIA Newton's SolverMuJoCo: MuJoCo Warp on a CUDA GPU
                       (MuJoCo's C solver on a machine without one). The scene
                       goes to Newton as MJCF; joint positions come back by name.

    --renderer opengl  MuJoCo's OpenGL renderer: shadow maps, any machine.
    --renderer mjwarp  MuJoCo Warp's ray-traced batch renderer on the GPU:
                       ray-traced shadows, textures, anti-aliasing.

Newton, Warp and MuJoCo Warp are optional (`pip install -e ".[newton]"`) and
only imported when asked for.
"""

from __future__ import annotations

import logging
import warnings

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

ENGINES = ("auto", "mujoco", "newton")
RENDERERS = ("auto", "opengl", "mjwarp")


def cuda_available() -> bool:
    try:
        import warp as wp
    except ImportError:
        return False
    try:
        return bool(wp.is_cuda_available())
    except Exception:  # noqa: BLE001 - no driver, broken install...
        return False


def resolve_engine(engine: str) -> str:
    if engine not in ENGINES:
        raise ValueError(f"--engine must be one of {ENGINES}, not {engine!r}.")
    if engine != "auto":
        return engine
    try:
        import newton  # noqa: F401
    except ImportError:
        return "mujoco"
    return "newton" if cuda_available() else "mujoco"


def resolve_renderer(renderer: str, engine: str) -> str:
    if renderer not in RENDERERS:
        raise ValueError(f"--renderer must be one of {RENDERERS}, not {renderer!r}.")
    if renderer != "auto":
        return renderer
    try:
        import mujoco_warp  # noqa: F401
    except ImportError:
        return "opengl"
    return "mjwarp" if engine == "newton" and cuda_available() else "opengl"


# --------------------------------------------------------------- Newton


def _free_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[:3], q[4:7], q[3:4]])


def _free_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[:3], q[6:7], q[3:6]])


class NewtonEngine:
    """Steps the scene with Newton, mirroring it into the server's MuJoCo data.

    `push(data)` copies a pose set in the MuJoCo data (a reset, a teleport) into
    Newton; `set_ctrl` the actuator targets; `step(n)` advances Newton; `pull(data)`
    copies the joint positions back for the state, the window and the cameras.
    """

    def __init__(self, mjcf: str, model: mujoco.MjModel, device: str | None = None):
        import newton
        import warp as wp

        self.newton, self.wp = newton, wp
        wp.config.quiet = True
        self.device = device or ("cuda:0" if cuda_available() else "cpu")
        builder = newton.ModelBuilder()
        # ctrl_direct: the MJCF's actuators take `control.mujoco.ctrl`, in MJCF
        # order, the same vector as MuJoCo's data.ctrl.
        builder.add_mjcf(mjcf, ctrl_direct=True)
        self.nm = builder.finalize(device=self.device)
        self.cpu = not self.device.startswith("cuda")
        self.solver = newton.solvers.SolverMuJoCo(self.nm, use_mujoco_cpu=self.cpu)
        self.dt = float(model.opt.timestep)
        self.s0, self.s1 = self.nm.state(), self.nm.state()
        self.control = self.nm.control()
        if self.control.mujoco.ctrl.shape[0] != model.nu:
            raise RuntimeError(f"Newton has {self.control.mujoco.ctrl.shape[0]} actuators, MuJoCo {model.nu}.")
        self._map_joints(model)
        self._map_mocap(model)
        newton.eval_fk(self.nm, self.nm.joint_q, self.nm.joint_qd, self.s0)
        self._graph = None
        self._carry = 0
        if not self.cpu:
            self._capture()
        logger.info("Newton engine on %s (%s).", self.device, "MuJoCo C" if self.cpu else "MuJoCo Warp")

    def _map_joints(self, model: mujoco.MjModel) -> None:
        labels = [str(l).split("/")[-1] for l in self.nm.joint_label]
        types = self.nm.joint_type.numpy()
        starts = self.nm.joint_q_start.numpy()
        free_type = int(self.newton.JointType.FREE)
        self.slices = []  # (mujoco qpos adr, newton q start, size, is free)
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            jt = model.jnt_type[j]
            size = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(jt, 1)
            matches = [k for k, l in enumerate(labels) if l == name or l.endswith(f"_{name}")]
            if not matches:
                raise RuntimeError(f"Joint {name!r} is not in the Newton model ({labels[:10]}...).")
            k = matches[0]
            free = int(types[k]) == free_type
            self.slices.append((int(model.jnt_qposadr[j]), int(starts[k]), size, free))

    def _map_mocap(self, model: mujoco.MjModel) -> None:
        """MuJoCo mocap bodies (the place pad) to the solver's mocap slots, by body name."""
        smj = self.solver.mj_model
        names = [mujoco.mj_id2name(smj, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(smj.nbody)]
        self.mocap = []
        for b in range(model.nbody):
            m = model.body_mocapid[b]
            if m < 0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            hits = [i for i, n in enumerate(names) if n == name or n.endswith(f"_{name}")]
            if hits and smj.body_mocapid[hits[0]] >= 0:
                self.mocap.append((int(m), int(smj.body_mocapid[hits[0]])))
            else:
                logger.warning("Mocap body %r has no counterpart in Newton; it will not move there.", name)

    def _capture(self) -> None:
        """Two steps as one CUDA graph (two, so the states end where they began)."""
        wp = self.wp
        try:
            self._raw_step()
            self._raw_step()  # Warm up outside the capture.
            with wp.ScopedCapture() as capture:
                self._raw_step()
                self._raw_step()
            self._graph = capture.graph
        except Exception as exc:  # noqa: BLE001
            logger.warning("No CUDA graph (%s); stepping without one.", exc)
            self._graph = None

    def _raw_step(self) -> None:
        self.s0.clear_forces()
        self.solver.step(self.s0, self.s1, self.control, None, self.dt)
        self.s0, self.s1 = self.s1, self.s0

    def step(self, n: int) -> None:
        if self._graph is None:
            for _ in range(n):
                self._raw_step()
            return
        # The graph runs steps in pairs; an odd step waits for the next call.
        self._carry += n
        for _ in range(self._carry // 2):
            self.wp.capture_launch(self._graph)
        self._carry %= 2

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self.control.mujoco.ctrl.assign(np.asarray(ctrl, dtype=np.float32))

    def push(self, data: mujoco.MjData) -> None:
        """Make Newton's state the MuJoCo data's pose, at rest."""
        q = self.s0.joint_q.numpy()
        for adr, start, size, free in self.slices:
            v = data.qpos[adr : adr + size]
            q[start : start + size] = _free_wxyz_to_xyzw(v) if free else v
        for s in (self.s0, self.s1):
            s.joint_q.assign(q)
            s.joint_qd.zero_()
            self.newton.eval_fk(self.nm, s.joint_q, s.joint_qd, s)
        for ours, theirs in self.mocap:
            if self.cpu:
                self.solver.mj_data.mocap_pos[theirs] = data.mocap_pos[ours]
                self.solver.mj_data.mocap_quat[theirs] = data.mocap_quat[ours]
            else:
                pos = self.solver.mjw_data.mocap_pos.numpy()
                quat = self.solver.mjw_data.mocap_quat.numpy()
                pos[:, theirs] = data.mocap_pos[ours]
                quat[:, theirs] = data.mocap_quat[ours]
                self.solver.mjw_data.mocap_pos.assign(pos)
                self.solver.mjw_data.mocap_quat.assign(quat)
        self.set_ctrl(data.ctrl)

    def pull(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Copy Newton's joint positions into the MuJoCo data and update its poses."""
        q = self.s0.joint_q.numpy()
        for adr, start, size, free in self.slices:
            v = q[start : start + size]
            data.qpos[adr : adr + size] = _free_xyzw_to_wxyz(v) if free else v
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)


# ------------------------------------------------------------- renderers


class OpenGLRenderer:
    """MuJoCo's own renderer, one framebuffer per image size."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def looks_changed(self) -> None:
        pass  # It reads the model's materials and lights on every frame.

    def render(self, data: mujoco.MjData, cameras: list[str], width: int, height: int) -> dict[str, np.ndarray]:
        key = (int(width), int(height))
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=key[1], width=key[0])
        r = self._renderers[key]
        out = {}
        for cam in cameras:
            r.update_scene(data, camera=cam)
            out[cam] = r.render()
        return out

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


# Model arrays the scene changes at reset (desk looks, room lights), copied to
# the GPU model before the next frame.
_LOOK_FIELDS = (
    "mat_rgba", "mat_texid", "mat_specular", "mat_shininess", "mat_reflectance", "mat_emission",
    "light_active", "light_diffuse", "light_specular", "light_dir", "light_castshadow",
)


class MjWarpRenderer:
    """MuJoCo Warp's ray-traced renderer: shadows, textures, anti-aliasing, on the GPU."""

    def __init__(self, model: mujoco.MjModel, samples_per_pixel: int = 2):
        import mujoco_warp as mjw
        import warp as wp

        self.mjw, self.wp = mjw, wp
        self.model = model
        self.samples = samples_per_pixel
        with warnings.catch_warnings():
            # About contact pairs: this copy only renders, it never simulates.
            warnings.simplefilter("ignore", UserWarning)
            self.m = mjw.put_model(model)
        self._data = mujoco.MjData(model)
        self.d = mjw.put_data(model, self._data)
        self._contexts: dict[tuple, object] = {}
        self._out: dict[tuple[int, int], object] = {}

    def looks_changed(self) -> None:
        for name in _LOOK_FIELDS:
            dev = getattr(self.m, name, None)
            if dev is None or not hasattr(dev, "assign"):
                continue
            cur = dev.numpy()
            host = np.asarray(getattr(self.model, name)).astype(cur.dtype)
            if host.size == cur.size:
                dev.assign(host.reshape(cur.shape))
            elif host.size * cur.shape[0] == cur.size:  # One copy per world.
                dev.assign(np.broadcast_to(host.reshape(cur.shape[1:]), cur.shape).copy())
            else:
                logger.debug("Not refreshing %s on the GPU: shape %s vs %s.", name, cur.shape, host.shape)

    def _context(self, cameras: tuple[str, ...], width: int, height: int):
        key = (cameras, width, height)
        if key not in self._contexts:
            self._contexts[key] = self.mjw.create_render_context(
                self.model, nworld=1, cam_res=(width, height), render_rgb=True, render_depth=False,
                use_textures=True, use_shadows=True, cam_active=list(cameras),
                samples_per_pixel=self.samples,
            )
        return self._contexts[key]

    def render(self, data: mujoco.MjData, cameras: list[str], width: int, height: int) -> dict[str, np.ndarray]:
        wp, mjw = self.wp, self.mjw
        cams = tuple(cameras)
        rc = self._context(cams, int(width), int(height))
        self.d.qpos.assign(data.qpos[None].astype(np.float32))
        if self.model.nmocap:
            self.d.mocap_pos.assign(data.mocap_pos[None].astype(np.float32))
            self.d.mocap_quat.assign(data.mocap_quat[None].astype(np.float32))
        mjw.kinematics(self.m, self.d)
        mjw.camlight(self.m, self.d)
        mjw.refit_bvh(self.m, self.d, rc)  # The ray tracer's bounds, where things are now.
        mjw.render(self.m, self.d, rc)
        key = (int(height), int(width))
        if key not in self._out:
            self._out[key] = wp.zeros((1, key[0], key[1]), dtype=wp.vec3)
        out = {}
        for i, cam in enumerate(cams):
            mjw.get_rgb(rc, i, self._out[key])
            img = self._out[key].numpy()[0]
            out[cam] = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return out

    def close(self) -> None:
        self._contexts.clear()
