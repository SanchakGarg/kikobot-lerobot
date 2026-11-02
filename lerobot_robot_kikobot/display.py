"""Live view of the real arm's commanded motion, rendered from the MuJoCo model.

It shows the shaped trajectory the servos are being sent (the same motion the
sim robot would make), not the measured pose. In rerun it also plots each
joint's commanded and measured angle, so tracking error is visible at a glance.

Modes: `rerun` (default), `mujoco` (native window), `both`, `none`.

Sending to rerun happens on a thread of its own and drops frames when the
viewer falls behind: a blocked log call must never stall the control loop.
"""

from __future__ import annotations

import logging
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from .config import ARM_JOINTS
from .paths import resolve_scene

logger = logging.getLogger(__name__)

DISPLAY_MODES = ("rerun", "mujoco", "both", "none")
SIM_IMAGE = "sim"
# Rerun frame rate. Each frame is a JPEG of a few tens of kB; raw 640x480
# frames at the 30 Hz loop rate (27 MB/s) are more than the viewer keeps up with.
RERUN_HZ = 15.0
JPEG_QUALITY = 80


class SimDisplay:
    def __init__(self, mode: str, scene: str = "pick_place", width: int = 640, height: int = 480):
        if mode not in DISPLAY_MODES:
            raise ValueError(f"display must be one of {DISPLAY_MODES}, not {mode!r}.")
        self.mode = mode
        m = self.model = mujoco.MjModel.from_xml_path(resolve_scene(scene))
        self.data = mujoco.MjData(m)

        def joint_adr(name: str) -> int:
            return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])

        self._arm_adr = [joint_adr(j) for j in ARM_JOINTS]
        # finger_r mirrors finger_l through an equality constraint, which only
        # acts under simulation; set both directly here.
        self._finger_adr = [joint_adr("finger_l"), joint_adr("finger_r")]
        grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_l")
        self._grip_closed, self._grip_open = m.actuator_ctrlrange[grip]

        self._viewer = None
        self._rec = None
        self._renderer = None
        if mode in ("mujoco", "both"):
            self._viewer = mujoco.viewer.launch_passive(m, self.data)
        if mode in ("rerun", "both"):
            self._open_rerun(width, height)

    def _open_rerun(self, width: int, height: int) -> None:
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except ImportError:
            logger.warning("rerun-sdk is not installed (pip install rerun-sdk); no rerun display.")
            return
        self._rr = rr
        # A recording of its own, so it does not clash with LeRobot's
        # `--display_data` recording if that is on too.
        self._rec = rr.RecordingStream("kikobot")
        self._rec.spawn(memory_limit="10%")
        self._rec.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=SIM_IMAGE, name="Sim (commanded)"),
                    rrb.Grid(*[rrb.TimeSeriesView(origin=f"joints/{j}", name=f"{j} (deg)") for j in ARM_JOINTS]),
                    column_shares=[3, 2],
                )
            )
        )
        # Rendered on the caller's thread (GL contexts are bound to it), sent on
        # this one. One pending frame at most; a newer one replaces it.
        self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._last_frame_t = 0.0
        self._pending = None
        self._pending_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._sender = threading.Thread(target=self._send_loop, name="kikobot-rerun", daemon=True)
        self._sender.start()
        self._camera = mujoco.MjvCamera()
        # The whole desk with the arm on it.
        self._camera.lookat[:] = (0.1, 0.05, 0.1)
        self._camera.distance = 1.1
        self._camera.azimuth = 135.0
        self._camera.elevation = -25.0

    def update(self, commanded: np.ndarray, gripper_pct: float, measured: np.ndarray) -> None:
        """Show `commanded` (radians, J1..J6) and plot it against `measured`."""
        d = self.data
        d.qpos[self._arm_adr] = commanded[: len(ARM_JOINTS)]
        finger = self._grip_closed + np.clip(gripper_pct, 0.0, 100.0) / 100.0 * (self._grip_open - self._grip_closed)
        d.qpos[self._finger_adr] = finger
        mujoco.mj_forward(self.model, d)

        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
        now = time.perf_counter()
        if self._rec is not None and now - self._last_frame_t >= 1.0 / RERUN_HZ:
            self._last_frame_t = now
            self._renderer.update_scene(d, camera=self._camera)
            frame = (self._renderer.render(), np.rad2deg(commanded), np.rad2deg(measured))
            with self._pending_lock:
                self._pending = frame
            self._wake.set()

    def _send_loop(self) -> None:
        rr = self._rr
        while not self._stop.is_set():
            if not self._wake.wait(timeout=0.5):
                continue
            self._wake.clear()
            with self._pending_lock:
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            image, commanded, measured = frame
            try:
                self._rec.log(SIM_IMAGE, rr.Image(image).compress(jpeg_quality=JPEG_QUALITY))
                for i, j in enumerate(ARM_JOINTS):
                    self._rec.log(f"joints/{j}/commanded", rr.Scalars(float(commanded[i])))
                    self._rec.log(f"joints/{j}/measured", rr.Scalars(float(measured[i])))
            except Exception as exc:  # The view is optional; never take teleop down.
                logger.warning("Rerun display stopped: %s", exc)
                return

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._rec is not None:
            self._stop.set()
            self._wake.set()
            # A sender stuck on a closed viewer is a daemon; do not wait on it.
            self._sender.join(timeout=1.0)
            self._rec = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
