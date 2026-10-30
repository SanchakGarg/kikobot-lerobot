"""Live view of the real arm's commanded motion, rendered from the MuJoCo model.

It shows the shaped trajectory the servos are being sent (the same motion the
sim robot would make), not the measured pose. In rerun it also plots each
joint's commanded and measured angle, so tracking error is visible at a glance.

Modes: `rerun` (default), `mujoco` (native window), `both`, `none`.
"""

from __future__ import annotations

import logging

import mujoco
import mujoco.viewer
import numpy as np

from .config import ARM_JOINTS
from .paths import ARM_SCENE_PATH

logger = logging.getLogger(__name__)

DISPLAY_MODES = ("rerun", "mujoco", "both", "none")
SIM_IMAGE = "sim"


class SimDisplay:
    def __init__(self, mode: str, width: int = 640, height: int = 480):
        if mode not in DISPLAY_MODES:
            raise ValueError(f"display must be one of {DISPLAY_MODES}, not {mode!r}.")
        self.mode = mode
        m = self.model = mujoco.MjModel.from_xml_path(ARM_SCENE_PATH)
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
        self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._camera = mujoco.MjvCamera()
        self._camera.lookat[:] = (0.0, 0.0, 0.15)
        self._camera.distance = 0.75
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
        if self._rec is not None:
            rr = self._rr
            self._renderer.update_scene(d, camera=self._camera)
            self._rec.log(SIM_IMAGE, rr.Image(self._renderer.render()))
            for i, j in enumerate(ARM_JOINTS):
                self._rec.log(f"joints/{j}/commanded", rr.Scalars(float(np.rad2deg(commanded[i]))))
                self._rec.log(f"joints/{j}/measured", rr.Scalars(float(np.rad2deg(measured[i]))))

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._rec is not None:
            self._rec.flush()
            self._rec = None
