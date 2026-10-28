"""The Kiko-Bot MuJoCo scene behind LeRobot's `Robot` interface.

Same keys and units as the real arm. Each `send_action` advances physics by
one control period, running the same `MotionShaper` as the hardware on every
physics step, so what you see in sim is what the servos will do.
"""

from __future__ import annotations

import math
from functools import cached_property
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config import ARM_JOINTS, KikobotSimConfig
from .kinematics import ArmKinematics
from .motion import MotionShaper
from .paths import resolve_scene

# Physics steps per control tick; `connect` sets the timestep to fit.
SUBSTEPS = 17
# Objects respawn in a fan in front of the arm, within comfortable reach.
SPAWN_RADIUS_M = (0.22, 0.32)
SPAWN_HALF_ANGLE_RAD = math.radians(40.0)
MIN_OBJECT_PAD_GAP_M = 0.12


class KikobotSim(Robot):
    config_class = KikobotSimConfig
    name = "kikobot_sim"

    def __init__(self, config: KikobotSimConfig):
        super().__init__(config)
        self.config = config
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.rng = np.random.default_rng(config.seed)
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None
        self._shaper: MotionShaper | None = None
        self._target = np.zeros(len(ARM_JOINTS))
        self._gripper_pct = 100.0

    # ---------------------------------------------------------------- features

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {**{f"{j}.pos": float for j in ARM_JOINTS}, "gripper.pos": float}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        cams = {c: (self.config.height, self.config.width, 3) for c in self.config.camera_names}
        return {**self._motors_ft, **cams}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # -------------------------------------------------------------- connection

    @property
    def is_connected(self) -> bool:
        return self.data is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        m = self.model = mujoco.MjModel.from_xml_path(resolve_scene(self.config.scene))
        m.opt.timestep = 1.0 / (self.config.fps * SUBSTEPS)
        self.data = mujoco.MjData(m)

        def id_of(kind, name):
            return mujoco.mj_name2id(m, kind, name)

        joints = [id_of(mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self._qpos_adr = m.jnt_qposadr[joints]
        self._arm_act = np.array([id_of(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ARM_JOINTS])
        self._grip_act = id_of(mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_l")
        self._grip_qpos = m.jnt_qposadr[id_of(mujoco.mjtObj.mjOBJ_JOINT, "finger_l")]
        self._grip_closed, self._grip_open = m.actuator_ctrlrange[self._grip_act]
        self._lower = m.jnt_range[joints, 0].copy()
        self._upper = m.jnt_range[joints, 1].copy()
        # The first free joint is the object to pick; the pad is a mocap body.
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self._object_adr = int(m.jnt_qposadr[free[0]]) if free else None
        pad = id_of(mujoco.mjtObj.mjOBJ_BODY, "place_pad")
        self._pad_mocap = int(m.body_mocapid[pad]) if pad >= 0 else -1

        self._shaper = MotionShaper(len(ARM_JOINTS), self.config.motion)
        self._renderer = mujoco.Renderer(m, height=self.config.height, width=self.config.width)
        self.reset_scene()
        if self.config.viewer:
            self._viewer = mujoco.viewer.launch_passive(m, self.data)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._renderer.close()
        self._renderer = None
        self.model = self.data = None

    # ----------------------------------------------------------------- episode

    def reset_scene(self) -> None:
        """Arm to the ready pose (where the gamepad teleop starts), jaws open,
        object and pad respawned. LeRobot has no episode hook; scripts call it."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        ready = ArmKinematics().ready_q()
        d.qpos[self._qpos_adr] = ready
        d.ctrl[self._arm_act] = ready
        self._gripper_pct = 100.0
        d.ctrl[self._grip_act] = self._grip_open
        self._target = ready.copy()
        self._respawn_objects()
        mujoco.mj_forward(m, d)
        self._shaper.reset(ready, d.time)

    def _sample_xy(self) -> np.ndarray:
        r = self.rng.uniform(*SPAWN_RADIUS_M)
        a = self.rng.uniform(-SPAWN_HALF_ANGLE_RAD, SPAWN_HALF_ANGLE_RAD)
        return np.array([r * math.cos(a), r * math.sin(a)])

    def _respawn_objects(self) -> None:
        obj_xy = self._sample_xy()
        if self._object_adr is not None:
            adr = self._object_adr
            # Keep the XML's height and tilt (a lying cylinder stays lying),
            # then spin about world Z.
            base = self.model.qpos0[adr + 3 : adr + 7]
            yaw = np.zeros(4)
            mujoco.mju_axisAngle2Quat(yaw, np.array([0.0, 0.0, 1.0]), self.rng.uniform(-math.pi, math.pi))
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, yaw, base)
            self.data.qpos[adr : adr + 3] = (*obj_xy, self.model.qpos0[adr + 2])
            self.data.qpos[adr + 3 : adr + 7] = quat
        if self._pad_mocap >= 0:
            pad_xy = self._sample_xy()
            for _ in range(100):
                if np.linalg.norm(pad_xy - obj_xy) >= MIN_OBJECT_PAD_GAP_M:
                    break
                pad_xy = self._sample_xy()
            self.data.mocap_pos[self._pad_mocap][:2] = pad_xy

    # --------------------------------------------------------------------- i/o

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        d = self.data
        obs: dict[str, Any] = {
            f"{j}.pos": float(np.rad2deg(d.qpos[a])) for j, a in zip(ARM_JOINTS, self._qpos_adr)
        }
        span = self._grip_open - self._grip_closed
        obs["gripper.pos"] = float(100.0 * (d.qpos[self._grip_qpos] - self._grip_closed) / span)
        for cam in self.config.camera_names:
            self._renderer.update_scene(d, camera=cam)
            obs[cam] = self._renderer.render()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        m, d = self.model, self.data
        for i, j in enumerate(ARM_JOINTS):
            if f"{j}.pos" in action:
                self._target[i] = math.radians(action[f"{j}.pos"])
        self._target = np.clip(self._target, self._lower, self._upper)
        if "gripper.pos" in action:
            self._gripper_pct = float(np.clip(action["gripper.pos"], 0.0, 100.0))

        self._shaper.set_target(self._target, d.time)
        d.ctrl[self._grip_act] = self._grip_closed + self._gripper_pct / 100.0 * (
            self._grip_open - self._grip_closed
        )
        for _ in range(SUBSTEPS):
            d.ctrl[self._arm_act] = self._shaper.step(d.time)
            mujoco.mj_step(m, d)
        if self._viewer is not None:
            self._viewer.sync()

        sent = {f"{j}.pos": float(np.rad2deg(self._target[i])) for i, j in enumerate(ARM_JOINTS)}
        sent["gripper.pos"] = self._gripper_pct
        return sent
