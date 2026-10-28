"""Full 6-DOF inverse kinematics for the Kiko-Bot arm, on the MuJoCo model.

The task point is the grasp centre between the jaws (TCP), and the task is
its full pose: position and orientation. All six joints take part, solved by
damped least squares with the damping raised only near a singularity.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .config import ARM_JOINTS
from .paths import ARM_SCENE_PATH

# Grasp centre in the `eof` body frame, measured between the two jaw meshes.
# The jaws hang along -Z of `eof`.
TCP_OFFSET_EOF = np.array([0.0, 0.0, -0.0758])

# A bent, non-singular pose with the jaws pointing straight down. The all-zero
# `home` keyframe is the arm standing straight up, where the Jacobian loses rank.
READY_POSE_DEG = np.array([90.0, -45.0, 105.0, 60.0, -91.0, 0.0])
READY_TCP_HEIGHT_M = 0.15

# Metres of position error that weigh the same as one radian of orientation
# error. Small enough that position wins when both cannot be met.
ORIENTATION_WEIGHT = 0.15
DAMPING = 0.01
# Below this smallest singular value, damping ramps up to MAX_DAMPING.
SINGULAR_THRESHOLD = 0.04
MAX_DAMPING = 0.08
MAX_STEP_RAD = np.deg2rad(6.0)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(m).reshape(9))
    return q


def axis_angle_quat(rotvec: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        q[0] = 1.0
        return q
    mujoco.mju_axisAngle2Quat(q, rotvec / angle, angle)
    return q


class ArmKinematics:
    """FK and IK of the six arm joints on a private copy of the model."""

    def __init__(self, scene_path: str = ARM_SCENE_PATH):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        if min(ids) < 0:
            raise RuntimeError(f"The model must contain joints {ARM_JOINTS}.")
        self.qpos_adr = self.model.jnt_qposadr[ids]
        self.dof_adr = self.model.jnt_dofadr[ids]
        self.lower = self.model.jnt_range[ids, 0].copy()
        self.upper = self.model.jnt_range[ids, 1].copy()
        self.eof_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "eof")
        if self.eof_id < 0:
            raise RuntimeError("The model must contain the 'eof' body.")
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self._ready_q: np.ndarray | None = None

    def clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP position (m) and rotation matrix (world frame) at joints `q`."""
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        rot = self.data.xmat[self.eof_id].reshape(3, 3).copy()
        pos = self.data.xpos[self.eof_id] + rot @ TCP_OFFSET_EOF
        return pos, rot

    def pose_error(self, q: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Position and orientation error (world frame) of `q` from a target pose."""
        cur_pos, cur_rot = self.fk(q)
        dq = np.zeros(3)
        mujoco.mju_subQuat(dq, mat_to_quat(rot), mat_to_quat(cur_rot))
        # mju_subQuat returns the rotation in the current body frame.
        return pos - cur_pos, cur_rot @ dq

    def ik_step(self, q: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """One damped-least-squares update of `q` toward the target pose."""
        e_pos, e_rot = self.pose_error(q, pos, rot)  # Leaves data at `q`.
        mujoco.mj_comPos(self.model, self.data)
        tcp = self.data.xpos[self.eof_id] + self.data.xmat[self.eof_id].reshape(3, 3) @ TCP_OFFSET_EOF
        mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, tcp, self.eof_id)
        jac = np.vstack((self._jacp[:, self.dof_adr], ORIENTATION_WEIGHT * self._jacr[:, self.dof_adr]))
        err = np.concatenate((e_pos, ORIENTATION_WEIGHT * e_rot))

        sigma_min = np.linalg.svd(jac, compute_uv=False)[-1]
        damping = DAMPING
        if sigma_min < SINGULAR_THRESHOLD:
            damping += (1.0 - (sigma_min / SINGULAR_THRESHOLD) ** 2) * MAX_DAMPING
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)

        biggest = np.max(np.abs(step))
        if biggest > MAX_STEP_RAD:
            step *= MAX_STEP_RAD / biggest
        return self.clip(q + step)

    def solve(self, q: np.ndarray, pos: np.ndarray, rot: np.ndarray, iterations: int) -> np.ndarray:
        for _ in range(iterations):
            q = self.ik_step(q, pos, rot)
        return q

    def ready_q(self) -> np.ndarray:
        """The ready pose, raised so the jaws clear the table. Solved once."""
        if self._ready_q is None:
            q0 = np.deg2rad(READY_POSE_DEG)
            pos, rot = self.fk(q0)
            pos[2] = READY_TCP_HEIGHT_M
            self._ready_q = self.solve(q0, pos, rot, iterations=300)
        return self._ready_q.copy()
