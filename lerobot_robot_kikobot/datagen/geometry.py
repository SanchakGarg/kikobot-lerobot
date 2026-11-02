"""Small rotation and timing helpers for the scripted expert."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ..kinematics import mat_to_quat, quat_to_mat

UP = np.array([0.0, 0.0, 1.0])


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis / n, angle)
    return quat_to_mat(q)


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation that turns direction `a` onto direction `b`."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # Opposite: half a turn about any axis normal to a.
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return axis_angle(perp, math.pi)
    return axis_angle(axis, math.atan2(s, c))


def rot_angle(r0: np.ndarray, r1: np.ndarray) -> float:
    """Angle (rad) of the rotation from r0 to r1."""
    c = (np.trace(r0.T @ r1) - 1.0) / 2.0
    return float(math.acos(min(1.0, max(-1.0, c))))


def slerp(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    q0, q1 = mat_to_quat(r0), mat_to_quat(r1)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    dot = min(1.0, float(np.dot(q0, q1)))
    if dot > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        th = math.acos(dot)
        q = (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)
    return quat_to_mat(q / np.linalg.norm(q))


def min_jerk(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s**3 * (10.0 - 15.0 * s + 6.0 * s * s)


def mat_to_rpy(r: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw (extrinsic X, Y, Z: R = Rz(yaw) Ry(pitch) Rx(roll))."""
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(r[2, 0]) < 1.0 - 1e-9:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:  # Gimbal lock: put it all in yaw.
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    return np.array([roll, pitch, yaw])


def rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def free_pose(qpos7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position and rotation matrix of a free joint's qpos (wxyz quaternion)."""
    return np.array(qpos7[:3], dtype=float), quat_to_mat(np.asarray(qpos7[3:7], dtype=float))


def aabb_half(rot: np.ndarray, half: np.ndarray) -> np.ndarray:
    """World-axis half extents of a box of `half` extents turned by `rot`."""
    return np.abs(rot) @ np.asarray(half, dtype=float)
