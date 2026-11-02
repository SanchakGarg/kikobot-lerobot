"""Procedural meshes for the desk sorting station, shared by both simulators.

Every item kind and every station part is built from a few primitives
(revolved profiles, tubes, boxes) as unshared-vertex meshes with normals and
UVs. The Isaac Sim server turns them into USD meshes (isaac/desk_assets.py),
the MuJoCo scene into MuJoCo meshes (desk/mujoco_desk.py).

Item frame, as in layout.py: long axis X, resting face down, centred.

Pure numpy, no package imports: the Isaac Sim server loads this file by path.
"""

from __future__ import annotations

import math

import numpy as np

# Looks of the parts that are not randomized: name -> (rgb, roughness, metallic).
FIXED_LOOKS = {
    "cup_blue": ((0.10, 0.22, 0.45), 0.35, 0.6),
    "felt": ((0.20, 0.30, 0.24), 0.95, 0.0),
    "metal": ((0.62, 0.63, 0.66), 0.25, 1.0),
    "wood": ((0.80, 0.62, 0.42), 0.8, 0.0),
    "graphite": ((0.08, 0.08, 0.09), 0.45, 0.3),
    "card": ((0.55, 0.42, 0.28), 0.9, 0.0),
    "cup": ((0.18, 0.19, 0.21), 0.35, 0.8),
    "tray": ((0.88, 0.88, 0.86), 0.55, 0.0),
    "tray_dark": ((0.30, 0.32, 0.35), 0.55, 0.0),
}

class MeshBuilder:
    """Unshared-vertex mesh with per-vertex normals and UVs. `face` fixes the
    winding itself from the outward direction it is given."""

    def __init__(self):
        self.points: list[np.ndarray] = []
        self.normals: list[np.ndarray] = []
        self.st: list[np.ndarray] = []
        self.counts: list[int] = []

    def face(self, pts, nrm, st, outward) -> None:
        pts, st = np.asarray(pts, float), np.asarray(st, float)
        nrm = np.asarray(nrm, float)
        if nrm.ndim == 1:
            nrm = np.repeat(nrm[None], len(pts), 0)
        # Newell's method: polygon normal from its winding.
        n = np.zeros(3)
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
        if n @ np.asarray(outward, float) < 0:
            pts, nrm, st = pts[::-1], nrm[::-1], st[::-1]
        self.points.extend(pts)
        self.normals.extend(nrm)
        self.st.extend(st)
        self.counts.append(len(pts))

    def extend(self, other: "MeshBuilder", transform=None) -> "MeshBuilder":
        R, t = transform if transform is not None else (np.eye(3), np.zeros(3))
        # einsum, not @: NumPy 2 on macOS's Accelerate raises spurious
        # floating-point warnings from matmul on small arrays.
        self.points.extend(np.einsum("ij,nj->ni", R, np.asarray(other.points, float)) + t)
        self.normals.extend(np.einsum("ij,nj->ni", R, np.asarray(other.normals, float)))
        self.st.extend(other.st)
        self.counts.extend(other.counts)
        return self

    def arrays(self):
        return (np.asarray(self.points), np.asarray(self.normals), np.asarray(self.st), list(self.counts))


def _norm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def revolve_x(r0: float, r1: float, x0: float, x1: float, n: int = 24, cap0: bool = True, cap1: bool = True,
              flat: bool = False, offset: float = 0.0, inward: bool = False, v_range=(0.0, 1.0)) -> MeshBuilder:
    """Frustum along X from (x0, r0) to (x1, r1). u around, v along."""
    m = MeshBuilder()
    a = offset + 2 * math.pi * np.arange(n + 1) / n
    dx = x1 - x0
    slope = -(r1 - r0) / dx if dx else 0.0
    sgn = -1.0 if inward else 1.0
    for k in range(n):
        ak, ak1 = a[k], a[k + 1]
        p = [(x0, r0 * math.cos(ak), r0 * math.sin(ak)), (x0, r0 * math.cos(ak1), r0 * math.sin(ak1)),
             (x1, r1 * math.cos(ak1), r1 * math.sin(ak1)), (x1, r1 * math.cos(ak), r1 * math.sin(ak))]
        if flat:
            am = 0.5 * (ak + ak1)
            nr = [sgn * _norm((slope, math.cos(am), math.sin(am)))] * 4
        else:
            nr = [sgn * _norm((slope, math.cos(t), math.sin(t))) for t in (ak, ak1, ak1, ak)]
        u0, u1 = k / n, (k + 1) / n
        st = [(u0, v_range[0]), (u1, v_range[0]), (u1, v_range[1]), (u0, v_range[1])]
        am = 0.5 * (ak + ak1)
        m.face(p, nr, st, sgn * np.array([0.0, math.cos(am), math.sin(am)]))
    for cap, x, r, d in ((cap0, x0, r0, -1.0), (cap1, x1, r1, 1.0)):
        if not cap or r <= 0:
            continue
        ring = [(x, r * math.cos(t), r * math.sin(t)) for t in a[:-1]]
        st = [(0.5 + 0.5 * math.cos(t), 0.5 + 0.5 * math.sin(t)) for t in a[:-1]]
        m.face(ring, (d, 0, 0), st, (d, 0, 0))
    return m


def tube_x(r_in: float, r_out: float, x0: float, x1: float, n: int = 32) -> MeshBuilder:
    """Hollow cylinder along X (tape, cup, scissor loops)."""
    m = revolve_x(r_out, r_out, x0, x1, n, cap0=False, cap1=False)
    m.extend(revolve_x(r_in, r_in, x0, x1, n, cap0=False, cap1=False, inward=True))
    a = 2 * math.pi * np.arange(n + 1) / n
    for x, d in ((x0, -1.0), (x1, 1.0)):
        for k in range(n):
            t0, t1 = a[k], a[k + 1]
            p = [(x, r_in * math.cos(t0), r_in * math.sin(t0)), (x, r_out * math.cos(t0), r_out * math.sin(t0)),
                 (x, r_out * math.cos(t1), r_out * math.sin(t1)), (x, r_in * math.cos(t1), r_in * math.sin(t1))]
            st = [(k / n, 0), (k / n, 1), ((k + 1) / n, 1), ((k + 1) / n, 0)]
            m.face(p, (d, 0, 0), st, (d, 0, 0))
    return m


def box_mesh(half, center=(0, 0, 0)) -> MeshBuilder:
    m = MeshBuilder()
    h, c = np.asarray(half, float), np.asarray(center, float)
    for ax in range(3):
        t1, t2 = (ax + 1) % 3, (ax + 2) % 3
        for s in (-1.0, 1.0):
            pts = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[ax], p[t1], p[t2] = s * h[ax], a * h[t1], b * h[t2]
                pts.append(c + p)
            nrm = np.zeros(3)
            nrm[ax] = s
            m.face(pts, nrm, [(0, 0), (1, 0), (1, 1), (0, 1)], nrm)
    return m


# X axis -> Z axis (cyclic permutation, a proper rotation).
X_TO_Z = (np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float), np.zeros(3))


def x_to_z(m: MeshBuilder, shift=(0, 0, 0)) -> MeshBuilder:
    R, _ = X_TO_Z
    return MeshBuilder().extend(m, (R, np.asarray(shift, float)))


# ============================================================ procedural items

# Each item is a list of (part, MeshBuilder), part = body | accent | metal | wood | graphite | card.
# Item frame as in desk_layout: long axis X, resting face down, centred.


def procedural_item(kind: str) -> list[tuple[str, MeshBuilder]]:
    if kind == "pen":
        r = 0.005
        return [
            ("metal", revolve_x(0.0008, 0.0035, -0.070, -0.057, 20, cap0=True)),
            ("body", revolve_x(r, r, -0.057, 0.040, 24)),
            ("accent", revolve_x(0.0056, 0.0056, 0.024, 0.068, 24)),
            ("accent", revolve_x(0.0056, 0.0045, 0.068, 0.070, 24, cap0=False)),
            ("accent", box_mesh((0.018, 0.0011, 0.0007), (0.050, 0.0, 0.0062))),
        ]
    if kind == "pencil":
        rc = 0.0075 / 2 / math.sin(math.radians(60))  # corner radius for 7.5 mm across the flats
        return [
            ("graphite", revolve_x(0.0, 0.0012, -0.0875, -0.0855, 6, flat=True)),
            ("wood", revolve_x(0.0012, rc, -0.0855, -0.070, 6, cap0=False, cap1=False, flat=True)),
            ("body", revolve_x(rc, rc, -0.070, 0.0735, 6, flat=True)),
            ("metal", revolve_x(0.0040, 0.0040, 0.0735, 0.0815, 20)),
            ("accent", revolve_x(0.0037, 0.0037, 0.0815, 0.0875, 20)),
        ]
    if kind == "marker":
        return [
            ("accent", revolve_x(0.0014, 0.0040, -0.065, -0.058, 20)),
            ("body", revolve_x(0.0085, 0.0085, -0.058, 0.030, 28)),
            ("accent", revolve_x(0.0090, 0.0090, 0.018, 0.063, 28)),
            ("accent", revolve_x(0.0090, 0.0060, 0.063, 0.065, 28, cap0=False)),
        ]
    if kind == "highlighter":
        return [
            ("accent", box_mesh((0.003, 0.004, 0.002), (-0.0605, 0.0, -0.002))),
            ("body", box_mesh((0.046, 0.011, 0.007), (-0.0115, 0.0, 0.0))),
            ("accent", box_mesh((0.0175, 0.0114, 0.0074), (0.0425, 0.0, 0.0))),
        ]
    if kind == "glue_stick":
        return [
            ("body", revolve_x(0.011, 0.011, -0.0425, 0.018, 32)),
            ("accent", revolve_x(0.0114, 0.0114, 0.012, 0.0425, 32)),
        ]
    if kind == "tape_roll":
        return [
            ("body", x_to_z(tube_x(0.0125, 0.022, -0.0065, 0.0065, 40))),
            ("card", x_to_z(tube_x(0.0105, 0.0125, -0.0066, 0.0066, 40))),
        ]
    if kind == "scissors":
        loops = MeshBuilder()
        for y in (-0.011, 0.011):
            loops.extend(x_to_z(tube_x(0.0065, 0.0105, -0.004, 0.004, 28), (-0.052, y, 0.0)))
        grips = box_mesh((0.012, 0.014, 0.0035), (-0.030, 0.0, 0.0))
        return [
            ("metal", box_mesh((0.046, 0.0065, 0.0012), (0.024, 0.0025, -0.0028))),
            ("metal", box_mesh((0.046, 0.0065, 0.0012), (0.024, -0.0025, -0.0016))),
            ("metal", x_to_z(revolve_x(0.0025, 0.0025, -0.0005, 0.0012, 16), (-0.012, 0.0, 0.0))),
            ("body", loops),
            ("body", grips),
        ]
    if kind == "eraser":
        return [
            ("body", box_mesh((0.022, 0.011, 0.006))),
            ("accent", box_mesh((0.010, 0.0113, 0.0062), (0.008, 0.0, 0.0))),
        ]
    if kind == "sticky_notes":
        return [
            ("body", box_mesh((0.038, 0.038, 0.0075), (0.0, 0.0, -0.0005))),
            ("accent", box_mesh((0.0378, 0.0378, 0.0004), (0.0, 0.0, 0.0074))),  # the top sheet
        ]
    if kind == "stapler":
        return [
            ("accent", box_mesh((0.062, 0.017, 0.006), (0.0, 0.0, -0.016))),  # base
            ("metal", box_mesh((0.050, 0.012, 0.004), (0.008, 0.0, -0.006))),  # magazine
            ("body", box_mesh((0.060, 0.016, 0.010), (0.003, 0.0, 0.012))),  # top arm
            ("body", box_mesh((0.012, 0.016, 0.008), (-0.052, 0.0, -0.004))),  # hinge block
        ]
    if kind == "ruler":
        return [
            ("body", box_mesh((0.100, 0.015, 0.0018))),
            ("accent", box_mesh((0.098, 0.004, 0.0002), (0.0, 0.009, 0.0019))),  # the scale
        ]
    if kind == "paper_clip_box":
        return [
            ("body", box_mesh((0.025, 0.018, 0.0105), (0.0, 0.0, -0.0015))),
            ("accent", box_mesh((0.0252, 0.0182, 0.0015), (0.0, 0.0, 0.0105))),  # lid
        ]
    if kind == "sharpener":
        return [
            ("body", box_mesh((0.013, 0.010, 0.010))),
            ("metal", box_mesh((0.009, 0.0101, 0.0035), (0.002, 0.0, 0.004))),  # blade plate
            ("graphite", revolve_x(0.004, 0.004, 0.0130, 0.0133, 16)),  # the hole
        ]
    raise KeyError(kind)


def station_tube(r_in: float, r_out: float, half_h: float, center=(0.0, 0.0, 0.0)) -> MeshBuilder:
    """An upright hollow cup wall (a station's `tube` part)."""
    c = np.asarray(center, float)
    return x_to_z(tube_x(r_in, r_out, -half_h, half_h, 48), c)


