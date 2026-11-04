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


def scaled(m: MeshBuilder, scale) -> MeshBuilder:
    """`m` stretched by `scale` (x, y, z); normals follow (inverse transpose)."""
    s = np.asarray(scale, float)
    out = MeshBuilder()
    out.points = [np.asarray(p, float) * s for p in m.points]
    out.normals = [_norm(np.asarray(n, float) / s) for n in m.normals]
    out.st = list(m.st)
    out.counts = list(m.counts)
    return out


def prism(profile, axis: int, a0: float, a1: float) -> MeshBuilder:
    """A convex polygon extruded along `axis` (0 X, 1 Y, 2 Z) from a0 to a1.
    `profile` is (u, v) points in the other two axes, in cyclic order
    (X: (y, z), Y: (x, z), Z: (x, y))."""
    m = MeshBuilder()
    others = [i for i in range(3) if i != axis]
    prof = np.asarray(profile, float)
    n = len(prof)
    centre2 = prof.mean(0)

    def point(uv, a):
        p = np.zeros(3)
        p[others[0]], p[others[1]], p[axis] = uv[0], uv[1], a
        return p

    span = np.ptp(prof, axis=0)
    span[span == 0] = 1.0
    for a, d in ((a0, -1.0), (a1, 1.0)):
        nrm = np.zeros(3)
        nrm[axis] = d
        st = (prof - prof.min(0)) / span
        m.face([point(uv, a) for uv in prof], nrm, st, nrm)
    for i in range(n):
        p, q = prof[i], prof[(i + 1) % n]
        edge = q - p
        out2 = np.array([edge[1], -edge[0]])
        if out2 @ ((p + q) / 2 - centre2) < 0:
            out2 = -out2
        nrm = np.zeros(3)
        nrm[others[0]], nrm[others[1]] = out2
        nrm = _norm(nrm)
        u0, u1 = i / n, (i + 1) / n
        m.face([point(p, a0), point(q, a0), point(q, a1), point(p, a1)], nrm,
               [(u0, 0), (u1, 0), (u1, 1), (u0, 1)], nrm)
    return m


def rounded_rect(hu: float, hv: float, r: float, n: int = 4) -> list[tuple[float, float]]:
    """A rectangle's outline with rounded corners (convex), counter-clockwise."""
    r = min(r, hu, hv)
    pts = []
    for cu, cv, a0 in ((hu - r, hv - r, 0.0), (-hu + r, hv - r, 90.0), (-hu + r, -hv + r, 180.0), (hu - r, -hv + r, 270.0)):
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            pts.append((cu + r * math.cos(a), cv + r * math.sin(a)))
    return pts


def rounded_box(half, r: float, center=(0, 0, 0), n: int = 4) -> MeshBuilder:
    """A box whose four edges along Z are rounded (corner radius r, seen from above)."""
    hx, hy, hz = half
    c = np.asarray(center, float)
    m = prism(rounded_rect(hx, hy, r, n), 2, -hz, hz)
    return MeshBuilder().extend(m, (np.eye(3), c))


def rounded_bar_x(hy: float, hz: float, r: float, x0: float, x1: float, n: int = 4) -> MeshBuilder:
    """A bar along X whose cross-section is a rounded rectangle (2hy x 2hz)."""
    return prism(rounded_rect(hy, hz, r, n), 0, x0, x1)


def _ribs_x(r_lo: float, r_hi: float, x0: float, x1: float, count: int, n: int = 24) -> MeshBuilder:
    """Alternating raised rings along X: a ribbed grip."""
    m = MeshBuilder()
    edges = np.linspace(x0, x1, 2 * count + 1)
    for k in range(2 * count):
        r = r_hi if k % 2 == 0 else r_lo
        m.extend(revolve_x(r, r, edges[k], edges[k + 1], n, cap0=True, cap1=True,
                           v_range=((edges[k] - x0) / (x1 - x0), (edges[k + 1] - x0) / (x1 - x0))))
    return m


# ============================================================ procedural items

# Each item is a list of (part, MeshBuilder), part = body | accent | metal | wood | graphite | card.
# Item frame as in layout.py: long axis X, resting face down, centred. Every
# item is built from its own size (`half`, its collider's half extents), so
# a pool item of any of its kind's sizes looks right.


def _kinds():
    """layout.KINDS: from the package, or, in the Isaac Sim server (which
    loads these files by path), from the module it loaded as `desk_layout`."""
    try:
        from .layout import KINDS  # noqa: PLC0415
    except ImportError:
        from desk_layout import KINDS  # noqa: PLC0415
    return KINDS


def procedural_item(kind: str, half=None) -> list[tuple[str, MeshBuilder]]:
    """The item's parts at size `half` (its collider's half extents; the
    kind's usual size if None)."""
    return _ITEMS[kind](*(half if half is not None else _kinds()[kind].half))


def _pen(hx, hy, hz):
    """A click ballpoint: metal tip cone, ribbed rubber grip, barrel, a clip
    near the end, and the push button."""
    r = hy
    L = 2 * hx
    x_tip, x_cone = -hx, -hx + 0.012 * L / 0.14
    x_grip = x_cone + 0.032 * L / 0.14
    x_clip0, x_end = hx - 0.05 * L / 0.14, hx - 0.004
    clip_z = r + 0.0009
    return [
        ("metal", revolve_x(0.0003, 0.0009, x_tip, x_tip + 0.0012, 12)),
        ("metal", revolve_x(0.0009, 0.42 * r, x_tip + 0.0012, x_tip + 0.004, 16, cap0=False)),
        ("body", revolve_x(0.42 * r, 0.9 * r, x_tip + 0.004, x_cone, 24, cap0=False)),
        ("accent", _ribs_x(0.97 * r, r, x_cone, x_grip, 6)),
        ("body", revolve_x(r, r, x_grip, x_end, 24, v_range=(0.25, 1.0))),
        ("body", revolve_x(r, 0.85 * r, x_end, hx - 0.0025, 24, cap0=False)),
        ("accent", revolve_x(0.5 * r, 0.45 * r, hx - 0.0025, hx, 16, cap0=False)),
        # The clip: a strip along the barrel, standing off it, with a bead at its tip.
        ("metal", box_mesh((0.5 * (x_end - x_clip0), 0.0011, 0.00045), (0.5 * (x_clip0 + x_end), 0.0, clip_z))),
        ("metal", box_mesh((0.002, 0.0011, 0.0009), (x_end - 0.002, 0.0, r + 0.0003))),
        ("metal", box_mesh((0.0015, 0.0013, 0.0008), (x_clip0 + 0.0015, 0.0, clip_z - 0.0003))),
    ]


def _pencil(hx, hy, hz):
    """A hex pencil: graphite point, the sharpened wood cone, the painted
    body, a crimped metal ferrule and a rounded eraser."""
    rc = hy / math.cos(math.radians(30))  # Corner radius of the hexagon, flats 2*hy apart.
    x0 = -hx
    point, cone = 0.003, 0.016
    ferrule, eraser = 0.009, 0.006
    xb0, xb1 = x0 + point + cone, hx - ferrule - eraser
    xf1 = xb1 + ferrule
    return [
        ("graphite", revolve_x(0.0002, 0.0011, x0, x0 + point, 12)),
        ("wood", revolve_x(0.0011, 0.92 * hy, x0 + point, xb0, 24, cap0=False, cap1=False)),
        ("wood", revolve_x(0.92 * hy, rc, xb0 - 0.0005, xb0, 6, flat=True, cap0=False)),
        ("body", revolve_x(rc, rc, xb0, xb1, 6, flat=True)),
        ("metal", revolve_x(1.02 * hy, 1.02 * hy, xb1, xf1, 24)),
        ("metal", _ribs_x(1.02 * hy, 1.07 * hy, xb1 + 0.0015, xb1 + 0.0045, 2)),
        ("accent", revolve_x(0.95 * hy, 0.95 * hy, xf1, hx - 0.0015, 20, cap0=False)),
        ("accent", revolve_x(0.95 * hy, 0.6 * hy, hx - 0.0015, hx, 20, cap0=False)),
    ]


def _marker(hx, hy, hz):
    """A capped marker: the cap over the nib, with a clip and a band, the
    barrel and an end plug in the ink's colour."""
    r = hy
    L = 2 * hx
    rc = 1.06 * r
    x_cap1 = -hx + 0.42 * L
    return [
        ("accent", revolve_x(0.55 * rc, rc, -hx, -hx + 0.003, 24)),
        ("accent", revolve_x(rc, rc, -hx + 0.003, x_cap1, 28, v_range=(0.0, 0.42))),
        ("body", revolve_x(1.0 * rc, 1.0 * rc, x_cap1 - 0.006, x_cap1 - 0.003, 28)),  # band
        ("body", revolve_x(r, r, x_cap1, hx - 0.006, 28, v_range=(0.42, 1.0))),
        ("body", revolve_x(r, 0.9 * r, hx - 0.006, hx - 0.003, 28, cap0=False)),
        ("accent", revolve_x(0.85 * r, 0.7 * r, hx - 0.003, hx, 24, cap0=False)),
        # Clip on the cap, standing off it.
        ("accent", box_mesh((0.36 * (x_cap1 + hx) / 2, 0.0016, 0.0006),
                            (-hx + 0.55 * (x_cap1 + hx), 0.0, rc + 0.0009))),
        ("accent", box_mesh((0.003, 0.0016, 0.001), (x_cap1 - 0.006, 0.0, rc + 0.0003))),
    ]


def _highlighter(hx, hy, hz):
    """A flat highlighter: chisel nib under a cap with a clip, a bevelled barrel."""
    L = 2 * hx
    rcap = (1.03 * hy, 1.05 * hz)
    x_cap0, x_cap1 = -hx + 0.004, -hx + 0.38 * L
    nib = [(-0.35 * hy, -0.45 * hz), (0.35 * hy, -0.45 * hz), (0.35 * hy, 0.1 * hz), (-0.35 * hy, 0.45 * hz)]
    return [
        ("accent", prism(nib, 0, -hx, x_cap0 + 0.0005)),
        ("accent", rounded_bar_x(rcap[0], rcap[1], 0.45 * rcap[1], x_cap0, x_cap1)),
        ("body", rounded_bar_x(hy, hz, 0.5 * hz, x_cap1, hx - 0.003)),
        ("body", rounded_bar_x(0.9 * hy, 0.85 * hz, 0.45 * hz, hx - 0.003, hx)),
        ("accent", box_mesh((0.3 * (x_cap1 - x_cap0), 0.002, 0.0006),
                            (x_cap0 + 0.55 * (x_cap1 - x_cap0), 0.0, rcap[1] + 0.0009))),
        ("accent", box_mesh((0.003, 0.002, 0.001), (x_cap1 - 0.005, 0.0, rcap[1] + 0.0003))),
    ]


def _glue_stick(hx, hy, hz):
    """A twist-up glue stick: the twist ring, the printed tube and a ribbed cap."""
    r = hy
    L = 2 * hx
    x_cap = hx - 0.36 * L
    return [
        ("accent", revolve_x(0.96 * r, r, -hx, -hx + 0.007, 32)),
        ("body", revolve_x(r, r, -hx + 0.007, x_cap, 32, v_range=(0.0, 0.64))),
        ("accent", revolve_x(1.04 * r, 1.04 * r, x_cap, hx - 0.003, 32)),
        ("accent", _ribs_x(1.04 * r, 1.07 * r, x_cap + 0.003, x_cap + 0.012, 3, 32)),
        ("accent", revolve_x(1.04 * r, 0.8 * r, hx - 0.003, hx, 32, cap0=False)),
    ]


def _tape_roll(hx, hy, hz):
    """A roll of tape on a 25 mm card core, lying flat."""
    r_in = _kinds()["tape_roll"].ring_inner
    core = r_in + 0.0015
    return [
        ("body", x_to_z(tube_x(core, hx, -hz, hz, 48))),
        ("card", x_to_z(tube_x(r_in, core, -hz - 0.0003, hz + 0.0003, 48))),
    ]


def _scissors(hx, hy, hz):
    """Scissors lying flat, closed: two tapered, pointed blades over each
    other, the pivot screw, and a small and a large handle loop on shanks.
    Designed at 14 cm (0.070, 0.022, 0.004) and stretched to the size."""
    blade_lo = prism([(-0.014, -0.0045), (0.040, -0.0035), (0.070, 0.0), (0.040, 0.0015), (-0.014, 0.0025)],
                     2, -0.0040, -0.0026)
    blade_hi = prism([(-0.014, -0.0025), (0.040, -0.0015), (0.068, 0.0), (0.040, 0.0035), (-0.014, 0.0045)],
                     2, -0.0026, -0.0012)
    pivot = x_to_z(revolve_x(0.0028, 0.0028, -0.0040, -0.0004, 16), (-0.008, 0.0, 0.0))
    loops = MeshBuilder()
    loops.extend(x_to_z(tube_x(0.0075, 0.0110, -0.004, 0.004, 32), (-0.0590, 0.0105, 0.0)))
    loops.extend(x_to_z(tube_x(0.0055, 0.0085, -0.004, 0.004, 28), (-0.0500, -0.0125, 0.0)))
    shanks = MeshBuilder()
    shanks.extend(prism([(-0.050, 0.004), (-0.012, 0.0005), (-0.012, 0.0045), (-0.048, 0.010)], 2, -0.0035, 0.0005))
    shanks.extend(prism([(-0.044, -0.006), (-0.012, -0.0045), (-0.012, -0.0005), (-0.042, -0.0035)], 2, -0.0035, 0.0005))
    s = (hx / 0.070, hy / 0.022, hz / 0.004)
    return [
        ("metal", scaled(blade_lo, s)),
        ("metal", scaled(blade_hi, s)),
        ("graphite", scaled(pivot, s)),
        ("body", scaled(loops, s)),
        ("body", scaled(shanks, s)),
    ]


def _eraser(hx, hy, hz):
    """A rubber block with rounded corners in a printed card sleeve."""
    L = 2 * hx
    return [
        ("body", rounded_box((hx, hy, hz), 0.25 * hy)),
        ("accent", rounded_box((0.27 * L, 1.02 * hy, 1.03 * hz), 0.25 * hy, (0.1 * L, 0.0, 0.0))),
    ]


def _sticky_notes(hx, hy, hz):
    """A pad of sticky notes: the stack, its glued edge, and the top sheet
    lifting a little at its free edge."""
    t = 0.0006
    curl = [(-hx, hz - t), (hx * 0.6, hz - t), (hx, hz - 0.0002), (-hx, hz - 0.0002)]
    return [
        ("body", rounded_box((hx, hy, hz - t), 0.0015, (0.0, 0.0, -t))),
        ("accent", prism([(x, y) for x, y in curl], 1, -0.998 * hy, 0.998 * hy)),
        ("card", box_mesh((0.004, 0.999 * hy, hz - t - 0.0002), (-hx + 0.004, 0.0, -t + 0.0002))),
    ]


def _stapler(hx, hy, hz):
    """A desk stapler: base with a rubber foot, the magazine, the hinge, and
    a top arm curving down to its nose. Designed at 13 cm and stretched."""
    arm = prism([(-0.064, -0.004), (-0.052, 0.020), (0.030, 0.022), (0.060, 0.015), (0.063, 0.004), (0.063, -0.004)],
                1, -0.016, 0.016)
    s = (hx / 0.065, hy / 0.018, hz / 0.022)
    return [
        ("accent", scaled(rounded_box((0.063, 0.017, 0.0045), 0.008, (0.0, 0.0, -0.0165)), s)),
        ("graphite", scaled(box_mesh((0.054, 0.012, 0.0008), (0.002, 0.0, -0.0212)), s)),
        ("metal", scaled(box_mesh((0.050, 0.012, 0.0035), (0.010, 0.0, -0.0085)), s)),
        ("accent", scaled(box_mesh((0.010, 0.0165, 0.010), (-0.054, 0.0, -0.004)), s)),
        ("body", scaled(arm, s)),
    ]


def _ruler(hx, hy, hz):
    """A ruler: one bevelled edge, and cm and half-cm ticks printed along it."""
    edge = [(-hy, -hz), (hy, -hz), (hy, -hz + 0.0007), (0.45 * hy, hz), (-hy, hz)]
    out = [("body", prism(edge, 0, -hx, hx))]
    ticks = MeshBuilder()
    n_cm = int(round(2 * hx * 100)) - 1
    for k in range(2 * n_cm + 1):
        x = -hx + 0.005 + 0.005 * k
        if x > hx - 0.004:
            break
        length = 0.006 if k % 2 == 0 else 0.0035
        ticks.extend(box_mesh((0.00025, length / 2, 0.00012), (x, 0.45 * hy - 0.001 - length / 2, hz + 0.00012)))
    out.append(("accent", ticks))
    return out


def _paper_clip_box(hx, hy, hz):
    """A small cardboard box of paper clips with a lift-off lid."""
    lid = 0.3 * hz
    return [
        ("body", rounded_box((0.985 * hx, 0.985 * hy, hz - 0.5 * lid), 0.0015, (0.0, 0.0, -0.5 * lid))),
        ("accent", rounded_box((hx, hy, 0.5 * lid), 0.0015, (0.0, 0.0, hz - 0.5 * lid))),
    ]


def _sharpener(hx, hy, hz):
    """A pencil sharpener: a wedge-shaped body, the blade plate on its
    sloping top, and the dark hole at its front."""
    body = prism([(-hx, -hz), (hx, -hz), (hx, 0.55 * hz), (-0.3 * hx, hz), (-hx, hz)], 1, -hy, hy)
    slope = math.atan2(hz - 0.55 * hz, 1.3 * hx)
    plate = box_mesh((0.5 * hx, 0.8 * hy, 0.0006))
    R = np.array([[math.cos(slope), 0, -math.sin(slope)], [0, 1, 0], [math.sin(slope), 0, math.cos(slope)]])
    return [
        ("body", body),
        ("metal", MeshBuilder().extend(plate, (R, np.array([0.35 * hx, 0.0, 0.78 * hz + 0.0006])))),
        ("graphite", revolve_x(0.0045 * hy / 0.010, 0.0045 * hy / 0.010, hx, hx + 0.0003, 20)),
    ]


_ITEMS = {
    "pen": _pen,
    "pencil": _pencil,
    "marker": _marker,
    "highlighter": _highlighter,
    "glue_stick": _glue_stick,
    "tape_roll": _tape_roll,
    "scissors": _scissors,
    "eraser": _eraser,
    "sticky_notes": _sticky_notes,
    "stapler": _stapler,
    "ruler": _ruler,
    "paper_clip_box": _paper_clip_box,
    "sharpener": _sharpener,
}


def station_tube(r_in: float, r_out: float, half_h: float, center=(0.0, 0.0, 0.0)) -> MeshBuilder:
    """An upright hollow cup wall (a station's `tube` part)."""
    c = np.asarray(center, float)
    return x_to_z(tube_x(r_in, r_out, -half_h, half_h, 48), c)


