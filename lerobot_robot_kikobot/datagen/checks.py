"""What makes an episode perfect, and the bookkeeping to tell.

An episode is kept only if all of these hold:

  * the target ends inside its station (cup, peg, tray or tray compartment),
    at rest, after a settle with the arm away;
  * the arm never touched a station, another item, or the table (the
    fingertips may brush the table while grasping, nothing more);
  * the grasp held from lift to release (no slip, no drop);
  * no other item on the desk moved more than DISTURB_M;
  * the planned motion was within IK tolerance at every tick (expert.py).

`SceneIndex` sorts every geom of the scene into who owns it; `contacts`
lists who touches whom; `ContactRules` says which touches are allowed in
which phase; `station_check` judges the final pose of the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from ..desk import layout as L
from .geometry import aabb_half

# Geom owners.
ROBOT, FINGER, TABLE, STATION, ITEM, OTHER = range(6)
FINGER_BODIES = ("lm6uu_bearing", "lm6uu_bearing_2")

DISTURB_M = 0.015  # Largest move allowed for an item that is not the target.
REST_SPEED_M_S = 0.01
PENETRATION_M = 0.0002  # A contact deeper than this is a touch.
FINGER_TABLE_M = 0.003  # How deep the fingertips may press on the table while grasping.
GRASP_PHASES = ("descend", "close", "lift")
SLIP_M = 0.015  # Target moving this far in the jaws between lift and release: a lost grip.


class SceneIndex:
    """Who owns each geom: the arm, a finger, the table, a station, or a desk item."""

    def __init__(self, model: mujoco.MjModel, pool: list[L.PoolItem]):
        m = model
        self.cls = np.full(m.ngeom, OTHER, dtype=np.int32)
        self.ref = np.full(m.ngeom, -1, dtype=np.int32)

        def body(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

        base = body("base_plate")
        if base < 0:
            raise RuntimeError("The scene has no 'base_plate' body: is it the Kiko-Bot arm?")
        fingers = {body(n) for n in FINGER_BODIES}
        robot = set()
        for b in range(m.nbody):
            # Walk up to the world; the arm's bodies all hang under base_plate.
            p = b
            while p > 0 and p != base:
                p = m.body_parentid[p]
            if p == base:
                robot.add(b)
        item_body = {body(f"desk_{it.name}"): k for k, it in enumerate(pool)}
        stations = {s.name: i for i, s in enumerate(L.STATIONS)}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if b in fingers:
                self.cls[g] = FINGER
            elif b in robot:
                self.cls[g] = ROBOT
            elif b in item_body:
                self.cls[g], self.ref[g] = ITEM, item_body[b]
            elif name.startswith("room_"):
                self.cls[g] = TABLE
            elif name.startswith("desk_"):
                for sname, si in stations.items():
                    if name.startswith(f"desk_{sname}_"):
                        self.cls[g], self.ref[g] = STATION, si
                        break
        self.station_names = [s.name for s in L.STATIONS]


def refresh_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Contacts for the current qpos, whichever engine stepped it."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)


def contacts(index: SceneIndex, data: mujoco.MjData):
    """(class a, ref a, class b, ref b, distance) for every contact, a <= b by class."""
    out = []
    for c in data.contact[: data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        a = (int(index.cls[g1]), int(index.ref[g1]))
        b = (int(index.cls[g2]), int(index.ref[g2]))
        if a > b:
            a, b = b, a
        out.append((a[0], a[1], b[0], b[1], float(c.dist)))
    return out


@dataclass
class ContactRules:
    """Which touches end an episode, for a given target item."""

    index: SceneIndex
    target: int

    def violation(self, data: mujoco.MjData, phase: str, held: bool = False) -> str | None:
        for ca, ra, cb, rb, dist in contacts(self.index, data):
            if dist > -PENETRATION_M:
                continue
            arm_a = ca in (ROBOT, FINGER)
            arm_b = cb in (ROBOT, FINGER)
            if not (arm_a or arm_b or (held and (ra == self.target or rb == self.target))):
                continue
            # The arm and the target item: that is the grasp.
            if arm_a and cb == ITEM and rb == self.target:
                continue
            if arm_a and arm_b:
                continue  # The arm's own parts (the jaws on each other).
            if arm_a and cb == TABLE:
                if ca == FINGER and phase in GRASP_PHASES and dist > -FINGER_TABLE_M:
                    continue
                return f"arm_hit_table@{phase}"
            if arm_a and cb == STATION:
                return f"arm_hit_{self.index.station_names[rb]}@{phase}"
            if arm_a and cb == ITEM:
                return f"arm_touched_other_item@{phase}"
            # The carried target hitting something on its way (planning only).
            if held:
                other = (ca, ra) if not (ca == ITEM and ra == self.target) else (cb, rb)
                if other[0] == STATION:
                    return f"item_hit_{self.index.station_names[other[1]]}@{phase}"
                if other[0] == ITEM:
                    return f"item_hit_other_item@{phase}"
                if other[0] == TABLE:
                    return f"item_hit_table@{phase}"
        return None


def item_box_corners(pos: np.ndarray, rot: np.ndarray, half) -> np.ndarray:
    hx, hy, hz = half
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return pos + (signs * np.array([hx, hy, hz])) @ rot.T


def _station_frame(st: L.Station) -> tuple[np.ndarray, np.ndarray]:
    c, s = math.cos(st.yaw), math.sin(st.yaw)
    return np.array([st.xy[0], st.xy[1], 0.0]), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _part(st: L.Station, name: str) -> L.StationPart:
    return next(p for p in st.parts if p.name == name)


def cup_inner(st: L.Station) -> tuple[float, float]:
    """Inner radius and rim height of a cup station."""
    visual = _part(st, "visual")
    return visual.half[0], 2.0 * visual.half[2]


def tray_inner(st: L.Station) -> tuple[float, float, float, int]:
    """Inner half length, inner half width, wall height and compartment count of a tray."""
    wall = _part(st, "wall_px")
    t = wall.half[0]
    n = 1 + sum(1 for p in st.parts if p.name.startswith("divider_"))
    return st.half[0] - 2 * t, st.half[1] - 2 * t, 2.0 * wall.half[2], n


def compartment_center(st: L.Station, k: int) -> np.ndarray:
    """World position (z = 0) of the centre of compartment k of a tray."""
    hx, _, _, n = tray_inner(st)
    o, r = _station_frame(st)
    x = -st.half[0] + 2.0 * st.half[0] * (k + 0.5) / n
    return o + r @ np.array([x, 0.0, 0.0])


def station_check(task, pos: np.ndarray, rot: np.ndarray, half) -> str | None:
    """None if the target, at `pos`/`rot` with collider `half` extents, is where it belongs."""
    st = task.station_spec
    o, r = _station_frame(st)
    local = r.T @ (pos - o)
    if task.strategy == "cup_insert":
        r_in, rim = cup_inner(st)
        if math.hypot(local[0], local[1]) > r_in:
            return "not_in_cup"
        axis = rot[:, 0]
        # Both ends inside the cup's footprint, the low one below the rim.
        hx = half[0]
        ends = [pos + axis * hx, pos - axis * hx]
        low = min(ends, key=lambda e: e[2])
        if low[2] > rim * 0.5:
            return "not_down_in_cup"
        for e in ends:
            if e[2] < rim and math.hypot(*(r.T @ (e - o))[:2]) > r_in + 0.002:
                return "poking_through_cup"
        if abs(axis[2]) < math.sin(math.radians(45.0)):
            return "not_upright_in_cup"
        return None
    if task.strategy == "ring_drop":
        if math.hypot(local[0], local[1]) > 0.006:
            return "not_on_peg"
        if pos[2] > 0.025:
            return "not_down_on_peg"
        return None
    # tray_place
    hx, hy, wall_h, n = tray_inner(st)
    corners = (item_box_corners(pos, rot, half) - o) @ r
    if np.any(np.abs(corners[:, 1]) > hy + 0.001) or np.any(np.abs(corners[:, 0]) > hx + 0.001):
        return "not_in_tray"
    floor_top = 0.004
    if corners[:, 2].min() + o[2] > floor_top + 0.008:
        return "not_down_in_tray"
    if task.compartment is not None and n > 1:
        edges = np.linspace(-st.half[0], st.half[0], n + 1)
        lo, hi = edges[task.compartment], edges[task.compartment + 1]
        if corners[:, 0].min() < lo - 0.004 or corners[:, 0].max() > hi + 0.004:
            return "wrong_compartment"
    return None


def aabb_top(pos: np.ndarray, rot: np.ndarray, half) -> float:
    return float(pos[2] + aabb_half(rot, half)[2])
