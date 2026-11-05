"""The scripted expert: picks the task's item off the desk and puts it where it belongs.

It has privileged access to the sim: the item poses, the station geometry
and a kinematic copy of the scene to check its plans against. An episode
runs in two plans:

  pick   ready -> above the item -> straight down -> close -> lift
  place  (planned from where the item really sits in the jaws after the lift)
         -> up -> over the station, turned as the station needs -> down ->
         open -> back out -> ready

Every plan is a dense joint trajectory at CONTROL_HZ, checked before any
physics runs: within IK tolerance at every tick of the straight moves, no
jump between ticks, clear of the joint limits, and no forbidden contact at
any tick (the arm against a station, the table or another item; the carried
item against anything before it is let go). Long moves (to the item, to
the station, home) are minimum-jerk in joint space, which keeps the arm on
one IK branch and inside its joint limits; the short moves near the desk
are straight lines for the TCP. A pick or place that cannot be planned is
not tried: the desk is skipped instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..desk import layout as L
from ..kinematics import TCP_OFFSET_EOF, ArmKinematics
from . import checks as C
from .geometry import UP, aabb_half, axis_angle, free_pose, min_jerk, rot_angle, rot_between, rot_z, slerp, wrap
from .tasks import TaskSpec

CONTROL_HZ = 50.0  # Joint targets to the actuators.
CONTROL_DT = 1.0 / CONTROL_HZ

# The jaws (models/robot.xml): the TCP sits this far above the fingertips, they
# open to this width at 100 %, and each finger is this wide along the item.
FINGERTIP_BELOW_TCP_M = 0.0472
JAW_OPEN_M = 0.0536
FINGER_HALF_WIDTH_M = 0.0175
FINGER_THICK_M = 0.02  # Outside the jaw opening.

# Tolerances on the plan.
IK_POS_TOL_M = 0.0015
IK_ROT_TOL_RAD = math.radians(1.5)
JOINT_MARGIN_RAD = math.radians(2.0)
MAX_JOINT_STEP_RAD = 0.06  # Per control tick: 3 rad/s.
REACH_M = 0.40

# Speeds.
V_APPROACH = 0.05  # m/s: the last centimetres to the item or into the station
V_TRANSIT = 0.14
W_TRANSIT = math.radians(70.0)  # rad/s of TCP turn
JOINT_SPEED = 1.0  # rad/s, fastest joint in a joint-space move
MIN_SEGMENT_S = 0.5

GRASP_PITCHES_DEG = (0.0, 12.0, -12.0, 24.0, -24.0)
CUP_GRIP_FROM_END_M = 0.025  # Pen-like items are held no nearer one end than this.
CUP_JAWS_OVER_RIM_M = 0.04  # The TCP stays this far above a cup's rim (the fingers clear it).
CUP_TIP_Z = 0.025  # Where the item's low end is let go, above the desk (the cup floor is at 4 mm).
PREGRASP_UP_M = 0.07
LIFT_TCP_Z = 0.13
FINGERTIP_MIN_Z = 0.0004
JAW_GRIP_MIN_M = 0.012  # Fingertips at least this far below the item's top (or 60 % of its height).
PRE_PLACE_CLEAR_M = 0.012  # Lowest point of the carried item above the station, before the last move down.
RING_DROP_M = 0.006  # The tape roll is let go this far above the peg's post.
HUB_RADIUS_M = (0.24, 0.30)  # Where the carry rises to, from the arm's base.
TRANSIT_MIN_TCP_Z = 0.09
APPROACH_HOVER_Z = 0.19  # The TCP crosses over to an item this high: fingertips above the cups' rims.

MAX_GRASP_TRIES = 6
MAX_PLACE_TRIES = 4


class PlanError(RuntimeError):
    """The expert cannot do this desk."""


def _most_common(errors: list[str], default: str) -> str:
    if not errors:
        return default
    keys = [e.split(" (")[0] for e in errors]
    return max(set(keys), key=keys.count)


@dataclass
class Plan:
    q: list[np.ndarray] = field(default_factory=list)
    grip: list[float] = field(default_factory=list)
    phase: list[str] = field(default_factory=list)
    tcp_pos: list[np.ndarray] = field(default_factory=list)
    tcp_rot: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.q)

    def add(self, q, grip, phase, pos, rot) -> None:
        self.q.append(np.array(q, dtype=float))
        self.grip.append(float(grip))
        self.phase.append(phase)
        self.tcp_pos.append(np.array(pos, dtype=float))
        self.tcp_rot.append(np.array(rot, dtype=float))


def grip_pct_for(width_m: float) -> float:
    return float(np.clip(100.0 * width_m / JAW_OPEN_M, 0.0, 100.0))


def down_rot(close_axis: np.ndarray) -> np.ndarray:
    """TCP rotation with the jaws pointing straight down, closing along `close_axis` (horizontal)."""
    x = np.array([close_axis[0], close_axis[1], 0.0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(UP, x), UP])


def _horizontal(v: np.ndarray) -> np.ndarray:
    h = np.array([v[0], v[1], 0.0])
    n = np.linalg.norm(h)
    return h / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


class Planner:
    """IK and dense trajectories on a kinematic copy of the arm."""

    def __init__(self, kin: ArmKinematics):
        self.kin = kin
        self.q_ready = kin.ready_q()
        p, _ = kin.fk(self.q_ready)
        # J1 against the azimuth of the TCP, to seed IK toward far-off targets.
        self._az_ready = math.atan2(p[1], p[0])
        self.lower = kin.lower + JOINT_MARGIN_RAD
        self.upper = kin.upper - JOINT_MARGIN_RAD

    def error(self, q, pos, rot) -> tuple[float, float]:
        e_p, e_r = self.kin.pose_error(q, pos, rot)
        return float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))

    def in_limits(self, q) -> bool:
        return bool(np.all(q >= self.lower) and np.all(q <= self.upper))

    def ok(self, q, pos, rot) -> bool:
        ep, er = self.error(q, pos, rot)
        return ep <= IK_POS_TOL_M and er <= IK_ROT_TOL_RAD and self.in_limits(q)

    def solve(self, pos, rot, seed: np.ndarray | None = None) -> np.ndarray | None:
        """A joint solution for the TCP pose: straight on from `seed` if that works,
        else the one nearest `seed` from a spread of starting points."""
        if seed is not None:
            q = self.kin.solve(np.array(seed, dtype=float), pos, rot, iterations=150)
            if self.ok(q, pos, rot):
                return q
        seeds = []
        az = math.atan2(pos[1], pos[0])
        for dj1 in (0.0, math.radians(30), -math.radians(30)):
            for dj6 in (0.0, math.pi, -math.pi):
                s = self.q_ready.copy()
                # J1 turns against the azimuth (its axis points down).
                s[0] = wrap(s[0] - (az - self._az_ready) + dj1)
                s[5] = wrap(s[5] + dj6)
                seeds.append(self.kin.clip(s))
        best, best_cost = None, math.inf
        for s in seeds:
            q = self.kin.solve(s, pos, rot, iterations=100)
            ep, er = self.error(q, pos, rot)
            if ep > 0.03 or er > 0.3:
                continue  # Not converging toward it from here.
            if not self.ok(q, pos, rot):
                q = self.kin.solve(q, pos, rot, iterations=150)
                if not self.ok(q, pos, rot):
                    continue
            cost = 0.0 if seed is None else float(np.sum(np.abs(q - seed)))
            if cost < best_cost:
                best, best_cost = q, cost
        return best

    # ------------------------------------------------------------ segments

    def line(self, plan: Plan, q0: np.ndarray, pose0, pose1, grip0: float, grip1: float, phase: str,
             v_lin: float = V_TRANSIT, w_ang: float = W_TRANSIT, min_s: float = MIN_SEGMENT_S) -> np.ndarray:
        """A minimum-jerk straight line of the TCP (orientation slerped); returns the last joint target."""
        p0, r0 = pose0
        p1, r1 = pose1
        dist = float(np.linalg.norm(p1 - p0))
        duration = max(min_s, dist / v_lin, rot_angle(r0, r1) / w_ang)
        n = max(1, int(math.ceil(duration / CONTROL_DT)))
        q = q0.copy()
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            pos = p0 + s * (p1 - p0)
            rot = slerp(r0, r1, s)
            q_new = self.kin.solve(q, pos, rot, iterations=12)
            if not self.ok(q_new, pos, rot):
                q_new = self.kin.solve(q_new, pos, rot, iterations=80)
                if not self.ok(q_new, pos, rot):
                    ep, er = self.error(q_new, pos, rot)
                    raise PlanError(f"ik@{phase} ({ep * 1000:.1f} mm, {math.degrees(er):.1f} deg)")
            if np.max(np.abs(q_new - q)) > MAX_JOINT_STEP_RAD:
                raise PlanError(f"joint_jump@{phase}")
            q = q_new
            plan.add(q, grip0 + (grip1 - grip0) * s, phase, pos, rot)
        return q

    def joints(self, plan: Plan, q0: np.ndarray, q1: np.ndarray, grip0: float, grip1: float, phase: str,
               min_tcp_z: float = TRANSIT_MIN_TCP_Z, min_s: float = MIN_SEGMENT_S) -> np.ndarray:
        """A minimum-jerk move in joint space, the TCP kept above `min_tcp_z`."""
        if not (self.in_limits(q0) and self.in_limits(q1)):
            raise PlanError(f"joint_limit@{phase}")
        p0, _ = self.kin.fk(q0)
        p1, _ = self.kin.fk(q1)
        dq = q1 - q0
        duration = max(min_s, float(np.max(np.abs(dq))) / JOINT_SPEED, float(np.linalg.norm(p1 - p0)) / V_TRANSIT)
        n = max(1, int(math.ceil(duration / CONTROL_DT)))
        floor = min(min_tcp_z, p0[2] - 0.005, p1[2] - 0.005)
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            q = q0 + s * dq
            pos, rot = self.kin.fk(q)
            if pos[2] < floor:
                raise PlanError(f"too_low@{phase}")
            plan.add(q, grip0 + (grip1 - grip0) * s, phase, pos, rot)
        return q1.copy()

    def hold(self, plan: Plan, q: np.ndarray, grip0: float, grip1: float, seconds: float, phase: str) -> None:
        pos, rot = self.kin.fk(q)
        n = max(1, int(round(seconds / CONTROL_DT)))
        ramp = max(1, int(0.6 * n))
        for i in range(1, n + 1):
            plan.add(q, grip0 + (grip1 - grip0) * min_jerk(min(1.0, i / ramp)), phase, pos, rot)


# ------------------------------------------------------------- the expert


@dataclass
class Target:
    index: int  # In the desk pool.
    item: L.PoolItem
    pos: np.ndarray
    rot: np.ndarray


@dataclass
class PlaceOption:
    item_pos: np.ndarray  # Where the item's centre goes...
    item_rot: np.ndarray  # ...turned like this, at the moment it is let go.
    pre_up: float  # The TCP comes down this far onto it.
    retreat_dir: np.ndarray  # The open jaws back out this way.
    opening: float  # Jaw opening to let go, %.


class Expert:
    """Plans the task on the world's current desk."""

    def __init__(self, world, task: TaskSpec, kin: ArmKinematics | None = None):
        self.world = world
        self.task = task
        self.kin = kin or ArmKinematics()
        self.planner = Planner(self.kin)
        self.desk = world._desk
        self.index = C.SceneIndex(world.model, self.desk.pool)
        # A private copy of the scene for checking plans by kinematics.
        self._scratch = mujoco.MjData(world.model)
        self._arm_qpos = world._qpos_adr
        self._finger_qpos = world._finger_qpos
        self._grip_closed, self._grip_open = world._grip_closed, world._grip_open
        self._eof = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "eof")

    # -------------------------------------------------------------- state

    def item_pose(self, k: int, data: mujoco.MjData | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = data if data is not None else self.world.data
        a = self.desk.qpos_adr[k]
        return free_pose(d.qpos[a : a + 7])

    def tcp_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.kin.fk(q)

    def measured_hold(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The item's pose in the TCP frame, as it really sits in the jaws now."""
        q = self.world.data.qpos[self._arm_qpos].copy()
        tp, tr = self.tcp_pose(q)
        ip, ir = self.item_pose(k)
        return tr.T @ (ip - tp), tr.T @ ir

    def targets(self) -> list[Target]:
        """Items of the task's kind on the desk, within reach, with nothing on them, nearest first."""
        w = self.world
        C.refresh_contacts(w.model, w.data)
        cons = C.contacts(self.index, w.data)
        candidates = []
        for k, (it, pl) in enumerate(zip(self.desk.pool, self.desk.placements)):
            if it.kind != self.task.kind or not pl.active:
                continue
            pos, rot = self.item_pose(k)
            if math.hypot(pos[0], pos[1]) > REACH_M or pos[2] < -0.01 or pos[2] > 0.06:
                continue
            # Lying the way it was put down: long axis level and, for boxes, face down.
            if abs(rot[2, 0]) > 0.2:
                continue
            if it.spec.collider not in ("capsule", "cylinder_x") and abs(rot[2, 2]) < 0.9:
                continue
            buried = False
            for ca, ra, cb, rb, dist in cons:
                if ca == C.ITEM and cb == C.ITEM and k in (ra, rb) and dist < 0.001:
                    other = rb if ra == k else ra
                    if self.item_pose(other)[0][2] > pos[2] + 0.002:
                        buried = True
            if not buried:
                candidates.append(Target(k, it, pos, rot))
        if not candidates:
            raise PlanError("no_free_target")
        # The nearest to the arm's ready pose first: the most natural pick.
        ready_xy = self.kin.fk(self.planner.q_ready)[0][:2]
        candidates.sort(key=lambda t: float(np.linalg.norm(t.pos[:2] - ready_xy)))
        return candidates

    # ---------------------------------------------------------- the grasp

    def grasp_candidates(self, t: Target) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """(TCP position, TCP rotation, jaw opening in %) for grasping `t` from above."""
        it = t.item
        half = np.asarray(it.half)
        u = _horizontal(t.rot[:, 0])
        perp = np.cross(UP, u)
        top = C.aabb_top(t.pos, t.rot, half)
        bottom = t.pos[2] - aabb_half(t.rot, half)[2]
        height = top - bottom
        # Fingertips a little below the item's middle (no deeper than 2 cm), never into the desk.
        tip_z = max(min(bottom + 0.5 * height - 0.004, bottom + 0.6 * height), top - 0.02, FINGERTIP_MIN_Z)
        deepest_tip = top - min(0.6 * height, JAW_GRIP_MIN_M)  # Grip at least this much of it.
        width = 2.0 * float(np.abs(perp @ t.rot) @ half)
        opening = grip_pct_for(width + 0.016)
        if it.kind == "tape_roll":
            # Round: any closing direction will do.
            closings = [rot_z(a) @ np.array([1.0, 0.0, 0.0]) for a in np.linspace(0.0, math.pi, 6, endpoint=False)]
            offsets = [0.0]
            opening = 100.0
        else:
            closings = [perp, -perp]
            offsets = [0.0]
            if self.task.strategy == "cup_insert":
                # Toward one end, so enough of it hangs below the jaws, upright, to reach
                # down into the cup while the jaws stay above the rim.
                _, rim = C.cup_inner(self.task.station_spec)
                g = rim + CUP_JAWS_OVER_RIM_M - CUP_TIP_Z - half[0]
                g = min(max(g, 0.0), half[0] - CUP_GRIP_FROM_END_M)
                offsets = [g, -g] if g > 0 else [0.0]
            elif it.kind == "scissors":
                offsets = [0.012]  # The blades, just in front of the pivot.
        half_w = 0.5 * JAW_OPEN_M * opening / 100.0 + FINGER_THICK_M
        out = []
        for g in offsets:
            c = t.pos[:2] + g * u[:2]
            for closing in closings:
                for pitch in GRASP_PITCHES_DEG:
                    # Pitched about the closing axis: the jaws lean along the item, for
                    # reach near and far without the fingers digging into the desk.
                    rot = axis_angle(closing, math.radians(pitch)) @ down_rot(closing)
                    # The fingertips' centre over the grasp point, at tip_z...
                    tcp = np.array([c[0], c[1], tip_z]) + FINGERTIP_BELOW_TCP_M * rot[:, 2]
                    # ...raised if a finger's corner would go into the desk.
                    low = min((rot @ np.array([sx * half_w, sy * FINGER_HALF_WIDTH_M, -FINGERTIP_BELOW_TCP_M]))[2]
                              for sx in (-1, 1) for sy in (-1, 1))
                    if tcp[2] + low < FINGERTIP_MIN_Z:
                        tcp[2] += FINGERTIP_MIN_Z - (tcp[2] + low)
                    if tcp[2] - FINGERTIP_BELOW_TCP_M * rot[2, 2] > deepest_tip + 1e-4:
                        continue  # Too shallow to hold it.
                    out.append((tcp, rot, opening))
        return out

    # ----------------------------------------------------- checks by kinematics

    def _set_arm(self, d: mujoco.MjData, q: np.ndarray, grip_pct: float) -> None:
        d.qpos[self._arm_qpos] = q
        g = self._grip_closed + float(np.clip(grip_pct, 0, 100)) / 100.0 * (self._grip_open - self._grip_closed)
        d.qpos[self._finger_qpos] = g

    def check_plan(self, plan: Plan, target: Target, held: tuple[np.ndarray, np.ndarray] | None = None,
                   start: int = 0, every: int = 2) -> None:
        """Forbidden contacts anywhere along `plan`, by kinematics. `held`: the item's
        pose in the TCP frame while it is carried (from `start` until it is let go)."""
        m = self.world.model
        d = self._scratch
        d.qpos[:] = self.world.data.qpos
        rules = C.ContactRules(self.index, target.index)
        a = self.desk.qpos_adr[target.index]
        carrying = held is not None
        for i in range(start, len(plan), every):
            ph = plan.phase[i]
            if ph in ("release", "retreat", "return", "rest"):
                carrying = False
            self._set_arm(d, plan.q[i], plan.grip[i])
            if carrying:
                mujoco.mj_kinematics(m, d)
                rot = d.xmat[self._eof].reshape(3, 3)
                tp = d.xpos[self._eof] + rot @ TCP_OFFSET_EOF
                p_rel, r_rel = held
                d.qpos[a : a + 3] = tp + rot @ p_rel
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (rot @ r_rel).reshape(9))
                d.qpos[a + 3 : a + 7] = quat
            C.refresh_contacts(m, d)
            # The carried item may touch its station only on the last move into it.
            bad = rules.violation(d, ph, held=carrying and ph != "place")
            if bad:
                raise PlanError(f"plan_{bad}")

    # ------------------------------------------------------------ planning

    def plan_pick(self, q_start: np.ndarray) -> tuple[Plan, Target, np.ndarray]:
        """Ready -> above the item -> down -> close -> lift. Raises PlanError.

        A grasp is only taken if the place after it can be planned too
        (assuming the item stays in the jaws as it lay)."""
        errors: list[str] = []
        for t in self.targets():
            try:
                plan, q_end = self._pick_target(q_start, t, errors)
                return plan, t, q_end
            except PlanError as exc:
                errors.append(str(exc))
        raise PlanError(_most_common(errors, "no_grasp"))

    def _pick_target(self, q_start, t: Target, errors: list[str]) -> tuple[Plan, np.ndarray]:
        options = []
        for pos, rot, opening in self.grasp_candidates(t):
            q_grasp = self.planner.solve(pos, rot, seed=q_start)
            if q_grasp is None:
                errors.append("ik_grasp")
                continue
            options.append((float(np.sum(np.abs(q_grasp - q_start))), pos, rot, opening, q_grasp))
        options.sort(key=lambda o: o[0])
        for _, pos, rot, opening, q_grasp in options[:MAX_GRASP_TRIES]:
            try:
                plan, q_end = self._pick_plan(q_start, t, pos, rot, opening, q_grasp)
                held = (rot.T @ (t.pos - pos), rot.T @ t.rot)
                self.plan_place(q_end, t, held)
                return plan, q_end
            except PlanError as exc:
                errors.append(str(exc))
        raise PlanError(_most_common(errors, "no_grasp"))

    def _pick_plan(self, q_start, t: Target, pos, rot, opening, q_grasp) -> tuple[Plan, np.ndarray]:
        P = self.planner
        plan = Plan()
        # Back along the jaws' axis, so the last stretch is straight in.
        pre = pos + PREGRASP_UP_M * rot[:, 2]
        q_pre = P.solve(pre, rot, seed=q_grasp)
        if q_pre is None:
            raise PlanError("ik_pregrasp")
        # Over the top: up to a hover high enough to clear the cups, across,
        # then straight down onto the pregrasp. A direct joint-space sweep
        # from the ready pose passes low over whatever stands in between.
        # Where that cannot be followed smoothly, the direct sweep instead.
        hover = np.array([pre[0], pre[1], max(pre[2], APPROACH_HOVER_Z)])
        q_hover = P.solve(hover, rot, seed=q_pre) if hover[2] > pre[2] + 1e-4 else None
        q = None
        if q_hover is not None:
            try:
                q = P.joints(plan, q_start, q_hover, 100.0, opening, "approach",
                             min_tcp_z=min(hover[2], self.tcp_pose(q_start)[0][2]) - 0.01)
                q = P.line(plan, q, (hover, rot), (pre, rot), opening, opening, "approach", v_lin=V_TRANSIT * 0.7)
            except PlanError:
                plan, q = Plan(), None
        if q is None:
            q = P.joints(plan, q_start, q_pre, 100.0, opening, "approach")
        q = P.line(plan, q, (pre, rot), (pos, rot), opening, opening, "descend", v_lin=V_APPROACH)
        P.hold(plan, q, opening, 0.0, 0.7, "close")
        lift = np.array([pos[0], pos[1], max(LIFT_TCP_Z, pos[2] + 0.06)])
        n_before_lift = len(plan)
        q = P.line(plan, q, (pos, rot), (lift, rot), 0.0, 0.0, "lift", v_lin=V_APPROACH)
        self.check_plan(plan, t)
        held = (rot.T @ (t.pos - pos), rot.T @ t.rot)
        self.check_plan(plan, t, held=held, start=n_before_lift + 5)
        return plan, q

    def plan_place(self, q_start: np.ndarray, t: Target, held) -> tuple[Plan, PlaceOption]:
        """From the lifted grasp to the station and back to ready. `held`: the item's
        pose in the TCP frame (p_rel, r_rel). Raises PlanError."""
        P = self.planner
        tcp0 = self.tcp_pose(q_start)
        p_rel, r_rel = held
        item_rot0 = tcp0[1] @ r_rel
        grip_in_item = r_rel.T @ (-p_rel)  # Where the jaws hold it, in the item's frame.
        errors: list[str] = []
        feasible = []
        for opt in self._place_candidates(t, item_rot0, grip_in_item):
            # The TCP pose that puts the item there, sitting in the jaws as it does.
            r_tcp = opt.item_rot @ r_rel.T
            p_tcp = opt.item_pos - r_tcp @ p_rel
            q_place = P.solve(p_tcp, r_tcp, seed=q_start)
            if q_place is None:
                errors.append("ik_place")
                continue
            q_pre = P.solve(p_tcp + np.array([0.0, 0.0, opt.pre_up]), r_tcp, seed=q_place)
            if q_pre is None:
                errors.append("ik_preplace")
                continue
            feasible.append((float(np.sum(np.abs(q_pre - q_start))), opt, p_tcp, r_tcp, q_pre))
        feasible.sort(key=lambda f: f[0])
        for _, opt, p_tcp, r_tcp, q_pre in feasible[:MAX_PLACE_TRIES]:
            try:
                plan = self._place_plan(q_start, tcp0, opt, p_tcp, r_tcp, q_pre)
                self.check_plan(plan, t, held=held)
                return plan, opt
            except PlanError as exc:
                errors.append(str(exc))
        raise PlanError(_most_common(errors, "no_place"))

    def _place_candidates(self, t: Target, item_rot0: np.ndarray, grip_in_item: np.ndarray):
        task, st = self.task, self.task.station_spec
        half = np.asarray(t.item.half)
        x0 = item_rot0[:, 0]
        # Flat items go down level, however they hang in the jaws: the wrist makes up for it.
        level = rot_between(item_rot0[:, 2] if item_rot0[2, 2] >= 0 else -item_rot0[:, 2], UP) @ item_rot0
        if task.strategy == "cup_insert":
            r_in, rim = C.cup_inner(st)
            c = np.array([st.xy[0], st.xy[1]])
            radius = max(half[1], half[2])
            opening = grip_pct_for(2.0 * radius + 0.025)
            # The end away from the jaws goes down: the -X end when held at +X.
            s_low = -1.0 if grip_in_item[0] > 0 else 1.0
            low_h = _horizontal(s_low * x0)  # Where the low end points now, flat.
            tip_z = CUP_TIP_Z
            for tilt_deg in (90.0, 80.0, 70.0, 60.0):
                th = math.radians(tilt_deg)
                drift = (rim - tip_z) / math.tan(th) if tilt_deg < 90.0 else 0.0
                if 0.5 * drift + radius > r_in - 0.008:
                    continue
                low0 = math.cos(th) * low_h - math.sin(th) * UP
                tilted = rot_between(s_low * x0, low0) @ item_rot0
                for psi in np.linspace(-math.pi, math.pi, 24, endpoint=False):
                    spin = rot_z(psi)
                    low_dir = spin @ low0
                    horiz = _horizontal(low_dir)
                    # It leans back toward -horiz as it rises: centre the lean on the cup.
                    tip = np.array([c[0] + 0.5 * drift * horiz[0], c[1] + 0.5 * drift * horiz[1], tip_z])
                    yield PlaceOption(tip - low_dir * half[0], spin @ tilted, rim + PRE_PLACE_CLEAR_M - tip_z,
                                      -low_dir, opening)
        elif task.strategy == "ring_drop":
            post_top = 0.068
            centre = np.array([st.xy[0], st.xy[1], post_top + RING_DROP_M + half[2]])
            for dyaw in np.linspace(-math.pi, math.pi, 12, endpoint=False):
                yield PlaceOption(centre, rot_z(dyaw) @ level, 0.05, UP, 100.0)
        else:
            _, _, wall_h, _ = C.tray_inner(st)
            if task.compartment is not None:
                centre_xy = C.compartment_center(st, task.compartment)[:2]
            else:
                centre_xy = np.array(st.xy)
            # Across a compartment (it is short), else along the tray.
            along = st.yaw + (math.pi / 2 if task.compartment is not None else 0.0)
            xl = level[:, 0]
            yaw_now = math.atan2(xl[1], xl[0])
            centre_z = wall_h + 0.008 + aabb_half(level, half)[2]
            for flip in (0.0, math.pi):
                r_des = rot_z(wrap(along + flip - yaw_now)) @ level
                yield PlaceOption(np.array([centre_xy[0], centre_xy[1], centre_z]), r_des, 0.05, UP, 100.0)

    def _place_plan(self, q_start, tcp0, opt: PlaceOption, p_place, r_place, q_pre) -> Plan:
        plan = self._carry_plan(q_start, tcp0, opt, p_place, q_pre)
        self._final_plan(plan, q_pre, opt, p_place, r_place)
        return plan

    def _carry_plan(self, q_start, tcp0, opt: PlaceOption, p_place, q_pre) -> Plan:
        """Lifted grasp -> up to a comfortable reach -> over the station (pre-place)."""
        P = self.planner
        plan = Plan()
        p0 = tcp0[0]
        rad = math.hypot(p0[0], p0[1])
        hub_rad = min(max(rad, HUB_RADIUS_M[0]), HUB_RADIUS_M[1])
        hub_xy = p0[:2] * (hub_rad / rad) if rad > 1e-6 else p0[:2]
        pre = p_place + np.array([0.0, 0.0, opt.pre_up])
        hub = np.array([hub_xy[0], hub_xy[1], max(pre[2], p0[2] + 0.04)])
        # Up first, so the item swings round high above the desk.
        q = P.line(plan, q_start, tcp0, (hub, tcp0[1]), 0.0, 0.0, "carry_up", v_lin=V_TRANSIT * 0.7)
        P.joints(plan, q, q_pre, 0.0, 0.0, "carry", min_tcp_z=min(pre[2], hub[2]) - 0.01)
        return plan

    def _final_plan(self, plan: Plan, q_now, opt: PlaceOption, p_place, r_place) -> None:
        """From wherever the TCP is over the station: onto the pre-place pose for
        (p_place, r_place), down, let go, back out, home."""
        P = self.planner
        here = self.tcp_pose(q_now)
        pre = p_place + np.array([0.0, 0.0, opt.pre_up])
        q = q_now
        if np.linalg.norm(here[0] - pre) > 1e-4 or rot_angle(here[1], r_place) > 1e-3:
            q = P.line(plan, q, here, (pre, r_place), 0.0, 0.0, "align", v_lin=V_APPROACH, min_s=0.3)
        q = P.line(plan, q, (pre, r_place), (p_place, r_place), 0.0, 0.0, "place", v_lin=V_APPROACH)
        P.hold(plan, q, 0.0, opt.opening, 0.6, "release")
        out = p_place + 0.07 * opt.retreat_dir
        q = P.line(plan, q, (p_place, r_place), (out, r_place), opt.opening, opt.opening, "retreat",
                   v_lin=V_APPROACH)
        if out[2] < 0.18:
            high = np.array([out[0], out[1], 0.18])
            q = P.line(plan, q, (out, r_place), (high, r_place), opt.opening, opt.opening, "retreat")
        q = P.joints(plan, q, P.q_ready, opt.opening, 100.0, "return")
        P.hold(plan, q, 100.0, 100.0, 0.5, "rest")

    def replan_final(self, q_now: np.ndarray, t: Target, opt: PlaceOption, held) -> Plan:
        """The last stretch again, aimed with where the item sits in the jaws now."""
        p_rel, r_rel = held
        r_tcp = opt.item_rot @ r_rel.T
        p_tcp = opt.item_pos - r_tcp @ p_rel
        plan = Plan()
        self._final_plan(plan, q_now, opt, p_tcp, r_tcp)
        self.check_plan(plan, t, held=held)
        return plan
