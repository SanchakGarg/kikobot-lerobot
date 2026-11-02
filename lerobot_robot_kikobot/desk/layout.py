"""The desk sorting station: what is on the desk each episode, where, and how it looks.

Pure numpy, no Isaac Sim: the server samples a layout here and the backend
applies it. Tested in tests/test_desk_layout.py.

Frames: the arm's base is at the origin on the tabletop (z = 0), the record
camera looks back at it from +X. Every item's local frame has its long axis
along X and its resting face down, so a placement is just (x, y, z, yaw).

Pens, pencils, markers and highlighters may land on top of each other: a
stacked one is dropped from just above whatever it crosses and settles during
the reset's settle phase. Everything else gets a clear patch of desk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

# ------------------------------------------------------------------ the items


@dataclass(frozen=True)
class KindSpec:
    """Default size of one kind of item, lying on the desk.

    `half` are the collider's half extents in the item frame (X = long axis);
    `collider` is capsule | cylinder_x | cylinder_z | box | scissors. The
    layout only needs the footprint (a 2D capsule: segment half length along X
    plus radius) and `half[2]`, the height of the item's centre above the desk.
    """

    collider: str
    half: tuple[float, float, float]
    mass: float
    friction: float
    writing: bool = False  # pen-like: may be stacked, goes in the pen cup

    def footprint(self) -> tuple[float, float]:
        hx, hy, _ = self.half
        if self.collider in ("capsule", "cylinder_x", "cylinder_z"):
            return max(hx - hy, 0.0), hy
        # Boxes: the capsule that covers the corners.
        r = math.hypot(hy, hy)
        return max(hx - hy, 0.0), r


KINDS: dict[str, KindSpec] = {
    # Ballpoint: 14 cm, 10 mm barrel.
    "pen": KindSpec("capsule", (0.070, 0.005, 0.005), 0.008, 0.8, writing=True),
    # Hex pencil, 17.5 cm, 7.5 mm across the flats (box collider: no rolling).
    "pencil": KindSpec("box", (0.0875, 0.00375, 0.00375), 0.006, 0.9, writing=True),
    # Felt-tip marker, 13 cm, 17 mm.
    "marker": KindSpec("capsule", (0.065, 0.0085, 0.0085), 0.015, 0.8, writing=True),
    # Flat highlighter, 12 cm x 22 x 14 mm.
    "highlighter": KindSpec("box", (0.060, 0.011, 0.007), 0.014, 0.8, writing=True),
    # Glue stick lying down, 8.5 cm, 22 mm.
    "glue_stick": KindSpec("cylinder_x", (0.0425, 0.011, 0.011), 0.025, 1.0),
    # Tape roll lying flat: 44 mm across (fits the open gripper), 13 mm wide.
    "tape_roll": KindSpec("cylinder_z", (0.022, 0.022, 0.0065), 0.030, 1.0),
    # Small scissors lying flat: blades + handle loops, 14 cm.
    "scissors": KindSpec("scissors", (0.070, 0.022, 0.004), 0.040, 0.6),
    # Block eraser, 44 x 22 x 12 mm.
    "eraser": KindSpec("box", (0.022, 0.011, 0.006), 0.010, 1.2),
}

WRITING_KINDS = tuple(k for k, s in KINDS.items() if s.writing)
SUPPLY_KINDS = tuple(k for k, s in KINDS.items() if not s.writing)


@dataclass(frozen=True)
class PoolItem:
    """One item that exists in the scene. Built once; shown or parked per episode."""

    name: str
    kind: str
    # Collider half extents. Defaults to KINDS[kind].half; the backend
    # replaces it with the real size when it loads a Nucleus asset.
    half: tuple[float, float, float]

    @property
    def spec(self) -> KindSpec:
        return KINDS[self.kind]

    def footprint(self) -> tuple[float, float]:
        return replace(self.spec, half=self.half).footprint()


# How many of each kind exist. Each episode shows a random subset.
DEFAULT_COUNTS: dict[str, int] = {
    "pen": 5,
    "pencil": 2,
    "marker": 2,
    "highlighter": 1,
    "glue_stick": 1,
    "tape_roll": 1,
    "scissors": 1,
    "eraser": 1,
}


def default_pool(counts: dict[str, int] = DEFAULT_COUNTS) -> list[PoolItem]:
    return [
        PoolItem(f"{kind}_{i}", kind, KINDS[kind].half)
        for kind, n in counts.items()
        for i in range(n)
    ]


# ------------------------------------------------------------- the stations


@dataclass(frozen=True)
class Station:
    """A fixed receptacle on the desk. `half` is its footprint, `yaw` its heading."""

    name: str
    xy: tuple[float, float]
    yaw: float
    half: tuple[float, float]


def _polar(r: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return (r * math.cos(a), r * math.sin(a))


# Pen cup on the arm's right, compartment tray on its left, both inside reach
# and outside the clutter fan in front of the arm.
PEN_CUP = Station("pen_cup", _polar(0.24, -60.0), 0.0, (0.038, 0.038))
CUP_HEIGHT = 0.095
TRAY = Station("tray", _polar(0.25, 62.0), math.radians(62.0 + 90.0), (0.095, 0.060))
TRAY_WALL_HEIGHT = 0.03
STATIONS = (PEN_CUP, TRAY)

# Clutter fan in front of the arm (item centres), and the base to keep clear.
CLUTTER_RADIUS_M = (0.17, 0.33)
CLUTTER_HALF_ANGLE_DEG = 42.0
BASE_KEEPOUT_M = 0.10
CLEARANCE_M = 0.006  # between items that may not touch
STACK_GAP_M = 0.004  # drop height above whatever a stacked pen crosses
MAX_STACK = 3  # layers in one pile

# Parking: inactive items lie on the floor under the table, out of view.
FLOOR_Z = -0.4
PARK_X = (-0.42, -0.05)
PARK_Y = (-0.42, 0.42)


# ----------------------------------------------------------- the appearance

# Linear RGB, 0..1.
BODY_COLORS = {
    "white": (0.85, 0.85, 0.83),
    "black": (0.03, 0.03, 0.035),
    "navy": (0.04, 0.07, 0.25),
    "royal_blue": (0.05, 0.20, 0.70),
    "sky": (0.35, 0.60, 0.85),
    "red": (0.70, 0.05, 0.05),
    "orange": (0.90, 0.35, 0.05),
    "yellow": (0.95, 0.75, 0.08),
    "green": (0.08, 0.45, 0.15),
    "teal": (0.05, 0.45, 0.45),
    "purple": (0.35, 0.10, 0.55),
    "pink": (0.90, 0.40, 0.60),
    "silver": (0.60, 0.62, 0.65),
    "gold": (0.75, 0.55, 0.20),
    "smoke": (0.25, 0.25, 0.28),
}
INK_COLORS = {
    "blue": (0.05, 0.15, 0.75),
    "black": (0.02, 0.02, 0.02),
    "red": (0.80, 0.05, 0.05),
    "green": (0.05, 0.55, 0.15),
    "purple": (0.40, 0.10, 0.65),
}
NEON_COLORS = {
    "neon_yellow": (0.95, 0.95, 0.10),
    "neon_pink": (0.98, 0.30, 0.65),
    "neon_green": (0.40, 0.95, 0.20),
    "neon_orange": (0.98, 0.50, 0.10),
    "neon_blue": (0.20, 0.75, 0.98),
}
PRINT_COLORS = {"white": (0.90, 0.90, 0.90), "black": (0.02, 0.02, 0.02), "gold": (0.80, 0.60, 0.20)}

# Grayscale patterns from desk_textures.py; the shader tints them between the
# body colour (pattern = 1) and the print colour (pattern = 0).
TEXTURES_BY_KIND = {
    "pen": ("solid", "stripes", "grip_dots", "bands", "brand"),
    "pencil": ("solid", "stripes", "brand", "bands"),
    "marker": ("solid", "brand", "bands"),
    "highlighter": ("solid", "bands", "brand"),
    "glue_stick": ("brand", "bands", "solid"),
    "tape_roll": ("solid", "stripes", "dots", "checker"),
    "scissors": ("solid", "speckle"),
    "eraser": ("solid", "brand", "stripes", "speckle"),
}

BODY_PALETTE_BY_KIND = {
    "pen": ("white", "black", "navy", "royal_blue", "sky", "red", "green", "purple", "pink", "silver", "gold", "smoke"),
    "pencil": ("yellow", "red", "green", "black", "royal_blue", "pink", "silver"),
    "marker": ("white", "black", "silver", "smoke"),
    "highlighter": tuple(NEON_COLORS),
    "glue_stick": ("white", "purple", "yellow", "red", "royal_blue"),
    "tape_roll": ("white", "sky", "pink", "yellow", "green", "smoke", "orange"),
    "scissors": ("red", "orange", "royal_blue", "black", "green", "purple", "pink"),
    "eraser": ("pink", "white", "sky", "green"),
}


@dataclass
class Placement:
    """One pool item in one episode."""

    name: str
    kind: str
    active: bool
    pos: tuple[float, float, float]
    yaw: float
    texture: str = "solid"
    body_rgb: tuple[float, float, float] = (0.8, 0.8, 0.8)
    print_rgb: tuple[float, float, float] = (0.02, 0.02, 0.02)
    accent_rgb: tuple[float, float, float] = (0.05, 0.15, 0.75)  # cap, ink, tip
    roughness: float = 0.5
    metallic: float = 0.0
    layer: int = 0  # 0 on the desk, 1+ stacked on other pens
    on_top_of: list[str] = field(default_factory=list)


def _all_colors() -> dict[str, tuple[float, float, float]]:
    return {**BODY_COLORS, **NEON_COLORS}


def sample_look(rng: np.random.Generator, kind: str) -> dict:
    colors = _all_colors()
    body_name = str(rng.choice(BODY_PALETTE_BY_KIND[kind]))
    body = colors[body_name]
    ink_name = str(rng.choice(list(INK_COLORS)))
    if kind == "highlighter":
        accent = body  # cap matches the ink
    elif kind in ("pen", "marker"):
        accent = INK_COLORS[ink_name]
    elif kind == "pencil":
        accent = (0.05, 0.05, 0.05) if rng.random() < 0.5 else BODY_COLORS["pink"]  # eraser end or plain
    else:
        accent = colors[str(rng.choice(list(BODY_COLORS)))]
    # Print in whichever of white/black contrasts the body; gold now and then.
    lum = 0.2126 * body[0] + 0.7152 * body[1] + 0.0722 * body[2]
    print_name = "gold" if rng.random() < 0.15 else ("black" if lum > 0.35 else "white")
    metallic = 0.0
    if body_name in ("silver", "gold") and rng.random() < 0.7:
        metallic = float(rng.uniform(0.6, 1.0))
    return dict(
        texture=str(rng.choice(TEXTURES_BY_KIND[kind])),
        body_rgb=tuple(float(c) for c in body),
        print_rgb=PRINT_COLORS[print_name],
        accent_rgb=tuple(float(c) for c in accent),
        roughness=float(rng.uniform(0.25, 0.75)) if metallic == 0.0 else float(rng.uniform(0.15, 0.4)),
        metallic=metallic,
    )


# -------------------------------------------------------------- the geometry


def _segment(xy, yaw: float, half_len: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.array([math.cos(yaw), math.sin(yaw)]) * half_len
    c = np.asarray(xy, dtype=float)
    return c - d, c + d


def segment_distance(a0, a1, b0, b1) -> float:
    """Shortest distance between 2D segments a0-a1 and b0-b1."""
    # Plain floats: this runs thousands of times per layout.
    ax, ay = float(a0[0]), float(a0[1])
    bx, by = float(b0[0]), float(b0[1])
    dax, day = float(a1[0]) - ax, float(a1[1]) - ay
    dbx, dby = float(b1[0]) - bx, float(b1[1]) - by

    def point_seg(px, py, sx, sy, dx, dy):
        n = dx * dx + dy * dy
        t = 0.0 if n == 0.0 else min(max(((px - sx) * dx + (py - sy) * dy) / n, 0.0), 1.0)
        return math.hypot(px - sx - t * dx, py - sy - t * dy)

    # Proper intersection: distance 0.
    den = dax * dby - day * dbx
    if den != 0.0:
        ex, ey = bx - ax, by - ay
        t = (ex * dby - ey * dbx) / den
        u = (ex * day - ey * dax) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0
    return min(
        point_seg(ax, ay, bx, by, dbx, dby),
        point_seg(ax + dax, ay + day, bx, by, dbx, dby),
        point_seg(bx, by, ax, ay, dax, day),
        point_seg(bx + dbx, by + dby, ax, ay, dax, day),
    )


@dataclass
class _Shape:
    """A placed footprint: 2D capsule, plus the top of the item for stacking."""

    name: str
    p0: np.ndarray
    p1: np.ndarray
    radius: float
    z_top: float
    writing: bool
    layer: int

    def gap(self, other: "_Shape") -> float:
        return segment_distance(self.p0, self.p1, other.p0, other.p1) - self.radius - other.radius


def _station_shapes() -> list[_Shape]:
    out = []
    for s in STATIONS:
        hx, hy = s.half
        r = hy
        p0, p1 = _segment(s.xy, s.yaw, max(hx - hy, 0.0))
        # Corners of a box tray poke out of the capsule; pad the radius.
        out.append(_Shape(s.name, p0, p1, math.hypot(r, r) + 0.01, 1.0, False, 0))
    base = np.zeros(2)
    out.append(_Shape("arm_base", base, base, BASE_KEEPOUT_M, 1.0, False, 0))
    return out


def _sample_xy(rng: np.random.Generator) -> np.ndarray:
    r = rng.uniform(*CLUTTER_RADIUS_M)
    a = math.radians(rng.uniform(-CLUTTER_HALF_ANGLE_DEG, CLUTTER_HALF_ANGLE_DEG))
    return np.array([r * math.cos(a), r * math.sin(a)])


def _in_fan(xy: np.ndarray) -> bool:
    r = float(np.linalg.norm(xy))
    a = math.degrees(math.atan2(xy[1], xy[0]))
    return CLUTTER_RADIUS_M[0] <= r <= CLUTTER_RADIUS_M[1] and abs(a) <= CLUTTER_HALF_ANGLE_DEG


def park_pose(index: int, item: PoolItem) -> tuple[float, float, float]:
    """A spot of its own on the floor under the table."""
    cols = 4
    row, col = divmod(index, cols)
    x = PARK_X[0] + (PARK_X[1] - PARK_X[0]) * (col + 0.5) / cols
    y = PARK_Y[0] + 0.06 * (row + 0.5)
    return (x, min(y, PARK_Y[1]), FLOOR_Z + item.half[2] + 0.002)


def sample_desk(
    rng: np.random.Generator,
    pool: list[PoolItem],
    n_writing: tuple[int, int] = (3, 8),
    n_supplies: tuple[int, int] = (1, 4),
    stack_p: float = 0.35,
    tries: int = 200,
) -> list[Placement]:
    """Which items are on the desk this episode, their poses and their looks.

    Returns one Placement per pool item, in pool order. Inactive items get a
    parking pose. `stack_p` is the chance that a pen-like item is dropped onto
    pens already placed instead of onto a clear patch of desk.
    """
    writing = [i for i, it in enumerate(pool) if it.spec.writing]
    supplies = [i for i, it in enumerate(pool) if not it.spec.writing]
    nw = int(rng.integers(min(n_writing[0], len(writing)), min(n_writing[1], len(writing)) + 1))
    ns = int(rng.integers(min(n_supplies[0], len(supplies)), min(n_supplies[1], len(supplies)) + 1))
    chosen_w = [int(i) for i in rng.permutation(writing)[:nw]]
    chosen_s = [int(i) for i in rng.permutation(supplies)[:ns]]

    placed: list[_Shape] = _station_shapes()
    out: dict[int, Placement] = {}

    def look(i):
        return sample_look(rng, pool[i].kind)

    # Bulky supplies first, so pens pile up in the gaps between them.
    for i in chosen_s:
        it = pool[i]
        half_len, radius = it.footprint()
        for _ in range(tries):
            xy, yaw = _sample_xy(rng), float(rng.uniform(-math.pi, math.pi))
            p0, p1 = _segment(xy, yaw, half_len)
            shape = _Shape(it.name, p0, p1, radius, 2 * it.half[2], False, 0)
            if all(shape.gap(o) >= CLEARANCE_M for o in placed):
                placed.append(shape)
                out[i] = Placement(it.name, it.kind, True, (float(xy[0]), float(xy[1]), it.half[2]), yaw, **look(i))
                break

    for i in chosen_w:
        it = pool[i]
        half_len, radius = it.footprint()
        hz = it.half[2]
        pens = [s for s in placed if s.writing]
        stack = bool(pens) and rng.random() < stack_p
        for _ in range(tries):
            yaw = float(rng.uniform(-math.pi, math.pi))
            if stack:
                # Cross an existing pen near its middle, at a good angle.
                base = pens[int(rng.integers(len(pens)))]
                mid = 0.5 * (base.p0 + base.p1)
                along = base.p1 - base.p0
                along = along / (np.linalg.norm(along) or 1.0)
                xy = mid + along * rng.uniform(-0.03, 0.03) + rng.normal(0.0, 0.01, 2)
                base_yaw = math.atan2(along[1], along[0])
                yaw = base_yaw + math.radians(rng.uniform(30.0, 150.0)) * rng.choice([-1, 1])
                if not _in_fan(xy):
                    continue
            else:
                xy = _sample_xy(rng)
            p0, p1 = _segment(xy, yaw, half_len)
            shape = _Shape(it.name, p0, p1, radius, 0.0, True, 0)
            hits = [o for o in placed if shape.gap(o) < CLEARANCE_M]
            if not stack:
                if hits:
                    continue
                z = hz
            else:
                # Only on pens, never on a supply, a station or the base.
                if not hits or any(not o.writing for o in hits):
                    continue
                layer = 1 + max(o.layer for o in hits)
                if layer >= MAX_STACK:
                    continue
                z = max(o.z_top for o in hits) + STACK_GAP_M + hz
                shape.layer = layer
            shape.z_top = z + hz
            placed.append(shape)
            out[i] = Placement(
                it.name, it.kind, True, (float(xy[0]), float(xy[1]), float(z)), float(yaw),
                layer=shape.layer, on_top_of=[o.name for o in hits], **look(i),
            )
            break
        else:
            if stack:
                # No good crossing found: put it on clear desk instead.
                for _ in range(tries):
                    xy, yaw = _sample_xy(rng), float(rng.uniform(-math.pi, math.pi))
                    p0, p1 = _segment(xy, yaw, half_len)
                    shape = _Shape(it.name, p0, p1, radius, 2 * hz, True, 0)
                    if all(shape.gap(o) >= CLEARANCE_M for o in placed):
                        placed.append(shape)
                        out[i] = Placement(it.name, it.kind, True, (float(xy[0]), float(xy[1]), hz), yaw, **look(i))
                        break

    result = []
    for i, it in enumerate(pool):
        if i in out:
            result.append(out[i])
        else:
            result.append(Placement(it.name, it.kind, False, park_pose(i, it), 0.0, **look(i)))
    return result
