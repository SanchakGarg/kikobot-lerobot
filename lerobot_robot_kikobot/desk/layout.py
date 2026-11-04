"""The desk sorting station: what is on the desk each episode, where, and how it looks.

Shared by both simulators: the MuJoCo scene (desk/mujoco_desk.py) and the
Isaac Sim server (isaac/desk_assets.py) build the same stations from
`STATIONS` and place the same items from `sample_desk`. Pure numpy, no
package imports: the Isaac Sim server loads this file by path. Tested in
tests/test_desk_layout.py.

Frames: the arm's base is at the origin on the desktop (z = 0), in the
desk's front-left corner (env/room.py); the desk runs to +X and -Y from it.
Every item's local frame has its long axis along X and its resting face
down, so a placement is just (x, y, z, yaw).

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
    """One kind of item, lying on the desk.

    `half` are the collider's half extents in the item frame (X = long axis)
    for the kind's most common size; `sizes` are the real sizes it comes in
    (half extents too, the first being `half`), which the pool cycles
    through. `collider` is capsule | cylinder_x | cylinder_z | ring | box |
    scissors. A ring is a hollow upright cylinder (a tape roll, which drops
    over the tape peg), `ring_inner` its inner radius. `mass` is for `half`;
    other sizes scale it with their volume. The layout only needs the
    footprint (a 2D capsule: segment half length along X plus radius) and
    `half[2]`, the height of the item's centre above the desk.
    """

    collider: str
    half: tuple[float, float, float]
    mass: float
    friction: float
    writing: bool = False  # pen-like: may be stacked, goes in a cup
    ring_inner: float = 0.0
    sizes: tuple[tuple[float, float, float], ...] = ()

    def all_sizes(self) -> tuple[tuple[float, float, float], ...]:
        return (self.half, *(h for h in self.sizes if h != self.half))

    def footprint(self) -> tuple[float, float]:
        hx, hy, _ = self.half
        if self.collider in ("capsule", "cylinder_x", "cylinder_z", "ring"):
            return max(hx - hy, 0.0), hy
        # Boxes: the capsule that covers the corners.
        r = math.hypot(hy, hy)
        return max(hx - hy, 0.0), r


# Sizes are real products'. Two limits keep every size workable: the open
# jaws span 53.6 mm, so anything gripped across is at most ~37 mm wide (a
# tape roll, gripped round its rim with the jaws wide open, at most 46 mm
# across), and the supply tray's compartments take items up to 11 cm long
# and 3.7 cm wide.
KINDS: dict[str, KindSpec] = {
    # Ballpoint, 14 cm x 10 mm; a slim stick pen, 13.5 cm x 8 mm; a chunky gel pen, 14.5 cm x 11 mm.
    "pen": KindSpec("capsule", (0.070, 0.005, 0.005), 0.008, 0.8, writing=True,
                    sizes=((0.0675, 0.004, 0.004), (0.0725, 0.0055, 0.0055))),
    # Hex pencil, 7.5 mm across the flats (box collider: no rolling): new
    # (17.5 cm), part used (13 cm) and a stub (9.5 cm).
    "pencil": KindSpec("box", (0.0875, 0.00375, 0.00375), 0.006, 0.9, writing=True,
                       sizes=((0.065, 0.00375, 0.00375), (0.0475, 0.00375, 0.00375))),
    # Felt-tip marker, 13 cm x 17 mm; a fineliner, 13.5 cm x 11 mm; a whiteboard marker, 12 cm x 19 mm.
    "marker": KindSpec("capsule", (0.065, 0.0085, 0.0085), 0.015, 0.8, writing=True,
                       sizes=((0.0675, 0.0055, 0.0055), (0.060, 0.0095, 0.0095))),
    # Flat highlighter, 12 cm x 22 x 14 mm; a pocket one, 10.5 cm; a big one, 12.5 cm.
    "highlighter": KindSpec("box", (0.060, 0.011, 0.007), 0.014, 0.8, writing=True,
                            sizes=((0.0525, 0.0095, 0.0065), (0.0625, 0.0115, 0.0075))),
    # Glue stick lying down: 15 g (8.5 cm x 22 mm), 8 g (7 cm x 18 mm), 21 g (9 cm x 25 mm).
    "glue_stick": KindSpec("cylinder_x", (0.0425, 0.011, 0.011), 0.025, 1.0,
                           sizes=((0.035, 0.009, 0.009), (0.045, 0.0125, 0.0125))),
    # Tape roll lying flat, on a 25 mm core: 13 mm x 44 mm, a mini 12 mm x
    # 33 mm, 19 mm x 44 mm (the classic desk roll) and 25 mm x 46 mm.
    "tape_roll": KindSpec("ring", (0.022, 0.022, 0.0065), 0.030, 1.0, ring_inner=0.0125,
                          sizes=((0.0165, 0.0165, 0.006), (0.022, 0.022, 0.0095), (0.023, 0.023, 0.0125))),
    # Scissors lying flat: blades + handle loops, 14 cm; 13 cm kids'; 17 cm office.
    "scissors": KindSpec("scissors", (0.070, 0.022, 0.004), 0.040, 0.6,
                         sizes=((0.065, 0.020, 0.004), (0.085, 0.022, 0.0045))),
    # Block eraser, 44 x 22 x 12 mm; 40 x 20 x 12 mm; a 60 x 25 x 13 mm one.
    "eraser": KindSpec("box", (0.022, 0.011, 0.006), 0.010, 1.2,
                       sizes=((0.020, 0.010, 0.006), (0.030, 0.0125, 0.0065))),
    # Sticky notes: a 76 mm pad, 16 mm thick; a 51 x 38 mm pad, 13 mm thick.
    "sticky_notes": KindSpec("box", (0.038, 0.038, 0.008), 0.030, 1.0,
                             sizes=((0.0255, 0.019, 0.0065),)),
    # Desk stapler, 13 cm x 36 mm x 44 mm; a mini one, 10 cm x 29 x 34 mm.
    "stapler": KindSpec("box", (0.065, 0.018, 0.022), 0.150, 0.8,
                        sizes=((0.050, 0.0145, 0.017),)),
    # Ruler, 20 cm or 15 cm, 30 / 25 mm wide, 4 mm thick.
    "ruler": KindSpec("box", (0.100, 0.015, 0.002), 0.012, 0.7,
                      sizes=((0.075, 0.0125, 0.002),)),
    # Box of paper clips: 50 x 36 x 24 mm; small 40 x 30 x 20 mm; large 56 x 37 x 26 mm.
    "paper_clip_box": KindSpec("box", (0.025, 0.018, 0.012), 0.030, 0.9,
                               sizes=((0.020, 0.015, 0.010), (0.028, 0.0185, 0.013))),
    # Pencil sharpener: single hole, 26 x 20 x 20 mm; double hole, 34 x 22 x 22 mm.
    "sharpener": KindSpec("box", (0.013, 0.010, 0.010), 0.010, 0.9,
                          sizes=((0.017, 0.011, 0.011),)),
}


@dataclass(frozen=True)
class PoolItem:
    """One item that exists in the scene. Built once; shown or parked per episode."""

    name: str
    kind: str
    # Collider half extents: one of KINDS[kind].all_sizes(). The Isaac
    # backend replaces it with the real size when it loads a Nucleus asset.
    half: tuple[float, float, float]

    @property
    def spec(self) -> KindSpec:
        return KINDS[self.kind]

    @property
    def mass(self) -> float:
        """The kind's mass, scaled to this item's size by volume."""
        return self.spec.mass * float(np.prod(self.half) / np.prod(self.spec.half))

    def footprint(self) -> tuple[float, float]:
        return replace(self.spec, half=self.half).footprint()


# How many of each kind exist. Each episode shows a random subset.
DEFAULT_COUNTS: dict[str, int] = {
    "pen": 9,
    "pencil": 4,
    "marker": 4,
    "highlighter": 3,
    "glue_stick": 3,
    "tape_roll": 4,
    "eraser": 3,
    "sticky_notes": 2,
    "scissors": 1,
    "stapler": 1,
    "ruler": 1,
    "paper_clip_box": 2,
    "sharpener": 2,
}


def default_pool(counts: dict[str, int] = DEFAULT_COUNTS) -> list[PoolItem]:
    """Every item, each kind cycling through its real sizes, so every desk
    shows a spread of them."""
    out = []
    for kind, n in counts.items():
        sizes = KINDS[kind].all_sizes()
        out += [PoolItem(f"{kind}_{i}", kind, sizes[i % len(sizes)]) for i in range(n)]
    return out


# ------------------------------------------------------------- the stations


@dataclass(frozen=True)
class StationPart:
    """One solid of a station, in the station's frame (z = 0 on the desk).

    box: `half` extents, turned by `yaw` about Z. cylinder: (radius, radius,
    half height), upright. tube: (inner radius, outer radius, half height),
    upright, for the eye only (its walls collide as boxes). `role` names its
    look in meshes.FIXED_LOOKS.
    """

    name: str
    shape: str
    center: tuple[float, float, float]
    half: tuple[float, float, float]
    role: str
    yaw: float = 0.0
    collide: bool = True
    visual: bool = True


@dataclass(frozen=True)
class Station:
    """A fixed receptacle on the desk, and the kinds of item that belong in it.

    `xy` and `yaw` place it; `half` is its footprint (for keeping the clutter
    off it); `parts` is its geometry, which both simulators build as is.
    """

    name: str
    xy: tuple[float, float]
    yaw: float
    half: tuple[float, float]
    accepts: tuple[str, ...] = ()
    parts: tuple[StationPart, ...] = ()


def _polar(r: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return (r * math.cos(a), r * math.sin(a))


def _cup(name, xy, r_out, height, accepts, role="cup", n_walls=16) -> Station:
    """A hollow cup: a tube to look at, a ring of box walls to collide with."""
    t = 0.004
    r_in = r_out - t
    r_mid = r_out - t / 2
    seg = r_mid * math.tan(math.pi / n_walls) + 0.0005
    parts = [StationPart("visual", "tube", (0, 0, height / 2), (r_in, r_out, height / 2), role, collide=False)]
    parts.append(StationPart("floor", "cylinder", (0, 0, 0.002), (r_in, r_in, 0.002), role))
    for k in range(n_walls):
        a = 2 * math.pi * k / n_walls
        parts.append(StationPart(
            f"wall_{k}", "box", (r_mid * math.cos(a), r_mid * math.sin(a), height / 2), (t / 2, seg, height / 2),
            role, yaw=a, visual=False,
        ))
    return Station(name, xy, 0.0, (r_out, r_out), accepts, tuple(parts))


def _peg(name, xy, accepts) -> Station:
    """A post on a round base, for tape rolls to drop over (their core is 25 mm across)."""
    parts = (
        StationPart("base", "cylinder", (0, 0, 0.004), (0.035, 0.035, 0.004), "wood"),
        StationPart("post", "cylinder", (0, 0, 0.008 + 0.03), (0.008, 0.008, 0.03), "metal"),
    )
    return Station(name, xy, 0.0, (0.035, 0.035), accepts, parts)


def _tray(name, xy, yaw, hx, hy, wall_h, compartments, accepts) -> Station:
    """An open tray, split into `compartments` along its length."""
    t = 0.002
    parts = [
        StationPart("floor", "box", (0, 0, 0.002), (hx, hy, 0.002), "tray_dark"),
        StationPart("wall_px", "box", (hx - t, 0, wall_h / 2), (t, hy, wall_h / 2), "tray"),
        StationPart("wall_nx", "box", (-hx + t, 0, wall_h / 2), (t, hy, wall_h / 2), "tray"),
        StationPart("wall_py", "box", (0, hy - t, wall_h / 2), (hx, t, wall_h / 2), "tray"),
        StationPart("wall_ny", "box", (0, -hy + t, wall_h / 2), (hx, t, wall_h / 2), "tray"),
    ]
    for k in range(1, compartments):
        x = -hx + 2 * hx * k / compartments
        parts.append(StationPart(f"divider_{k}", "box", (x, 0, wall_h * 0.4), (t, hy, wall_h * 0.4), "tray"))
    return Station(name, xy, yaw, (hx, hy), accepts, tuple(parts))


def _pad_zone(name, xy, yaw, h, accepts) -> Station:
    """A felt mat with a low rim, where the sticky-note pads go."""
    t, rim = 0.002, 0.004
    parts = (
        StationPart("mat", "box", (0, 0, 0.001), (h, h, 0.001), "felt"),
        StationPart("rim_px", "box", (h - t, 0, rim), (t, h, rim), "tray"),
        StationPart("rim_nx", "box", (-h + t, 0, rim), (t, h, rim), "tray"),
        StationPart("rim_py", "box", (0, h - t, rim), (h, t, rim), "tray"),
        StationPart("rim_ny", "box", (0, -h + t, rim), (h, t, rim), "tray"),
    )
    return Station(name, xy, yaw, (h, h), accepts, parts)


# The way the arm looks over the desk from its corner, as env/room.py's
# WORK_YAW_DEG: the middle of the quarter between -X and +Y.
WORK_YAW_DEG = 135.0

# Around the arm, inside its reach and on the desk, clear of the clutter fan
# that opens towards WORK_YAW_DEG and of the arm's blind spot (between about
# -90 and -50 degrees, where J1 runs out, which points off the desk): along
# the front edge the marker cup, the tape peg (under the ready pose, where
# only a low station may be) and the supply tray, the pen cup to the arm's
# right, and in the back-left corner the note mat and the long tray.
PEN_CUP = _cup("pen_cup", _polar(0.265, 90.0), 0.038, 0.095, ("pen", "pencil"))
MARKER_CUP = _cup("marker_cup", _polar(0.235, -33.0), 0.042, 0.085, ("marker", "highlighter"), role="cup_blue")
TAPE_PEG = _peg("tape_peg", _polar(0.23, 19.0), ("tape_roll",))
TRAY = _tray("supply_tray", _polar(0.31, 54.0), math.radians(54.0), 0.095, 0.060, 0.03, 3,
             ("glue_stick", "eraser", "sharpener", "paper_clip_box"))
NOTE_MAT = _pad_zone("note_mat", _polar(0.22, -127.0), math.radians(-37.0), 0.045, ("sticky_notes",))
LONG_TRAY = _tray("long_tray", _polar(0.31, -167.0), math.radians(-77.0), 0.12, 0.045, 0.025, 1,
                  ("scissors", "ruler", "stapler"))
STATIONS = (PEN_CUP, MARKER_CUP, TAPE_PEG, TRAY, NOTE_MAT, LONG_TRAY)


def station_for(kind: str) -> Station:
    """Where an item of `kind` belongs."""
    for s in STATIONS:
        if kind in s.accepts:
            return s
    raise KeyError(f"No station accepts {kind!r}.")


# Clutter fan over the desk (item centres), centred on WORK_YAW_DEG, and the
# base to keep clear.
CLUTTER_RADIUS_M = (0.15, 0.38)  # Within the gamepad's 0.42 m workspace.
CLUTTER_HALF_ANGLE_DEG = 30.0
BASE_KEEPOUT_M = 0.145  # The base plate is 20 cm square: this clears its corners.
CLEARANCE_M = 0.006  # between items that may not touch
STACK_GAP_M = 0.004  # drop height above whatever a stacked pen crosses
MAX_STACK = 3  # layers in one pile

# Parking: inactive items lie on the floor behind the camera, in front of
# the desk: out of reach and out of view.
FLOOR_Z = -0.75  # As env/room.py.
PARK_X = (1.75, 2.55)
PARK_Y = (-0.6, 0.6)


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
    "sticky_notes": ("solid", "solid", "stripes"),
    "stapler": ("solid", "speckle"),
    "ruler": ("solid", "bands", "brand"),
    "paper_clip_box": ("brand", "checker", "solid"),
    "sharpener": ("solid", "speckle"),
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
    "sticky_notes": ("yellow", *NEON_COLORS),
    "stapler": ("black", "red", "royal_blue", "silver", "smoke"),
    "ruler": ("white", "silver", "yellow", "sky", "smoke"),
    "paper_clip_box": ("white", "red", "royal_blue", "yellow", "green"),
    "sharpener": ("red", "royal_blue", "silver", "yellow", "green", "black"),
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
    a = math.radians(WORK_YAW_DEG + rng.uniform(-CLUTTER_HALF_ANGLE_DEG, CLUTTER_HALF_ANGLE_DEG))
    return np.array([r * math.cos(a), r * math.sin(a)])


def _in_fan(xy: np.ndarray) -> bool:
    r = float(np.linalg.norm(xy))
    a = math.degrees(math.atan2(xy[1], xy[0])) - WORK_YAW_DEG
    a = (a + 180.0) % 360.0 - 180.0
    return CLUTTER_RADIUS_M[0] <= r <= CLUTTER_RADIUS_M[1] and abs(a) <= CLUTTER_HALF_ANGLE_DEG


def park_pose(index: int, item: PoolItem) -> tuple[float, float, float]:
    """A spot of its own on the floor under the desk."""
    cols = 4
    row, col = divmod(index, cols)
    x = PARK_X[0] + (PARK_X[1] - PARK_X[0]) * (col + 0.5) / cols
    y = PARK_Y[0] + 0.06 * (row + 0.5)
    return (x, min(y, PARK_Y[1]), FLOOR_Z + item.half[2] + 0.002)


def sample_desk(
    rng: np.random.Generator,
    pool: list[PoolItem],
    n_writing: tuple[int, int] = (3, 7),
    n_supplies: tuple[int, int] = (2, 4),
    stack_p: float = 0.2,
    tries: int = 200,
    require: tuple[str, ...] | None = None,
) -> list[Placement]:
    """Which items are on the desk this episode, their poses and their looks.

    Returns one Placement per pool item, in pool order. Inactive items get a
    parking pose. `stack_p` is the chance that a pen-like item is dropped onto
    pens already placed instead of onto a clear patch of desk. `require`:
    kinds of which at least one item must be on the desk; each is placed
    first in its group, so it gets a clear patch of desk (later pens may
    still land on it).
    """
    writing = [i for i, it in enumerate(pool) if it.spec.writing]
    supplies = [i for i, it in enumerate(pool) if not it.spec.writing]
    nw = int(rng.integers(min(n_writing[0], len(writing)), min(n_writing[1], len(writing)) + 1))
    ns = int(rng.integers(min(n_supplies[0], len(supplies)), min(n_supplies[1], len(supplies)) + 1))
    chosen_w = [int(i) for i in rng.permutation(writing)[:nw]]
    chosen_s = [int(i) for i in rng.permutation(supplies)[:ns]]
    for kind in require or ():
        group = chosen_w if KINDS[kind].writing else chosen_s
        have = [i for i in group if pool[i].kind == kind]
        if have:
            pick = have[0]
            group.remove(pick)
        else:
            candidates = [i for i, it in enumerate(pool) if it.kind == kind]
            if not candidates:
                raise ValueError(f"The pool has no {kind!r} to require.")
            pick = int(candidates[int(rng.integers(len(candidates)))])
            if group:
                group.pop()  # Keep the item count.
        group.insert(0, pick)
    # Nothing is stacked on a required item: it stays free to pick up.
    keep_clear = {pool[i].name for i in chosen_w[:1] + chosen_s[:1] if pool[i].kind in (require or ())}

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
            shape = _Shape(it.name, p0, p1, radius, 0.0, it.name not in keep_clear, 0)
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
