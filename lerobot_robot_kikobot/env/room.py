"""The lab the arm works in, described once for both simulators.

`lab_room()` lists every surface, light and camera of a makerspace: the
arm's study desk with its hutch (two side panels, a top shelf and a tube
light under it), the floor, walls and ceiling with eight small downlights, a
window, a workbench with a pegboard, a steel shelving unit with bins, a
whiteboard and a stool. The
MuJoCo server builds it with `env/mujoco_scene.py`, the Isaac Sim server
with `isaac/room_usd.py`, so the two scenes have the same geometry, colours
and lights. `sample_lighting` varies the light every reset.

Frame: the arm's base sits on the desktop at the origin, z = 0 is the top
surface of the desk, +X is the way the arm faces in its ready pose. A person
sits at the desk on its +X side, facing -X. The arm is in the desk's
front-left corner: the desk runs back (-X, to the hutch) and right (+Y)
from it. The `record` camera looks over the person's shoulder at the arm
and the items around it.

Pure numpy and the standard library, no package imports: the Isaac Sim
server, which has no LeRobot, loads this file by path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The room: x from ROOM_X[0] to ROOM_X[1], y +-ROOM_HALF_Y, floor to ceiling.
FLOOR_Z = -0.75  # A 75 cm desk.
CEILING_Z = 2.25  # 3 m room.
ROOM_X = (-2.2, 3.8)
ROOM_HALF_Y = 2.5
WALL_T = 0.05
# The desktop, 0.95 m deep (X) and 1.50 m wide (Y). The arm's base is in its
# front-left corner, 30 cm in from both edges (room for the sorting stations
# around it); its ~0.40 m reach leaves the back and the right of the desk out
# of reach. The arm is turned so that its blind spot (where J1's +-165 degree
# range runs out, around -70 degrees) points off the desk, between the front
# and left edges: J1 sweeps over the whole desk without crossing it.
DESK_X = (-0.65, 0.30)
DESK_Y = (-0.30, 1.20)
DESK_TOP_T = 0.025
# The hutch: two side panels on the desk's left and right edges, a top shelf
# across them that is shallower than the desk, and a tube light under it.
HUTCH_PANEL_T = 0.018
HUTCH_SHELF_Z = 0.75  # Underside of the top shelf, above the desktop.
HUTCH_SHELF_DEPTH = 0.30
HUTCH_SHELF_T = 0.02
# The way the arm looks over the desk: the middle of the quarter between -X
# and +Y. desk/layout.py keeps a copy (it cannot import this file).
WORK_YAW_DEG = 135.0


@dataclass(frozen=True)
class Material:
    """A surface look. `pattern` names a grayscale texture from desk/textures.py
    that tints `rgb` (1 = rgb, 0 = black), softened by `contrast` (see
    `texture_values`); `repeat` tiles it per metre."""

    name: str
    rgb: tuple[float, float, float]
    roughness: float = 0.6
    metallic: float = 0.0
    pattern: str | None = None
    repeat: float = 1.0
    emission: float = 0.0  # Self-lit (light panels, the window), 0..1.
    contrast: float = 1.0


@dataclass(frozen=True)
class Part:
    """One static solid. Boxes: `half` extents. Cylinders: (radius, radius, half height), along Z.
    Prisms: a convex polygon in the XZ plane (`profile`, (x, z) corners in
    order, about `center`), extruded +-half[1] along Y; `half` is its
    bounding box."""

    name: str
    shape: str  # box | cylinder | prism
    center: tuple[float, float, float]
    half: tuple[float, float, float]
    material: str
    collide: bool = True
    profile: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class Light:
    """`downlight`: a small round ceiling light (`radius`), shining down.
    `tube`: a fluorescent-style tube light, `length` long along Y, shining
    along `dir`. `intensity` is relative: 1.0 is a downlight at full power;
    each simulator maps it to its own units."""

    name: str
    kind: str  # downlight | tube
    pos: tuple[float, float, float]
    dir: tuple[float, float, float] = (0.0, 0.0, -1.0)
    radius: float = 0.06
    length: float = 0.0
    intensity: float = 1.0
    temp_k: float = 6000.0  # Cool white LED.
    castshadow: bool = False


@dataclass(frozen=True)
class Camera:
    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fovy: float


@dataclass(frozen=True)
class RoomSpec:
    materials: dict[str, Material]
    parts: tuple[Part, ...]
    lights: tuple[Light, ...]
    cameras: tuple[Camera, ...]


@dataclass(frozen=True)
class LightState:
    """One light for one episode."""

    name: str
    on: bool
    intensity: float  # Relative, as Light.intensity.
    rgb: tuple[float, float, float]
    temp_k: float
    dir: tuple[float, float, float]


# ----------------------------------------------------------------- the lab


def _materials() -> dict[str, Material]:
    m = [
        Material("desk_wood", (0.62, 0.45, 0.30), 0.45, 0.0, "laminate", 1.0, contrast=0.8),  # walnut laminate
        Material("desk_edge", (0.30, 0.21, 0.14), 0.45),  # edge banding
        Material("desk_frame", (0.16, 0.16, 0.17), 0.4, 0.6),
        Material("tube_housing", (0.92, 0.92, 0.90), 0.5),
        Material("tube_glow", (1.0, 1.0, 1.0), 0.9, 0.0, None, 1.0, emission=1.0),
        Material("downlight_trim", (0.95, 0.95, 0.94), 0.4),
        Material("floor", (0.45, 0.46, 0.48), 0.7, 0.0, "speckle", 1.0, contrast=0.3),  # grey epoxy
        Material("wall", (0.86, 0.86, 0.83), 0.9),
        Material("wall_band", (0.33, 0.34, 0.36), 0.7),
        Material("ceiling", (0.92, 0.92, 0.90), 0.95),
        Material("downlight", (1.0, 1.0, 1.0), 0.9, 0.0, None, 1.0, emission=1.0),
        Material("window", (0.62, 0.70, 0.80), 0.1, 0.0, None, 1.0, emission=0.35),  # dusk outside
        Material("window_frame", (0.80, 0.80, 0.80), 0.4, 0.6),
        Material("bench_top", (0.66, 0.52, 0.36), 0.6, 0.0, "wood", 1.5, contrast=0.6),  # butcher block
        Material("bench_frame", (0.18, 0.19, 0.21), 0.4, 0.7),
        Material("pegboard", (0.74, 0.62, 0.46), 0.8, 0.0, "dots", 4.0, contrast=0.7),
        Material("tool_red", (0.75, 0.12, 0.10), 0.4),
        Material("tool_blue", (0.12, 0.30, 0.70), 0.4),
        Material("tool_steel", (0.62, 0.63, 0.66), 0.3, 1.0),
        Material("tool_black", (0.06, 0.06, 0.07), 0.5),
        Material("printer", (0.14, 0.15, 0.16), 0.5, 0.2),
        Material("printer_bed", (0.30, 0.30, 0.32), 0.3, 0.8),
        Material("shelf_steel", (0.58, 0.60, 0.62), 0.35, 0.9),
        Material("bin_blue", (0.15, 0.35, 0.75), 0.5),
        Material("bin_red", (0.80, 0.20, 0.15), 0.5),
        Material("bin_yellow", (0.92, 0.75, 0.15), 0.5),
        Material("bin_grey", (0.30, 0.31, 0.33), 0.5),
        Material("whiteboard", (0.96, 0.96, 0.96), 0.15),
        Material("whiteboard_frame", (0.62, 0.63, 0.65), 0.3, 0.9),
        Material("stool_seat", (0.10, 0.10, 0.11), 0.6),
        Material("stool_frame", (0.55, 0.56, 0.58), 0.3, 0.9),
    ]
    return {x.name: x for x in m}


def _box(name, center, half, material, collide=True) -> Part:
    return Part(name, "box", tuple(map(float, center)), tuple(map(float, half)), material, collide)


def _cyl(name, center, radius, half_height, material, collide=True) -> Part:
    return Part(name, "cylinder", tuple(map(float, center)), (radius, radius, half_height), material, collide)


def _prism(name, profile_xz, y, half_y, material, collide=True) -> Part:
    """A convex XZ polygon (absolute x, z corners), `half_y` thick about `y`."""
    pts = np.asarray(profile_xz, float)
    lo, hi = pts.min(0), pts.max(0)
    c = 0.5 * (lo + hi)
    rel = tuple((float(x - c[0]), float(z - c[1])) for x, z in pts)
    return Part(name, "prism", (float(c[0]), float(y), float(c[1])),
                (float(0.5 * (hi[0] - lo[0])), float(half_y), float(0.5 * (hi[1] - lo[1]))), material, collide, rel)


def _desk() -> list[Part]:
    """The arm's study desk: a laminate top on steel legs, a modesty panel, and
    the hutch: two side panels that run the full depth of the desk at the
    bottom and narrow to the top shelf's depth, the shelf, and a tube light
    fitting under the shelf. The front of the desk is its +X edge."""
    (x0, x1), (y0, y1) = DESK_X, DESK_Y
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hx, hy = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    t = DESK_TOP_T
    leg_h = (-t - FLOOR_Z) / 2
    parts = [
        _box("desk_top", (cx, cy, -t / 2), (hx, hy, t / 2), "desk_wood"),
        _box("desk_edge_front", (x1 + 0.001, cy, -t / 2), (0.001, hy, t / 2), "desk_edge", False),
    ]
    for i, (sx, sy) in enumerate(((1, 1), (-1, 1), (1, -1), (-1, -1))):
        parts.append(_box(f"desk_leg{i}", (cx + sx * (hx - 0.03), cy + sy * (hy - 0.03), FLOOR_Z + leg_h),
                          (0.02, 0.02, leg_h), "desk_frame"))
    # Modesty panel at the back, under the top.
    mz0, mz1 = FLOOR_Z + 0.3, -t
    parts.append(_box("desk_modesty", (x0 + 0.04, cy, 0.5 * (mz0 + mz1)), (0.008, hy - 0.06, 0.5 * (mz1 - mz0)),
                      "desk_wood", False))
    # The hutch.
    pt = HUTCH_PANEL_T
    z_top = HUTCH_SHELF_Z + HUTCH_SHELF_T
    side = ((x1, 0.0), (x0, 0.0), (x0, z_top), (x0 + HUTCH_SHELF_DEPTH, z_top))
    parts.append(_prism("hutch_side_left", side, y0 + pt / 2, pt / 2, "desk_wood"))
    parts.append(_prism("hutch_side_right", side, y1 - pt / 2, pt / 2, "desk_wood"))
    sx1 = x0 + HUTCH_SHELF_DEPTH
    parts.append(_box("hutch_shelf", (0.5 * (x0 + sx1), cy, HUTCH_SHELF_Z + HUTCH_SHELF_T / 2),
                      (0.5 * HUTCH_SHELF_DEPTH, hy - pt, HUTCH_SHELF_T / 2), "desk_wood"))
    # The tube light's fitting, near the shelf's front edge, and the glowing tube.
    tl = _tube_light()
    parts.append(_box("tube_housing", (tl.pos[0], cy, HUTCH_SHELF_Z - 0.012), (0.022, 0.5 * tl.length + 0.03, 0.012),
                      "tube_housing", False))
    parts.append(_box("tube_glow", (tl.pos[0], cy, tl.pos[2]), (0.012, 0.5 * tl.length, 0.012), "tube_glow", False))
    return parts


def _tube_light() -> Light:
    """Under the hutch's shelf, along Y, tilted a little towards the person."""
    (x0, x1), (y0, y1) = DESK_X, DESK_Y
    d = np.array([0.3, 0.0, -1.0])
    d /= np.linalg.norm(d)
    return Light("tube", "tube", (x0 + HUTCH_SHELF_DEPTH - 0.06, 0.5 * (y0 + y1), HUTCH_SHELF_Z - 0.036),
                 tuple(float(v) for v in d), radius=0.012, length=1.2, intensity=1.0, temp_k=6500.0, castshadow=True)


def _shell() -> list[Part]:
    """Floor, four walls with a grey base band, ceiling, window."""
    x0, x1 = ROOM_X
    cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
    hy = ROOM_HALF_Y
    h = (CEILING_Z - FLOOR_Z) / 2
    zc = FLOOR_Z + h
    t = WALL_T / 2
    band_h = 0.075
    parts = [
        _box("floor", (cx, 0, FLOOR_Z - 0.02), (hx, hy, 0.02), "floor"),
        _box("ceiling", (cx, 0, CEILING_Z + 0.02), (hx, hy, 0.02), "ceiling", False),
        _box("wall_back", (x0 - t, 0, zc), (t, hy, h), "wall"),  # behind the arm, -X
        _box("wall_front", (x1 + t, 0, zc), (t, hy, h), "wall"),
        _box("wall_left", (cx, hy + t, zc), (hx, t, h), "wall"),  # +Y, with the window
        _box("wall_right", (cx, -hy - t, zc), (hx, t, h), "wall"),
    ]
    bz = FLOOR_Z + band_h
    parts += [
        _box("band_back", (x0 + 0.005, 0, bz), (0.005, hy, band_h), "wall_band", False),
        _box("band_front", (x1 - 0.005, 0, bz), (0.005, hy, band_h), "wall_band", False),
        _box("band_left", (cx, hy - 0.005, bz), (hx, 0.005, band_h), "wall_band", False),
        _box("band_right", (cx, -hy + 0.005, bz), (hx, 0.005, band_h), "wall_band", False),
    ]
    # A strip window in the +Y wall, from 0.9 m above the floor to 2.3 m.
    wx, whx = 0.9, 1.4
    wz, whz = FLOOR_Z + 1.6, 0.7
    parts.append(_box("window", (wx, hy - 0.004, wz), (whx, 0.004, whz), "window", False))
    for i, z in enumerate((wz - whz, wz + whz)):
        parts.append(_box(f"window_sill{i}", (wx, hy - 0.03, z), (whx + 0.04, 0.03, 0.025), "window_frame", False))
    for i, x in enumerate((wx - whx, wx, wx + whx)):
        parts.append(_box(f"window_mullion{i}", (x, hy - 0.02, wz), (0.025, 0.02, whz), "window_frame", False))
    return parts


def _workbench() -> list[Part]:
    """Along the -X wall, behind the desk's hutch."""
    x0 = ROOM_X[0]
    top_z = FLOOR_Z + 0.9
    bx, bhx, bhy = x0 + 0.4, 0.35, 1.2
    parts = [_box("bench_top", (bx, 0, top_z - 0.025), (bhx, bhy, 0.025), "bench_top")]
    leg_h = (top_z - 0.05 - FLOOR_Z) / 2
    for i, (sx, sy) in enumerate(((1, 1), (-1, 1), (1, -1), (-1, -1))):
        parts.append(_box(f"bench_leg{i}", (bx + sx * (bhx - 0.04), sy * (bhy - 0.04), FLOOR_Z + leg_h), (0.025, 0.025, leg_h), "bench_frame"))
    parts.append(_box("bench_shelf", (bx, 0, FLOOR_Z + 0.2), (bhx - 0.05, bhy - 0.05, 0.012), "bench_frame"))
    # Pegboard and the tools hanging on it.
    pz = top_z + 0.55
    parts.append(_box("pegboard", (x0 + 0.012, 0, pz), (0.008, 1.1, 0.45), "pegboard", False))
    tools = [
        ("tool_hammer_handle", (0.012, 0.012, 0.13), (-0.85, pz - 0.05), "tool_black"),
        ("tool_hammer_head", (0.015, 0.06, 0.018), (-0.85, pz + 0.09), "tool_steel"),
        ("tool_wrench_l", (0.006, 0.012, 0.11), (-0.6, pz), "tool_steel"),
        ("tool_wrench_m", (0.006, 0.010, 0.09), (-0.52, pz + 0.02), "tool_steel"),
        ("tool_wrench_s", (0.006, 0.009, 0.07), (-0.45, pz + 0.04), "tool_steel"),
        ("tool_pliers", (0.008, 0.02, 0.09), (-0.25, pz), "tool_red"),
        ("tool_screwdriver_a", (0.01, 0.01, 0.1), (0.0, pz + 0.05), "tool_blue"),
        ("tool_screwdriver_b", (0.01, 0.01, 0.1), (0.08, pz + 0.05), "tool_red"),
        ("tool_saw", (0.004, 0.16, 0.05), (0.4, pz + 0.2), "tool_steel"),
        ("tool_saw_grip", (0.012, 0.04, 0.05), (0.6, pz + 0.2), "tool_black"),
        ("tool_tape_measure", (0.02, 0.035, 0.035), (0.8, pz - 0.15), "tool_blue"),
    ]
    for name, half, (y, z), mat in tools:
        parts.append(_box(name, (x0 + 0.02 + half[0], y, z), half, mat, False))
    # A desktop 3D printer on the bench, and a few parts bins.
    px, py = bx + 0.02, 0.75
    parts += [
        _box("printer_base", (px, py, top_z + 0.04), (0.2, 0.2, 0.04), "printer"),
        _box("printer_bed", (px, py, top_z + 0.09), (0.16, 0.16, 0.006), "printer_bed"),
        _box("printer_post_l", (px - 0.17, py, top_z + 0.27), (0.02, 0.02, 0.19), "printer"),
        _box("printer_post_r", (px + 0.17, py, top_z + 0.27), (0.02, 0.02, 0.19), "printer"),
        _box("printer_gantry", (px, py, top_z + 0.42), (0.19, 0.03, 0.025), "printer"),
        _box("bench_bin0", (bx + 0.1, -0.8, top_z + 0.05), (0.08, 0.12, 0.05), "bin_blue"),
        _box("bench_bin1", (bx + 0.1, -0.5, top_z + 0.05), (0.08, 0.12, 0.05), "bin_yellow"),
    ]
    return parts


def _shelves() -> list[Part]:
    """A steel shelving unit with storage bins, against the -Y wall."""
    sx, sy = 2.2, -ROOM_HALF_Y + 0.25
    hx, hy = 0.6, 0.2
    height = 1.8
    parts = []
    for i, (ux, uy) in enumerate(((1, 1), (-1, 1), (1, -1), (-1, -1))):
        parts.append(_box(f"shelf_post{i}", (sx + ux * hx, sy + uy * hy, FLOOR_Z + height / 2), (0.015, 0.015, height / 2), "shelf_steel"))
    colors = ("bin_blue", "bin_red", "bin_yellow", "bin_grey")
    for k, z in enumerate((0.1, 0.55, 1.0, 1.45)):
        shelf_z = FLOOR_Z + z
        parts.append(_box(f"shelf_board{k}", (sx, sy, shelf_z), (hx, hy, 0.01), "shelf_steel"))
        if z > 1.7:
            continue
        for j in range(3):
            parts.append(_box(
                f"shelf_bin{k}_{j}", (sx - 0.38 + 0.38 * j, sy, shelf_z + 0.01 + 0.1), (0.16, 0.17, 0.1),
                colors[(k + j) % len(colors)], False,
            ))
    return parts


def _furniture() -> list[Part]:
    wb_y = -ROOM_HALF_Y + 0.012
    parts = [
        _box("whiteboard", (0.1, wb_y, FLOOR_Z + 1.45), (0.9, 0.006, 0.55), "whiteboard", False),
        _box("whiteboard_frame_top", (0.1, wb_y + 0.004, FLOOR_Z + 2.0), (0.92, 0.012, 0.015), "whiteboard_frame", False),
        _box("whiteboard_frame_bottom", (0.1, wb_y + 0.004, FLOOR_Z + 0.9), (0.92, 0.012, 0.015), "whiteboard_frame", False),
        _box("whiteboard_tray", (0.1, wb_y + 0.04, FLOOR_Z + 0.89), (0.5, 0.04, 0.008), "whiteboard_frame", False),
    ]
    # A lab stool, pushed aside to the left of the desk.
    stx, sty = 0.9, -1.0
    seat_z = FLOOR_Z + 0.65
    parts += [
        _cyl("stool_seat", (stx, sty, seat_z), 0.17, 0.025, "stool_seat"),
        _cyl("stool_pole", (stx, sty, FLOOR_Z + 0.33), 0.022, 0.3, "stool_frame"),
        _cyl("stool_foot", (stx, sty, FLOOR_Z + 0.015), 0.22, 0.015, "stool_frame"),
        _cyl("stool_ring", (stx, sty, FLOOR_Z + 0.25), 0.16, 0.008, "stool_frame", False),
    ]
    return parts


# Eight small downlights in two rows of four across the ceiling.
DOWNLIGHT_X = (-1.45, 0.05, 1.55, 3.05)
DOWNLIGHT_Y = (-1.25, 1.25)


def _lights() -> list[Light]:
    lights = []
    for i, x in enumerate(DOWNLIGHT_X):
        for j, y in enumerate(DOWNLIGHT_Y):
            # The one nearest the desk also casts shadows, as a real room's would.
            lights.append(Light(f"down_{i}{j}", "downlight", (x, y, CEILING_Z - 0.012),
                                castshadow=(i, j) == (1, 0)))
    lights.append(_tube_light())
    return lights


def _light_parts(lights: list[Light]) -> list[Part]:
    """The downlights' trims and glowing faces, so they show in the images."""
    out = []
    for l in lights:
        if l.kind != "downlight":
            continue
        out.append(_cyl(f"{l.name}_trim", (l.pos[0], l.pos[1], CEILING_Z - 0.004), l.radius + 0.018, 0.004,
                        "downlight_trim", False))
        out.append(_cyl(f"{l.name}_face", (l.pos[0], l.pos[1], CEILING_Z - 0.009), l.radius, 0.0015,
                        "downlight", False))
    return out


# The one camera: in front of the desk, over the shoulder of a person sitting
# there, looking down at the arm and the items around it.
RECORD_CAMERA = Camera("record", (0.65, 0.30, 0.80), (-0.06, 0.12, 0.0), (0.0, 0.0, 1.0), 40.0)


def _cameras() -> list[Camera]:
    return [RECORD_CAMERA]


def lab_room() -> RoomSpec:
    lights = _lights()
    parts = _desk() + _shell() + _workbench() + _shelves() + _furniture() + _light_parts(lights)
    return RoomSpec(_materials(), tuple(parts), tuple(lights), tuple(_cameras()))


# ----------------------------------------------------------------- lighting


def kelvin_to_rgb(temp_k: float) -> tuple[float, float, float]:
    """Colour of a black body at `temp_k`, scaled so the brightest channel is 1.

    Tanner Helland's fit, good to a few percent from 1000 K to 40000 K.
    """
    t = float(np.clip(temp_k, 1000.0, 40000.0)) / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    rgb = np.clip([r, g, b], 0.0, 255.0) / 255.0
    return tuple(float(c) for c in rgb / rgb.max())


def fixed_lighting(spec: RoomSpec) -> list[LightState]:
    return [LightState(l.name, True, l.intensity, kelvin_to_rgb(l.temp_k), l.temp_k, l.dir) for l in spec.lights]


def sample_lighting(rng: np.random.Generator, spec: RoomSpec) -> list[LightState]:
    """The room's light for one episode.

    The downlights share one colour temperature (one batch of LEDs), each
    varies a little around a common dimmer level, and each may be off (at
    least two stay on). The tube light under the hutch is on in most
    episodes, a little brighter or dimmer each time.
    """
    level = rng.uniform(0.7, 1.3)
    temp = rng.uniform(5000.0, 6500.0)
    downs = [l for l in spec.lights if l.kind == "downlight"]
    on = rng.random(len(downs)) >= 0.1
    while on.sum() < 2:
        on[rng.integers(len(downs))] = True
    out = []
    for l, is_on in zip(downs, on):
        out.append(LightState(l.name, bool(is_on), l.intensity * level * rng.uniform(0.9, 1.1),
                              kelvin_to_rgb(temp), temp, l.dir))
    for l in spec.lights:
        if l.kind == "tube":
            t = rng.uniform(6000.0, 7000.0)
            out.append(LightState(l.name, bool(rng.random() < 0.85), l.intensity * rng.uniform(0.8, 1.2),
                                  kelvin_to_rgb(t), t, l.dir))
    return out


def texture_values(gray: np.ndarray, contrast: float) -> np.ndarray:
    """A pattern (0..1) softened towards 1: contrast 1 keeps it, 0 makes it plain."""
    return 1.0 - contrast * (1.0 - np.clip(gray, 0.0, 1.0))


def look_at_axes(eye, target, up) -> tuple[np.ndarray, np.ndarray]:
    """Camera right and up axes (MuJoCo's `xyaxes`): it looks along -Z, Y up."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, np.asarray(up, float))
    right /= np.linalg.norm(right)
    return right, np.cross(right, f)


__all__ = [
    "Camera", "Light", "LightState", "Material", "Part", "RoomSpec",
    "fixed_lighting", "kelvin_to_rgb", "lab_room", "look_at_axes", "sample_lighting", "texture_values",
    "FLOOR_Z", "CEILING_Z", "ROOM_X", "ROOM_HALF_Y", "DESK_X", "DESK_Y", "HUTCH_SHELF_Z", "WORK_YAW_DEG",
]
