"""The desk sorting station's layout sampler, textures and meshes (no Isaac Sim)."""

from __future__ import annotations

import importlib.util
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
# Loaded by path and bare name, the way the Isaac server loads them.
FILES = {
    "desk_layout": REPO / "lerobot_robot_kikobot" / "desk" / "layout.py",
    "desk_textures": REPO / "lerobot_robot_kikobot" / "desk" / "textures.py",
    "desk_meshes": REPO / "lerobot_robot_kikobot" / "desk" / "meshes.py",
    "desk_assets": REPO / "isaac" / "desk_assets.py",
}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, FILES[name])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L = _load("desk_layout")
T = _load("desk_textures")
_load("desk_meshes")
A = _load("desk_assets")


def _layouts(n=200, **kw):
    pool = L.default_pool()
    return pool, [L.sample_desk(np.random.default_rng(s), pool, **kw) for s in range(n)]


def test_same_seed_same_desk():
    pool = L.default_pool()
    a = L.sample_desk(np.random.default_rng(7), pool)
    b = L.sample_desk(np.random.default_rng(7), pool)
    assert [(p.pos, p.yaw, p.texture, p.body_rgb) for p in a] == [(p.pos, p.yaw, p.texture, p.body_rgb) for p in b]
    c = L.sample_desk(np.random.default_rng(8), pool)
    assert [p.pos for p in a] != [p.pos for p in c]


def test_one_placement_per_item_and_counts():
    pool, layouts = _layouts()
    for lay in layouts:
        assert [p.name for p in lay] == [it.name for it in pool]
        w = sum(p.active and L.KINDS[p.kind].writing for p in lay)
        s = sum(p.active and not L.KINDS[p.kind].writing for p in lay)
        # The arm's corner leaves room for a small desk: 2-4 writing tools, 1-2 supplies.
        assert 2 <= w <= 4 and 1 <= s <= 2


def test_active_items_in_reach_and_parked_ones_out_of_the_way():
    _, layouts = _layouts()
    for lay in layouts:
        for p in lay:
            x, y, z = p.pos
            if p.active:
                assert L._in_fan(np.array([x, y]))
                assert z > 0
            else:
                assert z < L.FLOOR_Z + 0.05
                assert L.PARK_X[0] <= x <= L.PARK_X[1]


def _shape(p, pool_by_name):
    it = pool_by_name[p.name]
    hl, r = it.footprint()
    p0, p1 = L._segment(p.pos[:2], p.yaw, hl)
    return p0, p1, r


def test_items_on_the_desk_do_not_intersect():
    """Two items at the same layer never overlap; stations and the base stay clear."""
    pool, layouts = _layouts()
    by_name = {it.name: it for it in pool}
    stations = L._station_shapes()
    for lay in layouts:
        on = [p for p in lay if p.active]
        for i, a in enumerate(on):
            a0, a1, ar = _shape(a, by_name)
            for st in stations:
                assert L.segment_distance(a0, a1, st.p0, st.p1) - ar - st.radius >= L.CLEARANCE_M - 1e-9
            for b in on[i + 1:]:
                b0, b1, br = _shape(b, by_name)
                touching = L.segment_distance(a0, a1, b0, b1) - ar - br < L.CLEARANCE_M - 1e-9
                if touching:
                    # Only pens stack, and then one is above the other.
                    assert L.KINDS[a.kind].writing and L.KINDS[b.kind].writing
                    assert a.layer != b.layer
                    lo, hi = sorted((a, b), key=lambda p: p.layer)
                    assert hi.pos[2] > lo.pos[2]


def test_pens_stack_on_top_of_each_other():
    _, layouts = _layouts(stack_p=1.0, n_writing=(8, 8))
    stacked = [p for lay in layouts for p in lay if p.layer > 0]
    assert len(stacked) > 100
    for p in stacked:
        assert L.KINDS[p.kind].writing
        assert p.on_top_of and p.layer < L.MAX_STACK
        assert p.pos[2] > L.KINDS[p.kind].half[2] + L.STACK_GAP_M - 1e-9


def test_no_stacking_when_disabled():
    _, layouts = _layouts(stack_p=0.0)
    assert all(p.layer == 0 for lay in layouts for p in lay)


def test_looks_use_known_textures_and_colours():
    _, layouts = _layouts(50)
    for lay in layouts:
        for p in lay:
            assert p.texture in T.PATTERNS
            assert all(0.0 <= c <= 1.0 for c in (*p.body_rgb, *p.print_rgb, *p.accent_rgb))
            assert 0.0 <= p.roughness <= 1.0 and 0.0 <= p.metallic <= 1.0


def test_segment_distance():
    assert L.segment_distance((0, 0), (1, 0), (0.5, -1), (0.5, 1)) == 0.0
    assert L.segment_distance((0, 0), (1, 0), (0, 1), (1, 1)) == pytest.approx(1.0)
    assert L.segment_distance((0, 0), (0, 0), (3, 4), (3, 4)) == pytest.approx(5.0)


def test_textures_are_valid_pngs(tmp_path):
    paths = T.ensure_textures(tmp_path)
    assert set(paths) == set(T.PATTERNS)
    for name, path in paths.items():
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        w, h = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        assert w == h and w in (256, 512)
        # Decode IDAT: one filter byte per row plus w gray bytes.
        idat = data[data.index(b"IDAT") + 4:data.index(b"IEND") - 8]
        assert len(zlib.decompress(idat)) == h * (w + 1)
        pattern = T.PATTERNS[name]()
        assert pattern.min() >= 0 and pattern.max() <= 1


SIZES = [(kind, half) for kind, spec in L.KINDS.items() for half in spec.all_sizes()]


@pytest.mark.parametrize(("kind", "half"), SIZES)
def test_procedural_items_fit_their_colliders(kind, half):
    """Visual meshes sit on the desk where the collider does: same bottom, about the same size."""
    half = np.array(half)
    pts = np.concatenate([m.arrays()[0] for _, m in A.procedural_item(kind, tuple(half))])
    lo, hi = pts.min(0), pts.max(0)
    assert lo[2] == pytest.approx(-half[2], abs=0.0015)
    assert hi[0] - lo[0] == pytest.approx(2 * half[0], rel=0.08)
    assert np.all(hi - lo < 2 * half + 0.008)


def test_every_size_is_workable():
    """Each size fits the jaws, its station and the stations' rules."""
    jaws = 0.0536
    for kind, half in SIZES:
        hx, hy, hz = half
        st = L.station_for(kind)
        if kind == "tape_roll":
            assert 2 * hx <= 0.046, half  # Gripped round the rim, the jaws wide open.
        elif kind not in ("scissors", "sticky_notes"):
            assert 2 * hy + 0.016 <= jaws, (kind, half)  # Gripped across, with clearance.
        if st.name == "supply_tray":
            # Laid along it.
            assert hx < st.half[0] - 0.01 and hy < st.half[1] - 0.004, (kind, half)
        elif st.name == "pen_cup":
            r_in = next(p for p in st.parts if p.name == "visual").half[0]
            assert max(hy, hz) < r_in - 0.01, (kind, half)


def test_pool_spreads_the_sizes():
    pool = L.default_pool()
    for kind, spec in L.KINDS.items():
        have = {it.half for it in pool if it.kind == kind}
        n = sum(it.kind == kind for it in pool)
        assert len(have) == min(n, len(spec.all_sizes())), kind
    big = max((it for it in pool if it.kind == "glue_stick"), key=lambda it: it.half[0])
    small = min((it for it in pool if it.kind == "glue_stick"), key=lambda it: it.half[0])
    assert big.mass > small.mass


def test_meshes_face_outward():
    """Every face's winding agrees with its normals (no inside-out faces)."""
    for kind, half in SIZES:
        for _, m in A.procedural_item(kind, half):
            pts, nrm, _, counts = m.arrays()
            i = 0
            for c in counts:
                poly = pts[i:i + c]
                n = np.zeros(3)
                for k in range(c):
                    a, b = poly[k], poly[(k + 1) % c]
                    n += np.cross(a, b)
                if np.linalg.norm(n) > 1e-12:
                    assert n @ nrm[i:i + c].sum(0) > 0, kind
                i += c


def test_asset_name_matching():
    assert A.match_kind("040_large_marker.usd") == "marker"
    assert A.match_kind("037_scissors.usd") == "scissors"
    assert A.match_kind("SM_BallPen_01.usdc") == "pen"
    assert A.match_kind("pencil_hb.usda") == "pencil"
    assert A.match_kind("open_box.usd") is None
    assert A.rank("a/Axis_Aligned_Physics/x.usd") > A.rank("a/Axis_Aligned/x.usd")


def test_stations_are_on_the_desk_and_in_reach():
    from lerobot_robot_kikobot.env import room as R

    (x0, x1), (y0, y1) = R.DESK_X, R.DESK_Y
    for st in L.STATIONS:
        r = math.hypot(*st.xy)
        assert 0.2 < r < 0.33, st.name
        # Every corner of its footprint on the desk, clear of the hutch's side panels.
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        for a in (-1, 1):
            for b in (-1, 1):
                x = st.xy[0] + c * a * st.half[0] - s * b * st.half[1]
                y = st.xy[1] + s * a * st.half[0] + c * b * st.half[1]
                assert x0 + R.HUTCH_PANEL_T < x < x1 - R.HUTCH_PANEL_T and y0 < y < y1, st.name
        # Out where the arm faces at rest, clear of the left side panel.
        assert st.xy[0] - max(st.half) > 0.05, st.name
        # Not in the arm's blind spot, where J1 runs out of range.
        assert not -100.0 < math.degrees(math.atan2(st.xy[1], st.xy[0])) < -40.0, st.name


def test_desk_bounds_agree_with_the_room():
    from lerobot_robot_kikobot.env import room as R

    assert np.allclose(L.DESK_X, R.DESK_X) and np.allclose(L.DESK_Y, R.DESK_Y)


def test_layout_agrees_with_the_room():
    from lerobot_robot_kikobot.env import room as R

    assert L.WORK_YAW_DEG == R.WORK_YAW_DEG
    assert L.FLOOR_Z == R.FLOOR_Z


def test_every_kind_has_a_station():
    for kind in L.KINDS:
        st = L.station_for(kind)
        assert kind in st.accepts
    accepted = [k for st in L.STATIONS for k in st.accepts]
    assert len(accepted) == len(set(accepted)) == len(L.KINDS)


def _station_outline(st, n=24):
    """Points round a station's footprint: a circle for a round one, else its box."""
    if st.half[0] == st.half[1] and any(p.shape in ("tube", "cylinder") for p in st.parts):
        return [(st.xy[0] + st.half[0] * math.cos(t), st.xy[1] + st.half[0] * math.sin(t))
                for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    c, s = math.cos(st.yaw), math.sin(st.yaw)
    out = []
    for u in np.linspace(-1, 1, 7):
        for v in (-1, 1):
            for a, b in ((u, v), (v, u)):
                out.append((st.xy[0] + c * a * st.half[0] - s * b * st.half[1],
                            st.xy[1] + s * a * st.half[0] + c * b * st.half[1]))
    return out


def _inside(st, p, pad):
    d = np.asarray(p) - np.asarray(st.xy)
    if st.half[0] == st.half[1] and any(q.shape in ("tube", "cylinder") for q in st.parts):
        return np.hypot(*d) < st.half[0] + pad
    c, s = math.cos(st.yaw), math.sin(st.yaw)
    lx, ly = c * d[0] + s * d[1], -s * d[0] + c * d[1]
    return abs(lx) < st.half[0] + pad and abs(ly) < st.half[1] + pad


def test_stations_do_not_overlap():
    for i, a in enumerate(L.STATIONS):
        for b in L.STATIONS[i + 1 :]:
            assert not any(_inside(b, p, 0.005) for p in _station_outline(a)), (a.name, b.name)
        # Clear of the arm's 20 cm square base plate.
        for p in _station_outline(a):
            assert max(abs(p[0]), abs(p[1])) > 0.105, a.name


def test_station_parts_sit_on_the_desk():
    for st in L.STATIONS:
        assert st.parts
        for p in st.parts:
            assert p.center[2] - p.half[2] >= -1e-9, (st.name, p.name)
            assert p.shape in ("box", "cylinder", "tube")


def test_tape_rolls_fit_over_the_peg():
    post = next(p for p in L.TAPE_PEG.parts if p.name == "post")
    assert L.KINDS["tape_roll"].ring_inner > post.half[0] + 0.002
