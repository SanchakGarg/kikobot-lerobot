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
        assert 1 <= w <= 8 and 1 <= s <= 4


def test_active_items_in_reach_and_parked_ones_under_the_table():
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


@pytest.mark.parametrize("kind", list(L.KINDS))
def test_procedural_items_fit_their_colliders(kind):
    """Visual meshes sit on the desk where the collider does: same bottom, about the same size."""
    half = np.array(L.KINDS[kind].half)
    pts = np.concatenate([m.arrays()[0] for _, m in A.procedural_item(kind)])
    lo, hi = pts.min(0), pts.max(0)
    assert lo[2] == pytest.approx(-half[2], abs=0.0015)
    assert hi[0] - lo[0] == pytest.approx(2 * half[0], rel=0.08)
    assert np.all(hi - lo < 2 * half + 0.008)


def test_meshes_face_outward():
    """Every face's winding agrees with its normals (no inside-out faces)."""
    for kind in L.KINDS:
        for _, m in A.procedural_item(kind):
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


def test_stations_are_inside_the_table_and_reach():
    for st in L.STATIONS:
        r = math.hypot(*st.xy)
        assert 0.15 < r < 0.33
        assert all(abs(c) + max(st.half) < 0.5 for c in st.xy)
