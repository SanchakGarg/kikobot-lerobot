"""Grayscale pattern textures for the desk items, written as PNGs.

Each pattern is a value in 0..1 per texel. The item's shader tints it: 1 is
the body colour, 0 the print colour (`desk_layout.Placement`), so one small
set of files gives every colour combination. Item meshes use u around the
barrel and v along its length (see desk_assets.py), so "stripes" run along a
pen and "bands" ring it.

Pure numpy plus the standard library (zlib) for PNG, so it runs anywhere.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

SIZE = 256


def _uv(size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    t = (np.arange(size) + 0.5) / size
    return np.meshgrid(t, t, indexing="xy")  # u varies along columns, v along rows


def solid(size: int = SIZE) -> np.ndarray:
    return np.ones((size, size))


def stripes(size: int = SIZE, n: int = 6, duty: float = 0.7) -> np.ndarray:
    u, _ = _uv(size)
    return ((u * n) % 1.0 < duty).astype(float)


def bands(size: int = SIZE) -> np.ndarray:
    """Rings near both ends and one in the middle, like a branded barrel."""
    _, v = _uv(size)
    out = np.ones((size, size))
    for lo, hi in ((0.08, 0.11), (0.14, 0.15), (0.48, 0.52), (0.86, 0.89)):
        out[(v >= lo) & (v < hi)] = 0.0
    return out


def grip_dots(size: int = SIZE) -> np.ndarray:
    """Rubber grip: a field of dots over the front quarter of the barrel."""
    u, v = _uv(size)
    out = np.ones((size, size))
    grip = (v > 0.05) & (v < 0.32)
    du, dv = (u * 12) % 1.0 - 0.5, (v * 48) % 1.0 - 0.5
    out[grip & (du**2 + dv**2 < 0.09)] = 0.0
    out[grip & ~(du**2 + dv**2 < 0.09)] = 0.35
    return out


def brand(size: int = SIZE, seed: int = 3) -> np.ndarray:
    """A printed label: a panel with blocky 'lettering' along the barrel."""
    rng = np.random.default_rng(seed)
    u, v = _uv(size)
    out = np.ones((size, size))
    panel = (v > 0.38) & (v < 0.72) & (u > 0.05) & (u < 0.45)
    out[panel] = 0.0
    # Letters: random blocks inside the panel, printed back in body colour.
    v0 = 0.40
    while v0 < 0.70:
        w = rng.uniform(0.012, 0.03)
        h = rng.uniform(0.12, 0.28)
        u0 = 0.08 + rng.uniform(0.0, 0.34 - h)  # letter stays inside the panel (u 0.05..0.45)
        out[(v >= v0) & (v < v0 + w) & (u >= u0) & (u < u0 + h)] = 1.0
        v0 += w + rng.uniform(0.004, 0.012)
    return out


def dots(size: int = SIZE) -> np.ndarray:
    u, v = _uv(size)
    du, dv = (u * 10) % 1.0 - 0.5, (v * 10) % 1.0 - 0.5
    return np.where(du**2 + dv**2 < 0.06, 0.0, 1.0)


def checker(size: int = SIZE, n: int = 8) -> np.ndarray:
    u, v = _uv(size)
    return (((u * n).astype(int) + (v * n).astype(int)) % 2).astype(float)


def speckle(size: int = SIZE, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(1.0 - (rng.random((size, size)) > 0.93) * rng.uniform(0.3, 1.0, (size, size)), 0, 1)


def wood(size: int = 512, seed: int = 11) -> np.ndarray:
    """Wood grain for the desk top: wavy rings plus fine streaks."""
    rng = np.random.default_rng(seed)
    u, v = _uv(size)
    # Smooth low-frequency warp from a few random sines.
    warp = np.zeros_like(u)
    for _ in range(6):
        f, ph, a = rng.uniform(1, 5), rng.uniform(0, 2 * np.pi), rng.uniform(0.01, 0.04)
        warp += a * np.sin(2 * np.pi * f * v + ph)
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * (u + warp) * 18)
    streaks = rng.random(size)[None, :].repeat(size, 0)
    g = 0.65 * rings**3 + 0.35 * streaks
    return np.clip(0.35 + 0.65 * g, 0, 1)


def laminate(size: int = 512, seed: int = 17) -> np.ndarray:
    """Wood-effect desk laminate: fine, nearly straight grain running along v,
    with a few darker growth lines and soft tone drifts. Tiles seamlessly."""
    rng = np.random.default_rng(seed)
    u, v = _uv(size)
    warp = np.zeros_like(u)
    for _ in range(5):  # Whole periods only, so the tile wraps.
        f, ph, a = rng.integers(1, 4), rng.uniform(0, 2 * np.pi), rng.uniform(0.002, 0.008)
        warp += a * np.sin(2 * np.pi * f * v + ph)
    grain = 0.5 + 0.5 * np.sin(2 * np.pi * (u + warp) * 90)
    lines = np.zeros_like(u)
    for _ in range(14):
        c, w = rng.uniform(0, 1), rng.uniform(0.0015, 0.004)
        d = np.abs(((u + warp - c) + 0.5) % 1.0 - 0.5)
        lines = np.maximum(lines, np.exp(-(d / w) ** 2) * rng.uniform(0.4, 1.0))
    tone = 0.5 + 0.5 * np.sin(2 * np.pi * (2 * u + warp * 20) + 1.3)
    g = 1.0 - 0.18 * grain**4 - 0.28 * lines - 0.12 * tone
    return np.clip(g, 0, 1)


PATTERNS = {
    "solid": solid,
    "stripes": stripes,
    "bands": bands,
    "grip_dots": grip_dots,
    "brand": brand,
    "dots": dots,
    "checker": checker,
    "speckle": speckle,
    "wood": wood,
    "laminate": laminate,
}


def write_png(path: Path, gray: np.ndarray) -> None:
    """8-bit grayscale PNG, no dependencies."""
    img = (np.clip(gray, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def ensure_textures(folder: Path) -> dict[str, Path]:
    """Write every pattern into `folder` (once) and return name -> path."""
    out = {}
    for name, fn in PATTERNS.items():
        p = Path(folder) / f"{name}.png"
        if not p.is_file():
            write_png(p, fn())
        out[name] = p
    return out
