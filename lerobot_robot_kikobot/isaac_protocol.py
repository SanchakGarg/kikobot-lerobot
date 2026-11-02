"""Wire protocol between the `kikobot_isaac` robot and the Isaac Sim server.

Isaac Sim runs in its own process (and its own Python environment), the
LeRobot side in another, possibly on another machine. They talk over one ZMQ
REQ/REP socket. Every message is multipart: a JSON header first, then zero or
more binary frames (camera images). No pickle, so neither side runs code the
other sends.

Requests (client -> server), all with a "cmd" key:

    hello    {fps, width, height, cameras, scene, seed, motion, image_encoding}
             -> {protocol, joints, lower_deg, upper_deg, engine}
    reset    {ready_deg: [6 floats]}                      -> {"state"}
    step     {action: {"J1.pos": deg, ..., "gripper.pos": %}} -> {"state"}
    observe  {}                                           -> observation
    close    {}                                           -> {}

`step` advances physics by one control period (1/fps) and renders the
cameras once; `observe` returns the latest state and those frames without
stepping. LeRobot calls `get_observation` once and `send_action` once per
tick, so each tick costs one render and the images cross the wire once.

An observation reply is the header {"state": {"J1.pos": ..., "gripper.pos":
...}, "images": [{"name", "shape", "encoding"}, ...]} followed by one frame
per image, in the same order. Errors come back as {"error": "message"}.

This module only needs numpy and pyzmq (plus OpenCV for JPEG), so the server
loads it by path under Isaac Sim's Python without importing LeRobot.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1
DEFAULT_PORT = 5556
IMAGE_ENCODINGS = ("raw", "jpeg")
JPEG_QUALITY = 90


class ProtocolError(RuntimeError):
    """The server answered with an error, or something that is not the protocol."""


def pack(header: dict[str, Any], frames: list[bytes] | None = None) -> list[bytes]:
    return [json.dumps(header).encode(), *(frames or [])]


def unpack(parts: list[bytes]) -> tuple[dict[str, Any], list[bytes]]:
    if not parts:
        raise ProtocolError("Empty message.")
    try:
        header = json.loads(parts[0].decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"Bad message header: {e}") from e
    if not isinstance(header, dict):
        raise ProtocolError("Message header is not a JSON object.")
    return header, list(parts[1:])


def encode_image(img: np.ndarray, encoding: str) -> tuple[dict[str, Any], bytes]:
    """HxWx3 uint8 RGB -> (meta, bytes)."""
    img = np.ascontiguousarray(img[..., :3], dtype=np.uint8)
    if encoding == "raw":
        return {"shape": list(img.shape), "encoding": "raw"}, img.tobytes()
    if encoding == "jpeg":
        import cv2

        ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise ProtocolError("JPEG encoding failed.")
        return {"shape": list(img.shape), "encoding": "jpeg"}, buf.tobytes()
    raise ValueError(f"image_encoding must be one of {IMAGE_ENCODINGS}, got {encoding!r}")


def decode_image(meta: dict[str, Any], data: bytes) -> np.ndarray:
    shape = tuple(meta["shape"])
    if meta["encoding"] == "raw":
        return np.frombuffer(data, dtype=np.uint8).reshape(shape).copy()
    if meta["encoding"] == "jpeg":
        import cv2

        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ProtocolError("JPEG decoding failed.")
        return bgr[..., ::-1].copy()
    raise ProtocolError(f"Unknown image encoding {meta['encoding']!r}.")


def pack_observation(
    state: dict[str, float], images: dict[str, np.ndarray], encoding: str
) -> list[bytes]:
    metas, frames = [], []
    for name, img in images.items():
        meta, data = encode_image(img, encoding)
        metas.append({"name": name, **meta})
        frames.append(data)
    return pack({"state": state, "images": metas}, frames)


def unpack_observation(header: dict[str, Any], frames: list[bytes]) -> dict[str, Any]:
    metas = header.get("images", [])
    if len(metas) != len(frames):
        raise ProtocolError(f"{len(metas)} image headers but {len(frames)} image frames.")
    obs: dict[str, Any] = {k: float(v) for k, v in header["state"].items()}
    for meta, data in zip(metas, frames):
        obs[meta["name"]] = decode_image(meta, data)
    return obs
