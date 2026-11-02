"""Wire protocol between the LeRobot side and the Kiko-Bot sim servers.

A sim server runs the physics in its own process, in real time, and shows it
in a window. The LeRobot robots connect to it over one ZMQ REQ/REP socket:
`kikobot_sim` and `kikobot_isaac` to drive and observe the simulated arm,
and `kikobot_follower` to mirror every command it sends the real servos, so
the sim and the physical arm move together. Every message is multipart: a
JSON header first, then zero or more binary frames (camera images). No pickle.

Requests (client -> server), all with a "cmd" key:

    hello    {protocol}                    -> {protocol, joints, lower_deg, upper_deg, scene, engine, ...}
    command  {q_deg: [6], gripper_pct}     -> {state}    new actuator targets, applied at once
    pose     {q_deg: [6], gripper_pct}     -> {state}    teleport the arm there and hold it
    observe  {cameras, width, height, encoding} -> observation
    reset    {ready_deg: [6]}              -> {state}    arm to ready, scene re-randomized
    close    {}                            -> {}

The MuJoCo server (`sim_server.py`) and the Isaac Sim server
(`isaac/kikobot_isaac_server.py`) both speak it, so `kikobot_sim`,
`kikobot_isaac` and the real arm's mirror share one client (`sim_client.py`).
The Isaac server's scene, cameras and image size are fixed at its start; the
MuJoCo server renders any size on request.

`state` is {"J1.pos": deg, ..., "J6.pos": deg, "gripper.pos": % open}, the
simulated arm's measured joints. An observation reply is the header {"state":
..., "images": [{"name", "shape", "encoding"}, ...]} followed by one frame per
image, in the same order. Errors come back as {"error": "message"}.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

PROTOCOL_VERSION = 2
# The arm's joints, in the order every message and action uses.
ARM_JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")
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
