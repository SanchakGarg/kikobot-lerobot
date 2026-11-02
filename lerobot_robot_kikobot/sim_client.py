"""The LeRobot side's connection to `kikobot-sim-server` (see `protocol.py`)."""

from __future__ import annotations

import threading
from typing import Any

from .protocol import PROTOCOL_VERSION, ProtocolError, pack, unpack


class SimClient:
    """One REQ socket to the sim server, with a timeout on every reply."""

    def __init__(self, host: str, port: int, timeout_s: float):
        import zmq

        self._zmq = zmq
        self.address = f"tcp://{host}:{port}"
        self.timeout_s = timeout_s
        self._ctx = zmq.Context()
        self._sock = None
        self._open()

    def _open(self) -> None:
        zmq = self._zmq
        if self._sock is not None:
            self._sock.close(linger=0)
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, int(self.timeout_s * 1000))
        sock.setsockopt(zmq.SNDTIMEO, int(self.timeout_s * 1000))
        sock.connect(self.address)
        self._sock = sock

    def set_timeout(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self._open()

    def call(self, header: dict[str, Any]) -> tuple[dict[str, Any], list[bytes]]:
        try:
            self._sock.send_multipart(pack(header))
            reply, frames = unpack(self._sock.recv_multipart())
        except self._zmq.Again as e:
            # A REQ socket that missed its reply is stuck; reopen it so the
            # next call goes through once the server is back.
            self._open()
            raise TimeoutError(
                f"No reply from the sim server at {self.address} to {header['cmd']!r}. "
                "Is the sim server running (kikobot-sim-server, or the Isaac server)?"
            ) from e
        if "error" in reply:
            raise ProtocolError(f"Sim server: {reply['error']}")
        return reply, frames

    def hello(self) -> dict[str, Any]:
        info, _ = self.call({"cmd": "hello", "protocol": PROTOCOL_VERSION})
        if info.get("protocol") != PROTOCOL_VERSION:
            raise ProtocolError(
                f"The sim server speaks protocol {info.get('protocol')}, this plugin "
                f"{PROTOCOL_VERSION}. Restart the server from the same checkout."
            )
        return info

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        self._ctx.term()


class SimMirror:
    """Sends the real arm's commands to a sim server from a thread of its own.

    `send` only stores the latest command and returns at once, so a slow or
    dead sim server can never hold up the real arm's control loop. Commands
    the thread had no time to send are skipped: only the latest one matters.
    After a failure it waits `retry_s` before trying again.
    """

    def __init__(self, client: SimClient, retry_s: float = 2.0):
        self.client = client
        self.retry_s = retry_s
        self.error: Exception | None = None
        self.failures = 0
        self._latest: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="kikobot-sim-mirror", daemon=True)
        self._thread.start()

    def send(self, header: dict[str, Any]) -> None:
        with self._lock:
            self._latest = header
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                header, self._latest = self._latest, None
            if header is None:
                continue
            try:
                self.client.call(header)
                self.error = None
            except Exception as exc:  # Reported by the owner; retried after a pause.
                self.error = exc
                self.failures += 1
                self._stop.wait(self.retry_s)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2.0)
        self.client.close()
