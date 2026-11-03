"""`kikobot-servo-web`: a web page to check and set the servos' current and torque limits.

It is a node of its own. With `kikobot-arm-server` running, it connects to it
and works while the arm is being driven: the arm server owns the bus and puts
each request on it between the teleop's goals.

    kikobot-servo-web                              # arm server on this machine
    kikobot-servo-web --arm-host 192.168.1.20      # arm server on another one
    kikobot-servo-web --bus /dev/ttyACM0           # no arm server: open the bus itself

Then open http://127.0.0.1:8080. The page shows each servo's live load,
current, voltage, temperature and protection flags, every limit with the
range it accepts, and sets a limit on one servo or all of them. `--host
0.0.0.0` serves it to other machines; anyone who can reach it can change the
limits, so only do that on a network you trust.

JSON API: GET /api/limits (spec and values), POST /api/limits {servo, key, value}
with `value` raw, as the arm server takes it (`servo_limits.py`).
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

from .protocol import ARM_DEFAULT_PORT, ProtocolError, pack, unpack
from .servo_limits import ServoLimits

logger = logging.getLogger(__name__)

DEFAULT_WEB_PORT = 8080


class ArmServerLimits:
    """Limits through `kikobot-arm-server`; one REQ socket, shared by the page's requests."""

    def __init__(self, host: str, port: int, timeout_s: float = 3.0):
        import zmq

        self._zmq = zmq
        self.address = f"tcp://{host}:{port}"
        self.timeout_s = timeout_s
        self._ctx = zmq.Context()
        self._sock = None
        self._lock = threading.Lock()
        self._open()

    def _open(self) -> None:
        zmq = self._zmq
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, int(self.timeout_s * 1000))
        self._sock.setsockopt(zmq.SNDTIMEO, int(self.timeout_s * 1000))
        self._sock.connect(self.address)

    def _call(self, header: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            try:
                self._sock.send_multipart(pack(header))
                reply, _ = unpack(self._sock.recv_multipart())
            except self._zmq.Again as e:
                self._open()  # A REQ socket that missed its reply is stuck.
                raise ConnectionError(
                    f"No reply from the arm server at {self.address}. Is kikobot-arm-server running?"
                ) from e
        if "error" in reply:
            raise ProtocolError(reply["error"])
        return reply

    def limits(self) -> dict[str, Any]:
        return self._call({"cmd": "limits"})

    def write(self, servo: str | int, key: str, value: int) -> dict[str, Any]:
        return self._call({"cmd": "limits_write", "servo": servo, "key": key, "value": value})

    def close(self) -> None:
        self._sock.close(linger=0)
        self._ctx.term()


class BusLimits:
    """Limits straight on the bus, when no arm server holds it."""

    def __init__(self, limits: ServoLimits):
        self._limits = limits
        self._lock = threading.Lock()

    def limits(self) -> dict[str, Any]:
        with self._lock:
            return {**self._limits.spec(), "values": self._limits.read()}

    def write(self, servo: str | int, key: str, value: int) -> dict[str, Any]:
        with self._lock:
            return self._limits.write(servo, key, value)

    def close(self) -> None:
        self._limits.bus.close()


def page() -> bytes:
    return resources.files(__package__).joinpath("web/servo_limits.html").read_bytes()


def make_handler(backend: Any) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("%s " + fmt, self.address_string(), *args)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, page(), "text/html; charset=utf-8")
            elif self.path == "/api/limits":
                try:
                    self._json(HTTPStatus.OK, backend.limits())
                except Exception as exc:
                    self._json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/api/limits":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                servo, key, value = body["servo"], body["key"], body["value"]
            except (ValueError, KeyError, TypeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"Bad request: {exc}"})
                return
            try:
                result = backend.write(servo, key, value)
            except (ValueError, ProtocolError, RuntimeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:
                self._json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
                return
            logger.info("Set %s of servo %s to %s (raw).", key, result["id"], result["value"])
            self._json(HTTPStatus.OK, result)

    return Handler


def make_server(backend: Any, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(backend))


def _bus_backend(args: argparse.Namespace) -> BusLimits:
    from .feetech import FeetechBus

    names = [*(f"J{i}" for i in range(1, 7)), "gripper"]
    ids = [int(i) for i in args.ids.split(",")]
    if len(ids) > len(names):
        raise SystemExit(f"--ids takes at most {len(names)} IDs (J1..J6, gripper).")
    bus = FeetechBus(args.bus, args.baudrate, ids)
    bus.ping_all()
    return BusLimits(ServoLimits(bus, dict(zip(names, ids))))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="kikobot-servo-web", description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1", help="address to serve the page on (0.0.0.0 for other machines)")
    p.add_argument("--port", type=int, default=DEFAULT_WEB_PORT, help="port to serve the page on")
    p.add_argument("--arm-host", default="127.0.0.1", help="where kikobot-arm-server runs")
    p.add_argument("--arm-port", type=int, default=ARM_DEFAULT_PORT)
    p.add_argument("--bus", default=None,
                   help="servo port to open directly instead of using the arm server (only with no arm server running)")
    p.add_argument("--baudrate", type=int, default=1_000_000)
    p.add_argument("--ids", default="1,2,3,4,5,6,7", help="with --bus: bus IDs of J1..J6 and the gripper")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Starting the servo limits page...")

    if args.bus:
        backend: Any = _bus_backend(args)
        source = f"the bus on {args.bus}"
    else:
        backend = ArmServerLimits(args.arm_host, args.arm_port)
        source = f"the arm server at {backend.address}"
        try:
            backend.limits()
        except Exception as exc:
            logger.warning("%s; the page will keep trying.", exc)
    server = make_server(backend, args.host, args.port)
    logger.info("Servo limits for %s: http://%s:%d", source, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        backend.close()


if __name__ == "__main__":
    main()
