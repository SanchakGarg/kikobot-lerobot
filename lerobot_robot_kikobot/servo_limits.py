"""Current, torque and safety limits of the Feetech STS servos, read and set on a live bus.

`ServoLimits` wraps a `FeetechBus` (or anything with `read_block` and
`write_register`). `kikobot-arm-server` answers `limits` / `limits_write`
requests with it, and `kikobot-servo-web` shows them on a web page, so the
limits can be checked and changed while the arm is being driven.

Register addresses, sizes and units are the STS3215 / STS3250 control table's.
Values travel raw (the servo's own integers); each `Register` says how to turn
one into a unit (`raw * scale`) and which raw range it accepts. EEPROM limits
survive a power cycle; `torque_limit` is RAM: it starts from `max_torque` at
power-up and changes the servo's torque cap at once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .feetech import _sign_magnitude

LIMIT, LIVE, INFO = "limit", "live", "info"


@dataclass(frozen=True)
class Register:
    key: str
    label: str
    address: int
    size: int
    group: str  # limit (settable), live (telemetry) or info (read-only setting)
    eeprom: bool = False
    lo: int = 0
    hi: int = 0
    scale: float = 1.0
    unit: str = ""
    sign_bit: int | None = None
    help: str = ""

    @property
    def writable(self) -> bool:
        return self.group == LIMIT


REGISTERS: tuple[Register, ...] = (
    # Limits.
    Register("torque_limit", "Torque limit (live)", 48, 2, LIMIT, False, 0, 1000, 0.1, "%",
             help="The torque cap in force now, in % of stall torque. Applies at once; "
                  "back to Max torque at the next power-up."),
    Register("max_torque", "Max torque", 16, 2, LIMIT, True, 0, 1000, 0.1, "%",
             help="Torque cap loaded into Torque limit at power-up (EEPROM)."),
    Register("protection_current", "Protection current", 28, 2, LIMIT, True, 0, 511, 6.5, "mA",
             help="Current above which over-current protection starts timing."),
    Register("overcurrent_time", "Over-current time", 38, 1, LIMIT, True, 0, 254, 10, "ms",
             help="How long the current may stay above Protection current before the servo cuts torque."),
    Register("overload_torque", "Overload torque", 36, 1, LIMIT, True, 0, 100, 1, "%",
             help="Load above which overload protection starts timing."),
    Register("protection_time", "Overload time", 35, 1, LIMIT, True, 0, 254, 10, "ms",
             help="How long the load may stay above Overload torque before protection trips."),
    Register("protective_torque", "Protective torque", 34, 1, LIMIT, True, 0, 100, 1, "%",
             help="Torque the servo falls back to once overload protection has tripped."),
    Register("max_temperature", "Max temperature", 13, 1, LIMIT, True, 0, 100, 1, "°C",
             help="Over-temperature protection threshold."),
    Register("max_voltage", "Max voltage", 14, 1, LIMIT, True, 0, 254, 0.1, "V",
             help="Over-voltage protection threshold."),
    Register("min_voltage", "Min voltage", 15, 1, LIMIT, True, 0, 254, 0.1, "V",
             help="Under-voltage protection threshold."),
    # Read-only settings.
    Register("unloading", "Protections on", 19, 1, INFO,
             help="Bit mask of protections that cut torque: 1 voltage, 4 temperature, 8 current, 32 overload."),
    Register("min_angle", "Min angle", 9, 2, INFO, unit="steps",
             help="Servo's own angle limit. Calibration relies on it: left as it is."),
    Register("max_angle", "Max angle", 11, 2, INFO, unit="steps",
             help="Servo's own angle limit. Calibration relies on it: left as it is."),
    # Telemetry.
    Register("load", "Load", 60, 2, LIVE, scale=0.1, unit="%", sign_bit=10,
             help="Present load, in % of stall torque; the sign is the direction."),
    Register("current", "Current", 69, 2, LIVE, scale=6.5, unit="mA", sign_bit=15),
    Register("voltage", "Voltage", 62, 1, LIVE, scale=0.1, unit="V"),
    Register("temperature", "Temperature", 63, 1, LIVE, unit="°C"),
    Register("status", "Status", 65, 1, LIVE,
             help="Protection flags set: 1 voltage, 2 angle sensor, 4 temperature, 8 current, 32 overload."),
)
BY_KEY = {r.key: r for r in REGISTERS}

# Two reads per servo cover every register: the EEPROM limits, then RAM.
BLOCKS = ((9, 30), (48, 23))


def decode(reg: Register, block_start: int, data: list[int]) -> int:
    i = reg.address - block_start
    raw = data[i] if reg.size == 1 else data[i] | data[i + 1] << 8
    return _sign_magnitude(raw, reg.sign_bit) if reg.sign_bit is not None else raw


class ServoLimits:
    """Read every limit and telemetry register of the arm's servos; set the limits."""

    def __init__(self, bus: Any, motor_ids: Mapping[str, int]):
        self.bus = bus
        self.motor_ids = dict(motor_ids)

    def spec(self) -> dict[str, Any]:
        return {
            "servos": [{"name": n, "id": i} for n, i in self.motor_ids.items()],
            "registers": [{**asdict(r), "writable": r.writable} for r in REGISTERS],
        }

    def read_servo(self, servo_id: int) -> dict[str, int]:
        blocks = [(start, self.bus.read_block(servo_id, start, n)) for start, n in BLOCKS]
        values = {}
        for reg in REGISTERS:
            start, data = next((s, d) for s, d in blocks if s <= reg.address < s + len(d))
            values[reg.key] = decode(reg, start, data)
        return values

    def read(self) -> dict[str, Any]:
        """Every servo's registers, raw. A servo that does not answer gets an `error`."""
        servos = []
        for name, servo_id in self.motor_ids.items():
            entry: dict[str, Any] = {"name": name, "id": servo_id}
            try:
                entry["values"] = self.read_servo(servo_id)
            except Exception as exc:
                entry["error"] = str(exc)
            servos.append(entry)
        return {"servos": servos}

    def write(self, servo: str | int, key: str, value: int) -> dict[str, Any]:
        """Set one limit on one servo (by name or bus ID), raw, and read it back."""
        servo_id = self.motor_ids.get(servo, servo) if isinstance(servo, str) else int(servo)
        if servo_id not in self.motor_ids.values():
            raise ValueError(f"No servo {servo!r} on this arm: {self.motor_ids}.")
        reg = BY_KEY.get(key)
        if reg is None or not reg.writable:
            raise ValueError(f"{key!r} is not a settable limit.")
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{reg.label} takes a whole number of raw steps, not {value!r}.")
        value = int(value)
        if not reg.lo <= value <= reg.hi:
            raise ValueError(
                f"{reg.label} must be {reg.lo}..{reg.hi} raw "
                f"({reg.lo * reg.scale:g}..{reg.hi * reg.scale:g} {reg.unit}), not {value}."
            )
        self.bus.write_register(servo_id, reg.address, reg.size, value, reg.eeprom)
        got = decode(reg, reg.address, self.bus.read_block(servo_id, reg.address, reg.size))
        if got != value:
            raise RuntimeError(
                f"Servo {servo_id} kept {reg.label} at {got} after writing {value}: "
                "its firmware refused it (some refuse EEPROM writes with torque on)."
            )
        return {"id": servo_id, "key": key, "value": got}
