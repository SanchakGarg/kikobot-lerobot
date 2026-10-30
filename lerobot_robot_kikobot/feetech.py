"""Minimal Feetech STS bus: group sync read/write over `scservo_sdk`.

Only what streaming needs. One sync-write packet sets every servo's goal and
gets no reply, so it costs well under a millisecond at 1 Mbps. Writing the
servos one by one and waiting for each reply is what made the old bridge slow.

Register addresses are from the STS3215 / STS3250 control table.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# EEPROM (persist; only written when the value differs).
ADDR_RETURN_DELAY = 7
ADDR_P_COEFFICIENT = 21
ADDR_D_COEFFICIENT = 22
ADDR_I_COEFFICIENT = 23
# RAM.
ADDR_TORQUE_ENABLE = 40
ADDR_ACCELERATION = 41
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_LOCK = 55
ADDR_PRESENT_POSITION = 56

RESOLUTION = 4096
# Attempts per request/reply transaction. A reply is occasionally lost on a
# busy half-duplex bus; one retry almost always gets it.
RETRIES = 3


def _le16(value: int) -> list[int]:
    return [value & 0xFF, (value >> 8) & 0xFF]


def _sign_magnitude(raw: int, sign_bit: int = 15) -> int:
    """Present_Position carries direction in bit 15, not two's complement."""
    return -(raw & ~(1 << sign_bit)) if raw & (1 << sign_bit) else raw


class FeetechBus:
    def __init__(self, port: str, baudrate: int, ids: Sequence[int]):
        try:
            import scservo_sdk as scs
        except ImportError as exc:
            raise ImportError(
                "The real arm needs the Feetech SDK: pip install feetech-servo-sdk"
            ) from exc
        self.scs = scs
        self.ids = list(ids)
        # Cleared on the first failed sync read if single reads still work:
        # some servo firmware does not answer the sync-read instruction.
        self.sync_read_supported = True
        self.port = scs.PortHandler(port)
        self.packet = scs.PacketHandler(0)  # 0: STS little-endian protocol.
        if not self.port.openPort():
            raise ConnectionError(
                f"Could not open {port}. Check the cable and the port name "
                "(`lerobot-find-port`); on Linux also that you are in the 'dialout' group."
            )
        if not self.port.setBaudRate(baudrate):
            self.port.closePort()
            raise ConnectionError(f"Could not set {port} to {baudrate} baud.")
        self._writer = scs.GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, 2)
        self._reader = scs.GroupSyncRead(self.port, self.packet, ADDR_PRESENT_POSITION, 2)
        for i in self.ids:
            self._reader.addParam(i)

    def close(self) -> None:
        self.port.closePort()

    def _check(self, result: int, error: int, what: str) -> None:
        if result != self.scs.COMM_SUCCESS:
            raise ConnectionError(f"{what}: {self.packet.getTxRxResult(result)}")
        if error:
            logger.warning("%s: servo reported %s", what, self.packet.getRxPacketError(error))

    def _txrx(self, request, *args):
        """Run a request/reply transaction, retrying lost replies."""
        for _ in range(RETRIES):
            out = request(self.port, *args)
            if out[-2] == self.scs.COMM_SUCCESS:
                break
        return out

    def ping_all(self) -> None:
        missing = []
        for i in self.ids:
            _, result, _ = self._txrx(self.packet.ping, i)
            if result != self.scs.COMM_SUCCESS:
                missing.append(i)
        if missing:
            raise ConnectionError(
                f"Servos {missing} did not answer. Check power (the bus needs 12 V), "
                "wiring and IDs (`--robot.servo_ids`)."
            )

    def read_byte(self, servo_id: int, address: int) -> int:
        value, result, error = self._txrx(self.packet.read1ByteTxRx, servo_id, address)
        self._check(result, error, f"read servo {servo_id} @{address}")
        return value

    def write_byte(self, servo_id: int, address: int, value: int) -> None:
        result, error = self._txrx(self.packet.write1ByteTxRx, servo_id, address, value)
        self._check(result, error, f"write servo {servo_id} @{address}")

    def write_word(self, servo_id: int, address: int, value: int) -> None:
        result, error = self._txrx(self.packet.write2ByteTxRx, servo_id, address, value)
        self._check(result, error, f"write servo {servo_id} @{address}")

    def set_eeprom_byte(self, servo_id: int, address: int, value: int) -> None:
        """Write an EEPROM register only if it differs, to spare the flash."""
        if self.read_byte(servo_id, address) == value:
            return
        self.write_byte(servo_id, ADDR_LOCK, 0)
        self.write_byte(servo_id, address, value)
        self.write_byte(servo_id, ADDR_LOCK, 1)

    def set_torque(self, enabled: bool) -> None:
        for i in self.ids:
            self.write_byte(i, ADDR_TORQUE_ENABLE, int(enabled))

    def sync_read_positions(self) -> dict[int, int]:
        if self.sync_read_supported:
            problem = ""
            for _ in range(RETRIES):
                result = self._reader.txRxPacket()
                if result != self.scs.COMM_SUCCESS:
                    problem = self.packet.getTxRxResult(result)
                    continue
                missing = [i for i in self.ids if not self._reader.isAvailable(i, ADDR_PRESENT_POSITION, 2)]
                if not missing:
                    return {
                        i: _sign_magnitude(self._reader.getData(i, ADDR_PRESENT_POSITION, 2)) for i in self.ids
                    }
                problem = f"no data from servos {missing}"
            # Fall back to one read per servo if that works; slower but fine
            # at the default read rate.
            out = self.read_positions()
            self.sync_read_supported = False
            logger.warning("Sync read failed (%s); reading servos one at a time instead.", problem)
            return out
        return self.read_positions()

    def read_positions(self) -> dict[int, int]:
        out = {}
        for i in self.ids:
            value, result, error = self._txrx(self.packet.read2ByteTxRx, i, ADDR_PRESENT_POSITION)
            self._check(result, error, f"read servo {i} position")
            out[i] = _sign_magnitude(value)
        return out

    def sync_write_goals(self, goals: dict[int, int]) -> None:
        self._writer.clearParam()
        for i, raw in goals.items():
            self._writer.addParam(i, _le16(int(min(max(raw, 0), RESOLUTION - 1))))
        result = self._writer.txPacket()
        if result != self.scs.COMM_SUCCESS:
            raise ConnectionError(f"sync write: {self.packet.getTxRxResult(result)}")
