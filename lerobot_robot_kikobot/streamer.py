"""Background thread that streams shaped goal positions to the servos.

The LeRobot loop only drops the latest target into `set_target`; this thread
owns the serial port and, at a fixed rate, advances the motion shaper and
sync-writes the result. The servos therefore get a goal every 10 ms that is
a tiny, continuous step ahead of where they are, however irregular the 30 Hz
loop (or the policy behind it) is.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from .feetech import FeetechBus
from .motion import MotionShaper

logger = logging.getLogger(__name__)

# Consecutive failed reads tolerated before the stream gives up.
MAX_READ_FAILURES = 25


class ServoStreamer:
    def __init__(
        self,
        bus: FeetechBus,
        shaper: MotionShaper,
        to_raw: Callable[[np.ndarray], dict[int, int]],
        from_raw: Callable[[dict[int, int]], np.ndarray],
        rate_hz: float,
        read_every: int,
    ):
        self.bus = bus
        self.shaper = shaper
        self.to_raw = to_raw
        self.from_raw = from_raw
        self.period = 1.0 / rate_hz
        self.read_every = max(1, read_every)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._new_target: tuple[np.ndarray, float] | None = None
        self.present: np.ndarray | None = None
        self.commanded: np.ndarray | None = None
        self.error: BaseException | None = None
        self.late_ticks = 0

    def start(self, initial: np.ndarray) -> None:
        """Start holding `initial` (the measured pose), so nothing jumps."""
        now = time.perf_counter()
        self.shaper.reset(initial, now)
        self.present = initial.copy()
        self.commanded = initial.copy()
        self._thread = threading.Thread(target=self._run, name="kikobot-servo-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def set_target(self, target: np.ndarray) -> None:
        if self.error is not None:
            raise ConnectionError("Servo stream stopped") from self.error
        with self._lock:
            self._new_target = (np.array(target, dtype=float), time.perf_counter())

    def latest_present(self) -> np.ndarray:
        with self._lock:
            return self.present.copy()

    def _run(self) -> None:
        tick = 0
        read_failures = 0
        next_t = time.perf_counter()
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                with self._lock:
                    new, self._new_target = self._new_target, None
                if new is not None:
                    self.shaper.set_target(*new)

                command = self.shaper.step(now)
                self.bus.sync_write_goals(self.to_raw(command))
                self.commanded = command

                if tick % self.read_every == 0:
                    try:
                        present = self.from_raw(self.bus.sync_read_positions())
                        with self._lock:
                            self.present = present
                        read_failures = 0
                    except ConnectionError as exc:
                        # A corrupted reply now and then is normal on a busy
                        # Feetech bus; only a run of them is a real fault.
                        read_failures += 1
                        if read_failures >= MAX_READ_FAILURES:
                            raise
                        logger.debug("Skipped a position read: %s", exc)
                tick += 1

                next_t += self.period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    self.late_ticks += 1
                    if delay < -self.period:
                        next_t = time.perf_counter()  # Resync after a stall.
        except BaseException as exc:  # Surface it on the next send_action.
            logger.error("Servo stream stopped: %s", exc)
            self.error = exc
