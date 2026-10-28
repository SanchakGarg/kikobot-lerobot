"""Measure a joint's ringing frequency to tune the input shaper.

The arm's analogue of Klipper's TEST_RESONANCES: one joint gets a sharp,
unshaped step, its encoder is logged as fast as the bus allows, and the
oscillation after the step gives the frequency and damping to pass as
`--robot.motion.shaper_freq_hz` and `--robot.motion.shaper_damping`.

    python scripts/measure_resonance.py --port /dev/ttyACM0 --joint J2

Test the joint that carries the most load (J2 or J3) in a pose where the
arm is stretched out, since that is where it rings most. The arm moves
--step-deg and back, so keep the space around it clear.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from lerobot_robot_kikobot.config import ARM_JOINTS, default_port
from lerobot_robot_kikobot.feetech import (
    ADDR_ACCELERATION,
    ADDR_GOAL_POSITION,
    ADDR_GOAL_SPEED,
    ADDR_PRESENT_POSITION,
    ADDR_TORQUE_ENABLE,
    RESOLUTION,
    FeetechBus,
)

DEG_PER_STEP = 360.0 / RESOLUTION


def log_step(bus: FeetechBus, servo_id: int, step: int, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    start = bus.sync_read_positions()[servo_id]
    bus.write_word(servo_id, ADDR_GOAL_POSITION, start + step)
    t0 = time.perf_counter()
    ts, xs = [], []
    while (t := time.perf_counter() - t0) < seconds:
        raw, result, _ = bus.packet.read2ByteTxRx(bus.port, servo_id, ADDR_PRESENT_POSITION)
        if result == bus.scs.COMM_SUCCESS:
            ts.append(t)
            xs.append(raw)
    bus.write_word(servo_id, ADDR_GOAL_POSITION, start)  # Back where it was.
    return np.array(ts), np.array(xs, dtype=float)


def analyse(ts: np.ndarray, xs: np.ndarray) -> tuple[float, float] | None:
    """Frequency (Hz) and damping ratio of the ringing, or None if there is none."""
    final = np.mean(xs[ts > ts[-1] - 0.3])
    resid = xs - final
    # Start once the servo has covered most of the move.
    moving = np.nonzero(np.abs(resid) < 0.5 * np.abs(resid[0]))[0]
    if len(moving) == 0:
        return None
    resid, ts = resid[moving[0]:], ts[moving[0]:]
    crossings = ts[1:][np.diff(np.sign(resid)) != 0]
    if len(crossings) < 3:
        return None
    period = 2.0 * np.median(np.diff(crossings))
    peaks = [
        np.max(np.abs(resid[(ts >= a) & (ts < b)])) for a, b in zip(crossings[:-1], crossings[1:])
    ]
    damping = 0.0
    if len(peaks) >= 2 and peaks[1] > 0:
        # Log decrement over half periods.
        delta = 2.0 * math.log(peaks[0] / peaks[1])
        damping = delta / math.sqrt(4 * math.pi**2 + delta**2)
    return 1.0 / period, min(max(damping, 0.0), 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=default_port())
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--joint", default="J2", choices=ARM_JOINTS)
    parser.add_argument("--servo-ids", type=int, nargs=6, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--step-deg", type=float, default=8.0)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    servo_id = args.servo_ids[ARM_JOINTS.index(args.joint)]
    bus = FeetechBus(args.port, args.baudrate, args.servo_ids)
    results = []
    try:
        bus.ping_all()
        here = bus.sync_read_positions()
        bus.sync_write_goals(here)
        for i in args.servo_ids:
            bus.write_byte(i, ADDR_TORQUE_ENABLE, 1)
        # Raw, unprofiled step: the strongest excitation of the ringing mode.
        bus.write_byte(servo_id, ADDR_ACCELERATION, 0)
        bus.write_word(servo_id, ADDR_GOAL_SPEED, 0)
        step = int(round(args.step_deg / DEG_PER_STEP))
        for k in range(args.repeats):
            ts, xs = log_step(bus, servo_id, step if k % 2 == 0 else -step, args.seconds)
            time.sleep(0.8)
            rate = len(ts) / ts[-1] if len(ts) > 1 else 0.0
            found = analyse(ts, xs)
            if found is None:
                print(f"run {k + 1}: no ringing visible ({rate:.0f} samples/s)")
            else:
                print(f"run {k + 1}: {found[0]:.1f} Hz, damping {found[1]:.2f} ({rate:.0f} samples/s)")
                results.append(found)
    finally:
        bus.close()  # Torque stays on: the arm holds its pose.

    if not results:
        print(
            "\nThe encoder shows no oscillation, so the servo loop itself is well damped.\n"
            "Flex in the links can still ring without showing here: if the gripper\n"
            "wobbles after a move, keep --robot.motion.shaper_type=zv and lower\n"
            "shaper_freq_hz until it stops (4-8 Hz is typical)."
        )
        return
    freq = float(np.median([f for f, _ in results]))
    damping = float(np.median([d for _, d in results]))
    print(
        f"\nUse:  --robot.motion.shaper_freq_hz={freq:.1f} --robot.motion.shaper_damping={damping:.2f}\n"
        "Start with shaper_type=zv (least delay); use zvd or ei if some ringing remains."
    )


if __name__ == "__main__":
    main()
