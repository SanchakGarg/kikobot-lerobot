"""Configs for the three LeRobot devices this plugin registers.

Every field is settable from the CLI, e.g. `--robot.motion.shaper_type=zvd`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

ARM_JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")


def default_port() -> str:
    # Feetech's URT-1 / CH343 boards enumerate as ttyACM*; CH340 clones as
    # ttyUSB*. `lerobot-find-port` tells you which one you have.
    return "COM3" if os.name == "nt" else "/dev/ttyACM0"


@dataclass
class MotionConfig:
    """How a stream of joint targets is turned into servo motion.

    Shared by the sim and the real arm, so the sim moves the way the hardware
    will. See `motion.py` for what each stage does.
    """

    # Trajectory follower: the commanded joints never exceed these.
    max_velocity_deg_s: float = 120.0
    max_acceleration_deg_s2: float = 600.0
    # Gain of the final approach (1/s). Higher tracks tighter, lower is softer.
    tracking_gain: float = 12.0
    # Extrapolate the incoming target between updates so the follower does
    # not lag a steadily moving stick by a whole control period.
    feedforward: bool = True
    # Input shaper, as in Klipper: none | zv | zvd | mzv | ei.
    shaper_type: str = "zv"
    # The arm's dominant ringing frequency. Measure it with
    # `python scripts/measure_resonance.py`; the default is a typical value
    # for a servo arm of this size.
    shaper_freq_hz: float = 6.0
    shaper_damping: float = 0.1
    # Velocity limit ramps from this fraction to 1 over `soft_start_s`, so the
    # first move after connecting (wherever the arm was) is gentle.
    soft_start_fraction: float = 0.25
    soft_start_s: float = 3.0


@RobotConfig.register_subclass("kikobot_sim")
@dataclass
class KikobotSimConfig(RobotConfig):
    # empty | pick_place | cylinder, or a path to an MJCF file.
    scene: str = "pick_place"
    fps: int = 30
    # Offscreen framebuffer is capped at 1280x720 by the scene XML.
    width: int = 640
    height: int = 480
    # Not `cameras`: RobotConfig reserves that name for CameraConfig objects.
    camera_names: tuple[str, ...] = ("record",)
    seed: int | None = None
    # Native MuJoCo window. macOS needs `mjpython` instead of `python`.
    viewer: bool = False
    motion: MotionConfig = field(default_factory=MotionConfig)


@RobotConfig.register_subclass("kikobot_follower")
@dataclass
class KikobotFollowerConfig(RobotConfig):
    port: str = field(default_factory=default_port)
    baudrate: int = 1_000_000
    # Bus IDs of J1..J6, in joint order.
    servo_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    # Set when a gripper servo is on the bus; `gripper.pos` is ignored otherwise.
    gripper_id: int | None = None
    gripper_open_deg: float = 0.0
    gripper_closed_deg: float = -60.0
    # Joints whose servo turns opposite to the model. Applied on top of any
    # calibration drive_mode, so it works with or without a calibration file.
    inverted_joints: tuple[str, ...] = ("J5",)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Goal positions are streamed by a background thread at this rate, in one
    # sync-write packet for all servos, independent of the LeRobot loop's fps.
    stream_hz: float = 100.0
    # Present positions are read every Nth stream tick. A read waits for the
    # replies; a write does not, so reads are what limit the stream rate.
    read_every: int = 3
    motion: MotionConfig = field(default_factory=MotionConfig)

    # Position-loop gains written to each servo (same as LeRobot's SO-101).
    # Feetech's factory P of 32 overshoots and buzzes on a loaded arm.
    p_coefficient: int = 16
    i_coefficient: int = 0
    d_coefficient: int = 32

    # Dropping torque lets a gravity-loaded arm fall, so the default keeps it on.
    disable_torque_on_disconnect: bool = False
    # MuJoCo window mirroring the measured joint angles.
    viewer: bool = False


@TeleoperatorConfig.register_subclass("kikobot_gamepad")
@dataclass
class KikobotGamepadConfig(TeleoperatorConfig):
    # Preferred GLFW joystick slot. If it holds no gamepad, the first one found is used.
    joystick: int = 0
    fps: int = 30
    linear_speed_m_s: float = 0.12
    angular_speed_deg_s: float = 90.0
    gripper_speed_pct_s: float = 120.0
    deadzone: float = 0.12
    # Stick input is low-passed with this time constant, so a flick of the
    # stick ramps the velocity up and down instead of stepping it.
    input_smoothing_s: float = 0.12
    # Fraction of full speed while precision mode (X) is on.
    precision_scale: float = 0.3
    # Work volume of the gripper centre, in metres from the base.
    workspace_radius_m: float = 0.42
    workspace_min_z_m: float = 0.01
    workspace_max_z_m: float = 0.50
