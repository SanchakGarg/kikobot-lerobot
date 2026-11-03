"""Configs for the three LeRobot devices this plugin registers.

Every field is settable from the CLI, e.g. `--robot.max_velocity_deg_s=150`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

from .protocol import ARM_JOINTS, DEFAULT_PORT  # noqa: F401 - ARM_JOINTS is re-exported

# The Isaac Sim server listens here by default, so both servers can run at once.
ISAAC_DEFAULT_PORT = DEFAULT_PORT + 1
SIM_BACKENDS = ("mujoco", "isaac", "none")



# Linux ports tried in order when --robot.port is not given. USB adapters
# (Feetech URT-1, Waveshare Bus Servo Adapter, or the Waveshare Bus Servo
# Driver HAT with its jumper on A) enumerate as ttyACM* (CH343) or ttyUSB*
# (CH340, FTDI). The Driver HAT with its jumper on B talks over the Raspberry
# Pi's GPIO UART, which is /dev/serial0.
LINUX_PORTS = ("/dev/kikobot", "/dev/ttyACM0", "/dev/ttyUSB0", "/dev/serial0")


def default_port() -> str:
    """The first servo port that exists; `lerobot-find-port` finds others."""
    if os.name == "nt":
        return "COM3"
    return next((p for p in LINUX_PORTS if os.path.exists(p)), "/dev/ttyACM0")


@RobotConfig.register_subclass("kikobot_sim")
@dataclass
class KikobotSimConfig(RobotConfig):
    """The simulated arm, run by `kikobot-sim-server` (start that first)."""

    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    # Seconds to wait for each server reply, and for the first one.
    timeout_s: float = 5.0
    connect_timeout_s: float = 10.0
    # Cameras of the server's scene to observe: record (front) and/or overhead.
    # Not `cameras`: RobotConfig reserves that name for CameraConfig objects.
    camera_names: tuple[str, ...] = ("record",)
    # Offscreen framebuffer is capped at 1280x720 by the scene XML.
    width: int = 640
    height: int = 480
    # raw: exact pixels (use on one machine). jpeg: ~20x smaller (over a network).
    image_encoding: str = "raw"
    # Reset the scene (arm to ready, objects respawned) when connecting.
    reset_on_connect: bool = True
    # Joint speed cap, the same as the real arm's.
    max_velocity_deg_s: float = 180.0


@RobotConfig.register_subclass("kikobot_isaac")
@dataclass
class KikobotIsaacConfig(KikobotSimConfig):
    """The arm in NVIDIA Isaac Sim, run by `isaac/kikobot_isaac_server.py` (start that first).

    Same protocol and behaviour as `kikobot_sim`. The scene, cameras and image
    size are chosen when starting the server. See docs/isaac-sim-teleop.md.
    """

    port: int = ISAAC_DEFAULT_PORT
    # Isaac's RTX renders are slower, and the first one compiles shaders.
    timeout_s: float = 10.0
    connect_timeout_s: float = 120.0


@RobotConfig.register_subclass("kikobot_follower")
@dataclass
class KikobotFollowerConfig(RobotConfig):
    port: str = field(default_factory=default_port)
    baudrate: int = 1_000_000
    # Bus IDs of J1..J6, in joint order.
    servo_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    # Bus ID of the gripper servo; None when there is none (`gripper.pos` is
    # then ignored).
    gripper_id: int | None = 7
    gripper_open_deg: float = 0.0
    gripper_closed_deg: float = -60.0
    # Joints whose servo turns opposite to the model. Applied on top of any
    # calibration drive_mode, so it works with or without a calibration file.
    inverted_joints: tuple[str, ...] = ("J5",)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Every send_action writes its goal straight to the servos in one
    # sync-write packet, as LeRobot's SO-101 does: no filter, no delay. This
    # only caps joint speed, for targets that jump (a policy's first action).
    max_velocity_deg_s: float = 180.0
    # Speed of the first move after connecting, from wherever the arm was.
    soft_start_deg_s: float = 30.0

    # Position-loop gains written to each servo (same as LeRobot's SO-101).
    # Feetech's factory P of 32 overshoots and buzzes on a loaded arm.
    p_coefficient: int = 16
    i_coefficient: int = 0
    d_coefficient: int = 32

    # Droop correction: joints that sag under the arm's weight settle short of
    # their goal. While the arm holds still, their goal is shifted until the
    # measured angle sits on the commanded one. `[]` turns it off.
    sag_joints: tuple[str, ...] = ("J2",)
    # How fast the correction builds up, and the most it may add.
    sag_time_constant_s: float = 0.5
    sag_max_deg: float = 15.0

    # Dropping torque lets a gravity-loaded arm fall, so the default keeps it on.
    disable_torque_on_disconnect: bool = False
    # Mirror every command to a sim server, so the sim arm moves with the real
    # one: mujoco (`kikobot-sim-server`), isaac (`isaac/kikobot_isaac_server.py`)
    # or none. If the server is not running, the real arm still runs.
    sim_backend: str = "mujoco"
    sim_host: str = "127.0.0.1"
    # None: the backend's default port (5556 mujoco, 5557 isaac).
    sim_port: int | None = None
    sim_timeout_s: float = 0.5

    @property
    def sim_address(self) -> tuple[str, int]:
        default = ISAAC_DEFAULT_PORT if self.sim_backend == "isaac" else DEFAULT_PORT
        return self.sim_host, self.sim_port if self.sim_port is not None else default
    # During lerobot-calibrate, pose the sim server's arm the way each step asks for.
    calibration_view: bool = True


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
    # Easing off the stick, the speed falls at this rate instead (full speed
    # per second): a small move stops almost where it is let go, a fast one
    # gets the braking distance the arm needs to stop without overshooting.
    stop_rate_per_s: float = 4.0
    # Fraction of full speed while precision mode (X) is on.
    precision_scale: float = 0.3
    # Work volume of the gripper centre, in metres from the base.
    workspace_radius_m: float = 0.42
    workspace_min_z_m: float = 0.01
    workspace_max_z_m: float = 0.50
    # `--robot.id` of the arm whose home pose (set with lerobot-calibrate) the
    # teleop starts at and returns to on Y. Without one, the built-in ready pose.
    robot_id: str | None = "kikobot"
