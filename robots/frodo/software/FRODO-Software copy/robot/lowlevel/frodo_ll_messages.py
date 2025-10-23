import ctypes
import dataclasses

from robot.lowlevel.frodo_ll_definition import motor_speed_struct
from utils.ctypes_utils import STRUCTURE


@STRUCTURE
class frodo_ll_sample_general:
    FIELDS = {
        'tick': ctypes.c_uint32,
        'state': ctypes.c_uint8,
        'update_time': ctypes.c_float
    }


@STRUCTURE
class frodo_ll_sample_drive:
    FIELDS = {
        'speed': motor_speed_struct,
        'goal_speed': motor_speed_struct,
        'rpm': motor_speed_struct
    }


@STRUCTURE
class frodo_ll_sample:
    FIELDS = {
        'general': frodo_ll_sample_general,
        'drive': frodo_ll_sample_drive,
    }

@dataclasses.dataclass
class FRODO_LL_MOTOR_SPEED:
    left: float
    right: float

@dataclasses.dataclass
class FRODO_LL_SAMPLE_GENERAL:
    tick: int
    state: int
    update_time: float

@dataclasses.dataclass
class FRODO_LL_SAMPLE_DRIVE:
    speed: FRODO_LL_MOTOR_SPEED
    goal_speed: FRODO_LL_MOTOR_SPEED
    rpm: FRODO_LL_MOTOR_SPEED

@dataclasses.dataclass
class FRODO_LL_SAMPLE:
    general: FRODO_LL_SAMPLE_GENERAL
    drive: FRODO_LL_SAMPLE_DRIVE