import ctypes
import dataclasses
import enum

from core.utils.ctypes_utils import STRUCTURE

FRODO_LL_ADDRESS_TABLE = 0x01

class FRODO_LL_Functions(enum.IntEnum):
    FRODO_LL_FUNCTION_FIRMWARE_STATE = 0x01
    FRODO_LL_FUNCTION_FIRMWARE_TICK = 0x02
    FRODO_LL_FUNCTION_SET_SPEED = 0x03
    FRODO_LL_FUNCTION_FIRMWARE_BEEP = 0x05
    FRODO_LL_FUNCTION_EXTERNAL_LED = 0x07

class FRODO_LL_Messages(enum.IntEnum):
    FRODO_LL_MESSAGE_SAMPLE_STREAM = 0x10

class motor_input_struct(ctypes.Structure):
    _fields_ = [("left", ctypes.c_float), ("right", ctypes.c_float)]

@STRUCTURE
class motor_speed_struct:
    FIELDS = {
        'left': ctypes.c_float,
        'right': ctypes.c_float,
    }


@STRUCTURE
class frodo_ll_sample_general:
    FIELDS = {
        'tick': ctypes.c_uint32,
        'state': ctypes.c_uint8,
        'update_time': ctypes.c_float,
        'battery_voltage': ctypes.c_float,
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
    battery_voltage: float

@dataclasses.dataclass
class FRODO_LL_SAMPLE_DRIVE:
    speed: FRODO_LL_MOTOR_SPEED
    goal_speed: FRODO_LL_MOTOR_SPEED
    rpm: FRODO_LL_MOTOR_SPEED

@dataclasses.dataclass
class FRODO_LL_SAMPLE:
    general: FRODO_LL_SAMPLE_GENERAL
    drive: FRODO_LL_SAMPLE_DRIVE