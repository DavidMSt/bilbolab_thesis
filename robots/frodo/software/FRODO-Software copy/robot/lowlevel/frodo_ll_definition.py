import ctypes
import enum

from utils.ctypes_utils import STRUCTURE

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

