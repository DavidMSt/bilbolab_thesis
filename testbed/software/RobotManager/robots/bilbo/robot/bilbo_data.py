import dataclasses
import enum
import math

import dacite
from dacite import from_dict

from robots.bilbo.robot.bilbo_definitions import BILBO_Control_Mode


@dataclasses.dataclass
class BILBO_LL_Sample_General:
    tick: int = 0
    status: int = 0


class TWIPR_ErrorType(enum.IntEnum):
    NONE = 0,
    MINOR = 1,
    MAJOR = 2,
    CRITICAL = 3,


class TWIPR_ErrorCodes(enum.IntEnum):
    UNSPECIFIED = 0,
    WHEEL_SPEED = 1,
    MANUAL_STOP = 2,
    INIT = 3,
    START = 4,
    IMU_INITIALIZE = 5,
    MOTOR_RACECONDITIONS = 6,
    FIRMWARE_RACECONDITION = 7


@dataclasses.dataclass
class BILBO_LL_Log_Entry:
    tick: int = 0
    type: TWIPR_ErrorType = TWIPR_ErrorType.NONE
    error: TWIPR_ErrorCodes = TWIPR_ErrorCodes.UNSPECIFIED


@dataclasses.dataclass
class BILBO_LL_Sample_Errors:
    state: TWIPR_ErrorType = TWIPR_ErrorType.NONE
    last_entry: BILBO_LL_Log_Entry = dataclasses.field(default_factory=BILBO_LL_Log_Entry)


@dataclasses.dataclass
class BILBO_LL_GYR_Data:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclasses.dataclass
class BILBO_LL_Acc_Data:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclasses.dataclass
class BILBO_LL_Sensor_Data:
    speed_left: float = 0.0
    speed_right: float = 0.0
    acc: BILBO_LL_Acc_Data = dataclasses.field(default_factory=BILBO_LL_Acc_Data)
    gyr: BILBO_LL_GYR_Data = dataclasses.field(default_factory=BILBO_LL_GYR_Data)
    battery_voltage: float = 0.0


@dataclasses.dataclass
class BILBO_LL_Estimation_Data:
    v: float = 0.0
    theta: float = 0.0
    theta_dot: float = 0.0
    psi: float = 0.0
    psi_dot: float = 0.0


@dataclasses.dataclass
class BILBO_LL_Sample_Estimation:
    state: BILBO_LL_Estimation_Data = dataclasses.field(default_factory=BILBO_LL_Estimation_Data)


@dataclasses.dataclass
class BILBO_LL_Control_External_Input:
    u_direct_1: float = 0.0
    u_direct_2: float = 0.0
    u_balancing_1: float = 0.0
    u_balancing_2: float = 0.0
    u_velocity_forward: float = 0.0
    u_velocity_turn: float = 0.0


@dataclasses.dataclass
class BILBO_LL_Control_Data:
    input_velocity_forward: float = 0.0
    input_velocity_turn: float = 0.0
    input_balancing_1: float = 0.0
    input_balancing_2: float = 0.0
    input_left: float = 0.0
    input_right: float = 0.0
    output_left: float = 0.0
    output_right: float = 0.0


@dataclasses.dataclass
class BILBO_LL_Sample_Control:
    status: int = 0
    mode: int = 0
    external_input: BILBO_LL_Control_External_Input = dataclasses.field(default_factory=BILBO_LL_Control_External_Input)
    data: BILBO_LL_Control_Data = dataclasses.field(default_factory=BILBO_LL_Control_Data)


@dataclasses.dataclass
class BILBO_LL_Sample_Sequence:
    sequence_id: int = 0
    sequence_tick: int = 0


@dataclasses.dataclass
class BILBO_LL_Sample_Debug:
    debug1: int = 0
    debug2: int = 0
    debug3: int = 0
    debug4: int = 0
    debug5: int = 0
    debug6: int = 0
    debug7: float = 0.0
    debug8: float = 0.0


@dataclasses.dataclass
class BILBO_LL_Sample:
    general: BILBO_LL_Sample_General = dataclasses.field(default_factory=BILBO_LL_Sample_General)
    errors: BILBO_LL_Sample_Errors = dataclasses.field(default_factory=BILBO_LL_Sample_Errors)
    control: BILBO_LL_Sample_Control = dataclasses.field(default_factory=BILBO_LL_Sample_Control)
    estimation: BILBO_LL_Sample_Estimation = dataclasses.field(default_factory=BILBO_LL_Sample_Estimation)
    sensors: BILBO_LL_Sensor_Data = dataclasses.field(default_factory=BILBO_LL_Sensor_Data)
    sequence: BILBO_LL_Sample_Sequence = dataclasses.field(default_factory=BILBO_LL_Sample_Sequence)
    debug: BILBO_LL_Sample_Debug = dataclasses.field(default_factory=BILBO_LL_Sample_Debug)


@dataclasses.dataclass
class BILBO_Sample_General:
    id: str = ''
    status: str = ''
    configuration: str = ''
    time: float = 0.0
    time_global: float = 0.0
    tick: int = 0
    sample_time: float = 0.0
    sample_time_ll: float = 0.0


class TWIPR_Control_Status(enum.IntEnum):
    TWIPR_CONTROL_STATE_ERROR = 0
    TWIPR_CONTROL_STATE_NORMAL = 1


class TWIPR_Control_Status_LL(enum.IntEnum):
    TWIPR_CONTROL_STATE_LL_ERROR = 0
    TWIPR_CONTROL_STATE_LL_NORMAL = 1


class TWIPR_Control_Mode_LL(enum.IntEnum):
    TWIPR_CONTROL_MODE_LL_OFF = 0,
    TWIPR_CONTROL_MODE_LL_DIRECT = 1,
    TWIPR_CONTROL_MODE_LL_BALANCING = 2,
    TWIPR_CONTROL_MODE_LL_VELOCITY = 3


@dataclasses.dataclass
class TWIPR_ControlInput:
    @dataclasses.dataclass
    class velocity:
        forward: float = 0
        turn: float = 0

    class balancing:
        u_left: float = 0
        u_right: float = 0

    class direct:
        u_left: float = 0
        u_right: float = 0


@dataclasses.dataclass
class TWIPR_Control_Sample:
    status: TWIPR_Control_Status = dataclasses.field(
        default=TWIPR_Control_Status(TWIPR_Control_Status.TWIPR_CONTROL_STATE_ERROR))
    mode: BILBO_Control_Mode = dataclasses.field(default=BILBO_Control_Mode(BILBO_Control_Mode.OFF))
    vic_enabled: bool = False
    tic_enabled: bool = False
    configuration: str = ''
    input: TWIPR_ControlInput = dataclasses.field(default_factory=TWIPR_ControlInput)


@dataclasses.dataclass
class BILBO_DynamicState:
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    theta: float = 0.0
    theta_dot: float = 0.0
    psi: float = 0.0
    psi_dot: float = 0.0

@dataclasses.dataclass
class BILBO_ConfigurationState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    theta: float = 0.0
    psi: float = 0.0


class TWIPR_Estimation_Status(enum.IntEnum):
    ERROR = 0,
    NORMAL = 1,


@dataclasses.dataclass(frozen=True)
class TWIPR_Estimation_Sample:
    status: TWIPR_Estimation_Status = TWIPR_Estimation_Status.ERROR
    state: BILBO_DynamicState = dataclasses.field(default_factory=BILBO_DynamicState)
    state_optitrack: BILBO_ConfigurationState | None = dataclasses.field(default_factory=BILBO_ConfigurationState)


class TWIPR_Drive_Status(enum.IntEnum):
    TWIPR_DRIVE_STATUS_OFF = 1,
    TWIPR_DRIVE_STATUS_ERROR = 0
    TWIPR_DRIVE_STATUS_NORMAL = 2


@dataclasses.dataclass
class TWIPR_Drive_Data:
    status: TWIPR_Drive_Status = TWIPR_Drive_Status.TWIPR_DRIVE_STATUS_OFF
    torque: float = 0
    speed: float = 0
    input: float = 0


@dataclasses.dataclass
class TWIPR_Drive_Sample:
    left: TWIPR_Drive_Data = dataclasses.field(default_factory=TWIPR_Drive_Data)
    right: TWIPR_Drive_Data = dataclasses.field(default_factory=TWIPR_Drive_Data)


@dataclasses.dataclass
class TWIPR_Sensors_IMU:
    gyr: dict = dataclasses.field(default_factory=dict)
    acc: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TWIPR_Sensors_Power:
    bat_voltage: float = 0
    bat_current: float = 0


@dataclasses.dataclass
class TWIPR_Sensors_Drive_Data:
    speed: float = 0
    torque: float = 0
    slip: bool = False


@dataclasses.dataclass
class TWIPR_Sensors_Drive:
    left: TWIPR_Sensors_Drive_Data = dataclasses.field(default_factory=TWIPR_Sensors_Drive_Data)
    right: TWIPR_Sensors_Drive_Data = dataclasses.field(default_factory=TWIPR_Sensors_Drive_Data)


@dataclasses.dataclass
class TWIPR_Sensors_Distance:
    front: float = 0
    back: float = 0


@dataclasses.dataclass
class TWIPR_Sensors_Sample:
    imu: TWIPR_Sensors_IMU = dataclasses.field(default_factory=TWIPR_Sensors_IMU)
    power: TWIPR_Sensors_Power = dataclasses.field(default_factory=TWIPR_Sensors_Power)
    drive: TWIPR_Sensors_Drive = dataclasses.field(default_factory=TWIPR_Sensors_Drive)
    distance: TWIPR_Sensors_Distance = dataclasses.field(default_factory=TWIPR_Sensors_Distance)


@dataclasses.dataclass
class BILBO_Sample_Connection:
    strength: float = 0.0
    internet: bool = False


@dataclasses.dataclass(frozen=False)
class BILBO_Sample:
    general: BILBO_Sample_General = dataclasses.field(default_factory=BILBO_Sample_General)
    connection: BILBO_Sample_Connection = dataclasses.field(default_factory=BILBO_Sample_Connection)
    control: TWIPR_Control_Sample = dataclasses.field(default_factory=TWIPR_Control_Sample)
    estimation: TWIPR_Estimation_Sample = dataclasses.field(default_factory=TWIPR_Estimation_Sample)
    drive: TWIPR_Drive_Sample = dataclasses.field(default_factory=TWIPR_Drive_Sample)
    sensors: TWIPR_Sensors_Sample = dataclasses.field(default_factory=TWIPR_Sensors_Sample)
    lowlevel: BILBO_LL_Sample = dataclasses.field(default_factory=BILBO_LL_Sample)


type_hooks = {
    BILBO_Control_Mode: BILBO_Control_Mode,
    TWIPR_Control_Status: TWIPR_Control_Status,
    TWIPR_Control_Status_LL: TWIPR_Control_Status_LL,
    TWIPR_Control_Mode_LL: TWIPR_Control_Mode_LL,
    TWIPR_Estimation_Status: TWIPR_Estimation_Status,
    TWIPR_Drive_Status: TWIPR_Drive_Status,
    TWIPR_ErrorType: TWIPR_ErrorType,
    TWIPR_ErrorCodes: TWIPR_ErrorCodes
}


def bilboSampleFromDict(dict) -> BILBO_Sample:
    sample = from_dict(data_class=BILBO_Sample, data=dict, config=dacite.Config(type_hooks=type_hooks))
    return sample


BILBO_STATE_DATA_DEFINITIONS = {
    'x': {
        'type': 'float',
        'unit': 'm',
        'max': 3,
        'min': -3,
        'display_resolution': '.1f'
    },
    'y': {
        'type': 'float',
        'unit': 'm',
        'max': 3,
        'min': -3,
        'display_resolution': '.1f'
    },
    'theta': {
        'type': 'float',
        'unit': 'rad',
        'max': math.pi / 2,
        'min': -math.pi / 2,
        'display_resolution': '.1f'
    },
    'theta_dot': {
        'type': 'float',
        'unit': 'rad/s',
        'max': 10,
        'min': -10,
        'display_resolution': '.1f'
    },
    'v': {
        'type': 'float',
        'unit': 'm/s',
        'max': 10,
        'min': -10,
        'display_resolution': '.1f'
    },
    'psi': {
        'type': 'float',
        'unit': 'rad',
        'max': math.pi,
        'min': -math.pi,
        'display_resolution': '.1f'
    },
    'psi_dot': {
        'type': 'float',
        'unit': 'rad/s',
        'max': 10,
        'min': -10,
        'display_resolution': '.1f'
    }
}
