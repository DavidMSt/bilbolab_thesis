import dataclasses
import enum

BILBO_HOST_NAMES = ['bilbo1', 'bilbo2', 'bilbo3', 'bilbo4', 'bilbo5']

PATH_TO_MAIN = '/home/admin/robot/software/main.py'
PYENV_SHIM_PATH = '/home/admin/.pyenv/shims/python3'
BILBO_USER_NAME = 'admin'
BILBO_PASSWORD = 'beutlin'

BILBO_CONTROL_DT = 0.01
MAX_STEPS_TRAJECTORY = 3000


class BILBO_Control_Mode(enum.IntEnum):
    OFF = 0,
    DIRECT = 1,
    BALANCING = 2,
    VELOCITY = 3


@dataclasses.dataclass
class BILBO_Information:
    id: str = ''
    type: str = ''
    version: str = ''
    color: list | None = None
    address: str = ''
    data_stream_port: int = ''
    gui_port: int = ''
    ssid: str = ''
    username: str = ''
    password: str = ''
    hardware: dict = dataclasses.field(default_factory=dict)


# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class TIC_Config:
    enabled: bool = False
    ki: float = 0.0
    max_error: float = 0.0
    theta_limit: float = 0.0


@dataclasses.dataclass
class VIC_Config:
    enabled: bool = False
    ki: float = 0.0
    max_error: float = 0.0
    v_limit: float = 0.0


@dataclasses.dataclass
class TWIPR_Balancing_Control_Config:
    K: list = dataclasses.field(default_factory=list)  # State Feedback Gain
    tic: TIC_Config = dataclasses.field(default_factory=TIC_Config)
    vic: VIC_Config = dataclasses.field(default_factory=VIC_Config)


@dataclasses.dataclass
class TWIPR_PID_Control_Config:
    Kp: float = 0.0
    Kd: float = 0.0
    Ki: float = 0.0
    # anti_windup: float = 0
    # integrator_saturation: float = None


@dataclasses.dataclass
class SpeedControl_Config:
    v: TWIPR_PID_Control_Config = dataclasses.field(default_factory=TWIPR_PID_Control_Config)
    psidot: TWIPR_PID_Control_Config = dataclasses.field(default_factory=TWIPR_PID_Control_Config)
    max_speeds: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class General_Control_Config:
    max_wheel_speed: float = 0.0
    max_wheel_torque: float = 0.0
    enable_external_inputs: bool = False
    torque_offset: dict = dataclasses.field(default_factory=lambda: {'left': 0, 'right': 0})


@dataclasses.dataclass
class ExternalInputsConfig:
    balancing_input_gain: dict = dataclasses.field(default_factory=dict)  # 'forward' and 'turn'
    speed_input_gain: dict = dataclasses.field(default_factory=dict)  # 'forward' and 'turn'


@dataclasses.dataclass
class BILBO_ControlConfig:
    name: str = ''
    description: str = ''
    mode: BILBO_Control_Mode = dataclasses.field(default=BILBO_Control_Mode(BILBO_Control_Mode.OFF))
    general: General_Control_Config = dataclasses.field(default_factory=General_Control_Config)
    external_inputs: ExternalInputsConfig = dataclasses.field(default_factory=ExternalInputsConfig)
    balancing_control: TWIPR_Balancing_Control_Config = dataclasses.field(
        default_factory=TWIPR_Balancing_Control_Config)
    speed_control: SpeedControl_Config = dataclasses.field(default_factory=SpeedControl_Config)
