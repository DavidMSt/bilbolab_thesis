import enum
import threading

from robot.communication.frodo_communication import FRODO_Communication
from robot.control.frodo_joystick_control import StandaloneJoystickControl
from robot.control.frodo_navigation import FRODO_Navigator
from robot.lowlevel.frodo_ll_definition import motor_input_struct, FRODO_LL_ADDRESS_TABLE, FRODO_LL_Functions
from utils.callbacks import callback_handler, CallbackContainer
from utils.events import event_handler, ConditionEvent
from utils.exit import ExitHandler
from utils.logging_utils import Logger
from utils.time import IntervalTimer

logger = Logger('CONTROL')
logger.setLevel('INFO')


class FRODO_Control_Mode(enum.IntEnum):
    OFF = 1,
    EXTERNAL = 2,
    NAVIGATION = 3


# ======================================================================================================================
@callback_handler
class FRODO_Control_Callbacks:
    update: CallbackContainer


@event_handler
class FRODO_Control_Events:
    update: ConditionEvent


# ======================================================================================================================
class FRODO_Control:
    communication: FRODO_Communication
    mode: FRODO_Control_Mode
    navigation: FRODO_Navigator
    joystick: StandaloneJoystickControl

    task: threading.Thread

    callbacks: FRODO_Control_Callbacks
    events: FRODO_Control_Events
    _update_time = 0.01
    _exit: bool = False
    exit: ExitHandler

    def __init__(self, communication):
        self.communication = communication
        self.mode = FRODO_Control_Mode.OFF
        self.navigation = FRODO_Navigator()
        self.joystick = StandaloneJoystickControl()

        self.task = threading.Thread(target=self._task, daemon=True)
        self.update_timer = IntervalTimer(self._update_time)
        self.exit = ExitHandler(self.close)

        self.callbacks = FRODO_Control_Callbacks()
        self.events = FRODO_Control_Events()

    # ------------------------------------------------------------------------------------------------------------------
    def init(self):
        self.joystick.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.task.start()
        self.navigation.start()
        self.joystick.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        logger.info("Exit Control")
        self._exit = True
        self.task.join()

    # ------------------------------------------------------------------------------------------------------------------
    def _task(self):
        self.update_timer.reset()
        while not self._exit:
            self._update()
            self.update_timer.sleep_until_next()

    # ------------------------------------------------------------------------------------------------------------------
    def _update(self):
        speed_left = 0.0
        speed_right = 0.0
        if self.mode == FRODO_Control_Mode.EXTERNAL:
            speed_left, speed_right = self.joystick.getInputs()
        elif self.mode == FRODO_Control_Mode.NAVIGATION:
            speed_left, speed_right = self.navigation.getInputs()

        # self.setSpeed(speed_left=speed_left, speed_right=speed_right)

        
    # ------------------------------------------------------------------------------------------------------------------
    def setMode(self, mode: FRODO_Control_Mode):
        if isinstance(mode, int):
            mode = FRODO_Control_Mode(mode)
        self.mode = mode
        logger.info(f"Set control mode to {mode.name}")

    # ------------------------------------------------------------------------------------------------------------------
    def setSpeed(self, speed_left, speed_right):
        input_struct = motor_input_struct(left=speed_left, right=speed_right)
        self.communication.serial.executeFunction(module=FRODO_LL_ADDRESS_TABLE,
                                                  address=FRODO_LL_Functions.FRODO_LL_FUNCTION_SET_SPEED,
                                                  data=input_struct,
                                                  input_type=motor_input_struct)

    # === PRIVATE METHODS ==============================================================================================
