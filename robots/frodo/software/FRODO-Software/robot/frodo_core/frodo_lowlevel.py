import ctypes

from core.communication.serial.core.serial_protocol import UART_Message
from core.utils.callbacks import CallbackContainer, callback_definition
from core.utils.ctypes_utils import CType, bytes_to_value
from core.utils.dataclass_utils import from_dict
from core.utils.events import event_definition, Event
from core.utils.logging_utils import Logger
from hardware.control_board import RobotControl_Board
from robot.frodo_core.lowlevel_definitions import frodo_ll_sample, FRODO_LL_SAMPLE


@callback_definition
class FRODO_LowLevel_Callbacks:
    sample: CallbackContainer


@event_definition
class FRODO_LowLevel_Events:
    sample: Event
    initialized: Event


class FRODO_Lowlevel:
    board: RobotControl_Board
    battery_voltage: float = 0.0

    sample: FRODO_LL_SAMPLE
    _first_sample_received: bool = False

    # === INIT =========================================================================================================
    def __init__(self, board: RobotControl_Board):
        self.board = board
        self.logger = Logger('LOWLEVEL', 'DEBUG')
        self.callbacks = FRODO_LowLevel_Callbacks()
        self.events = FRODO_LowLevel_Events()

        self.board.serial_interface.callbacks.stream.register(self._rxStreamCallback)
        self.board.serial_interface.callbacks.event.register(self._rxEventCallback)

    # === METHODS ======================================================================================================
    def init(self):
        self.board.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.board.start()

    # ------------------------------------------------------------------------------------------------------------------
    def writeValue(self, module: int = 0, address: int | list = None, value=None, type=ctypes.c_uint8):
        self.board.serial_interface.write(module, address, value, type)

    # ------------------------------------------------------------------------------------------------------------------
    def readValue(self, address: int, module: int = 0, type=ctypes.c_uint8):
        return self.board.serial_interface.read(address, module, type)

    # ------------------------------------------------------------------------------------------------------------------
    def executeFunction(self, address, module: int = 0, data=None, input_type: CType = None, output_type=None,
                        timeout=1):
        return self.board.serial_interface.function(address, module, data, input_type, output_type, timeout)

    # === PRIVATE METHODS ==============================================================================================
    def _rxStreamCallback(self, data: UART_Message, *args, **kwargs):
        data = data.data
        sample = from_dict(FRODO_LL_SAMPLE, bytes_to_value(data, frodo_ll_sample))
        self.battery_voltage = sample.general.battery_voltage

        if not self._first_sample_received:
            self._first_sample_received = True
            self.events.initialized.set(sample)

        self.callbacks.sample.call(sample)
        self.events.sample.set(sample)
        self.sample = sample

    def _rxEventCallback(self, data, *args, **kwargs):
        self.logger.debug(f"RX EVENT: {data}")
