import time
from threading import Event, Condition

from core.communication.serial.core.serial_protocol import UART_Message
from core.communication.serial.serial_interface import SerialMessage
from robot.lowlevel.frodo_ll_messages import FRODO_LL_SAMPLE
# === OWN PACKAGES =====================================================================================================
from utils.callbacks import Callback
from control_board.control_board import RobotControl_Board
from bilbo_old.communication.serial.bilbo_comm_serial import BILBO_Serial_Communication
from bilbo_old.communication.spi.twipr_comm_spi import BILBO_SPI_Interface
from bilbo_old.communication.wifi.twipr_comm_wifi import BILBO_WIFI_Interface
from utils.callbacks import callback_handler, CallbackContainer
from utils.dataclass_utils import freeze_dataclass_instance
from utils.events import ConditionEvent, event_handler
from utils.logging_utils import Logger
from robot.lowlevel.frodo_ll_definition import FRODO_LL_Messages
from robot.lowlevel.frodo_ll_messages import frodo_ll_sample
from utils.ctypes_utils import bytes_to_value, struct_to_dataclass, struct_to_dict, bytes_to_ctype

# ======================================================================================================================
handler = None

logger = Logger("COMMUNICATION")
logger.setLevel("WARNING")

# time_previous = time.perf_counter()


# ======================================================================================================================
@callback_handler
class BILBO_Communication_Callbacks:
    rx_stm32_sample: CallbackContainer


# ======================================================================================================================
@event_handler
class BILBO_Communication_Events:
    rx_stm32_sample: ConditionEvent
    stm32_tick: ConditionEvent


# ======================================================================================================================
class FRODO_Communication:
    board: RobotControl_Board

    serial: BILBO_Serial_Communication
    wifi: BILBO_WIFI_Interface
    spi: BILBO_SPI_Interface

    events: BILBO_Communication_Events
    callbacks: BILBO_Communication_Callbacks

    def __init__(self, board: RobotControl_Board):
        self.board = board

        self.wifi = BILBO_WIFI_Interface(interface=self.board.wifi_interface)
        self.serial = BILBO_Serial_Communication(interface=self.board.serial_interface)
        self.spi = BILBO_SPI_Interface(interface=self.board.spi_interface,
                                       sample_notification_pin=self.board.board_config['pins']['new_samples_interrupt'][
                                           'pin'])

        self.callbacks = BILBO_Communication_Callbacks()
        self.events = BILBO_Communication_Events()

        # Configure the SPI Interface
        self.spi.callbacks.rx_samples.register(self._stm32_rx_sample_callback)

        self.serial.callbacks.stream.register(self._serial_stream_callback)

    # === METHODS ======================================================================================================
    def init(self):
        self.spi.init()
        self.serial.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.spi.start()
        self.serial.start()

        global handler
        handler = self

    # ------------------------------------------------------------------------------------------------------------------
    def close(self):
        logger.warning("Closing FRODO Communication")
        self.spi.close()

    # === PRIVATE METHODS ==============================================================================================
    def _stm32_rx_sample_callback(self, samples, *args, **kwargs):
        sample = samples[0]

        # Execute the callbacks
        for callback in self.callbacks.rx_stm32_sample:
            callback(sample)

        # Set the events
        self.events.rx_stm32_sample.set(samples)
        self.events.stm32_tick.set(sample.general.tick)

    # ------------------------------------------------------------------------------------------------------------------
    def _serial_stream_callback(self, message: UART_Message, *args, **kwargs):
        if message.address == FRODO_LL_Messages.FRODO_LL_MESSAGE_SAMPLE_STREAM:
            # global time_previous
            # print(f"Time: {time.perf_counter() - time_previous}")
            # time_previous = time.perf_counter()
            try:
                data_struct = bytes_to_ctype(message.data, frodo_ll_sample)
            except Exception as e:
                logger.error(f"Error while parsing sample stream: {e}")
                return
            data: FRODO_LL_SAMPLE = struct_to_dataclass(data_struct, FRODO_LL_SAMPLE)
            self.callbacks.rx_stm32_sample.call(data)
