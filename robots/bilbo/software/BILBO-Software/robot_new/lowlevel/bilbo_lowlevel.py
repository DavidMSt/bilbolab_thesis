import ctypes
import time

from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.events import event_definition, Event
from core.utils.logging_utils import Logger
from hardware.control_board import RobotControl_Board
from hardware.stm32.stm32 import resetSTM32
from robot_new.lowlevel.communication.bilbo_lowlevel_serial import BILBO_Lowlevel_Serial
from robot_new.lowlevel.communication.bilbo_lowlevel_spi import BILBO_Lowlevel_SPI


@callback_definition
class BILBO_LowLevel_Callbacks:
    samples: CallbackContainer


@event_definition
class BILBO_LowLevel_Events:
    samples: Event


# === BILBO LOWLEVEL ===================================================================================================
class BILBO_LowLevel:
    serial: BILBO_Lowlevel_Serial
    spi: BILBO_Lowlevel_SPI
    board: RobotControl_Board

    callbacks: BILBO_LowLevel_Callbacks
    events: BILBO_LowLevel_Events

    tick: int | None = None

    # === INIT =========================================================================================================
    def __init__(self):
        self.logger = Logger("BILBO_LOWLEVEL", "DEBUG")
        self.callbacks = BILBO_LowLevel_Callbacks()
        self.events = BILBO_LowLevel_Events()
        self.board = RobotControl_Board()
        self.spi = BILBO_Lowlevel_SPI()
        self.serial = BILBO_Lowlevel_Serial(self.board.serial_interface)

        self.spi.callbacks.rx_samples.register(self._samples_callback)

    # === METHODS ======================================================================================================
    def init(self):
        self.spi.init()
        self.serial.init()
        self.board.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.board.start()
        self.spi.start()
        self.serial.start()

    # ------------------------------------------------------------------------------------------------------------------
    def startSampleListener(self):
        self.spi.startSampleListener()

    # ------------------------------------------------------------------------------------------------------------------
    def initializeLowlevelCounter(self):
        self.serial.resetLowlevelCounter()

    # ------------------------------------------------------------------------------------------------------------------
    def resetStm32(self):
        self.logger.debug("Resetting STM32...")
        resetSTM32()

    # === PRIVATE METHODS ==============================================================================================
    def _samples_callback(self, samples, *args, **kwargs):
        self.tick = samples[-1]['general']['tick']
        self.callbacks.samples.call(samples)
        self.events.samples.set()


if __name__ == '__main__':
    bilbo_ll = BILBO_LowLevel()
    bilbo_ll.init()
    bilbo_ll.start()
    bilbo_ll.initializeLowlevelCounter()
    bilbo_ll.startSampleListener()

    while True:
        time.sleep(10)
