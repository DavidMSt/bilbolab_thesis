import threading
import time
from ctypes import sizeof

# === OWN PACKAGES =====================================================================================================
from core.communication.spi.spi import SPI_Interface
from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.dataclass_utils import from_dict
# from utils.exit import ExitHandler
from core.utils.ctypes_utils import bytes_to_value
from core.utils.logging_utils import Logger
from hardware.board_config import getBoardConfig
from robot.lowlevel.stm32_sample import SAMPLE_BUFFER_LL_SIZE
from hardware.hardware.gpio import GPIO_Input, InterruptFlank, PullupPulldown
from core.utils.time import precise_sleep
from core.utils.bytes_utils import intToByteList
from robot_new.lowlevel.definitions.stm32_sample import bilbo_ll_sample_struct, BILBO_LL_Sample


# ======================================================================================================================
@callback_definition
class BILBO_SPI_Callbacks:
    rx_latest_sample: CallbackContainer
    rx_samples: CallbackContainer


class BILBO_SPI_Command_Type:
    READ_SAMPLE = 1
    SEND_TRAJECTORY = 2


class BILBO_Lowlevel_SPI:
    interface: SPI_Interface
    sample_notification_pin: int

    _input_pin: GPIO_Input | None
    _startSampleListening: bool = False
    _lock: threading.Lock

    _lowlevel_tick: int | None = None

    # === INIT =========================================================================================================
    def __init__(self, sample_notification_pin: int = None, interface: SPI_Interface = None, ):

        if interface is None:
            interface = SPI_Interface()

        self.interface = interface

        if sample_notification_pin is None:
            board_config = getBoardConfig()
            sample_notification_pin = board_config.definitions.pins.new_samples_interrupt.pin

        self.sample_notification_pin = sample_notification_pin

        self.callbacks = BILBO_SPI_Callbacks()


        self.rx_buffer = bytearray(SAMPLE_BUFFER_LL_SIZE * sizeof(bilbo_ll_sample_struct))

        self._input_pin = None
        self._lock = threading.Lock()

        self.logger = Logger('BILBO_SPI', 'DEBUG')

    # === METHODS ======================================================================================================
    def init(self):
        self._input_pin = GPIO_Input(
            pin=self.sample_notification_pin,
            pin_type='internal',
            interrupt_flank=InterruptFlank.BOTH,
            pull_up_down=PullupPulldown.DOWN,
            callback=self._samplesReadyInterrupt,
            bouncetime=1
        )

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def startSampleListener(self):
        self._startSampleListening = True

    # === PRIVATE METHODS ==============================================================================================
    def _samplesReadyInterrupt(self, *args, **kwargs):
        if not self._startSampleListening:
            return

        samples, latest_sample = self._readSamples()
        for callback in self.callbacks.rx_samples:
            callback(samples)

        for callback in self.callbacks.rx_latest_sample:
            callback(latest_sample)

    # ------------------------------------------------------------------------------------------------------------------
    def _readSamples(self):

        with self._lock:
            self._sendCommand(BILBO_SPI_Command_Type.READ_SAMPLE, 0)
            precise_sleep(0.01)
            self.interface.readinto( self.rx_buffer, start=0,
                                    end=SAMPLE_BUFFER_LL_SIZE * sizeof(bilbo_ll_sample_struct), write_value=0x05)

        samples = []
        for i in range(0, SAMPLE_BUFFER_LL_SIZE):
            sample = bytes_to_value(
                byte_data= self.rx_buffer[i * sizeof(bilbo_ll_sample_struct):(i + 1) * sizeof(bilbo_ll_sample_struct)],
                ctype_type=bilbo_ll_sample_struct)
            samples.append(sample)

        latest_sample = from_dict(BILBO_LL_Sample, samples[-1])

        self.logger.debug(f"Received latest sample: {latest_sample.general.tick}")
        new_lowlevel_tick = latest_sample.general.tick

        if self._lowlevel_tick is not None:
            if (new_lowlevel_tick - self._lowlevel_tick) != SAMPLE_BUFFER_LL_SIZE:
                self.logger.warning(f"Lowlevel SPI Tick not incremented by {SAMPLE_BUFFER_LL_SIZE}."
                                    f"Current tick: {self._lowlevel_tick}, New tick: {new_lowlevel_tick}")

        self._lowlevel_tick = new_lowlevel_tick

        return samples, latest_sample

    # ------------------------------------------------------------------------------------------------------------------
    def _sendCommand(self, command: int, length: int):
        assert (command in [BILBO_SPI_Command_Type.READ_SAMPLE, BILBO_SPI_Command_Type.SEND_TRAJECTORY])

        data = bytearray(4)

        len_byte_list = intToByteList(length, 2, byteorder='little')
        # data[0] = 0x66
        # data[1] = command
        # data[2:4] = len_byte_list

        data[0] = 0x66
        data[1] = command
        data[2] = 0x05
        data[3] = 0x05
        self.interface.send(data)


# ======================================================================================================================
if __name__ == '__main__':
    bilbo_spi = BILBO_Lowlevel_SPI()
    bilbo_spi.init()
    bilbo_spi.start()
    bilbo_spi.startSampleListener()

    while True:
        time.sleep(20)
