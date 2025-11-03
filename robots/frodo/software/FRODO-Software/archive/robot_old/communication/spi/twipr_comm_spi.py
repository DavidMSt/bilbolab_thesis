import time
from ctypes import sizeof
from RPi import GPIO

# === OWN PACKAGES =====================================================================================================
from core.communication.spi.spi import SPI_Interface
from utils.callbacks import Callback, callback_handler, CallbackContainer
from utils.dataclass_utils import from_dict
from utils.exit import ExitHandler
from bilbo_old.lowlevel.stm32_sample import bilbo_ll_sample_struct, BILBO_LL_Sample
from utils.ctypes_utils import bytes_to_value, struct_to_dataclass
from bilbo_old.lowlevel.stm32_sample import SAMPLE_BUFFER_SIZE

# ======================================================================================================================
@callback_handler
class BILBO_SPI_Callbacks:
    rx_samples: CallbackContainer

# ======================================================================================================================
class BILBO_SPI_Interface:
    interface: SPI_Interface
    callbacks: BILBO_SPI_Callbacks
    sample_notification_pin: int

    def __init__(self, interface: SPI_Interface, sample_notification_pin):
        self.interface = interface
        self.sample_notification_pin = sample_notification_pin
        self.callbacks = BILBO_SPI_Callbacks()

        # self.exit = ExitHandler()
        # self.exit.register(self.close)

    # === METHODS ======================================================================================================
    def init(self):
        self._configureSampleGPIO()
        ...
    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        GPIO.cleanup()

    # === PRIVATE METHODS ==============================================================================================
    def _configureSampleGPIO(self):
        time.sleep(0.25)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.sample_notification_pin, GPIO.IN,
                   pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(self.sample_notification_pin, GPIO.BOTH,
                              callback=self._samplesReadyInterrupt, bouncetime=1)

    # ------------------------------------------------------------------------------------------------------------------
    def _samplesReadyInterrupt(self, *args, **kwargs):
        new_samples = self._readSamples()

        for callback in self.callbacks.rx_samples:
            callback(new_samples)

    # ------------------------------------------------------------------------------------------------------------------
    def _readSamples(self):
        data_rx_bytes = bytearray(SAMPLE_BUFFER_SIZE * sizeof(bilbo_ll_sample_struct))
        self.interface.readinto(data_rx_bytes, start=0,
                                end=SAMPLE_BUFFER_SIZE * sizeof(bilbo_ll_sample_struct))
        samples = []
        for i in range(0, SAMPLE_BUFFER_SIZE):
            sample = bytes_to_value(
                byte_data=data_rx_bytes[i * sizeof(bilbo_ll_sample_struct):(i + 1) * sizeof(bilbo_ll_sample_struct)],
                ctype_type=bilbo_ll_sample_struct)

            sample = from_dict(BILBO_LL_Sample, sample)
            samples.append(sample)

        return samples
