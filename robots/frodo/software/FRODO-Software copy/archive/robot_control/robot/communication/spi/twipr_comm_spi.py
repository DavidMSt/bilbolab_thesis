from ctypes import sizeof

from core.communication.spi.spi import SPI_Interface
import bilbo_old.settings as settings
from utils.callbacks import Callback
import bilbo_old.communication.spi.ll_sample as ll_sample
from utils.ctypes_utils import struct_to_dict

from RPi import GPIO


def printStuff():
    print("fhjdsvfhjsdvghf")


class TWIPR_SPI_Interface:
    interface: SPI_Interface
    callbacks: dict

    def __init__(self, interface: SPI_Interface):
        self.interface = interface

        self.callbacks = {
            'rx_samples': []
        }

    # === METHODS ======================================================================================================
    def registerCallback(self, callback_id, function: callable, parameters: dict = None, lambdas: dict = None,
                         **kwargs):
        callback = Callback(function, parameters, lambdas, **kwargs)

        if callback_id in self.callbacks:
            self.callbacks[callback_id].append(callback)
        else:
            raise Exception("Invalid Callback type")

    def init(self):
        self._configureSampleGPIO()
        ...

    def start(self):
        ...

    # === PRIVATE METHODS ==============================================================================================

    def _configureSampleGPIO(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(settings.TWIPR_GPIO_INTERRUPT_NEW_SAMPLES, GPIO.IN,
                   pull_up_down=GPIO.PUD_DOWN)  # pull_up_down=GPIO.PUD_DOWN
        GPIO.add_event_detect(settings.TWIPR_GPIO_INTERRUPT_NEW_SAMPLES, GPIO.BOTH,
                              callback=self._samplesReadyInterrupt, bouncetime=1)

    # ------------------------------------------------------------------------------------------------------------------
    def _samplesReadyInterrupt(self, *args, **kwargs):
        new_samples = self._readSamples()
        for callback in self.callbacks['rx_samples']:
            callback(new_samples)

    def _readSamples(self):
        data_rx_bytes = bytearray(settings.SAMPLE_BUFFER_SIZE * sizeof(ll_sample.twipr_sample))
        self.interface.readinto(data_rx_bytes, start=0,
                                end=settings.SAMPLE_BUFFER_SIZE * sizeof(ll_sample.twipr_sample))
        samples = []
        for i in range(0, settings.SAMPLE_BUFFER_SIZE):
            samples.append(
                struct_to_dict(
                    ll_sample.twipr_sample.from_buffer_copy(
                        data_rx_bytes[i * sizeof(ll_sample.twipr_sample):(i + 1) * sizeof(ll_sample.twipr_sample)])))

        return samples

    #
    #     def samples_ready_callback(self, *args, **kwargs):
    #         # logging.info("RX")
    #         new_samples = self.readSampleData()
    #         if not self.first_sample_received:
    #             # print("TEST")
    #             logging.info("READY")
    #             self.first_sample_received = True
    #
    #         theta = np.rad2deg(new_samples[0]['estimation']['state']['theta'])
    #         teleplot.sendValue('theta', theta)
    #         self.samples.extend(new_samples)
    #
    #     def readSampleData(self):
    #         data_rx_bytes = bytearray(SAMPLE_SIZE * SAMPLE_BUFFER_SIZE)
    #         self.spi.readinto(data_rx_bytes, start=0, end=SAMPLE_SIZE * SAMPLE_BUFFER_SIZE, write_value=2)
    #         samples = []
    #         for i in range(0, SAMPLE_BUFFER_SIZE):
    #             samples.append(
    #                 struct_to_dict(twipr_sample.from_buffer_copy(data_rx_bytes[i * SAMPLE_SIZE:(i + 1) * SAMPLE_SIZE])))
    #
    #         return samples
