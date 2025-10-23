from utils.callbacks import Callback
from board.board import RobotControl_Board
from bilbo_old.communication.serial.twipr_comm_serial import TWIPR_Serial_Interface
from bilbo_old.communication.spi.twipr_comm_spi import TWIPR_SPI_Interface
from bilbo_old.communication.wifi.twipr_comm_wifi import TWIPR_WIFI_Interface


class TWIPR_Communication:
    board: RobotControl_Board

    serial: TWIPR_Serial_Interface
    wifi: TWIPR_WIFI_Interface
    spi: TWIPR_SPI_Interface

    callbacks: dict

    def __init__(self, board: RobotControl_Board):
        self.board = board

        self.wifi = TWIPR_WIFI_Interface(interface=self.board.wifi_interface)
        self.serial = TWIPR_Serial_Interface(interface=self.board.serial_interface)
        self.spi = TWIPR_SPI_Interface(interface=self.board.spi_interface)

        self.callbacks = {
            'rx_sample': []
        }

        # Configure the SPI Interface
        self.spi.registerCallback('rx_samples', self._rx_sample_callback)

    # === METHODS ======================================================================================================

    def init(self):
        self.spi.init()
        self.serial.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.spi.start()
        self.serial.start()

    # ------------------------------------------------------------------------------------------------------------------
    def registerCallback(self, callback_id, function: callable, parameters: dict = None, lambdas: dict = None,
                         **kwargs):
        callback = Callback(function, parameters, lambdas, **kwargs)

        if callback_id in self.callbacks:
            self.callbacks[callback_id].append(callback)
        else:
            raise Exception("Invalid Callback type")

    # ------------------------------------------------------------------------------------------------------------------

    # === PRIVATE METHODS ==============================================================================================
    def _rx_sample_callback(self, samples, *args, **kwargs):

        sample = samples[0]
        for callback in self.callbacks['rx_sample']:
            callback(sample)
