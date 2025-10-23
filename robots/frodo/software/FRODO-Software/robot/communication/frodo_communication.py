from core.utils.exit import register_exit_callback
from core.utils.logging_utils import Logger
from robot.common import FRODO_Common
from robot.communication.frodo_wifi_interface import FRODO_WIFI_Interface


class FRODO_Communication:
    wifi: FRODO_WIFI_Interface

    # === INIT =========================================================================================================
    def __init__(self, common: FRODO_Common):
        self.common = common
        self.logger = Logger("COMMUNICATION", "DEBUG")
        self.wifi = FRODO_WIFI_Interface(common=self.common)

        register_exit_callback(self.close)

    # === METHODS ======================================================================================================
    def init(self):
        self.wifi.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.logger.info("Starting FRODO Communication")
        self.wifi.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        self.logger.info("Exit FRODO Communication")

    # ------------------------------------------------------------------------------------------------------------------
    def sendSample(self, sample):
        self.wifi.sendStream(data=sample, stream_id="sample")

    # === PRIVATE METHODS ==============================================================================================
