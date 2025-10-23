import time

from core.utils.logging_utils import Logger
from robot.common import FRODO_Common
from robot.control.frodo_control import FRODO_Control
from robot.estimation.frodo_estimation import FRODO_Estimation
from robot.frodo_core.frodo_core import FRODO_Core
from robot.logging.frodo_sample import FRODO_Sample, FRODO_Sample_General
from robot.sensing.frodo_sensors import FRODO_Sensors


class FRODO_Logging:
    sample: FRODO_Sample | None

    # === INIT =========================================================================================================
    def __init__(self, common: FRODO_Common,
                 core: FRODO_Core,
                 control: FRODO_Control,
                 estimation: FRODO_Estimation,
                 sensors: FRODO_Sensors):
        self.core = core
        self.common = common
        self.control = control
        self.estimation = estimation
        self.sensors = sensors

        self.logger = Logger("LOGGING", "DEBUG")
        self.sample = None

    # === METHODS ======================================================================================================
    def update(self):
        self.sample = FRODO_Sample(
            general=FRODO_Sample_General(
                id=self.common.id,
                step=self.common.step,
                time=time.time(),
                battery=self.core.lowlevel.battery_voltage,
                connection_strength=self.common.getConnectionStrength(),
                internet_connection=self.common.getInternetConnected()
            ),
            estimation=self.estimation.getSample(),
            measurements=self.sensors.getSample(),
            control=self.control.getSample()
        )


    # ------------------------------------------------------------------------------------------------------------------
    def getSample(self):
        return self.sample
