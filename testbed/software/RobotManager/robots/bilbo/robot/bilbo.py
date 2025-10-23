import time

from core.communication.device_server import Device
from core.utils.events import Event, event_definition
from robots.bilbo.robot.bilbo_control import BILBO_Control
from robots.bilbo.robot.bilbo_core import BILBO_Core
from robots.bilbo.robot.experiment.bilbo_experiment import BILBO_ExperimentHandler
from robots.bilbo.robot.bilbo_interfaces import BILBO_Interfaces
from robots.bilbo.robot.bilbo_data import BILBO_Sample, bilboSampleFromDict
from robots.bilbo.robot.bilbo_definitions import *
from robots.bilbo.robot.bilbo_utilities import BILBO_Utilities


# ======================================================================================================================
class BILBO:
    device: Device

    information: BILBO_Information

    core: BILBO_Core
    control: BILBO_Control
    experiment_handler: BILBO_ExperimentHandler
    interfaces: BILBO_Interfaces
    data: BILBO_Sample

    # ==================================================================================================================
    def __init__(self, device: Device, information: BILBO_Information, *args, **kwargs):
        self.device = device

        self.information = information

        self.core = BILBO_Core(device=device, robot_id=self.device.information.device_id)

        self.control = BILBO_Control(core=self.core)
        self.experiment_handler = BILBO_ExperimentHandler(core=self.core, control=self.control)
        self.utilities = BILBO_Utilities(core=self.core)
        self.interfaces = BILBO_Interfaces(core=self.core,
                                           control=self.control,
                                           experiments=self.experiment_handler,
                                           utilities=self.utilities)

        self.data = BILBO_Sample()

        self.device.callbacks.stream.register(self._onStreamCallback)
        self.device.callbacks.disconnected.register(self._disconnected_callback)

    # ------------------------------------------------------------------------------------------------------------------
    def setControlConfiguration(self, config):
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    def loadControlConfiguration(self, name):
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    def saveControlConfiguration(self, name):
        raise NotImplementedError

    # === CLASS METHODS ================================================================================================

    # === METHODS ======================================================================================================

    # === PROPERTIES ===================================================================================================
    @property
    def id(self):
        return self.device.information.device_id

    # === COMMANDS ===========================================================================
    def balance(self, state):
        self.control.setControlMode(BILBO_Control_Mode.BALANCING)

    # ------------------------------------------------------------------------------------------------------------------
    def stop(self):
        self.control.setControlMode(0)

    # ------------------------------------------------------------------------------------------------------------------
    def setLEDs(self, red, green, blue):
        self.device.executeFunction('setLEDs', arguments={'red': red, 'green': green, 'blue': blue})

    # ------------------------------------------------------------------------------------------------------------------
    def _onStreamCallback(self, stream, *args, **kwargs):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def _disconnected_callback(self, *args, **kwargs):
        del self.experiment_handler

    # ------------------------------------------------------------------------------------------------------------------
    def __del__(self):
        print(f"Deleting {self.id}")
