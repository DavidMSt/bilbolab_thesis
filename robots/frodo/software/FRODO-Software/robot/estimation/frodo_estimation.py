import dataclasses
import enum

import numpy as np

from core.utils.logging_utils import Logger
from core.utils.time import setInterval
from robot.common import FRODO_Common
from robot.communication.frodo_communication import FRODO_Communication
from robot.definitions import FRODO_DynamicState
from robot.estimation.optitrack.optitrack_objects import TrackedFRODO_State
from robot.frodo_core.frodo_core import FRODO_Core
from robot.frodo_core.lowlevel_definitions import FRODO_LL_SAMPLE
from robot.sensing.frodo_sensors import FRODO_Sensors
from robot.estimation.optitrack.frodo_optitrack import FRODO_OptitrackListener, FRODO_OptitrackListenerSettings, \
    FRODO_OptitrackListenerStatus
from robot.settings import OPTITRACK_HOST



@dataclasses.dataclass
class FRODO_EstimationStatus:
    optitrack_listener: FRODO_OptitrackListenerStatus = FRODO_OptitrackListenerStatus.ERROR


@dataclasses.dataclass
class FRODO_Estimation_Lowlevel_Data:
    speed_left: float = 0.0
    speed_right: float = 0.0

@dataclasses.dataclass
class FRODO_Estimation_Sample:
    state: FRODO_DynamicState
    lowlevel_data: FRODO_Estimation_Lowlevel_Data


# === FRODO ESTIMATION =================================================================================================
class FRODO_Estimation:
    state: FRODO_DynamicState

    optitrack_listener: FRODO_OptitrackListener

    _first_optitrack_sample_received: bool = False
    _first_lowlevel_sample_received: bool = False
    _lowlevel_data: FRODO_LL_SAMPLE = None

    # === INIT =========================================================================================================
    def __init__(self, common: FRODO_Common,
                 core: FRODO_Core,
                 communication: FRODO_Communication,
                 sensors: FRODO_Sensors):
        self.common = common
        self.core = core
        self.communication = communication
        self.sensors = sensors
        self.logger = Logger("ESTIMATION", "DEBUG")

        self.state = FRODO_DynamicState()

        # Register a sample listener from the core
        self.core.callbacks.lowlevel_sample.register(self._onLowlevelSample)

        # Optitrack Listener
        optitrack_listener_settings = FRODO_OptitrackListenerSettings(
            host=OPTITRACK_HOST
        )

        self.optitrack_listener = FRODO_OptitrackListener(common=common, settings=optitrack_listener_settings)
        self.optitrack_listener.callbacks.sample.register(self._optitrack_sample_callback)

        # setInterval(self._printState, 2)

    # === METHODS ======================================================================================================
    def init(self):
        self.optitrack_listener.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.optitrack_listener.start()

    # ------------------------------------------------------------------------------------------------------------------
    def update(self):
        self.common.setDynamicState(self.state)

    # ------------------------------------------------------------------------------------------------------------------
    def getSample(self)->FRODO_Estimation_Sample:
        sample=FRODO_Estimation_Sample(
            state=self.state,
            lowlevel_data = FRODO_Estimation_Lowlevel_Data(
                speed_left=self._lowlevel_data.drive.speed.left,
                speed_right=self._lowlevel_data.drive.speed.right
            )
        )
        return sample

    # === PRIVATE METHODS ==============================================================================================
    def _printState(self):
        self.logger.info(
            f"State: x: {self.state.x:.3f}, "
            f"y: {self.state.y:.3f}, "
            f"psi: {np.rad2deg(self.state.psi):.1f}, "
            f"v: {self.state.v:.2f}, "
            f"psi_dot: {np.rad2deg(self.state.psi_dot):.1f}")

    # ------------------------------------------------------------------------------------------------------------------
    def _onLowlevelSample(self, sample: FRODO_LL_SAMPLE):
        if not self._first_lowlevel_sample_received:
            self.logger.info("First Lowlevel sample received")
            self._first_lowlevel_sample_received = True

        self._lowlevel_data = sample

    # ------------------------------------------------------------------------------------------------------------------
    def _optitrack_sample_callback(self, state: TrackedFRODO_State):
        if not self._first_optitrack_sample_received:
            self.logger.info("First Optitrack sample received")
            self._first_optitrack_sample_received = True
        self.state.x = state.x
        self.state.y = state.y
        self.state.psi = state.psi
        self.state.v = state.v
        self.state.psi_dot = state.psi_dot


