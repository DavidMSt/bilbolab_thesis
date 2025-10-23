import ctypes
import dataclasses
import enum

import numpy as np

import robot.lowlevel.stm32_addresses as addresses
from robot.bilbo_definitions import BILBO_DynamicState
from robot.communication.bilbo_communication import BILBO_Communication
from robot.hardware import readHardwareDefinition
from robot.lowlevel.stm32_sample import BILBO_LL_Sample
from core.utils.logging_utils import Logger


class TWIPR_Estimation_Status(enum.IntEnum):
    ERROR = 0,
    NORMAL = 1,


class TWIPR_Estimation_Mode(enum.IntEnum):
    TWIPR_ESTIMATION_MODE_VEL = 0,
    TWIPR_ESTIMATION_MODE_POS = 1


@dataclasses.dataclass(frozen=True)
class TWIPR_Estimation_Sample:
    status: TWIPR_Estimation_Status = TWIPR_Estimation_Status.ERROR
    state: BILBO_DynamicState = dataclasses.field(default_factory=BILBO_DynamicState)
    mode: TWIPR_Estimation_Mode = TWIPR_Estimation_Mode.TWIPR_ESTIMATION_MODE_VEL


# ======================================================================================================================

class BILBO_Estimation:
    _comm: BILBO_Communication

    state: BILBO_DynamicState
    status: TWIPR_Estimation_Status

    mode: TWIPR_Estimation_Mode

    def __init__(self, comm: BILBO_Communication):
        self._comm = comm

        self.state = BILBO_DynamicState()
        self.status = TWIPR_Estimation_Status.NORMAL
        self.mode = TWIPR_Estimation_Mode.TWIPR_ESTIMATION_MODE_VEL
        # self._comm.callbacks.rx_stm32_sample.register(self._onSample)

        self.logger = Logger('Estimation')
        self.logger.setLevel('DEBUG')

    # ==================================================================================================================
    def init(self):
        hardware_definition = readHardwareDefinition()
        theta_offset = hardware_definition.model.theta_offset
        self.setThetaOffset(theta_offset)

    # ------------------------------------------------------------------------------------------------------------------
    def getSample(self) -> TWIPR_Estimation_Sample:
        # sample = TWIPR_Estimation_Sample(
        #     mode=self.mode,
        #     status=self.status,
        #     state=self.state
        # )
        sample = {
            'mode': self.mode,
            'status': self.status,
            'state': dataclasses.asdict(self.state)
        }
        return sample

    # ------------------------------------------------------------------------------------------------------------------
    def setThetaOffset(self, offset: float):
        self.logger.info(f'Setting theta offset to {np.rad2deg(offset):.2f} deg')
        success = self._comm.serial.executeFunction(
            module=addresses.TWIPR_AddressTables.REGISTER_TABLE_GENERAL,
            address=addresses.TWIPR_EstimationAddresses.SET_THETA_OFFSET,
            data=offset,
            input_type=ctypes.c_float,
            output_type=ctypes.c_bool
        )

        if not success:
            self.logger.error('Could not set theta offset')

        # self._comm.serial.executeFunction(
        #     module=addresses.TWIPR_AddressTables.REGISTER_TABLE_GENERAL,
        #     address=addresses.TWIPR_ControlAddresses.ADDRESS_CONTROL_SET_MODE,
        #     data=mode.value,
        #     input_type=ctypes.c_uint8
        # )

    # ==================================================================================================================
    def _update(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def _onSample(self, sample: BILBO_LL_Sample, *args, **kwargs):
        self.state.v = sample.estimation.state.v
        self.state.theta = sample.estimation.state.theta
        self.state.theta_dot = sample.estimation.state.theta_dot
        self.state.psi = sample.estimation.state.psi
        self.state.psi_dot = sample.estimation.state.psi_dot

    # ------------------------------------------------------------------------------------------------------------------
    def _readState_LL(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
