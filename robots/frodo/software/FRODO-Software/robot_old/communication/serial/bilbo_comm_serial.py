from core.communication.serial.serial_interface import Serial_Interface, SerialMessage
import bilbo_old.lowlevel.stm32_addresses as addresses
from bilbo_old.lowlevel.stm32_general import twipr_firmware_revision
from bilbo_old.lowlevel.stm32_messages import *
from utils.callbacks import callback_handler, CallbackContainer
from utils.ctypes_utils import CType
from utils.events import ConditionEvent


# === CALLBACKS ========================================================================================================
@callback_handler
class BILBO_Serial_Communication_Callbacks:
    rx: CallbackContainer
    event: CallbackContainer
    error: CallbackContainer
    debug: CallbackContainer


# === EVENTS ===========================================================================================================
class BILBO_Serial_Communication_Events:
    rx: ConditionEvent
    event: ConditionEvent
    error: ConditionEvent
    debug: ConditionEvent


# === BILBO_Serial_Communication =======================================================================================
class BILBO_Serial_Communication:
    interface: Serial_Interface
    callbacks: BILBO_Serial_Communication_Callbacks

    def __init__(self, interface: Serial_Interface):
        self.interface = interface
        self.interface.callbacks.rx.register(self._rx_callback)
        self.callbacks = BILBO_Serial_Communication_Callbacks()

    # === METHODS ======================================================================================================
    def init(self):
        ...
        # self.interface.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        if not self.interface._thread.is_alive():
            self.interface.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self):
        self.interface.close()

    # ------------------------------------------------------------------------------------------------------------------
    def writeValue(self, module: int = 0, address: (int, list) = None, value=None, type=ctypes.c_uint8):
        self.interface.write(module, address, value, type)

    # ------------------------------------------------------------------------------------------------------------------
    def readValue(self, address: int, module: int = 0, type=ctypes.c_uint8):
        return self.interface.read(address, module, type)

    # ------------------------------------------------------------------------------------------------------------------
    def executeFunction(self, address, module: int = 0, data=None, input_type: CType = None, output_type=None,
                        timeout=1):
        return self.interface.function(address, module, data, input_type, output_type, timeout)

    # ------------------------------------------------------------------------------------------------------------------
    def readTick(self):
        tick = self.interface.read(module=addresses.TWIPR_AddressTables.REGISTER_TABLE_GENERAL,
                                   address=addresses.TWIPR_GeneralAddresses.ADDRESS_FIRMWARE_TICK,
                                   type=ctypes.c_uint32)

        return tick

    # ------------------------------------------------------------------------------------------------------------------
    def readFirmwareRevision(self):
        revision = self.interface.read(module=addresses.TWIPR_AddressTables.REGISTER_TABLE_GENERAL,
                                       address=addresses.TWIPR_GeneralAddresses.ADDRESS_FIRMWARE_REVISION,
                                       type=twipr_firmware_revision)

        return revision

    # ------------------------------------------------------------------------------------------------------------------
    def debug(self, state):
        self.interface.function(module=addresses.TWIPR_AddressTables.REGISTER_TABLE_GENERAL,
                                address=addresses.TWIPR_GeneralAddresses.ADDRESS_FIRMWARE_DEBUG,
                                data=state,
                                input_type=ctypes.c_uint8)

    # ------------------------------------------------------------------------------------------------------------------

    # === PRIVATE METHODS ==============================================================================================
    def _rx_callback(self, message: SerialMessage, *args, **kwargs):
        print("got message")
        # message.executeCallback()
