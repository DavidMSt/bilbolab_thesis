import ctypes

from core.communication.serial.serial_interface import Serial_Interface, SerialMessage
from core.utils.callbacks import callback_definition, CallbackContainer, OPTIONAL
from core.utils.ctypes_utils import CType
from core.utils.events import event_definition, Event, EventFlag
from core.utils.exit import register_exit_callback
from robot_new.lowlevel.communication.bilbo_serial_messages import BILBO_SERIAL_MESSAGES
import robot_new.lowlevel.definitions.stm32_addresses as addresses
from robot_new.lowlevel.definitions.stm32_general import twipr_firmware_revision


# === BILBO LOWLEVEL SERIAL ============================================================================================
@callback_definition
class BILBO_Serial_Communication_Callbacks:
    rx: CallbackContainer
    event: CallbackContainer = CallbackContainer(parameters=[('messages', list, OPTIONAL)])
    error: CallbackContainer
    debug: CallbackContainer


@event_definition
class BILBO_Serial_Communication_Events:
    rx: Event
    event = Event(flags=EventFlag('type', type))
    error: Event
    debug: Event


class BILBO_Lowlevel_Serial:
    interface: Serial_Interface

    # === INIT =========================================================================================================
    def __init__(self, interface: Serial_Interface = None):
        if interface is None:
            ...

        self.interface = interface
        self.interface.callbacks.rx.register(self._rx_callback)
        self.interface.callbacks.event.register(self._rx_event_callback)
        self.callbacks = BILBO_Serial_Communication_Callbacks()
        self.events = BILBO_Serial_Communication_Events()
        self.interface.addMessage(BILBO_SERIAL_MESSAGES)

        register_exit_callback(self.close)

    # === METHODS ======================================================================================================
    def init(self):
        ...
    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        if not self.interface.isRunning():
            self.interface.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        self.interface.close()

    # ------------------------------------------------------------------------------------------------------------------
    def writeValue(self, module: int = 0, address: int | list = None, value=None, type=ctypes.c_uint8):
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
    def resetLowlevelCounter(self):
        return self.executeFunction(
            module=addresses.TWIPR_AddressTables.REGISTER_TABLE_GENERAL,
            address=addresses.TWIPR_GeneralAddresses.ADDRESS_FIRMWARE_RESET,
            input_type=None,
            output_type=ctypes.c_bool,
            timeout=1
        )
    # ------------------------------------------------------------------------------------------------------------------
    def debug(self, state):
        self.interface.function(module=addresses.TWIPR_AddressTables.REGISTER_TABLE_GENERAL,
                                address=addresses.TWIPR_GeneralAddresses.ADDRESS_FIRMWARE_DEBUG,
                                data=state,
                                input_type=ctypes.c_uint8)

    # === PRIVATE METHODS ==============================================================================================
    def _rx_callback(self, message: SerialMessage, *args, **kwargs):
        pass
        # message.executeCallback()

    def _rx_event_callback(self, message: SerialMessage, *args, **kwargs):
        for callback in self.callbacks.event:
            if callback.parameters['messages'] is not None:
                if type(message) in callback.parameters['messages']:
                    callback(message)
            else:
                callback(message)

        self.events.event.set(data=message, flags={'type': type(message)})
