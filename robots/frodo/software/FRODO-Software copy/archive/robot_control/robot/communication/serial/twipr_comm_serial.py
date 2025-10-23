import ctypes

from core.communication.serial.serial_interface import Serial_Interface
from utils.callbacks import Callback

import bilbo_old.settings as settings
import bilbo_old.communication.serial.adresses_general as addresses_general


class TWIPR_Serial_Interface:
    interface: Serial_Interface

    callbacks: dict

    def __init__(self, interface: Serial_Interface):
        self.interface = interface

        self.callbacks = {
            'rx': [],
            'rx_event': [],
            'rx_error': [],
            'rx_debug': [],
        }

    # === METHODS ======================================================================================================
    def registerCallback(self, callback_id, function: callable, parameters: dict = None, lambdas: dict = None,
                         **kwargs):
        callback = Callback(function, parameters, lambdas, **kwargs)

        if callback_id in self.callbacks:
            self.callbacks[callback_id].append(callback)
        else:
            raise Exception("Invalid Callback type")

    # ------------------------------------------------------------------------------------------------------------------
    def init(self):
        self.interface.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.interface.start()

    # ------------------------------------------------------------------------------------------------------------------
    def writeValue(self, module: int = 0, address: (int, list) = None, value=None, type=ctypes.c_uint8):
        self.interface.write(module, address, value, type)

    # ------------------------------------------------------------------------------------------------------------------
    def readValue(self, address, module: int = 0, type=ctypes.c_uint8):
        return self.interface.read(address, module, type)

    # ------------------------------------------------------------------------------------------------------------------
    def executeFunction(self, address, module: int = 0, data=None, input_type=None, output_type=None, timeout=1):
        self.interface.function(address, module, data, input_type, output_type, timeout)

    # ------------------------------------------------------------------------------------------------------------------
    def readTick(self):
        tick = self.interface.read(module=settings.REGISTER_TABLE_GENERAL, address=addresses_general.TICK,
                                   type=ctypes.c_uint32)

        return tick

    # ------------------------------------------------------------------------------------------------------------------
    def debug(self, state):
        self.interface.function(module=settings.REGISTER_TABLE_GENERAL, address=addresses_general.DEBUG, data=state,
                                input_type=ctypes.c_uint8)

    # === PRIVATE METHODS ==============================================================================================
