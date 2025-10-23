# Sample
import time

from core.interfaces.wifi_interface import WIFI_Interface
from utils.callbacks import Callback


class TWIPR_WIFI_Interface:
    interface: WIFI_Interface

    connected: bool
    callbacks: dict

    _time_sync_offset: float  # Offset to sync the device time with the server time

    def __init__(self, interface: WIFI_Interface):
        self.interface = interface

        self.connected = False

        self.interface.registerCallback('connected', self._connectedCallback)
        self.interface.registerCallback('disconnected', self._disconnectedCallback)
        self.interface.registerCallback('sync', self._timeSyncCallback)

        self.callbacks = {
            'connected': [],
            'disconnected': [],
        }

        self._time_sync_offset = 0

    # ------------------------------------------------------------------------------------------------------------------
    def registerCallback(self, callback_id, function: callable, parameters: dict = None, lambdas: dict = None,
                         **kwargs):
        callback = Callback(function, parameters, lambdas, **kwargs)

        if callback_id in self.callbacks:
            self.callbacks[callback_id].append(callback)
        else:
            raise Exception("Invalid Callback type")

    # ------------------------------------------------------------------------------------------------------------------
    def sendMessage(self, message):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def sendStream(self, data):
        if self.interface.connected:
            self.interface.sendStreamMessage(data)

    # ------------------------------------------------------------------------------------------------------------------
    def getTime(self):
        return time.time() + self._time_sync_offset

    # ------------------------------------------------------------------------------------------------------------------
    def addCommands(self, commands: dict):
        self.interface.addCommands(commands)

    # ------------------------------------------------------------------------------------------------------------------
    def addCommand(self, identifier: str, callback: (callable, Callback), arguments: list[str], description: str):
        self.interface.addCommand(identifier, callback, arguments, description)

    # ==================================================================================================================
    def _connectedCallback(self, *args, **kwargs):
        self.connected = True

        for callback in self.callbacks['connected']:
            callback(*args, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    def _disconnectedCallback(self, *args, **kwargs):
        self.connected = False

        for callback in self.callbacks['disconnected']:
            callback(*args, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    def _timeSyncCallback(self, data, *args, **kwargs):
        self._time_sync_offset = data['time'] - time.time()
