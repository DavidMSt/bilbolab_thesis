from core.archive.device import Device
from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.logging_utils import Logger


# ======================================================================================================================
@callback_definition
class Frodo_Callbacks:
    stream: CallbackContainer


# ======================================================================================================================
class Frodo:
    device: Device
    callbacks: Frodo_Callbacks

    def __init__(self, device: Device):
        self.device = device
        self.device.callbacks.stream.register(self._onStream_callback)
        self.callbacks = Frodo_Callbacks()

        self.logger = Logger(f"{self.id}")

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def id(self):
        return self.device.information.device_id

    # ------------------------------------------------------------------------------------------------------------------
    def setSpeed(self, speed_left, speed_right):
        self.device.function(function='setSpeed',
                             data={
                                 'speed_left': speed_left,
                                 'speed_right': speed_right
                             })

    # ------------------------------------------------------------------------------------------------------------------
    def beep(self):
        self.device.function(function='beep', data={'frequency': 250, 'time_ms': 250, 'repeats': 1})

    # ------------------------------------------------------------------------------------------------------------------
    def getData(self, timeout=0.05):
        try:
            data = self.device.function(function='getData',
                                        data=None,
                                        return_type=dict,
                                        request_response=True,
                                        timeout=timeout)
        except TimeoutError:
            data = None
        return data

    # ------------------------------------------------------------------------------------------------------------------
    def test(self, input, timeout=1):
        try:
            data = self.device.function(function='test',
                                        data={'input': input},
                                        return_type=dict,
                                        request_response=True,
                                        timeout=timeout)
        except TimeoutError:
            data = None
        return data

    # ------------------------------------------------------------------------------------------------------------------
    def addMovement(self, dphi, radius, time):
        self.device.function(function='addNavigationMovement', data={'dphi': dphi, 'radius': radius, 
                                                                        'vtime': time})

    # ------------------------------------------------------------------------------------------------------------------
    def startNavigationMovement(self):
        self.device.function(function='startNavigationMovement', data={})

    # ------------------------------------------------------------------------------------------------------------------
    def stopNavigationMovement(self):
        self.device.function(function='stopNavigationMovement', data={})

    # ------------------------------------------------------------------------------------------------------------------
    def pauseNavigationMovement(self):
        self.device.function(function='pauseNavigationMovement', data={})

    # ------------------------------------------------------------------------------------------------------------------
    def continueNavigationMovement(self):
        self.device.function(function='continueNavigationMovement', data={})

    # ------------------------------------------------------------------------------------------------------------------
    def clearNavigationMovementQueue(self):
        self.device.function(function='clearNavigationMovementQueue', data={})

    # ------------------------------------------------------------------------------------------------------------------
    def setControlMode(self, mode):
        self.device.function(function='setControlMode', data={'mode': mode})

    # ------------------------------------------------------------------------------------------------------------------
    def setExternalLEDs(self, color):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def runSensingStep(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def runControlStep(self):
        ...

    # === PRIVATE METHODS ==============================================================================================
    def _onStream_callback(self, message, *args, **kwargs):
        ...
