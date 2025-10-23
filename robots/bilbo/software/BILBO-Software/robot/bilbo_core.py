import dataclasses
import threading
import time
from typing import Callable

from core.utils.exit import register_exit_callback
from core.utils.network import getSignalStrength, check_internet
from .core import get_logging_provider
from core.utils.callbacks import callback_definition
from core.utils.events import event_definition, Event
from core.utils.files import fileExists
from core.utils.json_utils import readJSON
from core.utils.logging_utils import Logger
from robot.settings import settings_file_path, GUI_PORT, STREAM_UDP_PORT, USERNAME, PASSWORD
from .hardware import readHardwareDefinition, BILBO_Hardware


def error_handler(severity, message):
    print(
        f"[{severity}] {message}"
    )


@dataclasses.dataclass
class BILBO_Information:
    id: str = ''
    type: str = ''
    version: str = ''
    color: list | None = None
    address: str = ''
    data_stream_port: int = ''
    gui_port: int = ''
    ssid: str = ''
    username: str = ''
    password: str = ''
    hardware: BILBO_Hardware | None = None


@event_definition
class BILBO_Core_Interaction_Events:
    resume: Event
    repeat: Event
    abort: Event


@event_definition
class BILBO_Core_Events:
    # sample: Event = Event(data_type=BILBO_Sample)
    sample: Event
    control_mode_change: Event
    control_config_change: Event
    experiment_mode_change: Event
    error: Event
    server_connected: Event
    server_disconnected: Event
    joystick_connected: Event
    joystick_disconnected: Event


@callback_definition
class BILBO_Core_Callbacks:
    ...


# ======================================================================================================================
class BILBO_Core:
    interaction_events: BILBO_Core_Interaction_Events
    events: BILBO_Core_Events

    information: BILBO_Information

    joystick_connected: bool = False
    server_connected: bool = False

    _exit: bool = False

    # === INIT =========================================================================================================
    def __init__(self):
        self.interaction_events = BILBO_Core_Interaction_Events()
        self.events = BILBO_Core_Events()
        self.information = self._getInformation()

        self.connection_strength = 0
        self.internet_connected = False

        self.logger = Logger("CORE", "INFO")

        self._thread = threading.Thread(target=self._connection_check_task)
        self._thread.start()

        register_exit_callback(self.stop)

    # === PROPERTIES ===================================================================================================
    @property
    def tick(self):
        return get_logging_provider().tick

    # === METHODS ======================================================================================================
    @staticmethod
    def getID() -> str:
        if not fileExists(settings_file_path):
            raise FileNotFoundError("Settings file not found. Run Bilbo Setup first")

        data = readJSON(settings_file_path)
        if not 'ID' in data:
            raise KeyError("ID not found. Run Bilbo Setup first")

        return data['ID']

    # ------------------------------------------------------------------------------------------------------------------
    def stop(self):
        self._exit = True
        if self._thread.is_alive():
            self._thread.join()

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def getData(self, signals, start_tick, end_tick):
        # Instance method so you can log if needed
        try:
            return get_logging_provider().getData(signals, start_tick, end_tick)
        except RuntimeError as e:
            self.logger.error(str(e))
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def getConnectionStatus(self, as_dict: bool = False):
        return {
            'strength': self.connection_strength,
            'internet': self.internet_connected
        }

    # ------------------------------------------------------------------------------------------------------------------
    def getHardwareDefinition(self):
        hwd = readHardwareDefinition()
        return hwd

    # ------------------------------------------------------------------------------------------------------------------
    def setResumeEvent(self, data):
        self.interaction_events.resume.set(data=data)

    def setRepeatEvent(self, data):
        self.interaction_events.repeat.set(data=data)

    def setAbortEvent(self, data):
        self.interaction_events.abort.set(data=data)

    # === PRIVATE METHODS ==============================================================================================
    def _getInformation(self) -> BILBO_Information:
        information = BILBO_Information(
            id=self.getID(),
            type='normal',  # TODO
            version='',  # TODO
            color=[],  # TODO
            address='',
            data_stream_port=STREAM_UDP_PORT,
            gui_port=GUI_PORT,
            ssid='',
            username=USERNAME,
            password=PASSWORD,
            hardware=readHardwareDefinition()
        )
        return information

    # ------------------------------------------------------------------------------------------------------------------
    def _connection_check_task(self):
        while not self._exit:
            self.connection_strength = getSignalStrength('wlan0')['percent']
            self.internet_connected = check_internet(timeout=1)
            time.sleep(2)
