import dataclasses
import threading
import time
from typing import Callable

import yaml

from core.utils.dataclass_utils import from_dict_auto
# ======================================================================================================================
from core.utils.exit import register_exit_callback
from core.utils.network import getSignalStrength, check_internet
from .bilbo_definitions import BILBO_TestbedConfig
from .config import BILBO_Config, get_bilbo_config
from .core import get_logging_provider
from core.utils.callbacks import callback_definition
from core.utils.events import event_definition, Event
from core.utils.files import fileExists
from core.utils.json_utils import readJSON
from core.utils.logging_utils import Logger
from robot.paths import CONFIG_PATH, ROBOT_PATH


# ======================================================================================================================
def error_handler(severity, message):
    print(
        f"[{severity}] {message}"
    )


@event_definition
class BILBO_Common_Interaction_Events:
    resume: Event
    repeat: Event
    abort: Event


@event_definition
class BILBO_Common_Events:
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
class BILBO_Common_Callbacks:
    ...


# ======================================================================================================================
class BILBO_Common:
    interaction_events: BILBO_Common_Interaction_Events
    events: BILBO_Common_Events
    callbacks: BILBO_Common_Callbacks

    information: BILBO_Config

    joystick_connected: bool = False
    server_connected: bool = False

    _exit: bool = False

    id: str
    config: BILBO_Config
    testbed_config: BILBO_TestbedConfig

    # === INIT =========================================================================================================
    def __init__(self):
        self.interaction_events = BILBO_Common_Interaction_Events()
        self.events = BILBO_Common_Events()
        self.callbacks = BILBO_Common_Callbacks()

        self.id = self._get_id()
        self.config = self._get_config()
        self.testbed_config = self._get_testbed_config()

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

    # ------------------------------------------------------------------------------------------------------------------
    def stop(self):
        self._exit = True
        if self._thread.is_alive():
            self._thread.join()

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_data(self, signals, start_tick, end_tick):
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
    def setResumeEvent(self, data):
        self.interaction_events.resume.set(data=data)

    def setRepeatEvent(self, data):
        self.interaction_events.repeat.set(data=data)

    def setAbortEvent(self, data):
        self.interaction_events.abort.set(data=data)

    # === PRIVATE METHODS ==============================================================================================
    def _get_config(self) -> BILBO_Config:
        config = get_bilbo_config(self._get_id())
        return config

    # ------------------------------------------------------------------------------------------------------------------
    def _connection_check_task(self):
        while not self._exit:
            self.connection_strength = getSignalStrength('wlan0')['percent']
            self.internet_connected = check_internet(timeout=1)
            time.sleep(2)

    # ------------------------------------------------------------------------------------------------------------------
    def _get_testbed_config(self) -> BILBO_TestbedConfig:
        testbed_file = f"{CONFIG_PATH}/testbed.yaml"

        if not fileExists(testbed_file):
            raise FileNotFoundError("Testbed file not found. Run Bilbo Setup first")

        with open(testbed_file, 'r') as file:
            config = yaml.safe_load(file)

        config = from_dict_auto(BILBO_TestbedConfig, config)
        return config

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _get_id() -> str:
        id_file = f"{ROBOT_PATH}/ID"
        if not fileExists(id_file):
            raise FileNotFoundError("ID file not found. Run Bilbo Setup first")
        else:
            with open(id_file, 'r') as file:
                id = file.read()
            return id
