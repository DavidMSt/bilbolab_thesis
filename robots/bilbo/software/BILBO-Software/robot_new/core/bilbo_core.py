import time

from core.utils.events import EventListener
from core.utils.logging_utils import Logger
from robot_new.core.bilbo_core_common import BILBO_Core_Common
from robot_new.core.bilbo_core_control import BILBO_Core_Control
from robot_new.core.bilbo_core_estimation import BILBO_Core_Estimation
from robot_new.core.bilbo_core_interface import BILBO_Core_Interface
from robot_new.core.bilbo_core_logging import BILBO_Core_Logging
from robot_new.lowlevel.bilbo_lowlevel import BILBO_LowLevel


class BILBO_Core:
    common: BILBO_Core_Common
    control: BILBO_Core_Control
    estimation: BILBO_Core_Estimation
    logging: BILBO_Core_Logging
    interface: BILBO_Core_Interface

    lowlevel: BILBO_LowLevel

    # --- Private Variables ---
    _eventListener: EventListener
    _last_update_time: float | None = None

    # === INIT =========================================================================================================
    def __init__(self):
        self.logger = Logger("BILBO_CORE", "DEBUG")
        self.common = BILBO_Core_Common()
        self.lowlevel = BILBO_LowLevel()
        self.logging = BILBO_Core_Logging(self.common, self.lowlevel)

        self._eventListener = EventListener(event=self.lowlevel.events.samples, callback=self.step)

    # === METHODS ======================================================================================================
    def init(self):
        self.lowlevel.init()
        self.logging.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.lowlevel.start()
        self.logging.start()
        self.lowlevel.initializeLowlevelCounter()
        self._eventListener.start()
        self.lowlevel.startSampleListener()

    # ------------------------------------------------------------------------------------------------------------------
    def step(self):
        # 1. General things
        self.logger.debug(f"Step {self.lowlevel.tick}")
        current_time = time.perf_counter()
        if self._last_update_time is not None:
            update_time = current_time - self._last_update_time
            if update_time > 0.2:
                self.logger.warning(f"Update took {update_time*1000:.1f} ms")
        self._last_update_time = current_time

        # 2. Update the estimation
        # 3. Update the control
        # self.control.update()


        # 4. Update the logging
        self.logging.update()
    # ------------------------------------------------------------------------------------------------------------------

    # === PRIVATE METHODS ==============================================================================================


if __name__ == '__main__':
    core = BILBO_Core()
    core.init()
    core.start()

    while True:
        time.sleep(10)
