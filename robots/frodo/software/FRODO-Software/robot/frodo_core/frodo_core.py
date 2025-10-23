import time

from core.utils.callbacks import CallbackContainer, callback_definition
from core.utils.events import event_definition, Event
from core.utils.exit import register_exit_callback
from hardware.control_board import RobotControl_Board
from robot.common import FRODO_Common
from robot.frodo_core.frodo_lowlevel import FRODO_Lowlevel


@callback_definition
class FRODO_Core_Callbacks:
    lowlevel_sample: CallbackContainer


@event_definition
class FRODO_Core_Events:
    initialized: Event
    lowlevel_sample: Event


class FRODO_Core:
    lowlevel: FRODO_Lowlevel
    callbacks: FRODO_Core_Callbacks
    events: FRODO_Core_Events

    # === INIT =========================================================================================================
    def __init__(self, board: RobotControl_Board, common: FRODO_Common):
        self.lowlevel = FRODO_Lowlevel(board)
        self.common = common
        self.callbacks = FRODO_Core_Callbacks()
        self.events = FRODO_Core_Events()

        # Register callbacks to the lowlevel
        self.lowlevel.callbacks.sample.register(self.callbacks.lowlevel_sample.call)

        register_exit_callback(self.close)

    # === METHODS ======================================================================================================
    def init(self):
        self.lowlevel.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.lowlevel.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self):
        ...
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # == PRIVATE METHODS ===============================================================================================


if __name__ == '__main__':
    board = RobotControl_Board()
    core = FRODO_Core(board)
    core.init()
    core.start()

    while True:
        time.sleep(1)
