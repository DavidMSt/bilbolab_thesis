import threading
import time

from core.communication.wifi.data_link import CommandArgument
from core.utils.colors import color_to_255
from core.utils.logging_utils import Logger
from core.utils.time import IntervalTimer
from robot.common import FRODO_Common
from robot.communication.frodo_communication import FRODO_Communication
from robot.definitions import TASK_TS
from robot.estimation.frodo_estimation import FRODO_Estimation
from robot.frodo_core.frodo_core import FRODO_Core
from hardware.control_board import RobotControl_Board
from robot.control.frodo_control import FRODO_Control, FRODO_ControlMode
from core.utils.exit import register_exit_callback
from robot.interfaces.frodo_interfaces import FRODO_Interfaces
from robot.logging.frodo_logging import FRODO_Logging
from robot.sensing.frodo_sensors import FRODO_Sensors
from robot.utilities.frodo_utilities import FRODO_Utilities

# === FRODO ============================================================================================================
class FRODO:
    common: FRODO_Common
    core: FRODO_Core
    control: FRODO_Control
    estimation: FRODO_Estimation
    utilities: FRODO_Utilities
    logging: FRODO_Logging
    sensors: FRODO_Sensors
    communication: FRODO_Communication
    interfaces: FRODO_Interfaces

    # === INIT =========================================================================================================
    def __init__(self):
        self.logger = Logger("FRODO", "DEBUG")

        self.common = FRODO_Common()
        self.board = RobotControl_Board()
        self.core = FRODO_Core(self.board, self.common)
        self.communication = FRODO_Communication(self.common)

        self.utilities = FRODO_Utilities(self.common, self.communication, self.core)

        self.sensors = FRODO_Sensors(common=self.common)
        self.estimation = FRODO_Estimation(self.common, self.core, self.communication, self.sensors)
        self.control = FRODO_Control(self.common, self.core, self.communication)

        self.logging = FRODO_Logging(self.common, self.core, self.control, self.estimation, self.sensors)
        self.interfaces = FRODO_Interfaces(self.common, self.communication, self.sensors, self.control)

        self.settings = self.common.getDefinitions()

        self._addWifiCommands()

        self._thread = threading.Thread(target=self._task, daemon=True)

        register_exit_callback(self.close)
        register_exit_callback(self._preClose, 10)

    # === METHODS ======================================================================================================
    def init(self):
        self.core.init()
        self.sensors.init()
        self.communication.init()
        self.estimation.init()
        self.control.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.core.start()

        self.logger.info("Starting FRODO")
        self.board.beep()
        self.core.lowlevel.board.setRGBLEDExtern(color_to_255(self.settings.color), brightness=0.1)
        self.sensors.start()
        self.estimation.start()

        self.communication.start()
        self.control.start()
        self.interfaces.start()
        self._thread.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        self.logger.info("Exit FRODO")

    # === PRIVATE METHODS ==============================================================================================
    def _preClose(self):
        self.core.lowlevel.board.beep(440, 500, 2)
        self.control.setMode(FRODO_ControlMode.EXTERNAL)
        self.control.setTrackSpeed(0, 0)

    # ------------------------------------------------------------------------------------------------------------------
    def _task(self):

        success = self.core.lowlevel.events.initialized.wait(timeout=2, stale_event_time=10)
        if not success:
            self.logger.error("FRODO Lowlevel was not not initialized")
            return

        sample = self.core.lowlevel.sample
        self.logger.important(
            f"FRODO successfully initialized. (Battery voltage: {sample.general.battery_voltage:.1f} V)")

        timer = IntervalTimer(interval=TASK_TS)
        while True:
            # Increase the step
            self.common.incrementStep()

            # Update the estimation
            self.estimation.update()

            # Do the control update
            self.control.update()

            # Update the logging
            self.logging.update()

            # Send the data via wifi
            self.communication.sendSample(self.logging.getSample())

            # Sleep until next iteration
            timer.sleep_until_next()

    # ------------------------------------------------------------------------------------------------------------------

    def _addWifiCommands(self):
        # --- Direct speed control (EXTERNAL mode only) ---------------------------------------------
        self.communication.wifi.newCommand(
            identifier='setSpeed',
            function=self.control.setTrackSpeed,
            description='Set left/right track speeds in m/s (requires EXTERNAL mode).',
            arguments=[
                CommandArgument(name='speed_left', type=float, description='Left track speed [m/s]'),
                CommandArgument(name='speed_right', type=float, description='Right track speed [m/s]'),
            ]
        )

        self.communication.wifi.newCommand(
            identifier='setSpeedNormalized',
            function=self.control.setTrackSpeedNormalized,
            description='Set left/right track speeds normalized to [-1..1] (requires EXTERNAL mode).',
            arguments=[
                CommandArgument(name='speed_left_normalized', type=float,
                                description='Left track speed normalized [-1..1]'),
                CommandArgument(name='speed_right_normalized', type=float,
                                description='Right track speed normalized [-1..1]'),
            ]
        )

        # --- Mode control --------------------------------------------------------------------------
        self.communication.wifi.newCommand(
            identifier='setMode',
            function=self.control.setMode,
            description='Switch control mode: EXTERNAL or NAVIGATION.',
            arguments=[
                CommandArgument(name='mode', type=str, description='Target mode: "EXTERNAL" or "NAVIGATION"'),
            ]
        )

        # --- Navigator: high-level actions ---------------------------------------------------------
        self.communication.wifi.newCommand(
            identifier='startNavigation',
            function=self.control.startNavigation,
            description='Start the navigator (process queued elements).',
            arguments=[]
        )

        self.communication.wifi.newCommand(
            identifier='pauseNavigation',
            function=self.control.navigator.pauseNavigation,  # exposed on Navigator
            description='Pause the navigator (holds current element and commands zero speed).',
            arguments=[]
        )

        self.communication.wifi.newCommand(
            identifier='resumeNavigation',
            function=self.control.navigator.resumeNavigation,  # exposed on Navigator
            description='Resume the navigator if paused.',
            arguments=[]
        )

        self.communication.wifi.newCommand(
            identifier='stopNavigation',
            function=self.control.stopNavigation,
            description='Stop the navigator and command zero speed.',
            arguments=[]
        )

        self.communication.wifi.newCommand(
            identifier='clearNavigation',
            function=self.control.clearNavigation,
            description='Stop navigation and clear any queued elements.',
            arguments=[]
        )

        self.communication.wifi.newCommand(
            identifier='skip_element',
            function=self.control.skip_element,
            description='Skip current element and continue with next',
            arguments=[]
        )

        # --- Navigator: convenience motion primitives (auto-start) --------------------------------
        self.communication.wifi.newCommand(
            identifier='moveTo',
            function=self.control.moveTo,
            description='Queue a MoveTo(x, y) and start navigation if not running.',
            arguments=[
                CommandArgument(name='x', type=float, description='Target X [m] in world frame'),
                CommandArgument(name='y', type=float, description='Target Y [m] in world frame'),
            ]
        )

        # --- Queue-only: add elements without starting the navigator ------------------------------
        self.communication.wifi.newCommand(
            identifier='addMoveTo',
            function=self.control.addMoveTo,
            description='Enqueue MoveTo(x, y) without starting navigation.',
            arguments=[
                CommandArgument(name='x', type=float, description='Target X [m]'),
                CommandArgument(name='y', type=float, description='Target Y [m]'),
                CommandArgument(name="element_id", type=str, description="Element ID", optional=True, default=None),
            ]
        )

        self.communication.wifi.newCommand(
            identifier='addMoveToRelative',
            function=self.control.addMoveToRelative,
            description='Enqueue MoveToRelative(dx, dy) without starting navigation.',
            arguments=[
                CommandArgument(name='dx', type=float, description='Relative X [m]'),
                CommandArgument(name='dy', type=float, description='Relative Y [m]'),
            ]
        )

        self.communication.wifi.newCommand(
            identifier='addRelativeStraightMove',
            function=self.control.addRelativeStraightMove,
            description='Enqueue RelativeStraightMove(distance) without starting navigation.',
            arguments=[
                CommandArgument(name='distance', type=float, description='Distance along current heading [m]'),
            ]
        )

        self.communication.wifi.newCommand(
            identifier='addTurnTo',
            function=self.control.addTurnTo,
            description='Enqueue TurnTo(psi) without starting navigation.',
            arguments=[
                CommandArgument(name='psi', type=float, description='Absolute heading [rad]'),
            ]
        )

        self.communication.wifi.newCommand(
            identifier='addRelativeTurn',
            function=self.control.addRelativeTurn,
            description='Enqueue RelativeTurn(dpsi) without starting navigation.',
            arguments=[
                CommandArgument(name='dpsi', type=float, description='Relative heading change [rad]'),
            ]
        )

        self.communication.wifi.newCommand(
            identifier='addTurnToPoint',
            function=self.control.addTurnToPoint,
            description='Enqueue TurnToPoint(x, y) without starting navigation.',
            arguments=[
                CommandArgument(name='x', type=float, description='Point X [m]'),
                CommandArgument(name='y', type=float, description='Point Y [m]'),
            ]
        )

        self.communication.wifi.newCommand(
            identifier='addTimeWait',
            function=self.control.addTimeWait,
            description='Enqueue TimeWait(duration, reference) without starting navigation. reference: "PRIMITIVE" or "EXPERIMENT".',
            arguments=[
                CommandArgument(name='duration', type=float, description='Duration [s]'),
                CommandArgument(name='reference', type=str,
                                description='Time reference ("PRIMITIVE"|"EXPERIMENT")'),
            ]
        )

        self.communication.wifi.newCommand(
            identifier='addAbsoluteTimeWait',
            function=self.control.addAbsoluteTimeWait,
            description='Enqueue AbsoluteTimeWait(unix_time) without starting navigation.',
            arguments=[
                CommandArgument(name='unix_time', type=float, description='Unix timestamp [s]'),
            ]
        )

        self.communication.wifi.newCommand(
            identifier='addEventWait',
            function=self.control.addEventWait,
            description='Enqueue EventWait(event) without starting navigation.',
            arguments=[
                CommandArgument(name='event', type=str, description='Event name'),
            ]
        )

        self.communication.wifi.newCommand(
            identifier='addCoordinatedMoveTo',
            function=self.control.addCoordinatedMoveTo,
            description='Enqueue CoordinatedMoveTo(x, y, psi_end=None) without starting navigation.',
            arguments=[
                CommandArgument(name='x', type=float, description='Target X [m]'),
                CommandArgument(name='y', type=float, description='Target Y [m]'),
                CommandArgument(name='psi_end', type=float, optional=True, default=None,
                                description='Final heading [rad] (use NaN or omit for none)'),
            ]
        )


# ======================================================================================================================
if __name__ == '__main__':
    frodo = FRODO()
    frodo.start()

    while True:
        time.sleep(10)
