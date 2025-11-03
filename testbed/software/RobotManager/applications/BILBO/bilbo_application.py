import logging
import os
import sys
import time

from core.utils.callbacks import Callback
from robots.bilbo.robot.bilbo import BILBO

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one or more levels as needed
top_level_module = os.path.abspath(os.path.join(current_dir, '..', '..'))  # adjust as needed

if top_level_module not in sys.path:
    sys.path.insert(0, top_level_module)

# === CUSTOM MODULES ===================================================================================================
from applications.BILBO.gui.bilbo_application_gui import BILBO_Application_GUI
from applications.BILBO.settings import AUTOSTART_ROBOTS, AUTOSTOP_ROBOTS
from applications.BILBO.tracker.bilbo_tracker import BILBO_Tracker
# from extensions.cli.archive.cli_gui import CLI_GUI_Server
from extensions.cli.cli import CommandSet, CLI
from robots.bilbo.manager.bilbo_joystick_control import BILBO_JoystickControl
from robots.bilbo.manager.bilbo_manager import BILBO_Manager
from core.utils.exit import register_exit_callback
from core.utils.logging_utils import setLoggerLevel, Logger
from core.utils.loop import infinite_loop
from core.utils.sound.sound import speak, SoundSystem
from core.utils.files import relativeToFullPath

# ======================================================================================================================
ENABLE_SPEECH_OUTPUT = True
# EXPERIMENT_DIR = relativeToFullPath('~/bilbolab/experiments/bilbo')


# ======================================================================================================================
class BILBO_Application:
    robot_manager: BILBO_Manager
    tracker: BILBO_Tracker

    soundsystem: SoundSystem

    def __init__(self):
        self.robot_manager = BILBO_Manager(enable_scanner=AUTOSTART_ROBOTS, autostop_robots=AUTOSTOP_ROBOTS)

        # self.robot_manager.callbacks.stream.register(self.gui.sendRawStream)
        self.robot_manager.callbacks.new_robot.register(self._newRobot_callback)
        self.robot_manager.callbacks.robot_disconnected.register(self._robotDisconnected_callback)

        # Joystick Control
        self.joystick_control = BILBO_JoystickControl(bilbo_manager=self.robot_manager, run_in_thread=True)

        # CLI
        self.cli = CLI(id='bilbo_app_cli')

        # Logging
        self.logger = Logger('APP')
        self.logger.setLevel('INFO')

        # Sound System for speaking and sounds
        self.soundsystem = SoundSystem(primary_engine='etts', volume=1)
        self.soundsystem.start()

        # GUI
        self.gui = BILBO_Application_GUI(host=self.robot_manager.host,
                                         cli=self.cli,
                                         joystick_control=self.joystick_control,)

        self.gui.callbacks.emergency_stop.register(self.robot_manager.emergencyStop)

        # Exit Handling
        register_exit_callback(self.close)

    # ------------------------------------------------------------------------------------------------------------------
    def init(self):
        setLoggerLevel(logger=['tcp', 'server', 'UDP', 'UDP Socket', 'Sound'], level=logging.WARNING)

        self.robot_manager.init()
        self.joystick_control.init()

        self.cli.root.addChild(self.robot_manager.cli)
        self.cli.root.addChild(self.joystick_control.cli_command_set)

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        # self.cli_server.start()
        self.joystick_control.start()
        self.logger.info('Starting Bilbo application')
        speak('Start Bilbo application')
        self.robot_manager.start()
        self.gui.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        speak('Stop Bilbo application')
        self.logger.info('Closing Bilbo application')
        time.sleep(2)
        global ENABLE_SPEECH_OUTPUT
        ENABLE_SPEECH_OUTPUT = False

    # ==================================================================================================================
    def _newRobot_callback(self, bilbo: BILBO, *args, **kwargs):
        if ENABLE_SPEECH_OUTPUT:
            speak(f"Robot {bilbo.id} connected")

        # Wait until the first sample is received
        if not bilbo.core.initialized:
            bilbo.core.events.initialized.on(callback=Callback(function=self.gui.addRobot,
                                                               inputs={'robot': bilbo},
                                                               discard_inputs=True),
                                             once=True,
                                             discard_data=True)
        else:
            self.gui.addRobot(bilbo)

    # ------------------------------------------------------------------------------------------------------------------
    def _robotDisconnected_callback(self, bilbo, *args, **kwargs):
        if ENABLE_SPEECH_OUTPUT:
            speak(f"Robot {bilbo.id} disconnected")
        self.gui.removeRobot(bilbo)


# ======================================================================================================================
def run_bilbo_application():
    app = BILBO_Application()
    app.init()
    app.start()

    infinite_loop()


if __name__ == '__main__':
    run_bilbo_application()
