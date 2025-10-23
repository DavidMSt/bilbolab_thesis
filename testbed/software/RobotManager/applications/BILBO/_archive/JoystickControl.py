import time

from extensions.cli.src.cli import CommandSet
# === OWN PACKAGES =====================================================================================================
from robots.bilbo.manager.bilbo_manager import BILBO_Manager
from robots.bilbo.robot.utils import TWIPR_GUI
from robots.bilbo.robot.utils import BILBO_JoystickControl
from core.utils.exit import register_exit_callback
from core.utils.logging_utils import setLoggerLevel
from core.utils.sound.sound import playSound, SoundSystem
from core.utils.sound.sound import speak

# === GLOBAL VARIABLES =================================================================================================
setLoggerLevel(logger=['tcp', 'server', 'udp'], level='DEBUG')
ENABLE_SPEECH_OUTPUT = True


# ==================================================================================================================
# The `SimpleTwiprJoystickControl` class provides a simplified means of joystick control for the BILBO robot.
# It manages instances of a BILBO manager, a joystick controller, and a GUI.
class SimpleTwiprJoystickControl:
    """
    This class allows joystick control functionality for the BILBO robot.
    It provides a simplified interface for the BILBO Manager, the BILBO JoystickControl, and the TWIPR_GUI.
    Callbacks and functionalities are registered within the class allowing a seamless and straightforward control.
    """
    robot_manager: BILBO_Manager
    joystick_control: BILBO_JoystickControl
    gui: TWIPR_GUI

    command_set: CommandSet

    def __init__(self):
        """
        Initialize the BILBO manager, BILBO joystick control and GUI instances.
        Register callback methods for the manager, joystick control and GUI
        """
        self.robot_manager = BILBO_Manager(robot_auto_start=False)
        self.joystick_control = BILBO_JoystickControl(bilbo_manager=self.robot_manager, run_in_thread=True)
        self.gui = TWIPR_GUI()

        self.command_set = CommandSet(name='root')
        # self.cli = CLI()

        self.gui.callbacks.rx_message.register(self._rxMessageGUI_callback)
        self.robot_manager.callbacks.stream.register(self.gui.sendRawStream)
        self.robot_manager.callbacks.new_robot.register(self._newRobot_callback)
        self.robot_manager.callbacks.robot_disconnected.register(self._robotDisconnected_callback)
        self.joystick_control.callbacks.new_joystick.register(self.gui.addJoystick)
        self.joystick_control.callbacks.new_joystick.register(self._newJoystickConnected_callback)
        self.joystick_control.callbacks.joystick_disconnected.register(self.gui.removeJoystick)
        self.joystick_control.callbacks.joystick_disconnected.register(self._joystickDisconnected_callback)

        register_exit_callback(self.close)

        self.soundsystem = SoundSystem(primary_engine='etts')
        self.soundsystem.start()

        # self.command_set.addChild(TWIPR_Manager_CommandSet(self.robot_manager))
        # self.command_set.addChild(TWIPR_JoystickControl_CommandSet(self.joystick_control))

    # ==================================================================================================================
    def init(self):
        """
        Initializes the robot manager, joystick control and the GUI.
        The initialization process involves setting initial states and configurations.
        """
        self.robot_manager.init()
        self.joystick_control.init()
        self.gui.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        """
        Start the robot manager, joystick controller and the GUI.
        It calls their respective start method which should initiate their operation.
        """
        speak(f'Starting Bilbo joystick control') # on {self.robot_manager.deviceManager.address.replace(".", " dot ")}')
        self.robot_manager.start()
        self.joystick_control.start()
        self.cli.start(commandSet=self.command_set)
        self.gui.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        speak('Stop Bilbo joystick control')
        time.sleep(2)
        global ENABLE_SPEECH_OUTPUT
        ENABLE_SPEECH_OUTPUT = False

    # ------------------------------------------------------------------------------------------------------------------
    def _newRobot_callback(self, robot, *args, **kwargs):
        """
        Callback method when new robot is detected.
        Prints the ID of the new robot to the GUI.
        """
        if ENABLE_SPEECH_OUTPUT:
            speak(f"Robot {robot.column_id} connected")
        else:
            playSound('robot_connected')
        self.gui.print(f"New Robot connected: {robot.column_id}")

    # ------------------------------------------------------------------------------------------------------------------
    def _robotDisconnected_callback(self, robot, *args, **kwargs):
        """
        Callback method when a robot disconnects.
        Prints the ID of the disconnected robot to the GUI.
        """
        if ENABLE_SPEECH_OUTPUT:
            speak(f"Robot {robot.column_id} disconnected")
        else:
            playSound('robot_disconnected')
        self.gui.print(f"Robot disconnected: {robot.column_id}")

    # ------------------------------------------------------------------------------------------------------------------
    def _newJoystickConnected_callback(self, joystick, *args, **kwargs):
        if ENABLE_SPEECH_OUTPUT:
            speak(f"Joystick connected")
            ...

    # ------------------------------------------------------------------------------------------------------------------
    def _joystickDisconnected_callback(self, joystick, *args, **kwargs):
        if ENABLE_SPEECH_OUTPUT:
            ...
            speak(f"Joystick disconnected")

    # ------------------------------------------------------------------------------------------------------------------
    def _rxMessageGUI_callback(self, message, *args, **kwargs):
        """
        Callback method for GUI related messages.
        It handles messages related to 'command' and 'joysticksChanged' type events.
        """
        message_type = message.get('type')

        if message_type == 'command':

            if message['data']['command'] == 'emergency':
                self.robot_manager.emergencyStop()
                if ENABLE_SPEECH_OUTPUT:
                    ...
                    speak("Emergency Stop")

        elif message_type == 'joysticksChanged':

            joysticks = message['data']['joysticks']

            for joystick in joysticks:
                joystick_id = joystick['id']
                robot_id = joystick['assignedBot']

                # No Robot assigned to Joystick
                if robot_id == '':
                    # Check if it is currently assigned to a robot
                    if joystick_id in self.joystick_control.assignments.keys():
                        self.joystick_control.unassignJoystick(joystick_id)
                else:

                    # Check if it is assigned to another robot in the list of robots
                    if joystick_id in self.joystick_control.assignments.keys():
                        connected_robot_id = self.joystick_control.assignments[joystick_id].robot.column_id

                        if connected_robot_id == robot_id:
                            # Do nothing, we have already assigned this robot
                            pass
                        else:
                            self.joystick_control.assignJoystick(joystick=joystick_id, bilbo=robot_id)

                    # it is not connected to a robot yet:
                    else:
                        self.joystick_control.assignJoystick(joystick=joystick_id, bilbo=robot_id)


# ======================================================================================================================


def main():
    app = SimpleTwiprJoystickControl()
    app.init()
    app.start()

    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
