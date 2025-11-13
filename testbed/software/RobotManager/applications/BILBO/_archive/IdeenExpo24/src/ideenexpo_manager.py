import logging
import threading
import time

from applications.BILBO._archive.IdeenExpo24.src import ideenexpo_settings
from applications.BILBO._archive.IdeenExpo24.src.ideenexpo_gui import IdeenExpoGUI
from robots.bilbo.twipr_manager import BILBO_Manager
from core.utils.misc import clipValue
from extensions.joystick.joystick_manager import Joystick, JoystickManager

logger = logging.getLogger('IdeenExpo')
logger.setLevel('INFO')


class IdeenExpoManager:
    robotManager: BILBO_Manager
    gui: IdeenExpoGUI
    # simulation: BackgroundSimulation

    joystick_manager: JoystickManager
    _connected_joysticks: dict
    _manual_torque_limits: dict
    _masterJoystick: Joystick
    _masterJoystickActive: bool

    joystick_assignments: dict

    _thread: threading.Thread

    def __init__(self):

        self.robotManager = BILBO_Manager()
        self.joystick_manager = JoystickManager()

        self.joystick_manager.registerCallback('new_joystick', self._joystickConnected_callback)
        self.joystick_manager.registerCallback('joystick_disconnected', self._joystickDisconnected_callback)

        self.robotManager.registerCallback('stream', self._robotStream_callback)
        self.robotManager.registerCallback('new_robot', self._newRobot_callback)
        self.robotManager.registerCallback('robot_disconnected', self._robotDisconnected_callback)

        self.gui = IdeenExpoGUI()
        self.gui.registerCallback('rx_message', self._guiRxMessage_callback)

        self._manual_torque_limits = {
            'forward': 0,
            'turn': 0,
        }

        self.joystick_assignments = {}
        self._connected_joysticks = {}
        self._masterJoystick = None
        self._masterJoystickActive = False

        self._thread = threading.Thread(target=self._threadFunction)

    # ------------------------------------------------------------------------------------------------------------------
    def init(self):
        self.robotManager.init()
        self.joystick_manager.init()
        self.gui.init()

        self._setTorqueValues('forward', reset=True)
        self._setTorqueValues('turn', reset=True)

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.gui.start()
        self.joystick_manager.start()
        self.robotManager.start()
        self._thread.start()

    # === PRIVATE METHODS ==============================================================================================
    def _threadFunction(self):

        while True:

            for assignment in self.joystick_assignments.values():
                input = self._calculateInputValues(assignment['joystick'])
                assignment['robot'].setInput(input)
            time.sleep(0.1)

    # ------------------------------------------------------------------------------------------------------------------
    def _assignJoystick(self, robot, joystick):
        if isinstance(robot, str):
            if robot in self.robotManager.robots.keys():
                robot = self.robotManager.robots[robot]
            else:
                return

        if isinstance(joystick, str):
            if joystick in self.joystick_manager.joysticks.keys():
                joystick = self.joystick_manager.joysticks[joystick]
            else:
                return

        for key, assignment in self.joystick_assignments.items():
            if robot == assignment['robot']:
                self._unassignJoystick(assignment['joystick'])

        self.joystick_assignments[joystick.id] = {
            'robot': robot,
            'joystick': joystick
        }

        logger.info(
            f"Added Joystick assignment ({joystick.id} -> {self.joystick_assignments[joystick.id]['robot'].group_id})")
        self.gui.print(
            f"Added Joystick assignment ({joystick.id} -> {self.joystick_assignments[joystick.id]['robot'].group_id})")

        if joystick is not self._masterJoystick:
            joystick.setButtonCallback(button=1, event='down', function=robot.setControlMode, parameters={'mode': 2})
            joystick.setButtonCallback(button=0, event='down', function=robot.setControlMode, parameters={'mode': 0})

    # ------------------------------------------------------------------------------------------------------------------
    def _unassignJoystick(self, joystick):
        print("UNASSIGN")
        if isinstance(joystick, str) and joystick in self.joystick_manager.joysticks.keys():
            joystick = self.joystick_manager.joysticks[joystick]
        else:
            return

        # Check if there is a joystick assignment, if not, return
        if joystick.group_id not in self.joystick_assignments.keys():
            return

        logger.info(
            f"Remove Joystick assignment ({joystick.group_id} -> {self.joystick_assignments[joystick.group_id]['robot'].group_id})")
        self.gui.print(
            f"Remove Joystick assignment ({joystick.group_id} -> {self.joystick_assignments[joystick.group_id]['robot'].group_id})")
        self.joystick_assignments.pop(joystick.group_id)

        if joystick is not self._masterJoystick:
            joystick.clearAllButtonCallbacks()

    # ------------------------------------------------------------------------------------------------------------------
    def _assignMasterJoystickCallbacks(self):
        if self._masterJoystick is not None:
            self._masterJoystick.clearAllButtonCallbacks()
            self._masterJoystick.setJoyHatCallback(direction='up', function=self._changeTorqueValues,
                                                   parameters={'torque_direction': 'forward', 'value': 0.01})
            self._masterJoystick.setJoyHatCallback(direction='down', function=self._changeTorqueValues,
                                                   parameters={'torque_direction': 'forward', 'value': -0.01})

            self._masterJoystick.setJoyHatCallback(direction='right', function=self._changeTorqueValues,
                                                   parameters={'torque_direction': 'turn', 'value': 0.01})
            self._masterJoystick.setJoyHatCallback(direction='left', function=self._changeTorqueValues,
                                                   parameters={'torque_direction': 'turn', 'value': -0.01})

            self._masterJoystick.setButtonCallback(button=1, event='down', function=self._setControlModeAll,
                                                   parameters={'mode': 2})
            self._masterJoystick.setButtonCallback(button=0, event='down', function=self._setControlModeAll,
                                                   parameters={'mode': 0})
            self._masterJoystick.setButtonCallback(button=2, event='down', function=self._masterJoystickToggle)
            self._masterJoystick.setButtonCallback(button=3, event='down', function=self._setTorqueValues,
                                                   parameters={'direction': 'forward', 'reset': True})
            self._masterJoystick.setButtonCallback(button=3, event='down', function=self._setTorqueValues,
                                                   parameters={'direction': 'turn', 'reset': True})
            self._masterJoystick.setButtonCallback(button=4, event='down', function=self.robotManager.emergencyStop)

    # ------------------------------------------------------------------------------------------------------------------
    def _joystickConnected_callback(self, joystick: Joystick, *args, **kwargs):

        # Check if the joystick is in the list of known joysticks:
        # if joystick.uuid not in ideenexpo_settings.joysticks:
        #     logger.info(f"Joystick ({joystick.uuid}) connected, but not in list of known joysticks")
        #     return

        # Check if the joystick uuid is already given to another connected joystick
        # if joystick.uuid in self._connected_joysticks.keys():
        #     logger.info(f"Joystick ({joystick.uuid}) connected, but it is already in list of connected joysticks")
        #     return

        # Check if the joystick is the master joystick
        if joystick.guid in ideenexpo_settings.joysticks:
            if ideenexpo_settings.joysticks[joystick.guid]['master']:
                self.gui.print("Master Joystick connected")
                logger.info("Master Joystick connected")
                self._masterJoystick = joystick
                self._assignMasterJoystickCallbacks()

        self._connected_joysticks[joystick.id] = joystick

        # Add the joystick to the gui
        self.gui.addJoystick(joystick)

    # ------------------------------------------------------------------------------------------------------------------
    def _joystickDisconnected_callback(self, joystick, *args, **kwargs):
        if joystick.group_id not in self._connected_joysticks.keys():
            return

        # Remove the joystick from the list of connected joysticks
        self._connected_joysticks.pop(joystick.group_id)
        logger.info(f"Joystick disconnected. ID: {joystick.group_id})")
        self.gui.print(f"Joystick disconnected. ID: {joystick.group_id})")

        if joystick == self._masterJoystick:
            self.masterJoystick = None
            logger.info("Master Joystick disconnected!")
            self.gui.print("Master Joystick disconnected!")

        if joystick.group_id in self.joystick_assignments.keys():
            self._unassignJoystick(joystick)

        # Notify the GUI
        self.gui.removeJoystick(joystick)

    # ------------------------------------------------------------------------------------------------------------------
    def _setControlModeAll(self, mode):
        for robot in self.robotManager.robots.values():
            robot.setControlMode(mode)

    # ------------------------------------------------------------------------------------------------------------------
    def _masterJoystickToggle(self):
        self._masterJoystickActive = not self._masterJoystickActive

        if self._masterJoystickActive:
            self._masterJoystick.rumble(1, 750)
        else:
            self._masterJoystick.rumble(0.25, 250)

        logger.info(f"Master Joystick Toggle. Status: {self._masterJoystickActive}")
        self.gui.print(f"Master Joystick Toggle. Status: {self._masterJoystickActive}")

    # ------------------------------------------------------------------------------------------------------------------
    def _robotStream_callback(self, stream, robot, *args, **kwargs):
        sample = self.gui.convertTwiprSample(stream)
        self.gui.sendStream(sample)

    # ------------------------------------------------------------------------------------------------------------------
    def _newRobot_callback(self, robot, *args, **kwargs):
        self.gui.addRobot(robot)
        self.gui.print(f"Robot connected: {robot.group_id}")

    # ------------------------------------------------------------------------------------------------------------------
    def _robotDisconnected_callback(self, robot, *args, **kwargs):

        self.gui.removeRobot(robot)
        self.gui.print(f"Robot disconnected: {robot.group_id}")

        # Check if there were any joystick assignments
        for joystick_id, assignment in self.joystick_assignments.items():
            print(f"Remove assignment {joystick_id} and {robot.group_id}")
            if assignment['robot'] == robot:
                self._unassignJoystick(joystick_id)
                return

    # ------------------------------------------------------------------------------------------------------------------
    def _guiRxMessage_callback(self, message):
        message_type = message.get('type')

        if message_type == 'command':

            if message['data']['command'] == 'emergency':
                self.robotManager.emergencyStop()

        elif message_type == 'joysticksChanged':

            joysticks = message['data']['joysticks']

            for joystick in joysticks:
                joystick_id = joystick['id']
                robot_id = joystick['assignedBot']

                # No Robot assigned to Joystick
                if robot_id == '':
                    # Check if it is currently assigned to a robot
                    if joystick_id in self.joystick_assignments.keys():
                        self._unassignJoystick(joystick_id)
                else:

                    # Check if it is assigned to another robot in the list of robots
                    if joystick_id in self.joystick_assignments.keys():
                        connected_robot_id = self.joystick_assignments[joystick_id][
                            'robot'].device.config.device_id

                        if connected_robot_id == robot_id:
                            # Do nothing, we have already assigned this robot
                            pass
                        else:
                            self._assignJoystick(robot=robot_id, joystick=joystick_id)

                    # it is not connected to a robot yet:
                    else:
                        self._assignJoystick(robot=robot_id, joystick=joystick_id)

        elif message_type == 'set':
            robot_id = message['botId']
            data_key = message['data']['key']
            data_value = message['data']['value']
            timestamp = message['timestamp']

            self.robotManager.setRobotControlMode(robot_id, data_value)

        else:
            # Unknown Message
            pass

    # ------------------------------------------------------------------------------------------------------------------
    def _setTorqueValues(self, direction, value=0, reset=False):
        if direction == 'forward':
            if reset:
                self._manual_torque_limits['forward'] = ideenexpo_settings.manual_control_settings['forward']['default']
            else:
                self._manual_torque_limits['forward'] = clipValue(value,
                                                                  ideenexpo_settings.manual_control_settings['forward'][
                                                                      'max'],
                                                                  ideenexpo_settings.manual_control_settings['forward'][
                                                                      'min'])

        elif direction == 'turn':

            if reset:
                self._manual_torque_limits['turn'] = ideenexpo_settings.manual_control_settings['turn']['default']
            else:
                self._manual_torque_limits['turn'] = clipValue(value,
                                                               ideenexpo_settings.manual_control_settings['turn'][
                                                                   'max'],
                                                               ideenexpo_settings.manual_control_settings['turn'][
                                                                   'min'])
                print(self._manual_torque_limits['turn'])

        logger.info(f"Set Torque values to: Forward: {self._manual_torque_limits['forward']},"
                    f" Turn: {self._manual_torque_limits['turn']}")

        self.gui.print(f"Set Torque values to: Forward: {self._manual_torque_limits['forward']},"
                       f" Turn: {self._manual_torque_limits['turn']}")

    # ------------------------------------------------------------------------------------------------------------------
    def _changeTorqueValues(self, torque_direction, value, *args, **kwargs):
        self._setTorqueValues(torque_direction, self._manual_torque_limits[torque_direction] + value)

    def _calculateInputValues(self, joystick) -> list:
        forward = 0
        turn = 0
        master_boost = 0

        if self._masterJoystick is not None:
            master_boost = (self._masterJoystick.axis[5] + 1) * 0.25

        torque_forward_max_cmd = self._manual_torque_limits['forward'] + master_boost * (
                ideenexpo_settings.manual_control_settings['forward']['max'] - self._manual_torque_limits['forward'])

        if self._masterJoystick is not None and (
                self._masterJoystickActive or (abs(self._masterJoystick.axis[1]) > 0.05)):
            forward = self._masterJoystick.axis[1] * torque_forward_max_cmd
        else:
            forward = joystick.axis[1] * torque_forward_max_cmd

        if self._masterJoystick is not None and (
                self._masterJoystickActive or (abs(self._masterJoystick.axis[2]) > 0.05)):
            turn = self._masterJoystick.axis[2] * self._manual_torque_limits['turn']
        else:
            turn = joystick.axis[2] * self._manual_torque_limits['turn']
        return [forward + turn, forward - turn]
