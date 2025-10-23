import time

from core.archive.device_manager import DeviceManager
from robots.frodo.archive.frodo import Frodo
from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.events import event_definition, ConditionEvent
from core.utils.exit import register_exit_callback
from core.utils.logging_utils import Logger


# ======================================================================================================================
@callback_definition
class FrodoManager_Callbacks:
    new_robot: CallbackContainer
    robot_disconnected: CallbackContainer
    stream: CallbackContainer


# ======================================================================================================================
@event_definition
class FrodoManager_Events:
    new_robot: ConditionEvent
    robot_disconnected: ConditionEvent
    stream: ConditionEvent


logger = Logger('FRODO MANAGER')
logger.setLevel('INFO')


# ======================================================================================================================
class FrodoManager:
    deviceManager: DeviceManager
    robots: dict[str, Frodo]
    callbacks: FrodoManager_Callbacks

    def __init__(self):
        self.deviceManager = DeviceManager()
        self.robots = {}
        self.callbacks = FrodoManager_Callbacks()

        self.deviceManager.callbacks.new_device.register(self._newDevice_callback)
        self.deviceManager.callbacks.device_disconnected.register(self._deviceDisconnected_callback)
        self.deviceManager.callbacks.stream.register(self._deviceStream_callback)

        register_exit_callback(self.close)

    # ==================================================================================================================
    def init(self):
        self.deviceManager.init()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        logger.info('Starting Frodo Manager')
        self.deviceManager.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        logger.info('Closing Frodo Manager')

    # ------------------------------------------------------------------------------------------------------------------
    def getRobotByID(self, robot_id: str) -> (Frodo, None):
        if robot_id in self.robots:
            return self.robots[robot_id]
        else:
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def _newDevice_callback(self, device, *args, **kwargs):
        # Check if the device has the correct class and type
        if not (device.information.device_class == 'robot' and device.information.device_type == 'frodo'):
            if device.information.device_class == 'robot':
                logger.warning(f"Robot attempted to connect with type {device.information.device_type}")
            return

        robot = Frodo(device)
        # Check if this robot ID is already used
        if robot.device.information.device_id in self.robots.keys():
            logger.warning(f"New Robot connected, but ID {robot.device.information.device_id} is already in use")

        self.robots[robot.device.information.device_id] = robot
        logger.info(f"New Frodo connected with ID: \"{robot.device.information.device_id}\"")

        # robot.beep()
        # delayed_execution(robot.setSpeed, delay=5, speed_left=1, speed_right=-1)
        # delayed_execution(robot.setSpeed, delay=10, speed_left=0, speed_right=0)

        for callback in self.callbacks.new_robot:
            callback(robot, *args, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    def _deviceDisconnected_callback(self, device, *args, **kwargs):
        """
        Callback for handling device disconnections.

        :param device: The disconnected device
        """
        if device.information.device_id not in self.robots:
            return

        robot = self.robots[device.information.device_id]
        self.robots.pop(device.information.device_id)

        logger.info(f"Robot {device.information.device_id} disconnected")

        # Remove any joystick assignments
        for callback in self.callbacks.robot_disconnected:
            callback(robot, *args, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    def _deviceStream_callback(self, stream, device, *args, **kwargs):
        """
        Callback for handling data streams from devices.

        :param stream: The data stream
        :param device: The device sending the stream
        """
        if device.information.device_id in self.robots.keys():
            for callback in self.callbacks.stream:
                callback(stream, self.robots[device.information.device_id], *args, **kwargs)


if __name__ == '__main__':
    manager = FrodoManager()
    manager.init()
    manager.start()

    while True:
        time.sleep(10)
